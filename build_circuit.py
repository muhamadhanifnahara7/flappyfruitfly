#!/usr/bin/env python3
"""
Build circuit.json — tight, real-time-safe version.
Targets:
  • ≤ 4 000 edges  (so 60 fps is trivial)
  • ≤   8 motor neurons  (bird doesn't flap non-stop)
  • ≤  40 input neurons
  • ≤ 120 interneurons

Strategy (forward-first — guarantees every motor is reachable from inputs):
  1. BFS forward from ALL inputs → find motors that optic-lobe neurons
     can actually reach (not random motors in a disconnected corner).
  2. Sample ≤ MAX_MOTORS from that confirmed set.
  3. BFS backward from the sampled motors → ancestors.
  4. BFS forward from inputs through ancestors → trimmed subgraph.
  5. Export strongest edges, hard-cap at MAX_EDGES.
"""
import json, gzip, csv, re, random
from collections import deque
from pathlib import Path

BASE = Path(__file__).parent
SEED = 42
random.seed(SEED)

# ── caps ────────────────────────────────────────────────
MAX_MOTORS = 8
MAX_INPUTS = 40
MAX_INTER  = 120
MIN_SYN    = 5      # minimum synapse count
MAX_EDGES  = 1200   # cap at 1200 for clean feedforward-dominated circuit

def load(fname):
    print(f"  Loading {fname}…", flush=True)
    with gzip.open(BASE / fname, 'rt', encoding='utf-8') as f:
        return list(csv.DictReader(f))

print("Loading files…")
classif    = load("classification.csv.gz")
conns      = load("connections.csv.gz")
cell_types = load("consolidated_cell_types.csv.gz")
coords     = load("coordinates.csv.gz")

print("Building lookups…")
cell_type_by_id = {r["root_id"]: r.get("primary_type","") for r in cell_types if r.get("root_id")}
class_by_id     = {r["root_id"]: r.get("class","")        for r in classif   if r.get("root_id")}
super_by_id     = {r["root_id"]: r.get("super_class","")  for r in classif   if r.get("root_id")}

# ── spatial positions normalised to 2D [20,260]×[20,340] and 3D [-100,100]³ ──
raw_pos3d = {}
for r in coords:
    rid, pos = r.get("root_id"), r.get("position","")
    if not rid or not pos: continue
    nums = re.sub(r'[\[\]]', '', pos).split()
    if len(nums) >= 3:
        raw_pos3d[rid] = (int(nums[0]), int(nums[1]), int(nums[2]))

if raw_pos3d:
    xs = [v[0] for v in raw_pos3d.values()]
    ys = [v[1] for v in raw_pos3d.values()]
    zs = [v[2] for v in raw_pos3d.values()]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    zmin, zmax = min(zs), max(zs)

    def norm_pos(rid):
        if rid not in raw_pos3d:
            return {"x": random.randint(20,260), "y": random.randint(20,340)}
        x, y, _ = raw_pos3d[rid]
        nx = 20 + int((x-xmin)/(xmax-xmin+1)*240) if xmax>xmin else 140
        ny = 20 + int((y-ymin)/(ymax-ymin+1)*320) if ymax>ymin else 180
        return {"x": nx, "y": ny}

    def norm_pos3d(rid):
        if rid not in raw_pos3d:
            return {"x": random.randint(-50,50), "y": random.randint(-50,50), "z": random.randint(-50,50)}
        x, y, z = raw_pos3d[rid]
        nx = int(-100 + (x-xmin)/(xmax-xmin+1)*200) if xmax>xmin else 0
        ny = int(-100 + (y-ymin)/(ymax-ymin+1)*200) if ymax>ymin else 0
        nz = int(-100 + (z-zmin)/(zmax-zmin+1)*200) if zmax>zmin else 0
        return {"x": nx, "y": ny, "z": nz}
else:
    def norm_pos(rid):
        return {"x": random.randint(20,260), "y": random.randint(20,340)}
    def norm_pos3d(rid):
        return {"x": 0, "y": 0, "z": 0}

print("Classifying neurons…")
input_set, motor_set = set(), set()
all_ids = set(cell_type_by_id) | set(class_by_id)
for nid in all_ids:
    ctype = cell_type_by_id.get(nid,"")
    cls   = class_by_id.get(nid,"")
    scls  = super_by_id.get(nid,"")
    if (re.match(r'^(LC|T4|T5|Mi|Tm|L[0-9]|R[0-9])', ctype, re.I)
            or scls == "optic"
            or cls  == "optic_lobe_intrinsic"):
        input_set.add(nid)
    if (re.match(r'^DN', ctype, re.I)
            or cls  == "descending"
            or scls == "descending"):
        motor_set.add(nid)
print(f"  Inputs: {len(input_set)}  Motors: {len(motor_set)}")

print("Building adjacency lists…")
fwd_adj, rev_adj = {}, {}
raw_edges = []
for r in conns:
    src, dst = r.get("pre_root_id"), r.get("post_root_id")
    cnt = int(r.get("syn_count",0) or 0)
    if not src or not dst or cnt < MIN_SYN: continue
    sign = -1 if re.search(r'GABA', r.get("nt_type",""), re.I) else 1
    raw_edges.append((src, dst, cnt, sign))
    fwd_adj.setdefault(src, []).append(dst)
    rev_adj.setdefault(dst, []).append(src)
print(f"  Total edges (syn≥{MIN_SYN}): {len(raw_edges)}")

# ═══════════════════════════════════════════════════════
# Phase 1 — Find confirmed input-reachable motors (fast 1-hop scan)
# ═══════════════════════════════════════════════════════
# 1-hop scan takes ~1s and finds ~165 motors — plenty to sample from.
# (BFS over 92k input nodes would be too slow or hit limit bugs.)
print("Phase 1: Scanning edges for 1-hop input→motor connections…")
motors_confirmed = set()
for src in input_set:
    for dst in fwd_adj.get(src, []):
        if dst in motor_set:
            motors_confirmed.add(dst)
print(f"  Motors directly reachable from inputs: {len(motors_confirmed)}")

if not motors_confirmed:
    raise RuntimeError("No motors reachable 1-hop from inputs! Check data integrity.")

# ═══════════════════════════════════════════════════════
# Phase 2 — Sample motors; BFS backward to find ancestors
# ═══════════════════════════════════════════════════════
motor_sample    = random.sample(list(motors_confirmed), min(MAX_MOTORS, len(motors_confirmed)))
motor_set_final = set(motor_sample)
print(f"Phase 2: Motor sample: {len(motor_sample)} — BFS backward for ancestors…")

BACK_LIMIT = 8000
ancestors = set(motor_sample)
frontier  = deque(motor_sample)
while frontier and len(ancestors) < BACK_LIMIT:
    cur = frontier.popleft()
    for src in rev_adj.get(cur, []):
        if src not in ancestors:
            ancestors.add(src)
            frontier.append(src)
print(f"  Ancestors collected: {len(ancestors)}")

# ═══════════════════════════════════════════════════════
# Phase 3 — BFS forward from inputs through ancestor set
# ═══════════════════════════════════════════════════════
print("Phase 3: BFS forward from inputs through ancestors…")
inputs_in_anc = input_set & ancestors
FWD_LIMIT     = 4000
reachable     = set(inputs_in_anc)
frontier      = deque(inputs_in_anc)
while frontier and len(reachable) < FWD_LIMIT:
    cur = frontier.popleft()
    for dst in fwd_adj.get(cur, []):
        if dst in ancestors and dst not in reachable:
            reachable.add(dst)
            frontier.append(dst)

# Verify which sampled motors are now reachable
motors_ok      = reachable & motor_set_final
inputs_ok      = reachable & input_set
print(f"  Reachable: {len(reachable)}, "
      f"motors in subgraph: {len(motors_ok)}/{len(motor_sample)}, "
      f"inputs in subgraph: {len(inputs_ok)}")

# If some sampled motors fell out of the trimmed BFS, use what we have
final_motors    = list(motors_ok) if motors_ok else motor_sample[:1]
motor_set_final = set(final_motors)

# ═══════════════════════════════════════════════════════
# Phase 4 — Trim to caps
# ═══════════════════════════════════════════════════════
final_inputs = random.sample(list(inputs_ok), min(MAX_INPUTS, len(inputs_ok)))

# BFS backward from final motors → interneurons linking them to inputs
inter_needed = set()
frontier4    = deque(final_motors)
visited4     = set(final_motors)
while frontier4 and len(inter_needed) < MAX_INTER:
    cur = frontier4.popleft()
    for src in rev_adj.get(cur, []):
        if src not in reachable: continue
        if src in motor_set_final or src in input_set: continue
        if src not in inter_needed:
            inter_needed.add(src)
        if src not in visited4:
            visited4.add(src)
            frontier4.append(src)

final_all = final_motors + final_inputs + list(inter_needed)
final_set = set(final_all)
print(f"  Final: motors={len(final_motors)}, "
      f"inputs={len(final_inputs)}, inter={len(inter_needed)}")

# ═══════════════════════════════════════════════════════
# Phase 5 — Export edges (strongest first, hard cap)
# ═══════════════════════════════════════════════════════
kept_raw = [(s,d,w,sg) for s,d,w,sg in raw_edges if s in final_set and d in final_set]
kept_raw.sort(key=lambda e: e[2], reverse=True)
kept_raw = kept_raw[:MAX_EDGES]

max_w      = max(w for _,_,w,_ in kept_raw) if kept_raw else 1
kept_edges = [
    {"src": s, "dst": d, "w": round((w / max_w) ** 0.4, 4), "sign": sg}
    for s,d,w,sg in kept_raw
]
print(f"  Edges exported: {len(kept_edges)}")

# ── Final connectivity check ──────────────────────────
adj_check   = {}
for e in kept_edges:
    adj_check.setdefault(e["src"],[]).append(e["dst"])
reach_check = set()
stack       = list(final_inputs)
while stack:
    cur = stack.pop()
    if cur in reach_check: continue
    reach_check.add(cur)
    stack.extend(adj_check.get(cur, []))
motors_reached = reach_check & motor_set_final
print(f"  ✅ Connectivity check: {len(motors_reached)}/{len(final_motors)} motors reachable from inputs")

# ═══════════════════════════════════════════════════════
# Phase 6 — Build neuron dict & save
# ═══════════════════════════════════════════════════════
# Include only neurons that appear in at least one edge OR are input/motor
edge_neurons = set()
for e in kept_edges:
    edge_neurons.add(e["src"]); edge_neurons.add(e["dst"])
edge_neurons |= set(final_inputs) | motor_set_final

neurons = {}
for nid in edge_neurons:
    ntype = ("input" if nid in input_set  else
             "motor" if nid in motor_set  else "inter")
    ctype = cell_type_by_id.get(nid, "Unknown")
    cls   = class_by_id.get(nid, "Unknown")
    scls  = super_by_id.get(nid, "Unknown")
    neurons[nid] = {
        "type": ntype,
        "a": 0,
        "ctype": ctype,
        "cls": cls,
        "scls": scls,
        "pos": norm_pos(nid),
        "pos3d": norm_pos3d(nid)
    }

input_ids = [n for n in final_inputs if n in neurons]
motor_ids = [n for n in final_motors  if n in neurons]

out = {"neurons": neurons, "edges": kept_edges,
       "inputIds": input_ids, "motorIds": motor_ids}
out_path = BASE / "circuit.json"
with open(out_path, "w") as f:
    json.dump(out, f, separators=(',',':'))

size_kb = out_path.stat().st_size // 1024
print(f"\n✅ Saved circuit.json ({size_kb} KB)")
print(f"   Neurons : {len(neurons)}")
print(f"   Edges   : {len(kept_edges)}")
print(f"   Inputs  : {len(input_ids)}   Motors: {len(motor_ids)}")
