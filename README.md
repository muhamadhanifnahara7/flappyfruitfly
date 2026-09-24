# FlappyFruitFly

A Flappy Bird–style game flown by a real fruit-fly brain, not by you.

**Live demo:** [flappyfruitfly.vercel.app](https://flappyfruitfly.vercel.app)

You watch. A trimmed [*Drosophila melanogaster*](https://en.wikipedia.org/wiki/Drosophila_melanogaster) connectome from [FlyWire](https://flywire.ai/) is the pilot: optic-lobe input neurons see the upcoming pipe gap, activity travels through interneurons, and descending motor neurons flap the bird.

The sidebar shows that circuit live — neuron types, a 2D map or 3D brain view, and the cell classes that are currently firing.

## Run locally

Serve the project folder (the game loads `circuit.json` and `brain_mesh.json` over HTTP):

```bash
python3 -m http.server 8000
```

Then open [http://localhost:8000](http://localhost:8000).

## Rebuild the circuit

The published `circuit.json` is a small, real-time-safe subgraph (capped neurons and edges). To rebuild it from the full FlyWire tables, put these files next to `build_circuit.py` and run:

```bash
python3 build_circuit.py
```

Required local dumps (not in this repo; they are large):

- `classification.csv.gz`
- `connections.csv.gz`
- `consolidated_cell_types.csv.gz`
- `coordinates.csv.gz`

`brain_mesh.json` is a lightweight mesh derived from `VFB_00101567_(JRC2018Unisex).obj`.

## Credits

Connectome data from [FlyWire](https://flywire.ai/). Brain mesh from Virtual Fly Brain (`VFB_00101567`, JRC2018 Unisex).
