# Voronoi Half-Patch (VHP)

VHP decomposes a B-rep into local units defined on half-edges, each encoding geometry and topology in a fixed-dimensional format. This converts a B-rep — an irregular graph — into a set of uniform tokens directly consumable by deep learning, supporting both edge-level and vertex-level latent representations.

VHP is framework-agnostic: its uniform structure is compatible with autoregressive Transformers, diffusion models, or other generative approaches.

## Pipeline

```
STEP files  ──brep2VHP──►  VHP graphs (.bin)  ──VHP2brep──►  STEP files
              (training data preparation)          (reconstruction)
```

## Stage 1: brep2VHP

Convert B-rep models (STEP format) into VHP graph data for training.

Edit the `ROOT` path in `brep2VHP/brep2VHP.sh`, then run:

```bash
cd brep2VHP
bash brep2VHP.sh
```

The script runs four steps in sequence:

1. Scale and split closed faces/edges
2. Handle inner wires
3. Resolve duplicate edges
4. Sample VHP graphs

Output `.bin` files are DGL graphs.

## Stage 2: VHP2brep

Reconstruct B-rep STEP models from VHP graphs.

Edit the `ROOT` path in `VHP2brep/VHP2brep.sh`, then run:

```bash
cd VHP2brep
bash VHP2brep.sh
```

Two surface fitting strategies are available (toggle via `--use-uv`):

| | Standard mode | UV mode |
|---|---|---|
| Surface method | `BRepFill_Filling` | `GeomAPI_PointsToBSplineSurface` |
| UV data needed | No | Yes |
| Best for | Planar / simple curved surfaces | Complex freeform surfaces |

**Standard mode** (default): builds faces using `BRepFill_Filling` with Voronoi interior point constraints. No UV data required.

**UV mode** (`--use-uv`): fits a B-spline surface via `GeomAPI_PointsToBSplineSurface` using RBF-interpolated UV→XYZ mappings. More robust for complex freeform surfaces.

## Data

Sample data is provided under `data/`:

```
data/
├── brep/    # input STEP files
├── simple/  # after closed face/edge splitting
├── split/   # after inner wire handling
├── break/   # after duplicate edge resolution
└── VHP/     # output VHP graphs (.bin)
```
