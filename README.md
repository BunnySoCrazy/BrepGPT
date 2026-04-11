
# BrepGPT: Autoregressive B-rep Generation with Voronoi Half-Patch  [![arXiv](https://img.shields.io/badge/arXiv-2511.22171-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2511.22171)
![teaser](assets/teaser.jpg)

---

This repository provides the official code of **BrepGPT**, accepted to SIGGRAPH Asia 2025 (Journal Track).

---

B-rep is the standard CAD representation, but its irregular graph structure — where geometry and topology are tightly coupled at multiple levels — makes it difficult to handle with deep learning.

Beyond the generative network, one contribution of this work is the **Voronoi Half-Patch (VHP)** representation. VHP decomposes a B-rep into local units defined on half-edges, each encoding geometry and topology in a fixed-dimensional format. This converts a B-rep — an irregular graph — into a set of uniform tokens directly consumable by deep learning, supporting both edge-level and vertex-level latent representations. BrepGPT demonstrates this with an autoregressive Transformer, but VHP is framework-agnostic: its uniform structure is equally compatible with diffusion models or other generative approaches.

## Environment Setup

```bash
mamba create -n brepgpt python=3.9 -y
conda activate brepgpt

mamba install occwl -c lambouj -c conda-forge -y
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html
pip install pytorch-lightning
pip install vector-quantize-pytorch==1.22.15
```



## Voronoi Half-Patch (VHP)

VHP is a network-agnostic 3D representation that converts B-reps into uniform, fixed-dimensional tokens defined on half-edges — compatible with autoregressive models, diffusion models, or any other generative framework. See [`VHP/README.md`](VHP/README.md) for details.

## Data

### Pre-processed DeepCAD dataset

Pre-processed VHP graphs for the DeepCAD dataset are available on **[Google Drive](https://drive.google.com/drive/folders/1BczW0SsSlGo440C4QxmqjfSYI9qpK5lL?usp=drive_link)** — download and use directly, no data preparation needed.


### Prepare your own data

To convert your own STEP files into VHP graphs, see [`VHP/README.md`](VHP/README.md) for the full pipeline. In brief:

**Step 1 — STEP → VHP graphs**

Edit the `ROOT` path in `VHP/brep2VHP/brep2VHP.sh`, then run:

```bash
cd VHP/brep2VHP
bash brep2VHP.sh
```

This runs four preprocessing steps in sequence: scale and split closed faces/edges → handle inner wires → resolve duplicate edges → sample VHP graphs. Output `.bin` files are DGL graphs with the following fields:

| Field | Shape | Description |
|---|---|---|
| `graph.ndata['x']` | `[N, 3]` | Vertex coordinates |
| `graph.edata['x']` | `[E, 6, 4, 3]` | VHP samples (curve × surface normals × 3D) |
| `graph.edata['next_half_edge']` | `[E, 4, 3]` | Next half-edge curve samples |
| `graph.edata['edge_inner_outer']` | `[E, 1]` | Inner / outer loop flag |

**Step 2 — VHP graphs → STEP**

To reconstruct STEP files from generated VHP graphs after inference, see the VHP2brep section in [`VHP/README.md`](VHP/README.md).

---


## Usage

### Training

Update `data_root` in `specs.json` to point to the downloaded (or prepared) VHP graph directory.

Training follows three steps (see `scripts/train.sh`):

**Step 1 — Train VQVAE encoders**

```bash
python train_LT.py -e cnnt/DeepCAD -m cnnt_vq
python train_LT.py -e vhp/DeepCAD -m vhp_vq
```

**Step 2 — Encode dataset with trained VQVAE**

```bash
python encode_LT.py -e cnnt/DeepCAD -m cnnt_vq
python encode_LT.py -e vhp/DeepCAD -m vhp_vq
```

**Step 3 — Train GPT**

```bash
python train_LT.py -e gpt/DeepCAD -m gpt
```

### Inference

```bash
python infer_LT.py -g gpt/DeepCAD -c cnnt/DeepCAD -v vhp/DeepCAD -n 32
```

See `scripts/infer.sh` for reference.


## Release Progress

- [x] VHP data processing code
- [x] Model training and inference code

## Related Resources

- **[3DNB](https://github.com/BunnySoCrazy/3DNB)** — Extension of this work
- **[Awesome-Neural-CAD](https://github.com/BunnySoCrazy/Awesome-Neural-CAD)** — Curated list of neural CAD generation research
- **[Awesome-3D-Generation](https://github.com/BunnySoCrazy/Awesome-3D-Generation)** — Curated list of 3D generation research


## Citation

If you use this code in your research, please cite:

```bibtex
@article{li2025brepgpt,
  title     = {BrepGPT: Autoregressive B-rep Generation with Voronoi Half-Patch},
  author    = {Li, Pu and Zhang, Wenhao and Quan, Weize and Zhang, Biao and Wonka, Peter and Yan, Dong-Ming},
  journal   = {ACM Transactions on Graphics},
  volume    = {44},
  number    = {6},
  pages     = {226:1--226:18},
  year      = {2025},
  publisher = {ACM}
}
```
