# Reconstruct single-cell resolution from spatial transcriptomics with CellRefiner

 CellRefiner is a physical model-based method that integrates a scRNA-seq dataset with a paired ST dataset to generate single-cell resolution in the imputed ST data. CellRefiner models cells as particles connected by forces, and then optimizes cell locations with spatial proximity constraints, gene expression similarity, and ligand-receptor interactions between cells.

# Installation
CellRefiner requires Python >= 3.9 and NVIDIA GPU with CUDA support.

**Step 1: Create and activate a virtual environment**

We recommend using conda or mamba for environment management.
```bash
conda create -n cellrefiner_env python=3.9
conda activate cellrefiner_env
```

**Step 2: Install cellrefiner from GitHub**

```bash
pip install git+https://github.com/XiangyuKuang/cellrefiner.git
```

**Step 3: Install CuPy (Optional)**

CuPy installation will accelerate preprocessing.spatial_mapping()

We recommend installing CuPy and CUDA toolkit via conda to avoid compatibility issues:

```bash
conda install cupy cudatoolkit -c conda-forge
```

# Usage

For detailed examples, please refer to the `tutorials` folder.

Import packages

```python
import squidpy as sq
import cellrefiner as cr
```
Load datasets and ligand-receptor database

```python
adata_st = sq.datasets.visium_fluo_adata_crop()
adata_sc = sq.datasets.sc_mouse_cortex()
db_lr = cr.pp.ligand_receptor_database()
```

Map cells to spots and spatial refinement
```python
adata_cr = cr.pp.spatial_mapping(adata_st,adata_sc,db_lr,scale=125,cluster_key_sc = 'cell_subclass')
```

Cell shape modeling and visualization
```python
sem = cr.tl.cell_shape_modeling(adata_cr,cluster_key = 'cell_subclass')
cr.pl.plot_cell_shape(sem)
```

Contact-based communication analysis
```python
cr.tl.contact_communication(db_lr, adata = adata_cr)
```
