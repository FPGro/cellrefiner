# Reconstruct single-cell resolution from spatial transcriptomics with CellRefiner

 CellRefiner is a physical model-based method that integrates a scRNA-seq dataset with a paired spatial transcriptomics (ST) dataset to generate single-cell resolution in the imputed ST data. CellRefiner models cells as particles connected by forces, and then optimizes cell locations with spatial proximity constraints, gene expression similarity, and ligand-receptor interactions between cells.

# Installation
CellRefiner requires Python >= 3.9 and NVIDIA GPU with CUDA support.

**Step 1: Create and activate a virtual environment**

We recommend using Conda or Mamba for environment management. To install Mamba, see https://github.com/conda-forge/miniforge.
```bash
mamba create -n cellrefiner_env python=3.9
mamba activate cellrefiner_env
```

**Step 2: Install cellrefiner from GitHub**

```bash
pip install git+https://github.com/XiangyuKuang/cellrefiner.git
```

**Step 3: Install CuPy (Optional)**

CuPy installation will accelerate `cellrefiner.preprocessing.spatial_mapping()`

We recommend installing CuPy and CUDA toolkit via conda to avoid compatibility issues:

```bash
mamba install cupy cudatoolkit -c conda-forge
```

**Expected installation time:** Approximately 2 minutes on a desktop computer.

# Usage

For detailed examples, please refer to the `tutorials` folder.

Import packages

```python
import squidpy as sq
import cellrefiner as cr
```

Load spatial transcriptomics and scRNA-seq datasets (available via the Squidpy package).

```python
adata_st = sq.datasets.visium_fluo_adata_crop()
adata_st = adata_st[adata_st.obs.cluster.isin([f"Cortex_{i}" for i in np.arange(1, 5)])].copy() # select cortex region
adata_sc = sq.datasets.sc_mouse_cortex()
```

Load ligand-receptor database

```python
db_lr = cr.pp.ligand_receptor_database(species = 'mouse')
```

Map cells to spots and refine the spatial locations of mapped cells
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
db_lr = cr.pp.filter_lr_database(db_lr, adata_cr)
cr.tl.contact_communication(db_lr, adata = adata_cr)
```

**Expected run time:** Approximately 7 minutes on a desktop computer with NVIDIA GPU.

# Documentation

See detailed documentation at https://cellrefiner.readthedocs.io/en/latest/