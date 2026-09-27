# CaMa-Flood-GPU

**CaMa-Flood-GPU** is a GPU-accelerated re-implementation of the [CaMa-Flood](https://github.com/global-hydrodynamics/CaMa-Flood_v4) hydrodynamic model. Built on [PyTorch](https://github.com/pytorch/pytorch), it runs on NVIDIA, AMD, and Apple GPUs through CUDA/HIP, [Triton](https://github.com/openai/triton), and Metal backends. It supports global river simulations on Linux, Windows, and macOS, with multi-GPU execution where the platform and backend support it.

**Note:** This repository is under active development, and both the code structure and content are subject to significant changes at any time.

**Runtime setup:** Install a GPU-enabled `torch` build for your platform and a compatible `triton` package when using the Triton backend. The model's GPU kernels compile when first used; no manual kernel build is needed. On macOS, the Metal backend uses PyTorch MPS and does not require Triton.

**Target Audience:** This project is intended for advanced users who are already familiar with the original CaMa-Flood model. Users are strongly advised to run the original [CaMa-Flood](https://github.com/global-hydrodynamics/CaMa-Flood_v4) first to understand the data structure, input specifications, and general workflow before attempting to use this GPU-accelerated version.

---

## Citation

If you find CaMa-Flood-GPU helpful for your research or applications, we would appreciate it if you cite our paper:

**BibTeX**

```bibtex
@article{kang2026camafloodgpu,
  author  = {Kang, Shengyu and Yin, Jiabo and Yamazaki, Dai},
  title   = {CaMa-Flood-GPU: a GPU-based hydrodynamic model implementation for scalable global simulations},
  journal = {Geoscientific Model Development},
  year    = {2026},
  volume  = {19},
  number  = {12},
  pages   = {5623--5640},
  doi     = {10.5194/gmd-19-5623-2026},
  url     = {https://gmd.copernicus.org/articles/19/5623/2026/}
}
```

**APA**

Kang, S., Yin, J., & Yamazaki, D. (2026). CaMa-Flood-GPU: A GPU-based hydrodynamic model implementation for scalable global simulations. Geoscientific Model Development, 19(12), 5623–5640. https://doi.org/10.5194/gmd-19-5623-2026

---

## Prerequisites

- Python >= 3.11
- A GPU-enabled PyTorch build for your platform; install Triton when using the Triton backend
- Additional Python libraries (will be auto-installed, but listed here for clarity):
  - pydantic (for better data validation)
  - netCDF4
  - and other utility packages as needed

Choose a PyTorch build compatible with your GPU and operating system. The available GPU paths are NVIDIA CUDA on Linux and Windows, AMD ROCm on Linux, and Apple Metal on macOS. Windows users can also run the Linux CUDA path through WSL2.

---

## Installation

### 1. Clone the repository

```shell
git clone https://github.com/Kshy0/CaMa-Flood-GPU.git
cd CaMa-Flood-GPU
```

### 2. Install PyTorch and Triton

It is recommended to use a virtual environment (`venv` or `conda`):

```shell
conda create -n CMF python=3.14.*
conda activate CMF
```

#### 2a. NVIDIA CUDA example

Choose the build for your GPU from the official [PyTorch installation guide](https://pytorch.org/get-started/locally/). The command below is an NVIDIA CUDA example; AMD ROCm and macOS MPS use different builds.
For CUDA 13.2, you may use:

```shell
pip install torch --index-url https://download.pytorch.org/whl/cu132
```

> **Note:** On supported CUDA and ROCm installations, PyTorch may install Triton automatically. Verify that `import triton` works when using the Triton backend. For native Windows, use [triton-windows](https://github.com/triton-lang/triton-windows); on macOS, use the Metal backend without Triton. `torchvision` and `torchaudio` are **NOT** required.

#### 2b. Older GPUs / legacy systems

> **Warning:** Starting from PyTorch **2.11.0**, the pre-built wheels no longer include kernels for compute capability 7.0 (e.g. V100). Using these wheels on such GPUs will result in an `Unsupported compute capability` error. Install PyTorch ≤ 2.10.0 instead.

For older GPUs, pin a compatible torch version. For example, with CUDA 12.8:

```shell
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

You can use `pip index versions torch` to check the latest torch version supported by your environment, and then select a suitable torch–CUDA combination from [the PyTorch previous versions page](https://pytorch.org/get-started/previous-versions/). You do not need to install CUDA separately, as the torch wheel package already includes a precompiled CUDA runtime. Just make sure your GPU driver is correctly installed, and that the chosen CUDA version is compatible according to [the NVIDIA CUDA compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

### 3. Install Hydroforge

[Hydroforge](https://github.com/Kshy0/hydroforge) is the underlying abstract framework used by this project. Install it with:

```shell
pip install git+https://github.com/Kshy0/hydroforge.git
```

### 4. Install other dependencies

```shell
pip install -e .
```

This command installs the `cmfgpu` package in editable mode, along with its required dependencies such as `netCDF4`, `scipy`, and others.

---

## Updating

When pulling the latest version of this repository, **remember to also update `Hydroforge`**, as the two packages are developed in tandem and API changes in one may require a matching update in the other:

```shell
git pull
conda activate CMF
pip uninstall hydroforge -y
pip install --upgrade git+https://github.com/Kshy0/hydroforge.git
```

---

## Quick Start

> **Tip:** The `scripts/` folder is version-controlled and may be updated with each `git pull`. To avoid losing your local path edits, it is recommended to copy it to a personal working copy before making changes:
> ```shell
> cp -r scripts scripts_user
> ```
> Use `scripts_user/` for your day-to-day work. When you pull updates, check `scripts/` for any new or changed templates and merge them into your copy as needed.

All scripts in this repository are designed for maximum flexibility. Before running any script (such as `scripts_user/make_map_params.py` for model parameters or `scripts_user/make_runoff_map.py` for runoff mapping), **you must manually set the correct file and directory paths** inside the script. This may include:

- `map_dir`: Path to your CaMa-Flood input map directory (e.g., `cmf_v420_pkg/map/glb_15min`)
- `out_dir`: Path to your desired output directory
- `gauge_file`: Path to the GRDC gauge file (optional)
- Other runtime settings

These paths are currently **hardcoded in the Python script**.
 Please edit them to match your local file structure before execution.

```python
# ...
merit_map = MERITMap(
    map_dir="/your/path/to/map",
    out_dir="/your/path/to/output",
    bifori_file="/your/path/to/bifori.txt",  # Optional
    gauge_file="/your/path/to/gauge_file.txt",  # Optional
    visualized=False,
    target_gpus=4
)
```

The `target_gpus` parameter specifies the number of GPUs for basin workload distribution.

Regardless of this setting, the system always computes and reports GPU load assignments using LPT (Longest Processing Time) scheduling, and warns if the load imbalance exceeds 10%.

- ### 1. Prepare data

  - CaMa-Flood-GPU is fully compatible with CaMa-Flood input data (river maps, runoff, etc.).
  - Download the required datasets from the [official CaMa-Flood site](https://hydro.iis.u-tokyo.ac.jp/~yamadai/cama-flood/) or follow the instructions in the original CaMa-Flood documentation.
  - Typically, you will download a folder named `cmf_v420_pkg` from the official site and place it somewhere on your local machine.
  
  ### 2. Generate parameters

  Run `make_map_params.py` to convert raw map files into `parameters.nc`. This step is always required.

  For `glb_15min` maps, the `cmf_v420_pkg` already includes pre-estimated river channel parameters (width, height, etc.), so `make_map_params.py` alone is sufficient. For higher resolutions, you additionally need to run `update_river_params.py` to estimate river geometry from scratch.

  `update_river_params.py` reads a **daily runoff climatology** (e.g., `ELSE_GPCC_dayclm-1981-2010.one`, a 365-day global climatology bundled with `cmf_v420_pkg`), accumulates discharge along the flow network, and applies power-law scaling with optional satellite-width fusion. This is the Python equivalent of `calc_outclm`, `calc_rivwth`, and `set_gwdlr` in the original Fortran coded.

  You can also use your own runoff data to compute the climatology. The built-in dataset classes (`DailyBinDataset`, `NetCDFDataset`, `ERA5LandAccumDataset`, etc.) all provide an `export_climatology()` method that aggregates any time-series runoff into a catchment-level mean-annual climatology NetCDF, which can then be fed directly into `estimate_river_geometry()`.

  ```shell
  cd /path/to/CaMa-Flood-GPU
  # Edit scripts to set your paths, then run:
  python ./scripts_user/make_map_params.py
  python ./scripts_user/update_river_params.py
  ```
  
  > **Note:** Please re-execute these scripts when you pull updates from git. In most cases the output will be identical, but some updates may introduce new features or improved defaults that require regeneration.
  
  ### 3. Generate runoff input map
  
  In order to use external runoff datasets with CaMa-Flood-GPU, we first need to **generate a mapping table** that links **runoff grid cells** to the corresponding **catchments**.
  
The [Hydroforge](https://github.com/Kshy0/hydroforge) dependency includes dataset classes for both binary (`.bin`) and NetCDF (`.nc`) input formats. Each dataset type is defined in its own module under `hydroforge.data.datasets` (e.g., `hydroforge/datasets/daily_bin.py`).
  
  For convenience, we provide a script `scripts_user/make_runoff_map.py` to generate the mapping table using `DailyBinDataset`. You can modify it to use other dataset classes (like `NetCDFDataset` or `ERA5LandAccumDataset`) if needed.

  These classes include built-in methods such as `generate_mapping_table()` to create the required mapping `.npz` file.
  
  ```shell
  cd /path/to/CaMa-Flood-GPU
  # Edit scripts_user/make_runoff_map.py to set your runoff paths, then run:
  python ./scripts_user/make_runoff_map.py
  ```
  
  Once created, the `.npz` file will contain a **sparse matrix** mapping each runoff grid cell to the affected catchments, which is then used during simulation.
  
  
  ### 4. Run the model
  
  Several script templates are provided in the `./scripts` directory, such as `run_daily_bin.py` and `run_netcdf.py`. After copying to `scripts_user/`, you are encouraged to modify or customize these scripts to fit your specific workflow and simulation requirements.
  
  For 1 GPU on a single machine, simply run:
  
  ```shell
  cd /path/to/CaMa-Flood-GPU
  python ./scripts_user/run_daily_bin.py
  ```
  
  There's no need for complex multi-GPU configurations! As long as each catchment's basin is already specified (`catchment_basin_id` in `merit_map.py`), CaMa-Flood-GPU will automatically distribute tasks across different GPUs, ensuring a balanced workload.
  
  For 4 GPUs on a single machine:
  
  ```shell
  cd /path/to/CaMa-Flood-GPU
  torchrun --nproc_per_node=4 ./scripts_user/run_daily_bin.py
  ```

---

## Disclaimer

CaMa-Flood-GPU is intended for research, development, and educational purposes. Please verify results independently before relying on them for critical applications.

---

## License

CaMa-Flood-GPU follows the same Apache 2.0 license as CaMa-Flood, but the datasets are provided under a different license.

---

## Contact

For questions, bug reports, or contributions, please open an [issue](https://github.com/Kshy0/CaMa-Flood-GPU/issues) or contact the maintainer (Shengyu Kang): kshy0204@whu.edu.cn
