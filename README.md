# CaMa-Flood-GPU

**CaMa-Flood-GPU** is a high-performance, GPU-accelerated re-implementation of the [CaMa-Flood](https://github.com/global-hydrodynamics/CaMa-Flood_v4) hydrodynamic model. This project leverages the [Triton](https://github.com/openai/triton) language and the [PyTorch](https://github.com/pytorch/pytorch) tensor ecosystem to achieve rapid, scalable global river simulations. By using Triton's custom GPU kernels and PyTorch's tensor abstraction, CaMa-Flood-GPU delivers significant speed-ups over both the original Fortran and the MATLAB-based versions introduced during [the CaMa-Flood developer/user international meeting 2024](https://global-hydrodynamics.github.io/cmf-meet-2024/).

**Note:** This repository is under active development, and both the code structure and content are subject to significant changes at any time.

**Development Environment:** This project is currently developed under WSL2 (Windows Subsystem for Linux 2), and requires that `torch` and `triton` can be installed successfully.

**Target Audience:** This project is intended for advanced users who are already familiar with the original CaMa-Flood model. Users are strongly advised to run the original [CaMa-Flood](https://github.com/global-hydrodynamics/CaMa-Flood_v4) first to understand the data structure, input specifications, and general workflow before attempting to use this GPU-accelerated version.

---

## Prerequisites

- Python == 3.13.11  
- PyTorch (with CUDA support) == 2.9.1+cu130
- Triton == 3.5.1
- Additional Python libraries (will be auto-installed, but listed here for clarity):
  - pydantic (for better data validation)
  - netCDF4
  - and other utility packages as needed

The installable version of torch depends on your system. This project will always rely on the official latest releases of torch and triton for the newest features and optimal performance. Tests have confirmed that the project can also run with torch 2.6.0 and CUDA 12.4.

In theory, the codebase should also run on AMD GPUs, but I haven’t had the chance to test that setup yet.

---

## Installation

### 1. Clone the repository

```shell
git clone https://github.com/Kshy0/CaMa-Flood-GPU.git
cd CaMa-Flood-GPU
```

### 2. Install PyTorch

It is recommended to use a virtual environment (`venv` or `conda`).

Please follow the official [PyTorch installation guide](https://pytorch.org/get-started/locally/) for your environment. 
For CUDA 13.0, you may use:

```shell
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

> **Note:** By default, `triton` will be installed automatically when you install PyTorch. You don't need to install packages `torchvision` or `torchaudio` as stated in the official manual.
>
> Sometimes, the above command may not be compatible with your system. For example, on some clusters running older systems, you can use `pip index versions torch` to check the latest torch version supported by your environment, and then select a suitable torch–CUDA combination from [the PyTorch previous versions page](https://pytorch.org/get-started/previous-versions/). You do not need to install CUDA separately, as the torch wheel package already includes a precompiled CUDA runtime. Just make sure your GPU driver is correctly installed, and that the chosen CUDA version is compatible according to [the NVIDIA CUDA compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

### 3. Install other dependencies

```shell
pip install -e .
```

This command installs the `cmfgpu` package in editable mode, along with its required dependencies such as `netCDF4`, `scipy`, and others.

---

## Quick Start

All scripts in this repository are designed for maximum flexibility. Before running any script (such as `scripts/make_map_params.py` for model parameters or `scripts/make_runoff_map.py` for runoff mapping), **you must manually set the correct file and directory paths** inside the script. This may include:

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
    visualize_basins=False,
    basin_use_file=False,  # Set to True to use basin.bin for pruning bifurcations
    target_gpus=4
)
```

The `target_gpus` parameter specifies the number of GPUs for basin workload distribution.
The `basin_use_file` parameter (default False) controls whether to use a `basin.bin` file to cut bifurcations crossing basin boundaries, which can help balance loads in high-resolution maps.
When enabled, the system will prune bifurcation paths that connect different basins as defined in `basin.bin`, and report both pruned and unpruned load distributions.
Regardless of this setting, the system always computes and reports GPU load assignments using LPT (Longest Processing Time) scheduling, and warns if the load imbalance exceeds 10%.

Note: The `basin.bin` file is originally generated by the CPU version's `set_bif_basin.F90`, designed for 16-32 MPI processes. Since GPU counts are typically low, this option is usually set to False. Consider enabling it only when you have many GPUs and require fine-grained simulations.

- ### 1. Prepare data

  - CaMa-Flood-GPU is fully compatible with CaMa-Flood input data (river maps, runoff, etc.).
  - Download the required datasets from the [official CaMa-Flood site](https://hydro.iis.u-tokyo.ac.jp/~yamadai/cama-flood/) or follow the instructions in the original CaMa-Flood documentation.
  - Typically, you will download a folder named `cmf_v420_pkg` from the official site and place it somewhere on your local machine.
  
  ### 2. Generate parameters

  For `glb_15min` maps, the `cmf_v420_pkg` contains the estimated river channel parameters. So no additional work is required to run this GPU program. If you want to use a higher resolution map, please refer to the instructions in the original Fortran repository. You need to compile the Fortran code and [generate river channel parameters](https://github.com/global-hydrodynamics/CaMa-Flood_v4/blob/master/map/src/src_param/s01-channel_params.sh) , such as `rivhgt.bin`, `rivwth_gwdlr.bin`.
  
  ```shell
  cd /path/to/CaMa-Flood-GPU
  # Edit scripts/make_map_params.py to set your map paths, then run:
  python ./scripts/make_map_params.py
  ```
  
  > **Note:** Please re-execute the code when you get the update from git!
  
  ### 3. Generate runoff input map
  
  In order to use external runoff datasets with CaMa-Flood-GPU, we first need to **generate a mapping table** that links **runoff grid cells** to the corresponding **catchments**.
  
  This repository includes dataset classes for both binary (`.bin`) and NetCDF (`.nc`) input formats. Each dataset type is defined in its own script under the `./datasets/` directory (e.g., `./datasets/daily_bin_dataset.py`), and often includes example usage in the `if __name__ == "__main__":` block.
  
  For convenience, we provide a dedicated script `scripts/make_runoff_map.py` to generate the mapping table using `DailyBinDataset`. You can modify it to use other dataset classes (like `NetCDFDataset` or `ERA5LandDataset`) if needed.

  These classes include built-in methods such as `generate_runoff_mapping_table()` to create the required mapping `.npz` file.
  
  ```shell
  cd /path/to/CaMa-Flood-GPU
  # Edit scripts/make_runoff_map.py to set your runoff paths, then run:
  python ./scripts/make_runoff_map.py
  ```
  
  Once created, the `.npz` file will contain a **sparse matrix** mapping each runoff grid cell to the affected catchments, which is then used during simulation.
  
  
  ### 4. Run the model
  
  Several script templates are provided in the `./scripts` directory, such as `run_daily_bin.py` and `netcdf_nc.py`. You are encouraged to modify or customize these scripts to fit your specific workflow and simulation requirements.
  
  For 1 GPU on a single machine, simply run:
  
  ```shell
  cd /path/to/CaMa-Flood-GPU
  python ./scripts/run_daily_bin.py
  ```
  
  There's no need for complex multi-GPU configurations! As long as each catchment's basin is already specified (`catchment_basin_id` in `merit_map.py`), CaMa-Flood-GPU will automatically distribute tasks across different GPUs, ensuring a balanced workload.
  
  For 4 GPUs on a single machine:
  
  ```shell
  cd /path/to/CaMa-Flood-GPU
  torchrun --nproc_per_node=4 ./scripts/run_daily_bin.py
  ```
  
  - For distributed runs, please refer to upcoming documentation and code samples.
  
  ### [Optional] Choosing Block Size for Optimal Performance
  
  The `block_size` parameter varies depending on your hardware, impacting memory usage and computational efficiency. To find the optimal size for your system, run a benchmark across typical values `[64, 128, 256, 512, 1024]`. Smaller block sizes may improve memory utilization, while larger ones can speed up computation by reducing kernel launch overhead. Use the provided `benchmark_block_size()` function to test and select the best block size for your setup.
  
  ```bash
  cd /path/to/CaMa-Flood-GPU
  python ./scripts/benchmark_block_size.py
  ```
  
  Once the benchmark is complete, select the block size that provides the best balance between performance and resource usage for your system.


---

## Features

- **Ultra-fast computation:** Even with the standard CPython interpreter, CaMa-Flood-GPU achieves superior performance. 
  **Benchmark:** `test1-glb_15min`, simulation from 2000-01-01 to 2000-12-31 with `adaptive_time_step` enabled runs in ~32 seconds (single 4070Ti GPU + i7-13700 CPU)—even faster than the MATLAB version.
- **Modular design:** The codebase is structured for easy expansion and maintenance, allowing users to add or replace modules as needed.
- **Scalable architecture:** This codebase is designed to be suitable for multi-node, multi-GPU. It has the ability to simulate floods on extremely high-resolution geographic maps.

---

## Disclaimer

CaMa-Flood-GPU is is intended for research, development, and educational purposes. Please verify results independently before relying on them for critical applications.

---

## License

CaMa-Flood-GPU follows the same Apache 2.0 license as CaMa-Flood, but the datasets are provided under a different license.

---

## Contact

For questions, bug reports, or contributions, please open an [issue](https://github.com/Kshy0/CaMa-Flood-GPU/issues) or contact the maintainer (Shengyu Kang): kshy0204@whu.edu.cn