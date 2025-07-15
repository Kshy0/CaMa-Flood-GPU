# CaMa-Flood-GPU

## Introduction

**CaMa-Flood-GPU** is a high-performance, GPU-accelerated re-implementation of the [CaMa-Flood](https://github.com/global-hydrodynamics/CaMa-Flood_v4) hydrodynamic model. This project leverages the [Triton](https://github.com/openai/triton) language and the [PyTorch](https://github.com/pytorch/pytorch) tensor ecosystem to achieve rapid, scalable global river simulations. By using Triton's custom GPU kernels and PyTorch's tensor abstraction, CaMa-Flood-GPU delivers significant speed-ups over both the original Fortran and the MATLAB-based versions introduced during [the CaMa-Flood developer/user international meeting 2024](https://global-hydrodynamics.github.io/cmf-meet-2024/).

**Note:** This repository is under active development, and both the code structure and content are subject to significant changes at any time.

**Development Environment:** This project is currently developed under WSL2 (Windows Subsystem for Linux 2), and requires that `torch` and `triton` can be installed successfully.

---

## WSL2 & SLURM Configuration Notes

- **WSL2:** Ensure your WSL2 instance can access your GPU (NVIDIA drivers and CUDA toolkit installed on Windows).  
  - It's recommended to follow the [WSL2 CUDA support](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
  - A successful CUDA installation is confirmed when `nvidia-smi` shows your GPU information correctly in the terminal.
  
- **SLURM:** 

  - Ensure that the GPU drivers and CUDA toolkit are correctly installed on your compute cluster.

  - I have applied for a GPU node, connected to the corresponding compute node via `ssh`, and executed the following commands. (please adapt them as needed for your environment — for example, `scl/gcc11.2` may not be available on all systems):

    ```bash
    module load scl/gcc11.2
    OMP_NUM_THREADS=4
    ```
  
- Job submission via `sbatch` is being tested and will be updated soon.

---

## Prerequisites

- Python == 3.13.5  
- PyTorch (with CUDA support) == 2.7.1+cu128
- Triton == 3.3.1
- Additional Python libraries (will be auto-installed, but listed here for clarity):
  - pydantic
  - netCDF4
  - h5py
  - and other utility packages as needed

The codebase dependencies are not strict, and I think any newer version of torch will run smoothly. For example, I also successfully tested torch 2.7.0 with CUDA 12.6 version, even though CUDA 12.2 is installed on the cluster. This codebase will always rely on newer versions of python, torch, and triton for the latest feature support and optimal performance.

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
For CUDA 12.8, you may use:

```shell
pip3 install torch --index-url https://download.pytorch.org/whl/cu128
```

> **Note:** By default, `triton` will be installed automatically when you install PyTorch. You don't need to install packages `torchvision` and `torchaudio` as stated in the official manual.
>
> Sometimes, the above command may not be compatible with your system—for example, your environment might not yet support CUDA 12.8. In that case, you can slightly modify the command (e.g., simply run `pip3 install torch`) to install a version of PyTorch compatible with CUDA 12.6.

### 3. Install other dependencies

```shell
pip install -e .
```

This command installs the `cmfgpu` package in editable mode, along with its required dependencies such as `netCDF4`, `scipy`, `h5py`, and others.

If you later clone or pull a newer version of the repository and notice that `setup.py` includes updated or additional dependencies, it is recommended to uninstall `cmfgpu` and reinstall it to ensure all required packages are correctly installed:

```bash
pip uninstall cmfgpu
pip install -e .
```

---

## Quick Start

All scripts in this repository are designed for maximum flexibility. Before running any script (such as `params/merit_map.py` to generate model parameters), **you must manually set the correct file and directory paths** inside the script. This may include:

- `map_dir`: Path to your CaMa-Flood input map directory (e.g., `cmf_v420_pkg/map/glb_15min`)
- `out_dir`: Path to your desired output directory
- `gauge_file`: Path to the GRDC gauge file (optional)
- Other runtime settings

These paths are currently **hardcoded in the Python script**, typically under the `if __name__ == "__main__":` block at the bottom of the file.
 Please edit them to match your local file structure before execution.

```python
if __name__ == "__main__":
    merit_map = MERITMap(
        map_dir="/your/path/to/map",
        out_dir="/your/path/to/output",
        gauge_file="/your/path/to/gauge_file.txt",  # Optional
        visualize_basins=False
    )
```

- ### 1. Prepare data

  - CaMa-Flood-GPU is fully compatible with CaMa-Flood input data (river maps, runoff, etc.).
  - Download the required datasets from the [official CaMa-Flood site](https://hydro.iis.u-tokyo.ac.jp/~yamadai/cama-flood/) or follow the instructions in the original CaMa-Flood documentation.
  - Typically, you will download a folder named `cmf_v420_pkg` from the official site and place it somewhere on your local machine.
  - For `glb_15min` maps, the `cmf_v420_pkg` contains the estimated river channel parameters. So no additional work is required to run this GPU program. If you want to use a higher resolution map, please refer to the instructions in the original Fortran repository. You need to compile the Fortran code and [generate river channel parameters](https://github.com/global-hydrodynamics/CaMa-Flood_v4/blob/master/map/src/src_param/s01-channel_params.sh) , such as river width.
  
  ### 2. Generate parameters
  
  CaMa-Flood-GPU is fully compatible with CaMa-Flood input data (river maps, runoff, etc.).
  
  For `map_dir`, it is important to note that it should include river channel parameter files such as `rivhgt.bin`, `rivwth_gwdlr.bin`, and `bifprm.txt`, which are generated using the original Fortran code from the CaMa-Flood repository. To produce them, you may need to compile and run the script `s01-channel_params.sh` provided in the original CaMa-Flood repository.
  
  If you are using the `glb_15min` maps from the official CaMa-Flood package, the required river channel parameters are already included. However, if you are using higher resolution maps, you will need to refer to the original Fortran repository to compile the necessary data and generate the required parameters.
  
  ```shell
  cd /path/to/CaMa-Flood-GPU
  python ./cmfgpu/params/merit_map.py
  ```
  
  > **Note:** Please re-execute the code when you get the update from git!
  
  ### 3. Generate runoff input map
  
  In order to use external runoff datasets with CaMa-Flood-GPU, we first need to **generate a mapping table** that links **runoff grid cells** to the corresponding **catchments**.
  
  This repository includes dataset classes for both binary (`.bin`) and NetCDF (`.nc`) input formats. Each dataset type is defined in its own script under the `./datasets/` directory. For example:
  
  - `./datasets/daily_bin_dataset.py` — for daily binary runoff data
  - `./datasets/daily_nc_dataset.py` — for NetCDF-based runoff
  
  These classes include built-in methods such as `generate_runoff_mapping_table()` to create the required mapping `.npz` file.
  
  ```shell
  cd /path/to/CaMa-Flood-GPU
  python ./cmfgpu/datasets/daily_bin_dataset.py
  ```
  
  Once created, the `.npz` file will contain a **sparse matrix** mapping each runoff grid cell to the affected catchments, which is then used during simulation.
  
  Based on our tests, CaMa-Flood-GPU is likely **I/O (CPU) bound**, meaning the hydrodynamic simulation runs faster than the time it takes to read and decode input data (especially from NetCDF files). If you're highly sensitive to runtime performance:
  
  1. Consider converting data to `.bin` format or using `.nc` files with lower compression levels—though this may require significantly more disk space.
  2. If feasible, you can precompute and store runoff data **already mapped to each catchment**, eliminating the need for sparse matrix operations during runtime. This applies especially to cases where the size of the runoff grid is much larger than the size of the catchment. While we've made efforts to decouple I/O from model execution, you still need to understand how CaMa-Flood-GPU assigns basins to different GPUs in multi-GPU runs. You must ensure each rank (based on `rank` and `world_size`) reads only its assigned subset of catchments and handles multi-process data loading correctly.
  3. The framework already uses **multi-process and asynchronous prefetching**, so performance is generally acceptable for most scenarios.
  
  ### 4. Run the model
  
  Several script templates are provided in the `./scripts` directory, such as `run_daily_bin.py` and `run_daily_nc.py`. You are encouraged to modify or customize these scripts to fit your specific workflow and simulation requirements.
  
  For 1 GPU on a single machine, simply run:
  
  ```shell
  cd /path/to/CaMa-Flood-GPU
  python ./scripts/run_daily_bin.py
  ```
  
  For 4 GPUs on a single machine:
  
  ```shell
  cd /path/to/CaMa-Flood-GPU
  torchrun --nproc_per_node=4 ./scripts/run_daily_bin.py
  ```
  
  - [Optional] For distributed runs, please refer to upcoming documentation and code samples.
  
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

## Roadmap & To-Do

1. **Triton autotune support:** Add support for Triton's autotune functionality to automatically find the best kernel parameters for your specific GPU, maximizing performance.
4. **Parameter calibration:** Develop tools for easier model tuning and result analysis.

---

## Disclaimer

CaMa-Flood-GPU is in early-stage development and has not been comprehensively validated for scientific or operational use. It is intended for research, development, and educational purposes. Please verify results independently before relying on them for critical applications.

---

## License

Coming soon.

---

## Contact

For questions, bug reports, or contributions, please open an [issue](https://github.com/Kshy0/CaMa-Flood-GPU/issues) or contact the maintainer (Shengyu Kang): kshy0204@whu.edu.cn