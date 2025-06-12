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
    export OMP_NUM_THREADS=4
    ```

  - Job submission via `sbatch` is being tested and will be updated soon.

---

## Prerequisites

- Python == 3.13.3  
- PyTorch (with CUDA support) == 2.7.1+cu128
- Triton == 3.3.1
- Additional Python libraries (will be auto-installed, but listed here for clarity):
  - netCDF4
  - omegaconf
  - h5py
  - and other utility packages as needed

The codebase dependencies of this program are not strict, and I think any newer version of torch will run smoothly. For example, I also successfully tested torch 2.7.0 with CUDA 12.6 version, even though CUDA 12.2 is installed on the cluster. This codebase will always rely on newer versions of python, torch, and triton for the latest feature support and optimal performance.

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

This command installs the `CMF_GPU` package in editable mode, along with its required dependencies such as `netCDF4`, `omegaconf`, `h5py`, and others.

If you later clone or pull a newer version of the repository and notice that `setup.py` includes updated or additional dependencies, it is recommended to uninstall `CMF_GPU` and reinstall it to ensure all required packages are correctly installed:

```bash
pip uninstall CMF_GPU
pip install -e .
```

---

## Quick Start

- ### 1. Prepare data

  - CaMa-Flood-GPU is fully compatible with CaMa-Flood input data (river maps, runoff, etc.).
  - Download the required datasets from the [official CaMa-Flood site](https://hydro.iis.u-tokyo.ac.jp/~yamadai/cama-flood/) or follow the instructions in the original CaMa-Flood documentation.
  - Typically, you will download a folder named `cmf_v420_pkg` from the official site and place it somewhere on your local machine.
  - For `glb_15min` maps, the `cmf_v420_pkg` contains the estimated river channel parameters. So no additional work is required to run this GPU program. If you want to use a higher resolution map, please refer to the instructions in the original Fortran repository. You need to compile the Fortran code and [generate river channel parameters](https://github.com/global-hydrodynamics/CaMa-Flood_v4/blob/master/map/src/src_param/s01-channel_params.sh) , such as river width.
  
  ### 2. Configure your run
  
  **You only need to modify the following paths in `configs/glb_15min.yaml`:**
  
  - `parameter_config`: 
    1. `working_dir`: Set this to the path of this project (i.e., the directory where you cloned CaMa-Flood-GPU).
    2. `map_dir`, `hires_map_dir`: These three paths are easy to modify once you have downloaded `cmf_v420_pkg` (available on the CaMa-Flood official website). Just point them to the corresponding folders on your machine.
  - `runoff_config`: 
    1. `base_dir`: The location where the "test_1deg" runoff data is stored.
  
  Other parameters usually do not require changes unless you have specific needs.
  
  ### 3. Generate parameters
  
  ```shell
  cd /path/to/CaMa-Flood-GPU
  python ./CMF_GPU/generate_parameters.py ./configs/glb_15min.yaml
  ```
  
  This script generates the parameter file `parameters.h5`, the state file `init_states.h5`, and the runoff input matrix `runoff_input_matrix.npz`.
  
  > **Note:** Please re-execute `generate_parameters.py` when you get the update from git!
  
  ### 4. Run the model
  
  For 4 GPUs on a single machine:
  
  ```shell
  cd /path/to/CaMa-Flood-GPU
  torchrun --nproc_per_node=4 CMF_GPU/main.py --config ./configs/glb_15min.yaml
  ```
  
  - [Optional] For distributed runs, please refer to upcoming documentation and code samples.


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