# CaMa-Flood-GPU

## Introduction

**CaMa-Flood-GPU** is a high-performance, GPU-accelerated re-implementation of the [CaMa-Flood](https://github.com/global-hydrodynamics/CaMa-Flood_v4) hydrodynamic model. This project leverages the [Triton](https://github.com/openai/triton) language and the PyTorch tensor ecosystem to achieve rapid, scalable global river simulations. By using Triton's custom GPU kernels and PyTorch's tensor abstraction, CaMa-Flood-GPU delivers significant speed-ups over both the original Fortran and the MATLAB-based [Mat-CaMa-Flood](https://github.com/Kshy0/Mat-CaMa-Flood) versions.

**Note:** This repository is under active development!

**Development Environment:** This project is currently developed under WSL2 (Windows Subsystem for Linux 2), and only requires that `torch` and `triton` can be installed successfully.

---

## WSL2 & SLURM Configuration Notes

- **WSL2:** Ensure your WSL2 instance can access your GPU (NVIDIA drivers and CUDA toolkit installed on Windows).  
  - It's recommended to follow the [WSL2 CUDA support](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
  - A successful CUDA installation is confirmed when `nvidia-smi` shows your GPU information correctly in the terminal.
- **SLURM:** Coming soon.

---

## Prerequisites

- Python == 3.13.3  
- PyTorch (with CUDA support) == 2.6.0+cu126
- Triton == 3.2.0
- Additional Python libraries (will be auto-installed, but listed here for clarity):
  - `netCDF4`
  - `omegaconf`
  - `ipykernel` (for Jupyter support)
  - and other utility packages as needed

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
For CUDA 12.6, you may use:

```shell
pip3 install torch --index-url https://download.pytorch.org/whl/cu126
```

> **Note:** By default, `triton` will be installed automatically when you install PyTorch.

### 3. Install other dependencies

```shell
pip install -e .
```

This command installs the `CMF_GPU` package in editable mode, along with its required dependencies such as `netCDF4`, `omegaconf`, `ipykernel`, and others.

---

## Quick Start

- ### 1. Prepare data

  - CaMa-Flood-GPU is fully compatible with CaMa-Flood input data (river maps, runoff, etc.).
  - Download the required datasets from the [official CaMa-Flood site](https://hydro.iis.u-tokyo.ac.jp/~yamadai/cama-flood/) or follow the instructions in the original CaMa-Flood documentation.
  - Typically, you will download a folder named `cmf_v420_pkg` from the official site and place it somewhere on your local machine.

  ### 2. Configure your run

  **You only need to modify the following 4 paths in `configs/test1-glb_15min.yaml`:**

  - `working_dir`: Set this to the path of this project (i.e., the directory where you cloned CaMa-Flood-GPU).
  - `map_dir`, `hires_map_dir`, `base_dir`: These three paths are easy to modify once you have downloaded `cmf_v420_pkg` (available on the CaMa-Flood official website). Just point them to the corresponding folders on your machine.

  Other parameters usually do not require changes unless you have specific needs.

  ### 3. Generate parameters

  Run all cells in `CMF_GPU/generate_parameters.ipynb`.  

  If your `test1-glb_15min.yaml` is set up correctly, you should not need to modify the notebook—simply execute all cells. This process will generate a refined `config.yaml` in the `inp/test1-glb_15min/` folder, ready for model execution.

  ### 4. Run the model

  ```shell
  cd /path/to/CaMa-Flood-GPU
  python CMF_GPU/main.py ./inp/test1-glb_15min/config.yaml
  ```

  - [Optional] For multi-GPU or distributed runs, please refer to upcoming documentation and code samples.


---

## Features

- **Ultra-fast computation:** Even with the standard CPython interpreter, CaMa-Flood-GPU achieves superior performance.  
  **Benchmark:** `test1-glb_15min`, simulation from 2000-01-01 to 2000-12-31 with `enable_adaptive_time_step` enabled runs in ~32 seconds (single 4070Ti GPU + i7-13700 CPU)—even faster than the MATLAB version.

- **Asynchronous I/O:** The model supports asynchronous data loading and dumping via `Dataloader.py` and `Datadumper.py`, ensuring that I/O does not bottleneck simulation speed.

- **Modular design:** The codebase is structured for easy expansion and maintenance, allowing users to add or replace modules as needed.

- **Scalable architecture:** While the current release focuses on single-GPU execution, the code is being actively developed to support single-machine multi-GPU and distributed multi-machine, multi-GPU execution (the architecture has been designed with scalability in mind, but multi-GPU setups are not yet tested).

---

## Roadmap & To-Do

1. **Triton autotune support:** Add support for Triton's autotune functionality to automatically find the best kernel parameters for your specific GPU, maximizing performance.
2. **Multi-GPU/Distributed support:** Enable out-of-the-box single-node multi-GPU and multi-node execution.
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