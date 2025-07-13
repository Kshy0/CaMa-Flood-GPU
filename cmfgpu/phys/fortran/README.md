# Fortran Reference Implementation

This folder contains a reference implementation of the original **CaMa-Flood** model in Fortran, adapted to validate the **numerical stability and accuracy** of the new GPU-accelerated Triton version.


## Key Notes

* **Not part of the main simulation code**: This folder is **independent** of the production pipeline. It exists solely for **developer-level testing and validation**.
* **Python-Fortran bridging**: The variable name mapping between Fortran and Python is implemented in [`Conventions.py`](./Conventions.py), allowing consistent data transformation for cross-language testing.
* **Pure function transformation**: The original Fortran code has been **slightly modified**:

  * Converted to **pure functions** (no side effects, no global state)
  * All real variables are now **double-precision**

## Compilation Instructions

To use the Fortran modules in Python:

1. You may need to install the following build tools beforehand:

   ```bash
   pip install meson ninja
   ```

2. **Compile the code** using the provided script:

   ```bash
   bash compile.sh
   ```

3. The compilation process uses [**NumPy’s f2py**](https://numpy.org/doc/stable/f2py/) to generate Python-callable modules.

## Example Usage

See `do_one_substep_fortran()` in your Python integration code for how the Fortran routines are invoked. This function coordinates:

* Flow and storage updates
* River-flood interaction logic
* Bifurcation handling (optional)

These routines act as direct counterparts to the Triton kernels, enabling side-by-side step comparisons for validation.
