from setuptools import setup, find_packages
setup(
    name='cmfgpu',
    version='0.1',
    packages=find_packages(),
    install_requires=['netCDF4', 'h5py','scipy', 'numba', 'pydantic'],
)