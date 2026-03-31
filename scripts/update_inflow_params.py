"""Add inflow gauge parameters to an existing parameters.nc.

Usage::

    python update_inflow_params.py /path/to/parameters.nc 12345 67890 11111

This appends ``inflow_catchment_id`` and ``inflow_basin_id`` on a new
``inflow_gauge`` dimension so that the model can validate which catchments
receive observed-discharge injection.
"""

from pathlib import Path

import numpy as np
from hydroforge.modeling.distributed import find_indices_in
from netCDF4 import Dataset


def update_inflow_params(
    parameter_nc: Path,
    gauge_catchment_ids: np.ndarray,
    verbose: bool = True,
) -> None:
    """Append inflow gauge variables to *parameter_nc*.

    Creates dimension ``inflow_gauge`` and variables:

    - ``inflow_catchment_id``  *(inflow_gauge,)*  int64
    - ``inflow_basin_id``      *(inflow_gauge,)*  int64

    Parameters
    ----------
    parameter_nc : Path
        Path to the existing parameters.nc file.
    gauge_catchment_ids : np.ndarray
        Integer catchment IDs where gauge inflow should be injected.
    verbose : bool
        Print summary.
    """
    parameter_nc = Path(parameter_nc)
    gauge_catchment_ids = np.asarray(gauge_catchment_ids, dtype=np.int64)

    # Look up basin IDs for distributed sharding
    with Dataset(str(parameter_nc), "r") as ds:
        all_cids = np.asarray(ds.variables["catchment_id"][:]).astype(np.int64)
        all_basins = np.asarray(ds.variables["catchment_basin_id"][:]).astype(np.int64)

    idx = find_indices_in(gauge_catchment_ids, all_cids)
    if np.any(idx < 0):
        bad = gauge_catchment_ids[idx < 0].tolist()
        raise ValueError(f"Gauge catchment IDs not found in parameters.nc: {bad}")

    inflow_basin_id = all_basins[idx]
    n = len(gauge_catchment_ids)

    with Dataset(str(parameter_nc), "r+") as ds:
        if "inflow_gauge" not in ds.dimensions:
            ds.createDimension("inflow_gauge", n)

        def _put(name, data, dtype, long_name=""):
            if name in ds.variables:
                ds.variables[name][:] = data
            else:
                v = ds.createVariable(
                    name, dtype, ("inflow_gauge",), zlib=True, complevel=4,
                )
                v[:] = data
                if long_name:
                    v.setncattr("long_name", long_name)

        _put(
            "inflow_catchment_id",
            gauge_catchment_ids,
            "i8",
            "catchment id where gauge inflow is injected",
        )
        _put(
            "inflow_basin_id",
            inflow_basin_id,
            "i8",
            "basin id for each inflow gauge (for distributed sharding)",
        )

    if verbose:
        print(f"[update_inflow_params] Written {n} inflow gauges to {parameter_nc}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Add inflow gauge catchment IDs to parameters.nc"
    )
    parser.add_argument("parameter_nc", type=str, help="Path to parameters.nc")
    parser.add_argument(
        "catchment_ids",
        type=int,
        nargs="+",
        help="Catchment IDs for inflow injection",
    )
    args = parser.parse_args()

    update_inflow_params(
        Path(args.parameter_nc),
        np.array(args.catchment_ids, dtype=np.int64),
    )
