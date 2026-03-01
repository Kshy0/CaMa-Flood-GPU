# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
Example: estimate dam / reservoir parameters from aggregated outflow statistics.

1. Run Python dam allocation to map GRanD dams to the CaMa grid::

       hires = HiResMap(map_dir="/path/to/glb_15min")
       hires.build_dams("/path/to/GRanD_allocated.csv")

   This writes ``dam_alloc.txt`` with proper catchment IDs.

2. Run a naturalized (no-dam) CaMa-Flood simulation with the aggregator
   configured to save yearly max and mean of ``total_outflow``::

       variables_to_save = {
           "max_mean": ["total_outflow"],
           "mean_mean": ["total_outflow"],
       }

3. Call ``estimate_dam_params`` with the aggregator output NC and the
   allocation result.  The function reads annual-max peaks, fits Gumbel
   100-yr return period to derive Qf, and estimates flood / conservation
   storage.
4. Output both a Fortran-compatible CSV and updated ``parameters.nc``.

Prerequisites
-------------
* A ``GRanD_allocated.csv`` from the CaMa-Flood v4.20 package.
* A ``dam_alloc.txt`` from ``HiResMap.build_dams()``.
* Aggregator output NC with yearly ``total_outflow_max_mean`` and
  ``total_outflow_mean_mean`` variables.
* A ``parameters.nc`` (from ``MERITMap``).
"""

from pathlib import Path

from cmfgpu.params import estimate_dam_params


def main():
    ### Configuration Start ###
    resolution = "glb_15min"
    base_dir = Path("/home/eat/CaMa-Flood-GPU")
    inp_dir = base_dir / "inp" / resolution

    parameter_nc = inp_dir / "parameters.nc"
    # GRanD dam-list CSV from CaMa-Flood v4.20 package
    cmf_pkg = Path("/home/eat/cmf_v420_pkg")
    dam_list = cmf_pkg / "map" / "data" / "GRanD_allocated.csv"
    # Dam allocation result from HiResMap.build_dams()
    dam_alloc = cmf_pkg / "map" / "glb_15min" / "dam_alloc.txt"

    # Aggregator output NC from naturalized simulation
    # (must contain total_outflow_max_mean and total_outflow_mean_mean)
    outflow_stats_nc = base_dir / "out" / "glb_15min_dam_natural_nc"

    # Output paths
    output_csv = base_dir / "output" / "dam_params.csv"
    output_nc = inp_dir / "parameters_dam.nc"  # new copy with reservoir vars

    # Optional GRSAD / ReGeom directories (set to None to use 37% fallback)
    grsad_dir = base_dir / "inp" / "GRSAD"
    regeom_dir = base_dir / "inp" / "ReGeom"
    ### Configuration End ###

    # ------------------------------------------------------------------
    # Estimate dam parameters from aggregated outflow statistics
    # ------------------------------------------------------------------
    result = estimate_dam_params(
        dam_list=str(dam_list),
        dam_alloc=str(dam_alloc),
        parameter_nc=str(parameter_nc),
        outflow_stats_nc=str(outflow_stats_nc),
        output_csv=str(output_csv),
        output_nc=str(output_nc),
        # GRSAD / ReGeom (optional; None → use fallback ratio)
        grsad_dir=str(grsad_dir) if grsad_dir else None,
        regeom_dir=str(regeom_dir) if regeom_dir else None,
        flood_storage_ratio=0.37,
        min_uparea=0.0,
    )

    print(f"\nDone. Primary output: {result}")


if __name__ == "__main__":
    main()
