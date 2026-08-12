# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from contextlib import nullcontext
from datetime import datetime, timedelta

import torch
import torch.distributed as dist
from hydroforge.data.datasets import DailyBinDataset
from hydroforge.data.distributed import setup_distributed
from hydroforge.data.input import InputProxy
from torch.utils.data import DataLoader

from cmfgpu.models import CaMaFlood


def main():
    ### Configuration Start ###
    resolution = "glb_15min"
    experiment_name = f"{resolution}_bin"
    input_file = f"/home/eat/CaMa-Flood-GPU/inp/{resolution}/parameters.nc"
    output_dir = "/home/eat/CaMa-Flood-GPU/out/"
    opened_modules = ["base", "adaptive_time","bifurcation"]
    num_sub_steps = 360 if "adaptive_time" not in opened_modules else None
    variables_to_save = {"mean": ["total_outflow"], "last": ["river_depth"]}
    runoff_time_interval = timedelta(days=1)
    
    loader_workers = 2
    output_workers = 2
    output_netcdf_options = {"compression": "zlib", "complevel": 4}
    prefetch_factor = 2
    BLOCK_SIZE = 128
    save_state = False
    output_split_by_year = False

    runoff_dir = "/home/eat/cmf_v420_pkg/inp/test_1deg/runoff"
    runoff_mapping_file = f"/home/eat/CaMa-Flood-GPU/inp/{resolution}/runoff_mapping_bin.npz"
    runoff_shape = [180, 360]
    start_date = datetime(2000, 1, 1)
    end_date = datetime(2000, 12, 31)
    unit_factor = 86400000
    bin_dtype = "float32"
    prefix = "Roff____"
    suffix = ".one"
    lat_south_to_north = False
    lon_0_to_360 = False
    
    # Set cycles to 0 to disable spin-up without removing its source window.
    spin_up_start_date = datetime(2000, 1, 1)
    spin_up_end_date = datetime(2000, 12, 31)
    spin_up_cycles = 0
    ### Configuration End ###

    local_rank, _, world_size = setup_distributed()
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    input_proxy = InputProxy.from_nc(input_file)

    dataset = DailyBinDataset(
        base_dir=runoff_dir,
        shape=runoff_shape,
        start_date=start_date,
        end_date=end_date,
        model_step=runoff_time_interval,
        unit_factor=unit_factor,
        bin_dtype=bin_dtype,
        prefix=prefix,
        suffix=suffix,
        spin_up_cycles=spin_up_cycles,
        spin_up_start_date=spin_up_start_date,
        spin_up_end_date=spin_up_end_date,
        lat_south_to_north=lat_south_to_north,
        lon_0_to_360=lon_0_to_360,
    )
    schedule = dataset.simulation_schedule

    model = CaMaFlood(
        device=device,
        experiment_name=experiment_name,
        input_proxy=input_proxy,
        output_dir=output_dir,
        opened_modules=opened_modules,
        variables_to_save=variables_to_save,
        output_workers=output_workers,
        output_netcdf_options=output_netcdf_options,
        BLOCK_SIZE=BLOCK_SIZE,
        output_split_by_year=output_split_by_year,
        simulation_schedule=schedule,
    )
    local_mapping = dataset.build_local_mapping(
        mapping_file=runoff_mapping_file,
        desired_catchment_ids=model.base.catchment_id.to("cpu").numpy(),
        device=device,
    )

    loader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=loader_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor if loader_workers > 0 else None,
    )

    stream_ctx = torch.cuda.stream(torch.cuda.Stream(device=device)) if device.type == "cuda" else nullcontext()
    for runoff_chunk in loader:
        with stream_ctx:
            runoff_chunk = dataset.shard_forcing(
                runoff_chunk.to(device),
                local_mapping,
            )
            for runoff in runoff_chunk:
                model.set_inputs(runoff)
                model.step_advance(
                    num_sub_steps=num_sub_steps,
                )
    if save_state:  
        model.save_state()
    model.close()
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
