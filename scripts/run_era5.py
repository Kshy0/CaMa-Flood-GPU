# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from contextlib import nullcontext
from datetime import datetime, timedelta

import torch
import torch.distributed as dist
from hydroforge.data.datasets import ERA5LandAccumDataset
from hydroforge.data.distributed import setup_distributed
from hydroforge.data.input import InputProxy
from torch.utils.data import DataLoader
from hydroforge.contracts.temporal import (
    CalendarWindow,
    StatisticsPlan,
)

from cmfgpu.models import CaMaFlood


def main():
    ### Configuration Start ###
    resolution = "glb_06min"
    experiment_name = f"{resolution}_era5"
    input_file = f"/home/eat/CaMa-Flood-GPU/inp/{resolution}/parameters.nc"
    output_dir = "/home/eat/CaMa-Flood-GPU/out"
    opened_modules = ["base", "adaptive_time", "bifurcation"]
    num_sub_steps = 360 if "adaptive_time" not in opened_modules else None
    variables_to_save = {"mean": ["total_outflow"], "last": ["river_depth"]}
    runoff_time_interval_hour = 1
    runoff_time_interval = timedelta(hours=runoff_time_interval_hour)

    loader_workers = 1
    output_workers = 2
    runoff_chunk_len = 24
    unit_factor = 3600 * runoff_time_interval_hour
    prefetch_factor = 2
    BLOCK_SIZE = 128
    save_state = False
    start_date = datetime(2000, 1, 1, 0, 0, 0)
    end_date = datetime(2000, 12, 31, 23, 0, 0)
    runoff_dir = "/home/eat/ERA5_Runoff"
    runoff_mapping_file = f"/home/eat/CaMa-Flood-GPU/inp/{resolution}/runoff_mapping_era5.npz"
    prefix = "runoff_"
    suffix = ".nc"
    var_name = "ro"
    output_split_by_year = False
    
    # Set cycles to 0 to disable spin-up without removing its source window.
    spin_up_start_date = datetime(2000, 1, 1, 0, 0, 0)
    spin_up_end_date = datetime(2000, 12, 31, 23, 0, 0)
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

    dataset = ERA5LandAccumDataset(
        base_dir=runoff_dir,
        start_date=start_date,
        end_date=end_date,
        model_step=runoff_time_interval,
        unit_factor=unit_factor, # mm/day divided by unit_factor to get m/s
        time_interval=runoff_time_interval,
        chunk_len=runoff_chunk_len,
        var_name=var_name,
        prefix=prefix,
        suffix=suffix,
        spin_up_cycles=spin_up_cycles,
        spin_up_start_date=spin_up_start_date,
        spin_up_end_date=spin_up_end_date,
    )
    schedule = dataset.simulation_schedule
    statistics_plan = StatisticsPlan(
        inner=CalendarWindow("day"),
    )

    model = CaMaFlood(
        device=device,
        experiment_name=experiment_name,
        input_proxy=input_proxy,
        output_dir=output_dir,
        opened_modules=opened_modules,
        variables_to_save=variables_to_save,
        simulation_schedule=schedule,
        statistics_plan=statistics_plan,
        output_workers=output_workers,
        output_netcdf_options={"compression": "zlib", "complevel": 4},
        BLOCK_SIZE=BLOCK_SIZE,
        output_split_by_year=output_split_by_year
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
        pin_memory=device.type == "cuda",
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
