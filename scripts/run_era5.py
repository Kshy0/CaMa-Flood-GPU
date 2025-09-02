# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from datetime import datetime, timedelta

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from cmfgpu.datasets.era5_land_dataset import ERA5LandDataset
from cmfgpu.models.cama_flood_model import CaMaFlood
from cmfgpu.utils import setup_distributed


def main():
    ### Configuration Start ###
    resolution = "glb_06min"
    experiment_name = f"{resolution}_era5"
    input_file = f"/home/eat/CaMa-Flood-GPU/inp/{resolution}/parameters.nc"
    output_dir = "/home/eat/CaMa-Flood-GPU/out"
    opened_modules = ["base", "adaptive_time", "log", "bifurcation"]
    variables_to_save = {"mean": ["river_outflow"], "last": ["river_depth"]}
    precision = "float32"
    runoff_time_interval_hour = 24
    time_step = runoff_time_interval_hour * 3600
    default_num_sub_steps = 360

    loader_workers = 4
    output_workers = 2
    runoff_chunk_len = 24
    unit_factor = 3600 * runoff_time_interval_hour
    prefetch_factor = 2
    BLOCK_SIZE = 128
    save_state = False
    start_date = datetime(2000, 1, 1, 0, 0, 0)
    end_date = datetime(2000, 12, 31, 0, 0, 0)
    runoff_dir = "/home/eat/ERA5_Runoff"
    runoff_mapping_file = f"/home/eat/CaMa-Flood-GPU/inp/{resolution}/runoff_mapping_era5.npz"
    prefix = "runoff_"
    suffix = ".nc"
    var_name = "ro"
    ### Configuration End ###

    batch_size = loader_workers
    local_rank, rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    dataset = ERA5LandDataset(
        base_dir=runoff_dir,
        start_date=start_date,
        end_date=end_date,
        unit_factor=unit_factor, # mm/day divided by unit_factor to get m/s
        time_interval=timedelta(hours=runoff_time_interval_hour),
        chunk_len=runoff_chunk_len,
        out_dtype=precision,
        var_name=var_name,
        prefix=prefix,
        suffix=suffix,
    )

    model = CaMaFlood(
        rank=rank,
        world_size=world_size,
        device=device,
        experiment_name=experiment_name,
        input_file=input_file,
        output_dir=output_dir,
        opened_modules=opened_modules,
        variables_to_save=variables_to_save,
        precision=precision,
        output_workers=output_workers,
        output_complevel=4,
        BLOCK_SIZE=BLOCK_SIZE
    )

    

    local_runoff_matrix, local_runoff_indices = dataset.build_local_runoff_matrix(
        runoff_mapping_file=runoff_mapping_file,
        desired_catchment_ids=model.base.catchment_id.to("cpu").numpy(),
        precision=precision,
        device=device,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False, # must be False
        num_workers=loader_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor, 
    )

    current_time = start_date
    stream = torch.cuda.Stream(device=device)
    for batch_runoff in loader:
        with torch.cuda.stream(stream):
            batch_runoff = dataset.shard_forcing(batch_runoff.to(device), local_runoff_matrix, local_runoff_indices, world_size)
            for runoff in batch_runoff:
                # Skip padded steps beyond the configured end_date
                if current_time > end_date:
                    break
                model.step_advance(
                    runoff=runoff,
                    time_step=time_step,
                    default_num_sub_steps=default_num_sub_steps,
                    current_time=current_time,
                    stat_is_first=(current_time.hour == 0),
                    stat_is_last=(current_time.hour == 0)
                )
                current_time += timedelta(seconds=time_step)           
    if save_state:  
        model.save_state(current_time)
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
