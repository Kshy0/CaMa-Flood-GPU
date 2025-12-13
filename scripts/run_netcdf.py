# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from datetime import datetime, timedelta

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from cmfgpu.datasets.netcdf_dataset import NetCDFDataset
from cmfgpu.models.cama_flood_model import CaMaFlood
from cmfgpu.utils import setup_distributed


def main():
    ### Configuration Start ###
    resolution = "glb_15min"
    experiment_name = f"{resolution}_nc"
    input_file = f"/home/eat/CaMa-Flood-GPU/inp/{resolution}/parameters.nc"
    output_dir = "/home/eat/CaMa-Flood-GPU/out"
    opened_modules = ["base", "adaptive_time","log", "bifurcation"]
    variables_to_save = {"mean": ["river_outflow"], "last": ["river_depth"]}
    precision = "float32"
    time_step = 86400.0
    default_num_sub_steps = 360
    runoff_chunk_len = 48
    loader_workers = 3
    output_workers = 2
    unit_factor = 86400000
    prefetch_factor = 2
    BLOCK_SIZE = 128
    save_state = False

    start_date = datetime(2000, 1, 1)
    end_date = datetime(2000, 12, 31)
    runoff_dir = "/home/eat/E2O_ecmwf"
    runoff_mapping_file = f"/home/eat/CaMa-Flood-GPU/inp/{resolution}/runoff_mapping_nc.npz"
    runoff_time_interval = timedelta(days=1)
    prefix = "e2o_ecmwf_wrr2_glob15_day_Runoff_"
    suffix = ".nc"
    var_name = "Runoff"
    output_split_by_year = False
    
    # Spin-up configuration
    do_spin_up = False
    spin_up_start_date = datetime(2000, 1, 1)
    spin_up_end_date = datetime(2000, 12, 31)
    spin_up_cycles = 1
    ### Configuration End ###

    batch_size = loader_workers
    local_rank, rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    dataset = NetCDFDataset(
        base_dir=runoff_dir,
        start_date=start_date,
        end_date=end_date,
        unit_factor=unit_factor,
        out_dtype=precision,
        var_name=var_name,
        chunk_len=runoff_chunk_len,
        time_interval=runoff_time_interval,
        prefix=prefix,
        suffix=suffix,
        spin_up_cycles=spin_up_cycles if do_spin_up else 0,
        spin_up_start_date=spin_up_start_date,
        spin_up_end_date=spin_up_end_date,
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
        BLOCK_SIZE=BLOCK_SIZE,
        output_split_by_year=output_split_by_year
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

    stream = torch.cuda.Stream(device=device)
    time_iter = dataset.time_iter()
    for batch_runoff in loader:
        with torch.cuda.stream(stream):
            batch_runoff = dataset.shard_forcing(batch_runoff.to(device), local_runoff_matrix, local_runoff_indices, world_size)
            for runoff in batch_runoff:
                current_time, is_valid = next(time_iter)
                if not is_valid:
                    continue
                model.step_advance(
                    runoff=runoff,
                    time_step=time_step,
                    default_num_sub_steps=default_num_sub_steps,
                    current_time=current_time,
                )
    if save_state:  
        model.save_state(current_time + timedelta(seconds=time_step))
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
