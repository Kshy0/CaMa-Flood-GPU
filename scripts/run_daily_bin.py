# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from contextlib import nullcontext
from datetime import datetime, timedelta

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from cmfgpu.datasets import DailyBinDataset
from cmfgpu.models import CaMaFlood
from cmfgpu.params import InputProxy
from cmfgpu.utils import setup_distributed


def main():
    ### Configuration Start ###
    resolution = "glb_15min"
    experiment_name = f"{resolution}_bin"
    input_file = f"/home/eat/CaMa-Flood-GPU/inp/{resolution}/parameters.nc"
    output_dir = "/home/eat/CaMa-Flood-GPU/out/"
    opened_modules = ["base", "adaptive_time", "log", "bifurcation"]
    variables_to_save = {"mean": ["total_outflow"], "last": ["river_depth"]}
    time_step = 86400.0
    default_num_sub_steps = 360
    
    loader_workers = 2
    output_workers = 2
    output_complevel = 4 
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
    
    # Spin-up configuration
    do_spin_up = False
    spin_up_start_date = datetime(2000, 1, 1)
    spin_up_end_date = datetime(2000, 12, 31)
    spin_up_cycles = 1
    ### Configuration End ###

    batch_size = loader_workers
    local_rank, rank, world_size = setup_distributed()
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
        unit_factor=unit_factor,
        bin_dtype=bin_dtype,
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
        input_proxy=input_proxy,
        output_dir=output_dir,
        opened_modules=opened_modules,
        variables_to_save=variables_to_save,
        output_workers=output_workers,
        output_complevel=output_complevel,
        BLOCK_SIZE=BLOCK_SIZE,
        output_split_by_year=output_split_by_year
    )
    model.set_total_steps(dataset.total_steps)

    local_runoff_matrix = dataset.build_local_runoff_matrix(
        runoff_mapping_file=runoff_mapping_file,
        desired_catchment_ids=model.base.catchment_id.to("cpu").numpy(),
        device=device,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=loader_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
    )

    stream_ctx = torch.cuda.stream(torch.cuda.Stream(device=device)) if device.type == "cuda" else nullcontext()
    time_iter = dataset.time_iter()
    last_valid_time = start_date
    for batch_runoff in loader:
        with stream_ctx:
            batch_runoff = dataset.shard_forcing(batch_runoff.to(device), local_runoff_matrix, world_size)
            for runoff in batch_runoff:
                current_time, is_valid, is_spin_up = next(time_iter)
                if not is_valid:
                    continue
                last_valid_time = current_time
                model.step_advance(
                    runoff=runoff,
                    time_step=time_step,
                    default_num_sub_steps=default_num_sub_steps,
                    current_time=current_time,
                    output_enabled=not is_spin_up
                )
    if save_state:  
        model.save_state(last_valid_time + timedelta(seconds=time_step))
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
