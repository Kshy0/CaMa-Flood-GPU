# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from contextlib import nullcontext
from datetime import datetime, timedelta

import torch
import torch.distributed as dist
from hydroforge.data.datasets import NetCDFDataset
from hydroforge.data.distributed import setup_distributed
from hydroforge.data.input import InputProxy
from hydroforge.contracts.temporal import SimulationSchedule
from torch.utils.data import DataLoader

from cmfgpu.models import CaMaFlood


def main():
    ### Configuration Start ###
    resolution = "glb_15min"
    experiment_name = f"{resolution}_nc"
    input_file = f"/home/eat/CaMa-Flood-GPU/inp/{resolution}/parameters.nc"
    output_dir = "/home/eat/CaMa-Flood-GPU/out"
    opened_modules = ["base", "adaptive_time", "bifurcation"]
    variables_to_save = {"mean": ["total_outflow"], "last": ["river_depth"]}
    time_step = 86400.0
    default_num_sub_steps = 360
    runoff_chunk_len = 24
    loader_workers = 2
    output_workers = 2
    unit_factor = 86400000
    prefetch_factor = 2
    BLOCK_SIZE = 128
    save_state = True

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
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    input_proxy = InputProxy.from_nc(input_file)

    dataset = NetCDFDataset(
        base_dir=runoff_dir,
        start_date=start_date,
        end_date=end_date,
        unit_factor=unit_factor,
        var_name=var_name,
        chunk_len=runoff_chunk_len,
        time_interval=runoff_time_interval,
        prefix=prefix,
        suffix=suffix,
        spin_up_cycles=spin_up_cycles if do_spin_up else 0,
        spin_up_start_date=spin_up_start_date,
        spin_up_end_date=spin_up_end_date,
        clip_negative=True,
    )
    schedule = SimulationSchedule.from_contract(
        dataset.temporal_contract(), step=timedelta(seconds=time_step),
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
        output_netcdf_options={"compression": "zlib", "complevel": 4},
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
        batch_size=batch_size,
        shuffle=False, # must be False
        num_workers=loader_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor, 
    )

    stream_ctx = torch.cuda.stream(torch.cuda.Stream(device=device)) if device.type == "cuda" else nullcontext()
    step_iter = dataset.step_iter()
    last_valid_time = start_date
    for batch_runoff in loader:
        with stream_ctx:
            batch_runoff = dataset.shard_forcing(batch_runoff.to(device), local_mapping)
            for runoff in batch_runoff:
                step = next(step_iter)
                if not step.valid:
                    continue
                last_valid_time = step.model_time
                
                model.step_advance(
                    runoff=runoff,
                    time_step=time_step,
                    default_num_sub_steps=default_num_sub_steps,
                    current_time=step.model_time,
                    output_enabled=not step.is_spin_up
                )
    if save_state:  
        model.save_state(last_valid_time + timedelta(seconds=time_step))
    model.close()
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
