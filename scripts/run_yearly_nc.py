from datetime import datetime, timedelta

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from cmfgpu.datasets.yearly_nc_dataset import YearlyNetCDFDataset
from cmfgpu.models.cama_flood_model import CaMaFlood
from cmfgpu.utils import setup_distributed


def main():
    ### Configuration Start ###
    resolution = "glb_15min"
    experiment_name = f"{resolution}_nc"
    input_file = f"/home/eat/CaMa-Flood-GPU/inp/{resolution}/parameters.nc"
    output_dir = "/home/eat/CaMa-Flood-GPU/out"
    opened_modules = ["base", "adaptive_time", "log", "bifurcation"]
    variables_to_save = ["river_outflow"]
    precision = "float32"
    time_step = 86400.0
    default_num_sub_steps = 360
    batch_size = 32
    loader_workers = 2
    output_workers = 2
    unit_factor = 86400000
    prefetch_factor = 2
    BLOCK_SIZE = 128
    save_state = False

    start_date = datetime(2000, 1, 1)
    end_date = datetime(2000, 12, 31)
    runoff_dir = "/home/eat/cmf_v420_pkg/inp/test_15min_nc"
    runoff_mapping_file = f"/home/eat/CaMa-Flood-GPU/inp/{resolution}/runoff_mapping_nc.npz"
    prefix = "e2o_ecmwf_wrr2_glob15_day_Runoff_"
    suffix = ".nc"
    var_name = "Runoff"
    ### Configuration End ###

    local_rank, rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

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

    dataset = YearlyNetCDFDataset(
        base_dir=runoff_dir,
        start_date=start_date,
        end_date=end_date,
        unit_factor=unit_factor,
        out_dtype=precision,
        var_name=var_name,
        prefix=prefix,
        suffix=suffix,
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
                model.step_advance(
                    runoff=runoff,
                    time_step=time_step,
                    default_num_sub_steps=default_num_sub_steps,
                    current_time=current_time,
                )
                current_time += timedelta(seconds=time_step)           
    if save_state:  
        model.save_state(current_time)
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
