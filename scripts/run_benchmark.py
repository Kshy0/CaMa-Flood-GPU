import time
from datetime import datetime, timedelta

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from cmfgpu.datasets.daily_bin_dataset import DailyBinDataset
from cmfgpu.models.cama_flood_model import CaMaFlood
from cmfgpu.utils import setup_distributed

BLOCK_SIZE_LIST = [64, 128, 256, 512, 1024]

def benchmark_block_sizes():
    ### Benchmark Configuration ###
    experiment_name = f"benchmark_run"
    resolution = "glb_15min"
    input_file = f"/home/eat/CaMa-Flood-GPU/inp/{resolution}/parameters.nc"
    output_dir = "/home/eat/CaMa-Flood-GPU/out"
    opened_modules = ["base", "adaptive_time", "bifurcation"]
    variables_to_save = {}
    precision = "float32"
    time_step = 86400.0
    default_num_sub_steps = 360
    batch_size = 8
    loader_workers = 2
    prefetch_factor = 2
    save_state = False

    runoff_dir = "/home/eat/cmf_v420_pkg/inp/test_1deg/runoff"
    runoff_mapping_file = f"/home/eat/CaMa-Flood-GPU/inp/{resolution}/runoff_mapping_bin.npz"
    runoff_shape = [180, 360]
    start_date = datetime(2000, 1, 1)
    end_date = datetime(2000, 4, 1)
    unit_factor = 86400000
    bin_dtype = "float32"
    prefix = "Roff____"
    suffix = ".one"

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
        output_workers=0,
        output_complevel=0,
        BLOCK_SIZE=BLOCK_SIZE_LIST[0], 
    )

    dataset = DailyBinDataset(
        base_dir=runoff_dir,
        shape=runoff_shape,
        start_date=start_date,
        end_date=end_date,
        unit_factor=unit_factor,
        bin_dtype=bin_dtype,
        out_dtype=precision,
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
        shuffle=False,
        num_workers=loader_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
    )

    results = []
    if rank == 0:
        print("Benchmarking BLOCK_SIZE...")

    for block_size in BLOCK_SIZE_LIST:
        model.BLOCK_SIZE = block_size
        current_time = start_date
        stream = torch.cuda.Stream(device=device)
        torch.cuda.synchronize()
        start = time.time()

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

        torch.cuda.synchronize()
        end = time.time()

        elapsed_ms = (end - start) * 1000
        results.append((block_size, elapsed_ms))
    if save_state:  
        model.save_state(current_time)
    if world_size > 1:
        dist.destroy_process_group()
    if rank == 0:
        print("\n=== Benchmark Results ===")
        for bs, t in results:
            print(f"BLOCK_SIZE={bs} --> {t:.2f} ms")

if __name__ == "__main__":
    benchmark_block_sizes()
