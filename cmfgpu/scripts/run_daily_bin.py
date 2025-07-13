import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from datetime import timedelta
from cmfgpu.utils import get_local_rank
from cmfgpu.models.cama_flood_model import CaMaFlood
from cmfgpu.datasets.daily_bin_dataset import DailyBinDataset
from cmfgpu.configs.daily_bin_config import DailyBinConfig

def setup_distributed():
    torch.multiprocessing.set_start_method("spawn", force=True)
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = get_local_rank()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        rank = 0
        world_size = 1

    return local_rank, rank, world_size

def main(config_file):
    config = DailyBinConfig.from_toml(config_file)
    local_rank, rank, world_size = setup_distributed()
    time_step = config.time_step
    model = CaMaFlood(rank=rank, 
                      world_size=world_size, 
                      device=torch.device(f"cuda:{local_rank}"),
                      experiment_name=config.experiment_name,
                      input_file=config.input_file,
                      output_dir=config.output_dir,
                      opened_modules=config.opened_modules,
                      variables_to_save=config.variables_to_save,
                      precision=config.precision,
                      output_workers=config.output_workers,
                      )
    dataset = DailyBinDataset(
        base_dir=config.runoff_dir,
        shape=config.runoff_shape,
        start_date=config.start_date,
        end_date=config.end_date,
        unit_factor=config.unit_factor,
    )
    local_runoff_matrix = dataset.build_local_runoff_matrix(
        runoff_mapping_file=config.runoff_mapping_file,
        desired_catchment_ids=model.base.catchment_id.to("cpu").numpy(),
        precision=config.precision,
        device=torch.device(f"cuda:{local_rank}"),
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.loader_workers,
        pin_memory=True,
    )
    current_time = config.start_date
    stream = torch.cuda.Stream(device=torch.device(f"cuda:{local_rank}"))
    for batch_runoff in loader:
        with torch.cuda.stream(stream):
            batch_runoff = dataset.apply_runoff_to_catchments(batch_runoff, local_runoff_matrix)
            for batch in batch_runoff:
                print(f"Rank {rank} processed data for time {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                model.step_advance(
                    runoff=batch,
                    time_step=time_step,
                    default_num_sub_steps=config.default_num_sub_steps,
                    current_time=current_time,
                )
                current_time += timedelta(seconds=time_step)
                
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Path to the config file")
    args = parser.parse_args()
    main(args.config_file)
    # main("/home/eat/CaMa-Flood-GPU/configs/glb_15min.toml")