import pickle
import os
import shutil
from CMF_GPU.phys.triton.StepAdvance import advance_step
from CMF_GPU.utils.Preprocesser import load_csr_list_from_pkl
from CMF_GPU.utils.Checker import prepare_model_and_function
from CMF_GPU.utils.Dataloader import DataLoader, DailyBinDataset
from CMF_GPU.utils.Datadumper import DataDumper
from CMF_GPU.utils.Logger import Logger
from CMF_GPU.utils.utils import snapshot_to_pkl, gather_device_dicts
from datetime import datetime, timedelta
from omegaconf import OmegaConf


def main(config_file):

    config = OmegaConf.load(config_file)
    config = OmegaConf.to_container(config, resolve=True)
    with open(os.path.join(config["inp_dir"], "parameters.pkl"), "rb") as f:
        params = pickle.load(f)
    with open(os.path.join(config["inp_dir"], "init_states.pkl"), "rb") as f:
        states = pickle.load(f)
    with open(os.path.join(config["inp_dir"], "runoff_mask.pkl"), "rb") as f:
        runoff_mask = pickle.load(f)    
    
    runtime_flags, params, states, parallel_step_fn = prepare_model_and_function(
        params,
        states,
        advance_step,
        config["runtime_flags"],
    )
    runoff_matrix = load_csr_list_from_pkl(os.path.join(config["inp_dir"], "runoff_input_matrix.pkl"), device_indices=runtime_flags["device_indices"])
    start_date = datetime.strptime(config["start_date"], "%Y-%m-%d")
    end_date = datetime.strptime(config["end_date"], "%Y-%m-%d")
    time_starts = [start_date + timedelta(seconds=runtime_flags["time_step"]) * i for i in range((end_date - start_date).days + 1)]

    ds = DailyBinDataset(**config["runoff_dataset"]["params"])

    loader = DataLoader(
        time_starts=time_starts,
        dataset=ds,  
        unit_factor=runtime_flags["unit_factor"],
        num_workers=1,
        max_cache_steps=100,
        precision="float32",
        runoff_mask=runoff_mask
    )
    if os.path.exists(config["out_dir"]):
        shutil.rmtree(config["out_dir"])
    os.makedirs(config["out_dir"])
    dumper = DataDumper(
        base_dir=config["out_dir"],
        file_format="nc",
        num_workers=1,
        disabled=False if "aggregator" in runtime_flags["modules"] else True
    )
    logger = Logger(base_dir=config["out_dir"],
        buffer_size=params[0].get("log_buffer_size", None), 
        disabled=False if "log" in runtime_flags["modules"] else True
    )
    # TODO: multi-GPU support
    default_sub_iters = runtime_flags["default_sub_iters"]
    for time_start in time_starts:
        runoff_t, time_length = loader.get_data(time_start)
        current_time = time_start
        time_step = runtime_flags["time_step"]
        if time_length % time_step != 0:
            raise ValueError(
                f"Runoff step length {time_length} cannot be evenly divided by time_step={time_step}, please check your data or configuration."
            )
        iters_per_runoff_step = int(time_length // time_step)
        dT_def = time_step / default_sub_iters
        # print("total_storage:", f"{states["total_storage"].sum():.4e}")
        for _ in range(iters_per_runoff_step):
            logger.set_current_time(current_time)
            parallel_step_fn(
                runtime_flags, params, states, runoff_t, runoff_matrix, dT_def, logger
            )
            aggregator_cpu = dumper.gather_stats(states)
            dumper.submit_data(current_time, aggregator_cpu)
            current_time += timedelta(seconds=time_step)

    states_cpu = gather_device_dicts(states)
    snapshot_to_pkl(states_cpu, "state", runtime_flags["modules"], os.path.join(config["out_dir"], "final_states.pkl"))
    dumper.close()
    logger.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run CMF_GPU simulation.")
    parser.add_argument("config_file", type=str, help="Path to the config.yaml file")
    args = parser.parse_args()
    main(args.config_file)