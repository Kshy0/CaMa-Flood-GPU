import numpy as np
import torch
from datetime import datetime, timedelta
from pathlib import Path
from CMF_GPU.utils.utils import check_enabled
from CMF_GPU.utils.Variables import MODULES_CONFIG

class Logger:
    def __init__(self, base_dir, buffer_size = 400, disabled=False):
        self.disabled = disabled
        if self.disabled:
            return
        self.start_time = datetime.now()
        self.log_path = Path(base_dir) / "log_CaMa_GPU.txt"
        self.log_vars = MODULES_CONFIG["log"]["hidden_states"]
        self.buffer_size = buffer_size
        self.times = None
        self.write_header()
        
    @check_enabled
    def write_header(self):
        with open(self.log_path, 'w') as f:
            f.write("Program Start Time: ")
            self.start_time_pos = f.tell()
            f.write("{:<24}\n".format(''))

            f.write("Program End Time:   ")
            self.end_time_pos = f.tell()
            f.write("{:<24}\n".format(''))

            f.write("Duration:       ")
            self.duration_pos = f.tell()
            f.write("{:>24}\n".format(''))

            f.write("\n")

            headers = [
                "StepStartTime", "StoragePre", "StorageNext",
                "StorageNew", "InflowError", "Inflow", "Outflow",
                "StorageStage", "StageError","RiverStorage", 
                "FloodStorage", "FloodArea"
            ]
            widths = [18] + [16] * (len(headers) - 1)
            header_line = ''.join(
                f"{h:<{w}}" if i == 0 else f"{h:>{w}}"
                for i, (h, w) in enumerate(zip(headers, widths))
            )
            f.write(header_line + "\n")

    @check_enabled
    def set_current_time(self, current_time):
        self.current_time = current_time

    @check_enabled
    def set_time_step(self, time_step, num_steps, states):
        self.time_step = time_step
        self.num_steps = num_steps
        with open(self.log_path, 'a') as f:
            f.write(f"Time Step: {time_step:.4f} seconds    Number of Steps: {num_steps}\n")
            
        # Create time vector
        self.times = [self.current_time + timedelta(seconds=time_step * i) for i in range(num_steps)]
        if num_steps > self.buffer_size:
            self.buffer_size = num_steps + 20
            for k in self.log_vars:
                states[k] = torch.zeros((self.buffer_size,), device=states[k].device, dtype=states[k].dtype)
    
    @check_enabled
    def write_step(self, log_data):
        num_steps = self.num_steps
        time_strs = np.array([t.strftime('%Y-%m-%d %H:%M') for t in self.times[:num_steps]], dtype=str)

        total_storage_pre = log_data["total_storage_pre_sum"].cpu().numpy()[:num_steps]
        total_storage_next = log_data["total_storage_next_sum"].cpu().numpy()[:num_steps]
        total_storage_new = log_data["total_storage_new"].cpu().numpy()[:num_steps]
        total_river_storage = log_data["total_river_storage"].cpu().numpy()[:num_steps]
        total_flood_storage = log_data["total_flood_storage"].cpu().numpy()[:num_steps]
        total_outflow = log_data["total_outflow"].cpu().numpy()[:num_steps]
        total_inflow = log_data["total_inflow"].cpu().numpy()[:num_steps]
        total_storage_stage_new = log_data["total_storage_stage_new"].cpu().numpy()[:num_steps]
        total_flood_area = log_data["total_flood_area"].cpu().numpy()[:num_steps]
        inflow_error = log_data["total_inflow_error"].cpu().numpy()[:num_steps]
        stage_error = log_data["total_stage_error"].cpu().numpy()[:num_steps]

        fmt = ['%-18s', '%16.6g', '%16.6g', '%16.6g', '%16.3e', '%16.6g', '%16.6g', '%16.6g', '%16.3e', '%16.6g', '%16.6g', '%16.6g']

        with open(self.log_path, 'a') as f:
            for i in range(num_steps):
                row_data = (
                    time_strs[i],
                    total_storage_pre[i],
                    total_storage_next[i],
                    total_storage_new[i],
                    inflow_error[i],
                    total_inflow[i],
                    total_outflow[i],
                    total_storage_stage_new[i],
                    stage_error[i],
                    total_river_storage[i],
                    total_flood_storage[i],
                    total_flood_area[i]
                )
                line = ''.join(fmt_str % val for fmt_str, val in zip(fmt, row_data))
                f.write(line + '\n')

    @check_enabled
    def close(self):
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        with open(self.log_path, 'r+b') as f:
            f.seek(self.start_time_pos)
            f.write(f"{self.start_time:%Y-%m-%d %H:%M:%S}".ljust(24).encode())

            f.seek(self.end_time_pos)
            f.write(f"{end_time:%Y-%m-%d %H:%M:%S}".ljust(24).encode())

            f.seek(self.duration_pos)
            f.write(f"{duration:8.2f} Seconds".ljust(24).encode())