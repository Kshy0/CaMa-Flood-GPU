from __future__ import annotations
from typing import List, Self
from pydantic import BaseModel, Field, FilePath, field_validator
from pathlib import Path
from datetime import datetime
import tomllib

class TemplateConfig(BaseModel):

    experiment_name: str = Field(default="default_experiment", description="Name of the experiment")
    start_date: datetime = Field(description="Start date of the simulation")
    end_date: datetime = Field(description="End date of the simulation")
    input_file: FilePath = Field(description="Path to the input configuration file")
    output_dir: Path = Field(description="Path to the output directory")
    opened_modules: List[str] = Field(description="List of active modules")
    variables_to_save: List[str] = Field(description="Variables to be collected during the simulation")
    precision: str = Field(default="float32", description="Precision of the data, e.g., 'float32', 'float64'")
    time_step: float = Field(default=86400.0, description="Time step in seconds")
    default_num_sub_steps: int = Field(default=360, description="Default number of sub-steps for the simulation", gt=0)
    
    batch_size: int = Field(default=8, description="Batch size for data loading", gt=0)
    loader_workers: int = Field(default=2, description="Number of workers for data loading", ge=0)
    output_workers: int = Field(default=2, description="Number of workers for writing output files", ge=0)

    @classmethod
    def from_toml(cls, toml_path: str) -> Self:
        with open(toml_path, 'rb') as f:
            data = tomllib.load(f)
        
        return cls(**data)
    
    @field_validator('start_date', 'end_date', mode='before')
    def parse_date(cls, value: str | datetime) -> datetime:
        if isinstance(value, datetime):
            return value
        try:
            # Handles standard "YYYY‑MM‑DD"
            return datetime.fromisoformat(value)
        except ValueError:
            # Fallback: strip Z and try manually
            val = value.rstrip("Z")
            return datetime.strptime(val, "%Y-%m-%d")