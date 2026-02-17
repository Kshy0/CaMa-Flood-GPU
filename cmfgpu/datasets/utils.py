# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Shared time-to-key helpers used by all dataset classes."""

from datetime import datetime
from typing import Union

import cftime


def single_file_key(dt) -> str:
    """Constant key for single-file mode."""
    return ""


def daily_time_to_key(dt: Union[datetime, cftime.datetime]) -> str:
    """Default time-to-file key: one file per day (YYYYMMDD)."""
    return f"{dt.year:04d}{dt.month:02d}{dt.day:02d}"


def yearly_time_to_key(dt: Union[datetime, cftime.datetime]) -> str:
    """Default time-to-file key: one file per year."""
    return f"{dt.year}"


def monthly_time_to_key(dt: datetime) -> str:
    """Default time-to-file key: one file per month (YYYY_MM)."""
    return dt.strftime("%Y_%m")
