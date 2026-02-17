# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

import re
import sys


def is_wsl() -> bool:
    """Check if the current system is Windows Subsystem for Linux (WSL)."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        with open("/proc/version", "r") as f:
            version_info = f.read().lower()
            return "microsoft" in version_info or "wsl" in version_info
    except Exception:
        return False


def sanitize_symbol(name: str) -> str:
    """Sanitize a string to be a valid python identifier/filename."""
    # Replace common operators with text
    name = name.replace('**', '_pow_')
    name = name.replace('^', '_pow_')
    name = name.replace('+', '_plus_')
    name = name.replace('-', '_minus_')
    name = name.replace('*', '_mul_')
    name = name.replace('/', '_div_')
    name = name.replace('.', '_dot_')
    # Replace any other non-alphanumeric with _
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Remove leading numbers
    if name and name[0].isdigit():
        name = '_' + name
    # Collapse underscores
    name = re.sub(r'_+', '_', name)
    return name.strip('_')
