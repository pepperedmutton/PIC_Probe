from __future__ import annotations

import re
import numpy as np
from pathlib import Path
from dataclasses import dataclass

@dataclass
class CrossSectionData:
    energy: np.ndarray
    sigma: np.ndarray

def parse_cs_txt(file_path: str | Path) -> dict[str, CrossSectionData]:
    """
    Parses the custom CS.txt file format.
    
    Expected format:
    1. First block: Electron Elastic (assumed, if no header)
    2. Block with "IONIZATION": Electron Ionization
    3. Block with "Ar+ + Ar" or "Backscat": Ion CEX
    
    Returns:
        Dictionary with keys: 'electron_elastic', 'electron_ionization', 'ion_cex'
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Cross section file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    sections = {}
    current_data = []
    current_section = "electron_elastic" # Default first section
    
    # Helper to finalize a section
    def finalize_section(name, data_rows):
        if not data_rows:
            return
        # Sort by energy just in case
        arr = np.array(data_rows, dtype=np.float64)
        # Ensure sorted by energy
        idx = np.argsort(arr[:, 0])
        arr = arr[idx]
        sections[name] = CrossSectionData(energy=arr[:, 0], sigma=arr[:, 1])

    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for headers / separators
        if line.startswith("---------") or line.startswith("xxxxxxxxx") or line.startswith("*********"):
            continue
            
        # Section detection logic
        if "IONIZATION" in line:
            # Save previous
            finalize_section(current_section, current_data)
            current_section = "electron_ionization"
            current_data = []
            continue
        
        if "Backscat" in line:
            # Save previous
            finalize_section(current_section, current_data)
            current_section = "ion_cex"
            current_data = []
            continue
            
        # Metadata lines (skip)
        if any(line.startswith(p) for p in ["SPECIES:", "PROCESS:", "PARAM.:", "COMMENT:", "UPDATED:", "COLUMNS:", "DATABASE:", "PERMLINK:", "DESCRIPTION:", "CONTACT:", "HOW TO REFERENCE:"]):
            continue
            
        # Parse data lines
        # Expect: Energy  Sigma
        parts = line.split()
        if len(parts) >= 2:
            try:
                e = float(parts[0])
                s = float(parts[1])
                current_data.append((e, s))
            except ValueError:
                # Not a data line
                continue

    # Finalize last section
    finalize_section(current_section, current_data)
    
    return sections
