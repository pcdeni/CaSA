#!/usr/bin/env python3
"""
SiMRA-DRAM Zenodo Data Analysis
================================
Analyzes the experimental data from the SiMRA-DRAM paper (DSN'24)
WITHOUT requiring any FPGA hardware.

PURPOSE: Understand what "good" DRAM characterization looks like before
you test your own DIMMs. Learn the data format, success rate distributions,
and what parameters matter.

SETUP:
  1. Download dataset: https://doi.org/10.5281/zenodo.11165221
  2. Extract dsn_artifact.zip
  3. Move experimental_data/ directory to the SiMRA-DRAM repo:
     git clone https://github.com/CMU-SAFARI/SiMRA-DRAM.git
     # Then: cp -r experimental_data/ SiMRA-DRAM/experimental_data/
  4. Run this script from anywhere — just set DATA_DIR below.

ALSO CLONE (for reference, code study only):
  git clone https://github.com/CMU-SAFARI/DRAM-Bender.git
  git clone https://github.com/CMU-SAFARI/FCDRAM.git
"""

import os
import sys
import json
import glob
import numpy as np

# =============================================================================
# CONFIGURATION — EDIT THIS PATH
# =============================================================================
DATA_DIR = os.path.expanduser("~/SiMRA-DRAM/experimental_data")
# Alternative: set via environment variable
DATA_DIR = os.environ.get("SIMRA_DATA_DIR", DATA_DIR)

# =============================================================================
# STEP 1: Discover what's in the dataset
# =============================================================================
def explore_dataset(data_dir):
    """Walk the data directory and catalog all files."""
    print("=" * 70)
    print("STEP 1: Exploring SiMRA-DRAM Dataset")
    print("=" * 70)

    if not os.path.isdir(data_dir):
        print(f"\nERROR: Data directory not found: {data_dir}")
        print(f"\nTo set up:")
        print(f"  1. Download: https://doi.org/10.5281/zenodo.11165221")
        print(f"  2. Extract dsn_artifact.zip")
        print(f"  3. Set DATA_DIR in this script or:")
        print(f"     export SIMRA_DATA_DIR=/path/to/experimental_data")
        return None

    file_types = {}
    total_files = 0
    total_size_mb = 0

    for root, dirs, files in os.walk(data_dir):
        for f in files:
            full_path = os.path.join(root, f)
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            ext = os.path.splitext(f)[1].lower()
            rel_path = os.path.relpath(full_path, data_dir)

            if ext not in file_types:
                file_types[ext] = {"count": 0, "size_mb": 0, "examples": []}
            file_types[ext]["count"] += 1
            file_types[ext]["size_mb"] += size_mb
            if len(file_types[ext]["examples"]) < 3:
                file_types[ext]["examples"].append(rel_path)

            total_files += 1
            total_size_mb += size_mb

    print(f"\nTotal files: {total_files}")
    print(f"Total size:  {total_size_mb:.1f} MB")
    print(f"\nFile types:")
    for ext, info in sorted(file_types.items(), key=lambda x: -x[1]["size_mb"]):
        print(f"  {ext or '(no ext)':10s}  {info['count']:5d} files  {info['size_mb']:8.1f} MB")
        for ex in info["examples"]:
            print(f"    example: {ex}")

    # Look for key directories
    print(f"\nTop-level directories:")
    for item in sorted(os.listdir(data_dir)):
        full = os.path.join(data_dir, item)
        if os.path.isdir(full):
            n_files = sum(len(f) for _, _, f in os.walk(full))
            print(f"  {item}/  ({n_files} files)")

    return file_types


# =============================================================================
# STEP 2: Parse characterization results (adapt based on actual data format)
# =============================================================================
def parse_csv_results(data_dir):
    """Try to parse any CSV/JSON results in the dataset.

    The actual file format depends on what SiMRA-DRAM outputs.
    This function adapts to whatever is found.
    """
    print("\n" + "=" * 70)
    print("STEP 2: Parsing Characterization Results")
    print("=" * 70)

    csv_files = glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True)
    json_files = glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True)
    txt_files = glob.glob(os.path.join(data_dir, "**/*.txt"), recursive=True)
    npy_files = glob.glob(os.path.join(data_dir, "**/*.npy"), recursive=True)

    print(f"\nFound: {len(csv_files)} CSV, {len(json_files)} JSON, "
          f"{len(txt_files)} TXT, {len(npy_files)} NPY files")

    # Try CSV first
    if csv_files:
        print(f"\nFirst CSV file: {os.path.relpath(csv_files[0], data_dir)}")
        try:
            import csv
            with open(csv_files[0], 'r') as f:
                reader = csv.reader(f)
                headers = next(reader, None)
                print(f"  Headers: {headers}")
                first_rows = [next(reader, None) for _ in range(3)]
                for row in first_rows:
                    if row:
                        print(f"  Row: {row[:10]}{'...' if len(row) > 10 else ''}")
        except Exception as e:
            print(f"  Error reading CSV: {e}")

    # Try JSON
    if json_files:
        print(f"\nFirst JSON file: {os.path.relpath(json_files[0], data_dir)}")
        try:
            with open(json_files[0], 'r') as f:
                data = json.load(f)
            if isinstance(data, dict):
                print(f"  Keys: {list(data.keys())[:10]}")
            elif isinstance(data, list):
                print(f"  List of {len(data)} items, first: {str(data[0])[:100]}")
        except Exception as e:
            print(f"  Error reading JSON: {e}")

    # Try NPY
    if npy_files:
        print(f"\nFirst NPY file: {os.path.relpath(npy_files[0], data_dir)}")
        try:
            arr = np.load(npy_files[0])
            print(f"  Shape: {arr.shape}, dtype: {arr.dtype}")
            print(f"  Min: {arr.min()}, Max: {arr.max()}, Mean: {arr.mean():.4f}")
        except Exception as e:
            print(f"  Error reading NPY: {e}")

    # Try TXT (often raw logs)
    if txt_files:
        print(f"\nFirst TXT file: {os.path.relpath(txt_files[0], data_dir)}")
        try:
            with open(txt_files[0], 'r') as f:
                lines = f.readlines()[:5]
            for line in lines:
                print(f"  {line.rstrip()[:100]}")
        except Exception as e:
            print(f"  Error reading TXT: {e}")


# =============================================================================
# STEP 3: Analyze MAJ3 / AND success rates (if data available)
# =============================================================================
def analyze_success_rates(data_dir):
    """Look for and analyze MAJ3/AND success rate data.

    SiMRA-DRAM tests:
      - Simultaneous Many-Row Activation (SiMRA)
      - MAJ3 through MAJ9 (majority operations)
      - RowCopy (one-to-many)
      - AND/OR/NOT (FCDRAM)

    We care most about:
      - AND success rate per row/bank/subarray
      - How success rate varies with temperature
      - How success rate varies with timing parameters
    """
    print("\n" + "=" * 70)
    print("STEP 3: Analyzing Success Rates")
    print("=" * 70)

    # Look for files that might contain success rate data
    all_files = glob.glob(os.path.join(data_dir, "**/*"), recursive=True)
    success_files = [f for f in all_files if any(
        kw in os.path.basename(f).lower()
        for kw in ['success', 'maj', 'and', 'result', 'characteriz']
    )]

    if success_files:
        print(f"\nPotentially relevant files ({len(success_files)}):")
        for f in success_files[:20]:
            rel = os.path.relpath(f, data_dir)
            size = os.path.getsize(f) / 1024
            print(f"  {rel} ({size:.1f} KB)")
    else:
        print("\nNo obvious success-rate files found by name.")
        print("Check the SiMRA-DRAM repo's analysis/ directory for Jupyter notebooks.")
        print("The paper_plots.ipynb notebook should process the raw data.")


# =============================================================================
# STEP 4: Summary of what to look for
# =============================================================================
def print_analysis_guide():
    """Print a guide for what to look for in the data."""
    print("\n" + "=" * 70)
    print("STEP 4: What To Look For (Analysis Guide)")
    print("=" * 70)

    print("""
When analyzing the SiMRA-DRAM data, focus on these metrics:

1. MAJ3 SUCCESS RATE BY SUBARRAY
   - Each subarray has different physical characteristics
   - Success rate > 99% = excellent for PIM
   - Success rate 95-99% = usable with error correction
   - Success rate < 95% = blacklist this subarray

2. TEMPERATURE DEPENDENCE
   - Test at 25C, 35C, 45C, 55C
   - If success rate drops >10x between 25C and 45C, you need thermal control
   - If stable: no thermal management needed (big cost/complexity savings)

3. TIMING PARAMETER SENSITIVITY
   - t1 (violated tRAS): typically 1-5ns
   - t2 (violated tRP): typically 2-5ns
   - Find the "sweet spot" where success rate is highest
   - This tells you what timings to program into DRAM Bender

4. ROW PAIR CHARACTERIZATION
   - PIM AND requires both rows in the SAME subarray
   - Some row pairs work better than others
   - Build a "safe pair" map before attempting inference

5. MANUFACTURER DIFFERENCES
   - SK Hynix: consistently works for PIM (all SAFARI papers use it)
   - Micron: works but may have different optimal timings
   - Samsung: DOES NOT WORK for PIM (zero PUD operations succeed)

6. PIM-LLM SPECIFIC TARGETS
   - We need 2-input AND (simpler than MAJ3, should have HIGHER success rate)
   - We need it reliable for ~133K rows across 5-6 banks
   - BER < 0.01% is ideal (30-layer cos_sim > 0.98)
   - BER < 0.1% is acceptable with MSB voting (cos_sim > 0.97)
   - BER > 1% requires heavy error correction (still possible but slow)

COMPARISON POINT FOR YOUR HARDWARE:
   When you characterize your own DIMMs, compare against these published numbers:
   - FCDRAM (HPCA'24): 16-input AND success = 94.94% (256 chips)
   - SiMRA (DSN'24): MAJ3 tested on 120 chips
   - Your 2-input AND should be BETTER than 16-input AND
   - Target: >99.99% success rate per AND operation
""")


# =============================================================================
# STEP 5: Also run the official SiMRA-DRAM analysis
# =============================================================================
def print_official_analysis_instructions():
    """Instructions for running the official SiMRA-DRAM notebooks."""
    print("\n" + "=" * 70)
    print("STEP 5: Running Official SiMRA-DRAM Analysis Notebooks")
    print("=" * 70)

    print("""
The SiMRA-DRAM repo includes Jupyter notebooks that reproduce the paper's plots.
Run these to understand the data format and see published results:

  cd SiMRA-DRAM/
  pip install pandas scipy matplotlib seaborn jupyter
  jupyter notebook analysis/paper_plots.ipynb

This will generate plots in analysis/plots/ showing:
  - MAJ3-MAJ9 success rates vs timing parameters
  - RowCopy success rates
  - Temperature dependence
  - Per-subarray variation

ALSO check the FCDRAM repo:
  cd FCDRAM/
  jupyter notebook analysis/paper_plots.ipynb

This shows:
  - AND/OR/NOT success rates
  - NAND/NOR success rates
  - Up to 16-input operation success rates
  - Data pattern effects on success rate

DRAM Bender repo (code study only, can't run without FPGA):
  cd DRAM-Bender/
  # Study: sources/apps/ for example characterization programs
  # Study: sources/platform/alveo_u200/ for FPGA platform code
  # Study: prebuilt/ for pre-built bitstreams (flash on Day 1)
""")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("SiMRA-DRAM Zenodo Data Analysis")
    print("=" * 70)
    print(f"Data directory: {DATA_DIR}")
    print()

    file_types = explore_dataset(DATA_DIR)

    if file_types is not None:
        parse_csv_results(DATA_DIR)
        analyze_success_rates(DATA_DIR)

    print_analysis_guide()
    print_official_analysis_instructions()

    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("""
1. Download the Zenodo data: https://doi.org/10.5281/zenodo.11165221
2. Run this script to explore the data structure
3. Run the official paper_plots.ipynb notebooks
4. Study the DRAM Bender ISA and example programs
5. When hardware arrives, you'll know exactly what output to expect
""")
