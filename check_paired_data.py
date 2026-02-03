
import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path.cwd() / 'src'))

from data.vitaldb_loader import VitalDBLoader

def check_paired_data():
    print("Checking for paired Propofol + Remifentanil data in VitalDB...")
    
    loader = VitalDBLoader(cache_dir='data/vitaldb_cache', use_cache=True)
    
    # Get all available cases (cached)
    # Note: get_available_cases filters for BIS + Propofol. We need to check if these ALSO have Remifentanil.
    case_ids = loader.get_available_cases(min_duration=1800)
    print(f"Total cases with BIS + Propofol: {len(case_ids)}")
    
    paired_cases = []
    
    for caseid in tqdm(case_ids, desc="Scanning for Remifentanil"):
        # We can't rely just on load_case because it might check cache which might not have RFTN loaded if previously loaded without it.
        # But looking at loader.load_case implementation, it tries to load RFTN tracks.
        # If the cache file exists, it returns the DF. We need to check if that DF has RFTN columns.
        
        df = loader.load_case(caseid)
        
        if df is None:
            continue
            
        has_rftn = False
        if 'RFTN_RATE' in df.columns:
            # Check if it actually has non-zero/non-nan values
            if df['RFTN_RATE'].notna().any() and (df['RFTN_RATE'] > 0).any():
                has_rftn = True
        
        if has_rftn:
            paired_cases.append(caseid)
            
    print("\n" + "="*50)
    print(f"Total Paired Cases Found: {len(paired_cases)}")
    print(f"Percentage: {len(paired_cases)/len(case_ids)*100:.1f}%")
    print("="*50)
    
    if len(paired_cases) > 0:
        print(f"First 10 paired case IDs: {paired_cases[:10]}")

if __name__ == "__main__":
    check_paired_data()
