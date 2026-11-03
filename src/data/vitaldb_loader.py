"""
VitalDB Dataset Loader for Propofol Infusion Control
=====================================================

This module loads and preprocesses anesthesia data from VitalDB
for training model-based QRL agents.

VitalDB provides:
- BIS (Bispectral Index): 0-100 (target: 40-60)
- Propofol infusion rate: mg/kg/h
- Remifentanil infusion rate: μg/kg/min
- Patient vitals: HR, BP, SpO2, etc.
"""

import vitaldb
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pickle
from collections import deque
from tqdm import tqdm
import torch
from torch.utils.data import Dataset


class VitalDBLoader:
    """
    Load and preprocess VitalDB anesthesia data.
    
    Attributes:
        cache_dir: Local cache directory
        bis_range: Target BIS range for filtering
    """
    
    def __init__(
        self,
        cache_dir: str = "./data/vitaldb_cache",
        bis_range: Tuple[float, float] = (30, 70),
        use_cache: bool = True
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.bis_range = bis_range
        self.use_cache = use_cache
        
        print(f"VitalDB Loader initialized")
        print(f"  BIS range: {bis_range}")
        print(f"  Cache directory: {cache_dir}")
    
    def get_available_cases(self, min_duration: int = 1800) -> List[int]:
        """
        Get list of available cases with propofol and BIS data.
        
        Args:
            min_duration: Minimum case duration in seconds (default: 30 min)
            
        Returns:
            List of case IDs
        """
        cache_file = self.cache_dir / "available_cases.pkl"
        
        if self.use_cache and cache_file.exists():
            print("Loading cached case list...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        print("Fetching case list from VitalDB...")
        
        # Use vitaldb.caseids_ppf to get cases with propofol
        # Then filter for those with BIS data
        import vitaldb
        
        # Get propofol cases
        ppf_cases = list(vitaldb.caseids_ppf)  # Convert set to list
        print(f"  Found {len(ppf_cases)} propofol cases")
        
        # Filter for cases with BIS data and sufficient duration
        available_cases = []
        
        # Check first N cases
        max_check = min(15000, len(ppf_cases))  # Scan up to 15000 cases to find more valid data
        check_cases = ppf_cases[:max_check]
        
        print(f"  Scanning first {len(check_cases)} cases (out of {len(ppf_cases)} total)...")
        
        for caseid in tqdm(check_cases, desc="Checking cases"):
            try:
                # Try to load BIS track to verify it exists
                data = vitaldb.load_case(caseid, ['BIS/BIS'])
                
                if data is not None and len(data) > 0:
                    # Check duration (assuming 0.5 Hz sampling = 2 seconds per sample)
                    duration = len(data) * 2  # Approximate
                    
                    if duration >= min_duration:
                        available_cases.append(caseid)
                        
                        if len(available_cases) % 20 == 0:
                            print(f"  Found {len(available_cases)} valid cases...")
                
            except Exception as e:
                continue
        
        print(f"Found {len(available_cases)} valid cases with BIS and propofol")
        
        # Cache results
        with open(cache_file, 'wb') as f:
            pickle.dump(available_cases, f)
        
        return available_cases
    
    # VitalDB clinical information endpoint. Demographics are NOT exposed as
    # signal tracks, so requesting 'Clinical/age' and friends through
    # vitaldb.load_case() returns NaN and every case silently falls back to the
    # defaults (age 50, weight 70, height 170, male). That destroys all
    # inter-patient PK/PD variability, so demographics are joined from here
    # instead, keyed on caseid.
    CLINICAL_URL = "https://api.vitaldb.net/cases"
    _clinical_cache: Optional[pd.DataFrame] = None

    def load_clinical_info(self) -> pd.DataFrame:
        """Return the VitalDB clinical table indexed by caseid.

        Cached on disk after the first call so later runs work offline.

        Returns:
            DataFrame indexed by caseid with subjectid, age, sex, height,
            weight, and bmi.  ``subjectid`` is the grouping key used to keep
            repeat operations from the same person in one data partition.
        """
        if VitalDBLoader._clinical_cache is not None:
            return VitalDBLoader._clinical_cache

        path = self.cache_dir / "clinical_info.pkl"
        if path.exists():
            df = pickle.load(open(path, "rb"))
        else:
            df = None

        # Older caches deliberately kept only demographics and therefore do
        # not contain VitalDB's person identifier.  Such a cache cannot
        # support a subject-disjoint split, so refresh it from the same
        # official endpoint instead of silently falling back to case IDs.
        if df is None or "subjectid" not in df.columns:
            print(f"Fetching VitalDB clinical information from {self.CLINICAL_URL} ...")
            df = pd.read_csv(self.CLINICAL_URL)
            keep = [c for c in (
                "caseid", "subjectid", "age", "sex", "height", "weight", "bmi")
                    if c in df.columns]
            if "subjectid" not in keep:
                raise RuntimeError(
                    "VitalDB clinical metadata does not expose subjectid; "
                    "a subject-disjoint experiment cannot proceed")
            df = df[keep].set_index("caseid")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(df, f)
            print(f"  Cached {len(df):,} clinical records to {path}")

        VitalDBLoader._clinical_cache = df
        return df

    def _apply_clinical_info(self, df: pd.DataFrame, caseid: int) -> pd.DataFrame:
        """Overwrite a case frame's demographics with the real clinical record.

        Args:
            df: Case frame, possibly carrying fallback demographics.
            caseid: VitalDB case id.

        Returns:
            The frame with age/sex/height/weight/bmi taken from the clinical
            table and the per-kg infusion rates re-expressed per kg of that
            weight. Unchanged if the case has no clinical record.
        """
        try:
            rec = self.load_clinical_info().loc[caseid]
        except Exception:
            return df

        height = float(rec["height"])
        weight = float(rec["weight"])
        if not (height > 0 and weight > 0):
            return df

        # The rates were converted from pump mL/h with the weight stored in the
        # frame, which is the 70-kg fallback because load_case cannot read
        # Clinical/weight. Re-express them per kg of the recorded weight.
        conversion_weight = float(df["weight"].iloc[0]) if len(df) else weight
        if conversion_weight > 0 and conversion_weight != weight:
            for column in ("PPF20_RATE", "RFTN_RATE"):
                if column in df.columns:
                    df[column] = df[column] * (conversion_weight / weight)

        df["age"] = float(rec["age"])
        df["height"] = height
        df["weight"] = weight
        # Downstream code reads sex numerically, where >= 0.5 means male.
        df["sex"] = 1.0 if str(rec["sex"]).upper().startswith("M") else 0.0
        df["bmi"] = weight / ((height / 100.0) ** 2)
        return df

    def load_case(self, caseid: int, interval: float = 10.0) -> Optional[pd.DataFrame]:
        """
        Load a single case with all relevant tracks.
        
        Args:
            caseid: VitalDB case ID
            interval: Sampling interval in seconds
            
        Returns:
            DataFrame with time-aligned data or None if failed
        """
        cache_file = self.cache_dir / f"case_{caseid}.pkl"
        
        if self.use_cache and cache_file.exists():
            with open(cache_file, 'rb') as f:
                # Existing cache entries were written with fallback demographics,
                # so the real clinical record is applied on every load.
                return self._apply_clinical_info(pickle.load(f), caseid)

        try:
            # Load ALL tracks in a single API call (data + demographics)
            # This halves download time by avoiding a second API call for clinical data
            import vitaldb
            
            track_names = [
                'BIS/BIS',                    # [0] Bispectral Index
                'Orchestra/PPF20_RATE',       # [1] Propofol infusion rate (mL/hr)
                'Orchestra/PPF20_CE',         # [2] Propofol effect-site concentration (mcg/mL)
                'Orchestra/PPF20_CP',         # [3] Propofol plasma concentration (mcg/mL)
                'Orchestra/RFTN20_RATE',      # [4] Remifentanil infusion rate (mL/hr, 20 mcg/mL)
                'Orchestra/RFTN20_CE',        # [5] Remifentanil effect-site concentration (ng/mL)
                'Orchestra/RFTN50_RATE',      # [6] Remifentanil infusion rate (mL/hr, 50 mcg/mL)
                'Orchestra/RFTN50_CE',        # [7] Remifentanil effect-site concentration (ng/mL)
                'Clinical/weight',            # [8] Patient weight (kg)
                'Clinical/height',            # [9] Patient height (cm)
                'Clinical/age',               # [10] Patient age (years)
                'Clinical/sex',               # [11] Patient sex
            ]
            
            # Load data — single API call for all tracks
            data = vitaldb.load_case(caseid, track_names, interval)
            
            if data is None or len(data) == 0:
                return None
            
            # Create DataFrame from time-series tracks (first 8 columns)
            track_short_names = ['BIS', 'PPF20_RATE', 'PPF20_CE', 'PPF20_CP', 
                               'RFTN20_RATE', 'RFTN20_CE', 'RFTN50_RATE', 'RFTN50_CE']
            
            df_dict = {}
            for i, track_name in enumerate(track_short_names):
                if i < data.shape[1]:
                    df_dict[track_name] = data[:, i]
            
            if 'BIS' not in df_dict or 'PPF20_RATE' not in df_dict:
                return None
            
            # Extract demographics from the same data array (columns 8-11)
            # Clinical tracks are constant values repeated across all rows
            try:
                patient_weight = float(data[0, 8]) if data.shape[1] > 8 and not np.isnan(data[0, 8]) else 70.0
                patient_height = float(data[0, 9]) if data.shape[1] > 9 and not np.isnan(data[0, 9]) else 170.0
                patient_age = float(data[0, 10]) if data.shape[1] > 10 and not np.isnan(data[0, 10]) else 50.0
                patient_sex_raw = data[0, 11] if data.shape[1] > 11 else np.nan
            except (IndexError, TypeError):
                patient_weight = 70.0
                patient_height = 170.0
                patient_age = 50.0
                patient_sex_raw = np.nan
            
            # Validate demographics
            if patient_weight <= 0:
                patient_weight = 70.0
            if patient_height <= 0:
                patient_height = 170.0
            if patient_age <= 0:
                patient_age = 50.0
            
            # Sex: VitalDB encodes as F=0, M=1 (numeric) or string
            if pd.isna(patient_sex_raw):
                patient_sex = 'M'
            elif isinstance(patient_sex_raw, str):
                patient_sex = patient_sex_raw if patient_sex_raw in ['M', 'F'] else 'M'
            else:
                # Numeric: 0=F, 1=M (VitalDB convention)
                patient_sex = 'M' if float(patient_sex_raw) >= 0.5 else 'F'
            
            # Combine RFTN20 and RFTN50 into single RFTN columns (prefer RFTN20, fallback to RFTN50)
            # Store which concentration was used for later unit conversion
            rftn_concentration = 20.0  # Default
            if 'RFTN20_RATE' in df_dict and 'RFTN50_RATE' in df_dict:
                rftn20_rate = df_dict['RFTN20_RATE']
                rftn50_rate = df_dict['RFTN50_RATE']
                # Use RFTN20 where available, otherwise use RFTN50 (keep as-is for now)
                use_rftn20 = pd.notna(rftn20_rate) & (rftn20_rate > 0)
                df_dict['RFTN_RATE'] = np.where(use_rftn20, rftn20_rate, rftn50_rate)
                # Track which concentration: 20 or 50 mcg/mL
                rftn_concentration = np.where(use_rftn20, 20.0, 50.0)
                
                rftn20_ce = df_dict.get('RFTN20_CE', np.full(len(rftn20_rate), np.nan))
                rftn50_ce = df_dict.get('RFTN50_CE', np.full(len(rftn20_rate), np.nan))
                df_dict['RFTN_CE'] = np.where(
                    pd.notna(rftn20_ce) & (rftn20_ce > 0),
                    rftn20_ce,
                    rftn50_ce if pd.notna(rftn50_ce).any() else rftn20_ce
                )
            elif 'RFTN20_RATE' in df_dict:
                df_dict['RFTN_RATE'] = df_dict['RFTN20_RATE']
                df_dict['RFTN_CE'] = df_dict.get('RFTN20_CE', np.full(len(df_dict['RFTN20_RATE']), np.nan))
            elif 'RFTN50_RATE' in df_dict:
                df_dict['RFTN_RATE'] = df_dict['RFTN50_RATE']
                df_dict['RFTN_CE'] = df_dict.get('RFTN50_CE', np.full(len(df_dict['RFTN50_RATE']), np.nan))
            
            df = pd.DataFrame(df_dict)
            
            # Compute BMI
            height_m = patient_height / 100.0
            patient_bmi = patient_weight / (height_m ** 2)
            
            # Convert sex to numeric (0=Female, 1=Male)
            sex_numeric = 1.0 if patient_sex == 'M' else 0.0
            
            # Unit conversions
            # 1. PPF20_RATE: mL/hr (20 mg/mL) → mg/kg/h
            df['PPF20_RATE'] = df['PPF20_RATE'] * 20.0 / patient_weight
            
            # 2. RFTN_RATE: mL/hr (20 or 50 mcg/mL) → μg/kg/min
            # Convert using appropriate concentration (20 or 50 mcg/mL)
            if 'RFTN_RATE' in df.columns:
                if isinstance(rftn_concentration, np.ndarray):
                    # Element-wise conversion with varying concentrations
                    df['RFTN_RATE'] = df['RFTN_RATE'] * rftn_concentration / patient_weight / 60.0
                else:
                    # Fixed concentration
                    df['RFTN_RATE'] = df['RFTN_RATE'] * rftn_concentration / patient_weight / 60.0
            
            df['time'] = np.arange(len(df)) * interval
            df['caseid'] = caseid
            
            # Add demographics to DataFrame
            df['weight'] = patient_weight
            df['height'] = patient_height
            df['age'] = patient_age
            df['sex'] = sex_numeric
            df['bmi'] = patient_bmi
            
            # Cache results
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)

            return self._apply_clinical_info(df, caseid)

        except Exception as e:
            # print(f"  Error loading case {caseid}: {e}")
            return None
    
    def prepare_training_data(
        self,
        n_cases: int = 100,
        bis_range: Optional[Tuple[float, float]] = None,
        min_duration: int = 1800,
        save_path: Optional[str] = None,
        action_max: float = 12.0,  # Match clinical maximum propofol rate (mg/kg/h)
        sampling_interval: int = 60  # Sample every Nth second (1=all data, 60=98.3% reduction)
    ) -> Dict[str, np.ndarray]:
        """
        Prepare training data for model-based QRL.
        
        Args:
            n_cases: Number of cases to load
            bis_range: BIS range filter (default: self.bis_range)
            min_duration: Minimum case duration in seconds
            save_path: Path to save prepared data
            action_max: Maximum action scale (for normalization to [0, 1])
            sampling_interval: Sample every Nth second (1=all data, 60=98.3% reduction)
            
        Returns:
            Dictionary with states, actions, rewards, next_states, dones
        """
        if bis_range is None:
            bis_range = self.bis_range
        
        print(f"\nPreparing training data...")
        print(f"  Target cases: {n_cases}")
        print(f"  BIS range: {bis_range}")
        print(f"  Min duration: {min_duration}s")
        print(f"  Sampling interval: {sampling_interval}s ({'all data' if sampling_interval == 1 else f'{100 * (1 - 1/sampling_interval):.0f}% data reduction'})")
        
        # Get available cases
        case_ids = self.get_available_cases(min_duration=min_duration)
        
        if len(case_ids) < n_cases:
            print(f"  Warning: Only {len(case_ids)} cases available, using all")
            n_cases = len(case_ids)
        
        # Sample cases
        np.random.seed(42)
        selected_cases = np.random.choice(case_ids, n_cases, replace=False)
        
        states_list = []
        actions_list = []
        rewards_list = []
        next_states_list = []
        dones_list = []
        
        valid_cases = 0
        
        for caseid in tqdm(selected_cases, desc="Loading cases"):
            df = self.load_case(caseid)
            
            if df is None or len(df) < 10:
                continue
            
            # Filter by BIS range
            df_filtered = df[
                (df['BIS'] >= bis_range[0]) & 
                (df['BIS'] <= bis_range[1]) &
                (df['BIS'].notna()) &
                (df['PPF20_RATE'].notna())
            ].copy()
            
            if len(df_filtered) < 10:
                continue
            
            # Extract transitions
            for i in range(len(df_filtered) - 1):
                # Apply subsampling: only process every Nth row
                if i % sampling_interval != 0:
                    continue
                
                try:
                    # State at time t
                    state = self._extract_state(df_filtered, i)
                    
                    # Action at time t (propofol rate)
                    action = df_filtered.iloc[i]['PPF20_RATE']
                    
                    # State at time t+1
                    next_state = self._extract_state(df_filtered, i + 1)
                    
                    # Reward (based on BIS target = 50)
                    bis_t = df_filtered.iloc[i]['BIS']
                    reward = self._compute_reward(bis_t)
                    
                    # Done flag (end of episode)
                    done = (i == len(df_filtered) - 2)
                    
                    # Validate data
                    if np.any(np.isnan(state)) or np.any(np.isnan(next_state)) or np.isnan(action):
                        continue
                    
                    if action < 0 or action > 12:  # Clinical maximum propofol rate (mg/kg/h)
                        continue
                    
                    states_list.append(state)
                    actions_list.append(action)
                    rewards_list.append(reward)
                    next_states_list.append(next_state)
                    dones_list.append(done)
                    
                except Exception as e:
                    continue
            
            valid_cases += 1
        
        # Convert to numpy arrays
        data = {
            'states': np.array(states_list, dtype=np.float32),
            'actions': np.array(actions_list, dtype=np.float32).reshape(-1, 1) / action_max,  # Normalize to [0, 1]
            'rewards': np.array(rewards_list, dtype=np.float32),
            'next_states': np.array(next_states_list, dtype=np.float32),
            'dones': np.array(dones_list, dtype=np.bool_),
        }
        
        print(f"\n✓ Training data prepared:")
        print(f"  Valid cases: {valid_cases}/{n_cases}")
        print(f"  Total transitions: {len(data['states']):,}")
        print(f"  States shape: {data['states'].shape}")
        print(f"  Actions range (normalized): [{data['actions'].min():.4f}, {data['actions'].max():.4f}]")
        print(f"  Actions range (raw μg/kg/min): [{data['actions'].min() * action_max:.2f}, {data['actions'].max() * action_max:.2f}]")
        print(f"  BIS range: [{50 - data['states'][:, 0].max():.1f}, {50 - data['states'][:, 0].min():.1f}]")
        print(f"  Mean reward: {data['rewards'].mean():.3f}")
        
        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"  Saved to: {save_path}")
        
        return data
    
    def _extract_state_dualdrug(self, df: pd.DataFrame, idx: int) -> np.ndarray:
        """
        Extract dual drug state vector (13 dimensions).
        
        State s_t = [
            BIS_error,           # e_t = 50 - BIS_t
            Ce_PPF,              # Propofol effect-site concentration
            Ce_RFTN,             # Remifentanil effect-site concentration
            dBIS/dt,             # BIS derivative
            u_ppf_{t-1},         # Previous propofol action
            u_rftn_{t-1},        # Previous remifentanil action
            PPF_accumulation,    # Cumulative propofol (1 min)
            RFTN_accumulation,   # Cumulative remifentanil (1 min)
            BIS_slope,           # BIS slope (3 min)
            Age,                 # Patient demographics
            Weight,
            Height,
            Sex
        ]
        
        Returns:
            state: numpy array of shape (13,)
        """
        row = df.iloc[idx]
        
        # Core state variables
        bis_error = 50 - row['BIS']  # Target BIS = 50
        ce_ppf = row.get('Ce_PPF', 0.0) if 'Ce_PPF' in df.columns else 0.0
        ce_rftn = row.get('Ce_RFTN', 0.0) if 'Ce_RFTN' in df.columns else 0.0
        
        # BIS derivative (dBIS/dt)
        if idx > 0:
            bis_prev = df.iloc[idx - 1]['BIS']
            dt = 1  # 1 second interval
            dbis_dt = (row['BIS'] - bis_prev) / dt
        else:
            dbis_dt = 0.0
        
        # Previous actions
        if idx > 0:
            u_ppf_prev = df.iloc[idx - 1]['PPF20_RATE']
            u_rftn_prev = df.iloc[idx - 1].get('RFTN_RATE', 0.0)
        else:
            u_ppf_prev = 0.0
            u_rftn_prev = 0.0
        
        # Cumulative drug amounts (past 60 seconds)
        if idx >= 60:
            ppf_acc = df.iloc[idx-60:idx]['PPF20_RATE'].sum()
            rftn_acc = df.iloc[idx-60:idx].get('RFTN_RATE', pd.Series([0]*60)).sum()
        else:
            ppf_acc = df.iloc[:idx]['PPF20_RATE'].sum()
            rftn_acc = df.iloc[:idx].get('RFTN_RATE', pd.Series([0]*idx)).sum()
        
        # BIS slope (past 180 seconds)
        if idx >= 180:
            bis_values = df.iloc[idx-180:idx]['BIS'].values
            time_steps = np.arange(len(bis_values))
            if len(bis_values) > 1:
                bis_slope = np.polyfit(time_steps, bis_values, 1)[0]
            else:
                bis_slope = 0.0
        else:
            bis_slope = 0.0
        
        # Patient demographics
        age = row.get('age', 50.0) / 100.0  # Normalize
        weight = row.get('weight', 70.0) / 100.0  # Normalize
        height = row.get('height', 170.0) / 200.0  # Normalize
        sex = row.get('sex', 1.0)  # 1=male, 0=female
        
        state = np.array([
            bis_error,
            ce_ppf,
            ce_rftn,
            dbis_dt,
            u_ppf_prev,
            u_rftn_prev,
            ppf_acc,
            rftn_acc,
            bis_slope,
            age,
            weight,
            height,
            sex
        ], dtype=np.float32)
        
        return state
    
    def _extract_state(self, df: pd.DataFrame, idx: int) -> np.ndarray:
        """
        Extract state vector following CBIM paper formulation (36)-(39).
        
        State s_t = [
            BIS_error,           # e_t = g - BIS_t (target 50)
            Ce_PPF,              # Effect-site concentration
            dBIS/dt,             # BIS derivative
            u_{t-1},             # Previous action
            PPF_accumulation,    # Cumulative propofol (1 min)
            RFTN_accumulation,   # Cumulative remifentanil (1 min)
            BIS_slope,           # BIS slope (3 min)
            RFTN_current         # Current remifentanil rate
        ]
        """
        row = df.iloc[idx]
        
        # BIS error (target = 50)
        bis_error = 50 - row['BIS']
        
        # Effect-site concentration
        ce_ppf = row.get('PPF20_CE', 0.0)
        if pd.isna(ce_ppf):
            ce_ppf = 0.0
        
        # BIS derivative (approximate)
        if idx > 0:
            dbis_dt = row['BIS'] - df.iloc[idx-1]['BIS']
        else:
            dbis_dt = 0.0
        
        # Previous action
        if idx > 0:
            u_prev = df.iloc[idx-1]['PPF20_RATE']
        else:
            u_prev = row['PPF20_RATE']
        
        # Cumulative propofol (last 6 time steps = 1 min at 10s interval)
        start_idx = max(0, idx - 6)
        ppf_acc = df.iloc[start_idx:idx+1]['PPF20_RATE'].sum()
        
        # Cumulative remifentanil (convert from mL/hr to mcg/kg/min if needed)
        if 'RFTN_RATE' in df.columns:
            rftn_acc = df.iloc[start_idx:idx+1]['RFTN_RATE'].fillna(0).sum()
            rftn_current = row.get('RFTN_RATE', 0.0)
        else:
            rftn_acc = 0.0
            rftn_current = 0.0
        
        if pd.isna(rftn_current):
            rftn_current = 0.0
        
        # Note: RFTN_RATE is in mL/hr, will be converted later based on concentration
        # BIS derivative (approximate)
        if idx > 0:
            dbis_dt = row['BIS'] - df.iloc[idx-1]['BIS']
        else:
            dbis_dt = 0.0
        
        # Previous action
        if idx > 0:
            u_prev = df.iloc[idx-1]['PPF20_RATE']
        else:
            u_prev = row['PPF20_RATE']
        
        # Cumulative propofol (last 6 time steps = 1 min at 10s interval)
        start_idx = max(0, idx - 6)
        ppf_acc = df.iloc[start_idx:idx+1]['PPF20_RATE'].sum()
        
        # Cumulative remifentanil
        if 'RFTN_RATE' in df.columns:
            rftn_acc = df.iloc[start_idx:idx+1]['RFTN_RATE'].fillna(0).sum()
            rftn_current = row.get('RFTN_RATE', 0.0)
        else:
            rftn_acc = 0.0
            rftn_current = 0.0
        
        if pd.isna(rftn_current):
            rftn_current = 0.0
        
        # BIS slope (last 18 time steps = 3 min)
        if idx >= 18:
            bis_slope = (row['BIS'] - df.iloc[idx-18]['BIS']) / 18
        else:
            bis_slope = 0.0
        
        state = np.array([
            bis_error,
            ce_ppf,
            dbis_dt,
            u_prev,
            ppf_acc,
            rftn_acc,
            bis_slope,
            rftn_current
        ], dtype=np.float32)
        
        return state
    
    def _compute_reward(self, bis: float, target: float = 50.0, alpha: float = 1.0) -> float:
        """
        Compute reward based on BIS value.
        
        Following CBIM paper formulation (40):
        R_t = 1 / (|g - BIS| + α)
        
        Args:
            bis: Current BIS value
            target: Target BIS value (default: 50)
            alpha: Smoothing parameter (default: 1.0)
            
        Returns:
            Reward value
        """
        error = abs(target - bis)
        reward = 1.0 / (error + alpha)
        return reward
    
    def prepare_dual_drug_training_data(
        self,
        n_cases: int = 100,
        bis_range: Optional[Tuple[float, float]] = None,
        min_duration: int = 1800,
        sampling_interval: int = 60,
        save_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data for dual drug control (Propofol + Remifentanil).
        
        Loads VitalDB cases with real remifentanil data (Orchestra/RFTN20_RATE or RFTN50_RATE).
        Units are automatically converted: PPF (mg/kg/h), RFTN (μg/kg/min).
        
        Args:
            n_cases: Number of cases to load
            bis_range: BIS range filter (default: self.bis_range)
            min_duration: Minimum case duration in seconds
            sampling_interval: Sample every Nth second (1=all data, 5=every 5 seconds for 80% reduction)
            save_path: Path to save prepared data
            
        Returns:
            Tuple of (states, actions, next_states, rewards, dones, caseids) where:
            - states: (N, 13) - Extended state for dual drug with demographics
            - actions: (N, 2) - [propofol_rate, remifentanil_rate]
            - next_states: (N, 13)
            - rewards: (N,) - Computed from VitalDB BIS tracking
            - dones: (N,) - Episode termination flags
            - caseids: (N,) - VitalDB case id per transition, the grouping key
              for patient-level splitting. Splitting on anything finer leaks the
              test set into training, since transitions within a case are
              highly correlated.
        """
        if bis_range is None:
            bis_range = self.bis_range
        
        print(f"\nPreparing dual drug training data...")
        print(f"  Target cases: {n_cases}")
        print(f"  BIS range: {bis_range}")
        print(f"  Min duration: {min_duration}s")
        print(f"  Sampling interval: {sampling_interval}s ({'all data' if sampling_interval == 1 else f'{100 * (1 - 1/sampling_interval):.0f}% data reduction'})")
        
        print(f"  Mode: REAL VitalDB remifentanil data (Orchestra/RFTN20 or RFTN50)")
        
        # Get available cases
        case_ids = self.get_available_cases(min_duration=min_duration)
        
        states_list = []
        actions_list = []
        next_states_list = []
        rewards_list = []  # Real rewards from VitalDB data
        dones_list = []    # Episode termination flags
        caseids_list = []  # Grouping key for patient-level splitting

        valid_cases = 0
        processed_cases = 0
        
        # Debug counters
        debug_stats = {
            'no_data': 0,
            'no_remifentanil': 0,
            'no_demographics': 0,
            'insufficient_samples': 0,
            'no_transitions': 0
        }
        
        for caseid in tqdm(case_ids, desc="Scanning for dual drug cases"):
            if valid_cases >= n_cases:
                break
            
            processed_cases += 1
            
            df = self.load_case(caseid)
            
            if df is None or len(df) < 10:
                debug_stats['no_data'] += 1
                continue
            
            # Check if demographics exist (required for 13D state)
            if 'age' not in df.columns or 'sex' not in df.columns or 'bmi' not in df.columns:
                debug_stats['no_demographics'] += 1
                continue
            
            # Check if remifentanil data exists
            if 'RFTN_RATE' not in df.columns:
                debug_stats['no_remifentanil'] += 1
                continue
            
            # Filter for cases with both drugs present. The threshold is only
            # a non-zero pump-rate check; it is not a claimed clinical target.
            df_filtered = df[
                (df['BIS'] >= bis_range[0]) & 
                (df['BIS'] <= bis_range[1]) &
                (df['BIS'].notna()) &
                (df['PPF20_RATE'].notna()) &
                (df['RFTN_RATE'].notna()) &
                (df['RFTN_RATE'] > 0.01)  # Remifentanil > 0.01 μg/kg/min (clinically meaningful dual drug)
            ].copy()
            
            # Debug: Log case filtering details
            if processed_cases <= 10:  # Print first 10 cases
                rftn_mean = df['RFTN_RATE'].mean() if 'RFTN_RATE' in df.columns else 0
                rftn_max = df['RFTN_RATE'].max() if 'RFTN_RATE' in df.columns else 0
                print(f"    Case {caseid}: RFTN mean={rftn_mean:.2f}, max={rftn_max:.2f} μg/kg/min, filtered samples={len(df_filtered)}")
            
            if len(df_filtered) < 10:
                debug_stats['insufficient_samples'] += 1
                continue
            
            # Extract transitions
            transitions_added = 0
            
            for i in range(len(df_filtered) - 1):
                # Apply subsampling: only process every Nth row
                if i % sampling_interval != 0:
                    continue
                
                try:
                    # State at time t (10D for dual drug)
                    state = self._extract_dual_drug_state(df_filtered, i)
                    
                    # Action at time t: [propofol_rate, remifentanil_rate] (2D)
                    ppf_rate = df_filtered.iloc[i]['PPF20_RATE']
                    rftn_rate = df_filtered.iloc[i]['RFTN_RATE']
                    action = np.array([ppf_rate, rftn_rate], dtype=np.float32)
                    
                    # State at time t+1
                    next_state = self._extract_dual_drug_state(df_filtered, i + 1)
                    
                    # Compute reward from VitalDB data
                    bis_current = df_filtered.iloc[i]['BIS']
                    bis_next = df_filtered.iloc[i + 1]['BIS']
                    reward = self._compute_vitaldb_reward(
                        bis_current=bis_current,
                        bis_next=bis_next,
                        ppf_rate=ppf_rate,
                        rftn_rate=rftn_rate
                    )
                    
                    # Episode termination flag (last transition of this case)
                    done = (i == len(df_filtered) - 2)
                    
                    # Validate data
                    if np.any(np.isnan(state)) or np.any(np.isnan(next_state)) or np.any(np.isnan(action)):
                        continue
                    
                    if ppf_rate < 0 or ppf_rate > 30:  # Unrealistic propofol rates (mg/kg/h)
                        continue
                    
                    if rftn_rate < 0 or rftn_rate > 50:  # Unrealistic remifentanil rates (μg/kg/min)
                        continue
                    
                    states_list.append(state)
                    actions_list.append(action)
                    next_states_list.append(next_state)
                    rewards_list.append(reward)
                    dones_list.append(done)
                    caseids_list.append(int(caseid))
                    transitions_added += 1
                    
                except Exception as e:
                    continue
            
            if transitions_added > 0:
                valid_cases += 1
                if valid_cases % 10 == 0:
                    print(f"  Found {valid_cases} valid dual drug cases (scanned {processed_cases})...")
            else:
                debug_stats['no_transitions'] += 1
        
        # Print debug statistics
        print(f"\n📊 Debug Statistics:")
        print(f"  Scanned: {processed_cases} cases")
        print(f"  Valid: {valid_cases} cases")
        print(f"  💊 Remifentanil non-zero filter: > 0.01 μg/kg/min")
        print(f"  Filtered out:")
        print(f"    - No data/too short: {debug_stats['no_data']}")
        print(f"    - No demographics (age/sex/BMI): {debug_stats['no_demographics']}")
        print(f"    - No remifentanil data: {debug_stats['no_remifentanil']}")
        print(f"    - Insufficient samples: {debug_stats['insufficient_samples']}")
        print(f"    - No valid transitions: {debug_stats['no_transitions']}")
        
        # Convert to numpy arrays
        states = np.array(states_list, dtype=np.float32)
        actions_raw = np.array(actions_list, dtype=np.float32)
        next_states = np.array(next_states_list, dtype=np.float32)
        rewards = np.array(rewards_list, dtype=np.float32)
        dones = np.array(dones_list, dtype=np.bool_)
        caseids = np.array(caseids_list, dtype=np.int64)
        
        # Normalize actions to [0, 1] range
        # Propofol: 0-12 mg/kg/h → [0, 1]
        # Remifentanil: 0-2 μg/kg/min → [0, 1]
        PPF_MAX = 12.0   # mg/kg/h (typical max)
        RFTN_MAX = 2.0  # μg/kg/min (typical max)
        
        actions = actions_raw.copy()
        actions[:, 0] = actions_raw[:, 0] / PPF_MAX   # Normalize propofol
        actions[:, 1] = actions_raw[:, 1] / RFTN_MAX  # Normalize remifentanil

        # Clip to the action space the policy can actually express. Recorded
        # induction rates exceed the maintenance ceiling on a minority of
        # transitions (~2.5% for propofol, up to 2.4x PPF_MAX); a sigmoid actor
        # cannot represent those, so leaving them unclipped imposes an
        # irreducible behaviour-cloning loss that biases the policy toward
        # saturation. Clipping makes the targets consistent with the stated
        # action space.
        n_clipped = int((actions > 1.0).sum())
        if n_clipped:
            print(f"  Clipped {n_clipped:,} action targets above the ceiling "
                  f"({100 * n_clipped / actions.size:.2f}% of all targets)")
        actions = np.clip(actions, 0.0, 1.0)
        
        print(f"\n✓ Dual drug training data prepared:")
        print(f"  Valid cases: {valid_cases} (scanned {processed_cases})")
        print(f"  Total transitions: {len(states):,}")
        print(f"  States shape: {states.shape}")
        print(f"  Actions shape: {actions.shape}")
        print(f"  Rewards shape: {rewards.shape}")
        print(f"  Dones shape: {dones.shape}")
        if len(actions) > 0:
            print(f"  Propofol RAW: mean={actions_raw[:, 0].mean():.2f}, range=[{actions_raw[:, 0].min():.2f}, {actions_raw[:, 0].max():.2f}] mg/kg/h")
            print(f"  Remifentanil RAW: mean={actions_raw[:, 1].mean():.2f}, range=[{actions_raw[:, 1].min():.2f}, {actions_raw[:, 1].max():.2f}] μg/kg/min")
            ppf_q = np.quantile(actions_raw[:, 0], [0.05, 0.5, 0.95])
            rftn_q = np.quantile(actions_raw[:, 1], [0.05, 0.5, 0.95])
            print(f"  Propofol RAW q05/median/q95: "
                  f"{ppf_q[0]:.2f}/{ppf_q[1]:.2f}/{ppf_q[2]:.2f} mg/kg/h")
            print(f"  Remifentanil RAW q05/median/q95: "
                  f"{rftn_q[0]:.3f}/{rftn_q[1]:.3f}/{rftn_q[2]:.3f} μg/kg/min")
            print(f"  Above policy ceiling: propofol "
                  f"{100 * np.mean(actions_raw[:, 0] > PPF_MAX):.2f}%, "
                  f"remifentanil {100 * np.mean(actions_raw[:, 1] > RFTN_MAX):.2f}%")
            print(f"  Propofol NORMALIZED range: [{actions[:, 0].min():.4f}, {actions[:, 0].max():.4f}]")
            print(f"  Remifentanil NORMALIZED range: [{actions[:, 1].min():.4f}, {actions[:, 1].max():.4f}]")
            print(f"  BIS range: [{50 - states[:, 0].max():.1f}, {50 - states[:, 0].min():.1f}]")
            print(f"  Reward range: [{rewards.min():.3f}, {rewards.max():.3f}]")
            print(f"  Reward mean: {rewards.mean():.3f} ± {rewards.std():.3f}")
            print(f"  Episodes completed: {dones.sum()}")
        
        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'states': states,
                'actions': actions,
                'next_states': next_states,
                'rewards': rewards,
                'dones': dones,
                'caseids': caseids
            }
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"  Saved to: {save_path}")

        return states, actions, next_states, rewards, dones, caseids

    @staticmethod
    def _dual_drug_state_segment(
        frame: pd.DataFrame,
        observation_interval_seconds: int,
    ) -> np.ndarray:
        """Vectorise the 13-D state on one uninterrupted valid segment.

        Derivatives are expressed per minute and the accumulation features are
        physical dose over the previous minute.  Keeping this calculation
        segment-local prevents missing/BIS-excluded gaps from being stitched
        into a seemingly continuous history.
        """
        n = len(frame)
        if n == 0:
            return np.empty((0, 13), dtype=np.float32)

        dt_min = float(observation_interval_seconds) / 60.0
        bis = frame["BIS"].to_numpy(dtype=np.float64)
        ppf = frame["PPF20_RATE"].to_numpy(dtype=np.float64)
        rftn = frame["RFTN_RATE"].to_numpy(dtype=np.float64)
        ce_ppf = frame.get(
            "PPF20_CE", pd.Series(np.zeros(n), index=frame.index)
        ).to_numpy(dtype=np.float64)
        ce_ppf = np.nan_to_num(ce_ppf, nan=0.0)
        ce_rftn = frame.get(
            "RFTN_CE", pd.Series(np.zeros(n), index=frame.index)
        ).to_numpy(dtype=np.float64)
        ce_rftn = np.where(
            np.isfinite(ce_rftn) & (ce_rftn != 0.0),
            ce_rftn,
            np.where(rftn > 0.0, rftn / 0.6, 0.0),
        )

        dbis_dt = np.zeros(n, dtype=np.float64)
        dbis_dt[1:] = np.diff(bis) / dt_min
        previous_ppf = np.concatenate(([ppf[0]], ppf[:-1]))
        previous_rftn = np.concatenate(([rftn[0]], rftn[:-1]))

        window_1min = max(1, int(round(60 / observation_interval_seconds)))

        def rolling_dose(values: np.ndarray, scale: float) -> np.ndarray:
            prefix = np.concatenate(([0.0], np.cumsum(values)))
            stop = np.arange(1, n + 1)
            start = np.maximum(0, stop - window_1min)
            return (prefix[stop] - prefix[start]) * scale

        ppf_acc = rolling_dose(ppf, dt_min / 60.0)  # mg/kg
        rftn_acc = rolling_dose(rftn, dt_min)        # microgram/kg

        window_3min = max(1, int(round(180 / observation_interval_seconds)))
        bis_slope = np.zeros(n, dtype=np.float64)
        if n > window_3min:
            bis_slope[window_3min:] = (
                bis[window_3min:] - bis[:-window_3min]
            ) / (window_3min * dt_min)

        age = frame["age"].to_numpy(dtype=np.float64) / 100.0
        sex = frame["sex"].to_numpy(dtype=np.float64)
        bmi = frame["bmi"].to_numpy(dtype=np.float64) / 40.0
        return np.column_stack((
            50.0 - bis,
            ce_ppf,
            ce_rftn,
            dbis_dt,
            previous_ppf,
            previous_rftn,
            ppf_acc,
            rftn_acc,
            bis_slope,
            ce_ppf * ce_rftn,
            age,
            sex,
            bmi,
        )).astype(np.float32)

    def prepare_dual_drug_control_data(
        self,
        n_cases: int = 3000,
        bis_range: Optional[Tuple[float, float]] = None,
        min_duration: int = 1800,
        observation_interval_seconds: int = 10,
        control_interval_seconds: int = 60,
        sequence_length: int = 60,
        unique_subjects: bool = True,
        save_path: Optional[str] = None,
        caseids: Optional[List[int]] = None,
    ) -> Dict[str, np.ndarray]:
        """Build a gap-safe 10-s-observation/60-s-control offline dataset.

        ``caseids`` replaces the scan over all available cases, so a locked
        cohort can be rebuilt with the same subjects (and therefore the same
        patient-level split).

        Each transition corresponds to one control decision.  The commanded
        action is the time-average recorded infusion over the following
        control interval, while the state and next-state inputs are explicit
        observation windows sampled on the original regular grid.  Invalid
        observations divide a case into separate segments; neither windows nor
        transitions cross those boundaries.

        When ``unique_subjects`` is true, at most one operation is retained per
        VitalDB ``subjectid``.  This makes the requested 2,100/300/600 split
        exactly subject-disjoint while retaining exactly 3,000 cases.
        """
        if bis_range is None:
            bis_range = self.bis_range
        if observation_interval_seconds <= 0 or control_interval_seconds <= 0:
            raise ValueError("observation and control intervals must be positive")
        if control_interval_seconds % observation_interval_seconds:
            raise ValueError("control interval must be divisible by observation interval")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

        control_stride = control_interval_seconds // observation_interval_seconds
        clinical = self.load_clinical_info()
        if "subjectid" not in clinical.columns:
            raise RuntimeError("subjectid is required for the clean protocol")

        print("\nPreparing gap-safe dual-drug control data...")
        print(f"  Cases/subjects: {n_cases:,}")
        print(f"  Observation interval: {observation_interval_seconds}s")
        print(f"  Control interval: {control_interval_seconds}s ({control_stride} observations)")
        print(f"  History: {sequence_length} observations "
              f"({sequence_length * observation_interval_seconds / 60:.1f} min)")

        state_windows = []
        next_windows = []
        action_blocks = []
        reward_blocks = []
        done_blocks = []
        caseid_blocks = []
        subjectid_blocks = []
        time_blocks = []
        segment_blocks = []
        selected_caseids = []
        selected_subjectids = []
        seen_subjects = set()
        segment_id = 0
        debug = {
            "no_data": 0,
            "duplicate_subject": 0,
            "no_subject": 0,
            "invalid_clinical_parameters": 0,
            "no_remifentanil": 0,
            "no_control_transition": 0,
            "irregular_cache": 0,
        }

        if caseids is None:
            caseids = self.get_available_cases(min_duration=min_duration)
        for caseid in tqdm(caseids, desc="Building clean dual-drug cohort"):
            if len(selected_caseids) >= n_cases:
                break
            caseid = int(caseid)
            try:
                clinical_record = clinical.loc[caseid]
                subjectid = int(clinical_record["subjectid"])
            except (KeyError, TypeError, ValueError):
                debug["no_subject"] += 1
                continue
            age = float(clinical_record["age"])
            weight = float(clinical_record["weight"])
            height = float(clinical_record["height"])
            if not (
                np.isfinite([age, weight, height]).all()
                and 0.0 <= age <= 120.0
                and 30.0 <= weight <= 200.0
                and 100.0 <= height <= 250.0
            ):
                debug["invalid_clinical_parameters"] += 1
                continue
            if unique_subjects and subjectid in seen_subjects:
                debug["duplicate_subject"] += 1
                continue

            frame = self.load_case(
                caseid, interval=float(observation_interval_seconds))
            if frame is None or len(frame) <= control_stride:
                debug["no_data"] += 1
                continue
            if "RFTN_RATE" not in frame.columns:
                debug["no_remifentanil"] += 1
                continue

            times = frame["time"].to_numpy(dtype=np.float64)
            if len(times) > 1 and not np.allclose(
                np.diff(times), observation_interval_seconds, atol=1e-6):
                debug["irregular_cache"] += 1
                continue

            valid = (
                frame["BIS"].between(bis_range[0], bis_range[1], inclusive="both")
                & frame["BIS"].notna()
                & frame["PPF20_RATE"].notna()
                & frame["RFTN_RATE"].notna()
                & frame["RFTN_RATE"].gt(0.01)
                & frame["PPF20_RATE"].between(0.0, 30.0, inclusive="both")
                & frame["RFTN_RATE"].between(0.0, 50.0, inclusive="both")
            ).to_numpy()
            valid_indices = np.flatnonzero(valid)
            if not len(valid_indices):
                debug["no_control_transition"] += 1
                continue

            cuts = np.flatnonzero(np.diff(valid_indices) != 1) + 1
            candidate_segments = np.split(valid_indices, cuts)
            case_payload = []
            for indices in candidate_segments:
                if len(indices) <= control_stride:
                    continue
                segment = frame.iloc[indices]
                dense_states = self._dual_drug_state_segment(
                    segment, observation_interval_seconds)
                if not np.isfinite(dense_states).all():
                    continue

                padded = np.concatenate((
                    np.repeat(dense_states[:1], sequence_length - 1, axis=0),
                    dense_states,
                ))
                windows = np.lib.stride_tricks.sliding_window_view(
                    padded, sequence_length, axis=0).transpose(0, 2, 1)
                origins = np.arange(
                    0, len(segment) - control_stride, control_stride,
                    dtype=np.int64)
                if not len(origins):
                    continue
                next_origins = origins + control_stride

                rates = segment[["PPF20_RATE", "RFTN_RATE"]].to_numpy(
                    dtype=np.float64)
                prefix = np.vstack((np.zeros((1, 2)), np.cumsum(rates, axis=0)))
                actions_raw = (
                    prefix[origins + control_stride] - prefix[origins]
                ) / control_stride
                actions = actions_raw / np.array([12.0, 2.0])
                actions = np.clip(actions, 0.0, 1.0).astype(np.float32)

                bis = segment["BIS"].to_numpy(dtype=np.float64)
                current_bis = bis[origins]
                next_bis = bis[next_origins]
                r_bis = 1.0 / (np.abs(next_bis - 50.0) + 1.0)
                r_drug = -0.001 * (actions_raw[:, 0] + 0.1 * actions_raw[:, 1])
                r_stability = -0.1 * np.abs(next_bis - current_bis)
                r_safety = np.where(
                    next_bis < 20.0, -10.0,
                    np.where(next_bis > 70.0, -5.0,
                             np.where(np.abs(next_bis - 50.0) > 30.0, -1.0, 0.0)))
                rewards = (r_bis + r_drug + r_stability + r_safety).astype(
                    np.float32)
                dones = np.zeros(len(origins), dtype=np.bool_)
                dones[-1] = True

                case_payload.append((
                    np.ascontiguousarray(windows[origins], dtype=np.float32),
                    np.ascontiguousarray(windows[next_origins], dtype=np.float32),
                    actions,
                    rewards,
                    dones,
                    np.full(len(origins), caseid, dtype=np.int64),
                    np.full(len(origins), subjectid, dtype=np.int64),
                    segment["time"].to_numpy(dtype=np.float64)[origins],
                    np.full(len(origins), segment_id, dtype=np.int64),
                ))
                segment_id += 1

            if not case_payload:
                debug["no_control_transition"] += 1
                continue
            for payload in case_payload:
                (states, next_states, actions, rewards, dones, caseids,
                 subjectids, timestamps, segmentids) = payload
                state_windows.append(states)
                next_windows.append(next_states)
                action_blocks.append(actions)
                reward_blocks.append(rewards)
                done_blocks.append(dones)
                caseid_blocks.append(caseids)
                subjectid_blocks.append(subjectids)
                time_blocks.append(timestamps)
                segment_blocks.append(segmentids)
            selected_caseids.append(caseid)
            selected_subjectids.append(subjectid)
            seen_subjects.add(subjectid)

        if len(selected_caseids) != n_cases:
            raise RuntimeError(
                f"requested {n_cases} unique-subject cases, found "
                f"{len(selected_caseids)}")

        data = {
            "states": np.concatenate(state_windows),
            "actions": np.concatenate(action_blocks),
            "next_states": np.concatenate(next_windows),
            "rewards": np.concatenate(reward_blocks),
            "dones": np.concatenate(done_blocks),
            "caseids": np.concatenate(caseid_blocks),
            "subjectids": np.concatenate(subjectid_blocks),
            "timestamps": np.concatenate(time_blocks),
            "segmentids": np.concatenate(segment_blocks),
            "selected_caseids": np.asarray(selected_caseids, dtype=np.int64),
            "selected_subjectids": np.asarray(selected_subjectids, dtype=np.int64),
            "observation_interval_seconds": np.asarray(
                observation_interval_seconds, dtype=np.int64),
            "control_interval_seconds": np.asarray(
                control_interval_seconds, dtype=np.int64),
            "sequence_length": np.asarray(sequence_length, dtype=np.int64),
        }
        print(f"  Selected: {len(selected_caseids):,} cases / "
              f"{len(set(selected_subjectids)):,} subjects")
        print(f"  Transitions: {len(data['states']):,}")
        print(f"  State windows: {data['states'].shape}")
        print(f"  Segments: {len(np.unique(data['segmentids'])):,}")
        print(f"  Exclusions: {debug}")

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as stream:
                pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  Saved to: {save_path}")
        return data
    

    def _compute_vitaldb_reward(
        self,
        bis_current: float,
        bis_next: float,
        ppf_rate: float,
        rftn_rate: float
    ) -> float:
        """
        Compute reward from VitalDB observations.
        
        Reward components:
        1. BIS tracking: Closer to target (50) → higher reward
        2. Drug efficiency: Lower drug usage → higher reward
        3. Stability: Smaller BIS changes → higher reward
        4. Safety: Penalize dangerous BIS ranges
        
        Args:
            bis_current: Current BIS value
            bis_next: Next BIS value
            ppf_rate: Propofol rate (mg/kg/h)
            rftn_rate: Remifentanil rate (μg/kg/min)
            
        Returns:
            Reward value
        """
        target_bis = 50.0
        
        # 1. BIS tracking reward (primary objective)
        bis_error = abs(bis_next - target_bis)
        r_bis = 1.0 / (bis_error + 1.0)  # Range: [0, 1], higher when closer to target
        
        # 2. Drug efficiency penalty (minimize drug usage)
        # Remifentanil is ~100x more potent, so weight it higher
        r_drug = -0.001 * (ppf_rate + 0.1 * rftn_rate)
        
        # 3. Stability reward (penalize large BIS fluctuations)
        bis_change = abs(bis_next - bis_current)
        r_stability = -0.1 * bis_change
        
        # 4. Safety penalty (dangerous BIS ranges)
        r_safety = 0.0
        if bis_next < 20:  # Too deep (high risk)
            r_safety = -10.0
        elif bis_next > 70:  # Too light (inadequate anesthesia)
            r_safety = -5.0
        elif bis_error > 30:  # Far from target
            r_safety = -1.0
        
        # Total reward
        reward = r_bis + r_drug + r_stability + r_safety
        
        return reward
    
    def _extract_dual_drug_state(self, df: pd.DataFrame, idx: int) -> np.ndarray:
        """
        Extract state vector for dual drug control with patient demographics.
        
        State s_t = [
            BIS_error,              # [0] e_t = target - BIS_t (target 50)
            Ce_PPF,                 # [1] Propofol effect-site concentration
            Ce_RFTN,                # [2] Remifentanil effect-site concentration
            dBIS/dt,                # [3] BIS derivative
            u_{ppf,t-1},            # [4] Previous propofol action
            u_{rftn,t-1},           # [5] Previous remifentanil action
            PPF_accumulation,       # [6] Cumulative propofol (1 min)
            RFTN_accumulation,      # [7] Cumulative remifentanil (1 min)
            BIS_slope,              # [8] BIS slope (3 min)
            interaction_factor,     # [9] Drug interaction indicator
            age_normalized,         # [10] Age / 100 (0-1 range)
            sex,                    # [11] 0=Female, 1=Male
            bmi_normalized,         # [12] BMI / 40 (0-1 range)
        ]
        
        Returns:
            13D state vector (previously 10D)
        """
        row = df.iloc[idx]
        
        # [0] BIS error (target = 50)
        bis_error = 50 - row['BIS']
        
        # [1] Propofol effect-site concentration
        ce_ppf = row.get('PPF20_CE', 0.0)
        if pd.isna(ce_ppf):
            ce_ppf = 0.0
        
        # [2] Remifentanil effect-site concentration
        # Use RFTN_CE if available, otherwise estimate from rate
        ce_rftn = row.get('RFTN_CE', 0.0)
        if pd.isna(ce_rftn) or ce_rftn == 0.0:
            # Fallback: estimate from rate using ke0 approximation
            rftn_rate = row.get('RFTN_RATE', 0.0)
            if pd.isna(rftn_rate):
                rftn_rate = 0.0
            ce_rftn = rftn_rate / 0.6 if rftn_rate > 0 else 0.0
        
        # [3] BIS derivative
        if idx > 0:
            dbis_dt = row['BIS'] - df.iloc[idx-1]['BIS']
        else:
            dbis_dt = 0.0
        
        # [4-5] Previous actions
        if idx > 0:
            u_ppf_prev = df.iloc[idx-1]['PPF20_RATE']
            u_rftn_prev = df.iloc[idx-1]['RFTN_RATE']
        else:
            u_ppf_prev = row['PPF20_RATE']
            u_rftn_prev = row['RFTN_RATE']
        
        if pd.isna(u_rftn_prev):
            u_rftn_prev = 0.0
        
        # [6-7] Cumulative doses (last 6 time steps = 1 min at 10s interval)
        start_idx = max(0, idx - 6)
        # Convert rate × time to actual dose:
        # PPF20_RATE is in mg/kg/h → multiply by (10s / 3600s/h) to get mg/kg
        # RFTN_RATE is in μg/kg/min → multiply by (10s / 60s/min) to get μg/kg
        timestep_hours = 10.0 / 3600.0  # 10 seconds in hours
        timestep_minutes = 10.0 / 60.0  # 10 seconds in minutes
        ppf_acc = df.iloc[start_idx:idx+1]['PPF20_RATE'].sum() * timestep_hours  # mg/kg
        rftn_acc = df.iloc[start_idx:idx+1]['RFTN_RATE'].fillna(0).sum() * timestep_minutes  # μg/kg
        
        # [8] BIS slope (last 18 time steps = 3 min)
        if idx >= 18:
            bis_slope = (row['BIS'] - df.iloc[idx-18]['BIS']) / 18
        else:
            bis_slope = 0.0
        
        # [9] Interaction factor (synergy between drugs)
        # Simple multiplicative interaction model
        interaction = ce_ppf * ce_rftn
        
        # [10-12] Patient demographics (normalized)
        age_norm = df.iloc[idx]['age'] / 100.0      # 0-100 years → 0-1
        sex = df.iloc[idx]['sex']                    # 0 or 1
        bmi_norm = df.iloc[idx]['bmi'] / 40.0       # 15-40 BMI → 0.375-1.0
        
        state = np.array([
            bis_error,       # [0]
            ce_ppf,          # [1]
            ce_rftn,         # [2]
            dbis_dt,         # [3]
            u_ppf_prev,      # [4]
            u_rftn_prev,     # [5]
            ppf_acc,         # [6]
            rftn_acc,        # [7]
            bis_slope,       # [8]
            interaction,     # [9]
            age_norm,        # [10] NEW: Age normalized
            sex,             # [11] NEW: Sex (0=F, 1=M)
            bmi_norm         # [12] NEW: BMI normalized
        ], dtype=np.float32)
        
        return state


class VitalDBDataset(Dataset):
    """
    PyTorch Dataset for VitalDB data.
    """
    
    def __init__(self, data: Dict[str, np.ndarray]):
        """
        Initialize dataset.
        
        Args:
            data: Dictionary with states, actions, rewards, next_states, dones
        """
        self.states = torch.FloatTensor(data['states'])
        self.actions = torch.FloatTensor(data['actions'])
        self.rewards = torch.FloatTensor(data['rewards'])
        self.next_states = torch.FloatTensor(data['next_states'])
        self.dones = torch.FloatTensor(data['dones'].astype(np.float32))
    
    def __len__(self) -> int:
        return len(self.states)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx]
        )


class PatientSequenceDataset(Dataset):
    """Patient-boundary-safe temporal windows for offline training.

    Every transition remains a training example.  At the beginning of a
    patient trajectory the first observation is repeated on the left, so the
    online and offline policies see the same fixed window length without ever
    borrowing observations from another patient.
    """

    def __init__(self, data: Dict[str, np.ndarray], sequence_length: int = 10):
        if "caseids" not in data:
            raise ValueError("PatientSequenceDataset requires per-transition caseids")
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")

        raw_states = np.asarray(data["states"], dtype=np.float32)
        raw_next = np.asarray(data["next_states"], dtype=np.float32)
        caseids = np.asarray(data["caseids"])
        if not (len(raw_states) == len(raw_next) == len(caseids)):
            raise ValueError("states, next_states and caseids must have equal length")

        if raw_states.ndim == 3:
            if raw_next.shape != raw_states.shape:
                raise ValueError(
                    "pre-windowed states and next_states must have equal shape")
            if raw_states.shape[1] != sequence_length:
                raise ValueError(
                    f"pre-windowed data has T={raw_states.shape[1]}, "
                    f"requested T={sequence_length}")
            state_windows = raw_states
            next_windows = raw_next
        elif raw_states.ndim == 2:
            n, state_dim = raw_states.shape
            state_windows = np.empty(
                (n, sequence_length, state_dim), dtype=np.float32)
            next_windows = np.empty_like(state_windows)
            histories: Dict[int, deque] = {}

            for i, caseid in enumerate(caseids):
                key = int(caseid)
                history = histories.setdefault(key, deque(maxlen=sequence_length))
                history.append(raw_states[i])
                observed = list(history)
                padded = [observed[0]] * (sequence_length - len(observed)) + observed
                state_window = np.asarray(padded, dtype=np.float32)
                state_windows[i] = state_window
                if sequence_length == 1:
                    next_windows[i, 0] = raw_next[i]
                else:
                    next_windows[i] = np.concatenate(
                        [state_window[1:], raw_next[i][None, :]], axis=0)
        else:
            raise ValueError("states must have shape [N,D] or [N,T,D]")

        self.states = torch.from_numpy(state_windows)
        self.actions = torch.as_tensor(data["actions"], dtype=torch.float32)
        self.rewards = torch.as_tensor(data["rewards"], dtype=torch.float32)
        self.next_states = torch.from_numpy(next_windows)
        self.dones = torch.as_tensor(
            np.asarray(data["dones"], dtype=np.float32), dtype=torch.float32)
        self.caseids = torch.as_tensor(caseids)
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )


if __name__ == "__main__":
    # Test VitalDB loader
    print("Testing VitalDB Loader...")
    print("=" * 70)
    
    loader = VitalDBLoader(
        cache_dir="./data/vitaldb_cache",
        bis_range=(30, 70),
        use_cache=True
    )
    
    # Prepare small dataset for testing
    data = loader.prepare_training_data(
        n_cases=10,
        min_duration=1800,
        save_path="./data/offline_dataset/vitaldb_test.pkl"
    )
    
    print(f"\n✓ Test complete!")
    print(f"  Dataset size: {len(data['states']):,} transitions")
