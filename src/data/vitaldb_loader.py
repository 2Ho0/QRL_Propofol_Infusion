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
        
        # Check first 200 cases
        check_cases = ppf_cases[:min(200, len(ppf_cases))]
        
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
                return pickle.load(f)
        
        try:
            # Load multiple tracks using vitaldb.load_case
            import vitaldb
            
            track_names = [
                'BIS/BIS',                    # Bispectral Index
                'Orchestra/PPF20_RATE',       # Propofol infusion rate (mL/hr)
                'Orchestra/PPF20_CE',         # Propofol effect-site concentration (mcg/mL)
                'Orchestra/PPF20_CP',         # Propofol plasma concentration (mcg/mL)
                'Orchestra/RFTN20_RATE',      # Remifentanil infusion rate (mL/hr, 20 mcg/mL)
                'Orchestra/RFTN20_CE',        # Remifentanil effect-site concentration (ng/mL)
                'Orchestra/RFTN50_RATE',      # Remifentanil infusion rate (mL/hr, 50 mcg/mL)
                'Orchestra/RFTN50_CE',        # Remifentanil effect-site concentration (ng/mL)
            ]
            
            # Load data
            data = vitaldb.load_case(caseid, track_names, interval)
            
            if data is None or len(data) == 0:
                return None
            
            # Create DataFrame
            track_short_names = ['BIS', 'PPF20_RATE', 'PPF20_CE', 'PPF20_CP', 
                               'RFTN20_RATE', 'RFTN20_CE', 'RFTN50_RATE', 'RFTN50_CE']
            
            df_dict = {}
            for i, track_name in enumerate(track_short_names):
                if i < data.shape[1]:
                    df_dict[track_name] = data[:, i]
            
            if 'BIS' not in df_dict or 'PPF20_RATE' not in df_dict:
                return None
            
            # Combine RFTN20 and RFTN50 into single RFTN columns (prefer RFTN20, fallback to RFTN50)
            if 'RFTN20_RATE' in df_dict and 'RFTN50_RATE' in df_dict:
                rftn20_rate = df_dict['RFTN20_RATE']
                rftn50_rate = df_dict['RFTN50_RATE']
                # Use RFTN20 where available, otherwise use RFTN50 (scaled)
                df_dict['RFTN_RATE'] = np.where(
                    pd.notna(rftn20_rate) & (rftn20_rate > 0),
                    rftn20_rate,
                    rftn50_rate * (50.0 / 20.0) if pd.notna(rftn50_rate).any() else rftn20_rate
                )
                
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
            
            # Get patient weight for unit conversion
            try:
                clinical_data = vitaldb.load_case(caseid, ['Clinical/weight'], interval)
                patient_weight = clinical_data[0, 0] if clinical_data is not None and len(clinical_data) > 0 else 70.0
                if np.isnan(patient_weight) or patient_weight <= 0:
                    patient_weight = 70.0
            except:
                patient_weight = 70.0
            
            # Unit conversions
            # 1. PPF20_RATE: mL/hr (20 mg/mL) → mg/kg/h
            df['PPF20_RATE'] = df['PPF20_RATE'] * 20.0 / patient_weight
            
            # 2. RFTN_RATE: mL/hr (20 or 50 mcg/mL) → μg/kg/min
            # Already scaled to 20 mcg/mL equivalent in the combination logic above
            if 'RFTN_RATE' in df.columns:
                df['RFTN_RATE'] = df['RFTN_RATE'] * 20.0 / patient_weight / 60.0
            
            df['time'] = np.arange(len(df)) * interval
            df['caseid'] = caseid
            df['weight'] = patient_weight
            
            # Cache results
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)
            
            return df
            
        except Exception as e:
            # print(f"  Error loading case {caseid}: {e}")
            return None
    
    def prepare_training_data(
        self,
        n_cases: int = 100,
        bis_range: Optional[Tuple[float, float]] = None,
        min_duration: int = 1800,
        save_path: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Prepare training data for model-based QRL.
        
        Args:
            n_cases: Number of cases to load
            bis_range: BIS range filter (default: self.bis_range)
            min_duration: Minimum case duration in seconds
            save_path: Path to save prepared data
            
        Returns:
            Dictionary with states, actions, rewards, next_states, dones
        """
        if bis_range is None:
            bis_range = self.bis_range
        
        print(f"\nPreparing training data...")
        print(f"  Target cases: {n_cases}")
        print(f"  BIS range: {bis_range}")
        print(f"  Min duration: {min_duration}s")
        
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
                    
                    if action < 0 or action > 30:  # Unrealistic propofol rates
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
            'actions': np.array(actions_list, dtype=np.float32).reshape(-1, 1),
            'rewards': np.array(rewards_list, dtype=np.float32),
            'next_states': np.array(next_states_list, dtype=np.float32),
            'dones': np.array(dones_list, dtype=np.bool_),
        }
        
        print(f"\n✓ Training data prepared:")
        print(f"  Valid cases: {valid_cases}/{n_cases}")
        print(f"  Total transitions: {len(data['states']):,}")
        print(f"  States shape: {data['states'].shape}")
        print(f"  Actions range: [{data['actions'].min():.2f}, {data['actions'].max():.2f}]")
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
            save_path: Path to save prepared data
            
        Returns:
            Tuple of (states, actions, next_states, rewards, dones) where:
            - states: (N, 10) - Extended state for dual drug
            - actions: (N, 2) - [propofol_rate, remifentanil_rate]
            - next_states: (N, 10)
            - rewards: (N,) - Computed from VitalDB BIS tracking
            - dones: (N,) - Episode termination flags
        """
        if bis_range is None:
            bis_range = self.bis_range
        
        print(f"\nPreparing dual drug training data...")
        print(f"  Target cases: {n_cases}")
        print(f"  BIS range: {bis_range}")
        print(f"  Min duration: {min_duration}s")
        
        print(f"  Mode: REAL VitalDB remifentanil data (Orchestra/RFTN20 or RFTN50)")
        
        # Get available cases
        case_ids = self.get_available_cases(min_duration=min_duration)
        
        states_list = []
        actions_list = []
        next_states_list = []
        rewards_list = []  # Real rewards from VitalDB data
        dones_list = []    # Episode termination flags
        
        valid_cases = 0
        processed_cases = 0
        
        for caseid in tqdm(case_ids, desc="Scanning for dual drug cases"):
            if valid_cases >= n_cases:
                break
            
            processed_cases += 1
            
            df = self.load_case(caseid)
            
            if df is None or len(df) < 10:
                continue
            
            # Check if remifentanil data exists
            if 'RFTN_RATE' not in df.columns:
                continue
            
            # Filter for cases with both drugs present
            df_filtered = df[
                (df['BIS'] >= bis_range[0]) & 
                (df['BIS'] <= bis_range[1]) &
                (df['BIS'].notna()) &
                (df['PPF20_RATE'].notna()) &
                (df['RFTN_RATE'].notna()) &
                (df['RFTN_RATE'] > 0.01)  # Remifentanil must be actively used (> 0.01 μg/kg/min)
            ].copy()
            
            if len(df_filtered) < 10:
                continue
            
            # Extract transitions
            transitions_added = 0
            
            for i in range(len(df_filtered) - 1):
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
                    transitions_added += 1
                    
                except Exception as e:
                    continue
            
            if transitions_added > 0:
                valid_cases += 1
                if valid_cases % 10 == 0:
                    print(f"  Found {valid_cases} valid dual drug cases (scanned {processed_cases})...")
        
        # Convert to numpy arrays
        states = np.array(states_list, dtype=np.float32)
        actions = np.array(actions_list, dtype=np.float32)
        next_states = np.array(next_states_list, dtype=np.float32)
        rewards = np.array(rewards_list, dtype=np.float32)
        dones = np.array(dones_list, dtype=np.bool_)
        
        print(f"\n✓ Dual drug training data prepared:")
        print(f"  Valid cases: {valid_cases} (scanned {processed_cases})")
        print(f"  Total transitions: {len(states):,}")
        print(f"  States shape: {states.shape}")
        print(f"  Actions shape: {actions.shape}")
        print(f"  Rewards shape: {rewards.shape}")
        print(f"  Dones shape: {dones.shape}")
        if len(actions) > 0:
            print(f"  Propofol range: [{actions[:, 0].min():.2f}, {actions[:, 0].max():.2f}] mg/kg/h")
            print(f"  Remifentanil range: [{actions[:, 1].min():.2f}, {actions[:, 1].max():.2f}] μg/kg/min")
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
                'dones': dones
            }
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"  Saved to: {save_path}")
        
        return states, actions, next_states, rewards, dones
    

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
        Extract state vector for dual drug control.
        
        State s_t = [
            BIS_error,              # [0] e_t = target - BIS_t (target 50)
            Ce_PPF,                 # [1] Propofol effect-site concentration
            Ce_RFTN,                # [2] Remifentanil effect-site (estimated)
            dBIS/dt,                # [3] BIS derivative
            u_{ppf,t-1},            # [4] Previous propofol action
            u_{rftn,t-1},           # [5] Previous remifentanil action
            PPF_accumulation,       # [6] Cumulative propofol (1 min)
            RFTN_accumulation,      # [7] Cumulative remifentanil (1 min)
            BIS_slope,              # [8] BIS slope (3 min)
            interaction_factor      # [9] Drug interaction indicator
        ]
        
        Returns:
            10D state vector
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
        ppf_acc = df.iloc[start_idx:idx+1]['PPF20_RATE'].sum()
        rftn_acc = df.iloc[start_idx:idx+1]['RFTN_RATE'].fillna(0).sum()
        
        # [8] BIS slope (last 18 time steps = 3 min)
        if idx >= 18:
            bis_slope = (row['BIS'] - df.iloc[idx-18]['BIS']) / 18
        else:
            bis_slope = 0.0
        
        # [9] Interaction factor (synergy between drugs)
        # Simple multiplicative interaction model
        interaction = ce_ppf * ce_rftn
        
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
            interaction      # [9]
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
