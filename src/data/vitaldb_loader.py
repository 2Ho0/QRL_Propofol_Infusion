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
                'Orchestra/PPF20_RATE',       # Propofol infusion rate (mg/kg/h)
                'Orchestra/PPF20_CE',         # Propofol effect-site concentration
                'Orchestra/PPF20_CP',         # Propofol plasma concentration
                'Orchestra/RFTN_RATE',        # Remifentanil infusion rate (μg/kg/min)
            ]
            
            # Load data
            data = vitaldb.load_case(caseid, track_names, interval)
            
            if data is None or len(data) == 0:
                return None
            
            # Create DataFrame
            track_short_names = ['BIS', 'PPF20_RATE', 'PPF20_CE', 'PPF20_CP', 'RFTN_RATE']
            
            df_dict = {}
            for i, track_name in enumerate(track_short_names):
                if i < data.shape[1]:
                    df_dict[track_name] = data[:, i]
            
            if 'BIS' not in df_dict or 'PPF20_RATE' not in df_dict:
                return None
            
            df = pd.DataFrame(df_dict)
            df['time'] = np.arange(len(df)) * interval
            df['caseid'] = caseid
            
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
