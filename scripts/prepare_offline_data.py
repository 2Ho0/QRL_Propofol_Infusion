"""
Prepare Offline Dataset from VitalDB
=====================================

This script downloads and preprocesses VitalDB data for offline RL training.

Usage:
    python scripts/prepare_offline_data.py --n_cases 100 --bis_range 30 70
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import argparse
from data.vitaldb_loader import VitalDBLoader
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Prepare VitalDB offline dataset')
    parser.add_argument('--n_cases', type=int, default=100,
                        help='Number of cases to load (default: 100)')
    parser.add_argument('--bis_min', type=float, default=30,
                        help='Minimum BIS value (default: 30)')
    parser.add_argument('--bis_max', type=float, default=70,
                        help='Maximum BIS value (default: 70)')
    parser.add_argument('--min_duration', type=int, default=1800,
                        help='Minimum case duration in seconds (default: 1800)')
    parser.add_argument('--output', type=str, 
                        default='./data/offline_dataset/vitaldb_offline_data.pkl',
                        help='Output file path')
    parser.add_argument('--cache_dir', type=str,
                        default='./data/vitaldb_cache',
                        help='Cache directory for raw data')
    parser.add_argument('--no_cache', action='store_true',
                        help='Disable caching')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("VitalDB Offline Dataset Preparation")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Number of cases: {args.n_cases}")
    print(f"  BIS range: [{args.bis_min}, {args.bis_max}]")
    print(f"  Min duration: {args.min_duration}s ({args.min_duration/60:.0f} min)")
    print(f"  Output: {args.output}")
    print(f"  Cache directory: {args.cache_dir}")
    print("=" * 70)
    
    # Initialize loader
    loader = VitalDBLoader(
        cache_dir=args.cache_dir,
        bis_range=(args.bis_min, args.bis_max),
        use_cache=not args.no_cache
    )
    
    # Prepare data
    data = loader.prepare_training_data(
        n_cases=args.n_cases,
        bis_range=(args.bis_min, args.bis_max),
        min_duration=args.min_duration,
        save_path=args.output
    )
    
    # Print statistics
    print("\n" + "=" * 70)
    print("Dataset Statistics")
    print("=" * 70)
    
    states = data['states']
    actions = data['actions']
    rewards = data['rewards']
    
    print(f"\nState features (8D):")
    feature_names = [
        'BIS_error', 'Ce_PPF', 'dBIS/dt', 'u_prev',
        'PPF_acc', 'RFTN_acc', 'BIS_slope', 'RFTN_current'
    ]
    for i, name in enumerate(feature_names):
        mean = states[:, i].mean()
        std = states[:, i].std()
        min_val = states[:, i].min()
        max_val = states[:, i].max()
        print(f"  {name:15s}: {mean:8.3f} ± {std:6.3f}  [{min_val:8.3f}, {max_val:8.3f}]")
    
    print(f"\nActions (propofol rate):")
    print(f"  Mean: {actions.mean():.3f} mg/kg/h")
    print(f"  Std:  {actions.std():.3f} mg/kg/h")
    print(f"  Range: [{actions.min():.3f}, {actions.max():.3f}]")
    
    print(f"\nRewards:")
    print(f"  Mean: {rewards.mean():.3f}")
    print(f"  Std:  {rewards.std():.3f}")
    print(f"  Range: [{rewards.min():.3f}, {rewards.max():.3f}]")
    
    # BIS distribution
    bis_values = 50 - states[:, 0]  # Convert BIS error back to BIS
    print(f"\nBIS Distribution:")
    print(f"  Mean: {bis_values.mean():.1f}")
    print(f"  Std:  {bis_values.std():.1f}")
    print(f"  Range: [{bis_values.min():.1f}, {bis_values.max():.1f}]")
    
    # BIS range breakdown
    ranges = [
        (30, 40, "Too deep"),
        (40, 50, "Target lower"),
        (50, 60, "Target upper"),
        (60, 70, "Too light")
    ]
    
    print(f"\nBIS Range Breakdown:")
    for low, high, desc in ranges:
        count = np.sum((bis_values >= low) & (bis_values < high))
        pct = count / len(bis_values) * 100
        print(f"  {low:2d}-{high:2d} ({desc:15s}): {count:6d} ({pct:5.1f}%)")
    
    print("\n" + "=" * 70)
    print("✓ Dataset preparation complete!")
    print(f"  Output file: {args.output}")
    print(f"  Ready for offline RL training")
    print("=" * 70)


if __name__ == "__main__":
    main()
