"""
Quick VitalDB dataset preparation for testing
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

from data.vitaldb_loader import VitalDBLoader

print("=" * 70)
print("Preparing VitalDB Dataset (BIS 30-70)")
print("=" * 70)

loader = VitalDBLoader(
    cache_dir="./data/vitaldb_cache",
    bis_range=(30, 70),
    use_cache=True
)

# Prepare dataset with 20 cases
data = loader.prepare_training_data(
    n_cases=20,
    min_duration=1800,
    save_path="./data/offline_dataset/vitaldb_offline_data_small.pkl"
)

print("\n" + "=" * 70)
print("✓ Dataset prepared!")
print("=" * 70)
print(f"\nYou can now train with:")
print(f"  python experiments/train_offline.py \\")
print(f"    --data_path ./data/offline_dataset/vitaldb_offline_data_small.pkl \\")
print(f"    --n_epochs 50 \\")
print(f"    --batch_size 64")
print("=" * 70)
