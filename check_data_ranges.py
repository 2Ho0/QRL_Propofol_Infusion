"""
Check actual data ranges in prepared VitalDB dataset
"""
import pickle
import numpy as np
from pathlib import Path

# Load prepared dataset
dataset_path = Path('./data/offline_dataset/vitaldb_offline_data_small.pkl')

if not dataset_path.exists():
    print(f"Dataset not found: {dataset_path}")
    print("Available files:")
    for f in Path('./data/offline_dataset/').glob('*.pkl'):
        print(f"  - {f}")
    exit(1)

with open(dataset_path, 'rb') as f:
    data = pickle.load(f)

states = data['states']
actions = data['actions']

print("=" * 80)
print("VitalDB Dataset Analysis")
print("=" * 80)

print(f"\nDataset shape:")
print(f"  States: {states.shape}")
print(f"  Actions: {actions.shape}")

print(f"\nState features ({states.shape[1]}D):")
state_names_8d = [
    "[0] BIS error (target - current)",
    "[1] Propofol Ce (μg/mL)", 
    "[2] Remifentanil Ce (ng/mL)",
    "[3] dBIS/dt (rate of change)",
    "[4] Previous propofol action",
    "[5] Previous remifentanil action",
    "[6] Propofol accumulation (last 60s)",
    "[7] Remifentanil accumulation (last 60s)",
]

state_names_13d = [
    "[0] BIS error (target - current)",
    "[1] Propofol Ce (μg/mL)", 
    "[2] Remifentanil Ce (ng/mL)",
    "[3] dBIS/dt (rate of change)",
    "[4] Previous propofol action",
    "[5] Previous remifentanil action",
    "[6] Propofol accumulation (last 60s)",
    "[7] Remifentanil accumulation (last 60s)",
    "[8] BIS slope",
    "[9] Drug interaction",
    "[10] Age (normalized)",
    "[11] Sex (0/1)",
    "[12] BMI (normalized)"
]

state_names = state_names_8d if states.shape[1] == 8 else state_names_13d

for i, name in enumerate(state_names):
    if i >= states.shape[1]:
        break
    values = states[:, i]
    print(f"  {name}")
    print(f"    Min: {values.min():.4f}, Max: {values.max():.4f}, "
          f"Mean: {values.mean():.4f}, Std: {values.std():.4f}")

print(f"\nActions (normalized to [0,1]):")
print(f"  Min: {actions.min():.4f}, Max: {actions.max():.4f}")
print(f"  Mean: {actions.mean():.4f}, Std: {actions.std():.4f}")

print(f"\n" + "=" * 80)
print("CRITICAL: Propofol Accumulation Analysis [6]")
print("=" * 80)

ppf_acc = states[:, 6]
print(f"Propofol accumulation (ppf_acc):")
print(f"  Range: [{ppf_acc.min():.2f}, {ppf_acc.max():.2f}]")
print(f"  Mean: {ppf_acc.mean():.2f}")
print(f"  Std: {ppf_acc.std():.2f}")
print(f"  Percentiles:")
print(f"    50th: {np.percentile(ppf_acc, 50):.2f}")
print(f"    75th: {np.percentile(ppf_acc, 75):.2f}")
print(f"    90th: {np.percentile(ppf_acc, 90):.2f}")
print(f"    95th: {np.percentile(ppf_acc, 95):.2f}")
print(f"    99th: {np.percentile(ppf_acc, 99):.2f}")

print(f"\n" + "=" * 80)
print("CRITICAL: Remifentanil Accumulation Analysis [7]")
print("=" * 80)

rftn_acc = states[:, 7]
print(f"Remifentanil accumulation (rftn_acc):")
print(f"  Range: [{rftn_acc.min():.2f}, {rftn_acc.max():.2f}]")
print(f"  Mean: {rftn_acc.mean():.2f}")
print(f"  Std: {rftn_acc.std():.2f}")
print(f"  Percentiles:")
print(f"    50th: {np.percentile(rftn_acc, 50):.2f}")
print(f"    75th: {np.percentile(rftn_acc, 75):.2f}")
print(f"    90th: {np.percentile(rftn_acc, 90):.2f}")
print(f"    95th: {np.percentile(rftn_acc, 95):.2f}")
print(f"    99th: {np.percentile(rftn_acc, 99):.2f}")

print(f"\n" + "=" * 80)
print("Unit Consistency Check")
print("=" * 80)

print(f"\nExpected units:")
print(f"  ppf_acc [6]: Sum of last 7 timesteps of propofol rate (mg/kg/h)")
print(f"    - Each timestep = 10s")
print(f"    - 7 timesteps = 70s ≈ 1 minute")
print(f"    - If max rate = 12 mg/kg/h → theoretical max = 12 × 7 = 84")
print(f"\n  rftn_acc [7]: Sum of last 7 timesteps of remifentanil rate (μg/kg/min)")
print(f"    - Each timestep = 10s")
print(f"    - 7 timesteps = 70s ≈ 1 minute")
print(f"    - If max rate = 2 μg/kg/min → theoretical max = 2 × 7 = 14")

print(f"\n" + "=" * 80)
print("Validation Results")
print("=" * 80)

ppf_theoretical_max = 12 * 7  # 84
rftn_theoretical_max = 2 * 7  # 14

if ppf_acc.max() <= ppf_theoretical_max:
    print(f"✓ Propofol accumulation OK: {ppf_acc.max():.2f} <= {ppf_theoretical_max}")
else:
    print(f"✗ Propofol accumulation EXCEEDS: {ppf_acc.max():.2f} > {ppf_theoretical_max}")
    print(f"  Exceeded by: {ppf_acc.max() - ppf_theoretical_max:.2f}")

if rftn_acc.max() <= rftn_theoretical_max:
    print(f"✓ Remifentanil accumulation OK: {rftn_acc.max():.2f} <= {rftn_theoretical_max}")
else:
    print(f"✗ Remifentanil accumulation EXCEEDS: {rftn_acc.max():.2f} > {rftn_theoretical_max}")
    print(f"  Exceeded by: {rftn_acc.max() - rftn_theoretical_max:.2f}")

print("\n")
