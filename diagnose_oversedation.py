"""
Diagnose why the agent is over-sedating patients (BIS too low)
"""
import pickle
import numpy as np
from pathlib import Path

print("=" * 80)
print("DIAGNOSING OVER-SEDATION PROBLEM")
print("=" * 80)

# Check if we have the right data
dataset_path = Path('./data/offline_dataset/vitaldb_offline_data_small.pkl')

if not dataset_path.exists():
    print(f"\n❌ Dataset not found: {dataset_path}")
    print("Please regenerate data with fixed loader.")
    exit(1)

with open(dataset_path, 'rb') as f:
    data = pickle.load(f)

states = data['states']
actions = data['actions']

print(f"\n📊 Current Dataset Statistics")
print("=" * 80)

# Check action normalization
print(f"\nActions (should be normalized [0,1]):")
print(f"  Min: {actions.min():.4f}")
print(f"  Max: {actions.max():.4f}")
print(f"  Mean: {actions.mean():.4f}")

# WARNING: Actions > 1 means incorrect normalization!
if actions.max() > 1.0:
    print(f"\n⚠️  WARNING: Actions NOT NORMALIZED!")
    print(f"   Max action: {actions.max():.2f}")
    print(f"   This suggests action_max in data preparation doesn't match actual data")
    print(f"\n   Diagnosis:")
    print(f"   - Data was prepared with OLD action_max (likely 30.0)")
    print(f"   - But code now uses NEW action_max (12.0)")
    print(f"   - Action values like {actions.max():.2f} / 12.0 = {actions.max()/12.0:.2f} (WAY too high!)")
    print(f"\n   ✅ FIX: Regenerate data with new action_max=12.0")

# Check propofol rates in state [4]
ppf_prev = states[:, 4]
print(f"\n[4] Previous Propofol Action (mg/kg/h):")
print(f"  Min: {ppf_prev.min():.2f}")
print(f"  Max: {ppf_prev.max():.2f}")
print(f"  Mean: {ppf_prev.mean():.2f}")
print(f"  90th percentile: {np.percentile(ppf_prev, 90):.2f}")

if ppf_prev.max() > 12.0:
    print(f"\n⚠️  WARNING: Propofol rate exceeds 12 mg/kg/h!")
    print(f"   Max rate: {ppf_prev.max():.2f} mg/kg/h")
    print(f"   This is from OLD data (validation threshold was 30)")
    print(f"\n   ✅ FIX: Regenerate data (new validation threshold is 12)")

# Check accumulation
ppf_acc = states[:, 6]
rftn_acc = states[:, 7]

print(f"\n[6] Propofol Accumulation:")
print(f"  Min: {ppf_acc.min():.4f}")
print(f"  Max: {ppf_acc.max():.4f}")
print(f"  Mean: {ppf_acc.mean():.4f}")

print(f"\n[7] Remifentanil Accumulation:")
print(f"  Min: {rftn_acc.min():.4f}")
print(f"  Max: {rftn_acc.max():.4f}")
print(f"  Mean: {rftn_acc.mean():.4f}")

# Expected max with NEW calculation
expected_ppf_max = 12.0 * (10.0 / 3600.0) * 7  # 0.233 mg/kg
expected_rftn_max = 2.0 * (10.0 / 60.0) * 7    # 2.333 μg/kg

print(f"\n📐 Expected Maximum Values (with NEW calculation):")
print(f"  ppf_acc: {expected_ppf_max:.4f} mg/kg")
print(f"  rftn_acc: {expected_rftn_max:.4f} μg/kg")

if ppf_acc.max() > 2.0:
    print(f"\n⚠️  WARNING: Accumulation values too high!")
    print(f"   ppf_acc max: {ppf_acc.max():.4f} (expected: ≤{expected_ppf_max:.4f})")
    print(f"   This suggests OLD calculation (without timestep multiplication)")
    print(f"\n   ✅ FIX: Regenerate data (accumulation now multiplies by timestep)")

print("\n" + "=" * 80)
print("DIAGNOSIS SUMMARY")
print("=" * 80)

issues_found = []

if actions.max() > 1.0:
    issues_found.append("❌ Actions not normalized (OLD action_max)")
if ppf_prev.max() > 12.0:
    issues_found.append("❌ Propofol rates exceed 12 mg/kg/h (OLD validation)")
if ppf_acc.max() > 2.0:
    issues_found.append("❌ Accumulation values suggest OLD calculation")

if issues_found:
    print(f"\n🔴 ISSUES FOUND ({len(issues_found)}):")
    for issue in issues_found:
        print(f"   {issue}")
    print(f"\n📋 ROOT CAUSE:")
    print(f"   Using OLD dataset created BEFORE the fixes")
    print(f"\n✅ SOLUTION:")
    print(f"   1. Delete old data: rm data/offline_dataset/vitaldb_offline_data_small.pkl")
    print(f"   2. Regenerate: python prepare_vitaldb_quick.py")
    print(f"   3. Verify: python check_data_ranges.py")
    print(f"   4. Retrain: python experiments/compare_quantum_vs_classical_dualdrug.py")
else:
    print(f"\n✅ All checks passed!")
    print(f"   Dataset appears to use NEW calculation methods")
    print(f"\n   If still over-sedating, check:")
    print(f"   - Reward function design")
    print(f"   - Patient simulator parameters")
    print(f"   - Training hyperparameters")

print("\n")
