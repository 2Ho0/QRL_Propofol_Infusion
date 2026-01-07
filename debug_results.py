"""
실험 결과 상세 디버깅 스크립트
================================

MDAPE 90%의 원인을 파악하기 위한 분석
"""

import sys
sys.path.append('/home/mwilliam/QRL_Propofol_Infusion')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import torch

def analyze_vitaldb_training_data():
    """VitalDB 학습 데이터 분석"""
    print("="*70)
    print("1. VITALDB TRAINING DATA ANALYSIS")
    print("="*70)
    
    data_path = Path('data/offline_dataset/dual_drug_vitaldb_6000cases.npz')
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return
    
    data = np.load(data_path, allow_pickle=True)
    
    states = data['states']
    actions = data['actions']
    rewards = data['rewards']
    
    print(f"\nData shapes:")
    print(f"  States: {states.shape}")
    print(f"  Actions: {actions.shape}")
    print(f"  Rewards: {rewards.shape}")
    
    print(f"\n📊 Action Statistics (VitalDB training data):")
    print(f"\nPropofol (mg/kg/h):")
    print(f"  Mean: {actions[:, 0].mean():.3f}")
    print(f"  Std:  {actions[:, 0].std():.3f}")
    print(f"  Min:  {actions[:, 0].min():.3f}")
    print(f"  25%:  {np.percentile(actions[:, 0], 25):.3f}")
    print(f"  50%:  {np.percentile(actions[:, 0], 50):.3f}")
    print(f"  75%:  {np.percentile(actions[:, 0], 75):.3f}")
    print(f"  Max:  {actions[:, 0].max():.3f}")
    
    print(f"\nRemifentanil (μg/kg/min):")
    print(f"  Mean: {actions[:, 1].mean():.4f}")
    print(f"  Std:  {actions[:, 1].std():.4f}")
    print(f"  Min:  {actions[:, 1].min():.4f}")
    print(f"  25%:  {np.percentile(actions[:, 1], 25):.4f}")
    print(f"  50%:  {np.percentile(actions[:, 1], 50):.4f}")
    print(f"  75%:  {np.percentile(actions[:, 1], 75):.4f}")
    print(f"  Max:  {actions[:, 1].max():.4f}")
    
    # Count very low doses
    low_ppf = np.sum(actions[:, 0] < 2.0)
    low_rftn = np.sum(actions[:, 1] < 0.03)
    
    print(f"\n⚠️  Data Quality Issues:")
    print(f"  Propofol < 2.0 mg/kg/h: {low_ppf} / {len(actions)} ({low_ppf/len(actions)*100:.1f}%)")
    print(f"  Remifentanil < 0.03 μg/kg/min: {low_rftn} / {len(actions)} ({low_rftn/len(actions)*100:.1f}%)")
    
    print(f"\n📊 State Statistics:")
    print(f"  BIS error (state[0]): {states[:, 0].mean():.2f} ± {states[:, 0].std():.2f}")
    print(f"  Ce Propofol (state[1]): {states[:, 1].mean():.2f} ± {states[:, 1].std():.2f}")
    print(f"  Ce Remifentanil (state[2]): {states[:, 2].mean():.3f} ± {states[:, 2].std():.3f}")
    
    # Check for scaling issues
    print(f"\n🔍 Feature Scale Check:")
    for i in range(min(5, states.shape[1])):
        print(f"  State[{i}]: mean={states[:, i].mean():.3f}, std={states[:, i].std():.3f}, "
              f"range=[{states[:, i].min():.3f}, {states[:, i].max():.3f}]")
    
    print(f"\n📊 Reward Statistics:")
    print(f"  Mean: {rewards.mean():.3f}")
    print(f"  Std:  {rewards.std():.3f}")
    print(f"  Min:  {rewards.min():.3f}")
    print(f"  Max:  {rewards.max():.3f}")


def analyze_trained_model():
    """학습된 모델의 출력 분석"""
    print("\n" + "="*70)
    print("2. TRAINED MODEL OUTPUT ANALYSIS")
    print("="*70)
    
    log_dir = Path('logs/comparison_dualdrug_20260104_163311')
    
    # Load quantum agent checkpoint
    quantum_ckpt = log_dir / 'quantum' / 'checkpoints' / 'best_model.pt'
    
    if not quantum_ckpt.exists():
        print(f"❌ Checkpoint not found: {quantum_ckpt}")
        return
    
    try:
        checkpoint = torch.load(quantum_ckpt, map_location='cpu')
        print(f"\n✓ Loaded checkpoint from {quantum_ckpt}")
        print(f"  Keys: {list(checkpoint.keys())}")
        
        if 'episode' in checkpoint:
            print(f"  Trained episodes: {checkpoint['episode']}")
        if 'best_reward' in checkpoint:
            print(f"  Best reward: {checkpoint['best_reward']:.2f}")
            
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")


def analyze_simulator_results():
    """Simulator 테스트 결과 분석"""
    print("\n" + "="*70)
    print("3. SIMULATOR TEST RESULTS ANALYSIS")
    print("="*70)
    
    results_path = Path('logs/comparison_dualdrug_20260104_163311/results.pkl')
    
    if not results_path.exists():
        print(f"❌ Results file not found: {results_path}")
        return
    
    try:
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        
        print(f"\n✓ Loaded results")
        print(f"  Keys: {list(results.keys())}")
        
        for agent_name in ['quantum_simulator', 'classical_simulator']:
            if agent_name not in results:
                continue
                
            agent_results = results[agent_name]
            agent_display = agent_name.split('_')[0].title()
            
            print(f"\n{agent_display} Agent:")
            print(f"  MDAPE: {agent_results['mdape_mean']:.2f} ± {agent_results['mdape_std']:.2f}%")
            print(f"  Reward: {agent_results['reward_mean']:.2f} ± {agent_results['reward_std']:.2f}")
            print(f"  Time in Target: {agent_results['time_in_target_mean']:.1f}%")
            print(f"  Propofol: {agent_results['propofol_usage_mean']:.3f} ± {agent_results['propofol_usage_std']:.3f} mg/kg/h")
            print(f"  Remifentanil: {agent_results['remifentanil_usage_mean']:.4f} ± {agent_results['remifentanil_usage_std']:.4f} μg/kg/min")
            
            # MDAPE distribution
            if 'mdape_list' in agent_results:
                mdapes = agent_results['mdape_list']
                print(f"\n  MDAPE Distribution:")
                print(f"    Min: {np.min(mdapes):.2f}%")
                print(f"    25%: {np.percentile(mdapes, 25):.2f}%")
                print(f"    50%: {np.percentile(mdapes, 50):.2f}%")
                print(f"    75%: {np.percentile(mdapes, 75):.2f}%")
                print(f"    Max: {np.max(mdapes):.2f}%")
                print(f"    All similar: {np.std(mdapes) < 1.0}")
        
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        import traceback
        traceback.print_exc()


def analyze_vitaldb_predictions():
    """VitalDB 예측 결과 분석"""
    print("\n" + "="*70)
    print("4. VITALDB PREDICTION ANALYSIS")
    print("="*70)
    
    vitaldb_path = Path('logs/comparison_dualdrug_20260104_163311/vitaldb_dual_drug.pkl')
    
    if not vitaldb_path.exists():
        print(f"❌ VitalDB results not found: {vitaldb_path}")
        return
    
    try:
        with open(vitaldb_path, 'rb') as f:
            vitaldb_results = pickle.load(f)
        
        print(f"\n✓ Loaded VitalDB test results")
        print(f"  Keys: {list(vitaldb_results.keys())}")
        
        for agent_name in ['quantum', 'classical']:
            if agent_name not in vitaldb_results:
                continue
            
            agent_results = vitaldb_results[agent_name]
            print(f"\n{agent_name.title()} Agent:")
            print(f"  Available keys: {list(agent_results.keys())}")
            
            if 'mdape_propofol' in agent_results:
                print(f"  Propofol MDAPE: {agent_results['mdape_propofol']:.2f}%")
            if 'mdape_remifentanil' in agent_results:
                print(f"  Remifentanil MDAPE: {agent_results['mdape_remifentanil']:.2f}%")
            
            # 예측값과 실제값 비교
            if 'predictions' in agent_results and 'true_values' in agent_results:
                preds = agent_results['predictions']
                trues = agent_results['true_values']
                
                print(f"\n  Predictions shape: {preds.shape}")
                print(f"  True values shape: {trues.shape}")
                
                print(f"\n  Predicted Propofol:")
                print(f"    Mean: {preds[:, 0].mean():.3f} mg/kg/h")
                print(f"    Std:  {preds[:, 0].std():.3f}")
                print(f"    Range: [{preds[:, 0].min():.3f}, {preds[:, 0].max():.3f}]")
                
                print(f"\n  Predicted Remifentanil:")
                print(f"    Mean: {preds[:, 1].mean():.4f} μg/kg/min")
                print(f"    Std:  {preds[:, 1].std():.4f}")
                print(f"    Range: [{preds[:, 1].min():.4f}, {preds[:, 1].max():.4f}]")
                
                print(f"\n  True Propofol:")
                print(f"    Mean: {trues[:, 0].mean():.3f} mg/kg/h")
                print(f"    Range: [{trues[:, 0].min():.3f}, {trues[:, 0].max():.3f}]")
                
                print(f"\n  True Remifentanil:")
                print(f"    Mean: {trues[:, 1].mean():.4f} μg/kg/min")
                print(f"    Range: [{trues[:, 1].min():.4f}, {trues[:, 1].max():.4f}]")
                
                # Error analysis
                ppf_errors = np.abs(preds[:, 0] - trues[:, 0])
                rftn_errors = np.abs(preds[:, 1] - trues[:, 1])
                
                print(f"\n  Absolute Errors:")
                print(f"    Propofol MAE: {ppf_errors.mean():.3f} mg/kg/h")
                print(f"    Remifentanil MAE: {rftn_errors.mean():.4f} μg/kg/min")
                
    except Exception as e:
        print(f"❌ Error loading VitalDB results: {e}")
        import traceback
        traceback.print_exc()


def create_diagnostic_plots():
    """진단 플롯 생성"""
    print("\n" + "="*70)
    print("5. CREATING DIAGNOSTIC PLOTS")
    print("="*70)
    
    data_path = Path('data/offline_dataset/dual_drug_vitaldb_6000cases.npz')
    
    if not data_path.exists():
        print(f"❌ Data file not found")
        return
    
    data = np.load(data_path, allow_pickle=True)
    actions = data['actions']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('VitalDB Training Data Analysis', fontsize=14, fontweight='bold')
    
    # 1. Propofol distribution
    ax = axes[0, 0]
    ax.hist(actions[:, 0], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(actions[:, 0].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {actions[:, 0].mean():.2f}')
    ax.axvline(2.0, color='orange', linestyle='--', linewidth=2, label='Clinical Min: 2.0')
    ax.set_xlabel('Propofol Rate (mg/kg/h)')
    ax.set_ylabel('Frequency')
    ax.set_title('Propofol Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Remifentanil distribution
    ax = axes[0, 1]
    ax.hist(actions[:, 1], bins=50, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(actions[:, 1].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {actions[:, 1].mean():.3f}')
    ax.axvline(0.03, color='orange', linestyle='--', linewidth=2, label='Clinical Min: 0.03')
    ax.set_xlabel('Remifentanil Rate (μg/kg/min)')
    ax.set_ylabel('Frequency')
    ax.set_title('Remifentanil Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Joint distribution
    ax = axes[1, 0]
    scatter = ax.scatter(actions[:, 0], actions[:, 1], alpha=0.3, s=1)
    ax.axvline(2.0, color='red', linestyle='--', alpha=0.5, label='Clinical Min PPF')
    ax.axhline(0.03, color='red', linestyle='--', alpha=0.5, label='Clinical Min RFTN')
    ax.set_xlabel('Propofol (mg/kg/h)')
    ax.set_ylabel('Remifentanil (μg/kg/min)')
    ax.set_title('Drug Combination Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. Low dose analysis
    ax = axes[1, 1]
    low_ppf_pct = (actions[:, 0] < 2.0).sum() / len(actions) * 100
    low_rftn_pct = (actions[:, 1] < 0.03).sum() / len(actions) * 100
    both_low_pct = ((actions[:, 0] < 2.0) & (actions[:, 1] < 0.03)).sum() / len(actions) * 100
    
    categories = ['PPF < 2.0', 'RFTN < 0.03', 'Both Low']
    percentages = [low_ppf_pct, low_rftn_pct, both_low_pct]
    colors = ['blue', 'green', 'red']
    
    bars = ax.bar(categories, percentages, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Percentage of Data (%)')
    ax.set_title('Low Dose Samples')
    ax.set_ylim(0, 100)
    
    # Add percentage labels on bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path('logs/vitaldb_data_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Diagnostic plot saved to: {output_path}")
    plt.close()


def main():
    """메인 분석 함수"""
    print("\n" + "="*70)
    print("COMPREHENSIVE DIAGNOSTIC ANALYSIS")
    print("Why is MDAPE ~90%?")
    print("="*70)
    
    analyze_vitaldb_training_data()
    analyze_trained_model()
    analyze_simulator_results()
    analyze_vitaldb_predictions()
    create_diagnostic_plots()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\n📄 Detailed report saved to: diagnostic_report.md")
    print("📊 Diagnostic plots saved to: logs/vitaldb_data_analysis.png")
    print("\n💡 Key Findings:")
    print("   1. Check if training data has too many low-dose samples")
    print("   2. Verify model output ranges match clinical requirements")
    print("   3. Confirm MDAPE measures action prediction, not BIS control")
    print("   4. Evaluate using BIS tracking error instead of action MDAPE")
    print("\n")


if __name__ == '__main__':
    main()
