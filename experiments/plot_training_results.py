"""
Training Results Plotter
==========================
Stage 1 (Offline Pre-training) 의 Loss 곡선과
Stage 2 (Online Fine-tuning) 의 누적 Reward 곡선을 플로팅합니다.

Usage:
    # 최신 로그 디렉터리를 자동으로 찾아 플로팅
    python experiments/plot_training_results.py

    # 특정 로그 디렉터리 지정
    python experiments/plot_training_results.py --log_dir logs/comparison_20260212_154505

    # 저장 경로 지정
    python experiments/plot_training_results.py --log_dir logs/comparison_20260212_154505 --save_dir plots/
"""

import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from pathlib import Path


# ─────────────────────────────────────────────
# 전역 스타일 설정
# ─────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

COLORS = {
    'classical': '#4C72B0',   # 파란계
    'quantum':   '#DD8452',   # 주황계
    'val':       '#55A868',   # 녹색
    'bc':        '#C44E52',   # 빨간계
    'rl':        '#8172B2',   # 보라
    'warmup':    '#DEA82A',   # 황색
    'explore':   '#2AA1A4',   # 청록
}


# ─────────────────────────────────────────────
# 헬퍼 함수
# ─────────────────────────────────────────────

def find_latest_log_dir(base: str = 'logs') -> Path | None:
    """logs/ 아래에서 comparison_ 로 시작하는 가장 최신 디렉터리를 반환."""
    base_path = Path(base)
    if not base_path.exists():
        return None
    candidates = sorted(base_path.glob('comparison_*'), reverse=True)
    return candidates[0] if candidates else None


def smooth(values: np.ndarray, window: int = 10) -> np.ndarray:
    """단순 이동 평균 스무딩."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')


def load_loss_csv(path: Path) -> pd.DataFrame | None:
    """Stage1 loss_history.csv 로드."""
    if not path.exists():
        print(f"  [경고] 파일 없음: {path}")
        return None
    return pd.read_csv(path)


def load_episode_csv(path: Path) -> pd.DataFrame | None:
    """Stage2 episode_history.csv 로드."""
    if not path.exists():
        print(f"  [경고] 파일 없음: {path}")
        return None
    return pd.read_csv(path)


# ─────────────────────────────────────────────
# Figure 1: Stage 1 – Loss 곡선 (Classical vs Quantum)
# ─────────────────────────────────────────────

def plot_stage1_loss(
    classical_loss: pd.DataFrame | None,
    quantum_loss: pd.DataFrame | None,
    save_path: Path
):
    """
    Train Loss / Val Loss / BC Loss / Val MDAPE 를
    Classical / Quantum 나란히 4-panel 비교로 그립니다.
    """
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Stage 1 – Offline Pre-training Loss Curves', fontsize=15, y=0.98)

    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

    panel_cfg = [
        ('train_loss', 'Train Loss (Total)',     'MSE Loss'),
        ('val_loss',   'Validation Loss (MSE)',  'MSE Loss'),
        ('bc_loss',    'Behavioral Cloning Loss','MSE Loss'),
        ('val_mdape',  'Validation MDAPE',       'MDAPE (%)'),
    ]

    for ax, (col, title, ylabel) in zip(axes, panel_cfg):
        for label, df, color in [
            ('Classical', classical_loss, COLORS['classical']),
            ('Quantum',   quantum_loss,   COLORS['quantum']),
        ]:
            if df is None or col not in df.columns:
                continue
            epochs = df['epoch'].values
            vals   = df[col].values

            ax.plot(epochs, vals, color=color, alpha=0.25, linewidth=1)
            if len(vals) >= 3:
                sm = smooth(vals, window=3)
                ax.plot(
                    epochs[len(epochs) - len(sm):],
                    sm,
                    color=color, linewidth=2.2, label=label
                )
            else:
                ax.plot(epochs, vals, color=color, linewidth=2.2, label=label)

        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper right')

    plt.savefig(save_path, bbox_inches='tight')
    print(f"  ✓ Stage1 Loss 플롯 저장 → {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# Figure 2: Stage 1 – Train & Val Loss 상세 (per agent)
# ─────────────────────────────────────────────

def plot_stage1_detailed(
    df: pd.DataFrame | None,
    agent_name: str,
    save_path: Path,
    color_train: str,
    color_val: str,
):
    """
    단일 에이전트의 Train / Val / BC / RL Loss 를
    2×2 패널로 상세 플로팅합니다.
    """
    if df is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f'{agent_name} – Stage 1 Detailed Loss', fontsize=14)

    pairs = [
        (axes[0, 0], 'train_loss', 'val_loss', 'Train vs Val Loss',         'MSE Loss'),
        (axes[0, 1], 'bc_loss',    None,        'Behavioral Cloning Loss',   'MSE Loss'),
        (axes[1, 0], 'rl_loss',    None,        'RL Actor Loss (scaled)',     'RL Loss (scaled)'),
        (axes[1, 1], 'val_mdape',  None,        'Validation MDAPE',          'MDAPE (%)'),
    ]

    epochs = df['epoch'].values

    for ax, col1, col2, title, ylabel in pairs:
        # col1
        if col1 in df.columns:
            v1 = df[col1].values
            ax.plot(epochs, v1, color=color_train, alpha=0.2, linewidth=1)
            sm1 = smooth(v1, 5)
            ax.plot(
                epochs[len(epochs) - len(sm1):], sm1,
                color=color_train, linewidth=2, label=col1.replace('_', ' ').title()
            )
        # col2 (optional – val_loss)
        if col2 and col2 in df.columns:
            v2 = df[col2].values
            ax.plot(epochs, v2, color=color_val, alpha=0.2, linewidth=1)
            sm2 = smooth(v2, 5)
            ax.plot(
                epochs[len(epochs) - len(sm2):], sm2,
                color=color_val, linewidth=2, label=col2.replace('_', ' ').title(),
                linestyle='--'
            )

        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"  ✓ {agent_name} 상세 Loss 플롯 저장 → {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# Figure 3: Stage 2 – Episode Reward (Training)
# ─────────────────────────────────────────────

def plot_stage2_rewards(
    classical_ep: pd.DataFrame | None,
    quantum_ep: pd.DataFrame | None,
    save_path: Path,
    smooth_window: int = 20
):
    """
    Stage 2 에피소드별 보상(raw + 이동평균) 및 누적 보상을 플로팅합니다.
    warmup / exploration 위상을 배경색으로 구분합니다.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=False)
    fig.suptitle('Stage 2 – Online Fine-tuning: Episode Reward', fontsize=14)

    # 공통 플롯 함수
    def _draw(ax_raw, ax_cum, df, label, color):
        if df is None:
            return
        eps     = df['episode'].values
        rewards = df['reward'].values

        # warmup 구간 배경 표시
        if 'phase' in df.columns:
            warmup_end = df[df['phase'] == 'warmup']['episode'].max()
            if not np.isnan(warmup_end):
                ax_raw.axvspan(eps[0], warmup_end, alpha=0.07,
                               color=COLORS['warmup'], label='Warmup Phase')
                ax_cum.axvspan(eps[0], warmup_end, alpha=0.07, color=COLORS['warmup'])

        # Raw reward
        ax_raw.plot(eps, rewards, color=color, alpha=0.2, linewidth=0.8)
        sm = smooth(rewards, smooth_window)
        ax_raw.plot(
            eps[len(eps) - len(sm):], sm,
            color=color, linewidth=2.2, label=f'{label} (smoothed)'
        )

        # 누적 보상
        cumsum = np.cumsum(rewards)
        ax_cum.plot(eps, cumsum, color=color, linewidth=2.2, label=label)

    # Classical
    _draw(axes[0], axes[2], classical_ep, 'Classical', COLORS['classical'])
    # Quantum
    _draw(axes[1], axes[2], quantum_ep,   'Quantum',   COLORS['quantum'])

    # 제목 & 레이블
    for i, (ax, agent) in enumerate(zip(axes[:2], ['Classical', 'Quantum'])):
        ax.set_title(f'{agent} – Per-episode Reward')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
        ax.legend(loc='upper left', fontsize=9)

    axes[2].set_title('Cumulative Reward (Classical vs Quantum)')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Cumulative Reward')
    axes[2].axhline(0, color='grey', linewidth=0.8, linestyle='--')
    axes[2].legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"  ✓ Stage2 Reward 플롯 저장 → {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# Figure 4: Stage 2 – MDAPE & Time-in-Target
# ─────────────────────────────────────────────

def plot_stage2_metrics(
    classical_ep: pd.DataFrame | None,
    quantum_ep: pd.DataFrame | None,
    save_path: Path,
    smooth_window: int = 20
):
    """MDAPE와 Time-in-Target 의 에피소드별 추이를 비교합니다."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    fig.suptitle('Stage 2 – Online Fine-tuning: Clinical Metrics', fontsize=14)

    metrics_cfg = [
        ('mdape',          'MDAPE (%)',               axes[0]),
        ('time_in_target', 'Time in Target BIS (%)',  axes[1]),
    ]

    for col, ylabel, ax in metrics_cfg:
        for label, df, color in [
            ('Classical', classical_ep, COLORS['classical']),
            ('Quantum',   quantum_ep,   COLORS['quantum']),
        ]:
            if df is None or col not in df.columns:
                continue
            eps  = df['episode'].values
            vals = df[col].values

            # warmup 구간
            if 'phase' in df.columns:
                warmup_end = df[df['phase'] == 'warmup']['episode'].max()
                if not np.isnan(warmup_end):
                    ax.axvspan(eps[0], warmup_end, alpha=0.07, color=COLORS['warmup'])

            ax.plot(eps, vals, color=color, alpha=0.18, linewidth=0.8)
            sm = smooth(vals, smooth_window)
            ax.plot(
                eps[len(eps) - len(sm):], sm,
                color=color, linewidth=2, label=label
            )

        ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + ' per Episode')
        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"  ✓ Stage2 Metrics 플롯 저장 → {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# Figure 5: 통합 대시보드 (Summary 1-page)
# ─────────────────────────────────────────────

def plot_dashboard(
    c_loss: pd.DataFrame | None,
    q_loss: pd.DataFrame | None,
    c_ep: pd.DataFrame | None,
    q_ep: pd.DataFrame | None,
    save_path: Path,
    smooth_window: int = 20
):
    """
    Stage1 Loss + Stage2 Reward + MDAPE 를 한 페이지 대시보드로 요약합니다.
    """
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('QRL Propofol Infusion – Training Summary Dashboard',
                 fontsize=15, y=0.99)

    gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35)
    ax_loss   = fig.add_subplot(gs[0, 0])   # Stage1 Val Loss
    ax_mdape1 = fig.add_subplot(gs[0, 1])   # Stage1 Val MDAPE
    ax_bc     = fig.add_subplot(gs[0, 2])   # Stage1 BC Loss
    ax_reward = fig.add_subplot(gs[1, 0:2]) # Stage2 Reward (wide)
    ax_mdape2 = fig.add_subplot(gs[1, 2])   # Stage2 MDAPE

    # --- Stage1 panels ---
    for label, df, color in [
        ('Classical', c_loss, COLORS['classical']),
        ('Quantum',   q_loss, COLORS['quantum']),
    ]:
        if df is None:
            continue
        epochs = df['epoch'].values

        for ax, col, title, ylabel in [
            (ax_loss,   'val_loss',   'Stage1 – Val Loss',      'MSE Loss'),
            (ax_mdape1, 'val_mdape',  'Stage1 – Val MDAPE',     'MDAPE (%)'),
            (ax_bc,     'bc_loss',    'Stage1 – BC Loss',       'MSE Loss'),
        ]:
            if col not in df.columns:
                continue
            vals = df[col].values
            ax.plot(epochs, vals, color=color, alpha=0.15, linewidth=0.8)
            sm = smooth(vals, 3)
            ax.plot(epochs[len(epochs) - len(sm):], sm,
                    color=color, linewidth=2, label=label)
            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)

    # --- Stage2 panels ---
    for label, df, color in [
        ('Classical', c_ep, COLORS['classical']),
        ('Quantum',   q_ep, COLORS['quantum']),
    ]:
        if df is None:
            continue
        eps     = df['episode'].values
        rewards = df['reward'].values

        # Warmup 구간
        if 'phase' in df.columns:
            warmup_end = df[df['phase'] == 'warmup']['episode'].max()
            if not np.isnan(warmup_end):
                ax_reward.axvspan(eps[0], warmup_end, alpha=0.07, color=COLORS['warmup'])
                ax_mdape2.axvspan(eps[0], warmup_end, alpha=0.07, color=COLORS['warmup'])

        # Reward (smoothed)
        sm_r = smooth(rewards, smooth_window)
        ax_reward.plot(eps, rewards, color=color, alpha=0.1, linewidth=0.6)
        ax_reward.plot(eps[len(eps) - len(sm_r):], sm_r,
                       color=color, linewidth=2, label=label)

        # MDAPE
        if 'mdape' in df.columns:
            mdape = df['mdape'].values
            sm_m  = smooth(mdape, smooth_window)
            ax_mdape2.plot(eps, mdape, color=color, alpha=0.1, linewidth=0.6)
            ax_mdape2.plot(eps[len(eps) - len(sm_m):], sm_m,
                           color=color, linewidth=2, label=label)

    ax_reward.set_title('Stage2 – Episode Reward (smoothed)')
    ax_reward.set_xlabel('Episode')
    ax_reward.set_ylabel('Total Reward')
    ax_reward.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    ax_reward.legend()

    ax_mdape2.set_title('Stage2 – MDAPE')
    ax_mdape2.set_xlabel('Episode')
    ax_mdape2.set_ylabel('MDAPE (%)')
    ax_mdape2.legend()

    plt.savefig(save_path, bbox_inches='tight')
    print(f"  ✓ 대시보드 저장 → {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Training Results Plotter')
    p.add_argument('--log_dir',  type=str, default=None,
                   help='comparison_YYYYMMDD_HHMMSS 디렉터리 경로 '
                        '(미지정 시 최신 자동 탐색)')
    p.add_argument('--save_dir', type=str, default=None,
                   help='플롯 저장 디렉터리 (기본: log_dir/plots/)')
    p.add_argument('--smooth',   type=int, default=20,
                   help='이동 평균 윈도우 크기 (기본: 20)')
    return p.parse_args()


def main():
    args = parse_args()

    # ─── 로그 디렉터리 결정 ───
    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        log_dir = find_latest_log_dir('logs')
        if log_dir is None:
            print('[오류] logs/ 하위에 comparison_ 디렉터리를 찾을 수 없습니다.')
            print('       --log_dir 로 직접 경로를 지정해 주세요.')
            return
        print(f'[정보] 최신 로그 디렉터리 자동 탐지: {log_dir}')

    if not log_dir.exists():
        print(f'[오류] 디렉터리가 없습니다: {log_dir}')
        return

    # ─── 저장 디렉터리 ───
    save_dir = Path(args.save_dir) if args.save_dir else log_dir / 'plots'
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f'[정보] 결과 저장 경로: {save_dir}\n')

    # ─── CSV 로드 ───
    print('=== CSV 파일 로드 중 ===')
    c_stage1 = log_dir / 'classical' / 'stage1_offline' / 'loss_history.csv'
    q_stage1 = log_dir / 'quantum'   / 'stage1_offline' / 'loss_history.csv'
    c_stage2 = log_dir / 'classical' / 'stage2_online'  / 'episode_history.csv'
    q_stage2 = log_dir / 'quantum'   / 'stage2_online'  / 'episode_history.csv'

    c_loss = load_loss_csv(c_stage1)
    q_loss = load_loss_csv(q_stage1)
    c_ep   = load_episode_csv(c_stage2)
    q_ep   = load_episode_csv(q_stage2)

    # ─── 플롯 생성 ───
    print('\n=== 플롯 생성 중 ===')

    # 1) Stage1 Loss 비교
    plot_stage1_loss(
        c_loss, q_loss,
        save_dir / 'stage1_loss_comparison.png'
    )

    # 2) 각 에이전트 상세 Loss
    plot_stage1_detailed(
        c_loss, 'Classical', save_dir / 'stage1_classical_detailed.png',
        color_train=COLORS['classical'], color_val=COLORS['val']
    )
    plot_stage1_detailed(
        q_loss, 'Quantum', save_dir / 'stage1_quantum_detailed.png',
        color_train=COLORS['quantum'], color_val=COLORS['val']
    )

    # 3) Stage2 에피소드 Reward
    plot_stage2_rewards(
        c_ep, q_ep,
        save_dir / 'stage2_reward.png',
        smooth_window=args.smooth
    )

    # 4) Stage2 MDAPE / Time-in-Target
    plot_stage2_metrics(
        c_ep, q_ep,
        save_dir / 'stage2_clinical_metrics.png',
        smooth_window=args.smooth
    )

    # 5) 통합 대시보드
    plot_dashboard(
        c_loss, q_loss, c_ep, q_ep,
        save_dir / 'dashboard.png',
        smooth_window=args.smooth
    )

    print(f'\n✅ 모든 플롯 완료 → {save_dir}')


if __name__ == '__main__':
    main()
