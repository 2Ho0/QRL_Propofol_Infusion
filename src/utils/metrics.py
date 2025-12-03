"""
Performance Metrics for Propofol Infusion Control
===================================================

This module implements the standard clinical performance metrics
for evaluating anesthesia control systems, following the CBIM paper.

CBIM Paper Formulations (50)-(52):
-----------------------------------

Performance Error (PE) - Base metric:
    PE_t = (BIS_t - BIS_target) / BIS_target × 100 [%]

(50) MDPE (Median Performance Error): Measure of bias
    MDPE = Median(PE)
    - Positive: BIS tends to be higher than target (underdosing)
    - Negative: BIS tends to be lower than target (overdosing)

(51) MDAPE (Median Absolute Performance Error): Measure of accuracy
    MDAPE = Median(|PE|)
    - Lower is better
    - Clinical acceptability: MDAPE < 20%

(52) Wobble: Measure of intra-individual variability
    Wobble = Median(|PE - MDPE|)
    - Lower indicates more stable control
    - Computed as median absolute deviation from MDPE

Additional Metrics:
- Divergence: Measure of trend in PE over time
- Time in Target Range: Percentage of time BIS is in target window
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """
    Container for anesthesia control performance metrics.
    
    Following CBIM paper formulations (50)-(52).
    
    Attributes:
        mdpe: Median Performance Error (%) - Formulation (50)
        mdape: Median Absolute Performance Error (%) - Formulation (51)
        wobble: Intra-individual variability (%) - Formulation (52)
        divergence: Trend in PE (%.min^-1)
        time_in_target: Percentage of time in target range (%)
        mean_dose: Mean propofol infusion rate (μg/kg/min)
        total_dose: Total propofol administered (mg/kg)
        settling_time: Time to first reach target range (seconds)
    """
    mdpe: float
    mdape: float
    wobble: float
    divergence: float
    time_in_target: float
    mean_dose: float
    total_dose: float
    settling_time: Optional[float]


def calculate_performance_error(
    bis_values: np.ndarray,
    bis_target: float = 50.0
) -> np.ndarray:
    """
    Calculate Performance Error (PE) for each time point.
    
    Base formula for metrics (50)-(52):
        PE_t = (BIS_t - BIS_target) / BIS_target × 100
    
    Args:
        bis_values: Array of BIS measurements
        bis_target: Target BIS value (g in paper)
    
    Returns:
        Array of PE values in percentage
    """
    return (bis_values - bis_target) / bis_target * 100


def calculate_mdpe(
    bis_values: np.ndarray,
    bis_target: float = 50.0
) -> float:
    """
    Calculate Median Performance Error (MDPE) - Formulation (50).
    
    MDPE = Median(PE)
    
    MDPE measures the bias of the control system.
    - Positive MDPE: System tends to underdose (BIS > target)
    - Negative MDPE: System tends to overdose (BIS < target)
    
    Args:
        bis_values: Array of BIS measurements
        bis_target: Target BIS value (g in paper)
    
    Returns:
        MDPE in percentage
    """
    # PE_t = (BIS_t - g) / g × 100  (base formula)
    pe = calculate_performance_error(bis_values, bis_target)
    # MDPE = Median(PE)  (50)
    return float(np.median(pe))


def calculate_mdape(
    bis_values: np.ndarray,
    bis_target: float = 50.0
) -> float:
    """
    Calculate Median Absolute Performance Error (MDAPE) - Formulation (51).
    
    MDAPE = Median(|PE|)
    
    MDAPE measures the accuracy/precision of the control system.
    Lower values indicate better control.
    Clinical acceptability threshold: MDAPE < 20%
    
    Args:
        bis_values: Array of BIS measurements
        bis_target: Target BIS value (g in paper)
    
    Returns:
        MDAPE in percentage
    """
    # PE_t = (BIS_t - g) / g × 100  (base formula)
    pe = calculate_performance_error(bis_values, bis_target)
    # MDAPE = Median(|PE|)  (51)
    return float(np.median(np.abs(pe)))


def calculate_wobble(
    bis_values: np.ndarray,
    bis_target: float = 50.0
) -> float:
    """
    Calculate Wobble (intra-individual variability) - Formulation (52).
    
    Wobble = Median(|PE - MDPE|)
    
    Wobble measures the variability of control around the median.
    It's the median absolute deviation of PE from MDPE.
    Lower values indicate more stable control.
    
    Args:
        bis_values: Array of BIS measurements
        bis_target: Target BIS value (g in paper)
    
    Returns:
        Wobble in percentage
    """
    # PE_t = (BIS_t - g) / g × 100  (base formula)
    pe = calculate_performance_error(bis_values, bis_target)
    # MDPE = Median(PE)  (50)
    mdpe = np.median(pe)
    # Wobble = Median(|PE - MDPE|)  (52)
    return float(np.median(np.abs(pe - mdpe)))


def calculate_divergence(
    bis_values: np.ndarray,
    time_values: np.ndarray,
    bis_target: float = 50.0
) -> float:
    """
    Calculate Divergence (trend in PE over time).
    
    Divergence measures whether the control is improving or worsening.
    Computed as the slope of linear regression of |PE| vs time.
    - Positive: Control worsening
    - Negative: Control improving
    - Near zero: Stable control
    
    Args:
        bis_values: Array of BIS measurements
        time_values: Array of time points (in minutes)
        bis_target: Target BIS value
    
    Returns:
        Divergence in %.min^-1
    """
    pe = calculate_performance_error(bis_values, bis_target)
    abs_pe = np.abs(pe)
    
    # Convert time to minutes if in seconds
    if time_values.max() > 200:  # Likely in seconds
        time_min = time_values / 60.0
    else:
        time_min = time_values
    
    # Linear regression: |PE| = a + b * time
    # Divergence = b (slope)
    n = len(time_min)
    if n < 2:
        return 0.0
    
    mean_t = np.mean(time_min)
    mean_pe = np.mean(abs_pe)
    
    numerator = np.sum((time_min - mean_t) * (abs_pe - mean_pe))
    denominator = np.sum((time_min - mean_t) ** 2)
    
    if denominator == 0:
        return 0.0
    
    divergence = numerator / denominator
    return float(divergence)


def calculate_time_in_target(
    bis_values: np.ndarray,
    target_low: float = 40.0,
    target_high: float = 60.0
) -> float:
    """
    Calculate percentage of time BIS is in target range.
    
    Args:
        bis_values: Array of BIS measurements
        target_low: Lower bound of target range
        target_high: Upper bound of target range
    
    Returns:
        Percentage of time in target range
    """
    in_target = np.logical_and(
        bis_values >= target_low,
        bis_values <= target_high
    )
    return float(np.mean(in_target) * 100)


def calculate_settling_time(
    bis_values: np.ndarray,
    time_values: np.ndarray,
    target_low: float = 45.0,
    target_high: float = 55.0,
    sustained_steps: int = 6
) -> Optional[float]:
    """
    Calculate settling time (time to first reach and stay in target).
    
    Settling time is defined as the time when BIS first enters
    the target range and stays there for a minimum number of steps.
    
    Args:
        bis_values: Array of BIS measurements
        time_values: Array of time points (in seconds)
        target_low: Lower bound of target range
        target_high: Upper bound of target range
        sustained_steps: Minimum steps to sustain in target
    
    Returns:
        Settling time in seconds, or None if never settled
    """
    n = len(bis_values)
    if n < sustained_steps:
        return None
    
    in_target = np.logical_and(
        bis_values >= target_low,
        bis_values <= target_high
    )
    
    # Find first sustained period in target
    for i in range(n - sustained_steps + 1):
        if np.all(in_target[i:i + sustained_steps]):
            return float(time_values[i])
    
    return None


def calculate_total_dose(
    dose_values: np.ndarray,
    time_values: np.ndarray,
    weight: float = 70.0
) -> float:
    """
    Calculate total propofol dose administered.
    
    Args:
        dose_values: Array of infusion rates (μg/kg/min)
        time_values: Array of time points (in seconds)
        weight: Patient weight in kg
    
    Returns:
        Total dose in mg/kg
    """
    if len(dose_values) < 2:
        return 0.0
    
    # Calculate dt for each interval
    dt = np.diff(time_values)  # in seconds
    
    # Convert to minutes and calculate total
    dt_min = dt / 60.0
    
    # Trapezoidal integration
    avg_rates = (dose_values[:-1] + dose_values[1:]) / 2
    total_ug_per_kg = np.sum(avg_rates * dt_min)
    
    # Convert μg/kg to mg/kg
    total_mg_per_kg = total_ug_per_kg / 1000.0
    
    return float(total_mg_per_kg)


def calculate_all_metrics(
    bis_values: np.ndarray,
    dose_values: np.ndarray,
    time_values: np.ndarray,
    bis_target: float = 50.0,
    target_range: Tuple[float, float] = (40.0, 60.0),
    patient_weight: float = 70.0
) -> PerformanceMetrics:
    """
    Calculate all performance metrics.
    
    Args:
        bis_values: Array of BIS measurements
        dose_values: Array of infusion rates (μg/kg/min)
        time_values: Array of time points (in seconds)
        bis_target: Target BIS value
        target_range: (low, high) bounds for target range
        patient_weight: Patient weight in kg
    
    Returns:
        PerformanceMetrics dataclass with all metrics
    """
    return PerformanceMetrics(
        mdpe=calculate_mdpe(bis_values, bis_target),
        mdape=calculate_mdape(bis_values, bis_target),
        wobble=calculate_wobble(bis_values, bis_target),
        divergence=calculate_divergence(bis_values, time_values, bis_target),
        time_in_target=calculate_time_in_target(
            bis_values, target_range[0], target_range[1]
        ),
        mean_dose=float(np.mean(dose_values)),
        total_dose=calculate_total_dose(dose_values, time_values, patient_weight),
        settling_time=calculate_settling_time(
            bis_values, time_values,
            target_low=(bis_target - 5),
            target_high=(bis_target + 5)
        )
    )


def format_metrics_report(metrics: PerformanceMetrics) -> str:
    """
    Format metrics as a readable report.
    
    Args:
        metrics: PerformanceMetrics object
    
    Returns:
        Formatted string report
    """
    lines = [
        "=" * 50,
        "ANESTHESIA CONTROL PERFORMANCE REPORT",
        "=" * 50,
        "",
        "Performance Metrics:",
        f"  MDPE:           {metrics.mdpe:+.2f}%",
        f"  MDAPE:          {metrics.mdape:.2f}%",
        f"  Wobble:         {metrics.wobble:.2f}%",
        f"  Divergence:     {metrics.divergence:+.4f}%/min",
        "",
        "Control Quality:",
        f"  Time in Target: {metrics.time_in_target:.1f}%",
        f"  Settling Time:  {f'{metrics.settling_time:.0f}s' if metrics.settling_time else 'N/A'}",
        "",
        "Drug Usage:",
        f"  Mean Dose:      {metrics.mean_dose:.1f} μg/kg/min",
        f"  Total Dose:     {metrics.total_dose:.2f} mg/kg",
        "",
        "Clinical Assessment:",
    ]
    
    # Add clinical assessment
    if metrics.mdape < 10:
        lines.append("  ✓ Excellent accuracy (MDAPE < 10%)")
    elif metrics.mdape < 20:
        lines.append("  ✓ Good accuracy (MDAPE < 20%)")
    elif metrics.mdape < 30:
        lines.append("  ⚠ Acceptable accuracy (MDAPE < 30%)")
    else:
        lines.append("  ✗ Poor accuracy (MDAPE ≥ 30%)")
    
    if abs(metrics.mdpe) < 10:
        lines.append("  ✓ Low bias (|MDPE| < 10%)")
    else:
        bias_dir = "overdosing" if metrics.mdpe < 0 else "underdosing"
        lines.append(f"  ⚠ Bias towards {bias_dir} (MDPE = {metrics.mdpe:+.1f}%)")
    
    if metrics.time_in_target >= 80:
        lines.append("  ✓ Excellent target maintenance (≥80%)")
    elif metrics.time_in_target >= 60:
        lines.append("  ✓ Good target maintenance (≥60%)")
    else:
        lines.append("  ⚠ Poor target maintenance (<60%)")
    
    lines.append("=" * 50)
    
    return "\n".join(lines)


class MetricsTracker:
    """
    Track metrics across multiple episodes for training analysis.
    
    Attributes:
        episode_metrics: List of metrics for each episode
        window_size: Window for moving average calculation
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize metrics tracker.
        
        Args:
            window_size: Window size for moving average
        """
        self.window_size = window_size
        self.episode_metrics: List[PerformanceMetrics] = []
        
        # Running statistics
        self.rewards: List[float] = []
        self.mdpe_history: List[float] = []
        self.mdape_history: List[float] = []
        self.time_in_target_history: List[float] = []
    
    def add_episode(
        self,
        metrics: PerformanceMetrics,
        total_reward: float
    ):
        """
        Add episode metrics.
        
        Args:
            metrics: Episode performance metrics
            total_reward: Total episode reward
        """
        self.episode_metrics.append(metrics)
        self.rewards.append(total_reward)
        self.mdpe_history.append(metrics.mdpe)
        self.mdape_history.append(metrics.mdape)
        self.time_in_target_history.append(metrics.time_in_target)
    
    def get_moving_average(self, values: List[float]) -> float:
        """Calculate moving average of recent values."""
        if len(values) == 0:
            return 0.0
        window = values[-self.window_size:]
        return float(np.mean(window))
    
    def get_summary(self) -> Dict[str, float]:
        """
        Get summary statistics.
        
        Returns:
            Dictionary of summary statistics
        """
        if len(self.episode_metrics) == 0:
            return {}
        
        return {
            'n_episodes': len(self.episode_metrics),
            'avg_reward': self.get_moving_average(self.rewards),
            'avg_mdpe': self.get_moving_average(self.mdpe_history),
            'avg_mdape': self.get_moving_average(self.mdape_history),
            'avg_time_in_target': self.get_moving_average(self.time_in_target_history),
            'best_mdape': min(self.mdape_history),
            'best_time_in_target': max(self.time_in_target_history)
        }


if __name__ == "__main__":
    # Test metrics calculation
    print("Testing Performance Metrics...")
    
    # Generate synthetic data
    np.random.seed(42)
    n_steps = 720  # 1 hour at 5s intervals
    
    # Simulate BIS starting high and settling to target
    time = np.arange(n_steps) * 5.0  # seconds
    
    # Simulated BIS: starts at 97, drops to ~50 with some noise
    bis = 97 - 47 * (1 - np.exp(-time / 300)) + np.random.randn(n_steps) * 3
    bis = np.clip(bis, 0, 100)
    
    # Simulated dose: ramps up then stabilizes
    dose = 150 * (1 - np.exp(-time / 100)) + np.random.randn(n_steps) * 10
    dose = np.clip(dose, 0, 200)
    
    # Calculate all metrics
    metrics = calculate_all_metrics(
        bis_values=bis,
        dose_values=dose,
        time_values=time,
        bis_target=50.0,
        target_range=(40.0, 60.0),
        patient_weight=70.0
    )
    
    # Print report
    print(format_metrics_report(metrics))
    
    # Test tracker
    print("\nTesting MetricsTracker...")
    tracker = MetricsTracker(window_size=10)
    
    for ep in range(5):
        # Add some variation
        bis_ep = bis + np.random.randn(n_steps) * 2
        metrics_ep = calculate_all_metrics(bis_ep, dose, time, 50.0)
        reward = -np.sum((bis_ep - 50) ** 2) / n_steps
        tracker.add_episode(metrics_ep, reward)
    
    summary = tracker.get_summary()
    print(f"Summary after 5 episodes:")
    for key, value in summary.items():
        print(f"  {key}: {value:.2f}")
    
    print("\nTest complete!")
