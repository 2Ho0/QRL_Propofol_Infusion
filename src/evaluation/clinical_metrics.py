"""
Clinical Performance Metrics for Anesthesia Control
====================================================

Additional clinical metrics beyond MDPE, MDAPE, and Wobble:
- Induction time: Time to reach target BIS
- Maintenance quality: Stability in target range
- Recovery time: Time to wake up after drug cessation
- Safety violations: Hypotension, hypoxia, bradycardia events
- Drug efficiency: Total drug consumption

These metrics are clinically relevant for evaluating anesthesia controllers.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ClinicalMetrics:
    """Container for clinical performance metrics."""
    # Standard metrics
    mdpe: float  # Median Performance Error
    mdape: float  # Median Absolute Performance Error  
    wobble: float  # Wobble (median absolute deviation)
    
    # Time-based metrics
    induction_time: float  # Minutes to reach target
    time_in_target: float  # Percentage of time in target range
    recovery_time: Optional[float]  # Minutes to recover consciousness
    
    # Safety metrics
    safety_violations: Dict[str, int]  # Count of each violation type
    severe_events: int  # Count of severe adverse events
    
    # Drug efficiency
    total_drug_consumption: float  # Total propofol mg/kg
    drug_efficiency: float  # Drug per minute in target
    
    # Episode metrics
    episode_reward: float
    episode_length: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'mdpe': self.mdpe,
            'mdape': self.mdape,
            'wobble': self.wobble,
            'induction_time': self.induction_time,
            'time_in_target': self.time_in_target,
            'recovery_time': self.recovery_time,
            'safety_violations': self.safety_violations,
            'severe_events': self.severe_events,
            'total_drug_consumption': self.total_drug_consumption,
            'drug_efficiency': self.drug_efficiency,
            'episode_reward': self.episode_reward,
            'episode_length': self.episode_length
        }


def compute_induction_time(
    bis_history: np.ndarray,
    target_bis: float = 50,
    threshold: float = 5,
    sampling_rate: float = 1.0
) -> float:
    """
    Compute time to reach target BIS during induction.
    
    Induction is the period from consciousness (BIS ~90-100) to 
    surgical anesthesia (BIS ~40-60).
    
    Args:
        bis_history: Array of BIS values over time
        target_bis: Target BIS value (default: 50)
        threshold: Acceptable deviation from target (default: ±5)
        sampling_rate: Samples per minute (default: 1.0)
    
    Returns:
        induction_time: Time in minutes to reach target range
    """
    target_range = (target_bis - threshold, target_bis + threshold)
    
    for t, bis in enumerate(bis_history):
        if target_range[0] <= bis <= target_range[1]:
            return t / sampling_rate  # Convert to minutes
    
    # If never reached target, return total time
    return len(bis_history) / sampling_rate


def compute_time_in_target(
    bis_history: np.ndarray,
    target_bis: float = 50,
    threshold: float = 5,
    start_idx: int = 0
) -> float:
    """
    Compute percentage of time spent in target BIS range.
    
    Args:
        bis_history: Array of BIS values
        target_bis: Target BIS value
        threshold: Acceptable deviation
        start_idx: Start index (e.g., after induction)
    
    Returns:
        Percentage of time in target range (0-100)
    """
    target_range = (target_bis - threshold, target_bis + threshold)
    relevant_history = bis_history[start_idx:]
    
    if len(relevant_history) == 0:
        return 0.0
    
    in_target = np.sum(
        (relevant_history >= target_range[0]) & 
        (relevant_history <= target_range[1])
    )
    
    return 100.0 * in_target / len(relevant_history)


def compute_recovery_time(
    bis_history: np.ndarray,
    drug_history: np.ndarray,
    recovery_threshold: float = 70,
    sampling_rate: float = 1.0
) -> Optional[float]:
    """
    Compute time to recover consciousness after drug cessation.
    
    Recovery is defined as BIS rising above recovery_threshold
    after drug infusion stops.
    
    Args:
        bis_history: Array of BIS values
        drug_history: Array of drug infusion rates
        recovery_threshold: BIS threshold for consciousness (default: 70)
        sampling_rate: Samples per minute
    
    Returns:
        recovery_time: Time in minutes from drug stop to recovery,
                      or None if drug never stopped or no recovery
    """
    # Find when drug infusion stops
    cessation_time = None
    for t in range(len(drug_history) - 1):
        if drug_history[t] > 0 and drug_history[t + 1] == 0:
            cessation_time = t
            break
    
    if cessation_time is None:
        return None  # Drug never stopped
    
    # Find when BIS exceeds recovery threshold
    for t in range(cessation_time, len(bis_history)):
        if bis_history[t] > recovery_threshold:
            return (t - cessation_time) / sampling_rate
    
    return None  # No recovery observed


def detect_safety_violations(
    bis_history: np.ndarray,
    drug_history: np.ndarray,
    state_history: Optional[np.ndarray] = None,
    bis_limits: Tuple[float, float] = (20, 70),
    max_drug_rate: float = 20.0
) -> Dict[str, int]:
    """
    Detect safety violations during anesthesia.
    
    Safety criteria:
    - BIS should stay within safe limits (avoid overdose or awareness)
    - Drug rate should not exceed safe limits
    - If vital signs available, check for hypotension, bradycardia
    
    Args:
        bis_history: Array of BIS values
        drug_history: Array of drug infusion rates
        state_history: Optional array of full state (for vital signs)
        bis_limits: Safe BIS range (min, max)
        max_drug_rate: Maximum safe drug rate mg/kg/h
    
    Returns:
        Dictionary with counts of each violation type
    """
    violations = {
        'bis_too_low': 0,      # Deep anesthesia (BIS < 20)
        'bis_too_high': 0,     # Risk of awareness (BIS > 70)
        'drug_overdose': 0,    # Excessive drug rate
        'rapid_drug_change': 0  # Too rapid infusion changes
    }
    
    # Check BIS limits
    violations['bis_too_low'] = np.sum(bis_history < bis_limits[0])
    violations['bis_too_high'] = np.sum(bis_history > bis_limits[1])
    
    # Check drug rate limits
    violations['drug_overdose'] = np.sum(drug_history > max_drug_rate)
    
    # Check for rapid drug changes (potentially dangerous)
    if len(drug_history) > 1:
        drug_changes = np.abs(np.diff(drug_history))
        # Flag changes > 5 mg/kg/h per minute
        violations['rapid_drug_change'] = np.sum(drug_changes > 5.0)
    
    # If state history available, check vital signs
    if state_history is not None:
        # Assuming state includes MAP and HR in later dimensions
        # This is environment-specific
        pass
    
    return violations


def compute_drug_efficiency(
    bis_history: np.ndarray,
    drug_history: np.ndarray,
    target_bis: float = 50,
    threshold: float = 5,
    sampling_rate: float = 1.0,
    patient_weight: float = 70.0
) -> Tuple[float, float]:
    """
    Compute total drug consumption and efficiency.
    
    Drug efficiency = How much drug was needed to maintain target BIS.
    
    Args:
        bis_history: Array of BIS values
        drug_history: Array of drug infusion rates (mg/kg/h)
        target_bis: Target BIS
        threshold: Acceptable deviation
        sampling_rate: Samples per minute
        patient_weight: Patient weight in kg
    
    Returns:
        total_consumption: Total drug in mg/kg
        efficiency: Drug per minute in target (lower is better)
    """
    # Total drug consumption (mg/kg)
    # drug_rate is in mg/kg/h, integrate over time
    dt = 1.0 / sampling_rate  # Time step in hours
    total_consumption = np.sum(drug_history) * dt
    
    # Time in target
    target_range = (target_bis - threshold, target_bis + threshold)
    time_in_target = np.sum(
        (bis_history >= target_range[0]) & 
        (bis_history <= target_range[1])
    ) / sampling_rate  # in minutes
    
    # Efficiency: drug per minute in target
    if time_in_target > 0:
        efficiency = total_consumption / time_in_target
    else:
        efficiency = float('inf')  # Never reached target
    
    return total_consumption, efficiency


def compute_maintenance_quality(
    bis_history: np.ndarray,
    target_bis: float = 50,
    induction_time_idx: int = 0
) -> Dict[str, float]:
    """
    Evaluate quality of BIS maintenance after induction.
    
    Metrics:
    - Standard deviation of BIS (lower is better)
    - Number of overshoots/undershoots
    - Maximum deviation from target
    
    Args:
        bis_history: Array of BIS values
        target_bis: Target BIS value
        induction_time_idx: Index where induction ended
    
    Returns:
        Dictionary with maintenance quality metrics
    """
    maintenance_bis = bis_history[induction_time_idx:]
    
    if len(maintenance_bis) == 0:
        return {
            'std_dev': 0.0,
            'overshoots': 0,
            'undershoots': 0,
            'max_deviation': 0.0
        }
    
    # Standard deviation (stability)
    std_dev = np.std(maintenance_bis)
    
    # Count overshoots (BIS > target + 5) and undershoots (BIS < target - 5)
    overshoots = np.sum(maintenance_bis > target_bis + 5)
    undershoots = np.sum(maintenance_bis < target_bis - 5)
    
    # Maximum deviation
    max_deviation = np.max(np.abs(maintenance_bis - target_bis))
    
    return {
        'std_dev': std_dev,
        'overshoots': overshoots,
        'undershoots': undershoots,
        'max_deviation': max_deviation
    }


def evaluate_episode(
    bis_history: np.ndarray,
    drug_history: np.ndarray,
    reward_history: np.ndarray,
    target_bis: float = 50,
    threshold: float = 5,
    patient_weight: float = 70.0
) -> ClinicalMetrics:
    """
    Comprehensive clinical evaluation of an episode.
    
    Computes all clinical metrics for a complete anesthesia episode.
    
    Args:
        bis_history: BIS values over time
        drug_history: Drug infusion rates over time
        reward_history: Rewards received over time
        target_bis: Target BIS value
        threshold: Acceptable deviation
        patient_weight: Patient weight in kg
    
    Returns:
        ClinicalMetrics object with all computed metrics
    """
    # Standard metrics (MDPE, MDAPE, Wobble)
    pe = target_bis - bis_history  # Performance Error
    mdpe = float(np.median(pe))
    mdape = float(np.median(np.abs(pe)))
    wobble = float(np.median(np.abs(pe - mdpe)))
    
    # Time-based metrics
    induction_time = compute_induction_time(bis_history, target_bis, threshold)
    time_in_target = compute_time_in_target(bis_history, target_bis, threshold)
    recovery_time = compute_recovery_time(bis_history, drug_history)
    
    # Safety violations
    safety_violations = detect_safety_violations(bis_history, drug_history)
    severe_events = (
        safety_violations['bis_too_low'] + 
        safety_violations['drug_overdose']
    )
    
    # Drug efficiency
    total_consumption, efficiency = compute_drug_efficiency(
        bis_history, drug_history, target_bis, threshold,
        patient_weight=patient_weight
    )
    
    # Episode metrics
    episode_reward = float(np.sum(reward_history))
    episode_length = len(bis_history)
    
    return ClinicalMetrics(
        mdpe=mdpe,
        mdape=mdape,
        wobble=wobble,
        induction_time=induction_time,
        time_in_target=time_in_target,
        recovery_time=recovery_time,
        safety_violations=safety_violations,
        severe_events=severe_events,
        total_drug_consumption=total_consumption,
        drug_efficiency=efficiency,
        episode_reward=episode_reward,
        episode_length=episode_length
    )


def print_metrics_summary(metrics: ClinicalMetrics):
    """Print formatted summary of clinical metrics."""
    print("\n" + "=" * 70)
    print("CLINICAL PERFORMANCE METRICS")
    print("=" * 70)
    
    print("\n📊 Standard Metrics:")
    print(f"  MDPE (Median Performance Error):  {metrics.mdpe:>8.2f}")
    print(f"  MDAPE (Median Absolute PE):       {metrics.mdape:>8.2f}")
    print(f"  Wobble (Variability):             {metrics.wobble:>8.2f}")
    
    print("\n⏱️  Time-Based Metrics:")
    print(f"  Induction Time:                   {metrics.induction_time:>8.2f} min")
    print(f"  Time in Target Range:             {metrics.time_in_target:>8.2f}%")
    if metrics.recovery_time:
        print(f"  Recovery Time:                    {metrics.recovery_time:>8.2f} min")
    else:
        print(f"  Recovery Time:                    N/A (drug not stopped)")
    
    print("\n⚠️  Safety Metrics:")
    print(f"  BIS Too Low Events:               {metrics.safety_violations['bis_too_low']:>8}")
    print(f"  BIS Too High Events:              {metrics.safety_violations['bis_too_high']:>8}")
    print(f"  Drug Overdose Events:             {metrics.safety_violations['drug_overdose']:>8}")
    print(f"  Rapid Drug Changes:               {metrics.safety_violations['rapid_drug_change']:>8}")
    print(f"  Severe Events:                    {metrics.severe_events:>8}")
    
    print("\n💊 Drug Efficiency:")
    print(f"  Total Drug Consumption:           {metrics.total_drug_consumption:>8.2f} mg/kg")
    print(f"  Drug Efficiency:                  {metrics.drug_efficiency:>8.4f} mg/kg/min")
    
    print("\n🎯 Episode Summary:")
    print(f"  Episode Reward:                   {metrics.episode_reward:>8.2f}")
    print(f"  Episode Length:                   {metrics.episode_length:>8} steps")
    
    print("=" * 70)
