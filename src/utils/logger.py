"""
Logging Utilities for Quantum RL Propofol Infusion
===================================================

Provides consistent logging across the entire project with proper formatting,
log levels, and handlers.

Usage:
------
    from src.utils.logger import get_logger
    
    logger = get_logger(__name__)
    logger.info("Training started")
    logger.debug("State: %s", state)
    logger.warning("BIS out of range: %.2f", bis)
    logger.error("Failed to load model: %s", error)

Features:
---------
- Colored console output
- File logging with rotation
- Configurable log levels
- Training-specific loggers
- Performance logging
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import colorlog


# Default log format
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
COLOR_FORMAT = '%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(name)s%(reset)s - %(message)s'

# Log colors
LOG_COLORS = {
    'DEBUG': 'cyan',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red,bg_white',
}


def setup_logging(
    log_dir: Optional[Path] = None,
    log_level: str = 'INFO',
    experiment_name: Optional[str] = None
) -> None:
    """
    Setup global logging configuration.
    
    Args:
        log_dir: Directory for log files. If None, only console logging is enabled.
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        experiment_name: Name of the experiment for log file naming
    
    Example:
        >>> setup_logging(log_dir=Path('logs/experiment1'), log_level='DEBUG')
    """
    level = getattr(logging, log_level.upper())
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler with colors
    console_handler = colorlog.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = colorlog.ColoredFormatter(
        COLOR_FORMAT,
        log_colors=LOG_COLORS,
        reset=True,
        style='%'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if log_dir specified)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Main log file
        if experiment_name:
            log_file = log_dir / f"{experiment_name}.log"
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = log_dir / f"training_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(DEFAULT_FORMAT)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Error log file (only errors and critical)
        error_log_file = log_dir / "errors.log"
        error_handler = logging.FileHandler(error_log_file)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Args:
        name: Logger name (typically __name__ of the module)
    
    Returns:
        Logger instance
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
    """
    return logging.getLogger(name)


class TrainingLogger:
    """
    Specialized logger for training metrics and progress.
    
    Logs training statistics, episode information, and performance metrics
    in a structured format.
    
    Example:
        >>> train_logger = TrainingLogger(log_dir='logs/experiment1')
        >>> train_logger.log_episode(episode=1, reward=45.2, steps=720, bis_mean=48.5)
        >>> train_logger.log_evaluation(mdape=15.3, wobble=12.1, time_in_target=85.2)
    """
    
    def __init__(self, log_dir: Path, experiment_name: str = "training"):
        """
        Initialize training logger.
        
        Args:
            log_dir: Directory for log files
            experiment_name: Name of the experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger(f"{__name__}.{experiment_name}")
        
        # CSV log files for metrics
        self.episode_log = self.log_dir / "episodes.csv"
        self.evaluation_log = self.log_dir / "evaluation.csv"
        
        # Initialize CSV headers
        if not self.episode_log.exists():
            with open(self.episode_log, 'w') as f:
                f.write("timestamp,episode,reward,steps,bis_mean,bis_std,mdape,time_in_target,actor_loss,critic_loss\n")
        
        if not self.evaluation_log.exists():
            with open(self.evaluation_log, 'w') as f:
                f.write("timestamp,episode,mdpe,mdape,wobble,time_in_target,induction_time,recovery_time\n")
    
    def log_episode(
        self,
        episode: int,
        reward: float,
        steps: int,
        bis_mean: float,
        bis_std: float,
        mdape: float,
        time_in_target: float,
        actor_loss: Optional[float] = None,
        critic_loss: Optional[float] = None
    ) -> None:
        """
        Log episode training information.
        
        Args:
            episode: Episode number
            reward: Total episode reward
            steps: Number of steps in episode
            bis_mean: Mean BIS value
            bis_std: BIS standard deviation
            mdape: Median absolute performance error
            time_in_target: Percentage time in target range
            actor_loss: Actor network loss (optional)
            critic_loss: Critic network loss (optional)
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Console log
        self.logger.info(
            f"Episode {episode:4d} | Reward: {reward:7.2f} | Steps: {steps:3d} | "
            f"BIS: {bis_mean:5.2f}±{bis_std:4.2f} | MDAPE: {mdape:5.2f}% | "
            f"TIT: {time_in_target:5.2f}%"
        )
        
        # CSV log
        with open(self.episode_log, 'a') as f:
            f.write(f"{timestamp},{episode},{reward:.4f},{steps},{bis_mean:.4f},{bis_std:.4f},"
                   f"{mdape:.4f},{time_in_target:.4f},{actor_loss or ''},"
                   f"{critic_loss or ''}\n")
    
    def log_evaluation(
        self,
        episode: int,
        mdpe: float,
        mdape: float,
        wobble: float,
        time_in_target: float,
        induction_time: Optional[float] = None,
        recovery_time: Optional[float] = None
    ) -> None:
        """
        Log evaluation metrics.
        
        Args:
            episode: Episode number
            mdpe: Median performance error (bias)
            mdape: Median absolute performance error
            wobble: Intra-individual variability
            time_in_target: Percentage time in target (40-60)
            induction_time: Time to reach BIS <= 60 (seconds)
            recovery_time: Time from stop to BIS >= 80 (seconds)
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Console log
        self.logger.info(
            f"Evaluation @ Episode {episode} | MDPE: {mdpe:6.2f}% | "
            f"MDAPE: {mdape:6.2f}% | Wobble: {wobble:6.2f}% | TIT: {time_in_target:5.2f}%"
        )
        
        if induction_time:
            self.logger.info(f"  Induction time: {induction_time:.1f}s")
        if recovery_time:
            self.logger.info(f"  Recovery time: {recovery_time:.1f}s")
        
        # CSV log
        with open(self.evaluation_log, 'a') as f:
            f.write(f"{timestamp},{episode},{mdpe:.4f},{mdape:.4f},{wobble:.4f},"
                   f"{time_in_target:.4f},{induction_time or ''},"
                   f"{recovery_time or ''}\n")
    
    def log_phase(self, phase: str, message: str) -> None:
        """
        Log training phase transitions.
        
        Args:
            phase: Phase name ('Offline', 'Online', 'Evaluation')
            message: Phase message
        """
        separator = "=" * 70
        self.logger.info(separator)
        self.logger.info(f"{phase}: {message}")
        self.logger.info(separator)


def log_model_info(logger: logging.Logger, model: object, model_name: str) -> None:
    """
    Log model architecture information.
    
    Args:
        logger: Logger instance
        model: Model object (PyTorch module or similar)
        model_name: Name of the model for logging
    """
    logger.info(f"{model_name} Architecture:")
    
    # Try to get parameter count
    try:
        import torch.nn as nn
        if isinstance(model, nn.Module):
            n_params = sum(p.numel() for p in model.parameters())
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"  Total parameters: {n_params:,}")
            logger.info(f"  Trainable parameters: {n_trainable:,}")
    except Exception as e:
        logger.debug(f"Could not count parameters: {e}")
    
    # Log model structure
    logger.debug(f"{model}")


def log_config(logger: logging.Logger, config: dict, config_name: str = "Configuration") -> None:
    """
    Log configuration dictionary in a readable format.
    
    Args:
        logger: Logger instance
        config: Configuration dictionary
        config_name: Name for the configuration section
    """
    logger.info(f"{config_name}:")
    
    def log_dict(d: dict, indent: int = 2):
        for key, value in d.items():
            if isinstance(value, dict):
                logger.info(f"{' ' * indent}{key}:")
                log_dict(value, indent + 2)
            else:
                logger.info(f"{' ' * indent}{key}: {value}")
    
    log_dict(config)


# Example usage
if __name__ == "__main__":
    # Setup logging
    setup_logging(
        log_dir=Path("logs/test"),
        log_level="DEBUG",
        experiment_name="test_logging"
    )
    
    # Get logger
    logger = get_logger(__name__)
    
    # Test different log levels
    logger.debug("Debug message - detailed information")
    logger.info("Info message - general information")
    logger.warning("Warning message - something unexpected")
    logger.error("Error message - something failed")
    
    # Test training logger
    train_logger = TrainingLogger(log_dir=Path("logs/test"), experiment_name="test")
    
    train_logger.log_phase("Offline Training", "Starting behavioral cloning")
    
    for episode in range(1, 4):
        train_logger.log_episode(
            episode=episode,
            reward=45.2 + episode,
            steps=720,
            bis_mean=48.5,
            bis_std=5.2,
            mdape=15.3,
            time_in_target=85.2,
            actor_loss=0.01,
            critic_loss=0.05
        )
    
    train_logger.log_evaluation(
        episode=3,
        mdpe=-5.2,
        mdape=15.3,
        wobble=12.1,
        time_in_target=85.2,
        induction_time=180.0,
        recovery_time=300.0
    )
    
    logger.info("Logging test complete!")
