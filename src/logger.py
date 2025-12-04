"""Centralized logging configuration for the heart disease ML system."""

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
import json


LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Timestamp for log files
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs logs as JSON for better parsing."""
    
    def format(self, record):
        log_obj = {
            "timestamp": self.formatTime(record, "%Y-%m-%d %H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_obj)


def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.hasHandlers():
        return logger
    
    # File handler (JSON format)
    json_handler = logging.FileHandler(
        LOG_DIR / f"app_{TIMESTAMP}.log"
    )
    json_handler.setLevel(logging.DEBUG)
    json_formatter = JSONFormatter()
    json_handler.setFormatter(json_formatter)
    
    # File handler for errors (plain text)
    error_handler = logging.FileHandler(
        LOG_DIR / f"errors_{TIMESTAMP}.log"
    )
    error_handler.setLevel(logging.ERROR)
    error_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    error_handler.setFormatter(error_formatter)
    
    # Console handler (plain text)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(json_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)
    
    return logger


# Main application logger
logger = setup_logger("heart_disease_app", level=logging.INFO)


def log_model_metrics(metrics: dict, model_name: str):
    """Log model metrics in a structured way."""
    logger.info(
        f"Model metrics for {model_name}",
        extra={"metrics": metrics}
    )


def log_training_start(config: dict):
    """Log training configuration."""
    logger.info(f"Starting model training with config: {config}")


def log_training_end(results: dict):
    """Log training completion."""
    logger.info(f"Training completed with results: {results}")


def log_prediction(input_data: dict, prediction: dict, model_version: str):
    """Log prediction events."""
    logger.info(
        f"Prediction made with model {model_version}",
        extra={
            "input_features": list(input_data.keys()),
            "prediction_result": prediction
        }
    )
