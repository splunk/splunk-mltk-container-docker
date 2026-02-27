import logging, os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime

def setup_logger(name: str = "app",
    log_file: str = "app.log",
    level: int = logging.INFO
) -> logging.Logger:
    """
    Creates a logger that logs to both console and file.

    Args:
        name: Logger name (use different names for different modules if desired)
        log_file: Path to log file
        level: Logging level (logging.INFO, logging.DEBUG, etc.)

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate logs if setup_logger() is called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (rotating logs)
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=5)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def setup_logger_dir(userName,userSession, curr_log_working_dir):
    ## Path of log folder for current session: 
    log_folder = os.path.join(curr_log_working_dir, userName)
    os.makedirs(log_folder, exist_ok=True) # Check if the folder exists

    ## Create file for the log
    current_date = datetime.now().strftime("%Y-%m-%d")
    name_of_file = userSession + "_" + current_date + ".log"
    return os.path.join(log_folder, name_of_file)


def get_logger(userName: str, userSession: str, base_dir: str) -> logging.Logger:
    """
    Creates (or reuses) a session logger that writes to a session log file.
    """

    # Create folder: <cwd>/logs
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Create folder for each user:
    user_log_dir = os.path.join(log_dir, userName)
    os.makedirs(user_log_dir, exist_ok=True)

    # File: <session>_YYYY-MM-DD.log
    today = datetime.now().strftime("%Y-%m-%d")
    log_path = os.path.join(user_log_dir, f"{userSession}_{today}.log")

    logger_name = f"{userName}.{userSession}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Prevent duplicated handlers
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        fh = logging.FileHandler(log_path)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
