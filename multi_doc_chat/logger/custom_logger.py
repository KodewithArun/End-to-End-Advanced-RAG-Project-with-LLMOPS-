import os
import logging
from logging.handlers import RotatingFileHandler
import structlog


class CustomLogger:
    def __init__(self, log_dir="logs", log_filename="app.log", max_bytes=5 * 1024 * 1024, backup_count=5):
        """
        Rotating Logger:
        - max_bytes: Maximum size per log file before rotating (default: 5MB)
        - backup_count: Maximum number of backup log files to retain (default: 5)
        """
        self.logs_dir = os.path.join(os.getcwd(), log_dir)
        os.makedirs(self.logs_dir, exist_ok=True)
        self.log_file_path = os.path.join(self.logs_dir, log_filename)
        self.max_bytes = max_bytes
        self.backup_count = backup_count

    def get_logger(self, name=__file__):
        logger_name = os.path.basename(name)

        # Rotating file handler: 5MB max, 5 backup files
        file_handler = RotatingFileHandler(
            self.log_file_path,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(message)s"))

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(message)s"))

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # Prevent duplicate handlers if get_logger is called multiple times
        has_rotating = any(isinstance(h, RotatingFileHandler) for h in root_logger.handlers)
        has_console = any(
            isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
            for h in root_logger.handlers
        )

        if not has_console:
            root_logger.addHandler(console_handler)
        if not has_rotating:
            root_logger.addHandler(file_handler)

        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
                structlog.processors.add_log_level,
                structlog.processors.EventRenamer(to="event"),
                structlog.processors.JSONRenderer(),
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        return structlog.get_logger(logger_name)
