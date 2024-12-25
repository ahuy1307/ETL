import logging
import os

class AppLog:
    _logger = None

    @classmethod
    def _initialize_logger(cls):
        if cls._logger is not None:
            return  # Logger is already initialized

        # Set up a custom logger
        cls._logger = logging.getLogger("AppLogger")
        cls._logger.setLevel(logging.DEBUG)  # Set the base level to DEBUG to capture all levels

        # Create handlers
        console_handler = logging.StreamHandler()  # Logs to console
        os.makedirs("logs", exist_ok=True)  # Ensure the logs directory exists
        file_handler = logging.FileHandler("logs/app.log")  # Logs to a file

        # Set logging levels for each handler
        console_handler.setLevel(logging.INFO)
        file_handler.setLevel(logging.DEBUG)

        # Create and set the formatter for logs
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add handlers to the logger
        cls._logger.addHandler(console_handler)
        cls._logger.addHandler(file_handler)

    @classmethod
    def debug(cls, msg, *args, **kwargs):
        cls._initialize_logger()
        cls._logger.debug(msg, *args, **kwargs)

    @classmethod
    def info(cls, msg, *args, **kwargs):
        cls._initialize_logger()
        cls._logger.info(msg, *args, **kwargs)

    @classmethod
    def error(cls, msg, *args, **kwargs):
        cls._initialize_logger()
        cls._logger.error(msg, *args, **kwargs)