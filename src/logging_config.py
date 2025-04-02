import logging
import os
import yaml

def setup_logging(config_path):
    """
    Sets up logging based on the configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.
    """
    # Load the configuration file
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Access logging configuration
    log_level = config["logging"]["level"]
    log_file = config["logging"]["log_file"]
    log_format = config["logging"]["format"]

    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),  # Convert level string to logging constant
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),  # Log to file
            logging.StreamHandler()  # Log to console
        ]
    )

    logging.info("Logging is configured successfully.")