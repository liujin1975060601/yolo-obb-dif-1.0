
import json
import logging

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        logging.error(f"Config file not found: {config_path}")
        raise
    except json.JSONDecodeError:
        logging.error(f"Error decoding config file: {config_path}")
        raise

def save_config(config, config_path):
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        logging.info(f"Config saved to {config_path}")
    except IOError as e:
        logging.error(f"Error saving config file: {config_path} — {e}")
        raise
