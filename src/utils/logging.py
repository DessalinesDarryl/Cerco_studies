# src/utils/logging.py
import logging, os
def get_logger(name="app", logfile=None):
    logger = logging.getLogger(name); logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch = logging.StreamHandler(); ch.setFormatter(fmt); logger.handlers = [ch]
    if logfile:
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        fh = logging.FileHandler(logfile); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger
