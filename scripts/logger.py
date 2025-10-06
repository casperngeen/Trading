'''
Don't really need this for now
'''
import logging

def setup_logger(name=__name__, level=logging.INFO, log_file=None):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    log_format = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(log_format)
    logger.addHandler(ch)
    
    # Optional: File handler
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(log_format)
        logger.addHandler(fh)
    
    # Prevent duplicate logs
    logger.propagate = False
    return logger