import logging
from functools import wraps

def setup_logger(name="biztelai"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(ch)

    return logger

logger = setup_logger()



def log_start_end(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"➡️ Starting: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"✅ Finished: {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"❌ Error in {func.__name__}: {e}")
            raise
    return wrapper
