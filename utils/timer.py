import time
from functools import wraps
from utils.logger import logger

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        elapsed = end - start
        logger.info(f"'{func.__name__}' executed in {elapsed:.4f}s")
        return result
    return wrapper
