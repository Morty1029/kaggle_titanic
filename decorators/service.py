import time


def execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        print(f"ВРЕМЯ РАБОТЫ {func.__name__} = {time.time() - start_time}")
        return res

    return wrapper
