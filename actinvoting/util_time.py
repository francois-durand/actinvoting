import time


def current_time():
    """Return the current time."""
    return time.time()


def elapsed_time(start_time):
    """Return the elapsed time since `t_start`.

    Parameters
    ----------
    start_time: float
        Start time.

    Returns
    -------
    tuple
        First element: elapsed time in seconds. Second element: string representation of the elapsed time.
    """
    duration_seconds = time.time() - start_time
    seconds = duration_seconds % 60
    minutes = (duration_seconds // 60) % 60
    hours = duration_seconds // 3600
    duration_str = f'{hours:.0f}h' if hours > 0 else ''
    duration_str += f'{minutes:.0f}m' if hours > 0 or minutes > 0 else ''
    duration_str += f'{seconds:.0f}s'
    return duration_seconds, duration_str
