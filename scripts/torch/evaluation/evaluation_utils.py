def get_time_left(windows_left: int, window_size_seconds: int, window_overlap_seconds: int):
    return windows_left * (window_size_seconds - window_overlap_seconds)
