"""Shared helper functions for the project."""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/CI environments
import matplotlib.pyplot as plt

RANDOM_STATE = 67


def get_project_root():
    """Return absolute path to the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setup_directories():
    """Create all directories the project pipeline expects."""
    root = get_project_root()
    directories = [
        os.path.join(root, 'data', 'raw'),
        os.path.join(root, 'data', 'processed'),
        os.path.join(root, 'models'),
        os.path.join(root, 'assets', 'plots'),
        os.path.join(root, 'assets', 'screenshots'),
        os.path.join(root, 'assets', 'architecture'),
        os.path.join(root, 'reports'),
    ]
    for d in directories:
        os.makedirs(d, exist_ok=True)


def safe_divide(numerator, denominator, default=0.0):
    """Safely divide numerator by (denominator + 1), returning default for non-finite results."""
    result = numerator / (denominator + 1)
    if hasattr(result, '__len__'):
        result = np.where(np.isfinite(result), result, default)
    elif not np.isfinite(result):
        result = default
    return result


def save_plot(fig, filename, directory=None):
    """Save a matplotlib figure to disk (defaults to assets/plots/)."""
    if directory is None:
        directory = os.path.join(get_project_root(), 'assets', 'plots')
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [SAVED] {filepath}")


def set_random_seed(seed=RANDOM_STATE):
    """Set random seeds for Python's random and NumPy."""
    import random
    random.seed(seed)
    np.random.seed(seed)


def validate_columns(df, required_columns):
    """Raise ValueError if df is missing any required columns."""
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def print_separator(title):
    """Print a formatted section header to the console."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    print_separator('Utils Module Self-Test')
    print(f"Project root : {get_project_root()}")
    print(f"RANDOM_STATE : {RANDOM_STATE}")

    assert safe_divide(10, 4) == 10 / 5, "scalar safe_divide failed"
    arr = safe_divide(np.array([10, 20]), np.array([4, 9]))
    assert arr[0] == 2.0 and arr[1] == 2.0, "array safe_divide failed"
    print("[OK] safe_divide works correctly.")

    import pandas as pd
    test_df = pd.DataFrame({'a': [1], 'b': [2]})
    validate_columns(test_df, ['a', 'b'])
    print("[OK] validate_columns works correctly.")

    setup_directories()
    print("[OK] setup_directories completed.")
    print("\nAll self-tests passed.")
