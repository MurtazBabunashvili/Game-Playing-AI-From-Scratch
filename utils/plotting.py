import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def _smooth(values: list, window: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (x_indices, smoothed_values) for a moving average overlay.
    Only computed when len(values) >= window.
    """
    arr      = np.array(values, dtype=np.float32)
    smoothed = np.convolve(arr, np.ones(window) / window, mode="valid")
    x        = np.arange(window - 1, len(arr))
    return x, smoothed


def _base_ax(ax, xlabel: str, ylabel: str, title: str) -> None:
    """Apply consistent axis labels and title."""
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_or_show(fig, save_path: Optional[str]) -> None:
    try:
        fig.tight_layout()
    except Exception:
        pass
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    plt.close(fig)

def plot_training_curve(
    episode_rewards:  list,
    title:            str           = "Training Curve",
    window:           int           = 20,
    color:            str           = "steelblue",
    save_path:        Optional[str] = None,
) -> None:
    """
    Plot total reward per episode with a smoothed moving-average overlay.

    Parameters
        episode_rewards : list of float
            One value per episode.
        title   : str
        window  : int.  Moving-average window size.
        color   : str.  Line / fill colour.
        save_path : str or None.
            If given, save figure to disk instead of showing it.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    x_raw = np.arange(len(episode_rewards))
    ax.plot(x_raw, episode_rewards, alpha=0.25, color=color, linewidth=0.8, label="raw reward")
    ax.fill_between(x_raw, episode_rewards, alpha=0.06, color=color)

    if len(episode_rewards) >= window:
        x_s, smoothed = _smooth(episode_rewards, window)
        ax.plot(x_s, smoothed, color=color, linewidth=2.0,
                label=f"avg ({window} episodes)")

    _base_ax(ax, "Episode", "Total Reward", title)
    ax.legend(framealpha=0.4)
    _save_or_show(fig, save_path)


def plot_comparison(
    histories:  list[list],
    labels:     list[str],
    title:      str           = "Algorithm Comparison",
    window:     int           = 20,
    colors:     Optional[list[str]] = None,
    save_path:  Optional[str] = None,
) -> None:
    """
    Overlay multiple reward histories on one plot.

    Parameters
        histories : list of lists.  One reward list per algorithm.
        labels    : list of str.    One label per history.
        title     : str
        window    : int.            Moving-average window.
        colors    : list of str or None.  Auto-assigned if omitted.
        save_path : str or None.
    """
    default_colors = ["steelblue", "tomato", "seagreen", "darkorange", "mediumpurple"]
    colors = colors or default_colors[: len(histories)]

    fig, ax = plt.subplots(figsize=(9, 4))

    for history, label, color in zip(histories, labels, colors):
        arr = np.array(history, dtype=np.float32)
        ax.plot(arr, alpha=0.15, color=color, linewidth=0.8)
        if len(history) >= window:
            x_s, smoothed = _smooth(history, window)
            ax.plot(x_s, smoothed, color=color, linewidth=2.0, label=label)

    _base_ax(ax, "Episode", "Total Reward", title)
    ax.legend(framealpha=0.4)
    _save_or_show(fig, save_path)

def plot_dashboard(
    episode_rewards: list,
    metrics:         Optional[dict[str, list]] = None,
    title:           str           = "Training Dashboard",
    window:          int           = 20,
    save_path:       Optional[str] = None,
) -> None:
    """
    One figure with reward on top and one sub-plot per extra metric below.

    Parameters
        episode_rewards : list of float
        metrics : dict {metric_name: list of float}
            Extra series to plot below the reward panel.
            Example: {"epsilon": [...], "loss": [...]}
        title   : str
        window  : int
        save_path : str or None
    """
    metrics  = metrics or {}
    n_panels = 1 + len(metrics)

    fig = plt.figure(figsize=(9, 3 * n_panels))
    gs  = gridspec.GridSpec(n_panels, 1, hspace=0.45)

    # ── Top panel: reward ──────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    x_raw = np.arange(len(episode_rewards))
    ax0.plot(x_raw, episode_rewards, alpha=0.25, color="steelblue",
             linewidth=0.8, label="raw reward")
    ax0.fill_between(x_raw, episode_rewards, alpha=0.06, color="steelblue")

    if len(episode_rewards) >= window:
        x_s, smoothed = _smooth(episode_rewards, window)
        ax0.plot(x_s, smoothed, color="steelblue", linewidth=2.0,
                 label=f"avg ({window})")

    _base_ax(ax0, "", "Total Reward", title)
    ax0.legend(framealpha=0.4, fontsize=9)

    # ── Extra metric panels ────────────────────────────────────────────────
    metric_colors = ["tomato", "seagreen", "darkorange", "mediumpurple", "sienna"]

    for i, (name, values) in enumerate(metrics.items(), start=1):
        ax = fig.add_subplot(gs[i])
        color = metric_colors[(i - 1) % len(metric_colors)]
        arr = np.array(values, dtype=np.float32)
        ax.plot(arr, alpha=0.3, color=color, linewidth=0.8)

        if len(values) >= window:
            x_s, smoothed = _smooth(values, window)
            ax.plot(x_s, smoothed, color=color, linewidth=2.0)

        _base_ax(ax, "Episode" if i == len(metrics) else "", name, "")
        ax.set_ylabel(name, fontsize=10)

    _save_or_show(fig, save_path)


def plot_from_csv(
    csv_path:  str,
    reward_col: str           = "reward",
    extra_cols: Optional[list[str]] = None,
    window:     int           = 20,
    title:      Optional[str] = None,
    save_path:  Optional[str] = None,
) -> None:
    """
    Recreate a dashboard plot directly from a TrainingLogger CSV file.

    Parameters
        csv_path    : str.  Path to the CSV written by TrainingLogger.
        reward_col  : str.  Column to use as the main reward series.
        extra_cols  : list of str or None.
            Additional columns to show as sub-panels (e.g. ["epsilon", "loss"]).
            If None, all numeric columns except reward and episode are shown.
        window      : int
        title       : str or None.  Defaults to the CSV filename stem.
        save_path   : str or None.

    """
    import csv

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"CSV is empty: {csv_path}")
        return

    def _col(name):
        return [float(r[name]) for r in rows if r.get(name) not in (None, "")]

    rewards = _col(reward_col)

    # Auto-detect extra numeric columns
    if extra_cols is None:
        skip = {"episode", reward_col, "avg_reward", "elapsed_s"}
        extra_cols = [k for k in rows[0] if k not in skip]

    metrics = {}
    for col in extra_cols:
        if col in rows[0]:
            metrics[col] = _col(col)

    title = title or os.path.splitext(os.path.basename(csv_path))[0]
    plot_dashboard(rewards, metrics=metrics, title=title,
                   window=window, save_path=save_path)