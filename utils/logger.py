import csv
import os
import time
from collections import deque
from typing import Optional

class TrainingLogger:
    """
    Main job:
        Tracks episode rewards and optionally extra metrics (like loss, epsilon, score etc)
        Prints a formatted progress line every 'print_every' episodes
        Writes one CSV row per episode so runs are fully reproducible

        Usage example:
            logger = TrainingLogger(
                run_name = "dqn_cartpole",
                print_every = 10,
                window = 20,
                log_dir = "runs/"
            )
    """

    BASE_COLS = ["episode", "reward", "avg_reward", "elapsed_s"]

    def __init__(self, run_name: str, print_every: int, window: int, log_dir: Optional[str]):
        #log_dir is directory where CSV is written. Pass None to disable file logging
        self.run_name = run_name
        self.print_every = print_every
        self.window = window
        self.log_dir = log_dir

        self._rewards: list = []
        self._window_buffer: deque = deque(maxlen=window)
        self._start_time: float = time.time()
        self._episode_start: float = time.time()

        self._extra_keys: list = [] #Used for extra metrics

        self._csv_file = None
        self._csv_writer = None
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            csv_path = os.path.join(log_dir, f"{run_name}.csv")
            self._csv_file = open(csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames = self.BASE_COLS,
                extrasaction="ignore"
            )
        self._header_written = False

    def log(self, episode: int, reward: float, **metrics) -> None:
        self._rewards.append(reward)
        self._window_buffer.append(reward)
        avg = sum(self._window_buffer) / len(self._window_buffer)

        elapsed = time.time() - self._start_time

        for k in metrics: #To discover new metrics
            if k not in self._extra_keys:
                self._extra_keys.append(k)

        if self._csv_writer is not None and not self._header_written: #Write CSV once
            all_columns = self.BASE_COLS + self._extra_keys
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=all_columns,
                extrasaction="ignore"
            )
            self._csv_writer.writeheader()
            self._header_written = True

        row = {
            "episode": episode,
            "reward": round(reward, 4),
            "avg_reward": round(avg, 4),
            "elapsed_s": round(elapsed, 2),
            **{k: round(v,6) if isinstance(v, float) else v for k, v in metrics.items()}
        }

        if self._csv_writer is not None:
            self._csv_writer.writerow(row)
            self._csv_file.flush()

        if (episode + 1) %self.print_every == 0:
            self._print(episode, reward, avg, elapsed, metrics)

        self._episode_start = time.time()

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()

    def summary(self) -> dict:
        """Returns a dictionary with:
        total_episodes, best_reward, avg_last_N, total_time_s

        For example: s = logger.summary()
        """

        if not self._rewards:
            return {}

        last_n = list(self._window_buffer)
        return {
            "run_name": self.run_name,
            "total_episodes": len(self._rewards),
            "best_reward": max(self._rewards),
            "worst_reward": min(self._rewards),
            "mean_reward": sum(self._rewards) / len(self._rewards),
            f"avg_last_{self.window}": sum(last_n) / len(last_n),
            "total_time_s": round(time.time() - self._start_time, 1),
        }

    def print_summary(self) -> None:
        s = self.summary()
        if not s:
            return
        print("\n" + "=" * 50)
        print(f"  Run complete — {s['run_name']}")
        print("=" * 50)
        print(f"  Episodes  : {s['total_episodes']}")
        print(f"  Best      : {s['best_reward']:.2f}")
        print(f"  Mean      : {s['mean_reward']:.2f}")
        print(f"  Avg last {self.window:>3}: {s[f'avg_last_{self.window}']:.2f}")
        print(f"  Time      : {s['total_time_s']:.1f}s")
        print("=" * 50 + "\n")

    @property
    def rewards(self) -> list:
        return self._rewards

    @property
    def csv_path(self) -> Optional[str]:
        if self.log_dir is None:
            return None
        return os.path.join(self.log_dir, f"{self.run_name}.csv")

    def _print(self, episode, reward, avg, elapsed, metrics):
        line = (
            f"Episode {episode + 1:5d} | "
            f"Reward {reward:8.2f} | "
            f"Avg({self.window}) {avg:8.2f} | "
            f"Time {elapsed:6.1f}s"
        )

        for k, v in metrics.items():
            if isinstance(v, float):
                line += f" | {k} {v:.4f}"
            else:
                line += f" | {k} {v}"
        print(line)

