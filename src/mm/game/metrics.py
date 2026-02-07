from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Mapping


StepMetrics = Dict[str, Any]
EpisodeSummary = Dict[str, Any]


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def step_metrics_from_env(env: Any) -> StepMetrics:
    station_queues = [len(station.passengers) for station in env.stations]
    station_caps = [station.capacity for station in env.stations]
    metro_loads = [len(metro.passengers) for metro in env.metros]
    metro_caps = [metro.capacity for metro in env.metros]

    total_station_queue = sum(station_queues)
    total_station_capacity = sum(station_caps)
    total_metro_load = sum(metro_loads)
    total_metro_capacity = sum(metro_caps)
    active_passengers = total_station_queue + total_metro_load

    full_stations = sum(q >= c for q, c in zip(station_queues, station_caps))
    full_metros = sum(q >= c for q, c in zip(metro_loads, metro_caps))
    idle_metros = sum(m.current_station is None for m in env.metros)

    served_station_ids = set()
    for path in env.paths:
        for station in path.stations:
            served_station_ids.add(station.id)
    num_stations = len(env.stations)
    station_coverage_ratio = _safe_div(len(served_station_ids), num_stations)

    return {
        "time_ms": env.time_ms,
        "steps": env.steps,
        "score": env.score,
        "num_paths": len(env.paths),
        "num_looped_paths": sum(path.is_looped for path in env.paths),
        "num_stations": num_stations,
        "num_metros": len(env.metros),
        "active_passengers": active_passengers,
        "station_queue_total": total_station_queue,
        "station_queue_avg": _safe_div(total_station_queue, num_stations),
        "station_queue_max": max(station_queues) if station_queues else 0,
        "metro_load_total": total_metro_load,
        "metro_load_avg": _safe_div(total_metro_load, len(metro_loads)),
        "metro_load_max": max(metro_loads) if metro_loads else 0,
        "station_utilization": _safe_div(total_station_queue, total_station_capacity),
        "metro_utilization": _safe_div(total_metro_load, total_metro_capacity),
        "full_stations": full_stations,
        "full_metros": full_metros,
        "idle_metros": idle_metros,
        "station_coverage_ratio": station_coverage_ratio,
        "is_paused": env.is_paused,
    }


def summarize_episode(
    step_metrics: List[StepMetrics],
    rewards: List[float] | None = None,
) -> EpisodeSummary:
    if not step_metrics:
        return {
            "num_steps": 0,
            "duration_ms": 0,
            "final_score": 0,
            "avg_reward": 0.0,
            "throughput_per_min": 0.0,
        }

    first = step_metrics[0]
    last = step_metrics[-1]
    duration_ms = max(0, int(last["time_ms"]) - int(first["time_ms"]))
    total_reward = float(sum(rewards)) if rewards is not None else float(last["score"] - first["score"])

    return {
        "num_steps": len(step_metrics),
        "duration_ms": duration_ms,
        "final_score": int(last["score"]),
        "score_gain": int(last["score"] - first["score"]),
        "total_reward": total_reward,
        "avg_reward": _safe_div(total_reward, len(step_metrics)),
        "throughput_per_min": _safe_div(last["score"], max(last["time_ms"], 1) / 60000.0),
        "avg_active_passengers": _safe_div(sum(m["active_passengers"] for m in step_metrics), len(step_metrics)),
        "peak_active_passengers": max(m["active_passengers"] for m in step_metrics),
        "avg_station_queue": _safe_div(sum(m["station_queue_avg"] for m in step_metrics), len(step_metrics)),
        "peak_station_queue": max(m["station_queue_max"] for m in step_metrics),
        "avg_metro_utilization": _safe_div(sum(m["metro_utilization"] for m in step_metrics), len(step_metrics)),
        "avg_station_coverage_ratio": _safe_div(sum(m["station_coverage_ratio"] for m in step_metrics), len(step_metrics)),
        "peak_full_stations": max(m["full_stations"] for m in step_metrics),
        "peak_full_metros": max(m["full_metros"] for m in step_metrics),
    }


@dataclass
class MetricsTracker:
    stall_window_steps: int = 300
    overcrowded_station_ratio: float = 0.5
    step_rows: List[StepMetrics] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    score_window: Deque[int] = field(default_factory=deque)
    done_reason: str | None = None

    def reset(self) -> None:
        self.step_rows.clear()
        self.rewards.clear()
        self.score_window.clear()
        self.done_reason = None

    def record_step(
        self,
        env: Any,
        reward: float,
        *,
        terminated: bool = False,
        truncated: bool = False,
    ) -> StepMetrics:
        row = step_metrics_from_env(env)
        self.step_rows.append(row)
        self.rewards.append(float(reward))

        self.score_window.append(int(row["score"]))
        if len(self.score_window) > self.stall_window_steps:
            self.score_window.popleft()

        if terminated:
            self.done_reason = "terminated"
        elif truncated:
            self.done_reason = "truncated"
        elif self._is_stalled():
            self.done_reason = "stalled"
        elif self._is_overcrowded(row):
            self.done_reason = "overcrowded"

        return row

    def _is_stalled(self) -> bool:
        if len(self.score_window) < self.stall_window_steps:
            return False
        return self.score_window[0] == self.score_window[-1]

    def _is_overcrowded(self, row: Mapping[str, Any]) -> bool:
        num_stations = int(row["num_stations"])
        if num_stations == 0:
            return False
        ratio = float(row["full_stations"]) / num_stations
        return ratio >= self.overcrowded_station_ratio

    def summary(self) -> EpisodeSummary:
        summary = summarize_episode(self.step_rows, self.rewards)
        summary["done_reason"] = self.done_reason
        summary["is_stalled"] = self.done_reason == "stalled"
        summary["is_overcrowded"] = self.done_reason == "overcrowded"
        return summary
