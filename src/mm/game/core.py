from __future__ import annotations

import random
from typing import Any, Dict, List, Mapping

import numpy as np

from config import (
    framerate,
    num_metros,
    num_paths,
    num_stations,
    passenger_spawning_interval_step,
    passenger_spawning_start_step,
)
import dynamics
from entity.get_entity import get_random_stations
from entity.metro import Metro
from entity.passenger import Passenger
from entity.path import Path
from entity.station import Station
from geometry.type import ShapeType
from graph.node import Node
from travel_plan import TravelPlan
from type import Color
from utils import hue_to_rgb

TravelPlans = Dict[Passenger, TravelPlan]
Action = Mapping[str, Any]
Observation = Dict[str, Any]


class ActionError(ValueError):
    pass


class MiniMetroEnv:
    """
    Headless simulation environment for evaluation.

    It mirrors the game logic from `Mediator` but removes UI and event handling.
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        num_paths_override: int | None = None,
        num_metros_override: int | None = None,
        num_stations_override: int | None = None,
        dt_ms: int | None = None,
    ) -> None:
        self.default_dt_ms = dt_ms if dt_ms is not None else int(1000 / framerate)
        self.num_paths = num_paths_override or num_paths
        self.num_metros = num_metros_override or num_metros
        self.num_stations = num_stations_override or num_stations
        self.seed = seed

        self.passenger_spawning_step = passenger_spawning_start_step
        self.passenger_spawning_interval_step = passenger_spawning_interval_step

        self._init_color_pool()
        self.reset(seed=seed)

    def _init_color_pool(self) -> None:
        self.path_colors: Dict[Color, bool] = {}
        for i in range(self.num_paths):
            color = hue_to_rgb(i / (self.num_paths + 1))
            self.path_colors[color] = False

    def reset(
        self,
        *,
        seed: int | None = None,
        stations: List[Station] | None = None,
    ) -> Observation:
        if seed is not None:
            self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        self.stations = stations if stations is not None else get_random_stations(self.num_stations)
        self.metros: List[Metro] = []
        self.paths: List[Path] = []
        self.passengers: List[Passenger] = []

        self.path_to_color: Dict[Path, Color] = {}
        self.travel_plans: TravelPlans = {}
        self.path_being_created: Path | None = None

        self.time_ms = 0
        self.steps = 0
        self.steps_since_last_spawn = self.passenger_spawning_interval_step + 1
        self.is_paused = False
        self.score = 0
        self.terminated = False

        self._init_color_pool()
        return self.observe()

    def step(
        self,
        action: Action | None = None,
        *,
        dt_ms: int | None = None,
    ) -> tuple[Observation, float, bool, bool, Dict[str, Any]]:
        if action:
            self.apply_action(action)

        score_before = self.score
        self.increment_time(self.default_dt_ms if dt_ms is None else dt_ms)
        reward = float(self.score - score_before)

        info = {
            "score": self.score,
            "time_ms": self.time_ms,
            "steps": self.steps,
            "num_passengers": len(self.passengers),
            "num_paths": len(self.paths),
            "num_metros": len(self.metros),
        }
        return self.observe(), reward, self.terminated, False, info

    def apply_action(self, action: Action) -> None:
        action_type = action.get("type")
        if action_type == "noop":
            return
        if action_type == "toggle_pause":
            self.is_paused = not self.is_paused
            return

        if action_type == "start_path":
            station = self.get_station_by_id(self._required_str(action, "station_id"))
            self.start_path_on_station(station)
            return
        if action_type == "add_station":
            station = self.get_station_by_id(self._required_str(action, "station_id"))
            self.add_station_to_path(station)
            return
        if action_type == "end_path":
            station = self.get_station_by_id(self._required_str(action, "station_id"))
            self.end_path_on_station(station)
            return
        if action_type == "abort_path":
            self.abort_path_creation()
            return
        if action_type == "remove_path":
            path = self.get_path_by_id(self._required_str(action, "path_id"))
            self.remove_path(path)
            return

        raise ActionError(f"Unknown action type: {action_type}")

    def observe(self) -> Observation:
        station_rows = []
        for station in self.stations:
            station_rows.append(
                {
                    "id": station.id,
                    "shape_type": station.shape.type.value,
                    "position": {"x": station.position.left, "y": station.position.top},
                    "queue_size": len(station.passengers),
                    "capacity": station.capacity,
                }
            )

        path_rows = []
        for path in self.paths:
            path_rows.append(
                {
                    "id": path.id,
                    "is_looped": path.is_looped,
                    "is_being_created": path.is_being_created,
                    "station_ids": [station.id for station in path.stations],
                    "metro_ids": [metro.id for metro in path.metros],
                }
            )

        metro_rows = []
        for metro in self.metros:
            metro_rows.append(
                {
                    "id": metro.id,
                    "path_id": metro.path_id,
                    "passenger_count": len(metro.passengers),
                    "capacity": metro.capacity,
                    "current_station_id": metro.current_station.id if metro.current_station else None,
                    "position": {"x": metro.position.left, "y": metro.position.top},
                }
            )

        return {
            "time_ms": self.time_ms,
            "steps": self.steps,
            "score": self.score,
            "is_paused": self.is_paused,
            "is_creating_path": self.path_being_created is not None,
            "stations": station_rows,
            "paths": path_rows,
            "metros": metro_rows,
            "num_passengers": len(self.passengers),
        }

    def _required_str(self, action: Mapping[str, Any], key: str) -> str:
        value = action.get(key)
        if not isinstance(value, str):
            raise ActionError(f"Action `{action.get('type')}` requires string `{key}`")
        return value

    def get_station_by_id(self, station_id: str) -> Station:
        for station in self.stations:
            if station.id == station_id:
                return station
        raise ActionError(f"Unknown station_id: {station_id}")

    def get_path_by_id(self, path_id: str) -> Path:
        for path in self.paths:
            if path.id == path_id:
                return path
        raise ActionError(f"Unknown path_id: {path_id}")

    def remove_path(self, path: Path):
        for metro in path.metros:
            for passenger in list(metro.passengers):
                if passenger in self.passengers:
                    self.passengers.remove(passenger)
            if metro in self.metros:
                self.metros.remove(metro)
        self.release_color_for_path(path)
        self.paths.remove(path)
        self.find_travel_plan_for_passengers()

    def start_path_on_station(self, station: Station) -> None:
        if self.path_being_created is not None:
            raise ActionError("A path is already being created")
        if len(self.paths) >= self.num_paths:
            raise ActionError("Path limit reached")

        assigned_color = None
        for path_color, taken in self.path_colors.items():
            if not taken:
                assigned_color = path_color
                self.path_colors[path_color] = True
                break
        if assigned_color is None:
            raise ActionError("No available path color")

        path = Path(assigned_color)
        self.path_to_color[path] = assigned_color
        path.add_station(station)
        path.is_being_created = True
        self.path_being_created = path
        self.paths.append(path)

    def add_station_to_path(self, station: Station) -> None:
        if self.path_being_created is None:
            raise ActionError("No path is being created")

        if self.path_being_created.stations[-1] == station:
            return
        if (
            len(self.path_being_created.stations) > 1
            and self.path_being_created.stations[0] == station
        ):
            self.path_being_created.set_loop()
        elif self.path_being_created.stations[0] != station:
            if self.path_being_created.is_looped:
                self.path_being_created.remove_loop()
            self.path_being_created.add_station(station)

    def abort_path_creation(self) -> None:
        if self.path_being_created is None:
            raise ActionError("No path is being created")
        self.release_color_for_path(self.path_being_created)
        self.paths.remove(self.path_being_created)
        self.path_being_created = None

    def release_color_for_path(self, path: Path) -> None:
        self.path_colors[path.color] = False
        if path in self.path_to_color:
            del self.path_to_color[path]

    def finish_path_creation(self) -> None:
        if self.path_being_created is None:
            raise ActionError("No path is being created")
        self.path_being_created.is_being_created = False
        self.path_being_created.remove_temporary_point()
        if len(self.metros) < self.num_metros:
            metro = Metro()
            self.path_being_created.add_metro(metro)
            self.metros.append(metro)
        self.path_being_created = None

    def end_path_on_station(self, station: Station) -> None:
        if self.path_being_created is None:
            raise ActionError("No path is being created")
        if (
            len(self.path_being_created.stations) > 1
            and self.path_being_created.stations[-1] == station
        ):
            self.finish_path_creation()
        elif (
            len(self.path_being_created.stations) > 1
            and self.path_being_created.stations[0] == station
        ):
            self.path_being_created.set_loop()
            self.finish_path_creation()
        elif self.path_being_created.stations[0] != station:
            self.path_being_created.add_station(station)
            self.finish_path_creation()
        else:
            self.abort_path_creation()

    def get_station_shape_types(self) -> List[ShapeType]:
        return dynamics.get_station_shape_types(self)

    def is_passenger_spawn_time(self) -> bool:
        return dynamics.is_passenger_spawn_time(self)

    def spawn_passengers(self) -> None:
        dynamics.spawn_passengers(self)

    def increment_time(self, dt_ms: int) -> None:
        dynamics.increment_time(self, dt_ms)

    def move_passengers(self) -> None:
        dynamics.move_passengers(self)

    def get_stations_for_shape_type(self, shape_type: ShapeType):
        return dynamics.get_stations_for_shape_type(self, shape_type)

    def find_shared_path(self, station_a: Station, station_b: Station) -> Path | None:
        return dynamics.find_shared_path(self, station_a, station_b)

    def passenger_has_travel_plan(self, passenger: Passenger) -> bool:
        return dynamics.passenger_has_travel_plan(self, passenger)

    def find_next_path_for_passenger_at_station(self, passenger: Passenger, station: Station):
        dynamics.find_next_path_for_passenger_at_station(self, passenger, station)

    def skip_stations_on_same_path(self, node_path: List[Node]):
        return dynamics.skip_stations_on_same_path(node_path)

    def find_travel_plan_for_passengers(self) -> None:
        dynamics.find_travel_plan_for_passengers(self)
