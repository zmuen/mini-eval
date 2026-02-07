from __future__ import annotations

import random
from typing import Any, List, Dict, Set

from config import passenger_color, passenger_size
from entity.passenger import Passenger
from entity.path import Path
from entity.station import Station
from geometry.type import ShapeType
from travel_plan import TravelPlan
from utils import get_shape_from_type
from graph.node import Node


def get_station_shape_types(env: Any) -> List[ShapeType]:
    station_shape_types: List[ShapeType] = []
    for station in env.stations:
        if station.shape.type not in station_shape_types:
            station_shape_types.append(station.shape.type)
    return station_shape_types


def is_passenger_spawn_time(env: Any) -> bool:
    return (
        env.steps == env.passenger_spawning_step
        or env.steps_since_last_spawn == env.passenger_spawning_interval_step
    )


def spawn_passengers(env: Any) -> None:
    for station in env.stations:
        station_types = get_station_shape_types(env)
        other_station_shape_types = [x for x in station_types if x != station.shape.type]
        destination_shape_type = random.choice(other_station_shape_types)
        destination_shape = get_shape_from_type(
            destination_shape_type, passenger_color, passenger_size
        )
        passenger = Passenger(destination_shape)
        if station.has_room():
            station.add_passenger(passenger)
            env.passengers.append(passenger)


def increment_time(env: Any, dt_ms: int) -> None:
    if env.is_paused:
        return

    env.time_ms += dt_ms
    env.steps += 1
    env.steps_since_last_spawn += 1

    for path in env.paths:
        for metro in path.metros:
            path.move_metro(metro, dt_ms)

    if is_passenger_spawn_time(env):
        spawn_passengers(env)
        env.steps_since_last_spawn = 0

    find_travel_plan_for_passengers(env)
    move_passengers(env)


def move_passengers(env: Any) -> None:
    for metro in env.metros:
        if metro.current_station:
            passengers_to_remove = []
            passengers_from_metro_to_station = []
            passengers_from_station_to_metro = []

            for passenger in metro.passengers:
                if metro.current_station.shape.type == passenger.destination_shape.type:
                    passengers_to_remove.append(passenger)
                elif env.travel_plans[passenger].get_next_station() == metro.current_station:
                    passengers_from_metro_to_station.append(passenger)
            for passenger in metro.current_station.passengers:
                if (
                    env.travel_plans[passenger].next_path
                    and env.travel_plans[passenger].next_path.id == metro.path_id  # type: ignore
                ):
                    passengers_from_station_to_metro.append(passenger)

            for passenger in passengers_to_remove:
                passenger.is_at_destination = True
                metro.remove_passenger(passenger)
                env.passengers.remove(passenger)
                del env.travel_plans[passenger]
                env.score += 1

            for passenger in passengers_from_metro_to_station:
                if metro.current_station.has_room():
                    metro.move_passenger(passenger, metro.current_station)
                    env.travel_plans[passenger].increment_next_station()
                    find_next_path_for_passenger_at_station(
                        env, passenger, metro.current_station
                    )

            for passenger in passengers_from_station_to_metro:
                if metro.has_room():
                    metro.current_station.move_passenger(passenger, metro)


def get_stations_for_shape_type(env: Any, shape_type: ShapeType) -> List[Station]:
    stations: List[Station] = []
    for station in env.stations:
        if station.shape.type == shape_type:
            stations.append(station)
    random.shuffle(stations)
    return stations


def find_shared_path(env: Any, station_a: Station, station_b: Station) -> Path | None:
    for path in env.paths:
        stations = path.stations
        if (station_a in stations) and (station_b in stations):
            return path
    return None


def passenger_has_travel_plan(env: Any, passenger: Passenger) -> bool:
    if passenger not in env.travel_plans:
        return False
    next_path = env.travel_plans[passenger].next_path
    return next_path is not None and next_path in env.paths


def find_next_path_for_passenger_at_station(
    env: Any, passenger: Passenger, station: Station
) -> None:
    next_station = env.travel_plans[passenger].get_next_station()
    if next_station is None:
        return
    next_path = find_shared_path(env, station, next_station)
    env.travel_plans[passenger].next_path = next_path


def skip_stations_on_same_path(node_path: List[Node]) -> List[Node]:
    if len(node_path) < 2:
        return node_path
    if len(node_path) == 2:
        return node_path

    nodes_to_remove = []
    i = 0
    j = 1
    path_set_list = [x.paths for x in node_path]
    path_set_list.append(set())
    while j <= len(path_set_list) - 1:
        set_a = path_set_list[i]
        set_b = path_set_list[j]
        if set_a & set_b:
            j += 1
        else:
            for k in range(i + 1, j - 1):
                nodes_to_remove.append(node_path[k])
            i = j - 1
            j += 1
    for node in nodes_to_remove:
        node_path.remove(node)
    return node_path

def build_station_nodes_dict(stations: List[Station], paths: List[Path]):
    station_nodes: List[Node] = []
    connections: List[List[Node]] = []
    station_nodes_dict: Dict[Station, Node] = {}

    for station in stations:
        node = Node(station)
        station_nodes.append(node)
        station_nodes_dict[station] = node
    for path in paths:
        if path.is_being_created:
            continue
        connection = []
        for station in path.stations:
            station_nodes_dict[station].paths.add(path)
            connection.append(station_nodes_dict[station])
        connections.append(connection)

    while len(station_nodes) > 0:
        root = station_nodes[0]
        for connection in connections:
            for idx in range(len(connection)):
                node = connection[idx]
                if node == root:
                    if idx - 1 >= 0:
                        root.neighbors.add(connection[idx - 1])
                    if idx + 1 <= len(connection) - 1:
                        root.neighbors.add(connection[idx + 1])
        station_nodes.remove(root)
        station_nodes_dict[root.station] = root

    return station_nodes_dict

def bfs(start: Node, end: Node) -> List[Node]:
    # Create a queue and enqueue the start node\
    queue = [(start, [start])]

    # While the queue is not empty
    while queue:
        # Dequeue the first node
        (node, path) = queue.pop(0)

        # If the node is the end node, return the path
        if node == end:
            return path

        # Enqueue the neighbors of the node
        for next in node.neighbors:
            if next not in path:
                queue.append((next, path + [next]))

    # If no path was found, return an empty list
    return []

def find_travel_plan_for_passengers(env: Any) -> None:
    station_nodes_dict = build_station_nodes_dict(env.stations, env.paths)
    for station in env.stations:
        for passenger in list(station.passengers):
            if not passenger_has_travel_plan(env, passenger):
                possible_dst_stations = get_stations_for_shape_type(
                    env, passenger.destination_shape.type
                )
                should_set_null_path = True
                for possible_dst_station in possible_dst_stations:
                    start = station_nodes_dict[station]
                    end = station_nodes_dict[possible_dst_station]
                    node_path = bfs(start, end)
                    if len(node_path) == 1:
                        station.remove_passenger(passenger)
                        env.passengers.remove(passenger)
                        passenger.is_at_destination = True
                        if passenger in env.travel_plans:
                            del env.travel_plans[passenger]
                        should_set_null_path = False
                        break
                    if len(node_path) > 1:
                        node_path = skip_stations_on_same_path(node_path)
                        env.travel_plans[passenger] = TravelPlan(node_path[1:])
                        find_next_path_for_passenger_at_station(env, passenger, station)
                        should_set_null_path = False
                        break
                if should_set_null_path:
                    env.travel_plans[passenger] = TravelPlan([])