"""Breadth-first search for equal-cost campus paths and nearest parking."""

from __future__ import annotations

from collections import deque
from time import perf_counter
from typing import Callable, Iterable

from algorithms.graph import CampusGraph


def bfs_path(
    graph: CampusGraph,
    start: str,
    destination: str,
    blocked_roads: Iterable[str] = (),
) -> dict[str, object] | None:
    """Return an equal-edge-cost shortest path and traversal order."""
    if start not in graph.adjacency or destination not in graph.adjacency:
        return None

    started = perf_counter()
    queue = deque([start])
    previous: dict[str, str | None] = {start: None}
    explored: list[str] = []
    blocked = set(blocked_roads)

    while queue:
        current = queue.popleft()
        explored.append(current)
        if current == destination:
            break
        for edge in graph.neighbors(current, blocked):
            if edge.destination not in previous:
                previous[edge.destination] = current
                queue.append(edge.destination)

    if destination not in previous:
        return None
    path = _reconstruct_path(previous, destination)
    return {
        "path": path,
        "explored": explored,
        "cost": float(len(path) - 1),
        "path_length": len(path) - 1,
        "distance_meters": _path_distance(graph, path),
        "execution_ms": (perf_counter() - started) * 1000,
        "current_node": explored[-1],
    }


def bfs_nearest(
    graph: CampusGraph,
    start: str,
    is_target: Callable[[str], bool],
    blocked_roads: Iterable[str] = (),
) -> dict[str, object] | None:
    """Find the closest reachable target by number of graph edges."""
    if start not in graph.adjacency:
        return None

    started = perf_counter()
    queue = deque([start])
    previous: dict[str, str | None] = {start: None}
    explored: list[str] = []
    blocked = set(blocked_roads)

    while queue:
        current = queue.popleft()
        explored.append(current)
        if is_target(current):
            path = _reconstruct_path(previous, current)
            return {
                "path": path,
                "target": current,
                "explored": explored,
                "cost": float(len(path) - 1),
                "path_length": len(path) - 1,
                "distance_meters": _path_distance(graph, path),
                "execution_ms": (perf_counter() - started) * 1000,
                "current_node": current,
            }
        for edge in graph.neighbors(current, blocked):
            if edge.destination not in previous:
                previous[edge.destination] = current
                queue.append(edge.destination)
    return None


def _reconstruct_path(previous: dict[str, str | None], destination: str) -> list[str]:
    path = []
    current: str | None = destination
    while current is not None:
        path.append(current)
        current = previous[current]
    return list(reversed(path))


def _path_distance(graph: CampusGraph, path: list[str]) -> float:
    distance = 0.0
    for first, second in zip(path, path[1:]):
        edge = graph.edge_between(first, second)
        if edge is not None:
            distance += edge.cost
    return distance
