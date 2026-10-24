"""A* campus navigation using Euclidean coordinate heuristic."""

from __future__ import annotations

import heapq
from itertools import count
from time import perf_counter
from typing import Iterable

from algorithms.graph import CampusGraph
from osm_campus import haversine_meters


def astar_path(
    graph: CampusGraph,
    start: str,
    destination: str,
    blocked_roads: Iterable[str] = (),
) -> dict[str, object] | None:
    """Find a least-distance route with f(n) = g(n) + h(n)."""
    if start not in graph.adjacency or destination not in graph.adjacency:
        return None

    started = perf_counter()
    def heuristic(node: str) -> float:
        return haversine_meters(graph.positions[node], graph.positions[destination])

    serial = count()
    open_heap = [(heuristic(start), next(serial), start)]
    came_from: dict[str, str | None] = {start: None}
    g_score = {start: 0.0}
    explored: list[str] = []
    closed: set[str] = set()
    blocked = set(blocked_roads)

    while open_heap:
        current = heapq.heappop(open_heap)[2]
        if current in closed:
            continue
        closed.add(current)
        explored.append(current)
        if current == destination:
            path = _reconstruct_path(came_from, destination)
            return {
                "path": path,
                "explored": explored,
                "cost": g_score[destination],
                "path_length": len(path) - 1,
                "current_node": current,
                "execution_ms": (perf_counter() - started) * 1000,
            }

        for edge in graph.neighbors(current, blocked):
            candidate = g_score[current] + edge.cost
            if candidate >= g_score.get(edge.destination, float("inf")):
                continue
            came_from[edge.destination] = current
            g_score[edge.destination] = candidate
            heapq.heappush(
                open_heap,
                (candidate + heuristic(edge.destination), next(serial), edge.destination),
            )
    return None


def _reconstruct_path(came_from: dict[str, str | None], destination: str) -> list[str]:
    path = []
    current: str | None = destination
    while current is not None:
        path.append(current)
        current = came_from[current]
    return list(reversed(path))
