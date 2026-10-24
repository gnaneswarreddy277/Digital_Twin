"""Shared graph derived from OSM road nodes, ways, features and parking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from osm_campus import campus_places, haversine_meters


@dataclass(frozen=True)
class Edge:
    destination: str
    cost: float
    road_id: str
    road_name: str
    kind: str


class CampusGraph:
    def __init__(self) -> None:
        self.positions: dict[str, tuple[float, float]] = {}
        self.labels: dict[str, str] = {}
        self.adjacency: dict[str, list[Edge]] = {}
        self.road_nodes: set[str] = set()
        self.parking_status: dict[str, bool] = {}
        self.places: dict[str, dict[str, Any]] = {}
        self.road_ways: dict[str, dict[str, Any]] = {}
        self.building_ways: list[dict[str, Any]] = []

    def add_node(self, node: str, label: str, position: tuple[float, float]) -> None:
        self.positions[node] = position
        self.labels[node] = label
        self.adjacency.setdefault(node, [])

    def add_edge(
        self,
        first: str,
        second: str,
        road_id: str,
        road_name: str,
        kind: str = "osm",
        cost: float | None = None,
        bidirectional: bool = True,
    ) -> None:
        distance = cost if cost is not None else haversine_meters(self.positions[first], self.positions[second])
        self.adjacency[first].append(Edge(second, distance, road_id, road_name, kind))
        if bidirectional:
            self.adjacency[second].append(Edge(first, distance, road_id, road_name, kind))

    def add_access_edge(self, source: str, target: str, label: str, kind: str) -> None:
        distance = haversine_meters(self.positions[source], self.positions[target])
        first = Edge(target, distance, f"access/{source}", label, kind)
        second = Edge(source, distance, f"access/{source}", label, kind)
        if not any(edge.destination == target for edge in self.adjacency[source]):
            self.adjacency[source].append(first)
        if not any(edge.destination == source for edge in self.adjacency[target]):
            self.adjacency[target].append(second)

    def neighbors(self, node: str, blocked_roads: Iterable[str] = ()) -> list[Edge]:
        blocked = set(blocked_roads)
        return [edge for edge in self.adjacency.get(node, []) if edge.road_id not in blocked]

    def nearest_road_node(self, position: tuple[float, float]) -> str:
        road_nodes = list(self.road_nodes)
        if not road_nodes:
            raise ValueError("The OSM extract contains no usable campus road nodes.")
        return min(road_nodes, key=lambda node: haversine_meters(position, self.positions[node]))

    def edge_between(self, first: str, second: str) -> Edge | None:
        return next((edge for edge in self.adjacency.get(first, []) if edge.destination == second), None)


def build_campus_graph(osm_data: dict[str, Any], parking: list[dict[str, Any]]) -> CampusGraph:
    """Construct routing topology from actual OSM ways; only access links are simulated."""
    graph = CampusGraph()
    graph.building_ways = [
        way for way in osm_data["buildings"]
        if len(way.get("geometry", [])) >= 4
    ]

    for way in osm_data["roads"]:
        geometry = way.get("geometry", [])
        osm_nodes = way.get("nodes", [])
        if len(geometry) < 2 or len(osm_nodes) != len(geometry):
            continue
        tags = way.get("tags", {})
        way_id = f"way/{way['id']}"
        highway = str(tags.get("highway", "road"))
        display_name = str(tags.get("name") or f"Mapped {highway.replace('_', ' ')} path")
        oneway = str(tags.get("oneway", "")).casefold()
        graph.road_ways[way_id] = way
        for node_id, point in zip(osm_nodes, geometry):
            node_key = f"node/{node_id}"
            position = (float(point["lat"]), float(point["lon"]))
            graph.add_node(node_key, f"OSM junction {node_id}", position)
            graph.road_nodes.add(node_key)
        for first_id, second_id in zip(osm_nodes, osm_nodes[1:]):
            first, second = f"node/{first_id}", f"node/{second_id}"
            if first != second:
                if oneway == "-1":
                    graph.add_edge(second, first, way_id, display_name, "osm", bidirectional=False)
                else:
                    graph.add_edge(first, second, way_id, display_name, "osm", bidirectional=oneway not in {"yes", "1", "true"})

    for place in campus_places(osm_data):
        key = str(place["key"])
        graph.places[key] = place
        if key not in graph.positions:
            graph.add_node(key, str(place["label"]), (float(place["lat"]), float(place["lon"])))
        nearest = graph.nearest_road_node(graph.positions[key])
        if nearest != key:
            graph.add_access_edge(key, nearest, f"Mapped feature access · {place['label']}", "access")

    for slot in parking:
        key = str(slot["node"])
        graph.add_node(key, f"{slot['id']} · {slot['area']}", (float(slot["lat"]), float(slot["lon"])))
        anchor = str(slot.get("anchor_key", ""))
        nearest = anchor if anchor in graph.places else graph.nearest_road_node(graph.positions[key])
        graph.add_access_edge(key, nearest, f"Simulated slot access · {slot['id']}", "simulated_access")
        graph.parking_status[key] = not bool(slot["occupied"])
    return graph
