"""Software simulation entities positioned on real OSM campus geometry."""

from __future__ import annotations

import random
import math
from datetime import datetime, timedelta
from typing import Any

from algorithms.astar import astar_path
from algorithms.bfs import bfs_nearest
from algorithms.graph import build_campus_graph
from ai_model import predict_parking
from osm_campus import campus_parking_anchors, campus_places, haversine_meters


PARKING_SLOTS_PER_AREA = 10
SIMULATED_MEETING_START = (14, 0)
SIMULATED_MEETING_END = (14, 45)


def _make_slots(osm_data: dict[str, Any]) -> list[dict[str, Any]]:
    anchors = campus_parking_anchors(osm_data)
    if not anchors:
        raise ValueError("No OSM-tagged campus parking locations are present in this map extract.")
    slots: list[dict[str, Any]] = []
    for area_index, anchor in enumerate(anchors):
        for offset in range(PARKING_SLOTS_PER_AREA):
            row, column = divmod(offset, 5)
            east_meters = (column - 2) * 4.5
            north_meters = (row - 0.5) * 5.0
            latitude = float(anchor["lat"]) + north_meters / 111_320
            longitude = float(anchor["lon"]) + east_meters / (
                111_320 * max(0.1, math.cos(math.radians(float(anchor["lat"]))))
            )
            slots.append({
                "id": f"P{area_index * PARKING_SLOTS_PER_AREA + offset + 1:02d}",
                "area": str(anchor["label"]),
                "anchor_key": str(anchor["key"]),
                "lat": latitude,
                "lon": longitude,
                "node": f"parking/P{area_index * PARKING_SLOTS_PER_AREA + offset + 1:02d}",
                "occupied": False,
                "vehicle_id": None,
                "entry_time": None,
                "exit_time": None,
            })
            if len(slots) >= 20:
                return slots
    return slots


class OSMCampusTwin:
    """Live-looking simulation whose every position is anchored to OSM geometry."""

    def __init__(self, osm_data: dict[str, Any], seed: int | None = None) -> None:
        self.osm_data = osm_data
        self.rng = random.Random(seed)
        self.places = campus_places(osm_data)
        self.place_by_key = {place["key"]: place for place in self.places}
        self.parking = _make_slots(osm_data)
        self.graph = build_campus_graph(osm_data, self.parking)
        self.blocked_roads: set[str] = set()
        self.current_time = datetime.now().replace(hour=13, minute=50, second=0, microsecond=0)
        self.minute_remainder = 0.0
        self.running = True
        self._vehicle_number = 1
        self._meeting_override: bool | None = None
        self.events: list[dict[str, str]] = []
        self.history: list[dict[str, Any]] = []
        self.road_node_keys = sorted(self.graph.road_nodes)
        self.main_gate = self._find_place(("main gate", "gate"))
        self.meeting_place = self._find_place(("amriteshwari hall", "auditorium"))
        self.people = self._create_people()
        self.vehicles: list[dict[str, Any]] = []
        self._initialize_parking()
        self._create_gate_vehicle()
        self.record_event("OSM campus geometry loaded; all entity coordinates are simulated", "system")

    def _find_place(self, name_fragments: tuple[str, ...]) -> dict[str, Any] | None:
        for fragment in name_fragments:
            found = next((place for place in self.places if fragment in place["label"].casefold()), None)
            if found:
                return found
        return None

    @property
    def main_gate_key(self) -> str:
        return str(self.main_gate["key"] if self.main_gate else self.road_node_keys[0])

    def _initialize_parking(self) -> None:
        occupied_count = min(14, max(0, len(self.parking) - 1))
        for slot in self.parking[:occupied_count]:
            self._vehicle_number += 1
            slot["occupied"] = True
            slot["vehicle_id"] = f"V{self._vehicle_number:03d}"
            slot["entry_time"] = self.current_time.strftime("%H:%M")
            self.graph.parking_status[str(slot["node"])] = False

    def _create_gate_vehicle(self) -> None:
        current_node = self.main_gate_key
        self.vehicles.append({
            "id": "V001", "node": current_node, "status": "at_gate",
            "destination_slot": None, "path": [], "path_index": 0,
        })

    def _create_people(self) -> list[dict[str, Any]]:
        destinations = [place["key"] for place in self.places]
        people = []
        for index in range(1, 41):
            role = "Student" if index <= 32 else "Teacher"
            people.append({
                "id": f"S{index:03d}" if role == "Student" else f"T{index - 32:03d}",
                "role": role,
                "node": self.rng.choice(self.road_node_keys),
                "destination": self.rng.choice(destinations) if destinations else self.road_node_keys[0],
                "meeting_attendee": index <= 8,
                "next_move": self.rng.randint(2, 6),
            })
        return people

    def meeting_active(self) -> bool:
        if self._meeting_override is not None:
            return self._meeting_override
        minute = self.current_time.hour * 60 + self.current_time.minute
        return SIMULATED_MEETING_START[0] * 60 <= minute < SIMULATED_MEETING_END[0] * 60 + SIMULATED_MEETING_END[1]

    def record_event(self, message: str, kind: str = "movement") -> None:
        self.events.insert(0, {"time": self.current_time.strftime("%H:%M"), "message": message, "kind": kind})
        del self.events[30:]

    def start(self) -> None:
        self.running = True
        self.record_event("Simulation started", "system")

    def pause(self) -> None:
        self.running = False
        self.record_event("Simulation paused", "system")

    def toggle_road(self, road_id: str) -> bool:
        if road_id not in self.graph.road_ways:
            raise ValueError("Select a real OSM road way.")
        if road_id in self.blocked_roads:
            self.blocked_roads.remove(road_id)
            is_blocked = False
        else:
            self.blocked_roads.add(road_id)
            is_blocked = True
        road_name = str(self.graph.road_ways[road_id].get("tags", {}).get("name") or road_id)
        self.record_event(f"OSM road {road_name} {'blocked' if is_blocked else 'reopened'} in simulation", "road")
        return is_blocked

    def add_vehicle(self, area: str | None = None) -> str:
        candidate = next((slot for slot in self.parking if not slot["occupied"] and (area is None or slot["area"] == area)), None)
        if candidate is None:
            raise ValueError("No simulated parking slots are available at that mapped OSM parking location.")
        self._vehicle_number += 1
        vehicle_id = f"V{self._vehicle_number:03d}"
        candidate.update({"occupied": True, "vehicle_id": vehicle_id, "entry_time": self.current_time.strftime("%H:%M"), "exit_time": None})
        self.graph.parking_status[str(candidate["node"])] = False
        self.record_event(f"{vehicle_id} occupied simulated slot {candidate['id']}", "parking")
        return vehicle_id

    def remove_vehicle(self, vehicle_id: str | None = None) -> str:
        occupied = [slot for slot in self.parking if slot["occupied"]]
        slot = next((item for item in occupied if item["vehicle_id"] == vehicle_id), None) if vehicle_id else (occupied[0] if occupied else None)
        if slot is None:
            raise ValueError("No parked simulated vehicle was found.")
        removed = str(slot["vehicle_id"])
        slot.update({"occupied": False, "vehicle_id": None, "entry_time": None, "exit_time": self.current_time.strftime("%H:%M")})
        self.graph.parking_status[str(slot["node"])] = True
        self.record_event(f"{removed} left simulated slot {slot['id']}", "parking")
        return removed

    def start_meeting(self) -> None:
        self._meeting_override = True
        if self.meeting_place:
            node = min(self.road_node_keys, key=lambda key: haversine_meters(self.graph.positions[key], (self.meeting_place["lat"], self.meeting_place["lon"])))
            for person in self.people:
                if person["meeting_attendee"]:
                    person["node"] = node
        self.record_event("Simulated meeting started at " + (self.meeting_place["label"] if self.meeting_place else "unmapped venue"), "meeting")

    def end_meeting(self) -> None:
        self._meeting_override = False
        self.record_event("Simulated meeting ended", "meeting")

    def move_person(self, person_id: str, destination_key: str) -> None:
        person = next((item for item in self.people if item["id"] == person_id), None)
        if person is None or destination_key not in self.graph.places:
            raise ValueError("Choose a simulated person and an OSM-mapped destination.")
        person["destination"] = destination_key
        self.record_event(f"{person_id} assigned to mapped destination {self.graph.labels[destination_key]}")

    def _route_gate_vehicle(self, vehicle: dict[str, Any]) -> None:
        if vehicle["node"] not in self.graph.adjacency:
            return
        result = bfs_nearest(
            self.graph,
            str(vehicle["node"]),
            lambda node: self.graph.parking_status.get(node, False),
            self.blocked_roads,
        )
        if result:
            vehicle["path"] = result["path"]
            vehicle["path_index"] = 0
            vehicle["destination_slot"] = str(result["target"])
            vehicle["status"] = "moving"
        else:
            vehicle["path"] = []
            vehicle["destination_slot"] = None
            vehicle["status"] = "waiting"

    def _advance_entities(self) -> None:
        self.current_time += timedelta(minutes=1)
        if self.meeting_active() and self.meeting_place:
            meeting_node = min(self.road_node_keys, key=lambda key: haversine_meters(self.graph.positions[key], (self.meeting_place["lat"], self.meeting_place["lon"])))
            for person in self.people:
                if person["meeting_attendee"]:
                    person["node"] = meeting_node
                    continue
        for person in self.people:
            person["next_move"] -= 1
            if person["next_move"] > 0:
                continue
            path_result = astar_path(self.graph, str(person["node"]), str(person["destination"]), self.blocked_roads)
            if path_result and len(path_result["path"]) > 1:
                person["node"] = path_result["path"][1]
            else:
                options = self.graph.neighbors(str(person["node"]), self.blocked_roads)
                if options:
                    edge = self.rng.choice(options)
                    person["node"] = edge.destination
            person["next_move"] = self.rng.randint(2, 6)

        for vehicle in self.vehicles:
            if vehicle["status"] == "at_gate":
                self._route_gate_vehicle(vehicle)
            if vehicle["status"] != "moving":
                continue
            target = str(vehicle["destination_slot"])
            path = vehicle["path"]
            remaining_edges = zip(path[int(vehicle["path_index"]):], path[int(vehicle["path_index"]) + 1:])
            if any(
                (edge := self.graph.edge_between(first, second)) is not None
                and edge.road_id in self.blocked_roads
                for first, second in remaining_edges
            ):
                self._route_gate_vehicle(vehicle)
            if vehicle["status"] != "moving":
                continue
            target = str(vehicle["destination_slot"])
            path = vehicle["path"]
            if vehicle["path_index"] + 1 < len(path):
                vehicle["path_index"] += 1
                vehicle["node"] = path[vehicle["path_index"]]
            elif target.startswith("parking/"):
                slot = next((item for item in self.parking if item["node"] == target), None)
                if slot and not slot["occupied"]:
                    slot.update({"occupied": True, "vehicle_id": vehicle["id"], "entry_time": self.current_time.strftime("%H:%M")})
                    self.graph.parking_status[target] = False
                    vehicle["status"] = "parked"
                    self.record_event(f"{vehicle['id']} reached simulated slot {slot['id']}", "parking")
                else:
                    vehicle["status"] = "waiting"

        for slot in self.parking:
            if slot["occupied"] and slot["vehicle_id"] != "V001" and self.rng.random() < 0.012:
                vehicle_id = slot["vehicle_id"]
                slot.update({"occupied": False, "vehicle_id": None, "entry_time": None, "exit_time": self.current_time.strftime("%H:%M")})
                self.graph.parking_status[slot["node"]] = True
                self.record_event(f"{vehicle_id} departed simulated slot {slot['id']}", "parking")
        if self.rng.random() < 0.035 and any(not slot["occupied"] for slot in self.parking):
            self._vehicle_number += 1
            self.vehicles.append({"id": f"V{self._vehicle_number:03d}", "node": self.main_gate_key, "status": "at_gate", "destination_slot": None, "path": [], "path_index": 0})

    def advance(self, elapsed_seconds: float, speed: int = 30) -> int:
        if not self.running:
            return 0
        self.minute_remainder += max(0.0, elapsed_seconds) * speed / 60
        ticks = int(self.minute_remainder)
        self.minute_remainder -= ticks
        for _ in range(min(ticks, 60)):
            self._advance_entities()
            self.history.append({
                "timestamp": self.current_time.isoformat(timespec="minutes"),
                "occupied": sum(bool(slot["occupied"]) for slot in self.parking),
                "available": sum(not bool(slot["occupied"]) for slot in self.parking),
            })
        del self.history[:-240]
        return ticks

    def _closest_place_label(self, node: str) -> str:
        coordinate = self.graph.positions[node]
        if not self.places:
            return "OSM road node"
        closest = min(self.places, key=lambda place: haversine_meters(coordinate, (place["lat"], place["lon"])))
        return str(closest["label"])

    def snapshot(self) -> dict[str, Any]:
        occupied = sum(bool(slot["occupied"]) for slot in self.parking)
        people = []
        place_counts: dict[str, int] = {}
        for person in self.people:
            lat, lon = self.graph.positions[str(person["node"])]
            current_place = self._closest_place_label(str(person["node"]))
            destination = self.graph.labels.get(str(person["destination"]), str(person["destination"]))
            place_counts[current_place] = place_counts.get(current_place, 0) + 1
            people.append({**person, "lat": lat, "lon": lon, "location": current_place, "destination_label": destination})
        vehicles = []
        for vehicle in self.vehicles:
            node = str(vehicle["node"])
            lat, lon = self.graph.positions.get(node, (self.main_gate["lat"], self.main_gate["lon"]))
            vehicles.append({**vehicle, "lat": lat, "lon": lon})
        return {
            "timestamp": self.current_time.isoformat(timespec="minutes"),
            "parking": self.parking,
            "occupied": occupied,
            "available": len(self.parking) - occupied,
            "capacity": len(self.parking),
            "people": people,
            "people_count": len(people),
            "vehicles": vehicles,
            "place_counts": place_counts,
            "meeting_active": self.meeting_active(),
            "meeting_name": "Faculty Meeting",
            "meeting_place": self.meeting_place,
            "meeting_attendees": 8 if self.meeting_active() and self.meeting_place else 0,
            "blocked_roads": sorted(self.blocked_roads),
            "events": self.events,
            "running": self.running,
        }

    def forecast(self, horizon_minutes: int = 10) -> dict[str, int]:
        state = self.snapshot()
        return predict_parking(
            int(state["occupied"]),
            int(state["people_count"]),
            self.current_time,
            bool(state["meeting_active"]),
            horizon_minutes,
        )
