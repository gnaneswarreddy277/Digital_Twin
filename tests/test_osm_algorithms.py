import unittest

from algorithms.astar import astar_path
from algorithms.bfs import bfs_nearest
from algorithms.graph import build_campus_graph
from osm_campus import campus_parking_anchors
from osm_simulation import PARKING_SLOTS_PER_AREA, OSMCampusTwin
from osm_campus import OSM_BOUNDS


def make_osm_fixture():
    return {
        "source": "OpenStreetMap Overpass API",
        "attribution": "© OpenStreetMap contributors",
        "campus_osm_way_id": 128964693,
        "bounds": list(OSM_BOUNDS),
        "center": [10.901, 76.900],
        "fetched_at": "2026-09-27T12:00:00Z",
        "roads": [
            {"type": "way", "id": 9001, "nodes": [1, 2, 3],
             "tags": {"highway": "residential", "name": "OSM north road", "oneway": "yes"},
             "geometry": [{"lat": 10.900, "lon": 76.900}, {"lat": 10.901, "lon": 76.900}, {"lat": 10.902, "lon": 76.900}]},
            {"type": "way", "id": 9002, "nodes": [2, 4],
             "tags": {"highway": "footway", "name": "OSM east path"},
             "geometry": [{"lat": 10.901, "lon": 76.900}, {"lat": 10.901, "lon": 76.901}]},
            {"type": "way", "id": 9003, "nodes": [1, 5, 4],
             "tags": {"highway": "residential", "name": "OSM south alternative"},
             "geometry": [{"lat": 10.900, "lon": 76.900}, {"lat": 10.899, "lon": 76.901}, {"lat": 10.901, "lon": 76.901}]},
        ],
        "buildings": [{"type": "way", "id": 9100, "tags": {"building": "library", "name": "OSM Library"},
                       "geometry": [{"lat": 10.901, "lon": 76.901}, {"lat": 10.9011, "lon": 76.901}, {"lat": 10.9011, "lon": 76.9011}, {"lat": 10.901, "lon": 76.9011}]}],
        "features": [
            {"type": "node", "id": 9200, "lat": 10.900, "lon": 76.900, "tags": {"name": "OSM Main Gate", "entrance": "main"}},
            {"type": "node", "id": 9201, "lat": 10.902, "lon": 76.900, "tags": {"amenity": "parking"}},
            {"type": "node", "id": 9300, "lat": 10.901, "lon": 76.901, "tags": {"name": "OSM Library"}},
        ],
    }


class OSMGraphAlgorithmTests(unittest.TestCase):
    def setUp(self):
        self.osm = make_osm_fixture()
        self.parking = [
            {"id": "P1", "node": "parking/P1", "area": "node/9201", "anchor_key": "node/9201",
             "lat": 10.902, "lon": 76.9, "occupied": True, "vehicle_id": "V001"},
            {"id": "P2", "node": "parking/P2", "area": "node/9201", "anchor_key": "node/9201",
             "lat": 10.9021, "lon": 76.9, "occupied": False, "vehicle_id": None},
        ]
        self.graph = build_campus_graph(self.osm, self.parking)

    def test_graph_edges_reference_osm_way_ids_and_meter_costs(self):
        edge = self.graph.edge_between("node/1", "node/2")
        self.assertIsNotNone(edge)
        self.assertEqual(edge.road_id, "way/9001")
        self.assertEqual(edge.kind, "osm")
        self.assertGreater(edge.cost, 100)
        self.assertIsNone(self.graph.edge_between("node/2", "node/1"))
        self.assertEqual(set(self.graph.parking_status), {"parking/P1", "parking/P2"})

    def test_bfs_finds_nearest_available_real_anchor_slot_and_reacts_to_occupancy(self):
        result = bfs_nearest(self.graph, "node/1", lambda node: self.graph.parking_status.get(node, False))
        self.assertIsNotNone(result)
        self.assertEqual(result["target"], "parking/P2")
        self.graph.parking_status["parking/P2"] = False
        self.assertIsNone(bfs_nearest(self.graph, "node/1", lambda node: self.graph.parking_status.get(node, False)))

    def test_astar_uses_real_distance_and_avoids_blocked_osm_way(self):
        target = "node/4"
        original = astar_path(self.graph, "node/1", target)
        detour = astar_path(self.graph, "node/1", target, {"way/9001"})
        self.assertIsNotNone(original)
        self.assertIsNotNone(detour)
        self.assertGreater(detour["cost"], original["cost"])
        self.assertTrue(all(not (self.graph.edge_between(a, b).road_id == "way/9001") for a, b in zip(detour["path"], detour["path"][1:])))
        self.assertGreaterEqual(original["execution_ms"], 0)

    def test_real_extract_simulation_stays_inside_mapped_parking_anchors(self):
        twin = OSMCampusTwin(self.osm, seed=5)
        self.assertEqual(len(twin.parking), len(campus_parking_anchors(self.osm)) * PARKING_SLOTS_PER_AREA)
        self.assertTrue(all(slot["anchor_key"] in twin.graph.places for slot in twin.parking))
        self.assertEqual(twin.snapshot()["capacity"], len(twin.parking))

    def test_simulated_vehicle_moves_between_real_osm_graph_nodes(self):
        twin = OSMCampusTwin(self.osm, seed=5)
        vehicle = next(item for item in twin.vehicles if item["id"] == "V001")
        initial_node = vehicle["node"]

        twin.advance(6, speed=10)

        self.assertIn(vehicle["node"], twin.graph.road_nodes | set(twin.graph.places) | set(twin.graph.parking_status))
        self.assertNotEqual(vehicle["node"], initial_node)
        self.assertIn(vehicle["status"], {"moving", "parked", "waiting"})

    def test_manual_parking_updates_shared_graph_bfs_state(self):
        twin = OSMCampusTwin(self.osm, seed=5)
        slot = next(item for item in twin.parking if not item["occupied"])
        self.assertTrue(twin.graph.parking_status[slot["node"]])

        vehicle_id = twin.add_vehicle(slot["area"])

        self.assertFalse(twin.graph.parking_status[slot["node"]])
        twin.remove_vehicle(vehicle_id)
        self.assertTrue(twin.graph.parking_status[slot["node"]])


if __name__ == "__main__":
    unittest.main()
