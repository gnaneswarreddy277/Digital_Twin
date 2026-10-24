import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from osm_campus import (
    OSM_BOUNDS,
    OSMDataError,
    campus_parking_anchors,
    campus_places,
    element_position,
    fetch_osm_data,
    haversine_meters,
)


class OSMCampusDataTests(unittest.TestCase):
    def setUp(self):
        self.fixture = {
            "source": "OpenStreetMap Overpass API",
            "attribution": "© OpenStreetMap contributors",
            "campus_osm_way_id": 128964693,
            "bounds": list(OSM_BOUNDS),
            "center": [10.9036429, 76.8996595],
            "fetched_at": "2026-09-27T12:23:21Z",
            "roads": [
                {"type": "way", "id": 9001, "nodes": [1, 2, 3],
                 "tags": {"highway": "residential", "name": "OSM campus road"},
                 "geometry": [{"lat": 10.9, "lon": 76.9}, {"lat": 10.901, "lon": 76.9}, {"lat": 10.902, "lon": 76.9}]},
                {"type": "way", "id": 9002, "nodes": [2, 4],
                 "tags": {"highway": "footway"},
                 "geometry": [{"lat": 10.901, "lon": 76.9}, {"lat": 10.901, "lon": 76.901}]},
            ],
            "buildings": [
                {"type": "way", "id": 9100, "tags": {"building": "library", "name": "Mapped Library"},
                 "geometry": [{"lat": 10.9005, "lon": 76.9005}, {"lat": 10.9005, "lon": 76.9007},
                              {"lat": 10.9007, "lon": 76.9007}, {"lat": 10.9007, "lon": 76.9005}]}
            ],
            "features": [
                {"type": "node", "id": 9200, "lat": 10.9, "lon": 76.9,
                 "tags": {"name": "Main gate", "entrance": "main"}},
                {"type": "node", "id": 9201, "lat": 10.902, "lon": 76.9,
                 "tags": {"amenity": "parking"}},
            ],
        }

    def test_osm_positions_and_attribution_are_preserved(self):
        self.assertEqual(element_position(self.fixture["features"][0]), (10.9, 76.9))
        places = campus_places(self.fixture)
        self.assertTrue(any(place["label"] == "Mapped Library" for place in places))
        self.assertTrue(any(place["key"] == "node/9200" for place in places))
        anchors = campus_parking_anchors(self.fixture)
        self.assertEqual(anchors[0]["key"], "node/9201")
        self.assertGreater(haversine_meters((10.9, 76.9), (10.901, 76.9)), 100)

    def test_cache_reads_real_osm_extract_without_network(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "osm.json"
            path.write_text(json.dumps(self.fixture), encoding="utf-8")
            with patch("osm_campus.requests.Session.get", side_effect=AssertionError("network should not be used")):
                cached = fetch_osm_data(cache_path=path)
        self.assertEqual(len(cached["roads"]), 2)
        self.assertEqual(cached["roads"][0]["id"], 9001)

    def test_fresh_load_requires_real_roads_and_buildings(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "osm.json"
            session = Mock()
            session.headers = {}
            session.get.side_effect = [
                Mock(ok=True, status_code=200, json=lambda: {"elements": self.fixture["roads"]}, raise_for_status=lambda: None),
                Mock(ok=True, status_code=200, json=lambda: {"elements": []}, raise_for_status=lambda: None),
                Mock(ok=True, status_code=200, json=lambda: {"elements": self.fixture["features"]}, raise_for_status=lambda: None),
            ]
            with patch("osm_campus.requests.Session", return_value=session):
                with self.assertRaises(OSMDataError):
                    fetch_osm_data(force_refresh=True, cache_path=path)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
