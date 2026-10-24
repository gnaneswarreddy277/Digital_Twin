import unittest

import folium
import plotly.graph_objects as go

from osm_map_view import build_leaflet_map, build_osm_3d_figure
from osm_simulation import OSMCampusTwin
from tests.test_osm_algorithms import make_osm_fixture


class OSMMapRenderingTests(unittest.TestCase):
    def setUp(self):
        self.osm = make_osm_fixture()
        self.twin = OSMCampusTwin(self.osm, seed=9)
        self.state = self.twin.snapshot()

    def test_leaflet_map_uses_osm_tiles_roads_buildings_and_simulation_layers(self):
        map_graph = build_leaflet_map(self.osm, self.state, self.twin.graph)
        self.assertIsInstance(map_graph, folium.Map)
        html = map_graph.get_root().render()
        self.assertIn("OpenStreetMap", html)
        self.assertIn("OSM roads and paths", html)
        self.assertIn("OSM building footprints", html)
        self.assertIn("SIMULATED PARKING", html)
        self.assertIn("SIMULATED LIVE LOCATIONS", html)
        self.assertIn("openstreetmap.org/copyright", html)

    def test_3d_view_extrudes_only_osm_buildings_and_draws_osm_roads(self):
        figure = build_osm_3d_figure(self.osm, self.state, self.twin.graph)
        self.assertIsInstance(figure, go.Figure)
        mesh_traces = [trace for trace in figure.data if isinstance(trace, go.Mesh3d)]
        road_traces = [trace for trace in figure.data if isinstance(trace, go.Scatter3d) and "OSM way" in str(trace.hovertemplate)]
        self.assertEqual(len(mesh_traces), len(self.osm["buildings"]))
        self.assertEqual(len(road_traces), len(self.osm["roads"]))


if __name__ == "__main__":
    unittest.main()
