"""Leaflet OSM map and OSM-footprint-derived 3D campus view."""

from __future__ import annotations

import html
import math
from typing import Any

import folium
import plotly.graph_objects as go

from algorithms.graph import CampusGraph
from osm_campus import OSM_ATTRIBUTION, campus_places


def _element_label(element: dict[str, Any]) -> str:
    tags = element.get("tags", {})
    return str(tags.get("name") or tags.get("amenity") or tags.get("building") or f"OSM {element['type']} {element['id']}")


def build_leaflet_map(
    osm_data: dict[str, Any],
    state: dict[str, Any],
    graph: CampusGraph,
    search_results: list[tuple[str, dict[str, Any] | None]] | None = None,
    search_step: int = 0,
) -> folium.Map:
    center = osm_data["center"]
    campus_map = folium.Map(location=center, zoom_start=16, tiles=None, control_scale=True, prefer_canvas=True)
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        attr=OSM_ATTRIBUTION,
        max_zoom=20,
        overlay=False,
        control=True,
    ).add_to(campus_map)

    buildings_layer = folium.FeatureGroup(name="OSM building footprints", show=True)
    for building in osm_data["buildings"]:
        geometry = building.get("geometry") or []
        if len(geometry) < 4:
            continue
        label = _element_label(building)
        coordinates = [(float(point["lat"]), float(point["lon"])) for point in geometry]
        safe_label = html.escape(label)
        folium.Polygon(
            locations=coordinates,
            color="#5b7668",
            weight=1,
            fill=True,
            fill_color="#769989",
            fill_opacity=0.30,
            tooltip=f"{safe_label} · OSM building {building['id']}",
            popup=folium.Popup(f"{safe_label}<br>OpenStreetMap way {building['id']}", max_width=320),
        ).add_to(buildings_layer)
    buildings_layer.add_to(campus_map)

    roads_layer = folium.FeatureGroup(name="OSM roads and paths", show=True)
    blocked = set(state["blocked_roads"])
    for road in osm_data["roads"]:
        geometry = road.get("geometry") or []
        if len(geometry) < 2:
            continue
        road_id = f"way/{road['id']}"
        tags = road.get("tags", {})
        road_name = str(tags.get("name") or tags.get("highway") or f"OSM way {road['id']}")
        is_blocked = road_id in blocked
        coords = [(float(point["lat"]), float(point["lon"])) for point in geometry]
        folium.PolyLine(
            coords,
            color="#dc4d4d" if is_blocked else "#477f68",
            weight=6 if is_blocked else 3,
            opacity=0.95,
            dash_array="8 6" if is_blocked else None,
            tooltip=f"{road_name} · {'BLOCKED IN SIMULATION' if is_blocked else tags.get('highway', 'mapped path')}",
            popup=folium.Popup(f"OSM way: {road_id}<br>{html.escape(road_name)}", max_width=300),
        ).add_to(roads_layer)
    roads_layer.add_to(campus_map)

    locations_layer = folium.FeatureGroup(name="Mapped campus places", show=True)
    for place in campus_places(osm_data):
        label = str(place["label"])
        tags = place["tags"]
        is_parking_anchor = tags.get("amenity") == "parking" or "parking" in tags
        folium.CircleMarker(
            location=(float(place["lat"]), float(place["lon"])),
            radius=7 if is_parking_anchor else 5,
            color="#d57e31" if is_parking_anchor else "#1c8a77",
            weight=2,
            fill=True,
            fill_color="#f0ac50" if is_parking_anchor else "#4eb9a2",
            fill_opacity=0.9,
            tooltip=f"{label} · OSM {place['osm_type']} {place['osm_id']}",
            popup=folium.Popup(
                f"<b>{html.escape(label)}</b><br>OSM {place['osm_type']} {place['osm_id']}<br>"
                f"{html.escape(str(tags.get('amenity') or tags.get('building') or tags.get('shop') or 'mapped feature'))}",
                max_width=320,
            ),
        ).add_to(locations_layer)
    locations_layer.add_to(campus_map)

    parking_layer = folium.FeatureGroup(name="SIMULATED PARKING", show=True)
    for slot in state["parking"]:
        occupied = bool(slot["occupied"])
        folium.CircleMarker(
            location=(float(slot["lat"]), float(slot["lon"])),
            radius=5,
            color="#c74b48" if occupied else "#35a365",
            fill=True,
            fill_color="#df6259" if occupied else "#57c77f",
            fill_opacity=0.95,
            tooltip=f"SIMULATED SLOT {slot['id']} · {'OCCUPIED' if occupied else 'AVAILABLE'} · {slot['area']}",
            popup=folium.Popup(
                f"SIMULATED PARKING<br>{slot['id']} · {html.escape(slot['area'])}<br>"
                f"{'Occupied by ' + html.escape(str(slot['vehicle_id'])) if occupied else 'Available'}<br>"
                f"Entry: {slot['entry_time'] or '—'} · Exit: {slot['exit_time'] or '—'}",
                max_width=320,
            ),
        ).add_to(parking_layer)
    parking_layer.add_to(campus_map)

    people_layer = folium.FeatureGroup(name="SIMULATED LIVE LOCATIONS", show=True)
    for person in state["people"]:
        folium.CircleMarker(
            location=(float(person["lat"]), float(person["lon"])),
            radius=4,
            color="#8e58ae",
            fill=True,
            fill_color="#bd7bdf",
            fill_opacity=0.88,
            tooltip=f"SIMULATED LIVE LOCATION · {person['id']} · {person['role']} · {person['location']} → {person['destination_label']}",
        ).add_to(people_layer)
    people_layer.add_to(campus_map)

    vehicles_layer = folium.FeatureGroup(name="SIMULATED VEHICLES", show=True)
    for vehicle in state["vehicles"]:
        folium.Marker(
            location=(float(vehicle["lat"]), float(vehicle["lon"])),
            tooltip=f"SIMULATED VEHICLE {vehicle['id']} · {vehicle['status']}",
            popup=folium.Popup(f"SIMULATED VEHICLE<br>{vehicle['id']} · {html.escape(str(vehicle['status']))}", max_width=240),
            icon=folium.Icon(color="orange" if vehicle["status"] != "parked" else "gray", icon="car", prefix="fa"),
        ).add_to(vehicles_layer)
    vehicles_layer.add_to(campus_map)

    if state.get("meeting_active") and state.get("meeting_place"):
        venue = state["meeting_place"]
        folium.Marker(
            (float(venue["lat"]), float(venue["lon"])),
            tooltip=f"SIMULATED MEETING · {venue['label']} · {state['meeting_attendees']} participants",
            icon=folium.Icon(color="blue", icon="users", prefix="fa"),
        ).add_to(campus_map)

    colors = {"BFS": "#2479df", "A*": "#e1ad20"}
    for algorithm, result in search_results or []:
        if not result:
            continue
        explored = list(result["explored"])
        for node in explored[:search_step]:
            lat, lon = graph.positions[node]
            folium.CircleMarker(
                (lat, lon), radius=5, color=colors[algorithm], fill=True,
                fill_color=colors[algorithm], fill_opacity=0.8,
                tooltip=f"{algorithm} explored · {graph.labels[node]}",
            ).add_to(campus_map)
        path = list(result["path"])
        if search_step >= len(explored) and len(path) > 1:
            coords = [graph.positions[node] for node in path]
            folium.PolyLine(
                coords, color="#27c65d", weight=8, opacity=0.95,
                tooltip=f"{algorithm} selected route · {result.get('distance_meters', result.get('cost', 0)):.1f} m",
            ).add_to(campus_map)
        if algorithm == "BFS":
            origin = path[0]
            folium.Marker(
                graph.positions[origin],
                tooltip="SIMULATED VEHICLE · BFS starting location",
                icon=folium.Icon(color="orange", icon="car", prefix="fa"),
            ).add_to(campus_map)

    folium.LayerControl(collapsed=True).add_to(campus_map)
    south, west, north, east = osm_data["bounds"]
    campus_map.fit_bounds([[south, west], [north, east]], padding=(12, 12))
    return campus_map


def _local_xy(latitude: float, longitude: float, origin: tuple[float, float]) -> tuple[float, float]:
    origin_lat, origin_lon = origin
    x = (longitude - origin_lon) * 111_320 * math.cos(math.radians(origin_lat))
    y = (latitude - origin_lat) * 110_574
    return x, y


def _building_mesh(figure: go.Figure, building: dict[str, Any], origin: tuple[float, float]) -> None:
    geometry = building.get("geometry") or []
    if len(geometry) < 4:
        return
    points = [(float(point["lat"]), float(point["lon"])) for point in geometry]
    if points[0] == points[-1]:
        points = points[:-1]
    if len(points) < 3:
        return
    tags = building.get("tags", {})
    try:
        height = float(str(tags.get("height", "")).replace("m", "").strip())
    except ValueError:
        try:
            height = float(tags.get("building:levels", 3)) * 3.2
        except (TypeError, ValueError):
            height = 9.6
    xy = [_local_xy(lat, lon, origin) for lat, lon in points]
    count = len(xy)
    center_x = sum(point[0] for point in xy) / count
    center_y = sum(point[1] for point in xy) / count
    xs = [point[0] for point in xy] + [point[0] for point in xy] + [center_x]
    ys = [point[1] for point in xy] + [point[1] for point in xy] + [center_y]
    zs = [0.0] * count + [height] * count + [height]
    bottom_center = count * 2
    i_values: list[int] = []
    j_values: list[int] = []
    k_values: list[int] = []
    for index in range(count):
        following = (index + 1) % count
        top = count + index
        top_following = count + following
        i_values.extend((index, following, bottom_center))
        j_values.extend((following, top_following, top))
        k_values.extend((top, top, top_following))
    label = _element_label(building)
    figure.add_trace(go.Mesh3d(
        x=xs, y=ys, z=zs, i=i_values, j=j_values, k=k_values,
        color="#628874", opacity=0.78, flatshading=True,
        hovertemplate=f"{html.escape(label)}<br>OSM footprint · estimated extrusion {height:.1f} m<extra></extra>",
        showlegend=False,
    ))


def build_osm_3d_figure(
    osm_data: dict[str, Any],
    state: dict[str, Any],
    graph: CampusGraph,
    search_results: list[tuple[str, dict[str, Any] | None]] | None = None,
    search_step: int = 0,
) -> go.Figure:
    origin = tuple(osm_data["center"])
    figure = go.Figure()
    for building in osm_data["buildings"]:
        _building_mesh(figure, building, origin)

    for road in osm_data["roads"]:
        geometry = road.get("geometry") or []
        if len(geometry) < 2:
            continue
        road_id = f"way/{road['id']}"
        points = [_local_xy(float(point["lat"]), float(point["lon"]), origin) for point in geometry]
        is_blocked = road_id in state["blocked_roads"]
        tags = road.get("tags", {})
        figure.add_trace(go.Scatter3d(
            x=[point[0] for point in points], y=[point[1] for point in points], z=[0.25] * len(points),
            mode="lines", line={"color": "#ee544e" if is_blocked else "#869c91", "width": 5},
            name=f"{'BLOCKED · ' if is_blocked else ''}{tags.get('name') or tags.get('highway')} · OSM {road['id']}",
            showlegend=False,
            hovertemplate=f"{html.escape(str(tags.get('name') or tags.get('highway')))} · OSM way {road['id']}<extra></extra>",
        ))

    for slot in state["parking"]:
        x, y = _local_xy(float(slot["lat"]), float(slot["lon"]), origin)
        figure.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[0.5], mode="markers+text", text=[slot["id"]],
            marker={"size": 7, "color": "#c94d49" if slot["occupied"] else "#31a361", "symbol": "square"},
            name=f"SIMULATED SLOT {slot['id']}",
            hovertemplate=f"SIMULATED PARKING · {slot['id']} · {'OCCUPIED' if slot['occupied'] else 'AVAILABLE'}<extra></extra>",
            showlegend=False,
        ))

    for person in state["people"]:
        x, y = _local_xy(float(person["lat"]), float(person["lon"]), origin)
        figure.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[1.0], mode="markers",
            marker={"size": 4, "color": "#ab6bc8"},
            name=f"SIMULATED {person['role']} {person['id']}",
            hovertemplate=f"SIMULATED LIVE LOCATION · {person['id']} · {html.escape(person['location'])}<extra></extra>",
            showlegend=False,
        ))

    for vehicle in state["vehicles"]:
        x, y = _local_xy(float(vehicle["lat"]), float(vehicle["lon"]), origin)
        figure.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[1.4], mode="markers+text", text=[vehicle["id"]],
            marker={"size": 7, "color": "#df873b", "symbol": "diamond"},
            name=f"SIMULATED VEHICLE {vehicle['id']}",
            hovertemplate=f"SIMULATED VEHICLE {vehicle['id']} · {vehicle['status']}<extra></extra>",
            showlegend=False,
        ))

    for algorithm, result in search_results or []:
        if not result:
            continue
        visible = list(result["explored"][:search_step])
        if visible:
            coords = [_local_xy(*graph.positions[node], origin) for node in visible]
            color = "#3385eb" if algorithm == "BFS" else "#e3b52e"
            figure.add_trace(go.Scatter3d(
                x=[p[0] for p in coords], y=[p[1] for p in coords], z=[1.8] * len(coords),
                mode="markers", marker={"size": 5, "color": color},
                name=f"{algorithm} explored nodes", hoverinfo="skip",
            ))
        path = list(result["path"])
        if search_step >= len(result["explored"]) and len(path) > 1:
            coords = [_local_xy(*graph.positions[node], origin) for node in path]
            figure.add_trace(go.Scatter3d(
                x=[p[0] for p in coords], y=[p[1] for p in coords], z=[2.0] * len(coords),
                mode="lines+markers", line={"color": "#36cf65", "width": 9},
                marker={"size": 4, "color": "#36cf65"}, name=f"{algorithm} selected route",
            ))

    figure.update_layout(
        height=680, margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="#101916", plot_bgcolor="#101916",
        font={"family": "Manrope", "color": "#e4ece5", "size": 10},
        scene={
            "bgcolor": "#101916",
            "xaxis": {"title": "East · metres", "gridcolor": "#26352d"},
            "yaxis": {"title": "North · metres", "gridcolor": "#26352d"},
            "zaxis": {"title": "Height · metres", "gridcolor": "#26352d"},
            "aspectmode": "data",
        },
        legend={"orientation": "h", "y": 1.02, "font": {"color": "#e4ece5"}},
        uirevision="osm-campus-3d",
    )
    return figure


def build_osm_mapbox_figure(
    osm_data: dict[str, Any],
    state: dict[str, Any],
    graph: CampusGraph,
    search_results: list[tuple[str, dict[str, Any] | None]] | None = None,
    search_step: int = 0,
) -> go.Figure:
    center = osm_data["center"]
    figure = go.Figure()

    road_lons, road_lats = [], []
    for road in osm_data["roads"]:
        geom = road.get("geometry") or []
        if len(geom) < 2:
            continue
        for pt in geom:
            road_lons.append(float(pt["lon"]))
            road_lats.append(float(pt["lat"]))
        road_lons.append(None)
        road_lats.append(None)

    figure.add_trace(go.Scattermapbox(
        mode="lines",
        lon=road_lons,
        lat=road_lats,
        line=dict(width=3, color="#477f68"),
        name="OSM Roads",
        hoverinfo="skip"
    ))

    places = campus_places(osm_data)
    p_lons = [float(p["lon"]) for p in places]
    p_lats = [float(p["lat"]) for p in places]
    p_texts = [str(p["label"]) for p in places]
    figure.add_trace(go.Scattermapbox(
        mode="markers+text",
        lon=p_lons,
        lat=p_lats,
        marker=dict(size=7, color="#1c8a77"),
        text=p_texts,
        textposition="top center",
        textfont=dict(color="#e8eee8", size=10),
        name="Mapped Places"
    ))

    slots = state["parking"]
    if slots:
        s_lons = [float(s["lon"]) for s in slots]
        s_lats = [float(s["lat"]) for s in slots]
        s_colors = ["#df6259" if s["occupied"] else "#57c77f" for s in slots]
        s_texts = [f"Slot {s['id']} ({s['area']}) · {'Occupied' if s['occupied'] else 'Available'}" for s in slots]
        figure.add_trace(go.Scattermapbox(
            mode="markers",
            lon=s_lons,
            lat=s_lats,
            marker=dict(size=9, color=s_colors),
            text=s_texts,
            hoverinfo="text",
            name="Simulated Parking"
        ))

    vehicles = state["vehicles"]
    if vehicles:
        v_lons = [float(v["lon"]) for v in vehicles]
        v_lats = [float(v["lat"]) for v in vehicles]
        v_texts = [f"Vehicle {v['id']} ({v['status']})" for v in vehicles]
        figure.add_trace(go.Scattermapbox(
            mode="markers+text",
            lon=v_lons,
            lat=v_lats,
            marker=dict(size=12, color="orange"),
            text=v_texts,
            textposition="top right",
            textfont=dict(color="orange", size=10),
            name="Simulated Vehicles"
        ))

    people = state["people"]
    if people:
        pe_lons = [float(p["lon"]) for p in people]
        pe_lats = [float(p["lat"]) for p in people]
        pe_texts = [f"{p['id']} ({p['role']})" for p in people]
        figure.add_trace(go.Scattermapbox(
            mode="markers",
            lon=pe_lons,
            lat=pe_lats,
            marker=dict(size=6, color="#bd7bdf"),
            text=pe_texts,
            hoverinfo="text",
            name="Simulated People"
        ))

    colors = {"BFS": "#2479df", "A*": "#e1ad20"}
    for algorithm, result in search_results or []:
        if not result:
            continue
        explored = list(result["explored"])
        exp_lons = [graph.positions[node][1] for node in explored[:search_step]]
        exp_lats = [graph.positions[node][0] for node in explored[:search_step]]
        if exp_lons:
            figure.add_trace(go.Scattermapbox(
                mode="markers",
                lon=exp_lons,
                lat=exp_lats,
                marker=dict(size=6, color=colors[algorithm]),
                name=f"{algorithm} Explored"
            ))
        path = list(result["path"])
        if search_step >= len(explored) and len(path) > 1:
            path_lons = [graph.positions[node][1] for node in path]
            path_lats = [graph.positions[node][0] for node in path]
            figure.add_trace(go.Scattermapbox(
                mode="lines",
                lon=path_lons,
                lat=path_lats,
                line=dict(width=6, color="#27c65d"),
                name=f"{algorithm} Route"
            ))

    figure.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center[0], lon=center[1]),
            zoom=15.5
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=700,
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(16,25,22,0.85)", font=dict(color="#e8eee8")),
        paper_bgcolor="#101916",
        plot_bgcolor="#101916",
        uirevision="osm-campus-mapbox",
    )
    return figure
