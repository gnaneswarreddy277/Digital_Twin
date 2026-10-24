"""Real-OSM, software-only digital twin for Amrita Ettimadai campus."""

from __future__ import annotations

import re
import time
from datetime import datetime

import pandas as pd
import streamlit as st
from algorithms.astar import astar_path
from algorithms.bfs import bfs_nearest
from osm_campus import OSMDataError, fetch_osm_data
from osm_map_view import build_osm_3d_figure, build_osm_mapbox_figure
from osm_simulation import OSMCampusTwin


st.set_page_config(
    page_title="Amrita Coimbatore · Digital Twin",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--ink:#e8eee8;--muted:#a3b0a8;--paper:#101916;--line:#2a3931;--green:#53b981;--red:#df6259;--blue:#5d9cc2}
html,body,[class*="css"]{font-family:'Manrope',sans-serif}.stApp{background:var(--paper);color:var(--ink)}
[data-testid="stHeader"]{background:transparent}[data-testid="stSidebar"]{background:#15211b;border-right:1px solid var(--line)}
.hero{display:flex;justify-content:space-between;align-items:flex-end;border-bottom:1px solid var(--line);padding:8px 0 18px;margin-bottom:18px;gap:18px}
.hero h1{font-size:29px;line-height:1.15;margin:5px 0 0;font-weight:800}.hero p{color:var(--muted);margin:7px 0 0;font-size:13px}
.campus-title{display:flex;justify-content:space-between;align-items:end;gap:16px;padding:4px 0 12px}
.campus-title h2{margin:4px 0 0;font-size:23px;font-weight:800}.campus-title p{margin:4px 0 0;color:var(--muted);font-size:11px}
.brand{color:var(--green);font-size:10px;font-weight:800;letter-spacing:1.6px;text-transform:uppercase}
.live,.mono{font-family:'DM Mono',monospace}.live{color:var(--green);font-size:11px;white-space:nowrap}
.eyebrow{color:var(--muted);font-size:10px;font-weight:800;letter-spacing:1.4px;text-transform:uppercase;margin:10px 0}
[data-testid="stMetric"]{background:#16211c;border:1px solid var(--line);border-radius:5px;padding:12px 14px}
[data-testid="stMetricLabel"]{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.7px}[data-testid="stMetricValue"]{font-size:23px;font-weight:800}
.stButton button{border-radius:4px;font-weight:700}.footnote{color:var(--muted);font:10px 'DM Mono',monospace}
.st-key-kpi-primary [data-testid="stHorizontalBlock"]{display:grid!important;grid-template-columns:repeat(7,minmax(0,1fr));gap:8px}
.st-key-kpi-primary [data-testid="stColumn"]{width:auto!important;min-width:0!important;flex:initial!important}
@media(max-width:850px){.st-key-kpi-primary [data-testid="stHorizontalBlock"]{grid-template-columns:repeat(2,minmax(0,1fr))}.hero{align-items:flex-start;flex-direction:column}.hero h1{font-size:24px}.campus-title{align-items:flex-start;flex-direction:column}}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 7)
def cached_osm_data() -> dict:
    return fetch_osm_data()


try:
    osm_data = st.session_state.pop("osm_data_override", None) or cached_osm_data()
except OSMDataError as error:
    st.error("Could not load the real campus data from OpenStreetMap. No fictional campus map is substituted.")
    st.code(str(error))
    st.stop()

osm_signature = str(osm_data.get("fetched_at", "osm"))
if "osm_campus_twin" not in st.session_state or st.session_state.get("osm_signature") != osm_signature:
    st.session_state.osm_campus_twin = OSMCampusTwin(osm_data)
    st.session_state.osm_signature = osm_signature
    st.session_state.osm_last_tick = time.monotonic()
twin: OSMCampusTwin = st.session_state.osm_campus_twin
place_by_key = {str(place["key"]): place for place in twin.places}
place_keys = list(place_by_key)
road_ids = list(twin.graph.road_ways)

def place_label(key: str) -> str:
    return str(place_by_key[key]["label"])


def road_label(road_id: str) -> str:
    way = twin.graph.road_ways[road_id]
    tags = way.get("tags", {})
    return f"{tags.get('name') or tags.get('highway') or 'Mapped road'} · OSM {way['id']}"


def readable_osm_path(path: list[str]) -> str:
    segments: list[str] = []
    for index, node in enumerate(path):
        if node in twin.graph.places or node in twin.graph.parking_status:
            label = twin.graph.labels[node]
            if not segments or segments[-1] != label:
                segments.append(label)
        if index + 1 < len(path):
            edge = twin.graph.edge_between(node, path[index + 1])
            if edge and edge.kind == "osm" and (not segments or segments[-1] != edge.road_name):
                segments.append(edge.road_name)
    return " → ".join(segments or [twin.graph.labels[node] for node in path[-1:]])


def find_place_index(fragment: str) -> int:
    return next((i for i, place in enumerate(twin.places) if fragment in str(place["label"]).casefold()), 0)


def execute_search(mode: str, start_place: str, destination: str) -> dict[str, object]:
    results: dict[str, object] = {}
    if mode in {"BFS", "COMPARE"}:
        vehicle = next((item for item in twin.vehicles if item["id"] == "V001"), None)
        start_node = str(vehicle["node"] if vehicle else twin.main_gate_key)
        results["BFS"] = bfs_nearest(
            twin.graph, start_node,
            lambda node: twin.graph.parking_status.get(node, False),
            twin.blocked_roads,
        )
    if mode in {"A*", "COMPARE"}:
        results["A*"] = astar_path(twin.graph, start_place, destination, twin.blocked_roads)
    return results


def save_search(mode: str, start_place: str, destination: str) -> None:
    results = execute_search(mode, start_place, destination)
    st.session_state.osm_search_run_id = st.session_state.get("osm_search_run_id", 0) + 1
    st.session_state.osm_search = {"mode": mode, "start": start_place, "destination": destination}
    st.session_state.osm_search_results = results
    st.session_state[f"osm_search_step_{st.session_state.osm_search_run_id}"] = max(
        (len(result["explored"]) for result in results.values() if result), default=0
    )


pending_place = st.session_state.pop("pending_osm_place", None)
if pending_place in place_by_key:
    target_key = "route_start" if st.session_state.get("map_pick_target", "start") == "start" else "route_destination"
    st.session_state[target_key] = pending_place
pending_road = st.session_state.pop("clicked_osm_road", None)
if pending_road in road_ids:
    st.session_state["selected_osm_road"] = pending_road

with st.sidebar:
    st.markdown('<div class="brand">OPENSTREETMAP · REAL GEOGRAPHY</div>', unsafe_allow_html=True)
    st.title("Amrita · Ettimadai")
    st.caption("Coimbatore campus · all people and vehicles are simulated")
    st.divider()
    st.markdown("**Real map display**")
    map_view = st.radio("Map view", ["REAL OSM MAP", "3D DIGITAL TWIN"], horizontal=True, label_visibility="collapsed")
    map_pick_target = st.selectbox("Use next clicked map place as", ["start", "destination"], key="map_pick_target")
    speed = st.selectbox("Simulation speed", ["Slow", "Normal", "Fast"], index=1)
    speed_multiplier = {"Slow": 10, "Normal": 30, "Fast": 60}[speed]
    start_column, pause_column = st.columns(2)
    if start_column.button("▶ START", width="stretch"):
        twin.start()
    if pause_column.button("⏸ PAUSE", width="stretch"):
        twin.pause()
    if st.button("↻ RESET SIMULATION", width="stretch"):
        st.session_state.osm_campus_twin = OSMCampusTwin(osm_data)
        st.session_state.osm_last_tick = time.monotonic()
        st.session_state.pop("osm_search", None)
        st.rerun()

    st.divider()
    st.markdown("**BFS and A* · same OSM road graph**")
    start_default = find_place_index("main gate")
    destination_default = find_place_index("central library")
    route_start = st.selectbox("A* start location", place_keys, index=start_default, format_func=place_label, key="route_start")
    route_destination = st.selectbox("A* destination", place_keys, index=destination_default, format_func=place_label, key="route_destination")
    bfs_column, astar_column = st.columns(2)
    if bfs_column.button("BFS – FIND PARKING", width="stretch"):
        save_search("BFS", route_start, route_destination)
    if astar_column.button("A* – FIND ROUTE", width="stretch"):
        save_search("A*", route_start, route_destination)
    compare_column, clear_column = st.columns(2)
    if compare_column.button("COMPARE BFS & A*", width="stretch"):
        save_search("COMPARE", route_start, route_destination)
    if clear_column.button("RESET PATH", width="stretch"):
        st.session_state.pop("osm_search", None)
        st.session_state.pop("osm_search_results", None)

    st.divider()
    st.markdown("**Road block simulation · OSM ways**")
    if road_ids:
        selected_road = st.selectbox("Choose a mapped road/path", road_ids, format_func=road_label, key="selected_osm_road")
        if selected_road in twin.blocked_roads:
            if st.button("UNBLOCK ROAD", width="stretch"):
                twin.toggle_road(selected_road)
        elif st.button("BLOCK ROAD", width="stretch"):
            twin.toggle_road(selected_road)

    st.divider()
    st.markdown("**Simulated parking control**")
    parking_areas = sorted({str(slot["area"]) for slot in twin.parking})
    parking_area = st.selectbox("OSM parking point", parking_areas)
    add_column, remove_column = st.columns(2)
    if add_column.button("＋ VEHICLE", width="stretch"):
        try:
            twin.add_vehicle(parking_area)
        except ValueError as error:
            st.warning(str(error))
    if remove_column.button("− VEHICLE", width="stretch"):
        try:
            twin.remove_vehicle()
        except ValueError as error:
            st.warning(str(error))

    if st.button("START SIMULATED MEETING", width="stretch"):
        twin.start_meeting()
    if st.button("END SIMULATED MEETING", width="stretch"):
        twin.end_meeting()

    if st.button("↻ REFRESH OSM DATA", width="stretch"):
        try:
            with st.spinner("Refreshing mapped campus data from Overpass…"):
                refreshed = fetch_osm_data(force_refresh=True)
            cached_osm_data.clear()
            st.session_state.osm_data_override = refreshed
            st.session_state.pop("osm_campus_twin", None)
            st.rerun()
        except OSMDataError as error:
            st.error(str(error))


@st.fragment
def render_interactive_map(osm_data: dict, state: dict, graph, search_results: list, search_step: int, map_view: str) -> None:
    if map_view == "REAL OSM MAP":
        st.markdown('<div class="eyebrow">Real campus map · OpenStreetMap Mapbox WebGL view · smooth non-blinking rendering</div>', unsafe_allow_html=True)
        st.plotly_chart(
            build_osm_mapbox_figure(osm_data, state, graph, search_results, search_step),
            width="stretch", config={"displayModeBar": True, "scrollZoom": True},
        )
        st.caption("Real OpenStreetMap tiles and campus vectors rendered via WebGL (zero blink).")
    else:
        st.markdown('<div class="eyebrow">3D twin · extruded mapped OSM building footprints and OSM roads</div>', unsafe_allow_html=True)
        st.plotly_chart(
            build_osm_3d_figure(osm_data, state, graph, search_results, search_step),
            width="stretch", config={"displayModeBar": True, "scrollZoom": True},
        )
        st.caption("Footprints and road positions follow OSM geometry. Where OSM has no height/levels tag, building extrusion height is illustrative, not surveyed.")


def render_dashboard() -> None:
    now = time.monotonic()
    elapsed = now - st.session_state.get("osm_last_tick", now)
    st.session_state.osm_last_tick = now
    twin.advance(elapsed, speed_multiplier)
    state = twin.snapshot()
    search_run = st.session_state.get("osm_search")
    search_results: list[tuple[str, dict[str, object] | None]] = []
    search_step = 0
    if search_run:
        fresh = execute_search(str(search_run["mode"]), str(search_run["start"]), str(search_run["destination"]))
        st.session_state.osm_search_results = fresh
        search_results = [(name, result) for name, result in fresh.items()]
        maximum = max((len(result["explored"]) for result in fresh.values() if result), default=0)
        step_key = f"osm_search_step_{st.session_state.osm_search_run_id}"
        if st.session_state.get(step_key, 0) > maximum:
            st.session_state[step_key] = maximum
        if maximum:
            search_step = st.slider("Search playback · reveal explored real OSM nodes", 0, maximum, key=step_key)

    occupancy = twin.forecast(10)
    st.markdown(
        f'<div class="campus-title"><div><div class="brand">AI DIGITAL TWIN · REAL OPENSTREETMAP BASE</div>'
        f'<h2>AMRITA VISHWA VIDYAPEETHAM</h2><p>COIMBATORE CAMPUS · ALL PEOPLE, VEHICLES AND PARKING STATUS ARE SIMULATED</p>'
        f'</div><div class="live">{"● SIMULATION LIVE" if state["running"] else "Ⅱ SIMULATION PAUSED"}</div></div>',
        unsafe_allow_html=True,
    )
    with st.container(key="kpi-primary"):
        columns = st.columns(7)
        columns[0].metric("Roads", len(osm_data["roads"]))
        columns[1].metric("Bldgs", len(osm_data["buildings"]))
        columns[2].metric("Slots", state["capacity"])
        columns[3].metric("Taken", state["occupied"])
        columns[4].metric("Free", state["available"])
        columns[5].metric("People", state["people_count"])
        columns[6].metric("Time", datetime.fromisoformat(state["timestamp"]).strftime("%H:%M"))

    render_interactive_map(osm_data, state, twin.graph, search_results, search_step, map_view)

    if search_run:
        st.markdown('<div class="eyebrow">Path results · live OSM graph</div>', unsafe_allow_html=True)
        rows = []
        for algorithm, result in search_results:
            if result is None:
                st.warning(f"{algorithm}: no reachable {'available OSM-anchored parking slot' if algorithm == 'BFS' else 'mapped destination'} with current road closures.")
                continue
            path = list(result["path"])
            path_text = readable_osm_path(path)
            if algorithm == "BFS":
                st.markdown(
                    f"**BFS · nearest simulated parking** · V001 at {twin.graph.labels[path[0]]} · "
                    f"target {twin.graph.labels[str(result['target'])]} · {result['path_length']} edges · "
                    f"{result['distance_meters']:.1f} m · {len(result['explored'])} explored · {result['execution_ms']:.3f} ms"
                )
            else:
                st.markdown(
                    f"**A* · mapped destination** · {path_text} · {result['cost']:.1f} m · "
                    f"{len(result['explored'])} explored · {result['execution_ms']:.3f} ms"
                )
            rows.append({
                "Algorithm": algorithm,
                "Purpose": "Nearest available parking by edge count" if algorithm == "BFS" else "Shortest mapped route by distance",
                "OSM path": path_text,
                "Nodes explored": len(result["explored"]),
                "Path edges": result["path_length"],
                "Distance (m)": round(float(result.get("distance_meters", result.get("cost", 0))), 1),
                "Execution (ms)": round(float(result["execution_ms"]), 3),
            })
        if len(rows) > 1:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.divider()
    parking_column, locations_column, insights_column = st.columns([1.05, 1.3, 0.95], gap="large")
    with parking_column:
        st.markdown('<div class="eyebrow">SIMULATED PARKING · around OSM parking points</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([
            {"Slot": slot["id"], "Mapped OSM anchor": slot["area"],
             "Status": "OCCUPIED" if slot["occupied"] else "AVAILABLE", "Vehicle": slot["vehicle_id"] or "—",
             "Entry": slot["entry_time"] or "—", "Exit": slot["exit_time"] or "—"}
            for slot in state["parking"]
        ]), hide_index=True, width="stretch", height=300)
    with locations_column:
        st.markdown('<div class="eyebrow">SIMULATED LIVE LOCATION · generated positions on real OSM graph</div>', unsafe_allow_html=True)
        people_rows = [{"ID": p["id"], "Type": p["role"], "Nearest OSM place": p["location"],
                        "Mapped destination": p["destination_label"], "Lat": round(float(p["lat"]), 6), "Lon": round(float(p["lon"]), 6)}
                       for p in state["people"][:8]]
        vehicle_rows = [{"ID": v["id"], "Type": "SIMULATED VEHICLE", "Nearest OSM place": v["status"],
                         "Mapped destination": v.get("destination_slot") or "—", "Lat": round(float(v["lat"]), 6), "Lon": round(float(v["lon"]), 6)}
                        for v in state["vehicles"]]
        st.dataframe(pd.DataFrame(people_rows + vehicle_rows), hide_index=True, width="stretch", height=300)
    with insights_column:
        st.markdown('<div class="eyebrow">Campus operations</div>', unsafe_allow_html=True)
        if state["meeting_active"] and state["meeting_place"]:
            st.info(f"SIMULATED MEETING · {state['meeting_place']['label']} · {state['meeting_attendees']} participants")
        elif twin.meeting_place:
            st.caption(f"Mapped venue: {twin.meeting_place['label']} · meeting schedule is simulated (14:00–14:45).")
        else:
            st.caption("No mapped meeting hall name in the current OSM data.")
        st.markdown("**SIMULATED AI PREDICTION**")
        st.write(f"In {occupancy['horizon_minutes']} minutes: **{occupancy['available']} free slots** ({occupancy['available_low']}–{occupancy['available_high']} modeled range).")
        st.caption("Prediction uses generated parking histories only.")
        st.markdown("**Mapped high-activity places**")
        counts = sorted(state["place_counts"].items(), key=lambda item: item[1], reverse=True)[:5]
        for place_name, count in counts:
            st.write(f"{place_name}: {count} simulated people")
        for event in state["events"][:4]:
            st.caption(f"{event['time']} · {event['message']}")

    st.markdown(f'<div class="footnote">{osm_data["attribution"]} · {len(osm_data["roads"])} road ways · {len(osm_data["buildings"])} building footprints · GENERATED ACTIVITY ONLY</div>', unsafe_allow_html=True)
    if osm_data.get("cache_stale"):
        st.warning("Overpass could not refresh. Displaying the previously cached OSM data.")


@st.fragment(run_every="15s")
def live_dashboard() -> None:
    render_dashboard()


live_dashboard()
