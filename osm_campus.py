"""Fetch and normalize OpenStreetMap geometry for the Ettimadai campus."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import requests


OSM_BOUNDS = (10.8975929, 76.8930111, 10.9096162, 76.9063810)
OSM_CENTER = (10.9036429, 76.8996595)
OSM_CAMPUS_WAY_ID = 128964693
OSM_CACHE_PATH = Path(__file__).parent / "data" / "osm_campus.json"
OSM_ATTRIBUTION = "© OpenStreetMap contributors · ODbL 1.0 · https://www.openstreetmap.org/copyright"
OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.nchc.org.tw/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)
USER_AGENT = "AmritaCampusDigitalTwin/1.0 (software-only academic simulation)"
CACHE_TTL_SECONDS = 7 * 24 * 60 * 60


class OSMDataError(RuntimeError):
    """Raised when real campus OSM geometry is unavailable and uncached."""


def haversine_meters(first: tuple[float, float], second: tuple[float, float]) -> float:
    """Great-circle distance between (latitude, longitude) points."""
    lat1, lon1 = map(math.radians, first)
    lat2, lon2 = map(math.radians, second)
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    value = math.sin(delta_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    return 6_371_000 * 2 * math.atan2(math.sqrt(value), math.sqrt(1 - value))


def _bbox_text() -> str:
    south, west, north, east = OSM_BOUNDS
    return f"{south},{west},{north},{east}"


def _query_overpass(query: str, session: requests.Session) -> dict[str, Any]:
    last_error: Exception | None = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            response = session.get(endpoint, params={"data": query}, timeout=(10, 75))
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload.get("elements"), list):
                raise OSMDataError("Overpass returned no OSM elements collection.")
            return payload
        except (requests.RequestException, ValueError, OSMDataError) as error:
            last_error = error
    raise OSMDataError(f"Could not fetch campus data from OpenStreetMap Overpass: {last_error}")


def fetch_osm_data(force_refresh: bool = False, cache_path: str | Path = OSM_CACHE_PATH) -> dict[str, Any]:
    """Load cached OSM data or fetch real roads, buildings and named map features."""
    path = Path(cache_path)
    if path.is_file() and not force_refresh:
        age = time.time() - path.stat().st_mtime
        if age < CACHE_TTL_SECONDS:
            with path.open("r", encoding="utf-8") as cache_file:
                payload = json.load(cache_file)
            payload["cache_stale"] = False
            return payload

    bbox = _bbox_text()
    queries = {
        "roads": f'[out:json][timeout:25];way["highway"]({bbox});out body geom;',
        "buildings": f'[out:json][timeout:25];way["building"]({bbox});out body geom;',
        "features": (
            f'[out:json][timeout:20];('
            f'node["name"]({bbox});node["amenity"]({bbox});node["entrance"]({bbox});'
            f'node["shop"]({bbox});way["name"]({bbox});way["amenity"]({bbox}););out body center;'
        ),
    }
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        fetched = {key: _query_overpass(query, session)["elements"] for key, query in queries.items()}
        if len(fetched["roads"]) < 10 or len(fetched["buildings"]) < 1:
            raise OSMDataError("OSM query returned insufficient campus road/building geometry.")
        payload = {
            "source": "OpenStreetMap Overpass API",
            "attribution": OSM_ATTRIBUTION,
            "campus_osm_way_id": OSM_CAMPUS_WAY_ID,
            "bounds": list(OSM_BOUNDS),
            "center": list(OSM_CENTER),
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "roads": fetched["roads"],
            "buildings": fetched["buildings"],
            "features": fetched["features"],
            "cache_stale": False,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(".tmp")
        with temporary_path.open("w", encoding="utf-8") as cache_file:
            json.dump(payload, cache_file, separators=(",", ":"))
        temporary_path.replace(path)
        return payload
    except OSMDataError:
        if path.is_file():
            with path.open("r", encoding="utf-8") as cache_file:
                payload = json.load(cache_file)
            payload["cache_stale"] = True
            return payload
        raise
    finally:
        session.close()


def element_key(element: dict[str, Any]) -> str:
    return f"{element['type']}/{element['id']}"


def element_position(element: dict[str, Any]) -> tuple[float, float] | None:
    if element["type"] == "node" and "lat" in element and "lon" in element:
        return float(element["lat"]), float(element["lon"])
    center = element.get("center")
    if center and "lat" in center and "lon" in center:
        return float(center["lat"]), float(center["lon"])
    geometry = element.get("geometry") or []
    if geometry:
        return (
            sum(float(point["lat"]) for point in geometry) / len(geometry),
            sum(float(point["lon"]) for point in geometry) / len(geometry),
        )
    return None


def campus_places(osm_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return unique named OSM features with actual or geometry-derived positions."""
    candidates = [*osm_data["features"], *osm_data["buildings"]]
    by_key: dict[str, dict[str, Any]] = {}
    for element in candidates:
        tags = element.get("tags", {})
        position = element_position(element)
        if not position or not (tags.get("name") or tags.get("amenity") or tags.get("entrance")):
            continue
        key = element_key(element)
        by_key[key] = {
            "key": key,
            "label": tags.get("name") or f"OSM {tags.get('amenity') or tags.get('entrance')} {element['id']}",
            "lat": position[0],
            "lon": position[1],
            "tags": tags,
            "osm_type": element["type"],
            "osm_id": int(element["id"]),
            "geometry": element.get("geometry", []),
        }
    return sorted(by_key.values(), key=lambda place: place["label"].casefold())


def campus_parking_anchors(osm_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Get actual OSM-tagged parking features; never invent parking-area anchors."""
    anchors: dict[str, dict[str, Any]] = {}
    for element in osm_data["features"]:
        tags = element.get("tags", {})
        if tags.get("amenity") != "parking" and "parking" not in tags:
            continue
        position = element_position(element)
        if position is None:
            continue
        anchors[element_key(element)] = {
            "key": element_key(element),
            "label": tags.get("name") or f"OSM mapped parking {element['id']}",
            "lat": position[0],
            "lon": position[1],
            "tags": tags,
            "osm_id": int(element["id"]),
        }
    return list(anchors.values())
