#!/usr/bin/env python3
"""
Find the closest active precipitation echo to a GPS point.

Point is read from .env (LAT/LON). The local NWS radar station covering
that point is looked up via api.weather.gov, and the actual "where is the
nearest precip" answer comes from NOAA's MRMS PrecipRate product, which is
a national radar mosaic built from all NEXRAD sites (including the local
one) -- there's no NWS endpoint that answers "nearest precip" from a single
radar's raw feed directly.
"""

import gzip
import json
import math
import os
import sys

import numpy as np
import requests
from dotenv import load_dotenv

NWS_POINTS_URL = "https://api.weather.gov/points/{lat},{lon}"
MRMS_URL = "https://mrms.ncep.noaa.gov/data/2D/PrecipRate/MRMS_PrecipRate.latest.grib2.gz"
NO_COVERAGE = -3.0  # MRMS sentinel value for "no radar coverage" at a cell
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
CACHE_GRIB_PATH = os.path.join(CACHE_DIR, "mrms_precip_rate.grib2")
CACHE_META_PATH = os.path.join(CACHE_DIR, "mrms_precip_rate.meta.json")
PRECIP_THRESHOLD_MMHR = 0.1  # anything at/above this counts as "precip"
BBOX_STEPS_DEG = [1, 2, 4, 8, 16, 30]  # progressively widen the search

# distance is stored internally in km; "knots" isn't really a distance unit
# (it's nautical miles per hour), but we treat it as shorthand for nautical
# miles here since that's what people asking for knots actually want.
UNIT_CONVERSIONS = {
    "miles": (0.621371, "mi"),
    "km": (1.0, "km"),
    "knots": (0.539957, "nmi"),
}

RATE_UNIT_CONVERSIONS = {
    "mm": (1.0, "mm/hr"),
    "in": (0.0393701, "in/hr"),
}


def load_point():
    load_dotenv()
    try:
        lat = float(os.environ["LAT"])
        lon = float(os.environ["LON"])
    except (KeyError, ValueError) as exc:
        sys.exit(f"Set LAT and LON in .env (see .env.example): {exc}")

    units = os.environ.get("UNITS", "miles").strip().lower()
    if units not in UNIT_CONVERSIONS:
        sys.exit(f"Invalid UNITS '{units}' in .env - choose one of: {', '.join(UNIT_CONVERSIONS)}")

    rate_units = os.environ.get("RATE_UNITS", "in").strip().lower()
    if rate_units not in RATE_UNIT_CONVERSIONS:
        sys.exit(
            f"Invalid RATE_UNITS '{rate_units}' in .env - choose one of: "
            f"{', '.join(RATE_UNIT_CONVERSIONS)}"
        )

    return lat, lon, units, rate_units


def convert_distance(distance_km, units):
    factor, label = UNIT_CONVERSIONS[units]
    return distance_km * factor, label


def convert_rate(rate_mmhr, rate_units):
    factor, label = RATE_UNIT_CONVERSIONS[rate_units]
    return rate_mmhr * factor, label


def get_local_radar(lat, lon):
    contact = os.environ.get("CONTACT_EMAIL", "example@example.com")
    headers = {"User-Agent": f"(nearest-precip-script, {contact})"}
    resp = requests.get(NWS_POINTS_URL.format(lat=lat, lon=lon), headers=headers, timeout=15)
    resp.raise_for_status()
    props = resp.json()["properties"]
    return {
        "radar_station": props.get("radarStation"),
        "forecast_office": props.get("cwa"),
        "city": props.get("relativeLocation", {}).get("properties", {}).get("city"),
        "state": props.get("relativeLocation", {}).get("properties", {}).get("state"),
    }


def load_cache_meta():
    try:
        with open(CACHE_META_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def fetch_mrms_precip_rate():
    os.makedirs(CACHE_DIR, exist_ok=True)
    meta = load_cache_meta()

    headers = {}
    if meta and os.path.exists(CACHE_GRIB_PATH):
        headers["If-Modified-Since"] = meta["last_modified"]

    resp = requests.get(MRMS_URL, headers=headers, timeout=60)
    resp.raise_for_status()

    if resp.status_code == 304:
        print("NOAA has no newer data yet - using cached download.")
        grib_path = CACHE_GRIB_PATH
    else:
        raw = gzip.decompress(resp.content)
        with open(CACHE_GRIB_PATH, "wb") as f:
            f.write(raw)
        last_modified = resp.headers.get("Last-Modified")
        with open(CACHE_META_PATH, "w") as f:
            json.dump({"last_modified": last_modified}, f)
        grib_path = CACHE_GRIB_PATH

    import cfgrib

    ds = cfgrib.open_dataset(grib_path)
    lats = ds.latitude.values
    lons = ds.longitude.values  # 0-360
    data = ds["unknown"].values  # mm/hr
    valid_time = ds.valid_time.values

    return lats, lons, data, valid_time


def haversine_km(lat0, lon0, lats, lons):
    R = 6371.0
    phi1, phi2 = math.radians(lat0), np.radians(lats)
    dphi = np.radians(lats - lat0)
    dlmb = np.radians(lons - lon0)
    a = np.sin(dphi / 2) ** 2 + math.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_compass(lat0, lon0, lat1, lon1):
    phi1, phi2 = math.radians(lat0), math.radians(lat1)
    dlmb = math.radians(lon1 - lon0)
    x = math.sin(dlmb) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlmb)
    deg = (math.degrees(math.atan2(x, y)) + 360) % 360
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return dirs[round(deg / 22.5) % 16]


def find_nearest_precip(lat, lon, lats, lons, data):
    lon_360 = lon % 360

    for half_width in BBOX_STEPS_DEG:
        lat_idx = np.where(np.abs(lats - lat) < half_width)[0]
        lon_idx = np.where(np.abs(lons - lon_360) < half_width)[0]
        if lat_idx.size == 0 or lon_idx.size == 0:
            continue

        sub = data[np.ix_(lat_idx, lon_idx)]
        precip_mask = sub >= PRECIP_THRESHOLD_MMHR
        ii, jj = np.where(precip_mask)
        if ii.size == 0:
            continue

        sub_lats = lats[lat_idx][ii]
        sub_lons = lons[lon_idx][jj]
        dist = haversine_km(lat, lon_360, sub_lats, sub_lons)
        k = np.argmin(dist)

        hit_lat = float(sub_lats[k])
        hit_lon = float(sub_lons[k]) - 360  # back to -180..180
        return {
            "lat": hit_lat,
            "lon": hit_lon,
            "distance_km": float(dist[k]),
            "rate_mmhr": float(sub[ii[k], jj[k]]),
            "bearing": bearing_compass(lat, lon, hit_lat, hit_lon),
        }

    return None


def main():
    lat, lon, units, rate_units = load_point()

    print(f"Point: {lat:.4f}, {lon:.4f}")

    try:
        radar = get_local_radar(lat, lon)
        print(
            f"Local NWS radar: {radar['radar_station']} "
            f"(WFO {radar['forecast_office']}, near {radar['city']}, {radar['state']})"
        )
    except requests.RequestException as exc:
        print(f"Could not look up local radar station: {exc}")

    print("Fetching MRMS national radar mosaic (PrecipRate)...")
    lats, lons, data, valid_time = fetch_mrms_precip_rate()
    print(f"Mosaic valid: {valid_time}")

    hit = find_nearest_precip(lat, lon, lats, lons, data)
    if hit is None:
        print("No precipitation detected anywhere in the search radius.")
        return

    distance, unit_label = convert_distance(hit["distance_km"], units)
    rate, rate_label = convert_rate(hit["rate_mmhr"], rate_units)
    print(
        f"Nearest precip: {distance:.1f} {unit_label} {hit['bearing']} of you "
        f"at {hit['lat']:.4f}, {hit['lon']:.4f} "
        f"(rate {rate:.2f} {rate_label})"
    )


if __name__ == "__main__":
    main()
