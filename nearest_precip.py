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

import argparse
import collections
import copy
import datetime
import gzip
import json
import math
import os
import shutil
import sys
import textwrap
import threading
import time

import numpy as np
import requests
from dotenv import load_dotenv
from tqdm import tqdm

VERBOSE = False
APP_STATE = None  # set while running in --watch dashboard mode


def vprint(msg):
    if VERBOSE:
        print(msg)


def status(msg):
    """Report a download/processing status message. In one-shot mode this is
    just vprint (gated by --verbose); in --watch mode it goes into the
    dashboard's activity log instead of stdout, since the screen is being
    redrawn wholesale."""
    if APP_STATE is not None:
        APP_STATE.log(msg)
    else:
        vprint(msg)

NWS_POINTS_URL = "https://api.weather.gov/points/{lat},{lon}"
MRMS_URL = "https://mrms.ncep.noaa.gov/data/2D/PrecipRate/MRMS_PrecipRate.latest.grib2.gz"
NO_COVERAGE = -3.0  # MRMS sentinel value for "no radar coverage" at a cell
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
CACHE_GRIB_PATH = os.path.join(CACHE_DIR, "mrms_precip_rate.grib2")
CACHE_META_PATH = os.path.join(CACHE_DIR, "mrms_precip_rate.meta.json")
HISTORY_PATH = os.path.join(CACHE_DIR, "last_observation.json")
PRECIP_THRESHOLD_MMHR = 0.1  # anything at/above this counts as "precip"
BBOX_STEPS_DEG = [1, 2, 4, 8, 16, 30]  # progressively widen the search
# MRMS grid spacing is ~1km; treat smaller changes as noise, not real motion.
NOISE_FLOOR_KM = 1.0
KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON_AT_EQUATOR = 111.320
# A storm whose projected track passes farther than this from the point is
# treated as "will miss" rather than "will impact" - an approximation of a
# convective cell's footprint radius, not a precise threat radius.
IMPACT_MISS_THRESHOLD_KM = 15.0
# How wide a compass cone to search when looking for a trailing storm behind
# the primary one, along the line it's moving away from.
SECONDARY_SEARCH_HALF_ANGLE_DEG = 30.0

MAP_ROWS = 23
MAP_COLS = 45
# mm/hr threshold, fill char, 256-color code - loosely mirrors the classic
# green/yellow/orange/red/magenta NWS reflectivity color scale.
MAP_LEVELS = [
    (2.0, "░", 34),    # light - green
    (6.0, "▒", 226),   # moderate - yellow
    (12.0, "▓", 208),  # heavy - orange
    (25.0, "█", 196),  # very heavy - red
    (float("inf"), "█", 201),  # extreme - magenta
]

DASHBOARD_MIN_COLS = 80
DASHBOARD_MIN_ROWS = 50
# Radar map on top, primary/secondary storm boxes side by side below it,
# status bar across the bottom - each row of the layout sums to 80 cols.
MAP_BOX_WIDTH = MAP_COLS + 2
STORM_BOX_WIDTH = DASHBOARD_MIN_COLS // 2   # 40 + 40 = 80
STORM_BOX_HEIGHT = 11
STATUS_BOX_HEIGHT = 10
STATUS_LOG_MAXLEN = 5

# distance is stored internally in km; "knots" isn't really a distance unit
# (it's nautical miles per hour), but we treat it as shorthand for nautical
# miles here since that's what people asking for knots actually want.
UNIT_CONVERSIONS = {
    "miles": (0.621371, "mi"),
    "km": (1.0, "km"),
    "knots": (0.539957, "nmi"),
}

SPEED_UNIT_LABELS = {
    "miles": "mph",
    "km": "km/h",
    "knots": "kts",
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


def convert_speed(speed_kmh, units):
    factor, _ = UNIT_CONVERSIONS[units]
    return speed_kmh * factor, SPEED_UNIT_LABELS[units]


def format_duration(hours):
    total_minutes = round(hours * 60)
    if total_minutes < 60:
        return f"{total_minutes} min"
    h, m = divmod(total_minutes, 60)
    return f"{h}h {m}m"


def describe_track(t_hours, miss_km, units):
    """Turn a project_track() result into (will_impact, human-readable line)."""
    if t_hours is None:
        return False, "not moving - can't project a track"

    miss_display, unit_label = convert_distance(miss_km, units)
    if t_hours <= 0:
        return False, f"already past closest approach (came within ~{miss_display:.1f} {unit_label})"

    will_impact = miss_km <= IMPACT_MISS_THRESHOLD_KM
    if will_impact:
        return True, f"ON TRACK to reach you - ETA {format_duration(t_hours)} (closest approach ~{miss_display:.1f} {unit_label})"
    return False, f"will likely PASS BY - closest approach ~{miss_display:.1f} {unit_label} in {format_duration(t_hours)}"


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


def download_mrms(headers):
    """Stream one attempt at the MRMS file. Returns (status_code, raw_bytes_or_None, last_modified)."""
    resp = requests.get(MRMS_URL, headers=headers, stream=True, timeout=60)
    resp.raise_for_status()

    if resp.status_code == 304:
        return 304, None, None

    status("[download] New data available - downloading MRMS mosaic...")
    total = int(resp.headers.get("Content-Length", 0))
    chunks = []
    # A live tqdm bar would corrupt a full-screen --watch redraw, so it's
    # only shown in one-shot --verbose mode.
    with tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024,
              desc="downloading", disable=(APP_STATE is not None) or not VERBOSE) as bar:
        for chunk in resp.iter_content(chunk_size=64 * 1024):
            chunks.append(chunk)
            bar.update(len(chunk))
    compressed = b"".join(chunks)

    status("[process]  Decompressing download...")
    # NOAA overwrites the "latest" file in place (not an atomic swap), so a
    # download can occasionally straddle an update and land here truncated.
    raw = gzip.decompress(compressed)
    return 200, raw, resp.headers.get("Last-Modified")


def fetch_mrms_precip_rate():
    os.makedirs(CACHE_DIR, exist_ok=True)
    meta = load_cache_meta()

    headers = {}
    if meta and os.path.exists(CACHE_GRIB_PATH):
        headers["If-Modified-Since"] = meta["last_modified"]

    status("[checking] Asking NOAA if the MRMS mosaic has updated...")

    attempts = 3
    for attempt in range(1, attempts + 1):
        if attempt > 1:
            status(f"[download] Retrying download (attempt {attempt}/{attempts})...")
        try:
            http_status, raw, last_modified = download_mrms(headers)
            break
        except (requests.RequestException, gzip.BadGzipFile, EOFError) as exc:
            status(f"[download] Download failed: {exc}")
            if attempt == attempts:
                if os.path.exists(CACHE_GRIB_PATH):
                    status("[cache]    Giving up - falling back to last good cached download.")
                    http_status, raw, last_modified = 304, None, None
                    break
                sys.exit("Could not download MRMS data and no cached copy is available.")

    if http_status == 304:
        status("[cache]    No newer data yet - reusing cached download.")
        grib_path = CACHE_GRIB_PATH
    else:
        with open(CACHE_GRIB_PATH, "wb") as f:
            f.write(raw)
        with open(CACHE_META_PATH, "w") as f:
            json.dump({"last_modified": last_modified}, f)
        grib_path = CACHE_GRIB_PATH

    status("[process]  Parsing grib data locally...")
    import cfgrib

    ds = cfgrib.open_dataset(grib_path)
    lats = ds.latitude.values
    lons = ds.longitude.values  # 0-360
    data = ds["unknown"].values  # mm/hr
    valid_time = ds.valid_time.values

    is_new = http_status != 304
    return lats, lons, data, valid_time, is_new


def haversine_km(lat0, lon0, lats, lons):
    R = 6371.0
    phi1, phi2 = math.radians(lat0), np.radians(lats)
    dphi = np.radians(lats - lat0)
    dlmb = np.radians(lons - lon0)
    a = np.sin(dphi / 2) ** 2 + math.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_degrees(lat0, lon0, lat1, lon1):
    phi1, phi2 = math.radians(lat0), math.radians(lat1)
    dlmb = math.radians(lon1 - lon0)
    x = math.sin(dlmb) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlmb)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def bearing_compass(lat0, lon0, lat1, lon1):
    deg = bearing_degrees(lat0, lon0, lat1, lon1)
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return dirs[round(deg / 22.5) % 16]


def bearing_degrees_np(lat0, lon0, lats, lons):
    phi1, phi2 = math.radians(lat0), np.radians(lats)
    dlmb = np.radians(lons - lon0)
    x = np.sin(dlmb) * np.cos(phi2)
    y = math.cos(phi1) * np.sin(phi2) - math.sin(phi1) * np.cos(phi2) * np.cos(dlmb)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def angle_diff_deg(a, b):
    """Smallest angular difference between two compass bearings, in [0, 180]."""
    return np.abs((a - b + 180) % 360 - 180)


def project_track(cell_lat, cell_lon, user_lat, user_lon, bearing_deg, speed_kmh):
    """Assume the cell keeps moving in a straight line at (bearing_deg, speed_kmh).
    Returns (hours_to_closest_approach, miss_distance_km), or (None, None) if
    the cell isn't moving. A negative hours value means the closest approach
    is already behind it (it's moving away along its track)."""
    if speed_kmh <= 0:
        return None, None

    dx = (cell_lon - user_lon) * KM_PER_DEG_LON_AT_EQUATOR * math.cos(math.radians(user_lat))
    dy = (cell_lat - user_lat) * KM_PER_DEG_LAT
    vx = math.sin(math.radians(bearing_deg))
    vy = math.cos(math.radians(bearing_deg))

    t_hours = -(dx * vx + dy * vy) / speed_kmh
    closest_x = dx + t_hours * speed_kmh * vx
    closest_y = dy + t_hours * speed_kmh * vy
    miss_distance_km = math.hypot(closest_x, closest_y)
    return t_hours, miss_distance_km


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


def find_precip_in_sector(lat, lon, lats, lons, data, center_bearing_deg, half_angle_deg, exclude_lat, exclude_lon):
    """Nearest precip cell within a compass cone, skipping the cell at
    (exclude_lat, exclude_lon) - used to look for a trailing storm behind the
    primary one instead of re-finding it."""
    lon_360 = lon % 360
    exclude_lon_360 = exclude_lon % 360

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

        not_excluded = haversine_km(exclude_lat, exclude_lon_360, sub_lats, sub_lons) > NOISE_FLOOR_KM
        bearings = bearing_degrees_np(lat, lon_360, sub_lats, sub_lons)
        in_sector = angle_diff_deg(bearings, center_bearing_deg) <= half_angle_deg
        candidates = np.where(not_excluded & in_sector)[0]
        if candidates.size == 0:
            continue

        cand_dist = haversine_km(lat, lon_360, sub_lats[candidates], sub_lons[candidates])
        best = np.argmin(cand_dist)
        k = candidates[best]

        hit_lat = float(sub_lats[k])
        hit_lon = float(sub_lons[k]) - 360
        return {
            "lat": hit_lat,
            "lon": hit_lon,
            "distance_km": float(cand_dist[best]),
            "rate_mmhr": float(sub[ii[k], jj[k]]),
            "bearing": bearing_compass(lat, lon, hit_lat, hit_lon),
        }

    return None


def crop_grid(lat, lon, lats, lons, data, radius_km):
    lon_360 = lon % 360
    lat_buffer = (radius_km / KM_PER_DEG_LAT) * 1.5
    lon_buffer = (radius_km / (KM_PER_DEG_LON_AT_EQUATOR * math.cos(math.radians(lat)))) * 1.5

    lat_idx = np.where(np.abs(lats - lat) < lat_buffer)[0]
    lon_idx = np.where(np.abs(lons - lon_360) < lon_buffer)[0]
    return lats[lat_idx], lons[lon_idx], data[np.ix_(lat_idx, lon_idx)]


def level_for_rate(rate_mmhr):
    for threshold, char, color in MAP_LEVELS:
        if rate_mmhr < threshold:
            return char, color
    return MAP_LEVELS[-1][1], MAP_LEVELS[-1][2]


def map_legend_line():
    return (
        "@ = you   "
        "\x1b[38;5;34m░\x1b[0m light   "
        "\x1b[38;5;226m▒\x1b[0m moderate   "
        "\x1b[38;5;208m▓\x1b[0m heavy   "
        "\x1b[38;5;196m█\x1b[0m very heavy   "
        "\x1b[38;5;201m█\x1b[0m extreme"
    )


def build_map_box_lines(lat, lon, lats, lons, data, radius_miles):
    """The bordered radar grid only (no title, no legend) - exactly
    MAP_ROWS + 2 lines, so it can be lined up next to other fixed-height
    boxes in the --watch dashboard layout."""
    radius_km = radius_miles * 1.60934
    crop_lats, crop_lons, crop_data = crop_grid(lat, lon, lats, lons, data, radius_km)

    row_center = MAP_ROWS // 2
    col_center = MAP_COLS // 2
    km_per_row = (2 * radius_km) / (MAP_ROWS - 1)
    km_per_col = km_per_row / 2  # terminal chars are roughly twice as tall as wide

    lines = ["┌" + "─" * MAP_COLS + "┐"]
    for r in range(MAP_ROWS):
        dy_km = (row_center - r) * km_per_row
        cell_lat = lat + dy_km / KM_PER_DEG_LAT
        row_chars = []
        for c in range(MAP_COLS):
            if r == row_center and c == col_center:
                row_chars.append("\x1b[1;97m@\x1b[0m")
                continue

            dx_km = (c - col_center) * km_per_col
            cell_lon = lon + dx_km / (KM_PER_DEG_LON_AT_EQUATOR * math.cos(math.radians(lat)))
            cell_lon_360 = cell_lon % 360

            if crop_lats.size == 0 or crop_lons.size == 0:
                row_chars.append(" ")
                continue

            lat_idx = np.argmin(np.abs(crop_lats - cell_lat))
            lon_idx = np.argmin(np.abs(crop_lons - cell_lon_360))
            rate = crop_data[lat_idx, lon_idx]

            if rate <= NO_COVERAGE or rate < PRECIP_THRESHOLD_MMHR:
                row_chars.append(" ")
            else:
                char, color = level_for_rate(rate)
                row_chars.append(f"\x1b[38;5;{color}m{char}\x1b[0m")
        lines.append("│" + "".join(row_chars) + "│")

    lines.append("└" + "─" * MAP_COLS + "┘")
    return lines


def render_ansi_map(lat, lon, lats, lons, data, radius_miles):
    lines = [f"Radar mosaic within {radius_miles:.0f} mi of {lat:.4f}, {lon:.4f} (N up):"]
    lines.extend(build_map_box_lines(lat, lon, lats, lons, data, radius_miles))
    lines.append(map_legend_line())
    return "\n".join(lines)


def make_box(title, content_lines, width, height):
    """Render a bordered box of an exact width/height (borders included).
    Text lines are word-wrapped to fit and truncated/padded with blanks to
    fit the fixed height, so boxes always line up in the dashboard grid."""
    interior_w = width - 2
    interior_h = height - 2

    body = [title.ljust(interior_w)[:interior_w], "-" * interior_w]
    for line in content_lines:
        body.extend(textwrap.wrap(line, interior_w) or [""])

    body = body[:interior_h]
    while len(body) < interior_h:
        body.append("")

    box = ["┌" + "─" * interior_w + "┐"]
    for line in body:
        box.append("│" + line.ljust(interior_w)[:interior_w] + "│")
    box.append("└" + "─" * interior_w + "┘")
    return box


def load_last_observation():
    try:
        with open(HISTORY_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_observation(lat, lon, hit, valid_time):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump(
            {
                "user_lat": lat,
                "user_lon": lon,
                "hit_lat": hit["lat"],
                "hit_lon": hit["lon"],
                "distance_km": hit["distance_km"],
                "valid_time": str(valid_time),
            },
            f,
        )


def analyze_movement(prev, lat, lon, hit, valid_time):
    """Compare this observation to the last saved one. Returns None if there's
    no usable baseline (first run, point changed, or no new radar scan yet)."""
    if prev is None:
        return None
    if abs(prev["user_lat"] - lat) > 1e-6 or abs(prev["user_lon"] - lon) > 1e-6:
        return None  # .env point changed - old history isn't comparable

    prev_time = np.datetime64(prev["valid_time"])
    elapsed_hours = (valid_time - prev_time) / np.timedelta64(1, "h")
    if elapsed_hours <= 0:
        return None  # same or older radar scan as last time - nothing new to compare

    cell_move_km = float(haversine_km(prev["hit_lat"], prev["hit_lon"] % 360,
                                       hit["lat"], hit["lon"] % 360))
    speed_kmh = cell_move_km / elapsed_hours
    move_bearing = bearing_degrees(prev["hit_lat"], prev["hit_lon"], hit["lat"], hit["lon"])

    distance_change_km = hit["distance_km"] - prev["distance_km"]
    if abs(distance_change_km) < NOISE_FLOOR_KM:
        trend = "steady"
    elif distance_change_km < 0:
        trend = "approaching"
    else:
        trend = "receding"

    eta_hours = None
    if trend == "approaching":
        closing_speed_kmh = -distance_change_km / elapsed_hours
        if closing_speed_kmh > 0:
            eta_hours = hit["distance_km"] / closing_speed_kmh

    return {
        "trend": trend,
        "bearing_deg": move_bearing,
        "speed_kmh": speed_kmh,
        "eta_hours": eta_hours,
        "elapsed_hours": float(elapsed_hours),
    }


class AppState:
    """Shared between the background fetch/analysis thread and the render
    loop. All reads/writes go through the lock; render() takes a shallow
    snapshot so it never blocks the worker mid-cycle."""

    def __init__(self):
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.busy = True
        self.log_entries = collections.deque(maxlen=STATUS_LOG_MAXLEN)
        self.last_checked = None
        self.last_updated = None  # radar mosaic's own valid_time
        self.next_check_at = None
        self.error = None

        self.lats = None
        self.lons = None
        self.data = None
        self.hit = None
        self.movement = None
        self.primary_track = None      # (will_impact: bool, description: str)
        self.secondary = None
        self.secondary_track = None    # description: str

    def log(self, message):
        with self.lock:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            self.log_entries.append(f"{timestamp} {message}")

    def set_busy(self, busy):
        with self.lock:
            self.busy = busy

    def set_next_check(self, ts):
        with self.lock:
            self.next_check_at = ts

    def set_error(self, message):
        with self.lock:
            self.error = message

    def update_result(self, **fields):
        with self.lock:
            for key, value in fields.items():
                setattr(self, key, value)
            self.last_checked = datetime.datetime.now()
            self.error = None

    def mark_checked(self):
        """A check ran but there was no new radar scan - record that we
        checked without touching the existing hit/movement results."""
        with self.lock:
            self.last_checked = datetime.datetime.now()
            self.error = None

    def snapshot(self):
        with self.lock:
            return copy.copy(self)


def run_analysis_cycle(state, lat, lon, units, rate_units):
    lats, lons, data, valid_time, is_new = fetch_mrms_precip_rate()

    if not is_new:
        # No new radar scan since last check - leave the existing hit/movement
        # results alone rather than recomputing (and rewriting history) off
        # data that hasn't changed.
        state.mark_checked()
        return

    prev = load_last_observation()
    hit = find_nearest_precip(lat, lon, lats, lons, data)
    movement = None
    primary_track = None
    secondary = None
    secondary_track = None

    if hit is not None:
        movement = analyze_movement(prev, lat, lon, hit, valid_time)
        if movement is not None:
            t_hours, miss_km = project_track(
                hit["lat"], hit["lon"], lat, lon, movement["bearing_deg"], movement["speed_kmh"]
            )
            will_impact, track_desc = describe_track(t_hours, miss_km, units)
            primary_track = (will_impact, track_desc)

            if not will_impact:
                search_bearing = (movement["bearing_deg"] + 180) % 360
                secondary = find_precip_in_sector(
                    lat, lon, lats, lons, data,
                    search_bearing, SECONDARY_SEARCH_HALF_ANGLE_DEG,
                    hit["lat"], hit["lon"],
                )
                if secondary is not None:
                    t2_hours, miss2_km = project_track(
                        secondary["lat"], secondary["lon"], lat, lon,
                        movement["bearing_deg"], movement["speed_kmh"],
                    )
                    _, secondary_track = describe_track(t2_hours, miss2_km, units)

        save_observation(lat, lon, hit, valid_time)

    state.update_result(
        lats=lats, lons=lons, data=data,
        hit=hit, movement=movement, primary_track=primary_track,
        secondary=secondary, secondary_track=secondary_track,
        last_updated=valid_time,
    )


def worker_loop(state, lat, lon, units, rate_units, interval_seconds):
    while not state.stop_event.is_set():
        state.set_busy(True)
        try:
            run_analysis_cycle(state, lat, lon, units, rate_units)
        except Exception as exc:  # keep the dashboard alive on transient errors
            state.set_error(str(exc))
            state.log(f"[error] {exc}")
        state.set_busy(False)
        state.set_next_check(time.time() + interval_seconds)
        state.stop_event.wait(interval_seconds)


def build_primary_box_lines(units, rate_units, snapshot):
    if snapshot.hit is None:
        return make_box("PRIMARY STORM", ["No precipitation detected in range."],
                         STORM_BOX_WIDTH, STORM_BOX_HEIGHT)

    hit = snapshot.hit
    distance, unit_label = convert_distance(hit["distance_km"], units)
    rate, rate_label = convert_rate(hit["rate_mmhr"], rate_units)
    lines = [
        f"{distance:.1f} {unit_label} {hit['bearing']} of you",
        f"Pos {hit['lat']:.3f}, {hit['lon']:.3f}",
        f"Rate {rate:.2f} {rate_label}",
    ]
    if snapshot.movement is None:
        lines.append("Movement: gathering history...")
    else:
        speed, speed_label = convert_speed(snapshot.movement["speed_kmh"], units)
        lines.append(f"Heading {snapshot.movement['bearing_deg']:.0f} deg @ {speed:.1f} {speed_label}")
        if snapshot.primary_track is not None:
            lines.append(f"Track: {snapshot.primary_track[1]}")
    return make_box("PRIMARY STORM", lines, STORM_BOX_WIDTH, STORM_BOX_HEIGHT)


def build_secondary_box_lines(units, rate_units, snapshot):
    if snapshot.primary_track is not None and snapshot.primary_track[0]:
        lines = ["Primary storm is on track -", "no need to look for a trailing one."]
    elif snapshot.secondary is None:
        lines = ["None found upstream of the", "primary storm's track."]
    else:
        sec = snapshot.secondary
        sdist, sunit = convert_distance(sec["distance_km"], units)
        srate, sratelabel = convert_rate(sec["rate_mmhr"], rate_units)
        lines = [
            f"{sdist:.1f} {sunit} {sec['bearing']} of you",
            f"Pos {sec['lat']:.3f}, {sec['lon']:.3f}",
            f"Rate {srate:.2f} {sratelabel}",
            "(assumes primary's heading/speed)",
        ]
        if snapshot.secondary_track:
            lines.append(f"Track: {snapshot.secondary_track}")
    return make_box("SECONDARY STORM", lines, STORM_BOX_WIDTH, STORM_BOX_HEIGHT)


def next_check_text(snapshot, interval_seconds):
    if snapshot.busy:
        return "checking now..."
    if snapshot.next_check_at is None:
        return f"next check in {interval_seconds:.0f}s"
    next_in = max(0, int(snapshot.next_check_at - time.time()))
    return f"next check in {next_in}s"


def build_status_box_lines(radar_info, interval_seconds, snapshot, width, height):
    last_checked = snapshot.last_checked.strftime("%H:%M:%S") if snapshot.last_checked else "-"
    last_updated = str(snapshot.last_updated) if snapshot.last_updated is not None else "-"

    lines = [
        f"Radar: {radar_info.get('radar_station')} (WFO {radar_info.get('forecast_office')}) "
        f"near {radar_info.get('city')}, {radar_info.get('state')}",
        f"Last checked {last_checked}   Radar scan valid {last_updated}",
        "Status: " + next_check_text(snapshot, interval_seconds),
    ]
    if snapshot.error:
        lines.append(f"Last error: {snapshot.error}")
    lines.append("Recent activity:")
    if snapshot.log_entries:
        lines.extend(snapshot.log_entries)
    else:
        lines.append("(nothing logged yet)")
    return make_box("STATUS", lines, width, height)


def build_frame(lat, lon, units, rate_units, radar_info, map_radius, interval_seconds, snapshot):
    header = (
        f" NEAREST PRECIP MONITOR - {lat:.4f}, {lon:.4f} - "
        f"{next_check_text(snapshot, interval_seconds)} "
    )
    lines = [header.center(DASHBOARD_MIN_COLS, "=")]

    if snapshot.lats is None:
        map_box = make_box(
            f"RADAR ({map_radius:.0f} mi, N up)", ["Waiting for first radar fetch..."],
            MAP_BOX_WIDTH, MAP_ROWS + 2,
        )
    else:
        map_box = build_map_box_lines(lat, lon, snapshot.lats, snapshot.lons, snapshot.data, map_radius)

    # Radar map on top, centered in the 80-col frame.
    map_indent = " " * ((DASHBOARD_MIN_COLS - MAP_BOX_WIDTH) // 2)
    lines.extend(map_indent + line for line in map_box)
    lines.append(map_indent + map_legend_line() if snapshot.lats is not None else "")
    lines.append("")

    # Primary and secondary storm boxes side by side below the map.
    primary_box = build_primary_box_lines(units, rate_units, snapshot)
    secondary_box = build_secondary_box_lines(units, rate_units, snapshot)
    for left, right in zip(primary_box, secondary_box):
        lines.append(left + right)
    lines.append("")

    # Status bar across the full width at the bottom.
    lines.extend(build_status_box_lines(radar_info, interval_seconds, snapshot, DASHBOARD_MIN_COLS, STATUS_BOX_HEIGHT))
    return "\n".join(lines)


def run_dashboard(lat, lon, units, rate_units, radar_info, map_radius, interval_seconds):
    term_size = shutil.get_terminal_size(fallback=(DASHBOARD_MIN_COLS, DASHBOARD_MIN_ROWS))
    if term_size.columns < DASHBOARD_MIN_COLS or term_size.lines < DASHBOARD_MIN_ROWS:
        print(
            f"Warning: terminal is {term_size.columns}x{term_size.lines}, "
            f"this dashboard is designed for at least {DASHBOARD_MIN_COLS}x{DASHBOARD_MIN_ROWS}."
        )

    state = AppState()
    global APP_STATE
    APP_STATE = state

    worker = threading.Thread(
        target=worker_loop, args=(state, lat, lon, units, rate_units, interval_seconds), daemon=True
    )
    worker.start()

    try:
        print("\x1b[?25l", end="")  # hide cursor
        while True:
            snapshot = state.snapshot()
            frame = build_frame(lat, lon, units, rate_units, radar_info, map_radius, interval_seconds, snapshot)
            print("\x1b[H\x1b[J" + frame, end="", flush=True)
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        state.stop_event.set()
        worker.join(timeout=5)
        print("\x1b[?25h")  # restore cursor
        print("\nStopped.")


def main():
    global VERBOSE
    parser = argparse.ArgumentParser(description="Find the closest active precipitation to a GPS point.")
    parser.add_argument("-v", "--verbose", action="store_true",
                         help="show download/cache/processing status and a download progress bar")
    parser.add_argument("--map", action="store_true",
                         help="draw an ANSI radar map centered on your point")
    parser.add_argument("--map-radius", type=float, default=25.0,
                         help="radius in miles for --map (default: 25)")
    parser.add_argument("--watch", action="store_true",
                         help="run as a continuously-updating full-screen dashboard")
    parser.add_argument("--interval", type=float, default=60.0,
                         help="seconds between radar checks in --watch mode (default: 60)")
    args = parser.parse_args()
    VERBOSE = args.verbose

    lat, lon, units, rate_units = load_point()

    print(f"Point: {lat:.4f}, {lon:.4f}")

    vprint("[download] Looking up local NWS radar station...")
    try:
        radar = get_local_radar(lat, lon)
        print(
            f"Local NWS radar: {radar['radar_station']} "
            f"(WFO {radar['forecast_office']}, near {radar['city']}, {radar['state']})"
        )
    except requests.RequestException as exc:
        radar = {"radar_station": "?", "forecast_office": "?", "city": "?", "state": "?"}
        print(f"Could not look up local radar station: {exc}")

    if args.watch:
        run_dashboard(lat, lon, units, rate_units, radar, args.map_radius, args.interval)
        return

    lats, lons, data, valid_time, _is_new = fetch_mrms_precip_rate()
    vprint(f"[process]  Mosaic valid: {valid_time}")

    if args.map:
        print()
        print(render_ansi_map(lat, lon, lats, lons, data, args.map_radius))
        print()

    prev = load_last_observation()

    vprint("[process]  Searching locally for the nearest precip echo...")
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

    movement = analyze_movement(prev, lat, lon, hit, valid_time)
    if movement is None:
        print("Movement: not enough history yet - run this again after the next radar update.")
    else:
        speed, speed_label = convert_speed(movement["speed_kmh"], units)
        print(
            f"Movement: heading {movement['bearing_deg']:.0f} deg at {speed:.1f} {speed_label} "
            f"({movement['trend']} - based on last observation)"
        )

        t_hours, miss_km = project_track(
            hit["lat"], hit["lon"], lat, lon, movement["bearing_deg"], movement["speed_kmh"]
        )
        will_impact, track_desc = describe_track(t_hours, miss_km, units)
        print(f"Track: {track_desc}")

        if not will_impact:
            # This storm's own path will miss us (could be the parallel-track
            # case) - check behind it, along the line it's moving away from,
            # for a trailing storm that might actually be headed our way.
            search_bearing = (movement["bearing_deg"] + 180) % 360
            secondary = find_precip_in_sector(
                lat, lon, lats, lons, data,
                search_bearing, SECONDARY_SEARCH_HALF_ANGLE_DEG,
                hit["lat"], hit["lon"],
            )
            if secondary is None:
                print(
                    f"Secondary storm: none found upstream "
                    f"(within {SECONDARY_SEARCH_HALF_ANGLE_DEG:.0f} deg of {search_bearing:.0f} deg)."
                )
            else:
                sdist, sunit_label = convert_distance(secondary["distance_km"], units)
                srate, srate_label = convert_rate(secondary["rate_mmhr"], rate_units)
                print(
                    f"Secondary storm: {sdist:.1f} {sunit_label} {secondary['bearing']} of you "
                    f"at {secondary['lat']:.4f}, {secondary['lon']:.4f} "
                    f"(rate {srate:.2f} {srate_label})"
                )
                # Assumes the trailing cell shares the primary's heading/speed
                # (typical for cells in the same line/system) since we don't
                # have a second observation of it yet to measure its own.
                t2_hours, miss2_km = project_track(
                    secondary["lat"], secondary["lon"], lat, lon,
                    movement["bearing_deg"], movement["speed_kmh"],
                )
                _, track_desc2 = describe_track(t2_hours, miss2_km, units)
                print(f"Secondary track (assumes same heading/speed as primary): {track_desc2}")

    save_observation(lat, lon, hit, valid_time)


if __name__ == "__main__":
    main()
