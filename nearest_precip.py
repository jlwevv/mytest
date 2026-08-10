#!/usr/bin/env python3
"""
Estimate when rain will start and end at a GPS point.

Point is read from .env (LAT/LON). The local NWS radar station covering
that point is looked up via api.weather.gov purely for display; the actual
analysis comes from NOAA's MRMS PrecipRate product, a national radar mosaic
built from all NEXRAD sites. A rolling window of recent PrecipRate frames
(backfilled from NOAA's MRMS archive on first use, then kept warm locally)
is used to estimate how the precip field near the point is moving, which is
then advected forward in time to project when rain will arrive at and
depart from the point itself.
"""

import argparse
import collections
import copy
import dataclasses
import datetime
import glob
import gzip
import json
import math
import os
import re
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
MRMS_ARCHIVE_DIR_URL = "https://mrms.ncep.noaa.gov/data/2D/PrecipRate/"
ARCHIVE_FILENAME_RE = re.compile(r'href="(MRMS_PrecipRate_00\.00_(\d{8})-(\d{6})\.grib2\.gz)"')
NO_COVERAGE = -3.0  # MRMS sentinel value for "no radar coverage" at a cell
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
CACHE_GRIB_PATH = os.path.join(CACHE_DIR, "mrms_precip_rate.grib2")
CACHE_META_PATH = os.path.join(CACHE_DIR, "mrms_precip_rate.meta.json")
FRAMES_DIR = os.path.join(CACHE_DIR, "frames")
FRAMES_INDEX_PATH = os.path.join(CACHE_DIR, "frames_index.json")
FRAME_INTERVAL_MIN = 2.0  # MRMS's native publishing cadence
FRAME_PRUNE_BUFFER_MIN = 10.0  # keep a little past the lookback window before deleting
PRECIP_THRESHOLD_MMHR = 0.1  # anything at/above this counts as "precip"
# MRMS grid spacing is ~1km; treat smaller changes as noise, not real motion.
NOISE_FLOOR_KM = 1.0
KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON_AT_EQUATOR = 111.320
# Assumed upper bound on how fast a precip field can realistically move, used
# to size the crop used for motion estimation and to bound the archive
# backfill window - not a precise physical limit, just a generous cap.
MAX_STORM_SPEED_KMH_ASSUMPTION = 110.0
CROP_RADIUS_BUFFER = 1.3
# Phase correlation on a field with little/no actual precip in it just
# correlates numerical noise and produces a plausible-looking but bogus
# velocity - require at least this many above-threshold cells in a frame
# before trusting a motion estimate derived from it.
MIN_SIGNAL_CELLS = 25
# TREC-style local block correlation (see estimate_motion_field): rather than
# one whole-field correlation - which can lock onto a distant, unrelated
# storm cell if one falls inside the crop - several small blocks clustered
# around the point are each correlated independently and combined.
BLOCK_OFFSET_FRACTION = 0.5  # side-block offset from center, as a fraction of crop_radius_km
BLOCK_OFFSET_BEARINGS_DEG = (0.0, 90.0, 180.0, 270.0)  # N, E, S, W side blocks around the center
FORECAST_STEP_MIN = 2.0  # matches MRMS cadence

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
# Radar map on top, rain status / storm motion boxes side by side below it,
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


@dataclasses.dataclass(frozen=True)
class Config:
    lat: float
    lon: float
    units: str
    rate_units: str
    lookback_min: float
    forecast_min: float


def _positive_float(name, default):
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        sys.exit(f"Invalid {name} '{raw}' in .env - must be a number.")
    if value <= 0:
        sys.exit(f"Invalid {name} '{raw}' in .env - must be greater than 0.")
    return value


def load_config():
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

    lookback_min = _positive_float("LOOKBACK_MIN", 30.0)
    forecast_min = _positive_float("FORECAST_MIN", 90.0)

    return Config(lat, lon, units, rate_units, lookback_min, forecast_min)


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


def format_eta_minutes(minutes):
    return format_duration(minutes / 60.0)


def clock_time_str(base_utc, minutes):
    """Format base_utc + minutes as a local clock time, e.g. '3:45 PM'."""
    target = base_utc + datetime.timedelta(minutes=minutes)
    return target.astimezone().strftime("%-I:%M %p")


def format_eta(base_utc, minutes):
    """'~45 min (3:45 PM)' - duration plus the local clock time it lands on."""
    return f"{format_eta_minutes(minutes)} ({clock_time_str(base_utc, minutes)})"


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


def to_utc_datetime(valid_time):
    """Normalize a numpy.datetime64 (as returned by cfgrib) or a datetime
    into a timezone-aware UTC datetime."""
    if isinstance(valid_time, np.datetime64):
        return valid_time.astype("datetime64[s]").astype(datetime.datetime).replace(
            tzinfo=datetime.timezone.utc
        )
    if valid_time.tzinfo is None:
        return valid_time.replace(tzinfo=datetime.timezone.utc)
    return valid_time.astimezone(datetime.timezone.utc)


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


# --- rolling radar-frame history cache ---


def frame_key(valid_time_utc):
    return valid_time_utc.strftime("%Y%m%d-%H%M%S")


def _frame_grib_path(key):
    return os.path.join(FRAMES_DIR, f"{key}.grib2")


def load_frames_index():
    try:
        with open(FRAMES_INDEX_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_frames_index(index):
    os.makedirs(FRAMES_DIR, exist_ok=True)
    with open(FRAMES_INDEX_PATH, "w") as f:
        json.dump(index, f)


def prune_frames(index, lookback_min, now=None):
    """Delete cached frames (and their cfgrib .idx sidecars) older than
    lookback_min + FRAME_PRUNE_BUFFER_MIN. Returns the pruned index."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    cutoff = now - datetime.timedelta(minutes=lookback_min + FRAME_PRUNE_BUFFER_MIN)

    kept = {}
    for key, entry in index.items():
        valid_time = datetime.datetime.fromisoformat(entry["valid_time"])
        if valid_time >= cutoff:
            kept[key] = entry
        else:
            # cfgrib's .idx sidecar filename includes a hash suffix we don't
            # know ahead of time - glob for anything with this frame's stem.
            for path in glob.glob(_frame_grib_path(key) + "*"):
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
    return kept


def store_frame(valid_time_utc, raw_grib_bytes, source, index):
    os.makedirs(FRAMES_DIR, exist_ok=True)
    key = frame_key(valid_time_utc)
    with open(_frame_grib_path(key), "wb") as f:
        f.write(raw_grib_bytes)
    index[key] = {"valid_time": valid_time_utc.isoformat(), "source": source}
    return key


def open_frame(key):
    import cfgrib

    ds = cfgrib.open_dataset(_frame_grib_path(key))
    lats = ds.latitude.values
    lons = ds.longitude.values
    data = ds["unknown"].values
    valid_time = ds.valid_time.values
    return lats, lons, data, valid_time


def list_archive_frames():
    """GET NOAA's MRMS PrecipRate directory listing and regex-parse the
    timestamped frame filenames out of it (plain re, no bs4/lxml - this
    project keeps its dependency list minimal). Returns (valid_time_utc,
    filename) tuples, oldest first."""
    resp = requests.get(MRMS_ARCHIVE_DIR_URL, timeout=30)
    resp.raise_for_status()

    frames = []
    for filename, date_part, time_part in ARCHIVE_FILENAME_RE.findall(resp.text):
        valid_time = datetime.datetime.strptime(
            date_part + time_part, "%Y%m%d%H%M%S"
        ).replace(tzinfo=datetime.timezone.utc)
        frames.append((valid_time, filename))
    frames.sort(key=lambda item: item[0])
    return frames


def ensure_frame_cache(lookback_min):
    """Prune stale frames, then backfill only whatever is missing from the
    [now - lookback_min, now] window by consulting NOAA's MRMS archive. A
    warm cache (the common case - most runs just add one new "latest" frame
    each cycle) needs no network access here at all."""
    index = load_frames_index()
    index = prune_frames(index, lookback_min)

    now = datetime.datetime.now(datetime.timezone.utc)
    window_start = now - datetime.timedelta(minutes=lookback_min)
    have_times = sorted(
        datetime.datetime.fromisoformat(entry["valid_time"]) for entry in index.values()
    )
    # "Covered" means we already have a frame within about one interval of
    # the start of the window - not worth an archive round-trip for a gap
    # that small.
    covered = have_times and (have_times[0] - window_start) <= datetime.timedelta(
        minutes=FRAME_INTERVAL_MIN * 1.5
    )

    if not covered:
        status("[history]  Backfilling radar history from NOAA's MRMS archive...")
        try:
            archive_frames = list_archive_frames()
        except requests.RequestException as exc:
            status(f"[history]  Could not reach MRMS archive: {exc}")
            save_frames_index(index)
            return index

        wanted = [
            (valid_time, filename)
            for valid_time, filename in archive_frames
            if valid_time >= window_start and frame_key(valid_time) not in index
        ]
        for valid_time, filename in tqdm(
            wanted, desc="backfilling", disable=(APP_STATE is not None) or not VERBOSE
        ):
            try:
                resp = requests.get(MRMS_ARCHIVE_DIR_URL + filename, timeout=30)
                resp.raise_for_status()
                raw = gzip.decompress(resp.content)
            except (requests.RequestException, gzip.BadGzipFile, EOFError) as exc:
                status(f"[history]  Skipping {filename}: {exc}")
                continue
            store_frame(valid_time, raw, "archive", index)

    save_frames_index(index)
    return index


def fetch_frame_history(cfg):
    """Fetch the freshest MRMS scan (unchanged fetch_mrms_precip_rate()
    path), fold it into the rolling frame cache, then backfill/prune that
    cache against cfg.lookback_min. Returns (lats, lons, data, valid_time,
    frames, is_new) where lats/lons/data/valid_time are the freshest scan
    and frames is the oldest->newest list of {"key", "valid_time"} dicts
    currently in the cache window."""
    lats, lons, data, valid_time, is_new = fetch_mrms_precip_rate()

    index = load_frames_index()
    if is_new:
        utc_time = to_utc_datetime(valid_time)
        key = frame_key(utc_time)
        if key not in index:
            os.makedirs(FRAMES_DIR, exist_ok=True)
            shutil.copyfile(CACHE_GRIB_PATH, _frame_grib_path(key))
            index[key] = {"valid_time": utc_time.isoformat(), "source": "latest"}
            save_frames_index(index)

    index = ensure_frame_cache(cfg.lookback_min)

    frames = sorted(
        (
            {"key": key, "valid_time": datetime.datetime.fromisoformat(entry["valid_time"])}
            for key, entry in index.items()
        ),
        key=lambda item: item["valid_time"],
    )

    return lats, lons, data, valid_time, frames, is_new


# --- motion estimation / nowcasting ---


def _nearest_index_map(old_coords, new_coords):
    """Map each value in old_coords to the index of its nearest match in
    new_coords (new_coords assumed monotonic, ascending or descending).
    Fallback path for the rare case where two cached frames don't share an
    identical grid - MRMS's CONUS grid is fixed, so this should be a no-op
    in practice."""
    ascending = new_coords[-1] >= new_coords[0]
    ref = new_coords if ascending else new_coords[::-1]
    idx = np.clip(np.searchsorted(ref, old_coords), 1, len(ref) - 1)
    left, right = ref[idx - 1], ref[idx]
    idx = idx - (np.abs(old_coords - left) < np.abs(old_coords - right)).astype(int)
    if not ascending:
        idx = len(ref) - 1 - idx
    return idx


def _estimate_block_motion(lat_c, lon_c, lats, lons, data, lats_old, lons_old, data_old,
                            elapsed_hours, crop_radius_km):
    """One local block's motion vector, centered at (lat_c, lon_c), via
    windowed FFT phase correlation between the old/new frames (already
    grid-aligned by the caller). Returns {"bearing_deg", "speed_kmh",
    "signal_cells"}, or None on insufficient signal, an empty/mismatched
    crop (e.g. the block fell outside radar coverage), or an implausibly
    fast result."""
    crop_lats, crop_lons, crop_new = crop_grid(lat_c, lon_c, lats, lons, data, crop_radius_km)
    _, _, crop_old = crop_grid(lat_c, lon_c, lats_old, lons_old, data_old, crop_radius_km)

    if crop_new.shape != crop_old.shape or crop_new.size == 0:
        return None

    signal_cells = min(
        int(np.count_nonzero(crop_old >= PRECIP_THRESHOLD_MMHR)),
        int(np.count_nonzero(crop_new >= PRECIP_THRESHOLD_MMHR)),
    )
    if signal_cells < MIN_SIGNAL_CELLS:
        return None

    def _prep(field):
        return np.clip(np.where(field <= NO_COVERAGE, 0.0, field), 0.0, None)

    # Taper the crop edges to zero before the FFT. Without this, the hard
    # rectangular cut at the crop boundary leaks spectral energy across many
    # frequencies ("spectral leakage"), which phase correlation is very
    # sensitive to - the estimated peak can swing wildly (by 100+ degrees)
    # for crop-radius changes of just a few km. A window fixes that.
    window = np.outer(np.hanning(crop_old.shape[0]), np.hanning(crop_old.shape[1]))
    fa = np.fft.fft2(_prep(crop_old) * window)
    fb = np.fft.fft2(_prep(crop_new) * window)
    cross = fb * np.conj(fa)
    magnitude = np.abs(cross)
    cross = np.divide(cross, magnitude, out=np.zeros_like(cross), where=magnitude > 1e-9)
    corr = np.abs(np.fft.ifft2(cross))
    peak_row, peak_col = np.unravel_index(np.argmax(corr), corr.shape)

    n_rows, n_cols = corr.shape
    shift_row = peak_row if peak_row <= n_rows // 2 else peak_row - n_rows
    shift_col = peak_col if peak_col <= n_cols // 2 else peak_col - n_cols

    if crop_lats.size < 2 or crop_lons.size < 2:
        return None
    km_per_row = abs(float(np.mean(np.diff(crop_lats)))) * KM_PER_DEG_LAT
    km_per_col = abs(float(np.mean(np.diff(crop_lons)))) * KM_PER_DEG_LON_AT_EQUATOR * math.cos(
        math.radians(lat_c)
    )

    # Row index increases whichever way `lats` runs (ascending or
    # descending) - normalize dy so a positive value always means "moved
    # north", regardless of MRMS's row ordering convention.
    lat_sign = 1.0 if (crop_lats[-1] - crop_lats[0]) >= 0 else -1.0
    dy_km = shift_row * km_per_row * lat_sign
    dx_km = shift_col * km_per_col

    displacement_km = math.hypot(dx_km, dy_km)
    if displacement_km < NOISE_FLOOR_KM:
        return {"bearing_deg": 0.0, "speed_kmh": 0.0, "signal_cells": signal_cells}

    speed_kmh = displacement_km / elapsed_hours
    if speed_kmh > MAX_STORM_SPEED_KMH_ASSUMPTION:
        # The crop itself was sized assuming nothing moves faster than this,
        # so a correlation implying a faster shift isn't real storm motion -
        # it's sparse/disconnected echoes (or noise) aliasing onto some
        # other offset within the crop. Distrust it rather than report it.
        return None

    bearing_deg = (math.degrees(math.atan2(dx_km, dy_km)) + 360) % 360
    return {"bearing_deg": bearing_deg, "speed_kmh": speed_kmh, "signal_cells": signal_cells}


def _combine_block_motions(block_results):
    """Combine several block motion vectors (each {"bearing_deg",
    "speed_kmh", "signal_cells"}), weighted by signal_cells, into one
    (bearing_deg, speed_kmh). Bearing is circular - averaging degrees
    directly would make 350 deg and 10 deg average to 180 deg instead of
    ~0 deg - so combination happens in Cartesian (vx, vy) space. Caller
    guarantees block_results is non-empty.

    The combined speed can never exceed MAX_STORM_SPEED_KMH_ASSUMPTION
    without an explicit check: by the triangle inequality, the magnitude of
    a weighted average of vectors is at most the weighted average of their
    magnitudes, and every contributing block already satisfies that cap."""
    vx_sum = vy_sum = weight_sum = 0.0
    for r in block_results:
        w = r["signal_cells"]
        rad = math.radians(r["bearing_deg"])
        vx_sum += w * r["speed_kmh"] * math.sin(rad)
        vy_sum += w * r["speed_kmh"] * math.cos(rad)
        weight_sum += w

    vx, vy = vx_sum / weight_sum, vy_sum / weight_sum
    speed_kmh = math.hypot(vx, vy)
    if speed_kmh < 1e-9:
        return 0.0, 0.0
    return (math.degrees(math.atan2(vx, vy)) + 360) % 360, speed_kmh


def estimate_motion_field(lat, lon, lats, lons, data, valid_time, oldest_key):
    """Estimate the precip field's dominant motion near (lat, lon) using a
    TREC-style cluster of small local blocks (the point itself plus 4
    cardinal offsets) rather than one whole-field correlation. A single
    whole-field correlation over a crop wide enough to be useful can lock
    onto a distant, unrelated storm cell if one falls inside it; several
    small blocks clustered tightly around the point dilute that risk by
    construction, since a contaminating cell is unlikely to dominate more
    than one or two of them. Each block is phase-correlated between the
    oldest cached frame in the lookback window and the freshest one (rather
    than chaining consecutive 2-minute frames, which gives a more stable
    displacement estimate over the whole window and needs no per-cell
    identity tracking)."""
    if oldest_key is None:
        return None

    lats_old, lons_old, data_old, valid_time_old = open_frame(oldest_key)
    utc_new = to_utc_datetime(valid_time)
    utc_old = to_utc_datetime(valid_time_old)
    elapsed_hours = (utc_new - utc_old).total_seconds() / 3600.0
    if elapsed_hours <= 0:
        return None

    if not (np.array_equal(lats_old, lats) and np.array_equal(lons_old, lons)):
        lat_map = _nearest_index_map(lats_old, lats)
        lon_map = _nearest_index_map(lons_old, lons)
        remapped = np.full((lats.size, lons.size), NO_COVERAGE, dtype=data_old.dtype)
        remapped[np.ix_(lat_map, lon_map)] = data_old
        data_old, lats_old, lons_old = remapped, lats, lons

    # Sized to how far a storm could plausibly have moved in the interval
    # actually being compared (elapsed_hours) - NOT the forecast horizon.
    # A crop sized off forecast_min can span 150-200+ km, which on an active
    # day easily contains multiple independently-moving storm cells.
    crop_radius_km = elapsed_hours * MAX_STORM_SPEED_KMH_ASSUMPTION * CROP_RADIUS_BUFFER

    offset_km = crop_radius_km * BLOCK_OFFSET_FRACTION
    lon_km_per_deg = KM_PER_DEG_LON_AT_EQUATOR * math.cos(math.radians(lat))
    block_centers = [(lat, lon)]
    for bearing_deg in BLOCK_OFFSET_BEARINGS_DEG:
        bearing_rad = math.radians(bearing_deg)
        block_centers.append((
            lat + (offset_km * math.cos(bearing_rad)) / KM_PER_DEG_LAT,
            lon + (offset_km * math.sin(bearing_rad)) / lon_km_per_deg,
        ))

    block_results = []
    for lat_c, lon_c in block_centers:
        result = _estimate_block_motion(
            lat_c, lon_c, lats, lons, data, lats_old, lons_old, data_old,
            elapsed_hours, crop_radius_km,
        )
        if result is not None:
            block_results.append(result)

    if not block_results:
        return None

    bearing_deg, speed_kmh = _combine_block_motions(block_results)
    return {"bearing_deg": bearing_deg, "speed_kmh": speed_kmh, "elapsed_hours": elapsed_hours}


def advect_series(lat, lon, lats, lons, data, bearing_deg, speed_kmh, forecast_min,
                   step_min=FORECAST_STEP_MIN):
    """Sample the current field upstream of (lat, lon) at each future time
    step - upstream being the reverse of the motion bearing, i.e. wherever
    the rain now headed this way is currently sitting. Returns
    [(t_minutes, rate_mmhr), ...] from t=0 to forecast_min."""
    upstream_bearing = math.radians((bearing_deg + 180) % 360)
    vx = math.sin(upstream_bearing)
    vy = math.cos(upstream_bearing)
    lon_km_per_deg = KM_PER_DEG_LON_AT_EQUATOR * math.cos(math.radians(lat))

    series = []
    t = 0.0
    while t <= forecast_min + 1e-6:
        dist_km = speed_kmh * (t / 60.0)
        sample_lat = lat + (dist_km * vy) / KM_PER_DEG_LAT
        sample_lon_360 = (lon + (dist_km * vx) / lon_km_per_deg) % 360

        lat_idx = np.argmin(np.abs(lats - sample_lat))
        lon_idx = np.argmin(np.abs(lons - sample_lon_360))
        rate = float(data[lat_idx, lon_idx])
        if rate <= NO_COVERAGE:
            rate = 0.0
        series.append((t, rate))
        t += step_min
    return series


def summarize_rain_forecast(series, threshold_mmhr=PRECIP_THRESHOLD_MMHR):
    """Walk an advect_series() result for the current status and the first
    below->above (start) and, after that, above->below (end) threshold
    crossings."""
    if not series:
        return {"raining_now": False, "start_min": None, "end_min": None}

    raining_now = series[0][1] >= threshold_mmhr
    start_min = 0.0 if raining_now else None

    if start_min is None:
        for t, rate in series[1:]:
            if rate >= threshold_mmhr:
                start_min = t
                break

    end_min = None
    if start_min is not None:
        for t, rate in series:
            if t <= start_min:
                continue
            if rate < threshold_mmhr:
                end_min = t
                break

    return {"raining_now": raining_now, "start_min": start_min, "end_min": end_min}


def compute_rain_forecast(cfg, lats, lons, data, valid_time, frames):
    oldest_key = frames[0]["key"] if len(frames) > 1 else None
    motion = estimate_motion_field(
        cfg.lat, cfg.lon, lats, lons, data, valid_time, oldest_key
    )

    lat_idx = np.argmin(np.abs(lats - cfg.lat))
    lon_idx = np.argmin(np.abs(lons - (cfg.lon % 360)))
    current_rate = float(data[lat_idx, lon_idx])
    if current_rate <= NO_COVERAGE:
        current_rate = 0.0

    if motion is None or motion["speed_kmh"] <= 0:
        # Can't project without motion - fall back to status-only.
        series = [(0.0, current_rate)]
    else:
        series = advect_series(
            cfg.lat, cfg.lon, lats, lons, data,
            motion["bearing_deg"], motion["speed_kmh"], cfg.forecast_min,
        )

    forecast = summarize_rain_forecast(series)
    return {"motion": motion, "series": series, "forecast": forecast}


def run_analysis(cfg):
    lats, lons, data, valid_time, frames, is_new = fetch_frame_history(cfg)
    result = compute_rain_forecast(cfg, lats, lons, data, valid_time, frames)
    result.update(lats=lats, lons=lons, data=data, valid_time=valid_time, is_new=is_new)
    return result


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
        self.motion = None     # estimate_motion_field() result, or None
        self.forecast = None   # summarize_rain_forecast() result, or None
        self.rain_ended_at = None  # UTC datetime of the last observed raining_now True->False, or None

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
        checked without touching the existing forecast/motion results."""
        with self.lock:
            self.last_checked = datetime.datetime.now()
            self.error = None

    def snapshot(self):
        with self.lock:
            return copy.copy(self)


def run_analysis_cycle(state, cfg):
    result = run_analysis(cfg)

    if not result["is_new"]:
        # No new radar scan since last check - leave the existing
        # forecast/motion results alone rather than recomputing off data
        # that hasn't changed.
        state.mark_checked()
        return

    # Track the moment rain actually stops (raining_now flips True->False)
    # so the dynamic refresh schedule knows when to start its slow-down
    # ramp. Only run_analysis_cycle's own worker thread ever writes state,
    # so reading state.forecast/rain_ended_at here without the lock is safe.
    was_raining = state.forecast["raining_now"] if state.forecast else False
    now_raining = result["forecast"]["raining_now"]
    rain_ended_at = state.rain_ended_at
    if was_raining and not now_raining:
        rain_ended_at = datetime.datetime.now(datetime.timezone.utc)
    elif now_raining:
        rain_ended_at = None  # raining again - any slow-down ramp no longer applies

    state.update_result(
        lats=result["lats"], lons=result["lons"], data=result["data"],
        motion=result["motion"], forecast=result["forecast"],
        rain_ended_at=rain_ended_at,
        last_updated=result["valid_time"],
    )


# --- dynamic --watch refresh schedule ---

# Refresh cadence (minutes) when no rain is expected within the forecast
# window, and the fastest cadence used once rain is imminent or happening.
DYNAMIC_BASE_INTERVAL_MIN = 15.0
DYNAMIC_MIN_INTERVAL_MIN = 1.0
# (minutes until rain starts, refresh interval in minutes), descending by
# time - refresh ramps smoothly between these as rain approaches. Above the
# first anchor's time it's flat at DYNAMIC_BASE_INTERVAL_MIN; at/below the
# last anchor's time (including while it's actively raining, i.e. 0) it's
# flat at DYNAMIC_MIN_INTERVAL_MIN.
DYNAMIC_APPROACH_ANCHORS = [(45.0, 10.0), (20.0, 5.0), (10.0, 1.0)]
# Minutes to ease back up from DYNAMIC_MIN_INTERVAL_MIN to
# DYNAMIC_BASE_INTERVAL_MIN after rain ends, absent a new storm approaching.
DYNAMIC_RECOVERY_MIN = 30.0


def _approach_interval_min(minutes_until_start):
    """Interpolated refresh interval based on minutes until rain starts (or
    0 if it's already raining). None means no rain expected in the window."""
    anchors = DYNAMIC_APPROACH_ANCHORS
    if minutes_until_start is None or minutes_until_start > anchors[0][0]:
        return DYNAMIC_BASE_INTERVAL_MIN
    if minutes_until_start <= anchors[-1][0]:
        return DYNAMIC_MIN_INTERVAL_MIN
    for (t_hi, v_hi), (t_lo, v_lo) in zip(anchors, anchors[1:]):
        if t_lo <= minutes_until_start <= t_hi:
            frac = (minutes_until_start - t_lo) / (t_hi - t_lo)
            return v_lo + frac * (v_hi - v_lo)
    return DYNAMIC_MIN_INTERVAL_MIN


def _recovery_interval_min(rain_ended_at, now):
    """Interpolated refresh interval easing back up to base over
    DYNAMIC_RECOVERY_MIN minutes since rain ended, or None if that ramp
    doesn't apply (rain hasn't ended, or the ramp has already finished)."""
    if rain_ended_at is None:
        return None
    elapsed_min = (now - rain_ended_at).total_seconds() / 60.0
    if elapsed_min >= DYNAMIC_RECOVERY_MIN:
        return None
    frac = elapsed_min / DYNAMIC_RECOVERY_MIN
    return DYNAMIC_MIN_INTERVAL_MIN + frac * (DYNAMIC_BASE_INTERVAL_MIN - DYNAMIC_MIN_INTERVAL_MIN)


def compute_dynamic_interval_minutes(forecast, rain_ended_at, now):
    """Combine the approaching-storm ramp-down with the post-rain ramp-up.
    While still easing back up from a just-ended rain, a newly approaching
    storm should never be allowed to poll slower than its own ramp calls
    for, so the faster (smaller) of the two wins."""
    approach = _approach_interval_min(forecast["start_min"] if forecast else None)
    recovery = _recovery_interval_min(rain_ended_at, now)
    if recovery is None:
        return approach
    return min(approach, recovery)


def worker_loop(state, cfg, fixed_interval_seconds):
    while not state.stop_event.is_set():
        state.set_busy(True)
        try:
            run_analysis_cycle(state, cfg)
        except Exception as exc:  # keep the dashboard alive on transient errors
            state.set_error(str(exc))
            state.log(f"[error] {exc}")
        state.set_busy(False)

        if fixed_interval_seconds is not None:
            interval_seconds = fixed_interval_seconds
        else:
            snapshot = state.snapshot()
            now = datetime.datetime.now(datetime.timezone.utc)
            interval_min = compute_dynamic_interval_minutes(snapshot.forecast, snapshot.rain_ended_at, now)
            interval_seconds = interval_min * 60.0

        state.set_next_check(time.time() + interval_seconds)
        state.stop_event.wait(interval_seconds)


def build_rain_status_box_lines(snapshot):
    if snapshot.forecast is None:
        return make_box("RAIN STATUS", ["Gathering radar history..."],
                         STORM_BOX_WIDTH, STORM_BOX_HEIGHT)

    forecast = snapshot.forecast
    base_utc = to_utc_datetime(snapshot.last_updated)
    lines = ["Raining now at your location." if forecast["raining_now"]
             else "Not raining at your location."]

    if forecast["start_min"] is None:
        lines.append("No rain expected in the forecast window.")
    elif forecast["start_min"] > 0:
        lines.append(f"Rain starts in ~{format_eta(base_utc, forecast['start_min'])}.")

    if forecast["start_min"] is not None:
        if forecast["end_min"] is None:
            lines.append("No clear end within the forecast window.")
        else:
            lines.append(f"Rain ends in ~{format_eta(base_utc, forecast['end_min'])}.")

    return make_box("RAIN STATUS", lines, STORM_BOX_WIDTH, STORM_BOX_HEIGHT)


def build_motion_box_lines(units, snapshot, lookback_min):
    if snapshot.motion is None:
        lines = ["Not enough radar history or", "nearby precip signal yet."]
    else:
        speed, speed_label = convert_speed(snapshot.motion["speed_kmh"], units)
        lines = [
            f"Heading {snapshot.motion['bearing_deg']:.0f} deg @ {speed:.1f} {speed_label}",
            f"(measured over last {lookback_min:.0f} min of radar history)",
        ]
    return make_box("STORM MOTION", lines, STORM_BOX_WIDTH, STORM_BOX_HEIGHT)


def next_check_text(snapshot, fixed_interval_seconds):
    if snapshot.busy:
        return "checking now..."
    if snapshot.next_check_at is None:
        if fixed_interval_seconds is None:
            return "next check pending..."
        return f"next check in {fixed_interval_seconds:.0f}s"
    next_in = max(0, int(snapshot.next_check_at - time.time()))
    return f"next check in {next_in}s"


def build_status_box_lines(radar_info, fixed_interval_seconds, snapshot, width, height):
    last_checked = snapshot.last_checked.strftime("%H:%M:%S") if snapshot.last_checked else "-"
    last_updated = str(snapshot.last_updated) if snapshot.last_updated is not None else "-"

    lines = [
        f"Radar: {radar_info.get('radar_station')} (WFO {radar_info.get('forecast_office')}) "
        f"near {radar_info.get('city')}, {radar_info.get('state')}",
        f"Last checked {last_checked}   Radar scan valid {last_updated}",
        "Status: " + next_check_text(snapshot, fixed_interval_seconds),
    ]
    if snapshot.error:
        lines.append(f"Last error: {snapshot.error}")
    lines.append("Recent activity:")
    if snapshot.log_entries:
        lines.extend(snapshot.log_entries)
    else:
        lines.append("(nothing logged yet)")
    return make_box("STATUS", lines, width, height)


def build_frame(cfg, radar_info, map_radius, fixed_interval_seconds, snapshot):
    header = (
        f" NEAREST PRECIP MONITOR - {cfg.lat:.4f}, {cfg.lon:.4f} - "
        f"{next_check_text(snapshot, fixed_interval_seconds)} "
    )
    lines = [header.center(DASHBOARD_MIN_COLS, "=")]

    if snapshot.lats is None:
        map_box = make_box(
            f"RADAR ({map_radius:.0f} mi, N up)", ["Waiting for first radar fetch..."],
            MAP_BOX_WIDTH, MAP_ROWS + 2,
        )
    else:
        map_box = build_map_box_lines(cfg.lat, cfg.lon, snapshot.lats, snapshot.lons, snapshot.data, map_radius)

    # Radar map on top, centered in the 80-col frame.
    map_indent = " " * ((DASHBOARD_MIN_COLS - MAP_BOX_WIDTH) // 2)
    lines.extend(map_indent + line for line in map_box)
    lines.append(map_indent + map_legend_line() if snapshot.lats is not None else "")
    lines.append("")

    # Rain status and storm motion boxes side by side below the map.
    rain_box = build_rain_status_box_lines(snapshot)
    motion_box = build_motion_box_lines(cfg.units, snapshot, cfg.lookback_min)
    for left, right in zip(rain_box, motion_box):
        lines.append(left + right)
    lines.append("")

    # Status bar across the full width at the bottom.
    lines.extend(build_status_box_lines(radar_info, fixed_interval_seconds, snapshot, DASHBOARD_MIN_COLS, STATUS_BOX_HEIGHT))
    return "\n".join(lines)


def run_dashboard(cfg, radar_info, map_radius, fixed_interval_seconds):
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
        target=worker_loop, args=(state, cfg, fixed_interval_seconds), daemon=True
    )
    worker.start()

    try:
        print("\x1b[?25l", end="")  # hide cursor
        while True:
            snapshot = state.snapshot()
            frame = build_frame(cfg, radar_info, map_radius, fixed_interval_seconds, snapshot)
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
    parser = argparse.ArgumentParser(
        description="Estimate when rain will start and end at a GPS point, using radar history."
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                         help="show download/cache/processing status and a download progress bar")
    parser.add_argument("--map", action="store_true",
                         help="draw an ANSI radar map centered on your point")
    parser.add_argument("--map-radius", type=float, default=25.0,
                         help="radius in miles for --map (default: 25)")
    parser.add_argument("--watch", action="store_true",
                         help="run as a continuously-updating full-screen dashboard")
    parser.add_argument("--interval", type=float, default=None,
                         help="seconds between radar checks in --watch mode (default: dynamic - "
                              "polls every 15 min normally, ramping down to every 1 min as rain "
                              "approaches/occurs and easing back up after it ends; pass a value "
                              "for a fixed cadence instead)")
    args = parser.parse_args()
    VERBOSE = args.verbose

    cfg = load_config()

    print(f"Point: {cfg.lat:.4f}, {cfg.lon:.4f}")

    vprint("[download] Looking up local NWS radar station...")
    try:
        radar = get_local_radar(cfg.lat, cfg.lon)
        print(
            f"Local NWS radar: {radar['radar_station']} "
            f"(WFO {radar['forecast_office']}, near {radar['city']}, {radar['state']})"
        )
    except requests.RequestException as exc:
        radar = {"radar_station": "?", "forecast_office": "?", "city": "?", "state": "?"}
        print(f"Could not look up local radar station: {exc}")

    if args.watch:
        run_dashboard(cfg, radar, args.map_radius, args.interval)
        return

    result = run_analysis(cfg)
    vprint(f"[process]  Mosaic valid: {result['valid_time']}")

    if args.map:
        print()
        print(render_ansi_map(cfg.lat, cfg.lon, result["lats"], result["lons"], result["data"], args.map_radius))
        print()

    forecast = result["forecast"]
    motion = result["motion"]
    base_utc = to_utc_datetime(result["valid_time"])

    print("Status: Raining now at your location." if forecast["raining_now"]
          else "Status: Not raining at your location.")

    if forecast["start_min"] is None:
        print(f"Forecast: no rain expected in the next {format_eta_minutes(cfg.forecast_min)}.")
    elif forecast["start_min"] > 0:
        print(f"Forecast: rain starts in ~{format_eta(base_utc, forecast['start_min'])}.")

    if forecast["start_min"] is not None:
        if forecast["end_min"] is None:
            print(f"Forecast: no clear end within the {format_eta_minutes(cfg.forecast_min)} window.")
        else:
            print(f"Forecast: rain ends in ~{format_eta(base_utc, forecast['end_min'])}.")

    if motion is None:
        print("Motion: not enough radar history or nearby precip signal yet.")
    else:
        speed, speed_label = convert_speed(motion["speed_kmh"], cfg.units)
        print(
            f"Motion: heading {motion['bearing_deg']:.0f} deg @ {speed:.1f} {speed_label} "
            f"(measured over last {cfg.lookback_min:.0f} min of radar history)"
        )


if __name__ == "__main__":
    main()
