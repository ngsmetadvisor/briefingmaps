# -*- coding: utf-8 -*-
"""Forecast - GEM Upper Air + Sfc Map
GitHub Actions version — all packages pre-installed via requirements.txt
"""

# ── Standard library ──────────────────────────────────────────
import csv, io, json, math, os, re, sys, time, warnings
import concurrent.futures
from collections import defaultdict

# ── Third-party: core ─────────────────────────────────────────
import numpy as np
import pandas as pd
import requests

# ── Third-party: matplotlib (backend before pyplot) ───────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Third-party: scipy ────────────────────────────────────────
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import (gaussian_filter, maximum_filter,
                           minimum_filter, label)

# ── Third-party: geo / mapping ────────────────────────────────
import folium
import branca
from shapely.geometry import shape



print('✓ All packages ready')

# @title
# -- Cell 1.5 - Configuration ------------------------------------------



CSV_URL         = 'https://raw.githubusercontent.com/ngsmetadvisor/SfcMap/main/AP_location.csv'
METAR_API       = 'https://aviationweather.gov/api/data/metar'
COVERAGE        = 'essential'       # essential | standard | all | chart
EXPORT_TIME     = '1200Z'          # 0000Z | 0600Z | 1200Z | 1800Z
INTERP_METHOD   = 'rbf'            # rbf | kriging
SLP_INTERVAL    = 4                # hPa spacing between isobars (standard=4)
GRID_N          = 240              # grid points per axis (60–600, step 20)
RBF_SMOOTHING   = 0.0              # 0.0 = exact fit; 0.2–0.5 typical for SLP
SIGMA_SMOOTH    = 1.0              # gaussian blur after interpolation (2–4 typical)
SYMBOL_SCALE    = 28               # station model size px (10–80)
FONT_SCALE      = 10               # station label font size (4–20)
HL_NEIGHBORHOOD = 5                # H/L search radius in grid cells (1–60)
HL_MIN_DELTA    = 0.5              # min pressure diff hPa to accept a centre (0.1–10.0)
HL_SIGMA        = 1.0              # gaussian smooth before extrema search (0.1–20.0)

print(f'Coverage: {COVERAGE} | Interp: {INTERP_METHOD} | Grid: {GRID_N} | Export: {EXPORT_TIME}')

# @title
#cell 1.2
import urllib.request, xml.etree.ElementTree as ET, json

_KML_URL = 'http://orangecore.net/met/wxchart/Alberta_Fire_Weather_Forecast_Zones.kml'

def _kml_to_geojson(kml_bytes):
    root = ET.fromstring(kml_bytes)
    features = []
    for pm in root.iter('Placemark'):
        name = ''
        for sd in pm.iter('SimpleData'):
            if sd.get('name') == 'NAME':
                name = sd.text or ''
                break
        if not name:
            n_tag = pm.find('name')
            name  = n_tag.text if n_tag is not None else 'Unknown'

        wfz_id = ''
        for sd in pm.iter('SimpleData'):
            if sd.get('name') == 'WFZ_ID':
                wfz_id = sd.text or ''
                break

        rings = []
        for coords_el in pm.findall('.//coordinates'):
            pts = []
            for token in coords_el.text.strip().split():
                parts = token.split(',')
                if len(parts) >= 2:
                    pts.append([float(parts[0]), float(parts[1])])
            if pts:
                rings.append(pts)

        if not rings:
            continue

        features.append({
            'type': 'Feature',
            'properties': {'name': name, 'wfz_id': wfz_id},
            'geometry':   {'type': 'Polygon', 'coordinates': rings}
        })

    return {'type': 'FeatureCollection', 'features': features}

# ── Fetch ──────────────────────────────────────────────────────────────────
try:
    with urllib.request.urlopen(_KML_URL, timeout=15) as _r:
        _kml_bytes = _r.read()
    _fire_zones_geojson_str = json.dumps(_kml_to_geojson(_kml_bytes))
    _zone_count = len(json.loads(_fire_zones_geojson_str)['features'])
    print(f'Alberta Fire Zone KML fetched → {_zone_count} zones loaded')
except Exception as e:
    print(f'WARNING: Fire zone KML fetch failed ({e}) — layer will be skipped')
    _fire_zones_geojson_str = '{"type":"FeatureCollection","features":[]}'

# ── Build HTML ─────────────────────────────────────────────────────────────
fire_zones_html = (
    '<script>\n'
    'var _FIRE_ZONES_GEOJSON = ' + _fire_zones_geojson_str + ';\n'
    '(function() {\n'
    '  function loadFireZones() {\n'
    '    var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '    if (!keys.length) { setTimeout(loadFireZones, 300); return; }\n'
    '    var MAP = window[keys[0]];\n'
    '    var fireLayer = L.geoJSON(_FIRE_ZONES_GEOJSON, {\n'
    '      style: function() {\n'
    '        return {\n'
    '          color: "#cc0000",\n'
    '          weight: 1.8,\n'
    '          opacity: 0.85,\n'
    '          fillColor: "#ff9933",\n'
    '          fillOpacity: 0.00,\n'
    '          dashArray: "4 3"\n'
    '        };\n'
    '      },\n'
    '      onEachFeature: function(feature, layer) {\n'
    '        var name = (feature.properties && feature.properties.name) || "Fire Zone";\n'
    '        layer.bindTooltip(name, {sticky: true, opacity: 0.9});\n'
    '        layer.bindPopup(\n'
    '          \'<div style="font-family:Courier New,monospace;font-size:12px;">\'\n'
    '          + \'<b style="color:#cc4400">\' + name + \'</b><br>\'\n'
    '          + \'Alberta Fire Weather Forecast Zone\'\n'
    '          + \'</div>\'\n'
    '        );\n'
    '      }\n'
    '    });\n'
    '    var _fireVisible = true;\n'
    '    var btn = document.getElementById("btn-fire-zones");\n'
    '    if (btn) {\n'
    '      btn.onclick = function() {\n'
    '        _fireVisible = !_fireVisible;\n'
    '        if (_fireVisible) { fireLayer.addTo(MAP); btn.style.background = "#cc4400"; }\n'
    '        else { MAP.removeLayer(fireLayer); btn.style.background = "#b0b8c8"; }\n'
    '      };\n'
    '      btn.style.background = "#cc4400";\n'
    '    }\n'
    '    fireLayer.addTo(MAP);\n'
    '  }\n'
    '  if (document.readyState === "complete") { setTimeout(loadFireZones, 800); }\n'
    '  else { window.addEventListener("load", function(){ setTimeout(loadFireZones, 800); }); }\n'
    '})();\n'
    '</script>\n'
    '<style>#btn-fire-zones { transition: background 0.2s; }</style>\n'
)

print('fire_zones_html ready')

# @title
#Cell 1.3a
# ══════════════════════════════════════════════════════════════════════════
#  STATION MODEL CONTROLS  ← edit these values only
# ══════════════════════════════════════════════════════════════════════════

# ── Circle ────────────────────────────────────────────────────────────────
CIRCLE_RADIUS   = 0.05   # fraction of S  (0.10 small → 0.20 large)

# ── Wind barb ─────────────────────────────────────────────────────────────
BARB_STAFF_LEN  = 1.00   # staff length multiplier of S
BARB_FULL_LEN   = 0.30   # full-barb (10kt) length fraction of S
BARB_HALF_LEN   = 0.15   # half-barb (5kt) length fraction of S
BARB_SPACING    = 0.10 # spacing between barbs fraction of S
BARB_LINE_WIDTH = 0.03  # stroke width fraction of S  (min 0.9px)
FEATHER_ANGLE   = 110    # degrees from staff  (90=perp, >90 tilts toward tip)
FEATHER_SIDE    = +1     # +1=right side, -1=left side (WMO standard)

# ── Font ──────────────────────────────────────────────────────────────────
FONT_SIZE_SCALE = 0.4   # font size = max(FONT_MIN_PX, S * this)
FONT_MIN_PX     = 7     # absolute minimum font size in px

# ── Label spacing ─────────────────────────────────────────────────────────
LABEL_HORIZ_OFF = 0.12   # horizontal offset from circle edge, fraction of S
LABEL_VERT_OFF  = 12     # vertical offset top/bottom labels from centre (px)
LABEL_ROW_GAP   = 0.9    # multiplier between name/ceiling rows

# ── Canvas ────────────────────────────────────────────────────────────────
CANVAS_PAD      = 1.5    # padding fraction of S
CANVAS_H_FACTOR = 3.4    # canvas height = S * this + PAD*2

# ══════════════════════════════════════════════════════════════════════════
print('Open this session to change the met symbols sizes')

# ── Cell 1.3b - WMO station model as SVG string ---
import math


print("Building Met Symbols")

def cloud_circle_svg(cx, cy, R, oktas):
    lw = max(0.9, R * 0.13)
    s = []
    if oktas == 9:  # VV — full black + white X
        s.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="black" stroke="black" stroke-width="{lw}"/>')
        s.append(f'<line x1="{cx-R*.55:.2f}" y1="{cy-R*.55:.2f}" x2="{cx+R*.55:.2f}" y2="{cy+R*.55:.2f}" stroke="white" stroke-width="{lw*.85:.2f}"/>')
        s.append(f'<line x1="{cx+R*.55:.2f}" y1="{cy-R*.55:.2f}" x2="{cx-R*.55:.2f}" y2="{cy+R*.55:.2f}" stroke="white" stroke-width="{lw*.85:.2f}"/>')
        return ''.join(s)
    s.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="white" stroke="black" stroke-width="{lw}"/>')
    if oktas <= 0:
        return ''.join(s)
    if oktas >= 8:
        s.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="black" stroke="black" stroke-width="{lw}"/>')
        return ''.join(s)
    if oktas == 2:
        s.append(f'<path d="M{cx},{cy} L{cx},{cy-R:.2f} A{R:.2f},{R:.2f} 0 0,1 {cx+R:.2f},{cy} Z" fill="black"/>')
    elif oktas == 4:
        s.append(f'<path d="M{cx},{cy} L{cx},{cy-R:.2f} A{R:.2f},{R:.2f} 0 1,1 {cx},{cy+R:.2f} Z" fill="black"/>')
    elif oktas == 6:
        s.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="black" stroke="black" stroke-width="{lw}"/>')
        s.append(f'<path d="M{cx},{cy} L{cx-R:.2f},{cy} A{R:.2f},{R:.2f} 0 0,1 {cx},{cy-R:.2f} Z" fill="white"/>')
    s.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="none" stroke="black" stroke-width="{lw}"/>')
    return ''.join(s)


def wind_barb_svg(cx, cy, R, wind_dir, wind_spd, wind_gust, S):
    """WMO wind barb — direction/speed controlled by module constants."""
    if wind_dir is None or wind_spd is None:
        return ''
    if wind_spd < 3:
        return (f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{R*1.5:.2f}" '
                f'fill="none" stroke="black" stroke-width="1"/>')

    sl   = S * BARB_STAFF_LEN
    blen = S * BARB_FULL_LEN
    bspc = S * BARB_SPACING
    lw   = max(0.9, S * BARB_LINE_WIDTH)

    staff_base_y = -R
    staff_tip_y  = -(R + sl)

    fx_full = FEATHER_SIDE * blen
    fx_half = FEATHER_SIDE * S * BARB_HALF_LEN
    tilt    = math.tan(math.radians(FEATHER_ANGLE - 90)) * blen

    spd = int(round(wind_spd / 5.0)) * 5
    pn  = spd // 50;  spd -= pn * 50
    fu  = spd // 10;  spd -= fu * 10
    ha  = spd //  5

    parts = []
    parts.append(
        f'<line x1="0" y1="{staff_base_y:.2f}" x2="0" y2="{staff_tip_y:.2f}" '
        f'stroke="black" stroke-width="{lw:.2f}" stroke-linecap="round"/>'
    )

    pos = 0.0

    if pn == 0 and fu == 0 and ha == 1:
        # lone half-barb — draw slightly inset from tip
        hy = staff_tip_y + 0.28 * sl
        parts.append(
            f'<line x1="0" y1="{hy:.2f}" x2="{fx_half:.2f}" y2="{hy - tilt*0.5:.2f}" '
            f'stroke="black" stroke-width="{lw:.2f}" stroke-linecap="round"/>'
        )
    else:
        for _ in range(pn):  # 50-kt pennants
            ay  = staff_tip_y + pos
            by2 = staff_tip_y + pos + bspc * 2
            pts = f'0,{ay:.2f} {fx_full:.2f},{ay - tilt:.2f} 0,{by2:.2f}'
            parts.append(f'<polygon points="{pts}" fill="black"/>')
            pos += bspc * 1.5
        for _ in range(fu):  # 10-kt full barbs
            fy = staff_tip_y + pos
            parts.append(
                f'<line x1="0" y1="{fy:.2f}" x2="{fx_full:.2f}" y2="{fy - tilt:.2f}" '
                f'stroke="black" stroke-width="{lw:.2f}" stroke-linecap="round"/>'
            )
            pos += bspc
        for _ in range(ha):  # 5-kt half barbs
            hy = staff_tip_y + pos
            parts.append(
                f'<line x1="0" y1="{hy:.2f}" x2="{fx_half:.2f}" y2="{hy - tilt*0.5:.2f}" '
                f'stroke="black" stroke-width="{lw:.2f}" stroke-linecap="round"/>'
            )
            pos += bspc

    inner = ''.join(parts)
    return (
        f'<g transform="translate({cx:.2f},{cy:.2f}) rotate({wind_dir:.1f})">'
        f'{inner}</g>'
    )


def pressure_tendency_svg(cx, cy, R, tendency, S, fs):
    """
    WMO pressure tendency symbol to the right of the station circle.
    Accepts int codes (0–7) or string keys.
    """
    _map = {
        'rising':         2,
        'falling':        7,
        'steady':         4,
        'rising_falling': 0,
        'falling_rising': 5,
        'rising_steady':  1,
        'falling_steady': 6,
    }
    if isinstance(tendency, str):
        tendency = _map.get(tendency.lower())
    if tendency is None:
        return ''

    lw   = max(0.9, S * 0.042)
    off  = R + S * LABEL_HORIZ_OFF

    # align with the pressure-change number row
    ox   = cx + off + fs * 2.2
    oy   = cy - R * 0.6 - LABEL_VERT_OFF + S * 0.65

    arm  = S * 0.22
    rise = S * 0.20

    def seg(x1, y1, x2, y2):
        return (
            f'<line x1="{ox+x1:.2f}" y1="{oy+y1:.2f}" '
            f'x2="{ox+x2:.2f}" y2="{oy+y2:.2f}" '
            f'stroke="black" stroke-width="{lw:.2f}" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
        )

    parts = []
    if   tendency == 2: parts.append(seg(-arm,  rise*.5,  arm, -rise*.5))
    elif tendency == 7: parts.append(seg(-arm, -rise*.5,  arm,  rise*.5))
    elif tendency == 4: parts.append(seg(-arm,  0,        arm,  0))
    elif tendency == 0:
        parts += [seg(-arm, rise*.5, 0, -rise*.5), seg(0, -rise*.5, arm, rise*.5)]
    elif tendency == 5:
        parts += [seg(-arm, -rise*.5, 0, rise*.5), seg(0, rise*.5, arm, -rise*.5)]
    elif tendency == 1:
        parts += [seg(-arm, rise*.5, 0, -rise*.5), seg(0, -rise*.5, arm, -rise*.5)]
    elif tendency == 6:
        parts += [seg(-arm, -rise*.5, 0, rise*.5), seg(0, rise*.5, arm, rise*.5)]
    return ''.join(parts)


def station_model_svg(d, S=34):
    """Full WMO station model SVG."""
    PAD = S * CANVAS_PAD
    W   = S * 3 + PAD * 2
    H   = S * CANVAS_H_FACTOR + PAD * 2
    cx  = W / 2
    cy  = H / 2
    R   = S * CIRCLE_RADIUS
    fs  = max(FONT_MIN_PX, int(S * FONT_SIZE_SCALE))
    off = R + S * LABEL_HORIZ_OFF

    parts = []

    # ── Sky cover / triangle ──────────────────────────────────────────────
    if d.get('has_sky_obs', False):
        parts.append(cloud_circle_svg(cx, cy, R, d['oktas']))
    else:
        th = R * 1.6
        parts.append(
            f'<polygon points="{cx:.2f},{cy-th:.2f} '
            f'{cx-th:.2f},{cy+th*0.65:.2f} '
            f'{cx+th:.2f},{cy+th*0.65:.2f}" '
            f'fill="black" stroke="none"/>'
        )

    # ── Wind barb ─────────────────────────────────────────────────────────
    parts.append(wind_barb_svg(cx, cy, R,
                               d['wind_dir'], d['wind_spd'],
                               d.get('wind_gust', 0), S))

    # ── Text helper ───────────────────────────────────────────────────────
    def txt(x, y, text, anchor='end', size=None):
        sz = size or fs
        return (
            f'<text x="{x:.1f}" y="{y:.1f}" '
            f'text-anchor="{anchor}" dominant-baseline="central" '
            f'font-size="{sz}px" font-weight="bold" '
            f'font-family="Courier New,monospace" fill="black" '
            f'paint-order="stroke" stroke="white" '
            f'stroke-width="2" stroke-linejoin="round">'
            f'{text}</text>'
        )

    # ── Temperature (top-left) ────────────────────────────────────────────
    if d['temp'] is not None:
        parts.append(txt(cx - off, cy - R * 0.6 - LABEL_VERT_OFF, str(d['temp'])))

    # ── Vis + weather (left) ──────────────────────────────────────────────
    v  = d['vis']
    vs = (str(int(v))  if v is not None and v >= 10    else
          str(int(v))  if v is not None and v % 1 == 0 else
          f'{v:.1f}'   if v is not None                else None)
    wx = ' '.join(x for x in [vs, d['weather'] or None] if x)
    if wx:
        parts.append(txt(cx - off - 4, cy, wx))

    # ── Dewpoint (bottom-left) ────────────────────────────────────────────
    if d['dew'] is not None:
        parts.append(txt(cx - off, cy + R * 0.6 + LABEL_VERT_OFF, str(d['dew'])))

    # ── SLP label (top-right) ─────────────────────────────────────────────
    if d['slp_label']:
        parts.append(txt(cx + off, cy - R * 0.6 - LABEL_VERT_OFF,
                         d['slp_label'], anchor='start'))

    # ── Pressure change + tendency symbol (right) ─────────────────────────
    tendency        = d.get('tendency')
    pressure_change = d.get('pressure_change')
    if tendency is not None:
        tend_y     = cy - R * 0.6 - LABEL_VERT_OFF + S * 0.65
        has_number = tendency != 'steady' and pressure_change is not None
        if has_number:
            sign   = '+' if pressure_change > 0 else ('-' if pressure_change < 0 else '')
            pc_str = sign + str(abs(pressure_change))
            parts.append(txt(cx + off, tend_y, pc_str, anchor='start'))
        parts.append(pressure_tendency_svg(cx, cy, R, tendency, S, fs))

    # ── Ceiling height (below circle) ────────────────────────────────────
    if d['lowest_sig'] and d['lowest_sig']['height'] <= 120:
        _cb = math.ceil(d['lowest_sig']['height'] / 10)
        parts.append(txt(cx, cy + R + fs * LABEL_ROW_GAP,
                         str(_cb), anchor='middle'))

    # ── Station ID (bottom) ───────────────────────────────────────────────
    _name_y = cy + R + fs * LABEL_ROW_GAP + fs * (LABEL_ROW_GAP + 0.2)
    parts.append(txt(cx, _name_y, d['icao'][-3:], anchor='middle'))

    return (
        f'<svg width="{W:.0f}" height="{H:.0f}" '
        f'viewBox="0 0 {W:.2f} {H:.2f}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="overflow:visible">'
        + ''.join(parts)
        + '</svg>'
    ), W, H


def flight_cat_color(d):
    return {
        'VFR':  '#22aa44',
        'MVFR': '#2244cc',
        'IFR':  '#cc2222',
        'LIFR': '#880088',
    }.get(d.get('flt_cat', ''), '#888888')


# ══════════════════════════════════════════════════════════════════════════
#  DEMO
# ══════════════════════════════════════════════════════════════════════════
print(f'Station model SVG ready  '
      f'(FEATHER_SIDE={FEATHER_SIDE}, FEATHER_ANGLE={FEATHER_ANGLE}°, '
      f'FONT_SIZE_SCALE={FONT_SIZE_SCALE})')

# ── Cell UA-2a. Fetch GEM upper-air + surface: RDPS days 0-3, GDPS days 3-7 ──
# ── Source: dd.weather.gc.ca  WXO-DD layout (confirmed May 2026) ─────────────
# - new code, with better runs checking system before run

import xarray as xr
from scipy.spatial import cKDTree
import math as _math
import tempfile, os, asyncio, aiohttp
import numpy as np
import pandas as pd
from datetime import datetime, timezone as _tz, timedelta


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          USER CONTROLS                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝



# ── Forecast window ───────────────────────────────────────────────────────────
# RDPS covers days 0 → RDPS_FORECAST_DAYS
# GDPS covers days RDPS_FORECAST_DAYS → GDPS_FORECAST_DAYS
# e.g. RDPS=3, GDPS=7 → RDPS days 0-3, GDPS days 3-7
RDPS_FORECAST_DAYS = 3
GDPS_FORECAST_DAYS = 10

# ── Synoptic hours to fetch ───────────────────────────────────────────────────
UA_HOURS  = [0, 12]       # isobaric (upper-air) valid times — UTC
SFC_HOURS = [0, 12]        # surface output rows — UTC
# Surface fetch strategy:
#   MSLP      → fetched at 00Z (matches SFC_HOURS)
#   QPF12H    → Precip-Accum(06Z) − Precip-Accum(18Z prior day)
#               window: 18Z day D-1 → 06Z day D, labelled as 00Z day D

# ── Pressure levels ───────────────────────────────────────────────────────────
GEM_PRESSURE_LEVELS = [500, 850]   # hPa — isobaric levels to fetch
FETCH_SFC           = True         # fetch surface (MSLP + Precip-Accum → QPF12H)

# ── Grid domain ───────────────────────────────────────────────────────────────
GEM_LAT_MAX  =  76.425  # °N  northern boundary
GEM_LAT_MIN  =  32.025  # °N  southern boundary
GEM_LON_MIN  = -175.425 # °E  western boundary
GEM_LON_MAX  =  -51.225 # °E  eastern boundary
GEM_GRID_DEG =   0.5    # grid spacing in degrees

# ── Grid density ──────────────────────────────────────────────────────────────
# Approximate spacings for reference:
#   5.00° → ~500 km  |  2.00° → ~200 km  |  0.90° → ~100 km
#   0.50° →  ~55 km  |  0.15° →  ~15 km (GDPS native)
#   0.09° →  ~10 km  (RDPS native — very slow extraction, not recommended)

# ── Run selection ─────────────────────────────────────────────────────────────
RDPS_RUN_HOURS    = [0, 6, 12, 18]   # UTC hours RDPS runs are initialised
GDPS_RUN_HOURS    = [0, 12]          # UTC hours GDPS runs are initialised
RDPS_MIN_AGE_H    = 2.5              # min hours since init before considering run
GDPS_MIN_AGE_H    = 6.0
MAX_LOOKBACK_DAYS = 2                # how many days back to search for a valid run

# ── Probe settings ────────────────────────────────────────────────────────────
# PROBE_MODE = 'representative' → 1 representative variable per group (faster)
# PROBE_MODE = 'all'            → every variable at every level (thorough)
PROBE_MODE        = 'representative'
PROBE_RETRIES     = 2      # retries after first fail → 3 total attempts
PROBE_RETRY_DELAY = 2.0    # seconds between retries
PROBE_TIMEOUT     = 10     # seconds per HEAD request

# ── Download settings ─────────────────────────────────────────────────────────
MAX_CONCURRENT = 3     # simultaneous GRIB2 downloads
TIMEOUT_S      = 60    # per-file download timeout in seconds


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     DERIVED CONSTANTS  (do not edit)                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

GEM_LONGITUDE = [round(GEM_LON_MIN + i*GEM_GRID_DEG, 4)
                 for i in range(int(round((GEM_LON_MAX - GEM_LON_MIN) / GEM_GRID_DEG)) + 1)]
GEM_LATITUDES = [round(GEM_LAT_MAX - i*GEM_GRID_DEG, 4)
                 for i in range(int(round((GEM_LAT_MAX - GEM_LAT_MIN) / GEM_GRID_DEG)) + 1)]

_N_PTS = len(GEM_LATITUDES) * len(GEM_LONGITUDE)
print(f'Grid            : {len(GEM_LATITUDES)} lats × {len(GEM_LONGITUDE)} lons'
      f' = {_N_PTS:,} points  ({GEM_GRID_DEG}° spacing ≈ {GEM_GRID_DEG*111:.0f} km)')
print(f'Pressure levels : {GEM_PRESSURE_LEVELS} hPa')
print(f'Forecast window : RDPS days 0-{RDPS_FORECAST_DAYS}  |  GDPS days {RDPS_FORECAST_DAYS}-{GDPS_FORECAST_DAYS}')
print(f'UA synoptic hrs : {sorted(UA_HOURS)}Z')
print(f'Sfc output hrs  : {sorted(SFC_HOURS)}Z  (MSLP@00Z, QPF12H from 18Z−1→06Z)')
print(f'Probe mode      : {PROBE_MODE}')

_DD_BASE = 'https://dd.weather.gc.ca'

# ── Isobaric variable map ─────────────────────────────────────────────────────
_VAR_MAP = {
    'AirTemp':            'TEMP',
    'GeopotentialHeight': 'HGHT',
    'RelativeHumidity':   'RELH',
    'WindU':              '_UGRD',
    'WindV':              '_VGRD',
}
_WXO_VARS = list(_VAR_MAP.keys())

# Representative variable per group used when PROBE_MODE='representative'
_PROBE_ISOB_VAR = 'AirTemp'   # representative for all isobaric vars at a given level

# ── Surface variable map ──────────────────────────────────────────────────────
_SFC_VAR_MAP = {
    'Pressure_MSL': 'MSLP',
    'Precip-Accum': '_PACC',
}
_SFC_VARS = list(_SFC_VAR_MAP.keys())

_NOW_UTC     = datetime.now(_tz.utc).replace(minute=0, second=0, microsecond=0)
_RDPS_CUTOFF = (_NOW_UTC + timedelta(days=RDPS_FORECAST_DAYS)).replace(
    hour=0, minute=0, second=0, microsecond=0
)

_MDT_OFFSET   = timedelta(hours=6)
_base_day_utc = _NOW_UTC.replace(hour=0, minute=0, second=0, microsecond=0)
_base_day_mdt = (_NOW_UTC - _MDT_OFFSET).replace(
    hour=0, minute=0, second=0, microsecond=0
).replace(tzinfo=_tz.utc)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     THERMODYNAMIC HELPERS                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _icao(lat, lon, prefix='GDPS'):
    return f"{prefix}{abs(lat):05.1f}N{abs(lon):06.1f}"

def _is_valid(v):
    return v is not None and not (isinstance(v, float) and _math.isnan(v)) and abs(v) < 99000

def _rh_to_dwpt(temp_c, rh_pct):
    if not (_is_valid(temp_c) and _is_valid(rh_pct)) or rh_pct <= 0:
        return None
    a, b  = 17.625, 243.04
    alpha = _math.log(max(rh_pct, 0.1)/100.0) + (a*temp_c/(b+temp_c))
    return round(b*alpha/(a-alpha), 2)

def _dwpt_to_mixr(dwpt_c, pres_hpa):
    if not (_is_valid(dwpt_c) and _is_valid(pres_hpa)):
        return None
    e = 6.112 * _math.exp(17.67*dwpt_c/(dwpt_c+243.5))
    return round(621.97*e/(pres_hpa-e), 3)

def _theta(temp_c, pres_hpa):
    if not (_is_valid(temp_c) and _is_valid(pres_hpa)):
        return None
    return round((temp_c+273.15)*(1000.0/pres_hpa)**0.2854, 2)

def _theta_e(temp_c, dwpt_c, pres_hpa):
    if any(not _is_valid(v) for v in [temp_c, dwpt_c, pres_hpa]):
        return None
    tk = temp_c+273.15; td = dwpt_c+273.15
    e  = 6.112*_math.exp(17.67*dwpt_c/(dwpt_c+243.5))
    r  = 0.622*e/(pres_hpa-e)
    tlc = 1.0/(1.0/(td-56.0)+_math.log(tk/td)/800.0)+56.0
    return round(tk*(1000.0/pres_hpa)**(0.2854*(1-0.28e-3*r*1000))
                 *_math.exp((3376.0/tlc-2.54)*r*(1+0.81e-3*r*1000)), 2)

def _theta_v(temp_c, mixr_gkg, pres_hpa):
    if any(not _is_valid(v) for v in [temp_c, mixr_gkg, pres_hpa]):
        return None
    tk = temp_c+273.15; r = mixr_gkg/1000.0
    return round(tk*(1000.0/pres_hpa)**0.2854*(1+1.608*r)/(1+r), 2)

def _uv_to_spd_dir(u, v):
    if u is None or v is None:
        return None, None
    drct = round((_math.degrees(_math.atan2(-u, -v))+360) % 360, 1)
    sped = round(_math.sqrt(u**2+v**2)/0.51444, 1)
    return drct, sped


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     URL BUILDERS                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _rdps_url(run_dt, fxx, var_name, pres_hpa):
    d  = run_dt.strftime('%Y%m%d')
    hh = run_dt.strftime('%H')
    lv = f'{int(pres_hpa):04d}'
    fn = f'{d}T{hh}Z_MSC_RDPS_{var_name}_IsbL-{lv}_RLatLon0.09_PT{fxx:03d}H.grib2'
    return f'{_DD_BASE}/{d}/WXO-DD/model_rdps/10km/{hh}/{fxx:03d}/{fn}'

def _gdps_url(run_dt, fxx, var_name, pres_hpa):
    d  = run_dt.strftime('%Y%m%d')
    hh = run_dt.strftime('%H')
    lv = f'{int(pres_hpa):04d}'
    fn = f'{d}T{hh}Z_MSC_GDPS_{var_name}_IsbL-{lv}_LatLon0.15_PT{fxx:03d}H.grib2'
    return f'{_DD_BASE}/{d}/WXO-DD/model_gdps/15km/{hh}/{fxx:03d}/{fn}'

def _rdps_sfc_url(run_dt, fxx, var_name):
    d  = run_dt.strftime('%Y%m%d')
    hh = run_dt.strftime('%H')
    if var_name == 'Pressure_MSL':
        fn = f'{d}T{hh}Z_MSC_RDPS_{var_name}_RLatLon0.09_PT{fxx:03d}H.grib2'
    else:
        fn = f'{d}T{hh}Z_MSC_RDPS_{var_name}_Sfc_RLatLon0.09_PT{fxx:03d}H.grib2'
    return f'{_DD_BASE}/{d}/WXO-DD/model_rdps/10km/{hh}/{fxx:03d}/{fn}'

def _gdps_sfc_url(run_dt, fxx, var_name):
    d  = run_dt.strftime('%Y%m%d')
    hh = run_dt.strftime('%H')
    if var_name == 'Pressure_MSL':
        fn = f'{d}T{hh}Z_MSC_GDPS_{var_name}_LatLon0.15_PT{fxx:03d}H.grib2'
    else:
        fn = f'{d}T{hh}Z_MSC_GDPS_{var_name}_Sfc_LatLon0.15_PT{fxx:03d}H.grib2'
    return f'{_DD_BASE}/{d}/WXO-DD/model_gdps/15km/{hh}/{fxx:03d}/{fn}'

def _fxx(run_dt, valid_dt):
    return int((valid_dt - run_dt).total_seconds() / 3600)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     RUN VERIFICATION                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Strategy:
#   1. Build all candidate run times (newest first) for RDPS and GDPS separately.
#   2. For the latest candidate, probe every fxx/var/level actually needed.
#      - PROBE_MODE='representative': 1 var per group (AirTemp per level, MSLP, PACC)
#      - PROBE_MODE='all': every variable at every level
#   3. Each probe HEAD request retried up to PROBE_RETRIES times on any failure.
#      - HTTP 404           → file genuinely missing, no retry
#      - timeout / 5xx      → transient, worth retrying
#   4. Print a per-fxx report (✓/✗) for every probed file.
#   5. If any probe fails after all retries → print failure report, fall back to
#      previous run candidate and probe that run (show report, commit regardless).
#   6. Precip fxx_prior=0 → skip probe entirely (prior accumulation = 0 by definition).
#      fxx_prior=6+ → probe required; failure is a real data gap.

async def _probe_run(session, run_dt, is_rdps, target_vts, sfc_target_vts):
    """
    Probe all needed files for run_dt.
    Returns (all_ok, report_lines) where report_lines is the per-fxx status list.
    """
    url_isob_fn = _rdps_url    if is_rdps else _gdps_url
    url_sfc_fn  = _rdps_sfc_url if is_rdps else _gdps_sfc_url
    max_fxx     = 84 if is_rdps else 240
    model_label = 'RDPS' if is_rdps else 'GDPS'

    # ── Build probe list ──────────────────────────────────────────────────────
    # Each entry: (label, url, skip_on_404_ok)
    # skip_on_404_ok=True means a 404 is acceptable (precip fxx_prior=0 case)
    probes = []   # (fxx_label, url, is_ignorable)

    # Isobaric probes
    vars_to_probe = (
        [_PROBE_ISOB_VAR] if PROBE_MODE == 'representative' else _WXO_VARS
    )
    for vt in sorted(target_vts):
        fxx_val = _fxx(run_dt, vt)
        if not (0 <= fxx_val <= max_fxx):
            continue
        fxx_label = f'fxx={fxx_val:03d}'
        for pres in GEM_PRESSURE_LEVELS:
            for var_name in vars_to_probe:
                url = url_isob_fn(run_dt, fxx_val, var_name, pres)
                label = f'{fxx_label}  {var_name}@{int(pres)}hPa'
                probes.append((label, url, False))

    # Surface probes
    if FETCH_SFC:
        sfc_vars_to_probe = (
            ['Pressure_MSL', 'Precip-Accum'] if PROBE_MODE == 'representative'
            else _SFC_VARS
        )
        for vt in sorted(sfc_target_vts):
            if vt.hour == 6:
                vt_mslp   = vt - timedelta(hours=6)   # 00Z
                vt_pacc   = vt                         # 06Z target
                fxx_mslp  = _fxx(run_dt, vt_mslp)
                fxx_tgt   = _fxx(run_dt, vt_pacc)
                fxx_prior = fxx_tgt - 12
            else:  # 18Z
                vt_mslp   = vt - timedelta(hours=6)   # 12Z
                vt_pacc   = vt                         # 18Z target
                fxx_mslp  = _fxx(run_dt, vt_mslp)
                fxx_tgt   = _fxx(run_dt, vt_pacc)
                fxx_prior = fxx_tgt - 12              # 06Z prior

            if 'Pressure_MSL' in sfc_vars_to_probe and 0 <= fxx_mslp <= max_fxx:
                url = url_sfc_fn(run_dt, fxx_mslp, 'Pressure_MSL')
                probes.append((f'fxx={fxx_mslp:03d}  Pressure_MSL@Sfc', url, False))

            if 'Precip-Accum' in sfc_vars_to_probe and 0 <= fxx_tgt <= max_fxx:
                url = url_sfc_fn(run_dt, fxx_tgt, 'Precip-Accum')
                probes.append((f'fxx={fxx_tgt:03d}  Precip-Accum@Sfc[target]', url, False))

                # Prior accumulation
                if fxx_prior == 0:
                    # fxx_prior=0 → prior = 0 by definition, skip probe
                    pass
                elif fxx_prior > 0:
                    url_prior = url_sfc_fn(run_dt, fxx_prior, 'Precip-Accum')
                    probes.append((f'fxx={fxx_prior:03d}  Precip-Accum@Sfc[prior]', url_prior, False))
                # fxx_prior < 0 → valid time before run init, impossible

    # ── Execute probes with retry ─────────────────────────────────────────────
    results   = {}   # label → ('ok' | 'missing' | 'error', detail)
    all_ok    = True

    for label, url, ignorable in probes:
        last_status = None
        last_detail = ''
        succeeded   = False

        for attempt in range(PROBE_RETRIES + 1):
            try:
                async with session.head(
                        url, timeout=aiohttp.ClientTimeout(total=PROBE_TIMEOUT)) as r:
                    if r.status == 200:
                        results[label] = ('ok', '')
                        succeeded = True
                        break
                    elif r.status == 404:
                        # Genuine missing — no point retrying
                        last_status = 'missing'
                        last_detail = 'HTTP 404'
                        break
                    else:
                        # 5xx or other — transient, retry
                        last_status = 'error'
                        last_detail = f'HTTP {r.status}'
            except asyncio.TimeoutError:
                last_status = 'error'
                last_detail = 'timeout'
            except Exception as e:
                last_status = 'error'
                last_detail = str(e)

            if not succeeded and attempt < PROBE_RETRIES:
                await asyncio.sleep(PROBE_RETRY_DELAY)

        if not succeeded:
            results[label] = (last_status, last_detail)
            if not ignorable:
                all_ok = False

    # ── Build report lines ────────────────────────────────────────────────────
    # Group by fxx for the RDPS/GDPS style output
    fxx_groups = {}
    for label, (status, detail) in results.items():
        fxx_part = label.split('  ')[0]   # e.g. 'fxx=024'
        if fxx_part not in fxx_groups:
            fxx_groups[fxx_part] = []
        sym = '✓' if status == 'ok' else '✗'
        fxx_groups[fxx_part].append(f'{sym} {label.split("  ", 1)[1]}')

    report_lines = []
    for fxx_key in sorted(fxx_groups.keys()):
        items    = fxx_groups[fxx_key]
        statuses = ' | '.join(items)
        # Find the url for this fxx to provide a clickable link
        fxx_url  = next((url for lbl, url, _ in probes
                         if lbl.startswith(fxx_key)), '')
        # Build parent directory url (strip filename)
        dir_url  = fxx_url.rsplit('/', 1)[0] if fxx_url else ''
        link     = f'  <{dir_url}>' if dir_url else ''
        report_lines.append(f'  {fxx_key}: {statuses}{link}')

    return all_ok, report_lines


async def _select_run_verified(run_hours, min_age_h, is_rdps,
                                target_vts, sfc_target_vts):
    """
    Select the best available run for RDPS or GDPS.

    Steps:
      1. Build candidate run times newest→oldest (up to MAX_LOOKBACK_DAYS).
      2. Probe the latest candidate — print full per-fxx report.
      3. If all probes pass → commit and return.
      4. If any probe fails → print failure report, move to next candidate.
      5. For the fallback candidate → probe and print report, then commit regardless.
      6. If no candidates remain → raise RuntimeError.
    """
    model_label = 'RDPS' if is_rdps else 'GDPS'
    now = _NOW_UTC


    # Build candidates newest → oldest
    candidates = []
    for day_offset in range(MAX_LOOKBACK_DAYS + 1):
        for h in sorted(run_hours, reverse=True):
            cand = (now - timedelta(days=day_offset)).replace(
                hour=h, minute=0, second=0, microsecond=0)
            age_h = (now - cand).total_seconds() / 3600
            if age_h >= min_age_h:
                candidates.append(cand)

    # Deduplicate and sort newest → oldest
    candidates = sorted(set(candidates), reverse=True)

    if not candidates:
        raise RuntimeError(f'{model_label}: no run candidates found within lookback window')

    connector = aiohttp.TCPConnector(ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:

        primary     = candidates[0]
        fallbacks   = candidates[1:]

        print(f'\n{model_label}:')
        print(f'  {primary.strftime("%Y-%m-%d %HZ")} run (probing all needed fxx):')
        all_ok, report = await _probe_run(session, primary, is_rdps,
                                          target_vts, sfc_target_vts)
        for line in report:
            # lines with a url in angle brackets → print as clickable
            if '<http' in line:
                text, url = line.rsplit('<', 1)
                url = url.rstrip('>')
                print(f'{text}')
                print(f'    → {url}')
            else:
                print(line)

        if all_ok:
            print(f'  ✓ Verified {model_label} run: {primary.strftime("%Y-%m-%d %HZ")}')
            return primary

        # ── Primary failed — print summary and fall back ──────────────────────
        missing_lines = [l for l in report if '✗' in l]
        print(f'\n  ✗ {len(missing_lines)} probe(s) failed on '
              f'{primary.strftime("%Y-%m-%d %HZ")} — falling back:')
        for line in missing_lines:
            print(f'  {line}')

        for fallback in fallbacks:
            print(f'\n  → {fallback.strftime("%Y-%m-%d %HZ")} run (fallback probe):')
            _, fb_report = await _probe_run(session, fallback, is_rdps,
                                            target_vts, sfc_target_vts)
            for line in fb_report:
                print(line)
            print(f'  ✓ Committing to {model_label} fallback run: '
                  f'{fallback.strftime("%Y-%m-%d %HZ")}')
            return fallback

    raise RuntimeError(f'{model_label}: no available run found within {MAX_LOOKBACK_DAYS}-day lookback')


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     BUILD TARGET VALID TIMES                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Isobaric valid times: UA_HOURS each day, days 0–GDPS_FORECAST_DAYS
_target_vts = []
for _d in range(GDPS_FORECAST_DAYS + 1):
    for _h in sorted(UA_HOURS):
        _vt = _base_day_utc + timedelta(days=_d, hours=_h)
        _target_vts.append(_vt)

# Surface valid times: 06Z each day (precip accumulation endpoint)
# QPF12H = Precip-Accum(06Z day D) − Precip-Accum(18Z day D-1)
# MSLP fetched at 00Z day D (06Z − 6h)
# Output row labelled 00Z day D
_sfc_target_vts = []
for _d in range(1, GDPS_FORECAST_DAYS + 2):
    _vt_00 = _base_day_mdt + timedelta(days=_d, hours=6)    # 06Z → label 00Z (midnight)
    _vt_12 = _base_day_mdt + timedelta(days=_d, hours=18)   # 18Z → label 12Z (noon)
    _sfc_target_vts.append(_vt_00)
    _sfc_target_vts.append(_vt_12)

# Flat coordinate lists for extraction
_target_lats = [lat for lat in GEM_LATITUDES for _   in GEM_LONGITUDE]
_target_lons = [lon for _   in GEM_LATITUDES for lon in GEM_LONGITUDE]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     SELECT VERIFIED RUNS                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

async def _select_runs():
    rdps = await _select_run_verified(
        RDPS_RUN_HOURS, RDPS_MIN_AGE_H, is_rdps=True,
        target_vts=_target_vts, sfc_target_vts=_sfc_target_vts
    )
    gdps = await _select_run_verified(
        GDPS_RUN_HOURS, GDPS_MIN_AGE_H, is_rdps=False,
        target_vts=_target_vts, sfc_target_vts=_sfc_target_vts
    )
    return rdps, gdps

_rdps_run_dt, _gdps_run_dt = asyncio.run(_select_runs())

print(f'\nRDPS run        : {_rdps_run_dt.strftime("%Y-%m-%d %HZ")}')
print(f'GDPS run        : {_gdps_run_dt.strftime("%Y-%m-%d %HZ")}')
print(f'RDPS cutoff UTC : {_RDPS_CUTOFF.isoformat()}')
print(f'Valid times (UA): {[vt.strftime("%Y-%m-%d %HZ") for vt in _target_vts]}')
print(f'Valid times (Sfc): {[vt.strftime("%Y-%m-%d %HZ") for vt in _sfc_target_vts]}')


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     GRIB2 POINT EXTRACTION                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _extract_points_grib(grib_bytes, target_lats, target_lons, var_name):
    fd, tmp_path = tempfile.mkstemp(suffix='.grib2')
    os.close(fd)
    try:
        with open(tmp_path, 'wb') as f:
            f.write(grib_bytes)
        try:
            ds = xr.open_dataset(tmp_path, engine='cfgrib',
                                 backend_kwargs={'errors': 'ignore'})
        except Exception:
            ds = xr.open_dataset(tmp_path, engine='cfgrib')
        da = ds[list(ds.data_vars)[0]]
        da.load()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    lat_name = next((c for c in ('latitude', 'lat', 'y') if c in da.coords), None)
    lon_name = next((c for c in ('longitude', 'lon', 'x') if c in da.coords), None)

    lats = da[lat_name].values
    lons = da[lon_name].values
    lons = ((lons + 180) % 360) - 180

    # ── Crop to MAP_EXTENT before extraction ─────────────────────────────
    _W, _E, _S, _N = -175.425, -51.225, 32.025, 76.425
    _PAD = 2.0
    data = da.values
    if lats.ndim == 2:
        _mask = ((lats >= _S - _PAD) & (lats <= _N + _PAD) &
                 (lons >= _W - _PAD) & (lons <= _E + _PAD))
        _rows = np.any(_mask, axis=1)
        _cols = np.any(_mask, axis=0)
        lats  = lats[np.ix_(_rows, _cols)]
        lons  = lons[np.ix_(_rows, _cols)]
        data  = data[np.ix_(_rows, _cols)]
    else:
        _row_mask = (lats >= _S - _PAD) & (lats <= _N + _PAD)
        _col_mask = (lons >= _W - _PAD) & (lons <= _E + _PAD)
        lats  = lats[_row_mask]
        lons  = lons[_col_mask]
        data  = data[np.ix_(_row_mask, _col_mask)]

    units   = da.attrs.get('GRIB_units', '') or da.attrs.get('units', '')
    is_temp = 'Temp' in var_name

    target_lats = np.asarray(target_lats)
    target_lons = np.asarray(target_lons)

    if lats.ndim == 2:
        tree    = cKDTree(np.column_stack([lats.ravel(), lons.ravel()]))
        _, fidx = tree.query(np.column_stack([target_lats, target_lons]))
        iy, ix  = np.unravel_index(fidx, lats.shape)
        vals    = data[iy, ix].astype(float)
    else:
        ilats = np.argmin(np.abs(lats[:, None] - target_lats), axis=0)
        ilons = np.argmin(np.abs(lons[:, None] - target_lons), axis=0)
        vals  = data[ilats, ilons].astype(float)

    if is_temp and (units == 'K' or np.nanmax(vals) > 100):
        vals = vals - 273.15

    result = {}
    for (tlat, tlon), val in zip(zip(target_lats, target_lons), vals):
        if _math.isnan(val) or abs(val) > 1e8:
            result[(tlat, tlon)] = None
        else:
            result[(tlat, tlon)] = round(float(val), 2) if is_temp else float(val)

    return result, lats, lons, data


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     BUILD DOWNLOAD TASK LIST                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

_point_data  = {}
_sfc_data    = {}
_raw_grids   = {}
_GRIB_CACHE  = '/tmp/gem_grib_cache'
os.makedirs(_GRIB_CACHE, exist_ok=True)

# ── Isobaric tasks ────────────────────────────────────────────────────────────
_tasks = []
for vt in _target_vts:
    use_rdps = _NOW_UTC < vt < _RDPS_CUTOFF
    for pres in GEM_PRESSURE_LEVELS:
        for var_name in _WXO_VARS:
            if use_rdps:
                fxx = _fxx(_rdps_run_dt, vt)
                if 0 <= fxx <= 84:
                    _tasks.append(('RDPS', _rdps_url(_rdps_run_dt, fxx, var_name, pres),
                                   var_name, pres, vt))
            else:
                fxx = _fxx(_gdps_run_dt, vt)
                if 0 <= fxx <= 240:
                    _tasks.append(('GDPS', _gdps_url(_gdps_run_dt, fxx, var_name, pres),
                                   var_name, pres, vt))

# ── Surface tasks ─────────────────────────────────────────────────────────────
# MSLP stored at 00Z (vt_mslp), Precip stored at 06Z (vt_pacc)
# QPF12H = max(0, _PACC − _PACC_PRIOR) computed at assembly
_sfc_tasks = []

for vt in _sfc_target_vts:
    use_rdps  = _NOW_UTC < vt < _RDPS_CUTOFF
    run_dt    = _rdps_run_dt if use_rdps else _gdps_run_dt
    url_fn    = _rdps_sfc_url if use_rdps else _gdps_sfc_url
    max_fxx   = 84 if use_rdps else 240
    model_lbl = 'RDPS' if use_rdps else 'GDPS'

    if vt.hour == 6:
        vt_mslp   = vt - timedelta(hours=6)   # 00Z
        vt_pacc   = vt                         # 06Z target
    else:  # 18Z
        vt_mslp   = vt - timedelta(hours=6)   # 12Z
        vt_pacc   = vt                         # 18Z target
    fxx_mslp  = _fxx(run_dt, vt_mslp)
    fxx_tgt   = _fxx(run_dt, vt_pacc)
    fxx_prior = fxx_tgt - 12

    if not (0 <= fxx_tgt <= max_fxx):
        continue

    # MSLP at 00Z
    if 0 <= fxx_mslp <= max_fxx:
        _sfc_tasks.append((model_lbl,
                           url_fn(run_dt, fxx_mslp, 'Pressure_MSL'),
                           'Pressure_MSL', vt_mslp, ''))

    # MSLP at 12Z (for twice-daily SFC map)
    vt_mslp_12  = vt_mslp + timedelta(hours=12)   # 12Z same day
    fxx_mslp_12 = _fxx(run_dt, vt_mslp_12)
    if 0 <= fxx_mslp_12 <= max_fxx:
        _sfc_tasks.append((model_lbl,
                           url_fn(run_dt, fxx_mslp_12, 'Pressure_MSL'),
                           'Pressure_MSL', vt_mslp_12, ''))

    # Precip target at 06Z
    _sfc_tasks.append((model_lbl,
                       url_fn(run_dt, fxx_tgt, 'Precip-Accum'),
                       'Precip-Accum', vt_pacc, ''))

    # Precip prior — skip if fxx_prior=0 (prior = 0 by definition)
    if fxx_prior > 0:
        _sfc_tasks.append((model_lbl,
                           url_fn(run_dt, fxx_prior, 'Precip-Accum'),
                           'Precip-Accum', vt_pacc, '_PRIOR'))

print(f'\nIsobaric tasks  : {len(_tasks)} GRIB2 files')
print(f'Surface tasks   : {len(_sfc_tasks)} GRIB2 files '
      f'(MSLP×{sum(1 for t in _sfc_tasks if t[2]=="Pressure_MSL")} '
      f'+ PACC×{sum(1 for t in _sfc_tasks if t[2]=="Precip-Accum" and t[4]=="")} '
      f'+ PACC_PRIOR×{sum(1 for t in _sfc_tasks if t[4]=="_PRIOR")})')
print(f'Extraction pts  : {len(set(zip(_target_lats, _target_lons)))} grid points\n')


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     ASYNC FETCH + EXTRACT                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

async def _fetch_and_extract_all():
    sem    = asyncio.Semaphore(MAX_CONCURRENT)
    errors = []

    async def _worker_isob(model, url, var_name, pres, vt):
        vt_str = vt.strftime('%Y-%m-%d') + f' {vt.hour:02d}Z'
        col    = _VAR_MAP[var_name]
        import datetime as _dt
        print(f'  → {_dt.datetime.utcnow().strftime("%H:%M:%S")} GET {url.split("/")[-1]}', flush=True)
        async with sem:
            try:
                async with session.get(
                        url, timeout=aiohttp.ClientTimeout(total=TIMEOUT_S)) as r:
                    raw    = await r.read() if r.status == 200 else None
                    status = r.status
            except Exception as e:
                print(f'  ✗ {_dt.datetime.utcnow().strftime("%H:%M:%S")} EXCEPTION {url.split("/")[-1]}: {e}', flush=True)
                errors.append((url, str(e))); return
        print(f'  ← {_dt.datetime.utcnow().strftime("%H:%M:%S")} HTTP {status} {url.split("/")[-1]}', flush=True)
        if raw is None:
            errors.append((url, f'HTTP {status}')); return
        if len(raw) == 0:
            errors.append((url, 'HTTP 200 but zero bytes')); return
        print(f'  [cfgrib] {_dt.datetime.utcnow().strftime("%H:%M:%S")} parsing {var_name} {len(raw)} bytes', flush=True)
        try:
            extracted, _rlats, _rlons, _rdata = _extract_points_grib(raw, _target_lats, _target_lons, var_name)
        except Exception as e:
            print(f'  ✗ {_dt.datetime.utcnow().strftime("%H:%M:%S")} cfgrib FAILED {var_name}: {e}', flush=True)
            errors.append((url, str(e))); return
        print(f'  [cfgrib] {_dt.datetime.utcnow().strftime("%H:%M:%S")} done {var_name}', flush=True)
        for (lat, lon), val in extracted.items():
            key = (lat, lon, vt_str, float(pres))
            if key not in _point_data:
                _point_data[key] = {}
            _point_data[key][col] = val
        _raw_grids[(var_name, vt_str, float(pres))] = {
            'lats': _rlats, 'lons': _rlons, 'data': _rdata
        }
        import datetime as _dt
        print(f'  ✓ {_dt.datetime.utcnow().strftime("%H:%M:%S")} {model} {vt_str}  {var_name}@{pres}hPa  ({len(extracted)} pts)', flush=True)

    async def _worker_sfc(model, url, var_name, vt, col_suffix):
        vt_str   = vt.strftime('%Y-%m-%d') + f' {vt.hour:02d}Z'
        base_col = _SFC_VAR_MAP[var_name]
        col      = base_col + col_suffix
        import datetime as _dt
        print(f'  → {_dt.datetime.utcnow().strftime("%H:%M:%S")} GET {url.split("/")[-1]}', flush=True)
        async with sem:
            try:
                async with session.get(
                        url, timeout=aiohttp.ClientTimeout(total=TIMEOUT_S)) as r:
                    raw    = await r.read() if r.status == 200 else None
                    status = r.status
            except Exception as e:
                print(f'  ✗ {_dt.datetime.utcnow().strftime("%H:%M:%S")} EXCEPTION {url.split("/")[-1]}: {e}', flush=True)
                errors.append((url, str(e))); return
        print(f'  ← {_dt.datetime.utcnow().strftime("%H:%M:%S")} HTTP {status} {url.split("/")[-1]}', flush=True)
        if raw is None:
            errors.append((url, f'HTTP {status}')); return
        if len(raw) == 0:
            errors.append((url, 'HTTP 200 but zero bytes')); return
        print(f'  [cfgrib] {_dt.datetime.utcnow().strftime("%H:%M:%S")} parsing {var_name} {len(raw)} bytes', flush=True)
        try:
            extracted, _rlats, _rlons, _rdata = _extract_points_grib(raw, _target_lats, _target_lons, var_name)
        except Exception as e:
            print(f'  ✗ {_dt.datetime.utcnow().strftime("%H:%M:%S")} cfgrib FAILED {var_name}: {e}', flush=True)
            errors.append((url, str(e))); return
        print(f'  [cfgrib] {_dt.datetime.utcnow().strftime("%H:%M:%S")} done {var_name}', flush=True)
        for (lat, lon), val in extracted.items():
            key = (lat, lon, vt_str)
            if key not in _sfc_data:
                _sfc_data[key] = {}
            _sfc_data[key][col] = val
        # only stash TARGET (not PRIOR) for contouring
        if not col_suffix:
            _raw_grids[(var_name, vt_str)] = {
                'lats': _rlats, 'lons': _rlons, 'data': _rdata
            }
            if var_name == 'Pressure_MSL':
                import hashlib
                _cache_key  = hashlib.md5(url.encode()).hexdigest()[:12]
                _cache_path = os.path.join(_GRIB_CACHE, f'mslp_{_cache_key}.grib2')
                with open(_cache_path, 'wb') as _cf:
                    _cf.write(raw)
                _raw_grids[(var_name, vt_str)]['cache_path'] = _cache_path
        tag = 'PRIOR' if col_suffix else 'TARGET'
        print(f'  ✓ {_dt.datetime.utcnow().strftime("%H:%M:%S")} {model} {vt_str}  {var_name} [{tag}]  ({len(extracted)} pts)', flush=True)

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        await asyncio.gather(
            *[_worker_isob(*t) for t in _tasks],
            *[_worker_sfc(*t)  for t in _sfc_tasks],
        )

    # ── Completeness report ───────────────────────────────────────────────────
    _requested = set()
    _fetched   = set()

    for vt in _target_vts:
        use_rdps = _NOW_UTC < vt < _RDPS_CUTOFF
        prefix   = 'RDPS' if use_rdps else 'GDPS'
        fxx_val  = _fxx(_rdps_run_dt if use_rdps else _gdps_run_dt, vt)
        max_fxx  = 84 if use_rdps else 240
        if not (0 <= fxx_val <= max_fxx):
            continue
        vt_str = vt.strftime('%Y-%m-%d') + f' {vt.hour:02d}Z'
        for pres in GEM_PRESSURE_LEVELS:
            for var_name in _WXO_VARS:
                col   = _VAR_MAP[var_name]
                field = f'{var_name}@{int(pres)}hPa'
                key   = (prefix, vt_str, field)
                _requested.add(key)
                if any(_point_data.get((lat, lon, vt_str, float(pres)), {}).get(col) is not None
                       for lat, lon in zip(_target_lats, _target_lons)):
                    _fetched.add(key)

    for vt in _sfc_target_vts:
        use_rdps    = _NOW_UTC < vt < _RDPS_CUTOFF
        prefix      = 'RDPS' if use_rdps else 'GDPS'
        vt_mslp     = vt - timedelta(hours=6)
        vt_pacc     = vt
        vt_str_mslp = vt_mslp.strftime('%Y-%m-%d') + f' {vt_mslp.hour:02d}Z'
        vt_str_pacc = vt_pacc.strftime('%Y-%m-%d') + f' {vt_pacc.hour:02d}Z'
        fxx_val     = _fxx(_rdps_run_dt if use_rdps else _gdps_run_dt, vt_mslp)
        max_fxx     = 84 if use_rdps else 240
        if not (0 <= fxx_val <= max_fxx):
            continue
        for var_name in _SFC_VARS:
            for suffix, label in [('', '@Sfc'), ('_PRIOR', '@Sfc-Prior')]:
                if var_name == 'Pressure_MSL' and suffix == '_PRIOR':
                    continue
                col     = _SFC_VAR_MAP[var_name] + suffix
                field   = f'{var_name}{label}'
                chk_str = vt_str_mslp if var_name == 'Pressure_MSL' else vt_str_pacc
                key     = (prefix, vt_str_mslp, field)
                _requested.add(key)
                if any(_sfc_data.get((lat, lon, chk_str), {}).get(col) is not None
                       for lat, lon in zip(_target_lats, _target_lons)):
                    _fetched.add(key)

    _missing_keys = _requested - _fetched
    _n_dl  = len(_fetched)
    _n_mi  = len(_missing_keys)
    _n_tot = len(_requested)

    for prefix, vt_str, field in sorted(_missing_keys):
        print(f'  ✗ MISSING: {prefix}  {vt_str}  {field}')

    _all_vt_strs = sorted(set(vt_str for _, vt_str, _ in _requested))
    _isob_fields = sorted(f for f in set(field for _, _, field in _requested) if 'Sfc' not in f)
    _sfc_fields  = sorted(f for f in set(field for _, _, field in _requested) if 'Sfc' in f)
    _all_fields  = _isob_fields + _sfc_fields
    _status_map  = {}
    for key in _requested:
        prefix, vt_str, field = key
        _status_map[(vt_str, field)] = (
            ('✓ fetched', '#0a5c36', '#d1fae5') if key in _fetched
            else ('✗ missing', '#991b1b', '#fee2e2')
        )

    print(f'  Fetch summary: {_n_dl} fetched  |  {_n_mi} missing  |  {_n_tot} total')
    return errors

gem_ua_errors = asyncio.run(_fetch_and_extract_all())


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     RETRY MISSING TASKS                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _task_fetched_isob(model, var_name, pres, vt):
    vt_str = vt.strftime('%Y-%m-%d') + f' {vt.hour:02d}Z'
    col    = _VAR_MAP[var_name]
    return any(_point_data.get((lat, lon, vt_str, float(pres)), {}).get(col) is not None
               for lat, lon in zip(_target_lats, _target_lons))

def _task_fetched_sfc(var_name, vt, col_suffix=''):
    vt_str = vt.strftime('%Y-%m-%d') + f' {vt.hour:02d}Z'
    col    = _SFC_VAR_MAP[var_name] + col_suffix
    return any(_sfc_data.get((lat, lon, vt_str), {}).get(col) is not None
               for lat, lon in zip(_target_lats, _target_lons))

_tasks_missing     = [t for t in _tasks     if not _task_fetched_isob(t[0], t[2], t[3], t[4])]
_sfc_tasks_missing = [t for t in _sfc_tasks if not _task_fetched_sfc(t[2], t[3], t[4])]

if _tasks_missing or _sfc_tasks_missing:
    print(f'\n⚠ {len(_tasks_missing)} isobaric + {len(_sfc_tasks_missing)} surface tasks missing — retrying...')

    async def _retry_fetch():
        sem    = asyncio.Semaphore(MAX_CONCURRENT)
        errors = []

        async def _worker_isob_r(model, url, var_name, pres, vt):
            vt_str = vt.strftime('%Y-%m-%d') + f' {vt.hour:02d}Z'
            col    = _VAR_MAP[var_name]
            async with sem:
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=TIMEOUT_S)) as r:
                        raw = await r.read() if r.status == 200 else None
                        status = r.status
                except Exception as e:
                    errors.append((url, str(e))); return
            if raw is None:
                errors.append((url, f'HTTP {status}')); return
            try:
                extracted, _rlats, _rlons, _rdata = _extract_points_grib(raw, _target_lats, _target_lons, var_name)
            except Exception as e:
                errors.append((url, str(e))); return
            for (lat, lon), val in extracted.items():
                key = (lat, lon, vt_str, float(pres))
                if key not in _point_data: _point_data[key] = {}
                _point_data[key][col] = val
            print(f'  ✓ RETRY {model} {vt_str}  {var_name}@{pres}hPa')

        async def _worker_sfc_r(model, url, var_name, vt, col_suffix):
            vt_str = vt.strftime('%Y-%m-%d') + f' {vt.hour:02d}Z'
            col    = _SFC_VAR_MAP[var_name] + col_suffix
            async with sem:
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=TIMEOUT_S)) as r:
                        raw = await r.read() if r.status == 200 else None
                        status = r.status
                except Exception as e:
                    errors.append((url, str(e))); return
            if raw is None:
                errors.append((url, f'HTTP {status}')); return
            try:
                extracted, _rlats, _rlons, _rdata = _extract_points_grib(raw, _target_lats, _target_lons, var_name)
            except Exception as e:
                errors.append((url, str(e))); return
            for (lat, lon), val in extracted.items():
                key = (lat, lon, vt_str)
                if key not in _sfc_data: _sfc_data[key] = {}
                _sfc_data[key][col] = val
            print(f'  ✓ RETRY {model} {vt_str}  {var_name} [{"PRIOR" if col_suffix else "TARGET"}]')

        connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT, ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            await asyncio.gather(
                *[_worker_isob_r(*t) for t in _tasks_missing],
                *[_worker_sfc_r(*t)  for t in _sfc_tasks_missing],
            )
        return errors

    _retry_errors = asyncio.run(_retry_fetch())
    if _retry_errors:
        print(f'  Retry errors ({len(_retry_errors)}):')
        for _u, _e in _retry_errors:
            print(f'    [{_e}]  {_u}')

    _still_isob = [t for t in _tasks_missing     if not _task_fetched_isob(t[0], t[2], t[3], t[4])]
    _still_sfc  = [t for t in _sfc_tasks_missing if not _task_fetched_sfc(t[2], t[3], t[4])]
    if _still_isob or _still_sfc:
        print(f'✗ Retry incomplete — {len(_still_isob)} isobaric + {len(_still_sfc)} surface still missing after retries')
        for t in _still_isob:
            print(f'  ✗ {t[0]}  {t[4].strftime("%Y-%m-%d %HZ")}  {t[2]}@{t[3]}hPa')
        for t in _still_sfc:
            print(f'  ✗ {t[0]}  {t[3].strftime("%Y-%m-%d %HZ")}  {t[2]} [{t[4] or "TARGET"}]')
    else:
        print('✓ Retry successful — all missing tasks now populated')

else:
    print(f'✓ All tasks populated — no retries needed')

_n_err = len(gem_ua_errors)
print(f'\n✓ Fetch complete — {len(_tasks)} isobaric + {len(_sfc_tasks)} surface files, {_n_err} errors')
if _n_err:
    print(f'  All {_n_err} error(s):')
    for _u, _e in gem_ua_errors:
        print(f'    [{_e}]  {_u}')


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     ASSEMBLE gem_ua_df                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

_COLS = ['icao','wmo','stn_name','lat','lon','valid_time','hour',
         'PRES','HGHT','TEMP','DWPT','RELH','MIXR','DRCT','SPED',
         'THTA','THTE','THTV','MSLP','QPF12H']

rows = []

_target_vt_strs = (
    {vt.strftime('%Y-%m-%d') + f' {vt.hour:02d}Z' for vt in _target_vts} |
    {(vt - timedelta(hours=6)).strftime('%Y-%m-%d') + f' {(vt - timedelta(hours=6)).hour:02d}Z'
     for vt in _sfc_target_vts}
)

# ── Isobaric rows ─────────────────────────────────────────────────────────────
for (lat, lon, vt_str, pres), fields in _point_data.items():
    if vt_str not in _target_vt_strs:
        continue
    vt     = datetime.strptime(vt_str, '%Y-%m-%d %HZ').replace(tzinfo=_tz.utc)
    prefix = 'RDPS' if vt < _RDPS_CUTOFF else 'GDPS'
    temp   = fields.get('TEMP')
    hght   = fields.get('HGHT')
    rh     = fields.get('RELH')
    u      = fields.get('_UGRD')
    v      = fields.get('_VGRD')
    drct, sped = _uv_to_spd_dir(u, v)
    dwpt = _rh_to_dwpt(temp, rh)
    mixr = _dwpt_to_mixr(dwpt, pres)
    rows.append({
        'icao':       _icao(lat, lon, prefix),
        'wmo':        None,
        'stn_name':   f'{prefix} {lat:+.1f}N {abs(lon):.1f}W',
        'lat':        float(lat),
        'lon':        float(lon),
        'valid_time': vt_str,
        'hour':       int(vt.hour),
        'PRES':       float(pres),
        'HGHT':       hght,
        'TEMP':       temp,
        'DWPT':       dwpt,
        'RELH':       rh,
        'MIXR':       mixr,
        'DRCT':       drct,
        'SPED':       sped,
        'THTA':       _theta(temp, pres),
        'THTE':       _theta_e(temp, dwpt, pres),
        'THTV':       _theta_v(temp, mixr, pres),
        'MSLP':       None,
        'QPF12H':     None,
        '_model':     prefix,
    })

# ── Surface rows ──────────────────────────────────────────────────────────────
# MSLP keyed at 00Z; Precip keyed at 06Z → remap to 00Z label
_sfc_merged = {}
for (lat, lon, vt_str), fields in _sfc_data.items():
    vt_key = datetime.strptime(vt_str, '%Y-%m-%d %HZ').replace(tzinfo=_tz.utc)
    if vt_key.hour == 6:
        vt_label = (vt_key - timedelta(hours=6)).strftime('%Y-%m-%d') + f' 00Z'
    elif vt_key.hour == 18:
        vt_label = vt_key.strftime('%Y-%m-%d') + f' 12Z'
    else:
        vt_label = vt_str
    key = (lat, lon, vt_label)
    if key not in _sfc_merged:
        _sfc_merged[key] = {}
    _sfc_merged[key].update(fields)

for (lat, lon, vt_str), fields in _sfc_merged.items():
    if vt_str not in _target_vt_strs:
        continue
    vt     = datetime.strptime(vt_str, '%Y-%m-%d %HZ').replace(tzinfo=_tz.utc)
    prefix = 'RDPS' if vt < _RDPS_CUTOFF else 'GDPS'

    mslp_raw = fields.get('MSLP')
    mslp = (round(mslp_raw / 100.0, 1)
            if mslp_raw is not None and not _math.isnan(float(mslp_raw))
            else None)

    pacc       = fields.get('_PACC')
    pacc_prior = fields.get('_PACC_PRIOR', 0.0)
    if pacc_prior is None:
        pacc_prior = 0.0

    if _is_valid(pacc):
        qpf12h = round(max(0.0, float(pacc) - float(pacc_prior)), 2)
    else:
        qpf12h = None

    rows.append({
        'icao':       _icao(lat, lon, prefix),
        'wmo':        None,
        'stn_name':   f'{prefix} {lat:+.1f}N {abs(lon):.1f}W',
        'lat':        float(lat),
        'lon':        float(lon),
        'valid_time': vt_str,
        'hour':       int(vt.hour),
        'PRES':       0.0,
        'HGHT':       None,
        'TEMP':       None,
        'DWPT':       None,
        'RELH':       None,
        'MIXR':       None,
        'DRCT':       None,
        'SPED':       None,
        'THTA':       None,
        'THTE':       None,
        'THTV':       None,
        'MSLP':       mslp,
        'QPF12H':     qpf12h,
        '_model':     prefix,
    })

gem_ua_df = (pd.DataFrame(rows)[_COLS + ['_model']]
             if rows else pd.DataFrame(columns=_COLS + ['_model']))

# ── Merge into ua_raw_df ──────────────────────────────────────────────────────
if 'ua_raw_df' not in dir():
    print('⚠  ua_raw_df not found — creating empty frame. Re-run surface cell first.')
    ua_raw_df = pd.DataFrame(columns=_COLS)

ua_raw_df = ua_raw_df[
    ~ua_raw_df['icao'].str.startswith('GDPS') &
    ~ua_raw_df['icao'].str.startswith('RDPS')
].copy()

_before   = len(ua_raw_df)
ua_raw_df = pd.concat([ua_raw_df, gem_ua_df[_COLS]], ignore_index=True)

_sfc_rows  = gem_ua_df[gem_ua_df['PRES'] == 0.0]
_isob_rows = gem_ua_df[gem_ua_df['PRES'] != 0.0]

_qpf_valid   = _sfc_rows['QPF12H'].notna().sum()
_qpf_nonzero = (_sfc_rows['QPF12H'] > 0).sum()
_qpf_max     = _sfc_rows['QPF12H'].max()

print(f'\n✓ ua_raw_df: {_before} → {len(ua_raw_df)} rows (+{len(gem_ua_df)} GEM rows)')
print(f'  Isobaric rows   : {len(_isob_rows)}')
print(f'  Surface rows    : {len(_sfc_rows)}')
print(f'  MSLP valid      : {_sfc_rows["MSLP"].notna().sum()}')
print(f'  QPF12H valid    : {_qpf_valid}  (non-zero: {_qpf_nonzero}  max: {_qpf_max:.1f} mm)')
print(f'  RDPS stations   : {gem_ua_df[gem_ua_df["_model"]=="RDPS"]["icao"].nunique()}')
print(f'  GDPS stations   : {gem_ua_df[gem_ua_df["_model"]=="GDPS"]["icao"].nunique()}')
print(f'  Valid times     : {sorted(ua_raw_df["valid_time"].unique())}')
print(f'  Pres levels     : {sorted(ua_raw_df["PRES"].unique(), reverse=True)}')

_summary = (
    gem_ua_df
    .groupby(['_model', 'valid_time'])
    .agg(rows=('PRES', 'count'))
    .reset_index()
    .pivot(index='_model', columns='valid_time', values='rows')
    .fillna(0).astype(int)
    .reset_index()
)

print(f'✔ GEM upper-air + surface merged into ua_raw_df '
      f'(RDPS days 0-3 · GDPS days 3-7 · dd.weather.gc.ca WXO-DD GRIB2)')
print(f'  {gem_ua_df["icao"].nunique()} virtual stations | '
      f'{len(GEM_PRESSURE_LEVELS)} pressure levels + surface | '
      f'{gem_ua_df["valid_time"].nunique()} valid times | '
      f'{len(gem_ua_df)} total rows | MSLP & QPF12H included')
print(_summary.to_string(index=False))

# ── Cell UA-2b - processing . Standard-level summary table (850/700/500/250 hPa) ────────
print('--- Upper Air station - Data extract ---')
import pandas as pd
import numpy as np

STANDARD_LEVELS = [850, 500]
LEVEL_TOL       = 25

FIELDS = ['PRES','HGHT','TEMP','DWPT','RELH','MIXR','DRCT','SPED','THTA','THTE','THTV']

def find_closest_level(group_df, target_p, tol=LEVEL_TOL):
    sub = group_df.copy()
    sub['_dist'] = (sub['PRES'] - target_p).abs()
    sub = sub[sub['_dist'] <= tol]
    if sub.empty:
        return {f: None for f in FIELDS}
    best = sub.loc[sub['_dist'].idxmin()]
    return {f: best[f] if not pd.isna(best[f]) else None for f in FIELDS}

if 'ua_raw_df' not in globals():
    print('❌ Error: ua_raw_df not found. Please run Cell UA-2 first.')
else:
    # ── Build summary — group by icao + date + hour to preserve all timesteps ─
    ua_raw_df['_date'] = ua_raw_df['valid_time'].astype(str).str[:10]
    _ua_isob = ua_raw_df[ua_raw_df['PRES'] != 0.0].copy()

    if 'hour' not in ua_raw_df.columns:
        ua_raw_df['hour'] = pd.to_datetime(ua_raw_df['valid_time'].str[:13], utc=True).dt.hour

    # ── Vectorised pivot approach — no Python loop over stations ──────────────
    # For each standard level, find the closest pressure row per group in one pass
    _KEY = ['icao', '_date', 'hour']

    # Meta columns — take first row per group (same for all rows in group)
    _meta = (_ua_isob.groupby(_KEY, sort=False)
             [['wmo','stn_name','lat','lon','valid_time']]
             .first().reset_index())

    # For each target level, tag each row with distance to that level,
    # keep only rows within tolerance, then pick closest per group
    _level_dfs = []
    for lvl in STANDARD_LEVELS:
        _df = _ua_isob.copy()
        _df['_dist'] = (_df['PRES'] - lvl).abs()
        _df = _df[_df['_dist'] <= LEVEL_TOL]
        # keep only the closest row per group
        _df = (_df.sort_values('_dist')
                  .groupby(_KEY, sort=False)[FIELDS]
                  .first()
                  .reset_index())
        _df.columns = _KEY + [f'{f}_{lvl}' for f in FIELDS]
        _level_dfs.append(_df)

    # Merge all levels onto meta
    ua_summary_df = _meta.rename(columns={'_date': '_date'})
    for _ldf in _level_dfs:
        ua_summary_df = ua_summary_df.merge(_ldf, on=_KEY, how='left')

    ua_raw_df = ua_raw_df.drop(columns=['_date'])  # clean up temp col
    del _ua_isob  # clean up

    ua_summary_df = ua_summary_df.sort_values(['_date', 'hour', 'icao']).reset_index(drop=True)
    print(f'  Raw summary: {len(ua_summary_df)} rows  '
          f'({ua_summary_df["icao"].nunique()} stations x '
          f'{ua_summary_df[["_date","hour"]].drop_duplicates().shape[0]} date+hour combos)')

    # ── Keep real soundings; fill other date+hour combos with GDML ────────────
    _is_model = ua_summary_df['icao'].str.startswith(('GDPS', 'GDML', 'RDPS'))
    _real_df  = ua_summary_df[~_is_model].copy()
    _model_df = ua_summary_df[_is_model].copy()

    _has_real = len(_real_df) > 0

    if not _has_real:
        print('  No real soundings — using model data for all times.')
        _keep_df = ua_summary_df.copy()
    else:
        _real_combos = set(zip(_real_df['hour'], _real_df['_date']))
        print(f'  Real sounding combos: {sorted(_real_combos)}')

        _model_keep = _model_df[
            _model_df.apply(lambda r: (r['hour'], r['_date']) not in _real_combos, axis=1)
        ]
        print(f'  Model fill combos: {sorted(set(zip(_model_keep["hour"], _model_keep["_date"])))}')

        _keep_df = pd.concat([_real_df, _model_keep], ignore_index=True)

    ua_summary_df = _keep_df.sort_values(['_date', 'hour', 'icao']).reset_index(drop=True)
    print(f'  After filter: {len(ua_summary_df)} rows  '
          f'({ua_summary_df["icao"].nunique()} stations x '
          f'{ua_summary_df[["_date","hour"]].drop_duplicates().shape[0]} date+hour combos)')

    # ── Build _synoptic_times and _ua_date_map ────────────────────────────────
    _ua_times = (ua_summary_df[['_date', 'hour', 'valid_time']]
                 .drop_duplicates(subset=['_date', 'hour'])
                 .sort_values(['_date', 'hour'])
                 .reset_index(drop=True))

    _ua_date_map    = {}
    _synoptic_times = []
    for _, _row in _ua_times.iterrows():
        _hr  = int(_row['hour'])
        _dt  = pd.Timestamp(_row['_date']).date()
        _key = f'{_row["_date"].replace("-","")}_{_hr:02d}'
        _ua_date_map[_key] = f'{_row["_date"]} {_hr:02d}Z'
        _synoptic_times.append((_dt, _hr))
        print(f'  Synoptic time: {_dt}  {_hr:02d}Z')

    # Drop temp col from summary df
    ua_summary_df = ua_summary_df.drop(columns=['_date']).reset_index(drop=True)

    print(f'  _synoptic_times: {len(_synoptic_times)} steps')

    # ── Coverage check ────────────────────────────────────────────────────────
    for lvl in STANDARD_LEVELS:
        n_miss = ua_summary_df[f'TEMP_{lvl}'].isna().sum()
        if n_miss:
            print(f'  ⚠ {lvl} hPa: {n_miss} station-hours missing TEMP')
    print(f'✓ Summary table: {len(ua_summary_df)} rows  '
          f'({ua_summary_df["icao"].nunique()} stations x '
          f'{ua_summary_df["hour"].nunique()} hours)')

    # ── Display ───────────────────────────────────────────────────────────────
    def _style_summary(df):
        s = df.style.format(na_rep='—', precision=1)
        for cols, cmap in [
            ([c for c in df.columns if c.startswith('TEMP_') or c.startswith('DWPT_')], 'RdBu_r'),
            ([c for c in df.columns if c.startswith('RELH_')], 'Blues'),
            ([c for c in df.columns if c.startswith('SPED_')], 'Purples'),
            ([c for c in df.columns if c.startswith('HGHT_')], 'YlOrRd'),
            ([c for c in df.columns if c.startswith('MIXR_')], 'YlGn'),
            ([c for c in df.columns if c.startswith(('THTA_','THTE_','THTV_'))], 'RdYlBu_r'),
            ([c for c in df.columns if c.startswith('DRCT_')], 'twilight'),
        ]:
            existing = [c for c in cols if c in df.columns]
            if existing:
                s = s.background_gradient(subset=existing, cmap=cmap, axis=None)
        pres_cols = [c for c in df.columns if c.startswith('PRES_')]
        if pres_cols:
            s = s.bar(subset=pres_cols, color='#9ecae1', vmin=200, vmax=900)
        return s

    print(f'  Summary table: {len(ua_summary_df)} rows  '
          f'({ua_summary_df["icao"].nunique()} stations x '
          f'{ua_summary_df["hour"].nunique()} hours)')

# @title
# ── Block 01 SKIPPED — populate empty segment globals for downstream cells ──
print('000--------  500 & 700 Ridge and Trough')
import pandas as pd

ua_summary_df['_vt']   = pd.to_datetime(ua_summary_df['valid_time'])
ua_summary_df['_date'] = ua_summary_df['_vt'].dt.date
ua_summary_df['_hour'] = ua_summary_df['_vt'].dt.hour

_synoptic_times = sorted(ua_summary_df[['_date','_hour']].drop_duplicates()
                          .itertuples(index=False, name=None))

for (_date, _hr) in _synoptic_times:
    _key      = f'{pd.Timestamp(_date).strftime("%Y%m%d")}_{int(_hr):02d}'
    globals()[f'ridge_segs_{_key}']      = []
    globals()[f'trough_segs_{_key}']     = []
    globals()[f'ridge_segs_700_{_key}']  = []
    globals()[f'trough_segs_700_{_key}'] = []
    globals()[f'ridge_segs_500_{_key}']  = []
    globals()[f'trough_segs_500_{_key}'] = []


print('⚡ Block 01 skipped — empty segment globals ready')

# @title
# ══════════════════════════════════════════════════════════════════════════
# CELL UA-2bX — Build synthetic metar_records from GEM surface rows
#            (SLP + PRECIP ONLY)
# ══════════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np

print('000--------  Building synthetic metar_records from GEM surface data...')

if 'gem_ua_df' not in globals():
    raise RuntimeError('❌ gem_ua_df not found — run Cell UA-2b first')

_sfc = gem_ua_df[
    (gem_ua_df['PRES'] == 0.0) &
    (gem_ua_df['MSLP'].notna())
].copy()

if _sfc.empty:
    raise RuntimeError('❌ No GEM surface rows with valid MSLP')

print(f'  Surface rows : {len(_sfc):,}')

# ── Vectorised build — no Python loop ────────────────────────────────────────
_vt = pd.to_datetime(
    _sfc['valid_time'].str.replace('Z', '', regex=False),
    format='%Y-%m-%d %H'
)

_sfc = _sfc.copy()
_sfc['_vt']        = _vt
_sfc['_timestamp'] = _vt.dt.strftime('%d%H00')
_sfc['_qpf']       = pd.to_numeric(_sfc['QPF12H'], errors='coerce')

_sfc = _sfc.sort_values(['_vt', 'icao']).reset_index(drop=True)

# Build records from vectorised columns — to_dict('records') is C-speed
_core = _sfc[['_vt','icao','stn_name','lat','lon','MSLP','_qpf','_model','_timestamp']].copy()
_core.columns = ['time','icao','name','lat','lon','slp','precip_accum','_model','timestamp']

metar_records = _core.to_dict('records')

# Add fixed fields in one pass
_FIXED = {
    'altimeter': None, 'temp': None, 'dew': None,
    'wind_dir': None, 'wind_spd': None, 'wind_gust': None,
    'flt_cat': 'VFR', 'rh': None, 'vis': None,
    'clouds': [], 'weather': '', 'tendency': None,
    'pressure_change': None, '_synthetic': True,
}
for rec in metar_records:
    rec.update(_FIXED)
    rec['hour'] = rec['time'].hour
    rec['time'] = rec['time'].to_pydatetime()
    rec['clouds'] = []   # ensure each record gets its own list instance

_valid_times = sorted({
    pd.Timestamp(r['time']).strftime('%Y-%m-%d %HZ')
    for r in metar_records
})

print(f'  ✓ Created {len(metar_records):,} synthetic METAR records')
print(f'  ✓ Valid times : {len(_valid_times)}')
print(f'  ✓ First valid : {_valid_times[0]}')
print(f'  ✓ Last valid  : {_valid_times[-1]}')
print(f'  ✓ QPF field   : QPF12H (12h accumulation)')
print('\n✅ metar_records ready (SLP + PRECIP only)')

# ── Cell UA-2c. GEM Surface grids (MSLP + QPF12H) ───────────────────────────

import tempfile, requests
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from matplotlib import pyplot as plt

# ── Settings — exact match to Cell A ─────────────────────────────────────────
MSLP_SMOOTH_SIGMA = 6.0
MSLP_INTERVAL     = 4.0
QPF_SIGMA         = 1.0
QPF_INTERVAL      = 1.0
REGRID_NLON       = 300
REGRID_NLAT       = 200
MAP_EXTENT        = [-175.425, -51.225, 32.025, 76.425]
CONTOUR_PAD       = 2.0

# H/L detection — exact match to Cell A
HL_SIGMA        = 1.0
HL_NEIGHBORHOOD = 30
HL_MIN_DELTA    = 0.5
SLP_INTERVAL    = 4.0

print('UA-2c settings (matched to Cell A)')
print(f'  MSLP smooth σ={MSLP_SMOOTH_SIGMA}, interval={MSLP_INTERVAL}')
print(f'  QPF12H σ={QPF_SIGMA}, interval={QPF_INTERVAL}')
print(f'  Regrid {REGRID_NLON}×{REGRID_NLAT}')
print(f'  HL_NEIGHBORHOOD={HL_NEIGHBORHOOD}, HL_MIN_DELTA={HL_MIN_DELTA}')

if 'ua_raw_df' not in globals():
    raise RuntimeError('❌ ua_raw_df not found — run Cell UA-2b first')
if '_rdps_run_dt' not in globals():
    raise RuntimeError('❌ _rdps_run_dt not found — run Cell UA-2b first')
if '_gdps_run_dt' not in globals():
    raise RuntimeError('❌ _gdps_run_dt not found — run Cell UA-2b first')

# ── QPF from ua_raw_df (pivot — unchanged) ────────────────────────────────────
_sfc_df = ua_raw_df[ua_raw_df['PRES'] == 0.0].copy()
print(f'\nSurface rows   : {len(_sfc_df)}')
print(f'  QPF12H valid : {_sfc_df["QPF12H"].notna().sum()}')

_sfc_df['_vt']   = pd.to_datetime(
    _sfc_df['valid_time'].str.replace('Z', '', regex=False),
    format='%Y-%m-%d %H'
)
_sfc_df['_date'] = _sfc_df['_vt'].dt.date
_sfc_df['_hour'] = _sfc_df['_vt'].dt.hour

_sfc_times = sorted(
    _sfc_df[['_date', '_hour']].drop_duplicates()
    .itertuples(index=False, name=None)
)
print(f'  Valid times  : {len(_sfc_times)}')
print(f'  SFC hours present: {sorted(set(h for _,h in _sfc_times))}')

def _qpf_build_grid(df, sigma=0.5, lon_vec=None, lat_vec=None):
    sub = df[['lat', 'lon', 'QPF12H']].dropna(subset=['QPF12H'])
    if len(sub) < 8:
        return None
    lats = sub['lat'].values
    lons = sub['lon'].values
    vals = sub['QPF12H'].values.astype(float)
    if lon_vec is None:
        lon_vec = np.array(sorted(np.unique(lons)))
    if lat_vec is None:
        lat_vec = np.array(sorted(np.unique(lats)))
    grid = np.full((len(lat_vec), len(lon_vec)), np.nan)
    lon_idx = {round(v, 4): i for i, v in enumerate(lon_vec)}
    lat_idx = {round(v, 4): i for i, v in enumerate(lat_vec)}
    for la, lo, va in zip(lats, lons, vals):
        ri = lat_idx.get(round(la, 4))
        ci = lon_idx.get(round(lo, 4))
        if ri is not None and ci is not None:
            grid[ri, ci] = va
    if sigma > 0:
        grid = gaussian_filter(grid, sigma=sigma)
    return np.clip(grid, 0.0, None)

# ── MSLP helpers — identical to Cell A ───────────────────────────────────────
def _fetch_grib(url, tag='', retries=3, retry_delay=5):
    import cfgrib
    import datetime as _dt
    print(f'  [{_dt.datetime.utcnow().strftime("%H:%M:%S")}] Downloading {tag} ...', end=' ', flush=True)    for attempt in range(retries):
        try:
            r = requests.get(url, stream=True, timeout=(15, 90))
            r.raise_for_status()
            tmp = tempfile.NamedTemporaryFile(suffix='.grib2', delete=False)
            try:
                for chunk in r.iter_content(1 << 20):
                    tmp.write(chunk)
            except requests.exceptions.ChunkedEncodingError as e:
                tmp.close()
                os.unlink(tmp.name)
                raise RuntimeError(f'Stream stalled for {tag}: {e}')
            tmp.close()
            print(f'{os.path.getsize(tmp.name)/1e6:.1f} MB')
            return tmp.name
        except Exception as e:
            if attempt < retries - 1:
                print(f'\n  ⚠ Attempt {attempt+1}/{retries} failed ({e}) — retrying in {retry_delay}s...')
                time.sleep(retry_delay)
            else:
                raise RuntimeError(f'_fetch_grib failed after {retries} attempts for {tag}: {e}')

def _open_and_load_mslp(url, tag=''):
    path = _fetch_grib(url, tag)
    try:
        import cfgrib
        datasets = cfgrib.open_datasets(path)
        return _load_mslp(datasets)
    finally:
        if os.path.exists(path):
            os.unlink(path)

def _load_mslp(datasets):
    for ds in datasets:
        for name in ('prmsl', 'msl', 'PRMSL'):
            if name in ds:
                da   = ds[name]
                lats = da.coords['latitude'].values
                lons = da.coords['longitude'].values
                data = np.squeeze(da.values).astype(float)
                lons = np.where(lons > 180, lons - 360, lons)
                return lats, lons, data / 100.0
    avail = [v for ds in datasets for v in ds.data_vars]
    raise KeyError(f'MSLP not found. Available: {avail}')

def _crop_grid(lats_g, lons_g, grid):
    W, E, S, N = MAP_EXTENT
    pad = CONTOUR_PAD
    if lats_g.ndim == 2:
        in_box   = ((lats_g >= S-pad) & (lats_g <= N+pad) &
                    (lons_g >= W-pad) & (lons_g <= E+pad))
        row_mask = np.any(in_box, axis=1)
        col_mask = np.any(in_box, axis=0)
    else:
        row_mask = (lats_g >= S-pad) & (lats_g <= N+pad)
        col_mask = (lons_g >= W-pad) & (lons_g <= E+pad)
    g_c    = grid[np.ix_(row_mask, col_mask)]
    lats_c = lats_g[np.ix_(row_mask, col_mask)] if lats_g.ndim==2 else lats_g[row_mask]
    lons_c = lons_g[np.ix_(row_mask, col_mask)] if lons_g.ndim==2 else lons_g[col_mask]
    return lats_c, lons_c, g_c

def _regrid_mslp(lats_g, lons_g, grid):
    W, E, S, N = MAP_EXTENT
    lon_reg = np.linspace(W, E, REGRID_NLON)
    lat_reg = np.linspace(S, N, REGRID_NLAT)
    lon2d, lat2d = np.meshgrid(lon_reg, lat_reg)
    src_lons = lons_g.ravel() if lons_g.ndim==2 else np.tile(lons_g, lats_g.shape[0])
    src_lats = lats_g.ravel() if lats_g.ndim==2 else np.repeat(lats_g, lons_g.shape[0])
    src_vals = grid.ravel()
    valid    = np.isfinite(src_vals)
    # linear first, fill remaining NaN with nearest
    grid_reg = griddata(
        (src_lons[valid], src_lats[valid]),
        src_vals[valid],
        (lon2d, lat2d),
        method='linear'
    )
    _nan_mask = ~np.isfinite(grid_reg)
    if _nan_mask.any():
        grid_nn = griddata(
            (src_lons[valid], src_lats[valid]),
            src_vals[valid],
            (lon2d, lat2d),
            method='nearest'
        )
        grid_reg[_nan_mask] = grid_nn[_nan_mask]
    return lon_reg, lat_reg, grid_reg

def _count_contours(grid, lon_vec, lat_vec, interval):
    if grid is None or not np.isfinite(grid).any():
        return 0
    glon, glat = np.meshgrid(lon_vec, lat_vec)
    gmin = np.floor(np.nanmin(grid) / interval) * interval
    gmax = np.ceil(np.nanmax(grid) / interval) * interval
    levels = np.arange(gmin, gmax + interval, interval)
    fig, ax = plt.subplots(figsize=(1, 1))
    cs = ax.contour(glon, glat, grid, levels=levels)
    plt.close(fig)
    return sum(len(seg) > 1 for seg_list in cs.allsegs for seg in seg_list)

# ── Main loop ─────────────────────────────────────────────────────────────────
from datetime import timedelta

for (_date, _hr) in _sfc_times:

    _date_str  = pd.Timestamp(_date).strftime('%Y%m%d')
    _valid_str = f'{_date_str} {int(_hr):02d}Z'
    _key       = f'{_date_str}_{int(_hr):02d}'
    _vt        = pd.Timestamp(_date).replace(tzinfo=None) + pd.Timedelta(hours=int(_hr))
    _vt        = _vt.to_pydatetime().replace(tzinfo=__import__('datetime').timezone.utc)

    print(f'\n── {_valid_str} ──')

    # ── Subset for this valid time ────────────────────────────────────────────
    _sub = _sfc_df[(_sfc_df['_date'] == _date) & (_sfc_df['_hour'] == _hr)]

    # ── MSLP — download GRIB directly, process like Cell A ───────────────────
    use_rdps  = _NOW_UTC < _vt < _RDPS_CUTOFF
    run_dt    = _rdps_run_dt if use_rdps else _gdps_run_dt
    fxx       = int((_vt - run_dt).total_seconds() / 3600)
    url_mslp  = (_rdps_sfc_url(run_dt, fxx, 'Pressure_MSL') if use_rdps
                 else _gdps_sfc_url(run_dt, fxx, 'Pressure_MSL'))

    try:
        # use cached file from UA-2a if available — avoids re-download
        _vt_mslp_str = f'{_date_str[:4]}-{_date_str[4:6]}-{_date_str[6:8]} {int(_hr):02d}Z'
        _cache_entry = _raw_grids.get(('Pressure_MSL', _vt_mslp_str), {})
        _cache_path  = _cache_entry.get('cache_path')

        if _cache_path and os.path.exists(_cache_path):
            print(f'  [cache] MSLP {_vt_mslp_str}', end=' ')
            import cfgrib
            _datasets        = cfgrib.open_datasets(_cache_path)
            _mslp_lats, _mslp_lons, _mslp_data = _load_mslp(_datasets)
            print('✓')
        else:
            print(f'  [fetch] MSLP fxx={fxx:03d} — not in cache, downloading...')
            _mslp_lats, _mslp_lons, _mslp_data = _open_and_load_mslp(url_mslp,
                f'{"RDPS" if use_rdps else "GDPS"} MSLP fxx={fxx:03d}')

        _mslp_lats, _mslp_lons, _mslp_data = _crop_grid(_mslp_lats, _mslp_lons, _mslp_data)
        lon_vec, lat_vec, _mslp_reg         = _regrid_mslp(_mslp_lats, _mslp_lons, _mslp_data)
        _mslp_reg = np.where(np.isfinite(_mslp_reg), _mslp_reg, np.nanmean(_mslp_reg))
        slp_grid  = gaussian_filter(_mslp_reg, sigma=MSLP_SMOOTH_SIGMA)
        print(f'  ✓ MSLP {slp_grid.shape} {np.nanmin(slp_grid):.1f}–{np.nanmax(slp_grid):.1f} hPa')
    except Exception as _e:
        slp_grid = lon_vec = lat_vec = None
        print(f'  ✗ MSLP failed: {_e}')

    # ── QPF from ua_raw_df ────────────────────────────────────────────────────
    _qpf_lats = np.array(sorted(_sub['lat'].dropna().unique()))
    _qpf_lons = np.array(sorted(_sub['lon'].dropna().unique()))
    qpf_grid  = _qpf_build_grid(_sub, sigma=QPF_SIGMA,
                                 lon_vec=_qpf_lons, lat_vec=_qpf_lats)
    if qpf_grid is not None:
        print(f'  ✓ QPF12H {qpf_grid.shape} '
              f'{qpf_grid.min():.1f}–{qpf_grid.max():.1f} mm')

    _mslp_segs = _count_contours(slp_grid, lon_vec, lat_vec, MSLP_INTERVAL) if slp_grid is not None else 0
    _qpf_segs  = _count_contours(qpf_grid,
                                  np.array(sorted(np.unique(_sub['lon'].dropna()))),
                                  np.array(sorted(np.unique(_sub['lat'].dropna()))),
                                  QPF_INTERVAL) if qpf_grid is not None else 0

    print(f'  [DEBUG] MSLP segments (4 hPa): {_mslp_segs}')
    print(f'  [DEBUG] QPF segments  (1 mm)  : {_qpf_segs}')

    globals()[f'slp_grid_{_key}']    = slp_grid
    globals()[f'qpf_grid_{_key}']    = qpf_grid
    globals()[f'precip_grid_{_key}'] = qpf_grid
    globals()[f'lon_vec_{_key}']     = lon_vec
    globals()[f'lat_vec_{_key}']     = lat_vec
    globals()[f'qpf_lon_vec_{_key}'] = _qpf_lons
    globals()[f'qpf_lat_vec_{_key}'] = _qpf_lats

print('\n✅ Cell UA-2c complete')

if slp_grid is not None:
    globals()['slp_grid'] = slp_grid
    globals()['lon_vec']  = lon_vec
    globals()['lat_vec']  = lat_vec

# @title BLOCK 2D — H/L Pressure Centre Detection'
print('=' * 60)
print('  BLOCK 2D — H/L Pressure Centre Detection')
print('  Locates surface High and Low centres from SLP grid')
print('=' * 60)

import math
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, label, gaussian_filter


def find_hl_centers(grid, lon_vec, lat_vec,
                    neighborhood=20, min_delta=2.0):
    sg    = gaussian_filter(grid, sigma=HL_SIGMA)
    max_f = maximum_filter(sg, size=neighborhood)
    min_f = minimum_filter(sg, size=neighborhood)

    # use tolerance for float comparison
    is_max = (np.abs(sg - max_f) < 1e-6) & (sg - min_f > min_delta)
    is_min = (np.abs(sg - min_f) < 1e-6) & (max_f - sg > min_delta)

    # trim edge margin — use 10% of grid size, not neighborhood
    edge = max(3, int(min(grid.shape) * 0.10))

    centers = []
    for typ, mask in [('H', is_max), ('L', is_min)]:
        # force-add interior global extremum as fallback for both H and L
        interior = sg[edge:-edge, edge:-edge]
        if typ == 'H':
            gr, gc = np.unravel_index(np.argmax(interior), interior.shape)
        else:
            gr, gc = np.unravel_index(np.argmin(interior), interior.shape)
        gr += edge; gc += edge
        mask = mask.copy()
        mask[gr, gc] = True
        lbl, n = label(mask)
        for i in range(1, n+1):
            rows, cols = np.where(lbl == i)
            best = np.argmax(sg[rows, cols]) if typ == 'H' else np.argmin(sg[rows, cols])
            r, c = rows[best], cols[best]

            # drop edge candidates
            if r < edge or r > grid.shape[0] - edge: continue
            if c < edge or c > grid.shape[1] - edge: continue

            _grid_val = float(grid[r, c])

            def _grid_at(sta_lat, sta_lon, _lv=lat_vec, _lo=lon_vec, _g=grid):
                _ri = int(round((sta_lat - _lv[0]) /
                                (_lv[-1] - _lv[0]) * (len(_lv) - 1)))
                _ci = int(round((sta_lon - _lo[0]) /
                                (_lo[-1] - _lo[0]) * (len(_lo) - 1)))
                _ri = max(0, min(len(_lv) - 1, _ri))
                _ci = max(0, min(len(_lo) - 1, _ci))
                return float(_g[_ri, _ci])

            _thresh = SLP_INTERVAL

            if typ == 'H':
                _mask = (_mr_slp is not None) & (_mr_grid_vals >= _grid_val - _thresh)
                _inside = _mr_slp[_mask]
                _val = (math.floor(float(_inside.max())) + 1) if len(_inside) else (math.floor(_grid_val) + 1)
            else:
                _mask = (_mr_slp is not None) & (_mr_grid_vals <= _grid_val + _thresh)
                _inside = _mr_slp[_mask]
                _val = (math.ceil(float(_inside.min())) - 1) if len(_inside) else (math.ceil(_grid_val) - 1)

            centers.append(dict(
                type=typ,
                lat=float(lat_vec[r]), lon=float(lon_vec[c]),
                val=float(_val)
            ))
    return centers


HL_SIGMA        = globals().get('HL_SIGMA',        1.0)
HL_NEIGHBORHOOD = globals().get('HL_NEIGHBORHOOD', 5)
HL_MIN_DELTA    = globals().get('HL_MIN_DELTA',    0.0)
SLP_INTERVAL    = globals().get('SLP_INTERVAL',    4.0)

_sfc_keys = sorted(
    k.replace('slp_grid_', '')
    for k in globals() if k.startswith('slp_grid_')
)

hl_centers_by_key = {}

# ── Pre-build numpy SLP lookup arrays from ua_raw_df ─────────────────────────
# replaces per-centre 113k-row Python list comprehension
_sfc_raw = ua_raw_df[
    (ua_raw_df['PRES'] == 0.0) & ua_raw_df['MSLP'].notna()
][['lat','lon','MSLP']].copy()
_mr_lats      = _sfc_raw['lat'].values.astype(float)
_mr_lons      = _sfc_raw['lon'].values.astype(float)
_mr_slp       = _sfc_raw['MSLP'].values.astype(float)
_mr_grid_vals = None   # filled per key below

for _key in _sfc_keys:
    _slp  = globals().get(f'slp_grid_{_key}')
    _lonv = globals().get(f'lon_vec_{_key}')
    _latv = globals().get(f'lat_vec_{_key}')
    if _slp is None or _lonv is None or _latv is None:
        print(f'  ⚠ {_key}: missing grid — skipped')
        continue
    print(f'  {_key}: grid shape={_slp.shape} min={_slp.min():.1f} max={_slp.max():.1f} range={_slp.max()-_slp.min():.1f} hPa')
    print(f'  HL_SIGMA={HL_SIGMA} HL_NEIGHBORHOOD={HL_NEIGHBORHOOD} HL_MIN_DELTA={HL_MIN_DELTA}')
    from scipy.ndimage import maximum_filter, minimum_filter, gaussian_filter as _gf
    _sg = _gf(_slp, sigma=HL_SIGMA)
    _mxf = maximum_filter(_sg, size=HL_NEIGHBORHOOD)
    _mnf = minimum_filter(_sg, size=HL_NEIGHBORHOOD)
    _is_max = (np.abs(_sg - _mxf) < 1e-6) & (_sg - _mnf > HL_MIN_DELTA)
    _is_min = (np.abs(_sg - _mnf) < 1e-6) & (_mxf - _sg > HL_MIN_DELTA)
    print(f'  max candidates: {_is_max.sum()}  min candidates: {_is_min.sum()}')
    # vectorised grid lookup for all metar lats/lons at once
    _lat0, _lat1 = _latv[0], _latv[-1]
    _lon0, _lon1 = _lonv[0], _lonv[-1]
    _ri = np.clip(
        np.round((_mr_lats - _lat0) / (_lat1 - _lat0) * (len(_latv) - 1)).astype(int),
        0, len(_latv) - 1
    )
    _ci = np.clip(
        np.round((_mr_lons - _lon0) / (_lon1 - _lon0) * (len(_lonv) - 1)).astype(int),
        0, len(_lonv) - 1
    )
    _mr_grid_vals = _slp[_ri, _ci]

    _centers = find_hl_centers(_slp, _lonv, _latv,
                               neighborhood=HL_NEIGHBORHOOD,
                               min_delta=HL_MIN_DELTA)
    hl_centers_by_key[_key] = _centers
    _h = [c for c in _centers if c['type'] == 'H']
    _l = [c for c in _centers if c['type'] == 'L']
    print(f'  {_key}: {len(_h)} High(s), {len(_l)} Low(s)')
    for c in _centers:
        print(f"    {c['type']}  {c['val']:.1f} hPa  "
              f"@ {c['lat']:.2f}°N  {c['lon']:.2f}°E")

# keep last key's centers as default for backward compat
if _sfc_keys:
    hl_centers = hl_centers_by_key.get(_sfc_keys[-1], [])
    slp_grid   = globals().get(f'slp_grid_{_sfc_keys[-1]}')
    lon_vec    = globals().get(f'lon_vec_{_sfc_keys[-1]}')
    lat_vec    = globals().get(f'lat_vec_{_sfc_keys[-1]}')

print('\n✅ Block 03 complete — hl_centers list ready')

# @title TUNING PARAMETERS for UA
# ══════════════════════════════════════════════════════════════════════════
#  TUNING PARAMETERS
# ══════════════════════════════════════════════════════════════════════════

# Grid resolution: increase N for finer output, decrease for speed.
# At GRIB2 density, 300×300 is already over-resolved; 200 is plenty.
_GRID_N = 200

# Dense-data decimation: keep at most this many points per field per level.
# RBF is O(n³); griddata/linear is O(n log n). But we still don't need
_MAX_PTS = 1000   # after spatial dedupe; tune up if you want more detail


# Gaussian smoothing (sigma in grid cells, applied AFTER interpolation)
_SIGMA = {'HGHT': 2.0, 'TEMP': 2.0, 'TTDP': 10, 'SPED': 10}

# instability
sigmaT700500 = 5.0

# Contour intervals
_INTERVALS = {'HGHT': 6.0, 'TEMP': 2.0, 'TTDP': 2.0, 'SPED': 5.0}


HL_LEVELS           = [850, 700, 500]
HL_SMOOTH_N         = 5
HL_MIN_PERSISTENCE  = {850: 5.0,   700: 5.0,   500: 10.0}
HL_MIN_DISTANCE_KM  = {850: 250.0, 700: 280.0, 500: 300.0}
HL_EDGE_SKIP_DEG    = 2.5

UA_HGHT_LEVELS = {
    850: np.arange(1140, 1650, 30),
    700: np.arange(2520, 3180, 60),
    500: np.arange(4800, 6000, 60),
    250: None,
}

UA_TEMP_BAND_OPACITY = 0.25
UA_TEMP_BAND_SHOW    = True

print('=' * 60)
print('  TUNING PARAMETERS HERE - UA band opacity, smoothing, data density')
print('=' * 60)

# @title  BLOCK 04A · Normal Temperature Bands & Colour Tables
# ══════════════════════════════════════════════════════════════════════════
#  Run this cell BEFORE Cell 1.
#  Exports (used by Cell 1):
#    UA_TEMP_BANDS_850, UA_TEMP_BANDS_500, UA_TEMP_BANDS
#    _normal_850_hi, _normal_850_lo
#    _normal_500_hi, _normal_500_lo
#    _TODAY
# ══════════════════════════════════════════════════════════════════════════

import io as _io
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.patches as _mpatches
from datetime import date as _date

print('=' * 60)
print('  BLOCK 05 — Cell 2')
print('  Normal Temperature Bands & Colour Tables')
print('=' * 60)


# ══════════════════════════════════════════════════════════════════════════
#  DATE-AWARE NORMAL BAND TABLES
# ══════════════════════════════════════════════════════════════════════════

# Each entry: (month, day, normal_hi °C, normal_lo °C)
_NORMALS_850 = [
    ( 1,  1,  -6,  -8), ( 1,  4,  -8, -10), ( 1, 15,  -6,  -8),
    ( 1, 18,  -4,  -6), ( 1, 24,  -6,  -8), ( 1, 31,  -8, -10),
    ( 2,  4,  -6,  -8), ( 3,  9,  -4,  -6), ( 3, 12,  -2,  -4),
    ( 4,  3,   0,  -2), ( 4,  5,   2,   0), ( 4,  8,   4,   2),
    ( 5,  2,   6,   4), ( 5, 11,   8,   6), ( 5, 23,  10,   8),
    ( 6,  1,  12,  10), ( 6, 27,  14,  12), ( 8, 24,  12,  10),
    ( 9,  5,  10,   8), ( 9, 17,   8,   6), (10,  1,   6,   4),
    (10, 11,   4,   2), (10, 25,   2,   0), (10, 29,   0,  -2),
    (11,  8,  -2,  -4), (11, 12,   0,  -2), (11, 17,  -2,  -4),
    (11, 26,  -4,  -6), (12,  2,  -6,  -8),
]

_NORMALS_500 = [
    ( 1,  1, -28, -30), ( 1, 15, -26, -28), ( 1, 24, -28, -30),
    ( 2, 23, -30, -32), ( 2, 27, -28, -30), ( 3,  9, -26, -28),
    ( 3, 12, -28, -30), ( 4,  5, -26, -28), ( 4, 19, -24, -26),
    ( 4, 27, -22, -24), ( 5, 11, -20, -22), ( 5, 23, -18, -20),
    ( 6, 13, -16, -18), ( 6, 27, -14, -16), ( 8,  4, -12, -14),
    ( 8, 11, -14, -16), ( 8, 31, -16, -18), ( 9, 17, -18, -20),
    (10,  3, -20, -22), (10, 21, -22, -24), (10, 29, -24, -26),
    (11, 17, -26, -28), (12,  2, -28, -30),
]

_HEIGHT_CONTROL = {
    "Jan 1":  5400, "Apr 3":  5460, "Apr 19": 5520, "May 11": 5580,
    "May 30": 5640, "Jun 27": 5700, "Jul 26": 5760, "Aug 7":  5700,
    "Aug 31": 5640, "Oct 1":  5580, "Oct 17": 5520, "Oct 29": 5460,
    "Nov 17": 5400,
}

_MONTHS = ['Jan','Feb','Mar','Apr','May','Jun',
           'Jul','Aug','Sep','Oct','Nov','Dec']


def _doy(month, day):
    try:
        return _date(2001, month, day).timetuple().tm_yday
    except ValueError:
        return 999


def _get_normal_band(pressure_level, today=None):
    if today is None:
        today = _date.today()
    table     = _NORMALS_850 if pressure_level == 850 else _NORMALS_500
    today_doy = today.timetuple().tm_yday
    best_hi = best_lo = None
    best_doy = -1
    for (m, d, hi, lo) in table:
        entry_doy = _doy(m, d)
        if entry_doy <= today_doy and entry_doy > best_doy:
            best_doy = entry_doy
            best_hi, best_lo = hi, lo
    if best_hi is None:
        best_hi, best_lo = table[-1][2], table[-1][3]
    return best_hi, best_lo


# ══════════════════════════════════════════════════════════════════════════
#  COLOUR TABLES
# ══════════════════════════════════════════════════════════════════════════

# Base colour table; green band sits at [-6, -8] °C.
# Shifted at runtime to align with the current-day normal.
_UA_TEMP_BANDS_BASE = [

    ( 24,  22, '#8800cc'), ( 22,  20, '#ffffff'), ( 20,  18, '#aaaaaa'),
    ( 18,  16, '#ffffff'), ( 16,  14, '#8b4513'),
    ( 14,  12, '#ffffff'), ( 12,  10, '#ffc0cb'), ( 10,   8, '#ffffff'),
    (  8,   6, '#e35335'), (  6,   4, '#ffffff'), (  4,   2, '#ff8c00'),
    (  2,   0, '#ffffff'), (  0,  -2, '#ffff00'), ( -2,  -4, '#ffffff'),
    ( -4,  -6, '#00cc00'), ( -6,  -8, '#ffffff'), ( -8, -10, '#45caff'),
    (-10, -12, '#ffffff'), (-12, -14, '#0066ff'), (-14, -16, '#ffffff'),
    (-16, -18, '#8800cc'), (-18, -20, '#ffffff'), (-20, -22, '#aaaaaa'),
    (-22, -24, '#ffffff'), (-24, -26, '#8b4513'), (-26, -28, '#ffffff'),
]

_STATIC_GREEN_LO = -6   # the base table's green lower edge


def _make_ua_temp_bands(pressure_level, today=None):
    """Shift the base colour table so green aligns with today's normal."""
    if today is None:
        today = _date.today()
    normal_hi, normal_lo = _get_normal_band(pressure_level, today)
    shift = normal_lo - _STATIC_GREEN_LO
    return [(bhi + shift, blo + shift, col)
            for (bhi, blo, col) in _UA_TEMP_BANDS_BASE]


# ── Compute for today ─────────────────────────────────────────────────────
_TODAY = _date.today()

_normal_850_hi, _normal_850_lo = _get_normal_band(850, _TODAY)
_normal_500_hi, _normal_500_lo = _get_normal_band(500, _TODAY)

UA_TEMP_BANDS_850 = _make_ua_temp_bands(850, _TODAY)
UA_TEMP_BANDS_500 = _make_ua_temp_bands(500, _TODAY)
UA_TEMP_BANDS     = UA_TEMP_BANDS_850   # default alias


# ══════════════════════════════════════════════════════════════════════════
#  NORMAL TABLE PRINT-OUT
# ══════════════════════════════════════════════════════════════════════════

_all_dates = set()
for m, d, hi, lo in _NORMALS_850: _all_dates.add((m, d))
for m, d, hi, lo in _NORMALS_500: _all_dates.add((m, d))
for s in _HEIGHT_CONTROL:
    mon, day = s.split(' ')
    _all_dates.add((_MONTHS.index(mon) + 1, int(day)))

_today_doy = _TODAY.timetuple().tm_yday
_best_active_doy = max(
    (_doy(m, d) for m, d in _all_dates if _doy(m, d) <= _today_doy),
    default=max(_doy(m, d) for m, d in _all_dates)
)

_HIGHLIGHT = '\033[103m'
_RESET     = '\033[0m'

print(f'\n  Normal temperature bands  —  {_TODAY}')
print(f"  {'Date':<12} {'850 hPa':>12} {'500 hPa':>12} {'500 hPa Hgt':>13}")
print('  ' + '-' * 52)
for _m, _d in sorted(_all_dates):
    _lbl       = f"{_MONTHS[_m-1]} {_d}"
    _entry_doy = _doy(_m, _d)
    _mark      = ' ◀ active' if _entry_doy == _best_active_doy else ''
    _n850 = next(((hi, lo) for mm, dd, hi, lo in _NORMALS_850
                  if mm == _m and dd == _d), None)
    _n500 = next(((hi, lo) for mm, dd, hi, lo in _NORMALS_500
                  if mm == _m and dd == _d), None)
    _hgt  = next((v for k, v in _HEIGHT_CONTROL.items() if k == _lbl), None)
    _c850 = f"{_n850[1]}→{_n850[0]}°C" if _n850 else '—'
    _c500 = f"{_n500[1]}→{_n500[0]}°C" if _n500 else '—'
    _chgt = f"{_hgt} m"                 if _hgt  else '—'
    line  = f"  {_lbl:<12} {_c850:>12} {_c500:>12} {_chgt:>13}{_mark}"
    print(f"{_HIGHLIGHT}{line}{_RESET}" if _entry_doy == _best_active_doy else line)

print(f'\n  850 hPa normal: {_normal_850_lo} to {_normal_850_hi} °C')
print(f'  500 hPa normal: {_normal_500_lo} to {_normal_500_hi} °C')


# ══════════════════════════════════════════════════════════════════════════
#  BAND SCALE VISUALISATION
# ══════════════════════════════════════════════════════════════════════════

def _plot_band_legend(ax, bands, normal_hi, normal_lo, vmin=-60, vmax=60, title=''):
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title(title, fontsize=11, pad=6)

    for (hi, lo, col) in bands:
        lo_c = max(lo, vmin)
        hi_c = min(hi, vmax)
        if hi_c <= lo_c:
            continue
        ax.add_patch(_mpatches.FancyBboxPatch(
            (lo_c, 0), hi_c - lo_c, 1,
            boxstyle='square,pad=0',
            facecolor=col, edgecolor='#888888', linewidth=0.4))

    boundaries = (
        {lo for _, lo, _ in bands if vmin <= lo <= vmax} |
        {hi for hi, _, _ in bands if vmin <= hi <= vmax}
    )
    for x in sorted(boundaries):
        ax.axvline(x, color='#666666', linewidth=0.5, alpha=0.7, zorder=4)
        ax.text(x, -0.18, str(int(x)), ha='center', va='top', fontsize=7,
                color='#333333',
                transform=ax.get_xaxis_transform(), clip_on=False)

    mid = (normal_lo + normal_hi) / 2
    ax.axvline(mid, color='black', linewidth=1.2, linestyle='--',
               alpha=0.65, zorder=5)
    ax.annotate(
        f'Normal  {normal_lo}→{normal_hi} °C',
        xy=(mid, 1), xytext=(mid, 1.38),
        fontsize=8, ha='center', color='#333333',
        arrowprops=dict(arrowstyle='->', color='#555555', lw=0.8),
        annotation_clip=False)

    ax.set_xticks([])
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)


_fig_demo, _axes_demo = plt.subplots(
    2, 1, figsize=(14, 4), gridspec_kw={'hspace': 1.0})
_fig_demo.suptitle(
    f'UA Temperature Band Scales — {_TODAY.strftime("%d %b %Y")}',
    fontsize=12, y=1.04)

_plot_band_legend(
    _axes_demo[0], UA_TEMP_BANDS_850, _normal_850_hi, _normal_850_lo,
    title=f'850 hPa  (normal {_normal_850_lo}→{_normal_850_hi} °C = green)')
_plot_band_legend(
    _axes_demo[1], UA_TEMP_BANDS_500, _normal_500_hi, _normal_500_lo,
    title=f'500 hPa  (normal {_normal_500_lo}→{_normal_500_hi} °C = green)')

plt.close(_fig_demo)

print('\n✅ Cell 2 complete — band tables and colour scales ready.')
print(f'   Exports: UA_TEMP_BANDS_850, UA_TEMP_BANDS_500, UA_TEMP_BANDS')
print(f'            _normal_850_hi/lo, _normal_500_hi/lo, _TODAY')

# @title  BLOCK 5A · Contour Calculation & Upper-Air Processing
# ══════════════════════════════════════════════════════════════════════════
#  Depends on:
#    - ua_summary_df          (DataFrame from earlier block)
#    - UA_TEMP_BANDS_850      )
#    - UA_TEMP_BANDS_500      ) produced by Cell 2 (run that first)
#    - UA_HGHT_LEVELS         )
#    - HL_*  config constants )
#    - _GRID_N, _MAX_PTS, _SIGMA, _INTERVALS, sigmaT700500
#  Exports:
#    - _ts_ua                 (dict, patched by Cell 1B)
#    - _tgrid_cache           (dict, consumed by Cell 1B)
#    - _synoptic_times        (list, used by Cell 9)
#    - _ts_ua_json_str        (set to empty by Cell 1B after patching)
# ══════════════════════════════════════════════════════════════════════════

import metpy

print('=' * 60)
print('  BLOCK 05 — Cell 1A')
print('  Contour Calculation & Upper-Air Processing')
print('  HEIGHT, TEMP, T-Td, WIND SPEED')
print('  H/L detection via MetPy peak_persistence')
print('  OPTIMISED for dense GRIB2 grids (scipy/numpy griddata path)')
print('  NOTE: Band fills are built by Cell 1B — run that next.')
print('=' * 60)

import io as _io
import json as _json_ua
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor, as_completed

from metpy.calc import peak_persistence, smooth_gaussian
from metpy.units import units




# ══════════════════════════════════════════════════════════════════════════
#  FAST GRID HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _decimate(lons, lats, vals, max_pts=_MAX_PTS):
    if len(vals) <= max_pts:
        return lons, lats, vals
    n_side   = int(np.ceil(np.sqrt(max_pts)))
    lon_bins = np.linspace(lons.min(), lons.max(), n_side + 1)
    lat_bins = np.linspace(lats.min(), lats.max(), n_side + 1)
    lon_idx  = np.clip(np.searchsorted(lon_bins, lons) - 1, 0, n_side - 1)
    lat_idx  = np.clip(np.searchsorted(lat_bins, lats) - 1, 0, n_side - 1)
    cell_key = lat_idx * n_side + lon_idx
    _, first = np.unique(cell_key, return_index=True)
    return lons[first], lats[first], vals[first]


def _make_grid(lons, lats, vals, N=_GRID_N, pad=1.5, method='linear'):
    lon_vec = np.linspace(lons.min() - pad, lons.max() + pad, N)
    lat_vec = np.linspace(lats.min() - pad, lats.max() + pad, N)
    glon, glat = np.meshgrid(lon_vec, lat_vec)
    pts  = np.column_stack([lons, lats])
    grid = griddata(pts, vals, (glon, glat), method=method)
    nan_mask = np.isnan(grid)
    if nan_mask.any():
        grid_nn = griddata(pts, vals, (glon, glat), method='nearest')
        grid[nan_mask] = grid_nn[nan_mask]
    return grid, lon_vec, lat_vec


def _smooth(grid, sigma):
    return gaussian_filter(grid, sigma=sigma)


def _extract_contours(glon2d, glat2d, grid, levels):
    fig, ax = plt.subplots(figsize=(1, 1))
    try:
        cs = ax.contour(glon2d, glat2d, grid, levels=levels)
    except Exception:
        plt.close(fig)
        return []
    segs = []
    for li, lv in enumerate(cs.levels):
        for coords in cs.allsegs[li]:
            if len(coords) < 2:
                continue
            mid = coords[len(coords) // 2]
            segs.append({
                'level':     float(lv),
                'coords':    [[float(c[0]), float(c[1])] for c in coords],
                'label_lon': float(mid[0]),
                'label_lat': float(mid[1]),
            })
    plt.close(fig)
    return segs


# ══════════════════════════════════════════════════════════════════════════
#  H/L DETECTION
# ══════════════════════════════════════════════════════════════════════════

def _haversine_km(lat1, lon1, lat2, lon2):
    R    = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a    = (np.sin(dlat/2)**2
            + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
            * np.sin(dlon/2)**2)
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _find_ua_hl_metpy(hght_grid, lon_vec, lat_vec, pressure_level):
    min_pers    = HL_MIN_PERSISTENCE.get(pressure_level, 30.0)
    min_dist_km = HL_MIN_DISTANCE_KM.get(pressure_level, 400.0)
    edge_skip   = HL_EDGE_SKIP_DEG
    grid_q = hght_grid * units.meter
    sm     = smooth_gaussian(grid_q, n=HL_SMOOTH_N).magnitude
    all_centers = []
    for typ, maxima in [('H', True), ('L', False)]:
        pp       = peak_persistence(sm, maxima=maxima)
        accepted = []
        for (r, c), pers in pp:
            if pers != float('inf') and pers < min_pers:
                continue
            lat = float(lat_vec[r])
            lon = float(lon_vec[c])
            if (lat < lat_vec[0]  + edge_skip or lat > lat_vec[-1] - edge_skip or
                    lon < lon_vec[0]  + edge_skip or lon > lon_vec[-1] - edge_skip):
                continue
            too_close = any(
                _haversine_km(lat, lon, p['lat'], p['lon']) < min_dist_km
                for p in accepted
            )
            if not too_close:
                accepted.append({'type': typ, 'lat': lat, 'lon': lon,
                                 'val':  float(sm[r, c]),
                                 'persistence': float(pers)})
                print(f'        ✓ {typ}  lat={lat:.1f}  lon={lon:.1f}  val={sm[r,c]:.0f}m  pers={pers:.2f}')
        all_centers.extend(accepted)
    return all_centers




# ══════════════════════════════════════════════════════════════════════════
#  W/C DETECTION
# ══════════════════════════════════════════════════════════════════════════
# ── W/C (Warm/Cold centre) detection config ───────────────────────────────
WC_LEVELS         = [850, 500]          # pressure levels to detect on
WC_MIN_PERSISTENCE = {850: 1.0, 500: 1.0}   # °C
WC_MIN_DISTANCE_KM = {850: 400.0, 500: 400.0}
WC_EDGE_SKIP_DEG  = HL_EDGE_SKIP_DEG        # reuse same edge margin
WC_SMOOTH_N       = HL_SMOOTH_N             # reuse same MetPy smoothing


def _find_wc_metpy(temp_grid, lon_vec, lat_vec, pressure_level):
    """Detect warm (W) / cold (C) temperature centres via MetPy peak_persistence."""
    min_pers    = WC_MIN_PERSISTENCE.get(pressure_level, 2.0)
    min_dist_km = WC_MIN_DISTANCE_KM.get(pressure_level, 350.0)
    edge_skip   = WC_EDGE_SKIP_DEG

    grid_q = temp_grid * units.degC
    sm     = smooth_gaussian(grid_q, n=WC_SMOOTH_N).magnitude
    print(f'      W/C {pressure_level}hPa: T range [{sm.min():.1f}, {sm.max():.1f}]°C  min_pers={min_pers}')
    all_centers = []

    for typ, maxima in [('W', True), ('C', False)]:
        pp       = peak_persistence(sm, maxima=maxima)
        print(f'      W/C {pressure_level}hPa {typ}: {len(pp)} raw candidates from peak_persistence')
        accepted = []
        for (r, c), pers in pp:
            if pers != float('inf') and pers < min_pers:
                continue
            lat = float(lat_vec[r])
            lon = float(lon_vec[c])
            if (lat < lat_vec[0]  + edge_skip or lat > lat_vec[-1] - edge_skip or
                    lon < lon_vec[0]  + edge_skip or lon > lon_vec[-1] - edge_skip):
                continue
            too_close = any(
                _haversine_km(lat, lon, p['lat'], p['lon']) < min_dist_km
                for p in accepted
            )
            if not too_close:
                accepted.append({'type': typ, 'lat': lat, 'lon': lon,
                                 'val':  float(sm[r, c]),
                                 'persistence': float(pers)})
                print(f'        ✓ {typ}  lat={lat:.1f}  lon={lon:.1f}  val={sm[r,c]:.1f}°C  pers={pers:.2f}')
        all_centers.extend(accepted)
    return all_centers
# ══════════════════════════════════════════════════════════════════════════
#  PER-LEVEL WORKER
# ══════════════════════════════════════════════════════════════════════════

def _process_level(_df_hr, _plvl, _bands_850, _bands_500, _hght_levels, _key):
    _lvl_data = {}
    _col      = lambda f: f'{f}_{_plvl}'
    _n        = _GRID_N

    def _pts_for(col):
        mask = (_df_hr[col].notna()
                if col in _df_hr.columns
                else pd.Series(False, index=_df_hr.index))
        sub  = _df_hr[mask]
        if sub.empty:
            return None, None, None
        lats = sub['lat'].values.astype(float)
        lons = sub['lon'].values.astype(float)
        vals = sub[col].values.astype(float)
        keys = (np.round(lats, 2) * 1000 + np.round(lons, 2)).astype(np.int64)
        _, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
        if cnt.max() > 1:
            lats_u = np.zeros(cnt.size); lons_u = np.zeros(cnt.size)
            vals_u = np.zeros(cnt.size)
            np.add.at(lats_u, inv, lats); np.add.at(lons_u, inv, lons)
            np.add.at(vals_u, inv, vals)
            lats = lats_u / cnt; lons = lons_u / cnt; vals = vals_u / cnt
        lons, lats, vals = _decimate(lons, lats, vals, _MAX_PTS)
        return lons, lats, vals

    # ── Height ────────────────────────────────────────────────────────────
    _fixed          = _hght_levels.get(_plvl)
    _hght_segs      = []
    _hght_grid_out  = (None, None, None)
    if _fixed is not None:
        lons, lats, vals = _pts_for(_col('HGHT'))
        if lons is not None and len(lons) >= 8:
            grid, lv, ltv = _make_grid(lons, lats, vals, N=_n)
            grid           = _smooth(grid, _SIGMA['HGHT'])
            _hght_grid_out = (grid, lv, ltv)
            glon2d, glat2d = np.meshgrid(lv, ltv)
            _hght_segs     = _extract_contours(glon2d, glat2d, grid, _fixed)
    _lvl_data['hght'] = _hght_segs

    # ── Temperature (contours only — fills done in Cell 1B) ───────────────
    _temp_segs = []
    lons_t, lats_t, vals_t = _pts_for(_col('TEMP'))
    if lons_t is not None and len(lons_t) >= 8:
        tgrid, lv_t, ltv_t = _make_grid(lons_t, lats_t, vals_t, N=_n)
        tgrid = _smooth(tgrid, _SIGMA['TEMP'])
        glon2d, glat2d = np.meshgrid(lv_t, ltv_t)
        vmin     = np.floor(tgrid.min() / _INTERVALS['TEMP']) * _INTERVALS['TEMP']
        vmax     = np.ceil( tgrid.max() / _INTERVALS['TEMP']) * _INTERVALS['TEMP']
        t_levels = np.arange(vmin, vmax + _INTERVALS['TEMP'], _INTERVALS['TEMP'])
        if len(t_levels) >= 2:
            _temp_segs = _extract_contours(glon2d, glat2d, tgrid, t_levels)
        _bands_for_lvl = (_bands_500 if _plvl in [500, 250] else _bands_850)
        # cache grid for Cell 1B
        _tgrid_cache[f'{_plvl}_{_key}'] = (tgrid, lv_t, ltv_t, _bands_for_lvl)
    _lvl_data['temp']            = _temp_segs
    _lvl_data['temp_band_fills'] = []   # patched by Cell 1B

    # ── T-Td ──────────────────────────────────────────────────────────────
    _ttdp_segs = []
    _t_col, _d_col = _col('TEMP'), _col('DWPT')
    if _t_col in _df_hr.columns and _d_col in _df_hr.columns:
        mask = _df_hr[_t_col].notna() & _df_hr[_d_col].notna()
        sub  = _df_hr[mask]
        if len(sub) >= 8:
            lons_td = sub['lon'].values.astype(float)
            lats_td = sub['lat'].values.astype(float)
            vals_td = (sub[_t_col] - sub[_d_col]).values.astype(float)
            lons_td, lats_td, vals_td = _decimate(lons_td, lats_td, vals_td, _MAX_PTS)
            grid_td, lv_td, ltv_td = _make_grid(lons_td, lats_td, vals_td, N=_n)
            grid_td = _smooth(grid_td, _SIGMA['TTDP'])
            glon2d, glat2d = np.meshgrid(lv_td, ltv_td)
            vmin      = np.floor(grid_td.min() / _INTERVALS['TTDP']) * _INTERVALS['TTDP']
            vmax      = np.ceil( grid_td.max() / _INTERVALS['TTDP']) * _INTERVALS['TTDP']
            td_levels = np.arange(vmin, vmax + _INTERVALS['TTDP'], _INTERVALS['TTDP'])
            if len(td_levels) >= 2:
                _ttdp_segs = _extract_contours(glon2d, glat2d, grid_td, td_levels)
    _lvl_data['ttdp'] = _ttdp_segs

    # ── Wind speed ────────────────────────────────────────────────────────
    _sped_segs = []
    lons_s, lats_s, vals_s = _pts_for(_col('SPED'))
    if lons_s is not None and len(lons_s) >= 8:
        grid_s, lv_s, ltv_s = _make_grid(lons_s, lats_s, vals_s, N=_n)
        grid_s = _smooth(grid_s, _SIGMA['SPED'])
        glon2d, glat2d = np.meshgrid(lv_s, ltv_s)
        vmin     = np.floor(grid_s.min() / _INTERVALS['SPED']) * _INTERVALS['SPED']
        vmax     = np.ceil( grid_s.max() / _INTERVALS['SPED']) * _INTERVALS['SPED']
        s_levels = np.arange(vmin, vmax + _INTERVALS['SPED'], _INTERVALS['SPED'])
        if len(s_levels) >= 2:
            _sped_segs = _extract_contours(glon2d, glat2d, grid_s, s_levels)
    _lvl_data['sped'] = _sped_segs

    # ── H/L detection ─────────────────────────────────────────────────────
    _ua_hl = []
    if _plvl in HL_LEVELS:
        hg, lv_hl, ltv_hl = _hght_grid_out
        if hg is not None:
            _ua_hl = _find_ua_hl_metpy(hg, lv_hl, ltv_hl, _plvl)

    # ── W/C detection ─────────────────────────────────────────────────────
    _ua_wc = []
    if _plvl in WC_LEVELS:
        cached = _tgrid_cache.get(f'{_plvl}_{_key}')
        if cached is not None:
            tg, lv_wc, ltv_wc, _ = cached
            _ua_wc = _find_wc_metpy(tg, lv_wc, ltv_wc, _plvl)

    return _plvl, _lvl_data, _ua_hl, _ua_wc


# ══════════════════════════════════════════════════════════════════════════
#  BUILD SYNOPTIC TIME LIST
# ══════════════════════════════════════════════════════════════════════════

ua_summary_df['_vt']   = pd.to_datetime(ua_summary_df['valid_time'])
ua_summary_df['_date'] = ua_summary_df['_vt'].dt.date
ua_summary_df['_hour'] = ua_summary_df['_vt'].dt.hour

_synoptic_times = sorted(
    ua_summary_df[['_date', '_hour']]
    .drop_duplicates()
    .itertuples(index=False, name=None)
)


# ══════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════

_tgrid_cache = {}
_ts_ua       = {}

for (_date_val, _hr) in _synoptic_times:
    _t0        = time.time()
    _date_str  = pd.Timestamp(_date_val).strftime('%Y%m%d')
    _key       = f'{_date_str}_{int(_hr):02d}'
    _valid_str = f'{_date_str} {int(_hr):02d}Z'

    _df_hr = ua_summary_df[
        (ua_summary_df['_date'] == _date_val) &
        (ua_summary_df['_hour'] == _hr)
    ].copy()

    _dominant_vt = _df_hr['valid_time'].value_counts().idxmax()
    _df_hr       = _df_hr[_df_hr['valid_time'] == _dominant_vt].copy()

    print(f'\n  Processing {_valid_str} ({len(_df_hr)} stations)...')
    if len(_df_hr) < 8:
        print(f'  ⚠ Too few stations ({len(_df_hr)}) — skipping.')
        _ts_ua[_key] = {
            'levels': {str(pl): {'hght': [], 'temp': [], 'temp_band_fills': [],
                                 'ttdp': [], 'sped': []}
                       for pl in [850, 700, 500, 250]},
            'instab': [],
            'thermal_ridge_850': [], 'thermal_trough_850': [],
            'thermal_ridge_700': [], 'thermal_trough_700': [],
            'thermal_ridge_500': [], 'thermal_trough_500': [],
            'dtdx_zero_pts': [],
            **{f'hl_{pl}': [] for pl in HL_LEVELS},
        }
        continue

    _hr_data   = {}
    _ua_hl_all = {}
    _ua_wc_all = {}


    with ThreadPoolExecutor(max_workers=4) as _pool:
        _futs = {_pool.submit(_process_level, _df_hr, pl,
                              UA_TEMP_BANDS_850, UA_TEMP_BANDS_500,
                              UA_HGHT_LEVELS, _key): pl
                 for pl in [850, 700, 500, 250]}
        for _fut in as_completed(_futs):
            _plvl, _lvl_data, _hl_list, _wc_list = _fut.result()
            _hr_data[str(_plvl)] = _lvl_data
            _ua_hl_all[_plvl]    = _hl_list
            _ua_wc_all[_plvl]    = _wc_list
            print(f'    {_plvl} hPa  '
                  f'HGHT:{len(_lvl_data["hght"])}  '
                  f'TEMP:{len(_lvl_data["temp"])}  '
                  f'fills:0 (Cell 1B)  '
                  f'T-Td:{len(_lvl_data["ttdp"])}  '
                  f'WIND:{len(_lvl_data["sped"])}  '
                  f'HL:{len(_hl_list)}  WC:{len(_wc_list)}')
            for c in _hl_list:
                print(f'      {c["type"]}  lat={c["lat"]:.1f}  lon={c["lon"]:.1f}'
                      f'  val={c["val"]:.0f}m  pers={c["persistence"]:.1f}')
            for c in _wc_list:
                print(f'      {c["type"]}  lat={c["lat"]:.1f}  lon={c["lon"]:.1f}'
                      f'  val={c["val"]:.1f}°C  pers={c["persistence"]:.1f}')

    # ── Instability (T700 - T500) ─────────────────────────────────────────
    _instab_cts = []
    _col700, _col500 = 'TEMP_700', 'TEMP_500'
    if _col700 in _df_hr.columns and _col500 in _df_hr.columns:
        _mask_i = _df_hr[_col700].notna() & _df_hr[_col500].notna()
        _sub_i  = _df_hr[_mask_i]
        if len(_sub_i) >= 4:
            _lons_i = _sub_i['lon'].values.astype(float)
            _lats_i = _sub_i['lat'].values.astype(float)
            _vals_i = (_sub_i[_col700] - _sub_i[_col500]).values.astype(float)
            _pad = 1.5; _NI = 180
            _ilon  = np.linspace(_lons_i.min() - _pad, _lons_i.max() + _pad, _NI)
            _ilat  = np.linspace(_lats_i.min() - _pad, _lats_i.max() + _pad, _NI)
            _iglon, _iglat = np.meshgrid(_ilon, _ilat)
            _tree         = cKDTree(np.column_stack([_lons_i, _lats_i]))
            _diff_grid = griddata(
              np.column_stack([_lons_i, _lats_i]),
              _vals_i,
              (_iglon, _iglat),
              method='linear'
            )
            _nan_mask = np.isnan(_diff_grid)
            if _nan_mask.any():
                _dists, _idxs = _tree.query(
                    np.column_stack([_iglon.ravel(), _iglat.ravel()]), k=1)
                _nn = _vals_i[_idxs].astype(float)
                _nn[_dists > 3.5] = np.nan
                _diff_grid[_nan_mask] = _nn.reshape(_NI, _NI)[_nan_mask]
            _diff_grid_sm  = gaussian_filter(
                np.where(np.isnan(_diff_grid), 0, _diff_grid),
                sigma=max(1, int(sigmaT700500 * 2)))
            _diff_grid_sm[np.isnan(_diff_grid)] = np.nan
            for _band_lvl in [16, 18]:
                _binary = np.where(
                    (~np.isnan(_diff_grid_sm)) & (_diff_grid_sm >= _band_lvl) &
                    (_diff_grid_sm < (_band_lvl + 2) if _band_lvl == 16
                     else np.ones_like(_diff_grid_sm, bool)),
                    1.0, 0.0)
                if _binary.max() < 0.5:
                    continue
                _fig_i, _ax_i = plt.subplots(figsize=(1, 1))
                try:
                    _cs_i = _ax_i.contour(_iglon, _iglat, _binary, levels=[0.5])
                    for _seg in _cs_i.allsegs[0]:
                        if len(_seg) < 3:
                            continue
                        _mid_i = _seg[len(_seg) // 2]
                        _instab_cts.append({
                            'coords':    [[float(p[1]), float(p[0])] for p in _seg],
                            'label_lon': float(_mid_i[0]),
                            'label_lat': float(_mid_i[1]),
                        })
                except Exception:
                    pass
                plt.close(_fig_i)
    print(f'    Instability: {len(_instab_cts)} segs')

    def _seg_to_dict(seg):
      mid = seg[len(seg) // 2]
      return {'coords':    [[float(p[0]), float(p[1])] for p in seg],
            'label_lon': float(mid[0]),
            'label_lat': float(mid[1])}

    _ts_ua[_key] = {
        'levels':             _hr_data,
        'instab':             _instab_cts,
        'thermal_ridge_850':  [_seg_to_dict(s) for s in globals().get(f'ridge_segs_{_key}',        [])],
        'thermal_trough_850': [_seg_to_dict(s) for s in globals().get(f'trough_segs_{_key}',       [])],
        'thermal_ridge_700':  [_seg_to_dict(s) for s in globals().get(f'ridge_segs_700_{_key}',    [])],
        'thermal_trough_700': [_seg_to_dict(s) for s in globals().get(f'trough_segs_700_{_key}',   [])],
        'thermal_ridge_500':  [_seg_to_dict(s) for s in globals().get(f'ridge_segs_500_{_key}',    [])],
        'thermal_trough_500': [_seg_to_dict(s) for s in globals().get(f'trough_segs_500_{_key}',   [])],
        'dtdx_zero_pts':      [],
        **{f'hl_{pl}': _ua_hl_all.get(pl, []) for pl in HL_LEVELS},
        **{f'wc_{pl}': _ua_wc_all.get(pl, []) for pl in WC_LEVELS},

    }
    print(f'    → stored {_key}  ({time.time() - _t0:.1f}s)')

# ── H/L diagnostic ────────────────────────────────────────────────────────
print('\n  H/L diagnostic:')
for (_date_val, _hr) in _synoptic_times:
    _date_str = pd.Timestamp(_date_val).strftime('%Y%m%d')
    _k = f'{_date_str}_{int(_hr):02d}'
    for _pl in HL_LEVELS:
        _hl_list = _ts_ua[_k].get(f'hl_{_pl}', 'KEY MISSING')
        print(f'    {_k} {_pl}hPa HL: {_hl_list}')
    for _pl in WC_LEVELS:
        _wc_list = _ts_ua[_k].get(f'wc_{_pl}', 'KEY MISSING')
        print(f'    {_k} {_pl}hPa WC: {_wc_list}')

print(f'\n✅ Cell 1A complete — {len(_ts_ua)} synoptic time(s)')
print(f'   _tgrid_cache keys: {sorted(_tgrid_cache.keys())}')
print(f'   ▶ Run Cell 1B to build temperature band fills.')

_ts_ua_json_str = _json_ua.dumps(_ts_ua)
print(f'  _ts_ua_json_str: {len(_ts_ua_json_str)//1024} KB')

# @title BLOCK 5B - How temp band get filled

# How temp band get filled

def _build_temp_band_fills(grid, lon_vec, lat_vec, ua_temp_bands, tb_base):
    bands = [(round(min(b[0], b[1]), 1), round(max(b[0], b[1]), 1), b[2])
             for b in ua_temp_bands]
    bands = [(lo + tb_base, hi + tb_base, col) for lo, hi, col in bands]
    bands.sort(key=lambda x: x[0])
    colored_bands = bands
    if not colored_bands:
        return []

    all_edges = sorted(
        {lo for lo, hi, col in colored_bands} |
        {hi for lo, hi, col in colored_bands}
    )
    dmin, dmax = float(grid.min()), float(grid.max())
    MARGIN = 0.01
    active_edges = sorted(set(
        [dmin - MARGIN]
        + [e for e in all_edges]
        + [dmax + MARGIN]
    ))
# Still need at least one interior band boundary
    if len(active_edges) < 2:
        return []

    glon, glat = np.meshgrid(lon_vec, lat_vec)
    fig, ax = plt.subplots(figsize=(1, 1))
    all_fills = []

    try:
        cf = ax.contourf(glon, glat, grid, levels=active_edges)

        from shapely.geometry import Polygon as _SP, MultiPolygon as _MP
        from shapely.ops import unary_union as _UU

        def _segs_to_shapely(segs):
            """Convert matplotlib contourf segments to Shapely geometry.
            Uses spatial containment to identify holes vs outer fills —
            winding order is unreliable across matplotlib versions."""
            if not segs:
                return None

            polys = []
            for seg in segs:
                if len(seg) < 3:
                    continue
                try:
                    p = _SP(seg)
                    if not p.is_valid:
                        p = p.buffer(0)
                    if p.is_valid and not p.is_empty:
                        polys.append(p)
                except Exception:
                    continue

            if not polys:
                return None

            # Sort largest→smallest so containment tests run large-parent-first
            polys.sort(key=lambda p: p.area, reverse=True)

            # Build containment tree using winding order as primary signal,
            # containment only as tiebreaker for ambiguous cases.
            # In contourf: CCW = filled outer, CW = hole punched into parent.
            # Containment alone fails for siblings that share a bounding region.
            n = len(polys)
            depth = [0] * n
            for i in range(1, n):
                # Primary: use winding order
                try:
                    if not polys[i].exterior.is_ccw:
                        # CW = hole — find its containing parent
                        for j in range(i - 1, -1, -1):
                            try:
                                pt = polys[i].representative_point()
                                if polys[j].contains(pt) or polys[j].covers(pt):
                                    depth[i] = depth[j] + 1
                                    break
                            except Exception:
                                continue
                        else:
                            # No parent found — treat as outer despite CW winding
                            depth[i] = 0
                        continue
                except Exception:
                    pass
                # CCW = outer — depth stays 0 unless it is inside another CCW outer
                # (i.e. an island inside a hole — depth 2). Check containment.
                for j in range(i - 1, -1, -1):
                    try:
                        pt = polys[i].representative_point()
                        if polys[j].contains(pt) or polys[j].covers(pt):
                            depth[i] = depth[j] + 1
                            break
                    except Exception:
                        continue

            # Even depth = outer filled region, odd depth = hole
            outers = [polys[i] for i in range(n) if depth[i] % 2 == 0]
            holes  = [polys[i] for i in range(n) if depth[i] % 2 == 1]

            if not outers:
                return None

            try:
                outer_union = _UU(outers) if len(outers) > 1 else outers[0]
                if not outer_union.is_valid:
                    outer_union = outer_union.buffer(0)
                if holes:
                    hole_union = _UU(holes) if len(holes) > 1 else holes[0]
                    if not hole_union.is_valid:
                        hole_union = hole_union.buffer(0)
                    result = outer_union.difference(hole_union)
                else:
                    result = outer_union
                if not result.is_valid:
                    result = result.buffer(0)
                return result
            except Exception:
                return None

        # Build cumulative geometry per contourf band index
        cumulative = {}
        for si in range(len(active_edges) - 1):
            if si >= len(cf.allsegs) or not cf.allsegs[si]:
                continue
            geom = _segs_to_shapely(cf.allsegs[si])
            if geom is not None and not geom.is_empty:
                cumulative[si] = geom

        # Subtract lower cumulative area from each band to get the ring for that band
        sorted_sis = sorted(cumulative.keys())

        for idx, si in enumerate(sorted_sis):
            lo_edge = active_edges[si]
            hi_edge = active_edges[si + 1]
            interval_mid = (lo_edge + hi_edge) / 2.0

            col = None
            for blo, bhi, bcol in colored_bands:
                if blo <= interval_mid <= bhi:
                    col = bcol
                    break
            if col is None:
                continue

            ring = cumulative[si]
            if idx > 0:
                prev_si = sorted_sis[idx - 1]
                try:
                    ring = ring.difference(cumulative[prev_si])
                    if not ring.is_valid:
                        ring = ring.buffer(0)
                except Exception:
                    pass



            if ring is None or ring.is_empty:
                continue

            geoms = list(ring.geoms) if ring.geom_type in ('MultiPolygon', 'GeometryCollection') else [ring]
            for poly in geoms:
                if poly.is_empty or poly.geom_type != 'Polygon':
                    continue
                outer = [[float(v[1]), float(v[0])] for v in poly.exterior.coords]
                holes_out = [
                    [[float(v[1]), float(v[0])] for v in interior.coords]
                    for interior in poly.interiors
                ]
                all_fills.append({
                    'color':  col,
                    'coords': outer,
                    'holes':  holes_out,
                })

        # Rebuild white fills by subtracting all colored fills from them
        _colored_polys   = []
        _colored_entries = []
        _white_entries   = []

        for _f in all_fills:
            if _f['color'] != '#ffffff':
                try:
                    _p = _SP([[c[1], c[0]] for c in _f['coords']])
                    if not _p.is_valid:
                        _p = _p.buffer(0)
                    if _p.is_valid and not _p.is_empty:
                        _colored_polys.append(_p)
                except Exception:
                    pass
                _colored_entries.append(_f)
            else:
                _white_entries.append(_f)

        if _colored_polys:
            _all_colored = _UU(_colored_polys)
            _new_white   = []
            for _f in _white_entries:
                try:
                    _wp = _SP([[c[1], c[0]] for c in _f['coords']])
                    if not _wp.is_valid:
                        _wp = _wp.buffer(0)
                    _wp = _wp.difference(_all_colored)
                    if _wp.is_empty:
                        continue
                    _geoms = list(_wp.geoms) if _wp.geom_type == 'MultiPolygon' else [_wp]
                    for _wg in _geoms:
                        if _wg.is_empty or _wg.geom_type != 'Polygon':
                            continue
                        _outer = [[float(v[1]), float(v[0])] for v in _wg.exterior.coords]
                        _holes = [
                            [[float(v[1]), float(v[0])] for v in _i.coords]
                            for _i in _wg.interiors
                        ]
                        _new_white.append({'color': '#ffffff', 'coords': _outer, 'holes': _holes})
                except Exception:
                    _new_white.append(_f)
            all_fills = _new_white + _colored_entries
        else:
            all_fills = _white_entries + _colored_entries

    except Exception as e:
        print(f'    ⚠ band fill error: {e}')

    plt.close(fig)
    return all_fills


print("====== How temp band get filled ======")

# @title  BLOCK 5C · Temperature Band Fills
# ══════════════════════════════════════════════════════════════════════════
#  Depends on:
#    - _ts_ua         (from Cell 1A)
#    - _tgrid_cache   (from Cell 1A)
#  Exports:
#    - _ts_ua_json_str  (ready for Cell 9)
#
#  Re-run this cell alone to tweak band fill logic without
#  re-running the heavy gridding / H&L detection in Cell 1A.
# ══════════════════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import json as _json_ua
import time






# ══════════════════════════════════════════════════════════════════════════
#  PATCH FILLS INTO _ts_ua FROM CACHED GRIDS
# ══════════════════════════════════════════════════════════════════════════

_t0 = time.time()

if '_tgrid_cache' not in globals() or not _tgrid_cache:
    print('⚠ _tgrid_cache is empty — run Cell 1A first.')
else:
    for _cache_key, (tgrid, lv_t, ltv_t, _bands_for_lvl) in _tgrid_cache.items():
        # cache key format: "{plvl}_{datestr}_{hh}"  e.g. "850_20260521_12"
        _plvl_str = _cache_key.split('_')[0]
        _ts_key   = '_'.join(_cache_key.split('_')[1:])   # "20260521_12"
        _fills = _build_temp_band_fills(tgrid, lv_t, ltv_t, _bands_for_lvl, 0.0)
        _ts_ua[_ts_key]['levels'][_plvl_str]['temp_band_fills'] = _fills
        print(f'  {_cache_key}: {len(_fills)} fill polygons')

# Reduce JSON size — strip empty hole arrays and unused levels
import json as _json_ua

for _k in _ts_ua:
    for _lvl in ['700', '250']:
        lvl_data = _ts_ua[_k].get('levels', {}).get(_lvl, {})
        lvl_data['temp_band_fills'] = []
        lvl_data['ttdp'] = []
        lvl_data['sped'] = []
    for _lvl in ['850', '500']:
        fills = _ts_ua[_k].get('levels', {}).get(_lvl, {}).get('temp_band_fills', [])
        for f in fills:
            # Normalise holes — keep key present so renderer never errors on missing key
            if 'holes' not in f:
                f['holes'] = []
            # Truncate coord precision to 4 decimal places
            f['coords'] = [[round(c[0], 4), round(c[1], 4)] for c in f['coords']]

_ts_ua_json_str = _json_ua.dumps(_ts_ua)
print(f'\n✅ Cell 1B complete ...')

    # Rebuild JSON for Cell 9
_ts_ua_json_str = _json_ua.dumps(_ts_ua)
print(f'\n✅ Cell 1B complete ({time.time() - _t0:.1f}s) — '
      f'{len(_ts_ua_json_str) // 1024} KB  →  ready for Cell 9')

# @title
# 000---- BLOCK 06 — MOCKED (convergence computation removed)
# Variables are kept as empty stubs so downstream blocks don't break.

import json as _json_conv
import pandas as pd

ua_summary_df['_vt']   = pd.to_datetime(ua_summary_df['valid_time'])
ua_summary_df['_date'] = ua_summary_df['_vt'].dt.date
ua_summary_df['_hour'] = ua_summary_df['_vt'].dt.hour

_synoptic_times = sorted(ua_summary_df[['_date','_hour']].drop_duplicates()
                          .itertuples(index=False, name=None))

_conv_sfc_by_ts   = {f"{pd.Timestamp(d).strftime('%Y%m%d')}_{int(h):02d}": [] for d, h in _synoptic_times}
_conv_850_by_hr   = {f"{pd.Timestamp(d).strftime('%Y%m%d')}_{int(h):02d}": [] for d, h in _synoptic_times}
_sfc_trough_by_ts = {f"{pd.Timestamp(d).strftime('%Y%m%d')}_{int(h):02d}": [] for d, h in _synoptic_times}

CONV_THRESHOLD    = -1e-5
CONV_COLOR        = '#cc00cc'
CONV_FILL_COLOR   = '#dd88ff'
CONV_FILL_OPACITY = 0.30
CONV_WEIGHT       = 2.5

_conv_payload = {
    'sfc':        _conv_sfc_by_ts,
    '850':        _conv_850_by_hr,
    'sfc_trough': _sfc_trough_by_ts,
    'threshold':  CONV_THRESHOLD,
}
_conv_json_str = _json_conv.dumps(_conv_payload)

print(f'✅ Block 06 mocked — {len(_synoptic_times)} time slot(s), variables ready for downstream blocks.')

# @title STATION SYMBOL DENSITY, Met symbol , Tooltips

# ══════════════════════════════════════════════════════════════════════════
#  STATION SYMBOL DENSITY  (km between symbols — increase = fewer, decrease = more)
# ══════════════════════════════════════════════════════════════════════════
SURFACE_STN_SPACING_KM = 500   # surface METAR symbols
UA_STN_SPACING_KM      = 1000   # upper-air station symbols



# ══════════════════════════════════════════════════════════════════════════
#  Met symbol Show / Hide
# ══════════════════════════════════════════════════════════════════════════
SHOW_STATION_SYMBOLS = True # True / False → hides met symbols on map AND in exports

# ══════════════════════════════════════════════════════════════════════════
#  Tooltips Show / Hide
# ══════════════════════════════════════════════════════════════════════════
SHOW_TOOLTIPS = False   # False = no hover tooltips on any layer

# @title RUNTIME
print('RDPS runtime:' + _rdps_run_dt.strftime('%Y-%m-%d %HZ'))
print('GDPS runtime:' + _gdps_run_dt.strftime('%Y-%m-%d %HZ'))

# @title UA Forecast map
# Cell 9 — UA Forecast map
# Layers: height + temp always on, 850/500 selector, time slider
# Export 850, Export 500, Export All (all timesteps both levels)






import folium
from folium import Element
import json as _json
import numpy as np
import pandas as pd
from datetime import date as _date_kh, datetime, timezone, timedelta

# ── 500 hPa key height ────────────────────────────────────────────────────
_HEIGHT_CONTROL = {
    "Jan 1":  5400, "Apr 3":  5460, "Apr 19": 5520, "May 11": 5580,
    "May 30": 5640, "Jun 27": 5700, "Jul 26": 5760, "Aug 7":  5700,
    "Aug 31": 5640, "Oct 1":  5580, "Oct 17": 5520, "Oct 29": 5460,
    "Nov 17": 5400,
}

def _parse_height_entry(label_str, ref_year=2001):
    return datetime.strptime(f"{label_str} {ref_year}", "%b %d %Y").timetuple().tm_yday

def _get_key_hgt_500(today=None):
    if today is None:
        today = _date_kh.today()
    today_doy = today.timetuple().tm_yday
    best_val, best_doy = None, -1
    for label_str, hgt in _HEIGHT_CONTROL.items():
        entry_doy = _parse_height_entry(label_str)
        if entry_doy <= today_doy and entry_doy > best_doy:
            best_doy = entry_doy
            best_val = hgt
    if best_val is None:
        best_val = list(_HEIGHT_CONTROL.values())[-1]
    return best_val

_today_kh   = _date_kh.today()
KEY_HGT_500 = _get_key_hgt_500(_today_kh)
KEY_HGT_850 = 0
KEY_HGT_700 = 0
KEY_HGT_250 = 0
print(f'  500 hPa key height: {KEY_HGT_500} m  (date: {_today_kh})')

def _decimate_stations(records, spacing_km=500):
    if not records:
        return records
    dlat = spacing_km / 111.32
    seen = {}
    out  = []
    for d in records:
        lat = d.get('lat')
        lon = d.get('lon')
        if lat is None or lon is None:
            continue
        dlon = spacing_km / (111.32 * np.cos(np.radians(lat)))
        cell = (int(lat / dlat), int(lon / dlon))
        if cell not in seen:
            seen[cell] = True
            out.append(d)
    return out

center_lat = 53.3097
center_lon = -113.5797

# ── Safeguards ────────────────────────────────────────────────────────────
if '_ts_ua_json_str' not in globals():
    print("⚠ _ts_ua_json_str missing — run Cell 7.6 first.")
    _ts_ua_json_str = _json.dumps({})
if '_ts_ua_stn_json_str' not in globals():
    print("⚠ _ts_ua_stn_json_str missing.")
    _ts_ua_stn_json_str = _json.dumps({})
if '_ts_slp_json_str' not in globals():
    print("⚠ _ts_slp_json_str missing — run Cell 7.5 first.")
    _ts_slp_json_str = _json.dumps({})
if '_ua_date_map' not in globals():
    print("⚠ _ua_date_map missing.")
    _ua_date_map = {}

# ── Build timestamp→key maps ──────────────────────────────────────────────
_metar_ts_to_key = {}
for (_date_val, _hr) in _synoptic_times:
    _date_str = pd.Timestamp(_date_val).strftime('%Y%m%d')
    _key      = f'{_date_str}_{int(_hr):02d}'
    _day      = pd.Timestamp(_date_val).day
    for _d in metar_records:
        _ts = _d.get('timestamp', '')
        if len(_ts) < 5: continue
        try:
            _rec_day  = int(_ts[0:2])
            _rec_hour = int(_ts[2:4])
        except ValueError:
            continue
        if _rec_day == _day and abs(_rec_hour - _hr) <= 3:
            _metar_ts_to_key[_ts] = _key

_ua_hour_to_key = {}
for (_date_val, _hr) in reversed(_synoptic_times):
    _date_str = pd.Timestamp(_date_val).strftime('%Y%m%d')
    _key      = f'{_date_str}_{int(_hr):02d}'
    _short    = str(int(_hr))
    if _short not in _ua_hour_to_key:
        _ua_hour_to_key[_short] = _key

_ua_date_map = {}
for (_date_val, _hr) in _synoptic_times:
    _ua_date_map[str(int(_hr))] = f'{pd.Timestamp(_date_val).strftime("%Y-%m-%d")} {int(_hr):02d}Z'

# ── Build ordered time steps for slider ───────────────────────────────────
_EDMONTON_OFFSET = timedelta(hours=-6)
_now_edmonton    = datetime.now(timezone.utc) + _EDMONTON_OFFSET
_edmonton_today  = _now_edmonton.date()

_ua_export_hour = 12
print(f"  UA export hour: {_ua_export_hour:02d}Z")

_time_steps = []
for (_date_val, _hr) in _synoptic_times:
    _date_str = pd.Timestamp(_date_val).strftime('%Y%m%d')
    _key      = f'{_date_str}_{int(_hr):02d}'
    _dt       = pd.Timestamp(_date_val).date()
    _months   = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    _dows     = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    _dow      = _dows[_dt.weekday()]
    _mon      = _months[_dt.month - 1]
    _label    = f'{_dow} {_mon} {_dt.day} {int(_hr):02d}Z'
    _time_steps.append({'key': _key, 'label': _label, 'hour': int(_hr)})

_time_steps_str      = _json.dumps(_time_steps)
_metar_ts_to_key_str = _json.dumps(_metar_ts_to_key)
_ua_hour_to_key_str  = _json.dumps(_ua_hour_to_key)

print(f"Time steps: {[s['label'] for s in _time_steps]}")

# ── Build map ─────────────────────────────────────────────────────────────
m = folium.Map(location=[center_lat, center_lon], zoom_start=5,
               tiles=None, prefer_canvas=True)
folium.TileLayer(tiles='about:blank', attr=' ', name='Blank', max_zoom=19, show=True).add_to(m)
m.get_root().html.add_child(Element(
    '<style>.leaflet-container{background:#e0f2ff!important;}</style>'
))


# ── Borders ───────────────────────────────────────────────────────────────
borders_js = (
    '<script>\n'
    '(function(){\n'
    '  function loadBorders(){\n'
    '    var keys=Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '    if(!keys.length){setTimeout(loadBorders,200);return;}\n'
    '    var MAP=window[keys[0]];\n'
    '    if(!MAP.getPane("albertaPane")){\n'
    '      MAP.createPane("albertaPane");\n'
    '      MAP.getPane("albertaPane").style.zIndex="210";\n'
    '      MAP.getPane("albertaPane").style.pointerEvents="none";\n'
    '    }\n'
    '    if(!MAP.getPane("bordersPane")){\n'
    '      MAP.createPane("bordersPane");\n'
    '      MAP.getPane("bordersPane").style.zIndex="220";\n'
    '      MAP.getPane("bordersPane").style.pointerEvents="none";\n'
    '    }\n'
    '    if(!MAP.getPane("landPane")){\n'
    '      MAP.createPane("landPane");\n'
    '      MAP.getPane("landPane").style.zIndex="205";\n'
    '      MAP.getPane("landPane").style.pointerEvents="none";\n'
    '    }\n'
    '    if(!MAP.getPane("heightPane")){\n'
    '      MAP.createPane("heightPane");\n'
    '      MAP.getPane("heightPane").style.zIndex="490";\n'
    '      MAP.getPane("heightPane").style.pointerEvents="none";\n'
    '    }\n'
    '    fetch("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_land.geojson")\n'
    '      .then(function(r){return r.json();})\n'
    '      .then(function(gj){\n'
    '        L.geoJSON(gj,{style:function(){return {color:"none",weight:0,fill:true,fillColor:"#dedede",fillOpacity:1.0};},pane:"landPane"}).addTo(MAP);\n'
    '      }).catch(function(e){console.warn("land fill load failed",e);});\n'
    '    var items=[\n'
    '      ["https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_coastline.geojson",\n'
    '       {color:"#444",weight:2.5,opacity:1.0,fill:false}],\n'
    '      ["https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_boundary_lines_land.geojson",\n'
    '       {color:"#ffffff",weight:2.2,opacity:1.0,fill:false}],\n'
    '      ["https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_1_states_provinces_lines.geojson",\n'
    '       {color:"#ffffff",weight:1.8,opacity:0.85,fill:false}],\n'
    '      ["https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_lakes.geojson",\n'
    '       {color:"#5588aa",weight:1.8,opacity:0.9,fill:false}]\n'
    '    ];\n'

    '    items.forEach(function(item){\n'
    '      fetch(item[0]).then(function(r){return r.json();}).then(function(gj){\n'
    '        L.geoJSON(gj,{style:function(){return item[1];},pane:"bordersPane"}).addTo(MAP);\n'
    '      }).catch(function(e){console.warn("border load failed",e);});\n'
    '    });\n'
    '    fetch("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_1_states_provinces.geojson")\n'
    '      .then(function(r){return r.json();})\n'
    '      .then(function(gj){\n'
    '        var ab={type:"FeatureCollection",features:gj.features.filter(function(f){return f.properties.name==="Alberta";})};\n'
    '        L.geoJSON(ab,{style:function(){return {color:"#444444",weight:2.5,opacity:1.0,fill:true,fillColor:"#ffffff",fillOpacity:1.0};},pane:"albertaPane"}).addTo(MAP);\n'
    '      }).catch(function(e){console.warn("Alberta border load failed",e);});\n'
    '  }\n'
    '  if(document.readyState==="complete"){setTimeout(loadBorders,600);}\n'
    '  else{window.addEventListener("load",function(){setTimeout(loadBorders,600);});}\n'
    '})();\n'
    '</script>'
)
m.get_root().html.add_child(Element(borders_js))

# ── KML / fire zones ──────────────────────────────────────────────────────
if 'fire_zones_html' in globals():
    m.get_root().html.add_child(Element(fire_zones_html))

# ── Force KML zone lines above grey land fill ─────────────────────────────
m.get_root().html.add_child(Element(
    '<script>\n'
    '(function(){\n'
    '  function raiseKML(){\n'
    '    var keys=Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '    if(!keys.length){setTimeout(raiseKML,300);return;}\n'
    '    var MAP=window[keys[0]];\n'
    '    if(!MAP.getPane("kmlPane")){\n'
    '      MAP.createPane("kmlPane");\n'
    '      MAP.getPane("kmlPane").style.zIndex=460;\n'
    '      MAP.getPane("kmlPane").style.pointerEvents="none";\n'
    '    }\n'
    '    MAP.eachLayer(function(layer){\n'
    '      if(layer.options && layer.options.pane === "albertaPane") return;\n'
    '      if(layer.setStyle && layer.eachLayer){\n'
    '        layer.eachLayer(function(sub){\n'
    '          if(sub.options && sub.options.pane === "albertaPane") return;\n'
    '          if(sub.setStyle){\n'
    '      if(layer.setStyle && layer.eachLayer){\n'
    '        layer.eachLayer(function(sub){\n'
    '          if(sub.setStyle){\n'
    '            sub.options.pane="kmlPane";\n'
    '            if(sub._path) sub._path.style.stroke="#555555";\n'
    '            if(sub._path) sub._path.style.strokeWidth="1.5px";\n'
    '            if(sub._path) sub._path.style.strokeDasharray="none";\n'
    '            if(sub._path) sub._path.style.fill="none";\n'
    '            sub.setStyle({color:"#555555",weight:1.5,opacity:1.0,dashArray:null,fill:false,pane:"kmlPane"});\n'
    '          }\n'
    '        });\n'
    '      } else if(layer.setStyle && layer._path){\n'
    '        layer.options.pane="kmlPane";\n'
    '        layer.setStyle({color:"#555555",weight:1.5,opacity:1.0,dashArray:null,fill:false,pane:"kmlPane"});\n'
    '      }\n'
    '    });\n'
    '  }\n'
    '  if(document.readyState==="complete"){setTimeout(raiseKML,900);}\n'
    '  else{window.addEventListener("load",function(){setTimeout(raiseKML,900);});}\n'
    '})();\n'
    '</script>'
))

# ── Fullscreen button ─────────────────────────────────────────────────────
fullscreen_html = (
    '<style>\n'
    '#syn-fs-btn{\n'
    '  position:fixed;top:10px;left:10px;z-index:10001;\n'
    '  background:rgba(255,255,255,0.96);border:1px solid #aaa;border-radius:6px;\n'
    '  padding:5px 10px;font-family:Courier New,monospace;font-size:12px;\n'
    '  box-shadow:0 2px 8px rgba(0,0,0,0.15);cursor:pointer;color:#1a3a6a;\n'
    '}\n'
    '#syn-fs-btn:hover{background:#e8f0fe;}\n'
    '</style>\n'
    '<button id="syn-fs-btn" onclick="synToggleFS()">&#x26F6; Fullscreen</button>\n'
    '<script>\n'
    'var _synFS=false,_synMapEl=null,_synOrigStyle="";\n'
    'function synToggleFS(){\n'
    '  var btn=document.getElementById("syn-fs-btn");\n'
    '  var keys=Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '  if(!keys.length)return;\n'
    '  var MAP=window[keys[0]];\n'
    '  if(!_synMapEl){_synMapEl=document.getElementById(keys[0])||document.querySelector(".leaflet-container");}\n'
    '  if(!_synMapEl)return;\n'
    '  _synFS=!_synFS;\n'
    '  if(_synFS){\n'
    '    _synOrigStyle=_synMapEl.getAttribute("style")||"";\n'
    '    _synMapEl.setAttribute("style","position:fixed!important;top:0;left:0;width:100vw!important;height:100vh!important;z-index:9999!important;margin:0!important;");\n'
    '    btn.innerHTML="&#x274C; Exit Fullscreen";\n'
    '  } else {\n'
    '    _synMapEl.setAttribute("style",_synOrigStyle);\n'
    '    btn.innerHTML="&#x26F6; Fullscreen";\n'
    '  }\n'
    '  setTimeout(function(){MAP.invalidateSize();},100);\n'
    '}\n'
    '</script>\n'
)
m.get_root().html.add_child(Element(fullscreen_html))

# ── Build per-timestamp surface station data ──────────────────────────────
import json as _json2
_ts_all = sorted(set(d['timestamp'] for d in metar_records if d['timestamp']))

for _mr in metar_records:
    _mr.setdefault('tendency', None)
    _mr.setdefault('pressure_change', None)

_ts_data = {}
for _ts in _ts_all:
    _entries = []
    _display_set = {id(d) for d in _decimate_stations(
        [d for d in metar_records if d['timestamp'] == _ts], spacing_km=SURFACE_STN_SPACING_KM)}

    for _d in metar_records:
        if _d['timestamp'] != _ts: continue
        if id(_d) not in _display_set: continue
        _fc  = {'VFR':'green','MVFR':'steelblue','IFR':'crimson','LIFR':'red'}.get(_d['flt_cat'],'#888')
        _wg  = f' G{_d["wind_gust"]}' if _d.get('wind_gust') else ''
        _tend_raw = _d.get('tendency')
        _pc_raw   = _d.get('pressure_change')
        _TEND_SYM   = {'rising':'/','falling':'\\','steady':'—','rising_falling':'∧','falling_rising':'V','rising_steady':'⌐','falling_steady':'∟'}
        _TEND_LABEL = {'rising':'Rising','falling':'Falling','steady':'Steady','rising_falling':'Rising then falling','falling_rising':'Falling then rising','rising_steady':'Rising then steady','falling_steady':'Falling then steady'}
        if _tend_raw:
            _sym    = _TEND_SYM.get(_tend_raw, '?')
            _lbl    = _TEND_LABEL.get(_tend_raw, _tend_raw)
            _pc_str = ''
            if _pc_raw is not None and _tend_raw != 'steady':
                _sign   = '+' if _pc_raw > 0 else ''
                _pc_str = f' ({_sign}{_pc_raw/10:.1f} hPa)'
            _tend_html = (f'<span style="font-weight:bold;font-size:13px;font-family:Courier New,monospace">{_sym}</span> {_lbl}{_pc_str}')
        else:
            _tend_html = '<span style="color:#aaa">insufficient history</span>'
        _pop = (f'<div style="font-family:monospace;font-size:12px;min-width:200px">'
                f'<b style="font-size:14px;color:#1a4a8a">{_d["icao"]}</b> '
                f'<span style="color:{_fc};font-weight:bold">{_d["flt_cat"]}</span><br>'
                f'<span style="color:#888;font-size:10px">{_d["name"]}</span>'
                f'<hr style="margin:4px 0">'
                f'Temp/Dew: <b>{_d["temp"]}C / {_d["dew"]}C</b><br>'
                f'Wind: <b>{_d["wind_dir"]}/{_d["wind_spd"]}kt{_wg}</b><br>'
                f'Vis: <b>{_d["vis"]} SM</b> Wx: <b>{_d["weather"] or "NIL"}</b><br>'
                f'SLP: <b>{_d["slp"]} hPa</b> RH: <b>{_d["rh"]}%</b><br>'
                f'Tendency: {_tend_html}<br>'
                f'Cloud: <b>' + ' '.join(c['raw'] for c in _d['clouds']) + '</b><br>'
                + f'<a href="https://aviationweather.gov/api/data/metar?ids={_d["icao"]}&hours=24&taf=1" '
                f'target="_blank" style="font-size:10px;color:#1a4a8a;">METAR+TAF: {_d["icao"]} ↗</a></div>')
        _svg_str, _sw, _sh = station_model_svg(
            {**_d, 'is_surface': True, 'slp_label': '', 'lowest_sig': None}, S=34)
        _ttd_val = round(_d['temp'] - _d['dew'], 1) if _d.get('temp') is not None and _d.get('dew') is not None else None
        _pc_hpa  = (_d.get('pressure_change') or 0) / 10.0
        if   _pc_hpa <= -3: _tend_color = '#8B0000'
        elif _pc_hpa <= -2: _tend_color = '#cc0000'
        elif _pc_hpa <= -1: _tend_color = '#ff6666'
        elif _pc_hpa >=  3: _tend_color = '#00008B'
        elif _pc_hpa >=  2: _tend_color = '#1a4a8a'
        elif _pc_hpa >=  1: _tend_color = '#66aaff'
        else:                _tend_color = None
        _entries.append({
            'lat': _d['lat'], 'lon': _d['lon'], 'popup': _pop,
            'tip': f'{_d["icao"]} {_d["temp"]}C/{_d["dew"]}C {_d["wind_dir"]}/{_d["wind_spd"]}kt',
            'svg': _svg_str, 'svg_w': int(_sw), 'svg_h': int(_sh),
            'ttd': _ttd_val, 'tend_color': _tend_color,
        })
    _ts_data[_ts] = _entries

_ts_list_str = _json2.dumps(_ts_all)
_latest_ts   = _ts_all[-1] if _ts_all else ''

# ── Build UA station data — keyed by full date+hour e.g. "20260507_12" ───
import json as _json3
import math

_ua_stn_data = {}

# Deduplicate: if two rows share the same rounded lat/lon and valid_time,
# keep only one (prefer RDPS/GDPS model rows over any legacy entries)
_ua_dedup = ua_summary_df.copy()
_ua_dedup['_rlat'] = _ua_dedup['lat'].round(1)
_ua_dedup['_rlon'] = _ua_dedup['lon'].round(1)
# Sort so RDPS/GDPS rows come first, then drop duplicates keeping first
_ua_dedup = (_ua_dedup
    .sort_values('icao', key=lambda s: s.str.startswith(('RDPS','GDPS')).astype(int), ascending=False)
    .drop_duplicates(subset=['_rlat', '_rlon', 'valid_time'])
    .drop(columns=['_rlat', '_rlon'])
    .reset_index(drop=True))

for (date_val, hr), _grp in _ua_dedup.groupby(
        [_ua_dedup['valid_time'].str[:10], _ua_dedup['hour']], sort=True):
    _date_str = pd.Timestamp(date_val).strftime('%Y%m%d')
    _key      = f'{_date_str}_{int(hr):02d}'
    _stns     = []

    _PINNED_COORDS = [
        (54.00, -114.00),   # WSE; Edmonton Stony Plain  (was 53.55, -114.10)
        (60.00, -112.00),   # YSM; Fort Smith            (was 60.02, -111.95)
        (58.00, -122.00),   # YYE; Fort Nelson           (was 58.83, -122.58)
        (54.00, -122.00),   # ZXS; Prince George         (was 53.88, -122.68)
        (60.00, -136.00),   # YXY; Whitehorse            (was 60.72, -135.07)
        (66.00, -126.00),   # YVQ; Norman Wells          (was 65.28, -126.80)
    ]
    # For each pinned coord, find the single closest row in _grp
    _pinned_rows_list = []
    for (la, lo) in _PINNED_COORDS:
        _dists = (((_grp['lat'] - la)**2 + (_grp['lon'] - lo)**2))
        if _dists.empty:
            continue
        _closest_idx = _dists.idxmin()
        _closest = _grp.loc[_closest_idx]
        # Only accept if within 1.5° (avoid matching a completely wrong region)
        if _dists[_closest_idx] < 1.5**2:
            _pinned_rows_list.append(_closest)
    _grp = pd.DataFrame(_pinned_rows_list).drop_duplicates(subset=['lat', 'lon']) if _pinned_rows_list else pd.DataFrame(columns=_grp.columns)
    for _, _r in _grp.iterrows():
        def _fmt(v, dec=1):
            return f'{v:.{dec}f}' if v is not None and not (isinstance(v, float) and math.isnan(v)) else '—'
        def _fmti(v):
            return f'{int(round(v))}' if v is not None and not (isinstance(v, float) and math.isnan(v)) else '—'

        _pop = (f'<div style="font-family:monospace;font-size:11px;min-width:240px">'
                f'<b style="font-size:13px;color:#cc6600">{_r["icao"]}</b> '
                f'<span style="color:#888;font-size:10px">{_r["stn_name"]}</span><br>'
                f'<hr style="margin:4px 0">')
        for _lvl in [850, 700, 500, 250]:
            _h   = _fmti(_r.get(f'HGHT_{_lvl}'))
            _t   = _fmt(_r.get(f'TEMP_{_lvl}'))
            _td  = _fmt(_r.get(f'DWPT_{_lvl}'))
            _tv, _tdv = _r.get(f'TEMP_{_lvl}'), _r.get(f'DWPT_{_lvl}')
            _ttd = (f'{_tv - _tdv:.1f}'
                    if _tv is not None and _tdv is not None
                    and not (isinstance(_tv, float) and math.isnan(_tv))
                    and not (isinstance(_tdv, float) and math.isnan(_tdv)) else '—')
            _wd  = _fmti(_r.get(f'DRCT_{_lvl}'))
            _ws  = _fmt(_r.get(f'SPED_{_lvl}'))
            _pop += (f'<b style="color:#cc6600">{_lvl} hPa</b> '
                     f'Hgt:<b>{_h}m</b> T:<b>{_t}°C</b> '
                     f'Td:<b>{_td}°C</b> T-Td:<b>{_ttd}°C</b> '
                     f'Wnd:<b>{_wd}/{_ws}kt</b><br>')

        _t5 = _r.get('TEMP_500')
        _t7 = _r.get('TEMP_700')
        _instab_str, _instab_cat = '—', ''
        if (_t5 is not None and _t7 is not None
                and not (isinstance(_t5, float) and math.isnan(_t5))
                and not (isinstance(_t7, float) and math.isnan(_t7))):
            _tdiff = _t7 - _t5
            _instab_str = f'{_tdiff:.1f}'
            if _tdiff >= 18:
                _instab_cat = ' <span style="color:#cc2200;font-weight:bold">CB</span>'
            elif _tdiff >= 16:
                _instab_cat = ' <span style="color:#cc5500;font-weight:bold">TCU</span>'
        _pop += (f'<hr style="margin:4px 0">T700-500: <b>{_instab_str}°C</b>{_instab_cat}<br>'
                 f'</div>')

        _level_svgs = {}
        for _lvl in [850, 700, 500, 250]:
            _lt  = _r.get(f'TEMP_{_lvl}')
            _ltd = _r.get(f'DWPT_{_lvl}')
            _lwd = _r.get(f'DRCT_{_lvl}')
            _lws = _r.get(f'SPED_{_lvl}')
            _lh  = _r.get(f'HGHT_{_lvl}')
            _lttd = None
            if (_lt is not None and _ltd is not None
                    and not (isinstance(_lt, float) and math.isnan(_lt))
                    and not (isinstance(_ltd, float) and math.isnan(_ltd))):
                _lttd = round(_lt - _ltd, 1)
            _lh_label = ''
            if _lh is not None and not (isinstance(_lh, float) and math.isnan(_lh)):
                _lh_label = str(int(round(_lh / 10)))[1:]
            _lws_kt = None
            if _lws is not None and not (isinstance(_lws, float) and math.isnan(_lws)):
                _lws_kt = _lws * 1
            _ua_d = {
                'icao': {
                    (54.00, -114.00): 'WSE',
                    (60.00, -112.00): 'YSM',
                    (58.00, -122.00): 'YYE',
                    (54.00, -122.00): 'ZXS',
                    (60.00, -136.00): 'YXY',
                    (66.00, -126.00): 'YVQ',
                }.get((round(float(_r['lat']), 2), round(float(_r['lon']), 2)), '   '),
                'temp': round(_lt, 1) if _lt is not None and not (isinstance(_lt, float) and math.isnan(_lt)) else None,
                'dew':  round(_lttd, 1) if _lttd is not None else None,
                'wind_dir': int(_lwd) if _lwd is not None and not (isinstance(_lwd, float) and math.isnan(_lwd)) else None,
                'wind_spd': _lws_kt, 'wind_gust': 0,
                'vis': None, 'weather': '', 'slp_label': _lh_label,
                'oktas': 8, 'has_sky_obs': True, 'clouds': [], 'lowest_sig': None,
                'ceiling': 99999, 'flt_cat': 'VFR',
                'lat': 0, 'lon': 0, 'timestamp': '', 'rh': 0,
                'tendency': None, 'pressure_change': None, 'is_surface': False,
            }
            _svg_str, _sw, _sh = station_model_svg(_ua_d, S=34)
            _level_svgs[str(_lvl)] = {'svg': _svg_str, 'w': int(_sw), 'h': int(_sh)}

        _stns.append({
            'lat':   float(_r['lat']),
            'lon':   float(_r['lon']),
            'icao':  str(_r['icao']),
            'name':  str(_r['stn_name']),
            'popup': _pop,
            'tip':   f'{_r["icao"]} | 850:{_fmt(_r.get("TEMP_850"))}°C 500:{_fmt(_r.get("TEMP_500"))}°C',
            'svgs':  _level_svgs,
        })

    _ua_stn_data[_key] = _stns
    print(f'  UA stn key {_key!r}: {len(_stns)} stations')

if not SHOW_STATION_SYMBOLS:
    _ua_stn_data = {}
_ts_ua_stn_json_str = _json3.dumps(_ua_stn_data)
print(f'\n✓ _ts_ua_stn_json_str keys: {sorted(_ua_stn_data.keys())}')

folium.LayerControl(collapsed=False).add_to(m)

# ═══════════════════════════════════════════════════════════════════════════
#  CONTROL BAR  — level (850/500) + time slider
# ═══════════════════════════════════════════════════════════════════════════
_bar_html = '''
<style>
#syn-bar {
  position: fixed;
  bottom: 0; left: 0; right: 0;
  z-index: 10000;
  background: #1a1a2e;
  border-top: 2px solid #4a7fc1;
  padding: 8px 16px;
  display: flex;
  align-items: center;
  gap: 14px;
  font-family: "Courier New", monospace;
  font-size: 11px;
  color: #e0e0e0;
  box-shadow: 0 -3px 12px rgba(0,0,0,0.5);
  min-height: 52px;
}
#syn-bar .bar-label {
  font-size: 8px; color: #8888aa; font-weight: bold;
  text-transform: uppercase; letter-spacing: 0.5px;
  white-space: nowrap;
}
#syn-bar .bar-section {
  display: flex; align-items: center; gap: 6px;
  border-right: 1px solid #3a3a5a;
  padding-right: 14px;
  white-space: nowrap;
}
#syn-bar .bar-section:last-child { border-right: none; }
.syn-lvl-btn {
  font-size: 12px; padding: 4px 14px;
  cursor: pointer;
  border: 1px solid #3a4a6a;
  border-radius: 4px;
  background: #2a2a4a;
  color: #c0c8e0;
  font-family: "Courier New", monospace;
  font-weight: bold;
  transition: background 0.15s;
}
.syn-lvl-btn:hover { background: #3a4a7a; }
.syn-lvl-btn.active { background: #4a7fc1; color: #fff; border-color: #6a9fe1; }
.syn-exp-btn {
  font-size: 11px; padding: 4px 12px;
  cursor: pointer;
  border: 1px solid #4a7fc1;
  border-radius: 4px;
  background: #2a3a5a;
  color: #c0d0ff;
  font-family: "Courier New", monospace;
  font-weight: bold;
}
.syn-exp-btn:hover { background: #4a7fc1; color: #fff; }
.syn-exp-btn.export-all { border-color: #cc8800; color: #ffcc66; background: #3a2a00; }
.syn-exp-btn.export-all:hover { background: #cc8800; color: #fff; }
#syn-time-slider {
  width: 320px;
  accent-color: #4a7fc1;
  cursor: pointer;
}
#syn-ts-label {
  color: #c0d0ff;
  font-size: 11px;
  min-width: 200px;
}
#syn-export-status {
  color: #ffcc66;
  font-size: 10px;
  min-width: 120px;
}
#syn-export-panel {
  position: fixed;
  top: 50px; left: 10px;
  z-index: 10001;
  background: rgba(26,26,46,0.95);
  border: 1px solid #4a7fc1;
  border-radius: 6px;
  padding: 8px 10px;
  font-family: "Courier New", monospace;
  font-size: 11px;
  color: #e0e0e0;
  box-shadow: 0 2px 10px rgba(0,0,0,0.4);
  min-width: 130px;
}
</style>

<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>

<!-- Export panel — top left -->
<div id="syn-export-panel">
  <div class="bar-label" style="margin-bottom:4px;">Export</div>
  <button class="syn-exp-btn" style="display:block;width:100%;margin-top:4px;border-color:#22aa44;color:#88ffaa;background:#003311;" onclick="synExportBothPDFs()">Export PDFs</button>
  <button class="syn-exp-btn export-all" style="display:block;width:100%;" onclick="synExportAll()">Export All</button>
  <div id="syn-export-status" style="margin-top:5px;min-height:14px;"></div>
</div>

<!-- Bottom control bar -->
<div id="syn-bar">
  <div class="bar-section">
    <span class="bar-label">Export</span>
    <button class="syn-exp-btn export-all" onclick="synExportAll()">Export All</button>
    <button class="syn-exp-btn" style="display:block;width:100%;margin-top:4px;border-color:#22aa44;color:#88ffaa;background:#003311;" onclick="synExportBothPDFs()">Export PDFs</button>

    <span id="syn-export-status"></span>
  </div>
  <div class="bar-section">
    <span class="bar-label">Level</span>
    <button class="syn-lvl-btn active" id="btn-850" onclick="synSetLevel(\'850\')">850 hPa</button>
    <button class="syn-lvl-btn"        id="btn-500" onclick="synSetLevel(\'500\')">500 hPa</button>
  </div>
  <div class="bar-section">
    <span class="bar-label">Time</span>
    <input type="range" id="syn-time-slider" min="0" value="0"
           oninput="synSliderChange(this.value)">
  </div>
  <div class="bar-section">
    <span id="syn-ts-label">—</span>
  </div>
</div>
'''

# ── JavaScript ─────────────────────────────────────────────────────────────
_js = f'''
<script>
// ── Data ──────────────────────────────────────────────────────────────────
var _SYN_TIME_STEPS  = {_time_steps_str};
var _SYN_UA_STNS     = {_ts_ua_stn_json_str};
var _SYN_UA          = {_ts_ua_json_str};
var KEY_HGT_DAM      = {{"850":{int(KEY_HGT_850/10)},"700":{int(KEY_HGT_700/10)},"500":{int(KEY_HGT_500/10)},"250":{int(KEY_HGT_250/10)}}};
var KEY_HGT_M        = {{"850":{int(KEY_HGT_850)},"700":{int(KEY_HGT_700)},"500":{int(KEY_HGT_500)},"250":{int(KEY_HGT_250)}}};

// ── State ─────────────────────────────────────────────────────────────────
var _synLevel        = "850";
var _synStepIdx      = 0;
var _synUALayer      = null;
var _synStnLayer     = null;
var _synShowStations = {'true' if SHOW_STATION_SYMBOLS else 'false'};
var _synShowTooltips = {'true' if SHOW_TOOLTIPS else 'false'};




// ── Helpers ───────────────────────────────────────────────────────────────
function _getMap() {{
  var k = Object.keys(window).filter(function(k) {{ return k.startsWith("map_"); }});
  return k.length ? window[k[0]] : null;
}}
function _btnOn(id)  {{ var b = document.getElementById(id); if (b) b.classList.add("active"); }}
function _btnOff(id) {{ var b = document.getElementById(id); if (b) b.classList.remove("active"); }}
function _setExportStatus(msg) {{
  var el = document.getElementById("syn-export-status");
  if (el) el.textContent = msg;
}}

// ── Level selector ────────────────────────────────────────────────────────
function synSetLevel(lvl) {{
  _synLevel = lvl;
  _btnOff("btn-850"); _btnOff("btn-500");
  _btnOn("btn-" + lvl);
  synRender();
}}

// ── Time slider ───────────────────────────────────────────────────────────
function synSliderChange(v) {{
  _synStepIdx = parseInt(v);
  synRender();
}}

// ── Main render ───────────────────────────────────────────────────────────
function synRender() {{
  var MAP  = _getMap(); if (!MAP) return;
  var step = _SYN_TIME_STEPS[_synStepIdx];
  if (!step) return;
  var lbl  = document.getElementById("syn-ts-label");
  if (lbl) lbl.textContent = step.label;
  synRenderUA(step.key, step.label);
}}

// ── UA render: contours + colouring + station barbs ───────────────────────
function synRenderUA(fullKey, stepLabel) {{
  var MAP = _getMap(); if (!MAP) return;
  if (_synUALayer)  {{ MAP.removeLayer(_synUALayer);  _synUALayer  = null; }}
  if (_synStnLayer) {{ MAP.removeLayer(_synStnLayer); _synStnLayer = null; }}
  if (!fullKey || !_synLevel) return;

  var uaData   = (_SYN_UA[fullKey] || {{levels:{{}}}}).levels[_synLevel] || {{}};
  _synUALayer  = L.layerGroup();

  // ── Temperature band fills ────────────────────────────────────────────
  var _tbFills = uaData.temp_band_fills || [];
  if (_tbFills.length) {{
    if (!MAP.getPane("tempbandsPane")) {{
      MAP.createPane("tempbandsPane");
      MAP.getPane("tempbandsPane").style.zIndex       = 480;
      MAP.getPane("tempbandsPane").style.pointerEvents = "none";
    }}
    _tbFills.forEach(function(poly) {{
      if (!poly.coords || poly.coords.length < 3) return;
      var outerLL = poly.coords.map(function(c) {{ return [c[0], c[1]]; }});
      var holes = (poly.holes || []).map(function(hole) {{
        return hole.map(function(c) {{ return [c[0], c[1]]; }});
      }});
      var rings = [outerLL].concat(holes);
      var fillColor = poly.color === "#ffffff" ? "#ffffff" : poly.color;
      if (poly.color === "#ffffff") return;
      L.polygon(rings, {{
        color: "none", weight: 0,
        fillColor: poly.color, fillOpacity: 0.25,
        fillRule: "evenodd",
        interactive: false, pane: "tempbandsPane"
      }}).addTo(_synUALayer);
    }});
  }}

  // ── Height contours ───────────────────────────────────────────────────
  (uaData.hght || []).forEach(function(ct) {{
    var ll    = ct.coords.map(function(c) {{ return [c[1], c[0]]; }});
    var isKey = (KEY_HGT_DAM[_synLevel] && (
      Math.round(ct.level) === KEY_HGT_DAM[_synLevel] ||
      Math.round(ct.level) === KEY_HGT_M[_synLevel]));
    var _hLine = L.polyline(ll, {{
      color:   "#000000",
      weight:  isKey ? 5.5 : 2.5,
      opacity: isKey ? 1.0 : 1.0,
      pane:    "heightPane"
    }});
    if (_synShowTooltips) _hLine.bindTooltip(_synLevel + " hPa Hgt=" + Math.round(ct.level) + "dam");
    _hLine.addTo(_synUALayer);

    // Contour label near Saskatchewan
    var hgtInterval = (_synLevel === "850") ? 30  : (_synLevel === "700") ? 60  : (_synLevel === "500") ? 60  : 120;
    var hgtAnchor   = (_synLevel === "850") ? 1140 : (_synLevel === "700") ? 2520 : (_synLevel === "500") ? 4800 : 9600;
    var hgtRem = Math.round(ct.level - hgtAnchor);
    if (hgtRem >= 0 && hgtRem % hgtInterval < 1) {{
      var _skLat = 54.0, _skLon = -106.0, _best = null, _bestDist = 1e9;
      ct.coords.forEach(function(c) {{
        var d = (c[1]-_skLat)*(c[1]-_skLat) + (c[0]-_skLon)*(c[0]-_skLon);
        if (d < _bestDist) {{ _bestDist = d; _best = c; }}
      }});
      var _lblLat = _best ? _best[1] : ct.label_lat;
      var _lblLon = _best ? _best[0] : ct.label_lon;
      L.marker([_lblLat, _lblLon], {{ icon: L.divIcon({{
        html: '<div style="font-size:14px;font-weight:bold;color:#fff;'
            + 'font-family:Courier New,monospace;background:#000000;'
            + 'padding:0 3px;line-height:1.4;text-align:center;min-width:28px;">'
            + Math.round(ct.level / 10) + '</div>',
        iconSize: [32,14], iconAnchor: [16,7], className: ""
      }}), pane: "heightPane" }}).addTo(_synUALayer);
    }}
  }});

  // ── Temperature isotherms ─────────────────────────────────────────────
  (uaData.temp || []).forEach(function(ct) {{
    var t   = ct.level;
    var col = t > 0
      ? "rgb(" + Math.round(180 + 75*Math.min(t/40, 1)) + ",0,0)"
      : t < 0
      ? "rgb(0,0," + Math.round(180 + 75*Math.min(Math.abs(t)/40, 1)) + ")"
      : "#00bb00";
    var ll     = ct.coords.map(function(c) {{ return [c[1], c[0]]; }});
    var isBold = (Math.round(t) % 10 === 0);
    var _tLine = L.polyline(ll, {{
      color: col, weight: isBold ? 1.2 : 0.8,
      opacity: isBold ? 1.0 : 0.8, dashArray: "6 4"
    }});
    if (_synShowTooltips) _tLine.bindTooltip(_synLevel + " hPa T=" + t.toFixed(1) + "°C");
    _tLine.addTo(_synUALayer);

    // Isotherm label near BC coast
    var _bcLat = 54.0, _bcLon = -130.0, _bcBest = null, _bcBestDist = 1e9;
    ct.coords.forEach(function(c) {{
      var d = (c[1]-_bcLat)*(c[1]-_bcLat) + (c[0]-_bcLon)*(c[0]-_bcLon);
      if (d < _bcBestDist) {{ _bcBestDist = d; _bcBest = c; }}
    }});
    var _bcLblLat = _bcBest ? _bcBest[1] : ct.label_lat;
    var _bcLblLon = _bcBest ? _bcBest[0] : ct.label_lon;
    var _tVal = Math.round(t);
    var _tBg  = _tVal > 0 ? '#cc0000' : _tVal < 0 ? '#0044cc' : '#008800';
    L.marker([_bcLblLat, _bcLblLon], {{ icon: L.divIcon({{
      html: '<div style="font-size:12px;font-weight:' + (isBold ? "900" : "bold") + ';'
          + 'color:#ffffff;background:transparent;'
          + 'font-family:Courier New,monospace;'
          + 'text-shadow:-2px -2px 0 ' + _tBg + ',2px -2px 0 ' + _tBg + ',-2px 2px 0 ' + _tBg + ',2px 2px 0 ' + _tBg + ','
          + '-2px 0 0 ' + _tBg + ',2px 0 0 ' + _tBg + ',0 -2px 0 ' + _tBg + ',0 2px 0 ' + _tBg + ';'
          + 'padding:0 2px;line-height:1.4;text-align:center;">'
          + _tVal + '</div>',
      iconSize: [32,16], iconAnchor: [16,8], className: ""
    }}) }}).addTo(_synUALayer);
  }});

    // ── UA H/L centres ────────────────────────────────────────────────────
  var _uaHL = (_SYN_UA[fullKey] || {{}})["hl_" + _synLevel] || [];
  _uaHL.forEach(function(c) {{
    var _shadow = "1px 1px 0 white,-1px -1px 0 white,1px -1px 0 white,-1px 1px 0 white";
    var _html   = '<div style="font-size:61px;font-weight:bold;color:#000000;'
      + 'font-family:Palatino Linotype,Palatino,serif;line-height:1;'
      + 'text-shadow:' + _shadow + ';pointer-events:none;">' + c.type + '</div>';
    var _hlMark = L.marker([c.lat, c.lon], {{
      icon: L.divIcon({{ html: _html, iconSize: [70,65], iconAnchor: [35,32], className: "" }}),
      zIndexOffset: 200
    }});
    if (_synShowTooltips) _hlMark.bindTooltip(_synLevel + " hPa " + c.type);
    _hlMark.addTo(_synUALayer);
  }});

  // ── UA W/C centres ────────────────────────────────────────────────────
  var _uaWC = (_SYN_UA[fullKey] || {{}})["wc_" + _synLevel] || [];
  _uaWC.forEach(function(c) {{
    var _isW     = c.type === "W";
    var _fgColor = _isW ? "#cc0000" : "#0033cc";
    var _bgColor = _isW ? "#ffcccc" : "#cce0ff";
    var _shadow  = "2px 2px 0 " + _bgColor
      + ",-2px -2px 0 " + _bgColor
      + ",2px -2px 0 "  + _bgColor
      + ",-2px 2px 0 "  + _bgColor
      + ",-2px 0 0 "    + _bgColor
      + ",2px 0 0 "     + _bgColor
      + ",0 -2px 0 "    + _bgColor
      + ",0 2px 0 "     + _bgColor;
    var _html = '<div style="font-size:61px;font-weight:bold;color:' + _fgColor + ';'
      + 'font-family:Palatino Linotype,Palatino,serif;line-height:1;'
      + 'text-shadow:' + _shadow + ';pointer-events:none;">' + c.type + '</div>';
    var _wcMark = L.marker([c.lat, c.lon], {{
      icon: L.divIcon({{ html: _html, iconSize: [70,65], iconAnchor: [35,32], className: "" }}),
      zIndexOffset: 190
    }});
    if (_synShowTooltips) _wcMark.bindTooltip(
      _synLevel + " hPa " + c.type + " " + c.val.toFixed(1) + "\u00b0C"
    );
    _wcMark.addTo(_synUALayer);
  }});

  _synUALayer.addTo(MAP);

  // ── Pinned station code labels ────────────────────────────────────────
  var _PINNED_LABELS = [
    {{ lat: 54.00, lon: -114.00, code: "WSE" }},
    {{ lat: 60.00, lon: -112.00, code: "YSM" }},
    {{ lat: 58.00, lon: -122.00, code: "YYE" }},
    {{ lat: 54.00, lon: -122.00, code: "ZXS" }},
    {{ lat: 60.00, lon: -136.00, code: "YXY" }},
    {{ lat: 66.00, lon: -126.00, code: "YVQ" }},
  ];

  // ── Station wind barbs ────────────────────────────────────────────────
  var stns = _SYN_UA_STNS[fullKey] || [];
  if (!stns.length) console.warn("No UA stns for key:", fullKey);
  _synStnLayer = L.layerGroup();
  stns.forEach(function(s) {{
    var svgInfo = (s.svgs || {{}})[_synLevel];
    if (!svgInfo) return;
    if (s.force850 && _synLevel !== "850") return;
    var icon = L.divIcon({{
      html:       svgInfo.svg,
      iconSize:   [svgInfo.w, svgInfo.h],
      iconAnchor: [Math.round(svgInfo.w/2), Math.round(svgInfo.h/2)],
      className:  ""
    }});
    var _sMark = L.marker([s.lat, s.lon], {{ icon: icon }})
      .bindPopup(s.popup, {{ maxWidth: 320 }});
    if (_synShowTooltips) _sMark.bindTooltip(s.tip + " | " + (stepLabel || ""));
    _sMark.addTo(_synStnLayer);
  }});
  if (_synShowStations) _synStnLayer.addTo(MAP);

}}

// ═══════════════════════════════════════════════════════════════════════════
//  EXPORT: shared capture engine
// ═══════════════════════════════════════════════════════════════════════════

function _synCapture(cfg, onDone) {{
  var MAP  = _getMap();
  var keys = Object.keys(window).filter(function(k) {{ return k.startsWith("map_"); }});
  if (!keys.length) {{ _setExportStatus("Map not found"); if (onDone) onDone(false); return; }}
  var mapEl = document.getElementById(keys[0]) || document.querySelector(".leaflet-container");
  if (!mapEl)  {{ _setExportStatus("Map el not found"); if (onDone) onDone(false); return; }}

  _synLevel = cfg.level;
  _btnOff("btn-850"); _btnOff("btn-500");
  _btnOn("btn-" + cfg.level);
  synRender();

  var hideEls = [
    mapEl.querySelector(".leaflet-control-container"),
    document.querySelector(".leaflet-control-layers"),
    document.querySelector(".leaflet-control-zoom"),
    document.querySelector(".leaflet-control-attribution"),
    document.getElementById("syn-bar"),
    document.getElementById("syn-export-panel"),
    document.getElementById("syn-fs-btn")
  ].filter(Boolean);
  var prevVis = hideEls.map(function(el) {{ return el.style.visibility; }});
  hideEls.forEach(function(el) {{ el.style.visibility = "hidden"; }});
  // Hide all Leaflet tooltips during export
  document.querySelectorAll(".leaflet-tooltip").forEach(function(t) {{ t.style.display = "none"; }});

  var origW = mapEl.style.width;
  var origH = mapEl.style.height;
  function restore() {{
    mapEl.style.width  = origW;
    mapEl.style.height = origH;
    MAP.invalidateSize();
    hideEls.forEach(function(el, i) {{ el.style.visibility = prevVis[i]; }});
    document.querySelectorAll(".leaflet-tooltip").forEach(function(t) {{ t.style.display = ""; }});
  }}

  mapEl.style.width  = cfg.targetW + "px";
  mapEl.style.height = cfg.targetH + "px";
  MAP.invalidateSize();
  MAP.setView(cfg.center, cfg.zoom, {{ animate: false }});

  setTimeout(function() {{
    html2canvas(mapEl, {{
      useCORS: true, allowTaint: true,
      scale: 2, logging: false,
      width: cfg.targetW, height: cfg.targetH
    }}).then(function(canvas) {{

      var cropH = canvas.height;
      var cropW = Math.min(Math.round(cropH * cfg.cropRatioW / cfg.cropRatioH), canvas.width);

      var BANNER_H = 90;
      var CREDIT_H = 22;

      var out = document.createElement("canvas");
      out.width  = cropW;
      out.height = cropH + BANNER_H + CREDIT_H;
      var ctx = out.getContext("2d");
      ctx.drawImage(canvas, 0, 0, cropW, cropH, 0, 0, cropW, cropH);

      var step   = _SYN_TIME_STEPS[_synStepIdx] || {{}};
      var _key   = step.key || "";
      var _dYear = parseInt(_key.substring(0,4), 10);
      var _dMon  = parseInt(_key.substring(4,6), 10) - 1;
      var _dDay  = parseInt(_key.substring(6,8), 10);
      var _dH    = step.hour || 0;

      var _MONTHS_L = ["January","February","March","April","May","June",
                       "July","August","September","October","November","December"];
      var _DOWS_L   = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"];

      var _utcDate  = new Date(Date.UTC(_dYear, _dMon, _dDay, _dH, 0, 0));
      var _yr       = _utcDate.getUTCFullYear();
      var _mar1     = new Date(Date.UTC(_yr, 2, 1));
      var _dstStart = new Date(Date.UTC(_yr, 2, 1 + (7 - _mar1.getUTCDay()) % 7 + 7));
      _dstStart.setUTCHours(8);
      var _nov1     = new Date(Date.UTC(_yr, 10, 1));
      var _dstEnd   = new Date(Date.UTC(_yr, 10, 1 + (7 - _nov1.getUTCDay()) % 7));
      _dstEnd.setUTCHours(7);

      var _offsetH   = (_utcDate >= _dstStart && _utcDate < _dstEnd) ? -6 : -7;
      var _tzLabel   = _offsetH === -6 ? "MDT" : "MST";
      var _localDate = new Date(_utcDate.getTime() + _offsetH * 3600000);
      var _lMon      = _localDate.getUTCMonth();
      var _lDay      = _localDate.getUTCDate();
      var _lYear     = _localDate.getUTCFullYear();
      var _lH        = _localDate.getUTCHours();
      var _ampm      = _lH < 12 ? "AM" : "PM";
      var _hr12      = _lH === 0 ? 12 : (_lH > 12 ? _lH - 12 : _lH);

      var _tsStr = _DOWS_L[_localDate.getUTCDay()]
                 + " " + _MONTHS_L[_lMon]
                 + " " + _lDay + ", " + _lYear
                 + " - " + _hr12 + " " + _ampm + " " + _tzLabel;

      var blackW = Math.round(cropW * 0.28);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, cropH, cropW, BANNER_H);
      ctx.fillStyle = "#111111";
      ctx.fillRect(0, cropH, blackW, BANNER_H);

      ctx.font         = "bold 42px Arial, sans-serif";
      ctx.fillStyle    = "#ffffff";
      ctx.textAlign    = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(_tsStr, blackW / 2, cropH + BANNER_H / 2);

      var _lvlTitle = cfg.level === "500"
        ? "500 hPa Heights and Isotherms"
        : "850 hPa Heights and Isotherms";

      ctx.font         = "bold 42px Arial, sans-serif";
      ctx.fillStyle    = "#111111";
      ctx.textAlign    = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(_lvlTitle, blackW + 24, cropH + BANNER_H / 2);

      ctx.font         = "bold 38px Arial, sans-serif";
      ctx.fillStyle    = "#333333";
      ctx.textAlign    = "right";
      ctx.textBaseline = "middle";
      ctx.fillText("AWCC Weather Office", cropW - 24, cropH + BANNER_H / 2);

      var creditY = cropH + BANNER_H;
      ctx.fillStyle = "#f0f0f0";
      ctx.fillRect(0, creditY, cropW, CREDIT_H);

      ctx.font         = "13px Arial, sans-serif";
      ctx.fillStyle    = "#777777";
      ctx.textAlign    = "right";
      ctx.textBaseline = "middle";
      ctx.fillText("Based on data issued by Meteorological Service of Canada",
                   cropW - 14, creditY + CREDIT_H / 2);

      var _exportNow = new Date();
      var _expStr = "Exported "
        + String(_exportNow.getUTCFullYear())
        + "-" + String(_exportNow.getUTCMonth()+1).padStart(2,"0")
        + "-" + String(_exportNow.getUTCDate()).padStart(2,"0")
        + " " + String(_exportNow.getUTCHours()).padStart(2,"0")
        + ":" + String(_exportNow.getUTCMinutes()).padStart(2,"0") + "Z";
      ctx.font      = "13px Arial, sans-serif";
      ctx.fillStyle = "#555555";
      ctx.textAlign = "left";
      ctx.fillText(_expStr, 14, creditY + CREDIT_H / 2);


      var _thisDate = _key.substring(0,8);
      var _seenDates = [];
      for (var _di = 0; _di < _SYN_TIME_STEPS.length; _di++) {{
        var _dkey = (_SYN_TIME_STEPS[_di].key || "").substring(0,8);
        if (_seenDates.indexOf(_dkey) === -1) _seenDates.push(_dkey);
        if (_dkey === _thisDate) break;
      }}
      var _dayNum  = String(_seenDates.length).padStart(2,"0");
      var _lvlPfx  = cfg.level === "500" ? "500mb" : "850mb";
      var _localDD = String(_lDay).padStart(2,"0");
      var fname = _lvlPfx + "Day" + _seenDates.length + ".png";

      var link   = document.createElement("a");
      link.download = fname;
      link.href     = out.toDataURL("image/png");
      link.click();

      restore();
      if (onDone) onDone(true);

    }}).catch(function(e) {{
      console.error("html2canvas failed:", e);
      restore();
      _setExportStatus("✗ Capture error: " + e.message);
      if (onDone) onDone(false);
    }});
  }}, 600);
}}

// ── Export single level at current timestep ───────────────────────────────
function synSave850() {{
  _setExportStatus("Capturing 850mb...");
  _synCapture({{
    level: "850", center: [55, -104], zoom: 5,
    targetW: 1400, targetH: 1100, cropRatioW: 8.5, cropRatioH: 11.0
  }}, function(ok) {{
    _setExportStatus(ok ? "✓ 850mb saved!" : "✗ Failed");
    if (ok) setTimeout(function() {{ _setExportStatus(""); }}, 3000);
  }});
}}

function synSave500() {{
  _setExportStatus("Capturing 500mb...");
  _synCapture({{
    level: "500", center: [55, -118], zoom: 5,
    targetW: 1400, targetH: 1141, cropRatioW: 2944.0, cropRatioH: 2400.0
  }}, function(ok) {{
    _setExportStatus(ok ? "✓ 500mb saved!" : "✗ Failed");
    if (ok) setTimeout(function() {{ _setExportStatus(""); }}, 3000);
  }});
}}

// ── Export All: every timestep × both levels ──────────────────────────────
var _exportAllQueue   = [];
var _exportAllRunning = false;

function synExportAll() {{
  if (_exportAllRunning) {{ _setExportStatus("Already running..."); return; }}
  _exportAllQueue = [];
  var total = _SYN_TIME_STEPS.length;
  for (var i = 0; i < _SYN_TIME_STEPS.length; i++) {{
    _exportAllQueue.push({{ stepIdx: i, level: "500" }});
  }}
  for (var i = 0; i < _SYN_TIME_STEPS.length; i++) {{
    _exportAllQueue.push({{ stepIdx: i, level: "850" }});
  }}
  _exportAllRunning = true;
  _setExportStatus("Export All: 0/" + (total * 2));
  _runExportQueue(0, total * 2);
}}

function _runExportQueue(done, total) {{
  if (_exportAllQueue.length === 0) {{
    _exportAllRunning = false;
    _setExportStatus("✓ All " + total + " images saved!");
    setTimeout(function() {{ _setExportStatus(""); }}, 5000);
    return;
  }}
  var job = _exportAllQueue.shift();
  _synStepIdx = job.stepIdx;
  var slider  = document.getElementById("syn-time-slider");
  if (slider) slider.value = String(job.stepIdx);
  var step = _SYN_TIME_STEPS[job.stepIdx] || {{}};
  var lbl  = document.getElementById("syn-ts-label");
  if (lbl) lbl.textContent = step.label || "";
  synRenderUA(step.key, step.label);

  var cfg = (job.level === "850")
    ? {{ level: "850", center: [55,-104], zoom: 5, targetW: 1400, targetH: 1141, cropRatioW: 8.5,    cropRatioH: 11.0   }}
    : {{ level: "500", center: [55,-118], zoom: 5, targetW: 1400, targetH: 1141, cropRatioW: 2944.0, cropRatioH: 2400.0 }};

  _setExportStatus("Exporting " + job.level + " step " + (job.stepIdx+1) + "/" + _SYN_TIME_STEPS.length + " (" + (done+1) + "/" + total + ")");
  setTimeout(function() {{
    _synCapture(cfg, function() {{
      setTimeout(function() {{ _runExportQueue(done + 1, total); }}, 1000);
    }});
  }}, 600);
}}

// ── jsPDF multi-page PDF export ───────────────────────────────────────────
var _pdfRunning = false;

function _ensureJsPDF(cb) {{
  if (typeof window.jspdf !== "undefined") {{ cb(); return; }}
  _setExportStatus("Loading jsPDF...");
  var s = document.createElement("script");
  s.src = "https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js";
  s.onload  = function() {{ cb(); }};
  s.onerror = function() {{ _setExportStatus("✗ jsPDF load failed"); _pdfRunning = false; }};
  document.head.appendChild(s);
}}

// ── Single-level PDF builder (calls onComplete when done) ─────────────────
function _doExportPDF(level, onComplete) {{
  _pdfRunning = true;
  var total = _SYN_TIME_STEPS.length;
  var queue = [];
  for (var i = 0; i < _SYN_TIME_STEPS.length; i++) {{
    if (_SYN_TIME_STEPS[i].hour === 12) queue.push(i);
  }}
  var total = queue.length;

  var PAGE_W_MM = 279.4;
  var PAGE_H_MM = 215.9;
  var doc = new window.jspdf.jsPDF({{ orientation: "landscape", unit: "mm", format: [PAGE_W_MM, PAGE_H_MM] }});

  var cfg = (level === "850")
    ? {{ level: "850", center: [55,-118], zoom: 5, targetW: 1400, targetH: 1141, cropRatioW: 2944,    cropRatioH: 2400   }}
    : {{ level: "500", center: [55,-118], zoom: 5, targetW: 1400, targetH: 1141, cropRatioW: 2944.0, cropRatioH: 2400.0 }};

  _setExportStatus("PDF " + level + ": 0/" + total);

  function processNext(idx) {{
    _synLevel = level;
    _btnOff("btn-850"); _btnOff("btn-500");
    _btnOn("btn-" + level);
    if (idx >= queue.length) {{
      var now   = new Date();
      var fname = level + "mb_ALL_"
        + now.getUTCFullYear()
        + String(now.getUTCMonth()+1).padStart(2,"0")
        + String(now.getUTCDate()).padStart(2,"0")
        + "_" + String(now.getUTCHours()).padStart(2,"0")
        + String(now.getUTCMinutes()).padStart(2,"0") + "Z.pdf";
      doc.save(fname);
      _pdfRunning = false;
      _setExportStatus("✓ " + level + "mb PDF saved (" + total + " pages)!");
      if (onComplete) onComplete();
      return;
    }}

    var stepIdx = queue[idx];
    _synStepIdx = stepIdx;
    var slider = document.getElementById("syn-time-slider");
    if (slider) slider.value = String(stepIdx);
    var step = _SYN_TIME_STEPS[stepIdx] || {{}};
    var lbl  = document.getElementById("syn-ts-label");
    if (lbl) lbl.textContent = step.label || "";
    synRenderUA(step.key, step.label);
    _setExportStatus("PDF " + level + ": " + (idx+1) + "/" + total);

    setTimeout(function() {{
      var MAP  = _getMap();
      var keys = Object.keys(window).filter(function(k) {{ return k.startsWith("map_"); }});
      if (!keys.length) {{ processNext(idx+1); return; }}
      var mapEl = document.getElementById(keys[0]) || document.querySelector(".leaflet-container");
      if (!mapEl)  {{ processNext(idx+1); return; }}

      var hideEls = [
        mapEl.querySelector(".leaflet-control-container"),
        document.querySelector(".leaflet-control-layers"),
        document.querySelector(".leaflet-control-zoom"),
        document.querySelector(".leaflet-control-attribution"),
        document.getElementById("syn-bar"),
        document.getElementById("syn-export-panel"),
        document.getElementById("syn-fs-btn")
      ].filter(Boolean);
      var prevVis = hideEls.map(function(el) {{ return el.style.visibility; }});
      hideEls.forEach(function(el) {{ el.style.visibility = "hidden"; }});
      // Hide all Leaflet tooltips during export
      document.querySelectorAll(".leaflet-tooltip").forEach(function(t) {{ t.style.display = "none"; }});

      var origW = mapEl.style.width, origH = mapEl.style.height;
      function restore() {{
        mapEl.style.width  = origW;
        mapEl.style.height = origH;
        MAP.invalidateSize();
        hideEls.forEach(function(el, i) {{ el.style.visibility = prevVis[i]; }});
        document.querySelectorAll(".leaflet-tooltip").forEach(function(t) {{ t.style.display = ""; }});
      }}

      mapEl.style.width  = cfg.targetW + "px";
      mapEl.style.height = cfg.targetH + "px";
      MAP.invalidateSize();
      MAP.setView(cfg.center, cfg.zoom, {{ animate: false }});

      setTimeout(function() {{
        html2canvas(mapEl, {{
          useCORS: true, allowTaint: true,
          scale: 1.5, logging: false,
          width: cfg.targetW, height: cfg.targetH
        }}).then(function(canvas) {{

          var cropH    = canvas.height;
          var cropW    = Math.min(Math.round(cropH * cfg.cropRatioW / cfg.cropRatioH), canvas.width);
          var BANNER_H = 80, TITLE_H = 80, CREDIT_H = 28;

          var out    = document.createElement("canvas");
          out.width  = cropW;
          out.height = cropH + BANNER_H + TITLE_H + CREDIT_H;
          var ctx    = out.getContext("2d");
          ctx.drawImage(canvas, 0, 0, cropW, cropH, 0, 0, cropW, cropH);

          var _key   = step.key || "";
          var _dYear = parseInt(_key.substring(0,4),10);
          var _dMon  = parseInt(_key.substring(4,6),10)-1;
          var _dDay  = parseInt(_key.substring(6,8),10);
          var _dH    = step.hour || 0;
          var _ML    = ["January","February","March","April","May","June","July","August","September","October","November","December"];
          var _DL    = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"];
          var _utc   = new Date(Date.UTC(_dYear,_dMon,_dDay,_dH,0,0));
          var _yr    = _utc.getUTCFullYear();
          var _m1    = new Date(Date.UTC(_yr,2,1));
          var _ds    = new Date(Date.UTC(_yr,2,1+(7-_m1.getUTCDay())%7+7)); _ds.setUTCHours(8);
          var _n1    = new Date(Date.UTC(_yr,10,1));
          var _de    = new Date(Date.UTC(_yr,10,1+(7-_n1.getUTCDay())%7));  _de.setUTCHours(7);
          var _off   = (_utc>=_ds && _utc<_de) ? -6 : -7;
          var _loc   = new Date(_utc.getTime()+_off*3600000);
          var _lH    = _loc.getUTCHours();
          var _ampm  = _lH<12?"AM":"PM";
          var _h12   = _lH===0?12:(_lH>12?_lH-12:_lH);
          var _tsStr = _DL[_loc.getUTCDay()]+" "+_ML[_loc.getUTCMonth()]+" "+_loc.getUTCDate()+", "+_loc.getUTCFullYear()+" - "+_h12+" "+_ampm;

          var blackW = Math.round(cropW*0.33);
          ctx.fillStyle="#ffffff"; ctx.fillRect(0,cropH,cropW,BANNER_H * 1);
          ctx.fillStyle="#111111"; ctx.fillRect(0,cropH,blackW,BANNER_H * 1);
          ctx.font="bold 30px Arial,sans-serif"; ctx.fillStyle="#ffffff";
          ctx.textAlign="center"; ctx.textBaseline="middle";
          ctx.fillText(_tsStr,blackW/2,cropH+BANNER_H/2);

          var titleY=cropH+BANNER_H;
          ctx.fillStyle="#ffffff"; ctx.fillRect(0,titleY,cropW,TITLE_H);
          ctx.font="bold 32px Arial,sans-serif"; ctx.fillStyle="#111111";
          ctx.textAlign="left"; ctx.textBaseline="middle";
          ctx.fillText(cfg.level==="500"?"500 hPa Heights and Isotherms":"850 hPa Heights and Isotherms",24,titleY+TITLE_H/2);
          ctx.font="24px Arial,sans-serif"; ctx.fillStyle="#333333";
          ctx.textAlign="right";
          ctx.fillText("AWCC Weather Office",cropW-24,titleY+TITLE_H/2);

          var creditY=titleY+TITLE_H;
          ctx.fillStyle="#ffffff"; ctx.fillRect(0,creditY,cropW,CREDIT_H);
          ctx.font="12px Arial,sans-serif"; ctx.fillStyle="#555555";
          ctx.textAlign="right"; ctx.textBaseline="middle";
          ctx.fillText("Based on data issued by Meteorological Service of Canada",cropW-20,creditY+CREDIT_H/2);

          var imgData = out.toDataURL("image/jpeg", 0.88);
          if (idx > 0) doc.addPage([PAGE_W_MM, PAGE_H_MM], "landscape");
          doc.addImage(imgData, "JPEG", 0, 0, PAGE_W_MM, PAGE_H_MM);

          restore();
          setTimeout(function() {{ processNext(idx+1); }}, 800);

        }}).catch(function(e) {{
          console.error("html2canvas error:", e);
          restore();
          processNext(idx+1);
        }});
      }}, 600);
    }}, 400);
  }}

  processNext(0);
}}

// ── One button → two PDFs ─────────────────────────────────────────────────
function synExportBothPDFs() {{
  if (_pdfRunning) {{ _setExportStatus("PDF already running..."); return; }}
  _ensureJsPDF(function() {{
    _setExportStatus("Starting 850mb PDF...");
    _doExportPDF("850", function() {{
      _setExportStatus("850mb done — starting 500mb PDF...");
      setTimeout(function() {{
        _doExportPDF("500", function() {{
          _setExportStatus("✓ Both PDFs saved!");
          setTimeout(function() {{ _setExportStatus(""); }}, 5000);
        }});
      }}, 1200);
    }});
  }});
}}

// ── Init ──────────────────────────────────────────────────────────────────
function _maybeTip(marker, text) {{
  if (_synShowTooltips) marker.bindTooltip(text);
  return marker;
}}
function _synInit() {{
  var slider = document.getElementById("syn-time-slider");
  if (slider) {{
    slider.max   = String(Math.max(0, _SYN_TIME_STEPS.length - 1));
    slider.value = "0";
  }}
  synSetLevel("850");
  synRender();
}}

if (document.readyState === "complete") {{ setTimeout(_synInit, 700); }}
else {{ window.addEventListener("load", function() {{ setTimeout(_synInit, 700); }}); }}
</script>
'''

m.get_root().html.add_child(Element(_bar_html))
m.get_root().html.add_child(Element(_js))

# ── Save ──────────────────────────────────────────────────────────────────
os.makedirs('outputs', exist_ok=True)
out_path = 'outputs/synoptic_map.html'
m.save(out_path)
print(f'\n✅ Synoptic map saved → {out_path}')

"""Below is LLJ plot. Not running at the moment.

Ctrl A & Ctrl / to cancel comment out.
"""

# @title LLJ for future use'
# # @title
print('this is LLJ for future use')

# Cell 9 — Synoptic map
# Layers: height + temp always on, 850/500 selector, time slider
# Export 850, Export 500, Export All (all timesteps both levels)


###########################################################
###########################################################

##     Ctrl + / to cancel comment out

###########################################################
###########################################################




import folium
from folium import Element
import json as _json
import numpy as np
import pandas as pd
from datetime import date as _date_kh, datetime, timezone, timedelta

# ── 500 hPa key height ────────────────────────────────────────────────────
_HEIGHT_CONTROL = {
    "Jan 1":  5400, "Apr 3":  5460, "Apr 19": 5520, "May 11": 5580,
    "May 30": 5640, "Jun 27": 5700, "Jul 26": 5760, "Aug 7":  5700,
    "Aug 31": 5640, "Oct 1":  5580, "Oct 17": 5520, "Oct 29": 5460,
    "Nov 17": 5400,
}

def _parse_height_entry(label_str, ref_year=2001):
    return datetime.strptime(f"{label_str} {ref_year}", "%b %d %Y").timetuple().tm_yday

def _get_key_hgt_500(today=None):
    if today is None:
        today = _date_kh.today()
    today_doy = today.timetuple().tm_yday
    best_val, best_doy = None, -1
    for label_str, hgt in _HEIGHT_CONTROL.items():
        entry_doy = _parse_height_entry(label_str)
        if entry_doy <= today_doy and entry_doy > best_doy:
            best_doy = entry_doy
            best_val = hgt
    if best_val is None:
        best_val = list(_HEIGHT_CONTROL.values())[-1]
    return best_val

_today_kh   = _date_kh.today()
KEY_HGT_500 = _get_key_hgt_500(_today_kh)
KEY_HGT_850 = 0
KEY_HGT_700 = 0
KEY_HGT_250 = 0
print(f'  500 hPa key height: {KEY_HGT_500} m  (date: {_today_kh})')

def _decimate_stations(records, spacing_km=500):
    if not records:
        return records
    dlat = spacing_km / 111.32
    seen = {}
    out  = []
    for d in records:
        lat = d.get('lat')
        lon = d.get('lon')
        if lat is None or lon is None:
            continue
        dlon = spacing_km / (111.32 * np.cos(np.radians(lat)))
        cell = (int(lat / dlat), int(lon / dlon))
        if cell not in seen:
            seen[cell] = True
            out.append(d)
    return out

center_lat = 53.3097
center_lon = -113.5797

# ── Safeguards ────────────────────────────────────────────────────────────
if '_ts_ua_json_str' not in globals():
    print("⚠ _ts_ua_json_str missing — run Cell 7.6 first.")
    _ts_ua_json_str = _json.dumps({})
if '_ts_ua_stn_json_str' not in globals():
    print("⚠ _ts_ua_stn_json_str missing.")
    _ts_ua_stn_json_str = _json.dumps({})
if '_ts_slp_json_str' not in globals():
    print("⚠ _ts_slp_json_str missing — run Cell 7.5 first.")
    _ts_slp_json_str = _json.dumps({})
if '_ua_date_map' not in globals():
    print("⚠ _ua_date_map missing.")
    _ua_date_map = {}

# ── Build timestamp→key maps ──────────────────────────────────────────────
_metar_ts_to_key = {}
for (_date_val, _hr) in _synoptic_times:
    _date_str = pd.Timestamp(_date_val).strftime('%Y%m%d')
    _key      = f'{_date_str}_{int(_hr):02d}'
    _day      = pd.Timestamp(_date_val).day
    for _d in metar_records:
        _ts = _d.get('timestamp', '')
        if len(_ts) < 5: continue
        try:
            _rec_day  = int(_ts[0:2])
            _rec_hour = int(_ts[2:4])
        except ValueError:
            continue
        if _rec_day == _day and abs(_rec_hour - _hr) <= 3:
            _metar_ts_to_key[_ts] = _key

_ua_hour_to_key = {}
for (_date_val, _hr) in reversed(_synoptic_times):
    _date_str = pd.Timestamp(_date_val).strftime('%Y%m%d')
    _key      = f'{_date_str}_{int(_hr):02d}'
    _short    = str(int(_hr))
    if _short not in _ua_hour_to_key:
        _ua_hour_to_key[_short] = _key

_ua_date_map = {}
for (_date_val, _hr) in _synoptic_times:
    _ua_date_map[str(int(_hr))] = f'{pd.Timestamp(_date_val).strftime("%Y-%m-%d")} {int(_hr):02d}Z'

# ── Build ordered time steps for slider ───────────────────────────────────
_EDMONTON_OFFSET = timedelta(hours=-6)
_now_edmonton    = datetime.now(timezone.utc) + _EDMONTON_OFFSET
_edmonton_today  = _now_edmonton.date()

_ua_export_hour = 12
print(f"  UA export hour: {_ua_export_hour:02d}Z")

_time_steps = []
for (_date_val, _hr) in _synoptic_times:
    _date_str = pd.Timestamp(_date_val).strftime('%Y%m%d')
    _key      = f'{_date_str}_{int(_hr):02d}'
    _dt       = pd.Timestamp(_date_val).date()
    _months   = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    _dows     = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    _dow      = _dows[_dt.weekday()]
    _mon      = _months[_dt.month - 1]
    _label    = f'{_dow} {_mon} {_dt.day} {int(_hr):02d}Z'
    _time_steps.append({'key': _key, 'label': _label, 'hour': int(_hr)})

_time_steps_str      = _json.dumps(_time_steps)
_metar_ts_to_key_str = _json.dumps(_metar_ts_to_key)
_ua_hour_to_key_str  = _json.dumps(_ua_hour_to_key)

print(f"Time steps: {[s['label'] for s in _time_steps]}")

# ── Build map ─────────────────────────────────────────────────────────────
m = folium.Map(location=[center_lat, center_lon], zoom_start=5,
               tiles=None, prefer_canvas=True)
folium.TileLayer(tiles='about:blank', attr=' ', name='Blank', max_zoom=19, show=True).add_to(m)
m.get_root().html.add_child(Element(
    '<style>.leaflet-container{background:#e0f2ff!important;}</style>'
))


# ── Borders ───────────────────────────────────────────────────────────────
borders_js = (
    '<script>\n'
    '(function(){\n'
    '  function loadBorders(){\n'
    '    var keys=Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '    if(!keys.length){setTimeout(loadBorders,200);return;}\n'
    '    var MAP=window[keys[0]];\n'
    '    if(!MAP.getPane("albertaPane")){\n'
    '      MAP.createPane("albertaPane");\n'
    '      MAP.getPane("albertaPane").style.zIndex="210";\n'
    '      MAP.getPane("albertaPane").style.pointerEvents="none";\n'
    '    }\n'
    '    if(!MAP.getPane("bordersPane")){\n'
    '      MAP.createPane("bordersPane");\n'
    '      MAP.getPane("bordersPane").style.zIndex="220";\n'
    '      MAP.getPane("bordersPane").style.pointerEvents="none";\n'
    '    }\n'
    '    if(!MAP.getPane("landPane")){\n'
    '      MAP.createPane("landPane");\n'
    '      MAP.getPane("landPane").style.zIndex="205";\n'
    '      MAP.getPane("landPane").style.pointerEvents="none";\n'
    '    }\n'
    '    if(!MAP.getPane("heightPane")){\n'
    '      MAP.createPane("heightPane");\n'
    '      MAP.getPane("heightPane").style.zIndex="490";\n'
    '      MAP.getPane("heightPane").style.pointerEvents="none";\n'
    '    }\n'
    '    fetch("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_land.geojson")\n'
    '      .then(function(r){return r.json();})\n'
    '      .then(function(gj){\n'
    '        L.geoJSON(gj,{style:function(){return {color:"none",weight:0,fill:true,fillColor:"#d4d4d4",fillOpacity:1.0};},pane:"landPane"}).addTo(MAP);\n'
    '      }).catch(function(e){console.warn("land fill load failed",e);});\n'
    '    var items=[\n'
    '      ["https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_coastline.geojson",\n'
    '       {color:"#444",weight:2.5,opacity:1.0,fill:false}],\n'
    '      ["https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_boundary_lines_land.geojson",\n'
    '       {color:"#ffffff",weight:2.2,opacity:1.0,fill:false}],\n'
    '      ["https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_1_states_provinces_lines.geojson",\n'
    '       {color:"#ffffff",weight:1.8,opacity:0.85,fill:false}],\n'
    '      ["https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_lakes.geojson",\n'
    '       {color:"#5588aa",weight:1.8,opacity:0.9,fill:false}]\n'
    '    ];\n'

    '    items.forEach(function(item){\n'
    '      fetch(item[0]).then(function(r){return r.json();}).then(function(gj){\n'
    '        L.geoJSON(gj,{style:function(){return item[1];},pane:"bordersPane"}).addTo(MAP);\n'
    '      }).catch(function(e){console.warn("border load failed",e);});\n'
    '    });\n'
    '    fetch("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_1_states_provinces.geojson")\n'
    '      .then(function(r){return r.json();})\n'
    '      .then(function(gj){\n'
    '        var ab={type:"FeatureCollection",features:gj.features.filter(function(f){return f.properties.name==="Alberta";})};\n'
    '        L.geoJSON(ab,{style:function(){return {color:"#444444",weight:2.5,opacity:1.0,fill:true,fillColor:"#ffffff",fillOpacity:1.0};},pane:"albertaPane"}).addTo(MAP);\n'
    '      }).catch(function(e){console.warn("Alberta border load failed",e);});\n'
    '  }\n'
    '  if(document.readyState==="complete"){setTimeout(loadBorders,600);}\n'
    '  else{window.addEventListener("load",function(){setTimeout(loadBorders,600);});}\n'
    '})();\n'
    '</script>'
)
m.get_root().html.add_child(Element(borders_js))

# ── KML / fire zones ──────────────────────────────────────────────────────
if 'fire_zones_html' in globals():
    m.get_root().html.add_child(Element(fire_zones_html))

# ── Fullscreen button ─────────────────────────────────────────────────────
fullscreen_html = (
    '<style>\n'
    '#syn-fs-btn{\n'
    '  position:fixed;top:10px;left:10px;z-index:10001;\n'
    '  background:rgba(255,255,255,0.96);border:1px solid #aaa;border-radius:6px;\n'
    '  padding:5px 10px;font-family:Courier New,monospace;font-size:12px;\n'
    '  box-shadow:0 2px 8px rgba(0,0,0,0.15);cursor:pointer;color:#1a3a6a;\n'
    '}\n'
    '#syn-fs-btn:hover{background:#e8f0fe;}\n'
    '</style>\n'
    '<button id="syn-fs-btn" onclick="synToggleFS()">&#x26F6; Fullscreen</button>\n'
    '<script>\n'
    'var _synFS=false,_synMapEl=null,_synOrigStyle="";\n'
    'function synToggleFS(){\n'
    '  var btn=document.getElementById("syn-fs-btn");\n'
    '  var keys=Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '  if(!keys.length)return;\n'
    '  var MAP=window[keys[0]];\n'
    '  if(!_synMapEl){_synMapEl=document.getElementById(keys[0])||document.querySelector(".leaflet-container");}\n'
    '  if(!_synMapEl)return;\n'
    '  _synFS=!_synFS;\n'
    '  if(_synFS){\n'
    '    _synOrigStyle=_synMapEl.getAttribute("style")||"";\n'
    '    _synMapEl.setAttribute("style","position:fixed!important;top:0;left:0;width:100vw!important;height:100vh!important;z-index:9999!important;margin:0!important;");\n'
    '    btn.innerHTML="&#x274C; Exit Fullscreen";\n'
    '  } else {\n'
    '    _synMapEl.setAttribute("style",_synOrigStyle);\n'
    '    btn.innerHTML="&#x26F6; Fullscreen";\n'
    '  }\n'
    '  setTimeout(function(){MAP.invalidateSize();},100);\n'
    '}\n'
    '</script>\n'
)
m.get_root().html.add_child(Element(fullscreen_html))

# ── Build per-timestamp surface station data ──────────────────────────────
import json as _json2
_ts_all = sorted(set(d['timestamp'] for d in metar_records if d['timestamp']))

for _mr in metar_records:
    _mr.setdefault('tendency', None)
    _mr.setdefault('pressure_change', None)

_ts_data = {}
for _ts in _ts_all:
    _entries = []
    _display_set = {id(d) for d in _decimate_stations(
        [d for d in metar_records if d['timestamp'] == _ts], spacing_km=SURFACE_STN_SPACING_KM)}

    for _d in metar_records:
        if _d['timestamp'] != _ts: continue
        if id(_d) not in _display_set: continue
        _fc  = {'VFR':'green','MVFR':'steelblue','IFR':'crimson','LIFR':'red'}.get(_d['flt_cat'],'#888')
        _wg  = f' G{_d["wind_gust"]}' if _d.get('wind_gust') else ''
        _tend_raw = _d.get('tendency')
        _pc_raw   = _d.get('pressure_change')
        _TEND_SYM   = {'rising':'/','falling':'\\','steady':'—','rising_falling':'∧','falling_rising':'V','rising_steady':'⌐','falling_steady':'∟'}
        _TEND_LABEL = {'rising':'Rising','falling':'Falling','steady':'Steady','rising_falling':'Rising then falling','falling_rising':'Falling then rising','rising_steady':'Rising then steady','falling_steady':'Falling then steady'}
        if _tend_raw:
            _sym    = _TEND_SYM.get(_tend_raw, '?')
            _lbl    = _TEND_LABEL.get(_tend_raw, _tend_raw)
            _pc_str = ''
            if _pc_raw is not None and _tend_raw != 'steady':
                _sign   = '+' if _pc_raw > 0 else ''
                _pc_str = f' ({_sign}{_pc_raw/10:.1f} hPa)'
            _tend_html = (f'<span style="font-weight:bold;font-size:13px;font-family:Courier New,monospace">{_sym}</span> {_lbl}{_pc_str}')
        else:
            _tend_html = '<span style="color:#aaa">insufficient history</span>'
        _pop = (f'<div style="font-family:monospace;font-size:12px;min-width:200px">'
                f'<b style="font-size:14px;color:#1a4a8a">{_d["icao"]}</b> '
                f'<span style="color:{_fc};font-weight:bold">{_d["flt_cat"]}</span><br>'
                f'<span style="color:#888;font-size:10px">{_d["name"]}</span>'
                f'<hr style="margin:4px 0">'
                f'Temp/Dew: <b>{_d["temp"]}C / {_d["dew"]}C</b><br>'
                f'Wind: <b>{_d["wind_dir"]}/{_d["wind_spd"]}kt{_wg}</b><br>'
                f'Vis: <b>{_d["vis"]} SM</b> Wx: <b>{_d["weather"] or "NIL"}</b><br>'
                f'SLP: <b>{_d["slp"]} hPa</b> RH: <b>{_d["rh"]}%</b><br>'
                f'Tendency: {_tend_html}<br>'
                f'Cloud: <b>' + ' '.join(c['raw'] for c in _d['clouds']) + '</b><br>'
                + f'<a href="https://aviationweather.gov/api/data/metar?ids={_d["icao"]}&hours=24&taf=1" '
                f'target="_blank" style="font-size:10px;color:#1a4a8a;">METAR+TAF: {_d["icao"]} ↗</a></div>')
        _svg_str, _sw, _sh = station_model_svg({**_d, 'is_surface': True, 'slp_label': '', 'lowest_sig': None}, S=34)
        _ttd_val = round(_d['temp'] - _d['dew'], 1) if _d.get('temp') is not None and _d.get('dew') is not None else None
        _pc_hpa  = (_d.get('pressure_change') or 0) / 10.0
        if   _pc_hpa <= -3: _tend_color = '#8B0000'
        elif _pc_hpa <= -2: _tend_color = '#cc0000'
        elif _pc_hpa <= -1: _tend_color = '#ff6666'
        elif _pc_hpa >=  3: _tend_color = '#00008B'
        elif _pc_hpa >=  2: _tend_color = '#1a4a8a'
        elif _pc_hpa >=  1: _tend_color = '#66aaff'
        else:                _tend_color = None
        _entries.append({
            'lat': _d['lat'], 'lon': _d['lon'], 'popup': _pop,
            'tip': f'{_d["icao"]} {_d["temp"]}C/{_d["dew"]}C {_d["wind_dir"]}/{_d["wind_spd"]}kt',
            'svg': _svg_str, 'svg_w': int(_sw), 'svg_h': int(_sh),
            'ttd': _ttd_val, 'tend_color': _tend_color,
        })
    _ts_data[_ts] = _entries

if not SHOW_STATION_SYMBOLS:
    _ts_data = {}
_ts_json_str = _json2.dumps(_ts_data)
_ts_list_str = _json2.dumps(_ts_all)
_latest_ts   = _ts_all[-1] if _ts_all else ''

# ── Build UA station data — keyed by full date+hour e.g. "20260507_12" ───
import json as _json3
import math

_ua_stn_data = {}

for (date_val, hr), _grp in ua_summary_df.groupby(
        [ua_summary_df['valid_time'].str[:10], 'hour'], sort=True):
    _date_str = pd.Timestamp(date_val).strftime('%Y%m%d')
    _key      = f'{_date_str}_{int(hr):02d}'
    _stns     = []

    _FORCE_STN_COORDS = {
        'ANA':  (53.5513, -116.5031), 'B4':   (50.9258, -115.1240),
        'BRA':  (57.1677, -117.6640), 'BROO': (50.5500, -111.8500),
        'C4':   (49.6086, -114.4514), 'C5':   (49.6356, -110.3296),
        'ECA':  (54.7916, -118.2348), 'FLA':  (58.6109, -117.1600),
        'MUA':  (57.1353, -110.8942), 'PYA':  (58.7684, -111.1061),
        'FGA':  (58.6860, -114.9947), 'S5':   (57.1443, -115.0798),
        'SDA':  (54.7283, -115.3556), 'SHA':  (52.2367, -115.1967),
        'WGM':  (49.1333, -113.8000), 'WJW':  (52.9300, -118.0300),
        'WRA':  (55.2855, -112.4789), 'WZG':  (51.1934, -115.5522),
    }

    # Find nearest RDPS grid point to each station and build synthetic rows
    import numpy as _np2
    _stn_rows = []
    for _stn_icao, (_slat, _slon) in _FORCE_STN_COORDS.items():
        _dists = (_grp['lat'] - _slat)**2 + (_grp['lon'] - _slon)**2
        if _dists.empty: continue
        _nearest = _grp.loc[_dists.idxmin()].copy()
        _nearest['icao']     = _stn_icao
        _nearest['stn_name'] = _stn_icao
        _nearest['lat']      = _slat
        _nearest['lon']      = _slon
        _stn_rows.append(_nearest)
    if not _stn_rows: continue
    _grp = pd.DataFrame(_stn_rows).reset_index(drop=True)
    _grp['_force850'] = True
    for _, _r in _grp.iterrows():
        def _fmt(v, dec=1):
            return f'{v:.{dec}f}' if v is not None and not (isinstance(v, float) and math.isnan(v)) else '—'
        def _fmti(v):
            return f'{int(round(v))}' if v is not None and not (isinstance(v, float) and math.isnan(v)) else '—'

        _pop = (f'<div style="font-family:monospace;font-size:11px;min-width:240px">'
                f'<b style="font-size:13px;color:#cc6600">{_r["icao"]}</b> '
                f'<span style="color:#888;font-size:10px">{_r["stn_name"]}</span><br>'
                f'<hr style="margin:4px 0">')
        for _lvl in [850, 700, 500, 250]:
            _h   = _fmti(_r.get(f'HGHT_{_lvl}'))
            _t   = _fmt(_r.get(f'TEMP_{_lvl}'))
            _td  = _fmt(_r.get(f'DWPT_{_lvl}'))
            _tv, _tdv = _r.get(f'TEMP_{_lvl}'), _r.get(f'DWPT_{_lvl}')
            _ttd = (f'{_tv - _tdv:.1f}'
                    if _tv is not None and _tdv is not None
                    and not (isinstance(_tv, float) and math.isnan(_tv))
                    and not (isinstance(_tdv, float) and math.isnan(_tdv)) else '—')
            _wd  = _fmti(_r.get(f'DRCT_{_lvl}'))
            _ws  = _fmt(_r.get(f'SPED_{_lvl}'))
            _pop += (f'<b style="color:#cc6600">{_lvl} hPa</b> '
                     f'Hgt:<b>{_h}m</b> T:<b>{_t}°C</b> '
                     f'Td:<b>{_td}°C</b> T-Td:<b>{_ttd}°C</b> '
                     f'Wnd:<b>{_wd}/{_ws}kt</b><br>')

        _t5 = _r.get('TEMP_500')
        _t7 = _r.get('TEMP_700')
        _instab_str, _instab_cat = '—', ''
        if (_t5 is not None and _t7 is not None
                and not (isinstance(_t5, float) and math.isnan(_t5))
                and not (isinstance(_t7, float) and math.isnan(_t7))):
            _tdiff = _t7 - _t5
            _instab_str = f'{_tdiff:.1f}'
            if _tdiff >= 18:
                _instab_cat = ' <span style="color:#cc2200;font-weight:bold">CB</span>'
            elif _tdiff >= 16:
                _instab_cat = ' <span style="color:#cc5500;font-weight:bold">TCU</span>'
        _pop += (f'<hr style="margin:4px 0">T700-500: <b>{_instab_str}°C</b>{_instab_cat}<br>'
                 f'</div>')

        _level_svgs = {}
        for _lvl in [850, 700, 500, 250]:
            _lt  = _r.get(f'TEMP_{_lvl}')
            _ltd = _r.get(f'DWPT_{_lvl}')
            _lwd = _r.get(f'DRCT_{_lvl}')
            _lws = _r.get(f'SPED_{_lvl}')
            _lh  = _r.get(f'HGHT_{_lvl}')
            _lttd = None
            if (_lt is not None and _ltd is not None
                    and not (isinstance(_lt, float) and math.isnan(_lt))
                    and not (isinstance(_ltd, float) and math.isnan(_ltd))):
                _lttd = round(_lt - _ltd, 1)
            _lh_label = ''
            if _lh is not None and not (isinstance(_lh, float) and math.isnan(_lh)):
                _lh_label = str(int(round(_lh / 10)))[1:]
            _lws_kt = None
            if _lws is not None and not (isinstance(_lws, float) and math.isnan(_lws)):
                _lws_kt = _lws * 1
            _ua_d = {
                'icao': str(_r['icao']),
                'temp': None,
                'dew':  None,
                'wind_dir': int(_lwd) if _lwd is not None and not (isinstance(_lwd, float) and math.isnan(_lwd)) else None,
                'wind_spd': _lws_kt, 'wind_gust': 0,
                'vis': None, 'weather': '', 'slp_label': None,
                'oktas': 8, 'has_sky_obs': True, 'clouds': [], 'lowest_sig': None,
                'ceiling': 99999, 'flt_cat': 'VFR',
                'lat': 0, 'lon': 0, 'timestamp': '', 'rh': 0,
                'tendency': None, 'pressure_change': None, 'is_surface': False,
            }
            _svg_str, _sw, _sh = station_model_svg(_ua_d, S=34)
            _level_svgs[str(_lvl)] = {'svg': _svg_str, 'w': int(_sw), 'h': int(_sh)}

        _stns.append({
            'lat':   float(_r['lat']),
            'lon':   float(_r['lon']),
            'icao':  str(_r['icao']),
            'force850': bool(_r.get('_force850', False)),
            'name':  str(_r['stn_name']),
            'popup': _pop,
            'tip':   f'{_r["icao"]} | 850:{_fmt(_r.get("TEMP_850"))}°C 500:{_fmt(_r.get("TEMP_500"))}°C',
            'svgs':  _level_svgs,
        })

    _ua_stn_data[_key] = _stns
    print(f'  UA stn key {_key!r}: {len(_stns)} stations')

if not SHOW_STATION_SYMBOLS:
    _ua_stn_data = {}
_ts_ua_stn_json_str = _json3.dumps(_ua_stn_data)
print(f'\n✓ _ts_ua_stn_json_str keys: {sorted(_ua_stn_data.keys())}')

folium.LayerControl(collapsed=False).add_to(m)

# ═══════════════════════════════════════════════════════════════════════════
#  CONTROL BAR  — level (850/500) + time slider
# ═══════════════════════════════════════════════════════════════════════════
_bar_html = '''
<style>
#syn-bar {
  position: fixed;
  bottom: 0; left: 0; right: 0;
  z-index: 10000;
  background: #1a1a2e;
  border-top: 2px solid #4a7fc1;
  padding: 8px 16px;
  display: flex;
  align-items: center;
  gap: 14px;
  font-family: "Courier New", monospace;
  font-size: 11px;
  color: #e0e0e0;
  box-shadow: 0 -3px 12px rgba(0,0,0,0.5);
  min-height: 52px;
}
#syn-bar .bar-label {
  font-size: 8px; color: #8888aa; font-weight: bold;
  text-transform: uppercase; letter-spacing: 0.5px;
  white-space: nowrap;
}
#syn-bar .bar-section {
  display: flex; align-items: center; gap: 6px;
  border-right: 1px solid #3a3a5a;
  padding-right: 14px;
  white-space: nowrap;
}
#syn-bar .bar-section:last-child { border-right: none; }
.syn-lvl-btn {
  font-size: 12px; padding: 4px 14px;
  cursor: pointer;
  border: 1px solid #3a4a6a;
  border-radius: 4px;
  background: #2a2a4a;
  color: #c0c8e0;
  font-family: "Courier New", monospace;
  font-weight: bold;
  transition: background 0.15s;
}
.syn-lvl-btn:hover { background: #3a4a7a; }
.syn-lvl-btn.active { background: #4a7fc1; color: #fff; border-color: #6a9fe1; }
.syn-exp-btn {
  font-size: 11px; padding: 4px 12px;
  cursor: pointer;
  border: 1px solid #4a7fc1;
  border-radius: 4px;
  background: #2a3a5a;
  color: #c0d0ff;
  font-family: "Courier New", monospace;
  font-weight: bold;
}
.syn-exp-btn:hover { background: #4a7fc1; color: #fff; }
.syn-exp-btn.export-all { border-color: #cc8800; color: #ffcc66; background: #3a2a00; }
.syn-exp-btn.export-all:hover { background: #cc8800; color: #fff; }
#syn-time-slider {
  width: 320px;
  accent-color: #4a7fc1;
  cursor: pointer;
}
#syn-ts-label {
  color: #c0d0ff;
  font-size: 11px;
  min-width: 200px;
}
#syn-export-status {
  color: #ffcc66;
  font-size: 10px;
  min-width: 120px;
}
#syn-export-panel {
  position: fixed;
  top: 50px; left: 10px;
  z-index: 10001;
  background: rgba(26,26,46,0.95);
  border: 1px solid #4a7fc1;
  border-radius: 6px;
  padding: 8px 10px;
  font-family: "Courier New", monospace;
  font-size: 11px;
  color: #e0e0e0;
  box-shadow: 0 2px 10px rgba(0,0,0,0.4);
  min-width: 130px;
}
</style>

<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>

<!-- Export panel — top left -->
<div id="syn-export-panel">
  <div class="bar-label" style="margin-bottom:4px;">Export</div>
  <button class="syn-exp-btn" style="display:block;width:100%;margin-top:4px;border-color:#cc2200;color:#ffaaaa;background:#440000;" onclick="synExportBothPDFs()">Export PDFs</button>
  <button class="syn-exp-btn export-all" style="display:block;width:100%;border-color:#9944cc;color:#ddaaff;background:#2a0044;" onclick="synExportAll()">Export 850LLJ Prog</button>
  <div id="syn-export-status" style="margin-top:5px;min-height:14px;"></div>
</div>

<!-- Bottom control bar -->
<div id="syn-bar">
  <div class="bar-section">
    <span class="bar-label">Export</span>
    <button class="syn-exp-btn export-all" style="border-color:#9944cc;color:#ddaaff;background:#2a0044;" onclick="synExportAll()">Export 850LLJ Prog</button>
    <button class="syn-exp-btn" style="border-color:#cc2200;color:#ffaaaa;background:#440000;" onclick="synExportBothPDFs()">Export 850LLJ Prog PDF</button>

    <span id="syn-export-status"></span>
  </div>
  <div class="bar-section">
    <span class="bar-label">Level</span>
    <button class="syn-lvl-btn active" id="btn-850" onclick="synSetLevel(\'850\')">850 hPa</button>
    <button class="syn-lvl-btn"        id="btn-500" onclick="synSetLevel(\'500\')">500 hPa</button>
  </div>
  <div class="bar-section">
    <span class="bar-label">Time</span>
    <input type="range" id="syn-time-slider" min="0" value="0"
           oninput="synSliderChange(this.value)">
  </div>
  <div class="bar-section">
    <span id="syn-ts-label">—</span>
  </div>
</div>
'''

# ── JavaScript ─────────────────────────────────────────────────────────────
_js = f'''
<script>
// ── Data ──────────────────────────────────────────────────────────────────
var _SYN_TIME_STEPS  = {_time_steps_str};
var _SYN_UA_STNS     = {_ts_ua_stn_json_str};
var _SYN_UA          = {_ts_ua_json_str};
var KEY_HGT_DAM      = {{"850":{int(KEY_HGT_850/10)},"700":{int(KEY_HGT_700/10)},"500":{int(KEY_HGT_500/10)},"250":{int(KEY_HGT_250/10)}}};
var KEY_HGT_M        = {{"850":{int(KEY_HGT_850)},"700":{int(KEY_HGT_700)},"500":{int(KEY_HGT_500)},"250":{int(KEY_HGT_250)}}};

// ── State ─────────────────────────────────────────────────────────────────
var _synLevel        = "850";
var _synStepIdx      = 0;
var _synUALayer      = null;
var _synStnLayer     = null;
var _synShowStations = {'true' if SHOW_STATION_SYMBOLS else 'false'};
var _synShowTooltips = {'true' if SHOW_TOOLTIPS else 'false'};

// ── Helpers ───────────────────────────────────────────────────────────────
function _getMap() {{
  var k = Object.keys(window).filter(function(k) {{ return k.startsWith("map_"); }});
  return k.length ? window[k[0]] : null;
}}
function _btnOn(id)  {{ var b = document.getElementById(id); if (b) b.classList.add("active"); }}
function _btnOff(id) {{ var b = document.getElementById(id); if (b) b.classList.remove("active"); }}
function _setExportStatus(msg) {{
  var el = document.getElementById("syn-export-status");
  if (el) el.textContent = msg;
}}

// ── Level selector ────────────────────────────────────────────────────────
function synSetLevel(lvl) {{
  _synLevel = lvl;
  _btnOff("btn-850"); _btnOff("btn-500");
  _btnOn("btn-" + lvl);
  synRender();
}}

// ── Time slider ───────────────────────────────────────────────────────────
function synSliderChange(v) {{
  _synStepIdx = parseInt(v);
  synRender();
}}

// ── Main render ───────────────────────────────────────────────────────────
function synRender() {{
  var MAP  = _getMap(); if (!MAP) return;
  var step = _SYN_TIME_STEPS[_synStepIdx];
  if (!step) return;
  var lbl  = document.getElementById("syn-ts-label");
  if (lbl) lbl.textContent = step.label;
  synRenderUA(step.key, step.label);
}}

// ── UA render: contours + colouring + station barbs ───────────────────────
function synRenderUA(fullKey, stepLabel) {{
  var MAP = _getMap(); if (!MAP) return;
  if (_synUALayer)  {{ MAP.removeLayer(_synUALayer);  _synUALayer  = null; }}
  if (_synStnLayer) {{ MAP.removeLayer(_synStnLayer); _synStnLayer = null; }}
  if (!fullKey || !_synLevel) return;

  var uaData   = (_SYN_UA[fullKey] || {{levels:{{}}}}).levels[_synLevel] || {{}};
  _synUALayer  = L.layerGroup();

  // ── Temperature band fills ────────────────────────────────────────────
  var _tbFills = uaData.temp_band_fills || [];
  if (_tbFills.length) {{
    if (!MAP.getPane("tempbandsPane")) {{
      MAP.createPane("tempbandsPane");
      MAP.getPane("tempbandsPane").style.zIndex       = 380;
      MAP.getPane("tempbandsPane").style.pointerEvents = "none";
    }}
    _tbFills.forEach(function(poly) {{
      if (!poly.coords || poly.coords.length < 3) return;
      var outerLL = poly.coords.map(function(c) {{ return [c[0], c[1]]; }});
      var holes = (poly.holes || []).map(function(hole) {{
        return hole.map(function(c) {{ return [c[0], c[1]]; }});
      }});
      var rings = [outerLL].concat(holes);
      var fillColor = poly.color === "#ffffff" ? "#ffffff" : poly.color;
      if (poly.color === "#ffffff") return;
      L.polygon(rings, {{
        color: "none", weight: 0,
        fillColor: poly.color, fillOpacity: 0.25,
        fillRule: "evenodd",
        interactive: false, pane: "tempbandsPane"
      }}).addTo(_synUALayer);
    }});
  }}

  // ── Height contours ───────────────────────────────────────────────────
  (uaData.hght || []).forEach(function(ct) {{
    var ll    = ct.coords.map(function(c) {{ return [c[1], c[0]]; }});
    var isKey = (KEY_HGT_DAM[_synLevel] && (
      Math.round(ct.level) === KEY_HGT_DAM[_synLevel] ||
      Math.round(ct.level) === KEY_HGT_M[_synLevel]));
    var _hLine = L.polyline(ll, {{
      color:   "#000000",
      weight:  isKey ? 4.5 : 1.5,
      opacity: isKey ? 1.0 : 0.85,
      pane:    "heightPane"
    }});
    if (_synShowTooltips) _hLine.bindTooltip(_synLevel + " hPa Hgt=" + Math.round(ct.level) + "dam");
    _hLine.addTo(_synUALayer);

    // Contour label near Saskatchewan
    var hgtInterval = (_synLevel === "850") ? 30  : (_synLevel === "700") ? 60  : (_synLevel === "500") ? 60  : 120;
    var hgtAnchor   = (_synLevel === "850") ? 1140 : (_synLevel === "700") ? 2520 : (_synLevel === "500") ? 4800 : 9600;
    var hgtRem = Math.round(ct.level - hgtAnchor);
    if (hgtRem >= 0 && hgtRem % hgtInterval < 1) {{
      var _skLat = 54.0, _skLon = -106.0, _best = null, _bestDist = 1e9;
      ct.coords.forEach(function(c) {{
        var d = (c[1]-_skLat)*(c[1]-_skLat) + (c[0]-_skLon)*(c[0]-_skLon);
        if (d < _bestDist) {{ _bestDist = d; _best = c; }}
      }});
      var _lblLat = _best ? _best[1] : ct.label_lat;
      var _lblLon = _best ? _best[0] : ct.label_lon;
      L.marker([_lblLat, _lblLon], {{ icon: L.divIcon({{
        html: '<div style="font-size:14px;font-weight:bold;color:#fff;'
            + 'font-family:Courier New,monospace;background:#000000;'
            + 'padding:0 3px;line-height:1.4;text-align:center;min-width:28px;">'
            + Math.round(ct.level / 10) + '</div>',
        iconSize: [32,14], iconAnchor: [16,7], className: ""
      }}), pane: "heightPane" }}).addTo(_synUALayer);
    }}
  }});

  // ── Temperature isotherms ─────────────────────────────────────────────
  (uaData.temp || []).forEach(function(ct) {{
    var t   = ct.level;
    var col = t > 0
      ? "rgb(" + Math.round(180 + 75*Math.min(t/40, 1)) + ",0,0)"
      : t < 0
      ? "rgb(0,0," + Math.round(180 + 75*Math.min(Math.abs(t)/40, 1)) + ")"
      : "#00bb00";
    var ll     = ct.coords.map(function(c) {{ return [c[1], c[0]]; }});
    var isBold = (Math.round(t) % 10 === 0);
    var _tLine = L.polyline(ll, {{
      color: col, weight: isBold ? 1.2 : 0.8,
      opacity: isBold ? 1.0 : 0.8, dashArray: "6 4"
    }});
    if (_synShowTooltips) _tLine.bindTooltip(_synLevel + " hPa T=" + t.toFixed(1) + "°C");
    _tLine.addTo(_synUALayer);

    // Isotherm label near BC coast
    var _bcLat = 54.0, _bcLon = -130.0, _bcBest = null, _bcBestDist = 1e9;
    ct.coords.forEach(function(c) {{
      var d = (c[1]-_bcLat)*(c[1]-_bcLat) + (c[0]-_bcLon)*(c[0]-_bcLon);
      if (d < _bcBestDist) {{ _bcBestDist = d; _bcBest = c; }}
    }});
    var _bcLblLat = _bcBest ? _bcBest[1] : ct.label_lat;
    var _bcLblLon = _bcBest ? _bcBest[0] : ct.label_lon;
    var _tVal = Math.round(t);
    var _tBg  = _tVal > 0 ? '#cc0000' : _tVal < 0 ? '#0044cc' : '#008800';
    L.marker([_bcLblLat, _bcLblLon], {{ icon: L.divIcon({{
      html: '<div style="font-size:12px;font-weight:' + (isBold ? "900" : "bold") + ';'
          + 'color:#ffffff;background:transparent;'
          + 'font-family:Courier New,monospace;'
          + 'text-shadow:-2px -2px 0 ' + _tBg + ',2px -2px 0 ' + _tBg + ',-2px 2px 0 ' + _tBg + ',2px 2px 0 ' + _tBg + ','
          + '-2px 0 0 ' + _tBg + ',2px 0 0 ' + _tBg + ',0 -2px 0 ' + _tBg + ',0 2px 0 ' + _tBg + ';'
          + 'padding:0 2px;line-height:1.4;text-align:center;">'
          + _tVal + '</div>',
      iconSize: [32,16], iconAnchor: [16,8], className: ""
    }}) }}).addTo(_synUALayer);
  }});

  // ── UA H/L centres ────────────────────────────────────────────────────
  var _uaHL = (_SYN_UA[fullKey] || {{}})["hl_" + _synLevel] || [];
  _uaHL.forEach(function(c) {{
    var _shadow = "1px 1px 0 white,-1px -1px 0 white,1px -1px 0 white,-1px 1px 0 white";
    var _html   = '<div style="font-size:61px;font-weight:bold;color:#000000;'
      + 'font-family:Palatino Linotype,Palatino,serif;line-height:1;'
      + 'text-shadow:' + _shadow + ';pointer-events:none;">' + c.type + '</div>';
    var _hlMark = L.marker([c.lat, c.lon], {{
      icon: L.divIcon({{ html: _html, iconSize: [70,65], iconAnchor: [35,32], className: "" }}),
      zIndexOffset: 200
    }});
    if (_synShowTooltips) _hlMark.bindTooltip(_synLevel + " hPa " + c.type);
    _hlMark.addTo(_synUALayer);
  }});

  _synUALayer.addTo(MAP);

  // ── Station wind barbs ────────────────────────────────────────────────
  var stns = _SYN_UA_STNS[fullKey] || [];
  if (!stns.length) console.warn("No UA stns for key:", fullKey);
  _synStnLayer = L.layerGroup();
  stns.forEach(function(s) {{
    if (_synLevel !== "850") return;
    var svgInfo = (s.svgs || {{}})[_synLevel];
    if (!svgInfo) return;
    var icon = L.divIcon({{
      html:       svgInfo.svg,
      iconSize:   [svgInfo.w, svgInfo.h],
      iconAnchor: [Math.round(svgInfo.w/2), Math.round(svgInfo.h/2)],
      className:  ""
    }});
    var _sMark = L.marker([s.lat, s.lon], {{ icon: icon }})
      .bindPopup(s.popup, {{ maxWidth: 320 }});
    if (_synShowTooltips) _sMark.bindTooltip(s.tip + " | " + (stepLabel || ""));
    _sMark.addTo(_synStnLayer);
  }});
  if (_synShowStations) _synStnLayer.addTo(MAP);
}}

// ═══════════════════════════════════════════════════════════════════════════
//  EXPORT: shared capture engine
// ═══════════════════════════════════════════════════════════════════════════

function _synCapture(cfg, onDone) {{
  var MAP  = _getMap();
  var keys = Object.keys(window).filter(function(k) {{ return k.startsWith("map_"); }});
  if (!keys.length) {{ _setExportStatus("Map not found"); if (onDone) onDone(false); return; }}
  var mapEl = document.getElementById(keys[0]) || document.querySelector(".leaflet-container");
  if (!mapEl)  {{ _setExportStatus("Map el not found"); if (onDone) onDone(false); return; }}

  _synLevel = cfg.level;
  _btnOff("btn-850"); _btnOff("btn-500");
  _btnOn("btn-" + cfg.level);
  synRender();

  var hideEls = [
    mapEl.querySelector(".leaflet-control-container"),
    document.querySelector(".leaflet-control-layers"),
    document.querySelector(".leaflet-control-zoom"),
    document.querySelector(".leaflet-control-attribution"),
    document.getElementById("syn-bar"),
    document.getElementById("syn-export-panel"),
    document.getElementById("syn-fs-btn")
  ].filter(Boolean);
  var prevVis = hideEls.map(function(el) {{ return el.style.visibility; }});
  hideEls.forEach(function(el) {{ el.style.visibility = "hidden"; }});
  // Hide all Leaflet tooltips during export
  document.querySelectorAll(".leaflet-tooltip").forEach(function(t) {{ t.style.display = "none"; }});

  var origW = mapEl.style.width;
  var origH = mapEl.style.height;
  function restore() {{
    mapEl.style.width  = origW;
    mapEl.style.height = origH;
    MAP.invalidateSize();
    hideEls.forEach(function(el, i) {{ el.style.visibility = prevVis[i]; }});
    document.querySelectorAll(".leaflet-tooltip").forEach(function(t) {{ t.style.display = ""; }});
  }}

  mapEl.style.width  = cfg.targetW + "px";
  mapEl.style.height = cfg.targetH + "px";
  MAP.invalidateSize();
  MAP.setView(cfg.center, cfg.zoom, {{ animate: false }});

  setTimeout(function() {{
    html2canvas(mapEl, {{
      useCORS: true, allowTaint: true,
      scale: 2, logging: false,
      width: cfg.targetW, height: cfg.targetH
    }}).then(function(canvas) {{

      var cropH = canvas.height;
      var cropW = Math.min(Math.round(cropH * cfg.cropRatioW / cfg.cropRatioH), canvas.width);

      var BANNER_H = 90;
      var CREDIT_H = 22;

      var out = document.createElement("canvas");
      out.width  = cropW;
      out.height = cropH + BANNER_H + CREDIT_H;
      var ctx = out.getContext("2d");
      ctx.drawImage(canvas, 0, 0, cropW, cropH, 0, 0, cropW, cropH);

      var step   = _SYN_TIME_STEPS[_synStepIdx] || {{}};
      var _key   = step.key || "";
      var _dYear = parseInt(_key.substring(0,4), 10);
      var _dMon  = parseInt(_key.substring(4,6), 10) - 1;
      var _dDay  = parseInt(_key.substring(6,8), 10);
      var _dH    = step.hour || 0;

      var _MONTHS_L = ["January","February","March","April","May","June",
                       "July","August","September","October","November","December"];
      var _MONTHS_S = ["JAN","FEB","MAR","APR","MAY","JUN",
                       "JUL","AUG","SEP","OCT","NOV","DEC"];
      var _DOWS_L   = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"];
      var _DOWS_S   = ["SUNDAY","MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY"];

      var _utcDate = new Date(Date.UTC(_dYear, _dMon, _dDay, _dH, 0, 0));
      var _yr      = _utcDate.getUTCFullYear();

      var _mar1     = new Date(Date.UTC(_yr, 2, 1));
      var _dstStart = new Date(Date.UTC(_yr, 2, 1 + (7 - _mar1.getUTCDay()) % 7 + 7));
      _dstStart.setUTCHours(8);
      var _nov1     = new Date(Date.UTC(_yr, 10, 1));
      var _dstEnd   = new Date(Date.UTC(_yr, 10, 1 + (7 - _nov1.getUTCDay()) % 7));
      _dstEnd.setUTCHours(7);

      var _offsetH  = (_utcDate >= _dstStart && _utcDate < _dstEnd) ? -6 : -7;
      var _tzLabel  = _offsetH === -6 ? "MDT" : "MST";
      var _localDate = new Date(_utcDate.getTime() + _offsetH * 3600000);
      var _lYear    = _localDate.getUTCFullYear();
      var _lMon     = _localDate.getUTCMonth();
      var _lDay     = _localDate.getUTCDate();
      var _lH       = _localDate.getUTCHours();
      var _ampm     = _lH < 12 ? "AM" : "PM";
      var _hr12     = _lH === 0 ? 12 : (_lH > 12 ? _lH - 12 : _lH);

      var TITLE_H = BANNER_H;
      out.height = cropH + BANNER_H + TITLE_H + CREDIT_H;
      ctx.drawImage(canvas, 0, 0, cropW, cropH, 0, 0, cropW, cropH);

      var blackW = Math.round(cropW * 0.28);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, cropH, cropW, BANNER_H);
      ctx.fillStyle = "#111111";
      ctx.fillRect(0, cropH, blackW, BANNER_H);

      var _tsStr = _DOWS_L[_localDate.getUTCDay()]
                 + " " + _MONTHS_L[_lMon]
                 + " " + _lDay + ", " + _lYear
                 + " - " + _hr12 + " " + _ampm;

      ctx.font         = "bold 32px Arial, sans-serif";
      ctx.fillStyle    = "#ffffff";
      ctx.textAlign    = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(_tsStr, blackW / 2, cropH + BANNER_H / 2);

      var _lvlTitle = cfg.level === "500"
        ? "500 hPa Heights and Isotherms"
        : "850 hPa Heights and Isotherms";

      ctx.font         = "bold 42px Arial, sans-serif";
      ctx.fillStyle    = "#111111";
      ctx.textAlign    = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(_lvlTitle, blackW + 24, cropH + BANNER_H / 2);

      ctx.font         = "bold 38px Arial, sans-serif";
      ctx.fillStyle    = "#333333";
      ctx.textAlign    = "right";
      ctx.textBaseline = "middle";
      ctx.fillText("AWCC Weather Office", cropW - 24, cropH + BANNER_H / 2);

      var creditY = cropH + BANNER_H;
      ctx.fillStyle = "#f0f0f0";
      ctx.fillRect(0, creditY, cropW, CREDIT_H);

      ctx.font         = "13px Arial, sans-serif";
      ctx.fillStyle    = "#777777";
      ctx.textAlign    = "right";
      ctx.textBaseline = "middle";
      ctx.fillText("Based on data issued by Meteorological Service of Canada",
                   cropW - 14, creditY + CREDIT_H / 2);

      var _exportNow = new Date();
      var _expStr = "Exported "
        + String(_exportNow.getUTCFullYear())
        + "-" + String(_exportNow.getUTCMonth()+1).padStart(2,"0")
        + "-" + String(_exportNow.getUTCDate()).padStart(2,"0")
        + " " + String(_exportNow.getUTCHours()).padStart(2,"0")
        + ":" + String(_exportNow.getUTCMinutes()).padStart(2,"0") + "Z";
      ctx.font         = "13px Arial, sans-serif";
      ctx.fillStyle    = "#777777";
      ctx.textAlign    = "left";
      ctx.fillText(_expStr, 14, creditY + CREDIT_H / 2);

      var _exportNow = new Date();
      var _expStr = "Exported "
        + String(_exportNow.getUTCFullYear())
        + "-" + String(_exportNow.getUTCMonth()+1).padStart(2,"0")
        + "-" + String(_exportNow.getUTCDate()).padStart(2,"0")
        + " " + String(_exportNow.getUTCHours()).padStart(2,"0")
        + ":" + String(_exportNow.getUTCMinutes()).padStart(2,"0") + "Z";
      ctx.font         = "20px Arial, sans-serif";
      ctx.fillStyle    = "#555555";
      ctx.textAlign = "left";
      ctx.fillText(_expStr, 20, creditY + CREDIT_H / 2);

      var _expDay   = _DOWS_S[_localDate.getUTCDay()];
      var _expMon   = _MONTHS_S[_lMon];
      var _hrStr    = String(_hr12).padStart(2,"0") + _ampm + _tzLabel;
      var _thisDate = _key.substring(0,8);
      var _seenDates = [];
      for (var _di = 0; _di < _SYN_TIME_STEPS.length; _di++) {{
        var _dkey = (_SYN_TIME_STEPS[_di].key || "").substring(0,8);
        if (_seenDates.indexOf(_dkey) === -1) _seenDates.push(_dkey);
        if (_dkey === _thisDate) break;
      }}
      var _dayNum  = String(_seenDates.length).padStart(2,"0");
      var _lvlPfx  = cfg.level === "500" ? "500mb" : "850mb";
      var _localDD = String(_lDay).padStart(2,"0");
      var fname = _lvlPfx + "LLJ-Day" + _dayNum + "-" + _hrStr
                + "_" + _expDay + "_" + _expMon
                + "_" + _localDD + "_" + _lYear + ".png";

      var link   = document.createElement("a");
      link.download = fname;
      link.href     = out.toDataURL("image/png");
      link.click();

      restore();
      if (onDone) onDone(true);

    }}).catch(function(e) {{
      console.error("html2canvas failed:", e);
      restore();
      _setExportStatus("✗ Capture error: " + e.message);
      if (onDone) onDone(false);
    }});
  }}, 600);
}}

// ── Export single level at current timestep ───────────────────────────────
function synSave850() {{
  _setExportStatus("Capturing 850mb...");
  _synCapture({{
    level: "850", center: [55, -104], zoom: 5,
    targetW: 1400, targetH: 1100, cropRatioW: 8.5, cropRatioH: 11.0
  }}, function(ok) {{
    _setExportStatus(ok ? "✓ 850mb saved!" : "✗ Failed");
    if (ok) setTimeout(function() {{ _setExportStatus(""); }}, 3000);
  }});
}}

function synSave500() {{
  _setExportStatus("Capturing 500mb...");
  _synCapture({{
    level: "500", center: [55, -118], zoom: 5,
    targetW: 1400, targetH: 1141, cropRatioW: 2944.0, cropRatioH: 2400.0
  }}, function(ok) {{
    _setExportStatus(ok ? "✓ 500mb saved!" : "✗ Failed");
    if (ok) setTimeout(function() {{ _setExportStatus(""); }}, 3000);
  }});
}}

// ── Export All: every timestep × both levels ──────────────────────────────
var _exportAllQueue   = [];
var _exportAllRunning = false;

function synExportAll() {{
  if (_exportAllRunning) {{ _setExportStatus("Already running..."); return; }}
  _exportAllQueue = [];
  var total = _SYN_TIME_STEPS.length;
  for (var i = 0; i < total; i++) {{ _exportAllQueue.push({{ stepIdx: i, level: "850" }}); }}
  _exportAllRunning = true;
  _setExportStatus("Export All: 0/" + total);
  _runExportQueue(0, total);
}}

function _runExportQueue(done, total) {{
  if (_exportAllQueue.length === 0) {{
    _exportAllRunning = false;
    _setExportStatus("✓ All " + total + " images saved!");
    setTimeout(function() {{ _setExportStatus(""); }}, 5000);
    return;
  }}
  var job = _exportAllQueue.shift();
  _synStepIdx = job.stepIdx;
  var slider  = document.getElementById("syn-time-slider");
  if (slider) slider.value = String(job.stepIdx);
  var step = _SYN_TIME_STEPS[job.stepIdx] || {{}};
  var lbl  = document.getElementById("syn-ts-label");
  if (lbl) lbl.textContent = step.label || "";
  synRenderUA(step.key, step.label);

  var cfg = (job.level === "850")
    ? {{ level: "850", center: [55,-104], zoom: 5, targetW: 1400, targetH: 1100, cropRatioW: 8.5,    cropRatioH: 11.0   }}
    : {{ level: "500", center: [55,-118], zoom: 5, targetW: 1400, targetH: 1141, cropRatioW: 2944.0, cropRatioH: 2400.0 }};

  _setExportStatus("Exporting " + job.level + " step " + (job.stepIdx+1) + "/" + _SYN_TIME_STEPS.length + " (" + (done+1) + "/" + total + ")");
  setTimeout(function() {{
    _synCapture(cfg, function() {{
      setTimeout(function() {{ _runExportQueue(done + 1, total); }}, 1000);
    }});
  }}, 600);
}}

// ── jsPDF multi-page PDF export ───────────────────────────────────────────
var _pdfRunning = false;

function _ensureJsPDF(cb) {{
  if (typeof window.jspdf !== "undefined") {{ cb(); return; }}
  _setExportStatus("Loading jsPDF...");
  var s = document.createElement("script");
  s.src = "https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js";
  s.onload  = function() {{ cb(); }};
  s.onerror = function() {{ _setExportStatus("✗ jsPDF load failed"); _pdfRunning = false; }};
  document.head.appendChild(s);
}}

// ── Single-level PDF builder (calls onComplete when done) ─────────────────
function _doExportPDF(level, onComplete) {{
  _pdfRunning = true;
  var total = _SYN_TIME_STEPS.length;
  var queue = [];
  for (var i = 0; i < total; i++) queue.push(i);

  var PAGE_W_MM = 279.4;
  var PAGE_H_MM = 215.9;
  var doc = new window.jspdf.jsPDF({{ orientation: "landscape", unit: "mm", format: [PAGE_W_MM, PAGE_H_MM] }});

  var cfg = (level === "850")
    ? {{ level: "850", center: [55,-118], zoom: 5, targetW: 1400, targetH: 1141, cropRatioW: 2944,    cropRatioH: 2400   }}
    : {{ level: "500", center: [55,-118], zoom: 5, targetW: 1400, targetH: 1141, cropRatioW: 2944.0, cropRatioH: 2400.0 }};

  _setExportStatus("PDF " + level + ": 0/" + total);

  function processNext(idx) {{
    _synLevel = level;
    _btnOff("btn-850"); _btnOff("btn-500");
    _btnOn("btn-" + level);
    if (idx >= queue.length) {{
      var now   = new Date();
      var fname = level + "mb_ALL_"
        + now.getUTCFullYear()
        + String(now.getUTCMonth()+1).padStart(2,"0")
        + String(now.getUTCDate()).padStart(2,"0")
        + "_" + String(now.getUTCHours()).padStart(2,"0")
        + String(now.getUTCMinutes()).padStart(2,"0") + "Z.pdf";
      doc.save(fname);
      _pdfRunning = false;
      _setExportStatus("✓ " + level + "mb PDF saved (" + total + " pages)!");
      if (onComplete) onComplete();
      return;
    }}

    var stepIdx = queue[idx];
    _synStepIdx = stepIdx;
    var slider = document.getElementById("syn-time-slider");
    if (slider) slider.value = String(stepIdx);
    var step = _SYN_TIME_STEPS[stepIdx] || {{}};
    var lbl  = document.getElementById("syn-ts-label");
    if (lbl) lbl.textContent = step.label || "";
    synRenderUA(step.key, step.label);
    _setExportStatus("PDF " + level + ": " + (idx+1) + "/" + total);

    setTimeout(function() {{
      var MAP  = _getMap();
      var keys = Object.keys(window).filter(function(k) {{ return k.startsWith("map_"); }});
      if (!keys.length) {{ processNext(idx+1); return; }}
      var mapEl = document.getElementById(keys[0]) || document.querySelector(".leaflet-container");
      if (!mapEl)  {{ processNext(idx+1); return; }}

      var hideEls = [
        mapEl.querySelector(".leaflet-control-container"),
        document.querySelector(".leaflet-control-layers"),
        document.querySelector(".leaflet-control-zoom"),
        document.querySelector(".leaflet-control-attribution"),
        document.getElementById("syn-bar"),
        document.getElementById("syn-export-panel"),
        document.getElementById("syn-fs-btn")
      ].filter(Boolean);
      var prevVis = hideEls.map(function(el) {{ return el.style.visibility; }});
      hideEls.forEach(function(el) {{ el.style.visibility = "hidden"; }});
      // Hide all Leaflet tooltips during export
      document.querySelectorAll(".leaflet-tooltip").forEach(function(t) {{ t.style.display = "none"; }});

      var origW = mapEl.style.width, origH = mapEl.style.height;
      function restore() {{
        mapEl.style.width  = origW;
        mapEl.style.height = origH;
        MAP.invalidateSize();
        hideEls.forEach(function(el, i) {{ el.style.visibility = prevVis[i]; }});
        document.querySelectorAll(".leaflet-tooltip").forEach(function(t) {{ t.style.display = ""; }});
      }}

      mapEl.style.width  = cfg.targetW + "px";
      mapEl.style.height = cfg.targetH + "px";
      MAP.invalidateSize();
      MAP.setView(cfg.center, cfg.zoom, {{ animate: false }});

      setTimeout(function() {{
        html2canvas(mapEl, {{
          useCORS: true, allowTaint: true,
          scale: 1.5, logging: false,
          width: cfg.targetW, height: cfg.targetH
        }}).then(function(canvas) {{

          var cropH    = canvas.height;
          var cropW    = Math.min(Math.round(cropH * cfg.cropRatioW / cfg.cropRatioH), canvas.width);
          var BANNER_H = 80, TITLE_H = 80, CREDIT_H = 28;

          var out    = document.createElement("canvas");
          out.width  = cropW;
          out.height = cropH + BANNER_H + TITLE_H + CREDIT_H;
          var ctx    = out.getContext("2d");
          ctx.drawImage(canvas, 0, 0, cropW, cropH, 0, 0, cropW, cropH);

          var _key   = step.key || "";
          var _dYear = parseInt(_key.substring(0,4),10);
          var _dMon  = parseInt(_key.substring(4,6),10)-1;
          var _dDay  = parseInt(_key.substring(6,8),10);
          var _dH    = step.hour || 0;
          var _ML    = ["January","February","March","April","May","June","July","August","September","October","November","December"];
          var _DL    = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"];
          var _utc   = new Date(Date.UTC(_dYear,_dMon,_dDay,_dH,0,0));
          var _yr    = _utc.getUTCFullYear();
          var _m1    = new Date(Date.UTC(_yr,2,1));
          var _ds    = new Date(Date.UTC(_yr,2,1+(7-_m1.getUTCDay())%7+7)); _ds.setUTCHours(8);
          var _n1    = new Date(Date.UTC(_yr,10,1));
          var _de    = new Date(Date.UTC(_yr,10,1+(7-_n1.getUTCDay())%7));  _de.setUTCHours(7);
          var _off   = (_utc>=_ds && _utc<_de) ? -6 : -7;
          var _loc   = new Date(_utc.getTime()+_off*3600000);
          var _lH    = _loc.getUTCHours();
          var _ampm  = _lH<12?"AM":"PM";
          var _h12   = _lH===0?12:(_lH>12?_lH-12:_lH);
          var _tsStr = _DL[_loc.getUTCDay()]+" "+_ML[_loc.getUTCMonth()]+" "+_loc.getUTCDate()+", "+_loc.getUTCFullYear()+" - "+_h12+" "+_ampm;

          var blackW = Math.round(cropW*0.33);
          ctx.fillStyle="#ffffff"; ctx.fillRect(0,cropH,cropW,BANNER_H * 1);
          ctx.fillStyle="#111111"; ctx.fillRect(0,cropH,blackW,BANNER_H * 1);
          ctx.font="bold 30px Arial,sans-serif"; ctx.fillStyle="#ffffff";
          ctx.textAlign="center"; ctx.textBaseline="middle";
          ctx.fillText(_tsStr,blackW/2,cropH+BANNER_H/2);

          var titleY=cropH+BANNER_H;
          ctx.fillStyle="#ffffff"; ctx.fillRect(0,titleY,cropW,TITLE_H);
          ctx.font="bold 32px Arial,sans-serif"; ctx.fillStyle="#111111";
          ctx.textAlign="left"; ctx.textBaseline="middle";
          ctx.fillText(cfg.level==="500"?"500 hPa Heights and Isotherms":"850hPa LLJ Prog",24,titleY+TITLE_H/2);
          ctx.font="24px Arial,sans-serif"; ctx.fillStyle="#333333";
          ctx.textAlign="right";
          ctx.fillText("AWCC Weather Office",cropW-24,titleY+TITLE_H/2);

          var creditY=titleY+TITLE_H;
          ctx.fillStyle="#ffffff"; ctx.fillRect(0,creditY,cropW,CREDIT_H);
          ctx.font="12px Arial,sans-serif"; ctx.fillStyle="#555555";
          ctx.textAlign="right"; ctx.textBaseline="middle";
          ctx.fillText("Based on data issued by Meteorological Service of Canada",cropW-20,creditY+CREDIT_H/2);

          var imgData = out.toDataURL("image/jpeg", 0.88);
          if (idx > 0) doc.addPage([PAGE_W_MM, PAGE_H_MM], "landscape");
          doc.addImage(imgData, "JPEG", 0, 0, PAGE_W_MM, PAGE_H_MM);

          restore();
          setTimeout(function() {{ processNext(idx+1); }}, 800);

        }}).catch(function(e) {{
          console.error("html2canvas error:", e);
          restore();
          processNext(idx+1);
        }});
      }}, 600);
    }}, 400);
  }}

  processNext(0);
}}

// ── One button → two PDFs ─────────────────────────────────────────────────
function synExportBothPDFs() {{
  if (_pdfRunning) {{ _setExportStatus("PDF already running..."); return; }}
  _ensureJsPDF(function() {{
    _setExportStatus("Starting 850mb PDF...");
    _doExportPDF("850", function() {{
      _setExportStatus("✓ 850mb PDF saved!");
      setTimeout(function() {{ _setExportStatus(""); }}, 5000);
    }});
  }});
}}

// ── Init ──────────────────────────────────────────────────────────────────
function _maybeTip(marker, text) {{
  if (_synShowTooltips) marker.bindTooltip(text);
  return marker;
}}
function _synInit() {{
  var slider = document.getElementById("syn-time-slider");
  if (slider) {{
    slider.max   = String(Math.max(0, _SYN_TIME_STEPS.length - 1));
    slider.value = "0";
  }}
  synSetLevel("850");
  synRender();
}}

if (document.readyState === "complete") {{ setTimeout(_synInit, 700); }}
else {{ window.addEventListener("load", function() {{ setTimeout(_synInit, 700); }}); }}
</script>
'''

m.get_root().html.add_child(Element(_bar_html))
m.get_root().html.add_child(Element(_js))

os.makedirs('outputs', exist_ok=True)
m.save('outputs/llj_prog.html')
print(f'\n✅ LLJ map saved → outputs/llj_prog.html')

# ── Cell 5B. GEM Surface map — MSLP contours + QPF fill, time slider ──────

import folium
from folium import Element
import json as _json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.ndimage import zoom as _zoom

if 'ua_raw_df' not in globals():
    raise RuntimeError('❌ Run Cell UA-2b first')

_sfc_keys = sorted(
    k.replace('slp_grid_', '')
    for k in globals()
    if k.startswith('slp_grid_')
    and not k == 'slp_grid'
)
if not _sfc_keys:
    raise RuntimeError('❌ No slp_grid_* found — run Cell UA-2c first')

print(f'Found {len(_sfc_keys)} time steps: {_sfc_keys}')

_MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
_DOWS   = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

_sfc_time_steps = []
for _key in _sfc_keys:
    _y, _h  = int(_key[:4]), int(_key[9:])
    _mo, _dd = int(_key[4:6]), int(_key[6:8])
    _dt     = pd.Timestamp(year=_y, month=_mo, day=_dd)
    _label  = f'{_DOWS[_dt.dayofweek]} {_MONTHS[_mo-1]} {_dd} {_h:02d}Z'
    _sfc_time_steps.append({'key': _key, 'label': _label, 'hour': _h})

print('Time steps:', [s['label'] for s in _sfc_time_steps])

# ── Contour extractor — same pattern as Block 04 ─────────────────────────────
def _extract_contours(grid, lon_vec, lat_vec, interval):
    glon, glat = np.meshgrid(lon_vec, lat_vec)
    vmin  = np.floor((np.nanmin(grid) - 1000) / interval) * interval + 1000
    vmax  = np.ceil( (np.nanmax(grid) - 1000) / interval) * interval + 1000
    levels = np.arange(vmin, vmax + interval, interval)
    fig, ax = plt.subplots(figsize=(1, 1))
    cs = ax.contour(glon, glat, grid, levels=levels)
    plt.close(fig)
    contours = []
    for li, lvl in enumerate(cs.levels):
        is_major = (round((lvl - 1000) % 16) == 0)
        weight   = 3.5 if is_major else 3.0
        opacity  = 1.0 if is_major else 1.0
        for coords in cs.allsegs[li]:
            if len(coords) < 2: continue
            mid = coords[len(coords) // 2]
            contours.append({
                'level':     round(float(lvl), 1),
                'bold':      is_major,
                'weight':    weight,
                'opacity':   opacity,
                'coords':    [[round(float(c[1]), 3), round(float(c[0]), 3)] for c in coords],
                'label_lon': round(float(mid[0]), 3),
                'label_lat': round(float(mid[1]), 3),
            })
    return contours


# ── QPF fill bands — vectorised ───────────────────────────────────────────────
_QPF_LEVELS = [0.6, 1.5, 3, 5, 10, 20, 30, 40, 50, 60, 80, 100, 120]
_QPF_COLORS = {
    0.6:  '#c8f0a0', 1.5:  '#78d048', 3:    '#228b22',
    5:    '#00aaaa', 10:   '#1a78c2', 20:   '#6a0dad',
    30:   '#cc00cc', 40:   '#ffff00', 50:   '#ffaa00',
    60:   '#ff4400', 80:   '#cc0000', 100:  '#880000', 120: '#111111'
}

def _extract_qpf_bands(grid, lat_vec, lon_vec, n_interp=120):
    if grid is None:
        return []
    zoom_lat = n_interp / grid.shape[0]
    zoom_lon = n_interp / grid.shape[1]
    gq   = _zoom(grid, (zoom_lat, zoom_lon), order=1)
    gq   = np.clip(gq, 0, None)
    if gq.max() < 0.6:
        return []  # nothing above lowest QPF threshold
    latf = np.linspace(lat_vec[0], lat_vec[-1], n_interp)
    lonf = np.linspace(lon_vec[0], lon_vec[-1], n_interp)
    glon, glat = np.meshgrid(lonf, latf)
    fig, ax = plt.subplots(figsize=(1, 1))
    try:
        cs = ax.contourf(glon, glat, gq, levels=_QPF_LEVELS, extend='max')
    except Exception:
        plt.close(fig)
        return []
    plt.close(fig)
    bands = []
    for li, (lvl, seg_list) in enumerate(zip(cs.levels, cs.allsegs)):
        color = _QPF_COLORS.get(lvl, _QPF_COLORS[_QPF_LEVELS[-1]])
        for verts in seg_list:
            verts = np.array(verts)
            if len(verts) < 3:
                continue
            coords = [[round(float(v[1]),3), round(float(v[0]),3)]
                      for v in verts if not np.isnan(v).any()]
            if len(coords) < 3:
                continue
            bands.append({'level': float(lvl), 'color': color, 'coords': coords})
    return bands


# ── Bake all frames ───────────────────────────────────────────────────────────
_MSLP_INTERVAL = float(globals().get('MSLP_INTERVAL', 16.0))

print('Baking contours...')
_frame_data = {}

for _key in _sfc_keys:
    _slp  = globals().get(f'slp_grid_{_key}')
    _qpf  = globals().get(f'qpf_grid_{_key}')
    _lonv = globals().get(f'lon_vec_{_key}')
    _latv = globals().get(f'lat_vec_{_key}')
    if _slp is None or _lonv is None:
        continue

    _mslp_contours = _extract_contours(_slp, _lonv, _latv, _MSLP_INTERVAL)
    _qpf_latv  = globals().get(f'qpf_lat_vec_{_key}', _latv)
    _qpf_lonv  = globals().get(f'qpf_lon_vec_{_key}', _lonv)
    _qpf_bands = _extract_qpf_bands(_qpf, _qpf_latv, _qpf_lonv) if _qpf is not None else []
    _frame_data[_key] = {
        'mslp': _mslp_contours,
        'qpf':  _qpf_bands,
        'hl':   hl_centers_by_key.get(_key, []),
        'bbox': [float(_latv[0]), float(_lonv[0]), float(_latv[-1]), float(_lonv[-1])]
    }
    print(f'  {_key}: {len(_mslp_contours)} MSLP segs, {len(_qpf_bands)} QPF bands')
    if _qpf_bands:
        print(f'    QPF levels present: {sorted(set(b["level"] for b in _qpf_bands))}')
    else:
        print(f'    QPF max value: {_qpf.max():.2f} mm — may be below 0.5 mm threshold')

print(f'✓ Baked {len(_frame_data)} frames')

# ── Folium map ────────────────────────────────────────────────────────────────
_center_lat = float(globals()[f'lat_vec_{_sfc_keys[0]}'].mean())
_center_lon = float(globals()[f'lon_vec_{_sfc_keys[0]}'].mean())

m = folium.Map(location=[_center_lat, _center_lon], zoom_start=5,
               tiles=None, prefer_canvas=True)
folium.TileLayer(tiles='about:blank', attr=' ', name='Blank',
                 max_zoom=19, show=True).add_to(m)
m.get_root().html.add_child(Element(
    '<style>.leaflet-container{background:#e0f2ff!important;}.leaflet-control-zoom{display:none!important;}</style>'
))

_borders_js = '''
<script>
(function(){
  function loadBorders(){
    var keys=Object.keys(window).filter(function(k){return k.startsWith("map_");});
    if(!keys.length){setTimeout(loadBorders,200);return;}
    var MAP=window[keys[0]];
    MAP.options.zoomSnap=0;
    MAP.options.zoomDelta=0.1;
    ["landPane","bordersPane","heightPane","qpfPane"].forEach(function(pname,zi){
      if(!MAP.getPane(pname)){
        MAP.createPane(pname);
        MAP.getPane(pname).style.zIndex=[205,220,490,210][zi];
        MAP.getPane(pname).style.pointerEvents="none";
      }
    });
    fetch("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_land.geojson")
      .then(function(r){return r.json();})
      .then(function(gj){
        L.geoJSON(gj,{style:function(){return{color:"none",weight:0,fill:true,fillColor:"#dedede",fillOpacity:1.0};},pane:"landPane"}).addTo(MAP);
      });
    [
      ["https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_coastline.geojson",
       {color:"#444",weight:2.0,opacity:1.0,fill:false}],
      ["https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_boundary_lines_land.geojson",
       {color:"#ffffff",weight:2.0,opacity:1.0,fill:false}],
      ["https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_1_states_provinces_lines.geojson",
       {color:"#ffffff",weight:1.4,opacity:0.85,fill:false}],
      ["https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_lakes.geojson",
       {color:"#5588aa",weight:1.5,opacity:0.9,fill:false}]
    ].forEach(function(item){
      fetch(item[0]).then(function(r){return r.json();}).then(function(gj){
        L.geoJSON(gj,{style:function(){return item[1];},pane:"bordersPane"}).addTo(MAP);
      });
    });
    fetch("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_1_states_provinces.geojson")
      .then(function(r){return r.json();})
      .then(function(gj){
        var ab={type:"FeatureCollection",features:gj.features.filter(function(f){return f.properties.name==="Alberta";})};
        L.geoJSON(ab,{style:function(){return{color:"#cc0000",weight:2.5,opacity:1.0,fill:true,fillColor:"#ffffff",fillOpacity:1.0};},pane:"landPane"}).addTo(MAP);
      });
  }
  if(document.readyState==="complete"){setTimeout(loadBorders,600);}
  else{window.addEventListener("load",function(){setTimeout(loadBorders,600);});}
})();
</script>
'''
m.get_root().html.add_child(Element(_borders_js))

_fullscreen_html = '''
<style>
#syn-fs-btn{
  position:fixed;top:10px;left:10px;z-index:10001;
  background:rgba(255,255,255,0.96);border:1px solid #aaa;border-radius:6px;
  padding:5px 10px;font-family:Courier New,monospace;font-size:12px;
  box-shadow:0 2px 8px rgba(0,0,0,0.15);cursor:pointer;color:#1a3a6a;
}
#syn-fs-btn:hover{background:#e8f0fe;}
</style>
<button id="syn-fs-btn" onclick="synToggleFS()">&#x26F6; Fullscreen</button>
<script>
var _synFS=false,_synMapEl=null,_synOrigStyle="";
function synToggleFS(){
  var btn=document.getElementById("syn-fs-btn");
  var keys=Object.keys(window).filter(function(k){return k.startsWith("map_");});
  if(!keys.length)return;
  var MAP=window[keys[0]];
  if(!_synMapEl){_synMapEl=document.getElementById(keys[0])||document.querySelector(".leaflet-container");}
  if(!_synMapEl)return;
  _synFS=!_synFS;
  if(_synFS){
    _synOrigStyle=_synMapEl.getAttribute("style")||"";
    _synMapEl.setAttribute("style","position:fixed!important;top:0;left:0;width:100vw!important;height:100vh!important;z-index:9999!important;margin:0!important;");
    btn.innerHTML="&#x274C; Exit Fullscreen";
  } else {
    _synMapEl.setAttribute("style",_synOrigStyle);
    btn.innerHTML="&#x26F6; Fullscreen";
  }
  setTimeout(function(){MAP.invalidateSize();},100);
}
</script>
'''
m.get_root().html.add_child(Element(_fullscreen_html))

if 'fire_zones_html' in globals():
    m.get_root().html.add_child(Element(fire_zones_html))

_bar_html = '''
<style>
#gem-bar{
  position:fixed;bottom:0;left:0;right:0;z-index:10000;
  background:#1a1a2e;border-top:2px solid #4a7fc1;
  padding:8px 16px;display:flex;align-items:center;gap:14px;
  font-family:"Courier New",monospace;font-size:11px;color:#e0e0e0;
  box-shadow:0 -3px 12px rgba(0,0,0,0.5);min-height:52px;
}
#gem-bar .bar-label{font-size:8px;color:#8888aa;font-weight:bold;
  text-transform:uppercase;letter-spacing:0.5px;white-space:nowrap;}
#gem-bar .bar-section{display:flex;align-items:center;gap:6px;
  border-right:1px solid #3a3a5a;padding-right:14px;white-space:nowrap;}
#gem-bar .bar-section:last-child{border-right:none;}
.gem-layer-btn{font-size:12px;padding:4px 14px;cursor:pointer;
  border:1px solid #3a4a6a;border-radius:4px;background:#2a2a4a;
  color:#c0c8e0;font-family:"Courier New",monospace;font-weight:bold;}
.gem-layer-btn:hover{background:#3a4a7a;}
.gem-layer-btn.active{background:#4a7fc1;color:#fff;border-color:#6a9fe1;}
#gem-time-slider{width:340px;accent-color:#4a7fc1;cursor:pointer;}
#gem-ts-label{color:#c0d0ff;font-size:12px;min-width:220px;font-weight:bold;}
</style>
<div id="gem-bar">
   <div class="bar-section">
    <span class="bar-label">Export</span>
    <button class="gem-layer-btn" id="btn-export-all" onclick="gemExportAll()" style="border-color:#cc8800;color:#ffcc66;">Export All PNG</button>
    <span id="gem-export-status" style="font-size:10px;color:#ffcc66;min-width:120px;"></span>
  </div>
  <div class="bar-section">
    <span class="bar-label">Layers</span>
    <button class="gem-layer-btn active" id="btn-mslp" onclick="gemToggle('mslp')">MSLP</button>
    <button class="gem-layer-btn active" id="btn-qpf"  onclick="gemToggle('qpf')">QPF 12h</button>
  </div>
  <div class="bar-section">
    <span class="bar-label">Time</span>
    <input type="range" id="gem-time-slider" min="0" value="0"
           oninput="gemSliderChange(this.value)">
  </div>
  <div class="bar-section">
    <span id="gem-ts-label">—</span>
  </div>
  <div class="bar-section">
    <span class="bar-label">Legend</span>
    <span style="font-size:10px;color:#aac4ff;">━ MSLP 4 hPa &nbsp;┅ bold 16 hPa</span>
    <span style="font-size:10px;color:#aac4ff;margin-left:8px;">QPF 12h (mm):</span>
    <span style="display:inline-flex;align-items:center;gap:3px;margin-left:4px;">
      <span style="background:#c8f0a0;width:18px;height:14px;display:inline-block;border:1px solid #555;"></span><span style="font-size:9px;">0.6</span>
      <span style="background:#78d048;width:18px;height:14px;display:inline-block;border:1px solid #555;"></span><span style="font-size:9px;">1.5</span>
      <span style="background:#228b22;width:18px;height:14px;display:inline-block;border:1px solid #555;"></span><span style="font-size:9px;">3</span>
      <span style="background:#00aaaa;width:18px;height:14px;display:inline-block;border:1px solid #555;"></span><span style="font-size:9px;">5</span>
      <span style="background:#1a78c2;width:18px;height:14px;display:inline-block;border:1px solid #555;"></span><span style="font-size:9px;">10</span>
      <span style="background:#6a0dad;width:18px;height:14px;display:inline-block;border:1px solid #555;"></span><span style="font-size:9px;">20</span>
      <span style="background:#cc00cc;width:18px;height:14px;display:inline-block;border:1px solid #555;"></span><span style="font-size:9px;">30</span>
      <span style="background:#ffff00;width:18px;height:14px;display:inline-block;border:1px solid #555;"></span><span style="font-size:9px;">40</span>
      <span style="background:#ffaa00;width:18px;height:14px;display:inline-block;border:1px solid #555;"></span><span style="font-size:9px;">50</span>
      <span style="background:#ff4400;width:18px;height:14px;display:inline-block;border:1px solid #555;"></span><span style="font-size:9px;">60</span>
      <span style="background:#cc0000;width:18px;height:14px;display:inline-block;border:1px solid #555;"></span><span style="font-size:9px;">80</span>
      <span style="background:#880000;width:18px;height:14px;display:inline-block;border:1px solid #555;"></span><span style="font-size:9px;">100</span>
      <span style="background:#111111;width:18px;height:14px;display:inline-block;border:1px solid #555;"></span><span style="font-size:9px;">120+</span>
    </span>
  </div>
</div>
'''
m.get_root().html.add_child(Element(_bar_html))

_time_steps_str = _json.dumps(_sfc_time_steps)
_frame_data_str = _json.dumps(_frame_data)

_js = f'''
<script>
var _GEM_STEPS  = {_time_steps_str};
var _GEM_FRAMES = {_frame_data_str};

function _utcKeyToLocalStr(key) {{
  var yr  = parseInt(key.substring(0,4), 10);
  var mo  = parseInt(key.substring(4,6), 10) - 1;
  var dy  = parseInt(key.substring(6,8), 10);
  var hr  = parseInt(key.substring(9,11), 10);
  var ML  = ["January","February","March","April","May","June",
             "July","August","September","October","November","December"];
  var DL  = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"];
  var utc = new Date(Date.UTC(yr, mo, dy, hr, 0, 0));
  var y2  = utc.getUTCFullYear();
  var m1  = new Date(Date.UTC(y2, 2, 1));
  var ds  = new Date(Date.UTC(y2, 2, 1+(7-m1.getUTCDay())%7+7)); ds.setUTCHours(8);
  var n1  = new Date(Date.UTC(y2, 10, 1));
  var de  = new Date(Date.UTC(y2, 10, 1+(7-n1.getUTCDay())%7)); de.setUTCHours(7);
  var off = (utc >= ds && utc < de) ? -6 : -7;
  var tz  = off === -6 ? "MDT" : "MST";
  var loc = new Date(utc.getTime() + off * 3600000);
  var lh  = loc.getUTCHours();
  var h12 = lh === 0 ? 12 : (lh > 12 ? lh - 12 : lh);
  var ampm = lh < 12 ? "AM" : "PM";
  return DL[loc.getUTCDay()] + " " + ML[loc.getUTCMonth()] + " "
       + loc.getUTCDate() + ", " + loc.getUTCFullYear()
       + " \u2014 " + h12 + " " + ampm + " " + tz;
}}

var _gemStepIdx   = 0;
var _gemShowMslp  = true;
var _gemShowQpf   = true;
var _gemMslpLayer  = null;
var _gemQpfLayer   = null;
var _gemExporting  = false;

function _getMap(){{
  var k=Object.keys(window).filter(function(k){{return k.startsWith("map_");}});
  return k.length?window[k[0]]:null;
}}

function gemToggle(which){{
  if(which==="mslp"){{ _gemShowMslp=!_gemShowMslp; document.getElementById("btn-mslp").classList.toggle("active",_gemShowMslp); }}
  else              {{ _gemShowQpf =!_gemShowQpf;  document.getElementById("btn-qpf" ).classList.toggle("active",_gemShowQpf);  }}
  gemRender(_gemStepIdx);
}}

function gemSliderChange(v){{
  _gemStepIdx=parseInt(v);
  gemRender(_gemStepIdx);
}}

function gemRender(idx){{
  var MAP=_getMap(); if(!MAP) return;
  var step=_GEM_STEPS[idx]; if(!step) return;
  var lbl=document.getElementById("gem-ts-label");
  if(lbl) lbl.textContent=_utcKeyToLocalStr(step.key);

  if(_gemQpfLayer) {{ MAP.removeLayer(_gemQpfLayer);  _gemQpfLayer=null; }}
  if(_gemMslpLayer){{ MAP.removeLayer(_gemMslpLayer); _gemMslpLayer=null; }}

  var fd=_GEM_FRAMES[step.key]; if(!fd) return;

  // ── QPF dots ──────────────────────────────────────────────────────────
  if(_gemShowQpf && fd.qpf && fd.qpf.length){{
    _gemQpfLayer=L.layerGroup();
    if(!MAP.getPane("qpfPane")){{
      MAP.createPane("qpfPane");
      MAP.getPane("qpfPane").style.zIndex=480;
      MAP.getPane("qpfPane").style.pointerEvents="none";
    }}
    fd.qpf.slice().sort(function(a,b){{return a.level-b.level;}}).forEach(function(band){{
      if(!band.coords||band.coords.length<3) return;
      L.polygon([band.coords],{{
        color:"none",
        weight:0,
        fillColor:band.color,
        fillOpacity:0.45,
        interactive:false,
        pane:"qpfPane"
      }}).addTo(_gemQpfLayer);
    }});
    _gemQpfLayer.addTo(MAP);
  }}


  // ── Debug location pins ───────────────────────────────────────────
  if(!_gemQpfLayer){{
    _gemQpfLayer=L.layerGroup();
    _gemQpfLayer.addTo(MAP);
  }}
  var _debugPins=[
    {{lat:53.5461,lon:-113.4938,name:"Edmonton"}},

  ];
  _debugPins.forEach(function(p){{
    L.circleMarker([p.lat,p.lon],{{
      radius:5,
      color:"#ff0000",
      fillColor:"#ff0000",
      fillOpacity:1.0,
      weight:2,
      pane:"qpfPane"
    }}).addTo(_gemQpfLayer);
  }});

  // ── MSLP contours ─────────────────────────────────────────────────────
  if(_gemShowMslp && fd.mslp && fd.mslp.length){{
    _gemMslpLayer=L.layerGroup();
    if(!MAP.getPane("heightPane")){{
      MAP.createPane("heightPane");
      MAP.getPane("heightPane").style.zIndex=490;
      MAP.getPane("heightPane").style.pointerEvents="none";
    }}
    fd.mslp.forEach(function(ct){{
      if(!ct.coords||ct.coords.length<2) return;
      L.polyline(ct.coords,{{
        color:"#000000",
        weight:ct.weight,
        opacity:ct.opacity,
        pane:"heightPane"
      }}).addTo(_gemMslpLayer);
    }});

    // Labels on bold (every 20 hPa) contours
    var _labelled={{}};
    var _skLat=54.0,_skLon=-106.0;
    fd.mslp.forEach(function(ct){{
      if(!ct.coords||ct.coords.length<2) return;
      var _best=null,_bestDist=1e9;
      ct.coords.forEach(function(c){{
        var d=(c[0]-_skLat)*(c[0]-_skLat)+(c[1]-_skLon)*(c[1]-_skLon);
        if(d<_bestDist){{_bestDist=d;_best=c;}}
      }});
      if(!_best) return;
      L.marker([_best[0],_best[1]],{{
        icon:L.divIcon({{
          html:'<div style="font-size:11px;font-weight:bold;color:#fff;'
              +'font-family:Courier New,monospace;background:#000000;'
              +'padding:0 3px;line-height:1.4;text-align:center;min-width:28px;">'
              +ct.level.toFixed(0)+'</div>',
          iconSize:[36,14],iconAnchor:[18,7],className:""
        }}),
        pane:"heightPane"
      }}).addTo(_gemMslpLayer);
    }});

    // ── H/L centres ───────────────────────────────────────────────────
    (fd.hl||[]).forEach(function(c){{
      var _isH = c.type==="H";
      var _shadow = "1px 1px 0 white,-1px -1px 0 white,1px -1px 0 white,-1px 1px 0 white";
      L.marker([c.lat,c.lon],{{
        icon:L.divIcon({{
          html:'<div style="font-size:52px;font-weight:bold;'
              +'color:'+(_isH?'#000000':'#000000')+';'
              +'font-family:Palatino Linotype,Palatino,serif;line-height:1;'
              +'text-shadow:'+_shadow+';pointer-events:none;">'
              +c.type+'</div>',
          iconSize:[60,70],iconAnchor:[30,35],className:""
        }}),
        pane:"heightPane"
      }}).addTo(_gemMslpLayer);
    }});

    _gemMslpLayer.addTo(MAP);
  }}
}}

function _gemInit(){{
  var slider=document.getElementById("gem-time-slider");
  if(slider){{
    slider.max=String(Math.max(0,_GEM_STEPS.length-1));
    slider.value="0";
  }}
  gemRender(0);
}}



function _gemSetStatus(msg){{
  var el=document.getElementById("gem-export-status");
  if(el) el.textContent=msg;
}}

function _gemEnsureHtml2Canvas(cb){{
  if(typeof html2canvas!=="undefined"){{cb();return;}}
  var s=document.createElement("script");
  s.src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js";
  s.onload=cb;
  s.onerror=function(){{_gemSetStatus("✗ html2canvas failed");}};
  document.head.appendChild(s);
}}



function _gemSetStatus(msg){{
  var el=document.getElementById("gem-export-status");
  if(el) el.textContent=msg;
}}



// ── Export All — queue every timestep ────────────────────────────────────
var _gemExportAllQueue   = [];
var _gemExportAllRunning = false;

function gemExportAll(){{
  if(_gemExportAllRunning){{_gemSetStatus("Already running...");return;}}
  _gemExportAllRunning=true;
  _gemExportAllQueue=[];
  for(var i=0;i<_GEM_STEPS.length;i++) {{
    if(_GEM_STEPS[i].hour === 0) _gemExportAllQueue.push(i);
  }}
  var btn=document.getElementById("btn-export-all");
  if(btn) btn.classList.add("active");
  _gemSetStatus("Export All: 0/"+_GEM_STEPS.length);
  _gemEnsureHtml2Canvas(function(){{
    _gemRunExportQueue(0);
  }});
}}

function _gemRunExportQueue(done){{
  if(_gemExportAllQueue.length===0){{
    _gemExportAllRunning=false;
    var btn=document.getElementById("btn-export-all");
    if(btn) btn.classList.remove("active");
    _gemSetStatus("\u2713 All "+done+" saved!");
    setTimeout(function(){{_gemSetStatus("");}},5000);
    return;
  }}
  var idx=_gemExportAllQueue.shift();
  _gemStepIdx=idx;
  var slider=document.getElementById("gem-time-slider");
  if(slider) slider.value=String(idx);
  gemRender(idx);
  _gemSetStatus("Exporting "+(idx+1)+"/"+_GEM_STEPS.length+"...");

  var MAP=_getMap();
  var keys=Object.keys(window).filter(function(k){{return k.startsWith("map_");}});
  if(!keys.length){{setTimeout(function(){{_gemRunExportQueue(done);}},200);return;}}
  var mapEl=document.getElementById(keys[0])||document.querySelector(".leaflet-container");
  if(!mapEl){{setTimeout(function(){{_gemRunExportQueue(done);}},200);return;}}

  var hideEls=[
    document.getElementById("gem-bar"),
    document.getElementById("gem-banner"),
    document.getElementById("syn-fs-btn")
  ].filter(Boolean);
  var prevVis=hideEls.map(function(el){{return el.style.visibility;}});
  hideEls.forEach(function(el){{el.style.visibility="hidden";}});
  document.querySelectorAll(".leaflet-tooltip,.leaflet-popup,.leaflet-label").forEach(function(t){{t.style.display="none";}});

  var origW=mapEl.style.width,origH=mapEl.style.height;
  var TARGET_W=1400,TARGET_H=Math.round(1400*2400/2944);
  mapEl.style.width=TARGET_W+"px";
  mapEl.style.height=TARGET_H+"px";
  MAP.invalidateSize();
  MAP.setView([55.5, -118],5.6,{{animate:false}});

  setTimeout(function(){{
    html2canvas(mapEl,{{
      useCORS:true,allowTaint:true,scale:2,logging:false,
      width:TARGET_W,height:TARGET_H
    }}).then(function(canvas){{
      var cropH=canvas.height;
      var cropW=Math.min(Math.round(cropH*2944/2400),canvas.width);
      var BANNER_H=90,CREDIT_H=22;
      var out=document.createElement("canvas");
      out.width=cropW; out.height=cropH+BANNER_H+CREDIT_H;
      var ctx=out.getContext("2d");
      ctx.drawImage(canvas,0,0,cropW,cropH,0,0,cropW,cropH);

      // ── Run time top-left ─────────────────────────────────────────────────
      var _rdpsRun = {repr(_rdps_run_dt.strftime('%Y-%m-%d %HZ') if '_rdps_run_dt' in dir() else 'unknown')};
      var _gdpsRun = {repr(_gdps_run_dt.strftime('%Y-%m-%d %HZ') if '_gdps_run_dt' in dir() else 'unknown')};
      ctx.font         = "26px Arial, sans-serif";
      ctx.fillStyle    = "rgba(255,255,255,0.75)";
      ctx.textAlign    = "left";
      ctx.textBaseline = "top";
      ctx.fillText("RDPS run: " + _rdpsRun, 10, 10);
      ctx.fillText("GDPS run: " + _gdpsRun, 10, 38);
      ctx.font         = "26px Arial, sans-serif";
      ctx.fillStyle    = "#888888";
      ctx.fillText("RDPS run: " + _rdpsRun, 10, 10);
      ctx.fillText("GDPS run: " + _gdpsRun, 10, 38);

      var step=_GEM_STEPS[idx]||{{}};
      var key=step.key||"";
      var yr=parseInt(key.substring(0,4),10);
      var mo=parseInt(key.substring(4,6),10)-1;
      var dy=parseInt(key.substring(6,8),10);
      var hr=step.hour||0;
      var tsStr=_utcKeyToLocalStr(key);

      var blackW=Math.round(cropW*0.28);
      ctx.fillStyle="#ffffff"; ctx.fillRect(0,cropH,cropW,BANNER_H);
      ctx.fillStyle="#111111"; ctx.fillRect(0,cropH,blackW,BANNER_H);
      ctx.font="bold 42px Arial,sans-serif"; ctx.fillStyle="#ffffff";
      ctx.textAlign="center"; ctx.textBaseline="middle";
      ctx.fillText(tsStr,blackW/2,cropH+BANNER_H/2);
      ctx.font="bold 42px Arial,sans-serif"; ctx.fillStyle="#111111";
      ctx.textAlign="left"; ctx.textBaseline="middle";
      ctx.fillText("Surface & 12h Precipitation accumulation (noon to midnight)",blackW+24,cropH+BANNER_H/2);
      ctx.font="bold 38px Arial,sans-serif"; ctx.fillStyle="#333333";
      ctx.textAlign="right"; ctx.textBaseline="middle";
      ctx.fillText("AWCC Weather Office",cropW-24,cropH+BANNER_H/2);

      var creditY=cropH+BANNER_H;
      ctx.fillStyle="#f0f0f0"; ctx.fillRect(0,creditY,cropW,CREDIT_H);
      ctx.font="13px Arial,sans-serif"; ctx.fillStyle="#777777";
      ctx.textAlign="right"; ctx.textBaseline="middle";
      ctx.fillText("Based on data issued by Meteorological Service of Canada",cropW-14,creditY+CREDIT_H/2);
      var now=new Date();
      var expStr="Exported "+now.getUTCFullYear()+"-"+String(now.getUTCMonth()+1).padStart(2,"0")+"-"+String(now.getUTCDate()).padStart(2,"0")+" "+String(now.getUTCHours()).padStart(2,"0")+":"+String(now.getUTCMinutes()).padStart(2,"0")+"Z";
      ctx.textAlign="left";
      ctx.fillText(expStr,14,creditY+CREDIT_H/2);

      // ── QPF scale bar (above banner, reference-image style) ───────────
      var QPF_LEVELS=[0.6,1.5,3,5,10,20,30,40,50,60,80,100,120];
      var QPF_COLORS={{"0.6":"#c8f0a0","1.5":"#78d048","3":"#228b22","5":"#00aaaa","10":"#1a78c2","20":"#6a0dad","30":"#cc00cc","40":"#ffff00","50":"#ffaa00","60":"#ff4400","80":"#cc0000","100":"#880000","120":"#111111"}};
      var SWATCH_TOP=6, SWATCH_H=64, LABEL_GAP=3;
      var SCALE_H=SWATCH_TOP+SWATCH_H+LABEL_GAP+30+6;
      var scaleCanvas=document.createElement("canvas");
      scaleCanvas.width=cropW; scaleCanvas.height=SCALE_H;
      var sc=scaleCanvas.getContext("2d");

      var scaleBarW=Math.floor(cropW/3);
      var MARGIN_L=Math.floor(cropW/3), MARGIN_R=cropW-MARGIN_L-scaleBarW;
      var swatchArea=scaleBarW;
      var swatchW=Math.floor(swatchArea/QPF_LEVELS.length);
      QPF_LEVELS.forEach(function(lvl,i){{
        var x=MARGIN_L+i*swatchW;
        // swatch fill
        sc.fillStyle=QPF_COLORS[String(lvl)]||"#111111";
        sc.fillRect(x,SWATCH_TOP,swatchW,SWATCH_H);
        // swatch border
        sc.strokeStyle="#555555"; sc.lineWidth=1;
        sc.strokeRect(x+0.5,SWATCH_TOP+0.5,swatchW-1,SWATCH_H-1);
        // label left-aligned to swatch left edge
        sc.font="30px Arial,sans-serif";
        sc.fillStyle="#111111";
        sc.textAlign="left";
        sc.textBaseline="top";
        sc.fillText(String(lvl),x+2,SWATCH_TOP+SWATCH_H+LABEL_GAP);
      }});
      // "mm" label after last swatch
      var mmX=MARGIN_L+QPF_LEVELS.length*swatchW+5;
      sc.font="30px Arial,sans-serif";
      sc.font="11px Arial,sans-serif";
      sc.fillStyle="#111111";
      sc.textAlign="left";
      sc.textBaseline="top";
      sc.fillText("mm",mmX,SWATCH_TOP+SWATCH_H+LABEL_GAP);

      // Composite: map → scale bar → banner → credit (strict vertical order)
      var final=document.createElement("canvas");
      final.width=cropW; final.height=cropH+BANNER_H+CREDIT_H;
      var fc=final.getContext("2d");
      // 1. map pixels from the html2canvas capture
      fc.drawImage(canvas,0,0,cropW,cropH,0,0,cropW,cropH);

      // ── Run time top-left ─────────────────────────────────────────────────
      var _rdpsRun = {repr(_rdps_run_dt.strftime('%Y-%m-%d %HZ') if '_rdps_run_dt' in dir() else 'unknown')};
      var _gdpsRun = {repr(_gdps_run_dt.strftime('%Y-%m-%d %HZ') if '_gdps_run_dt' in dir() else 'unknown')};
      fc.font         = "26px Arial, sans-serif";
      fc.fillStyle    = "rgba(255,255,255,0.75)";
      fc.textAlign    = "left";
      fc.textBaseline = "top";
      fc.fillText("RDPS run: " + _rdpsRun, 10, 10);
      fc.fillText("GDPS run: " + _gdpsRun, 10, 38);
      fc.fillStyle    = "#888888";
      fc.fillText("RDPS run: " + _rdpsRun, 10, 10);
      fc.fillText("GDPS run: " + _gdpsRun, 10, 38);
      // 2. scale bar — floated bottom-left of map, above banner


      var scaleY=cropH-SCALE_H-8;
      var scaleX=Math.floor(cropW/3);
      // center horizontally: scaleBarW is 1/3 of cropW, centered = start at 1/3
      var scaleX=Math.floor((cropW-scaleBarW)/2);
      fc.drawImage(scaleCanvas,scaleX,scaleY);
      // 3. banner block (copy from `out` which has it at offset cropH)
      fc.drawImage(out,0,cropH,cropW,BANNER_H+CREDIT_H, 0,cropH,cropW,BANNER_H+CREDIT_H);

      var _firstKey=_GEM_STEPS[0].key;
      var _firstUTC=new Date(Date.UTC(parseInt(_firstKey.substring(0,4),10),parseInt(_firstKey.substring(4,6),10)-1,parseInt(_firstKey.substring(6,8),10)));
      var _validUTC=new Date(Date.UTC(yr,mo,dy));
      var _dayOffset=Math.round((_validUTC-_firstUTC)/(1000*60*60*24))+1;
      var fname="sfcDay"+_dayOffset+".png";
      var link=document.createElement("a");
      link.download=fname; link.href=final.toDataURL("image/png"); link.click();

      mapEl.style.width=origW; mapEl.style.height=origH;
      MAP.invalidateSize();
      hideEls.forEach(function(el,i){{el.style.visibility=prevVis[i];}});
      document.querySelectorAll(".leaflet-tooltip,.leaflet-popup,.leaflet-label").forEach(function(t){{t.style.display="";}});

      setTimeout(function(){{_gemRunExportQueue(done+1);}},1000);
    }}).catch(function(e){{
      console.error("Capture error:",e);
      mapEl.style.width=origW; mapEl.style.height=origH;
      MAP.invalidateSize();
      hideEls.forEach(function(el,i){{el.style.visibility=prevVis[i];}});
      setTimeout(function(){{_gemRunExportQueue(done);}},500);
    }});
  }},600);
}}

if(document.readyState==="complete"){{setTimeout(_gemInit,800);}}
else{{window.addEventListener("load",function(){{setTimeout(_gemInit,800);}});}}
</script>
'''
m.get_root().html.add_child(Element(_js))

_banner_html = f'''
<div id="gem-banner" style="
  position:fixed;top:0;left:0;right:0;z-index:10002;
  display:flex;align-items:stretch;height:56px;
  font-family:Arial,sans-serif;pointer-events:none;
  box-shadow:0 2px 8px rgba(0,0,0,0.4);
">
  <div style="
    background:#111111;color:#ffffff;
    display:flex;align-items:center;justify-content:center;
    padding:0 20px;min-width:260px;
    font-size:15px;font-weight:bold;line-height:1.3;text-align:center;
  ">
    <span id="gem-banner-time">—</span>
  </div>
  <div style="
    background:#ffffff;color:#111111;flex:1;
    display:flex;align-items:center;
    padding:0 20px;font-size:18px;font-weight:bold;
  ">
    GEM Surface — MSLP &amp; 12h Precipitation accumulation
  </div>
  <div style="
    background:#ffffff;color:#333333;
    display:flex;align-items:center;
    padding:0 20px;font-size:15px;font-weight:bold;
  ">
    AWCC Weather Office
  </div>
</div>
<div style="height:56px;"></div>
<script>
(function(){{
  function _updateBannerTime(){{
    var step = (_GEM_STEPS||[])[window._gemStepIdx||0];
    if(!step) return;
    var key  = step.key||"";
    var yr   = parseInt(key.substring(0,4),10);
    var mo   = parseInt(key.substring(4,6),10)-1;
    var dy   = parseInt(key.substring(6,8),10);
    var hr   = step.hour||0;
    var el = document.getElementById("gem-banner-time");
    if(el) el.textContent = _utcKeyToLocalStr(key);
  }}

  setTimeout(_updateBannerTime, 1000);
}})();
</script>
'''
m.get_root().html.add_child(Element(_banner_html))

os.makedirs('outputs', exist_ok=True)
_out_path = 'outputs/gem_surface_map.html'
m.save(_out_path)

print(f'\n✅ Cell UA-2d complete — map saved → {_out_path}')
