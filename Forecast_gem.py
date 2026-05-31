# -*- coding: utf-8 -*-
"""Forecast - GEM Upper Air + Sfc Map
Converted from Google Colab to standalone Python (GitHub Actions compatible).
"""

# ── Cell 1 . Install & import packages ────────────────────────────────────────
import csv, io, json, math, re, time, warnings
import asyncio
import concurrent.futures
import os
import sys
import subprocess
import tempfile
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import (gaussian_filter, maximum_filter,
                           minimum_filter, label)
import folium
import branca
from shapely.geometry import shape

print('✓ Core packages ready')

# ── Try optional heavy deps ────────────────────────────────────────────────────
try:
    import cfgrib
    import xarray as xr
    print('✓ cfgrib/xarray ready')
except ImportError:
    print('WARNING: cfgrib/xarray not available — GRIB fetch will fail')

try:
    from pykrige.ok import OrdinaryKriging
except ImportError:
    print('WARNING: pykrige not available')

# ── Cell 1.5 - Configuration ───────────────────────────────────────────────────

CSV_URL         = 'http://orangecore.net/met/wxchart/AP_location.csv'
METAR_API       = 'https://aviationweather.gov/api/data/metar'
COVERAGE        = 'essential'
EXPORT_TIME     = '1200Z'
INTERP_METHOD   = 'rbf'
SLP_INTERVAL    = 4
GRID_N          = 240
RBF_SMOOTHING   = 0.0
SIGMA_SMOOTH    = 1.0
SYMBOL_SCALE    = 28
FONT_SCALE      = 10
HL_NEIGHBORHOOD = 5
HL_MIN_DELTA    = 0.5
HL_SIGMA        = 1.0

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f'Coverage: {COVERAGE} | Interp: {INTERP_METHOD} | Grid: {GRID_N} | Export: {EXPORT_TIME}')
print(f'Output dir: {OUTPUT_DIR}')

# ── Fire zone KML ─────────────────────────────────────────────────────────────
import urllib.request, xml.etree.ElementTree as ET

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

try:
    with urllib.request.urlopen(_KML_URL, timeout=15) as _r:
        _kml_bytes = _r.read()
    _fire_zones_geojson_str = json.dumps(_kml_to_geojson(_kml_bytes))
    _zone_count = len(json.loads(_fire_zones_geojson_str)['features'])
    print(f'Alberta Fire Zone KML fetched → {_zone_count} zones loaded')
except Exception as e:
    print(f'WARNING: Fire zone KML fetch failed ({e}) — layer will be skipped')
    _fire_zones_geojson_str = '{"type":"FeatureCollection","features":[]}'

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
    '      }\n'
    '    });\n'
    '    fireLayer.addTo(MAP);\n'
    '  }\n'
    '  if (document.readyState === "complete") { setTimeout(loadFireZones, 800); }\n'
    '  else { window.addEventListener("load", function(){ setTimeout(loadFireZones, 800); }); }\n'
    '})();\n'
    '</script>\n'
)
print('fire_zones_html ready')

# ── Station model controls ─────────────────────────────────────────────────────
CIRCLE_RADIUS   = 0.05
BARB_STAFF_LEN  = 1.00
BARB_FULL_LEN   = 0.30
BARB_HALF_LEN   = 0.15
BARB_SPACING    = 0.10
BARB_LINE_WIDTH = 0.03
FEATHER_ANGLE   = 110
FEATHER_SIDE    = +1
FONT_SIZE_SCALE = 0.4
FONT_MIN_PX     = 7
LABEL_HORIZ_OFF = 0.12
LABEL_VERT_OFF  = 12
LABEL_ROW_GAP   = 0.9
CANVAS_PAD      = 1.5
CANVAS_H_FACTOR = 3.4

# ── Station model SVG functions ────────────────────────────────────────────────
def cloud_circle_svg(cx, cy, R, oktas):
    lw = max(0.9, R * 0.13)
    s = []
    if oktas == 9:
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
        hy = staff_tip_y + 0.28 * sl
        parts.append(
            f'<line x1="0" y1="{hy:.2f}" x2="{fx_half:.2f}" y2="{hy - tilt*0.5:.2f}" '
            f'stroke="black" stroke-width="{lw:.2f}" stroke-linecap="round"/>'
        )
    else:
        for _ in range(pn):
            ay  = staff_tip_y + pos
            by2 = staff_tip_y + pos + bspc * 2
            pts = f'0,{ay:.2f} {fx_full:.2f},{ay - tilt:.2f} 0,{by2:.2f}'
            parts.append(f'<polygon points="{pts}" fill="black"/>')
            pos += bspc * 1.5
        for _ in range(fu):
            fy = staff_tip_y + pos
            parts.append(
                f'<line x1="0" y1="{fy:.2f}" x2="{fx_full:.2f}" y2="{fy - tilt:.2f}" '
                f'stroke="black" stroke-width="{lw:.2f}" stroke-linecap="round"/>'
            )
            pos += bspc
        for _ in range(ha):
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
    _map = {
        'rising': 2, 'falling': 7, 'steady': 4,
        'rising_falling': 0, 'falling_rising': 5,
        'rising_steady': 1, 'falling_steady': 6,
    }
    if isinstance(tendency, str):
        tendency = _map.get(tendency.lower())
    if tendency is None:
        return ''
    lw  = max(0.9, S * 0.042)
    off = R + S * LABEL_HORIZ_OFF
    ox  = cx + off + fs * 2.2
    oy  = cy - R * 0.6 - LABEL_VERT_OFF + S * 0.65
    arm = S * 0.22
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
    PAD = S * CANVAS_PAD
    W   = S * 3 + PAD * 2
    H   = S * CANVAS_H_FACTOR + PAD * 2
    cx  = W / 2
    cy  = H / 2
    R   = S * CIRCLE_RADIUS
    fs  = max(FONT_MIN_PX, int(S * FONT_SIZE_SCALE))
    off = R + S * LABEL_HORIZ_OFF
    parts = []
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
    parts.append(wind_barb_svg(cx, cy, R,
                               d['wind_dir'], d['wind_spd'],
                               d.get('wind_gust', 0), S))
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
    if d['temp'] is not None:
        parts.append(txt(cx - off, cy - R * 0.6 - LABEL_VERT_OFF, str(d['temp'])))
    v  = d['vis']
    vs = (str(int(v))  if v is not None and v >= 10    else
          str(int(v))  if v is not None and v % 1 == 0 else
          f'{v:.1f}'   if v is not None                else None)
    wx = ' '.join(x for x in [vs, d['weather'] or None] if x)
    if wx:
        parts.append(txt(cx - off - 4, cy, wx))
    if d['dew'] is not None:
        parts.append(txt(cx - off, cy + R * 0.6 + LABEL_VERT_OFF, str(d['dew'])))
    if d['slp_label']:
        parts.append(txt(cx + off, cy - R * 0.6 - LABEL_VERT_OFF,
                         d['slp_label'], anchor='start'))
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
    if d['lowest_sig'] and d['lowest_sig']['height'] <= 120:
        _cb = math.ceil(d['lowest_sig']['height'] / 10)
        parts.append(txt(cx, cy + R + fs * LABEL_ROW_GAP,
                         str(_cb), anchor='middle'))
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
        'VFR':  '#22aa44', 'MVFR': '#2244cc',
        'IFR':  '#cc2222', 'LIFR': '#880088',
    }.get(d.get('flt_cat', ''), '#888888')


print('Station model SVG ready')

# ── Grid / model config ────────────────────────────────────────────────────────
GEM_GRID_DEG        = 2.0
SKIP_IF_RECENT      = False
UA_HOURS            = {12}
GEM_PRESSURE_LEVELS = [850, 500]
RDPS_FORECAST_DAYS  = 3
GDPS_FORECAST_DAYS  = 6

# ── Cell UA-2b. Fetch GEM upper-air + surface ──────────────────────────────────
import aiohttp
import xarray as xr
from scipy.spatial import cKDTree
from datetime import datetime, timezone as _tz, timedelta

GEM_LAT_MAX   =  80.0
GEM_LAT_MIN   =  30.0
GEM_LON_MIN   = -150.0
GEM_LON_MAX   =  -80.0

MAX_CONCURRENT = 3
TIMEOUT_S      = 60

GEM_LONGITUDE = [round(GEM_LON_MIN + i*GEM_GRID_DEG, 4)
                 for i in range(int(round((GEM_LON_MAX - GEM_LON_MIN) / GEM_GRID_DEG)) + 1)]
GEM_LATITUDES = [round(GEM_LAT_MAX - i*GEM_GRID_DEG, 4)
                 for i in range(int(round((GEM_LAT_MAX - GEM_LAT_MIN) / GEM_GRID_DEG)) + 1)]

_N_PTS = len(GEM_LATITUDES) * len(GEM_LONGITUDE)
print(f'Grid: {len(GEM_LATITUDES)} lats × {len(GEM_LONGITUDE)} lons = {_N_PTS:,} points')

_DD_BASE = 'https://dd.weather.gc.ca'

_VAR_MAP = {
    'AirTemp':            'TEMP',
    'GeopotentialHeight': 'HGHT',
    'RelativeHumidity':   'RELH',
    'WindU':              '_UGRD',
    'WindV':              '_VGRD',
}
_WXO_VARS = list(_VAR_MAP.keys())

_SFC_VAR_MAP = {
    'Pressure_MSL': 'MSLP',
    'Precip-Accum': '_PACC',
}
_SFC_VARS = list(_SFC_VAR_MAP.keys())

_NOW_UTC     = datetime.now(_tz.utc).replace(minute=0, second=0, microsecond=0)
_RDPS_CUTOFF = (_NOW_UTC + timedelta(days=RDPS_FORECAST_DAYS)).replace(
    hour=0, minute=0, second=0, microsecond=0
)

# ── Thermodynamic helpers ──────────────────────────────────────────────────────
def _icao(lat, lon, prefix='GDPS'):
    return f"{prefix}{abs(lat):05.1f}N{abs(lon):06.1f}"

def _is_valid(v):
    return v is not None and not (isinstance(v, float) and math.isnan(v)) and abs(v) < 99000

def _rh_to_dwpt(temp_c, rh_pct):
    if not (_is_valid(temp_c) and _is_valid(rh_pct)) or rh_pct <= 0:
        return None
    a, b  = 17.625, 243.04
    alpha = math.log(max(rh_pct, 0.1)/100.0) + (a*temp_c/(b+temp_c))
    return round(b*alpha/(a-alpha), 2)

def _dwpt_to_mixr(dwpt_c, pres_hpa):
    if not (_is_valid(dwpt_c) and _is_valid(pres_hpa)):
        return None
    e = 6.112 * math.exp(17.67*dwpt_c/(dwpt_c+243.5))
    return round(621.97*e/(pres_hpa-e), 3)

def _theta(temp_c, pres_hpa):
    if not (_is_valid(temp_c) and _is_valid(pres_hpa)):
        return None
    return round((temp_c+273.15)*(1000.0/pres_hpa)**0.2854, 2)

def _theta_e(temp_c, dwpt_c, pres_hpa):
    if any(not _is_valid(v) for v in [temp_c, dwpt_c, pres_hpa]):
        return None
    tk = temp_c+273.15; td = dwpt_c+273.15
    e  = 6.112*math.exp(17.67*dwpt_c/(dwpt_c+243.5))
    r  = 0.622*e/(pres_hpa-e)
    tlc = 1.0/(1.0/(td-56.0)+math.log(tk/td)/800.0)+56.0
    return round(tk*(1000.0/pres_hpa)**(0.2854*(1-0.28e-3*r*1000))
                 *math.exp((3376.0/tlc-2.54)*r*(1+0.81e-3*r*1000)), 2)

def _theta_v(temp_c, mixr_gkg, pres_hpa):
    if any(not _is_valid(v) for v in [temp_c, mixr_gkg, pres_hpa]):
        return None
    tk = temp_c+273.15; r = mixr_gkg/1000.0
    return round(tk*(1000.0/pres_hpa)**0.2854*(1+1.608*r)/(1+r), 2)

def _uv_to_spd_dir(u, v):
    if u is None or v is None:
        return None, None
    drct = round((math.degrees(math.atan2(-u, -v))+360) % 360, 1)
    sped = round(math.sqrt(u**2+v**2)/0.51444, 1)
    return drct, sped

# ── URL builders ──────────────────────────────────────────────────────────────
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

# ── GRIB2 point extraction ─────────────────────────────────────────────────────
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

    units   = da.attrs.get('GRIB_units', '') or da.attrs.get('units', '')
    is_temp = 'Temp' in var_name
    data    = da.values

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
        if math.isnan(val) or abs(val) > 1e8:
            result[(tlat, tlon)] = None
        else:
            result[(tlat, tlon)] = round(float(val), 2) if is_temp else float(val)

    return result


# ── Storage for fetched data ───────────────────────────────────────────────────
_point_data = {}
_sfc_data   = {}

# ── Async fetch helpers ────────────────────────────────────────────────────────
async def _check_run_exists(run_dt, is_rdps=True):
    fxx  = 24
    pres = GEM_PRESSURE_LEVELS[0]
    url  = _rdps_url(run_dt, fxx, 'AirTemp', pres) if is_rdps else _gdps_url(run_dt, fxx, 'AirTemp', pres)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.head(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
                return r.status == 200
    except:
        return False

async def _latest_run_verified(run_hours, min_age_h, is_rdps=True):
    now = datetime.now(_tz.utc)
    candidates = []
    for h in run_hours:
        for day_offset in [0, 1]:
            cand = (now - timedelta(days=day_offset)).replace(
                hour=h, minute=0, second=0, microsecond=0)
            age_h = (now - cand).total_seconds() / 3600
            if age_h >= min_age_h:
                candidates.append(cand)
    for run_dt in sorted(candidates, reverse=True):
        if await _check_run_exists(run_dt, is_rdps):
            print(f'  ✓ Verified run: {run_dt.strftime("%Y-%m-%d %HZ")}')
            return run_dt
        else:
            print(f'  ✗ Not available: {run_dt.strftime("%Y-%m-%d %HZ")}, trying previous...')
    raise RuntimeError('No available model run found on server')


async def _fetch_and_extract_all(tasks, sfc_tasks, target_lats, target_lons):
    sem    = asyncio.Semaphore(MAX_CONCURRENT)
    errors = []

    async def _worker_isob(model, url, var_name, pres, vt):
        vt_str = vt.strftime('%Y-%m-%d') + f' {vt.hour:02d}Z'
        col    = _VAR_MAP[var_name]
        async with sem:
            try:
                async with session.get(
                        url, timeout=aiohttp.ClientTimeout(total=TIMEOUT_S)) as r:
                    raw    = await r.read() if r.status == 200 else None
                    status = r.status
            except Exception as e:
                errors.append((url, str(e))); return
        if raw is None:
            errors.append((url, f'HTTP {status}')); return
        if len(raw) == 0:
            errors.append((url, 'HTTP 200 but zero bytes')); return
        try:
            extracted = _extract_points_grib(raw, target_lats, target_lons, var_name)
        except Exception as e:
            errors.append((url, str(e))); return
        for (lat, lon), val in extracted.items():
            key = (lat, lon, vt_str, float(pres))
            if key not in _point_data:
                _point_data[key] = {}
            _point_data[key][col] = val
        print(f'  ✓ {model} {vt_str}  {var_name}@{pres}hPa  ({len(extracted)} pts)')

    async def _worker_sfc(model, url, var_name, vt, col_suffix):
        vt_str   = vt.strftime('%Y-%m-%d') + f' {vt.hour:02d}Z'
        base_col = _SFC_VAR_MAP[var_name]
        col      = base_col + col_suffix
        async with sem:
            try:
                async with session.get(
                        url, timeout=aiohttp.ClientTimeout(total=TIMEOUT_S)) as r:
                    raw    = await r.read() if r.status == 200 else None
                    status = r.status
            except Exception as e:
                errors.append((url, str(e))); return
        if raw is None:
            errors.append((url, f'HTTP {status}')); return
        if len(raw) == 0:
            errors.append((url, 'HTTP 200 but zero bytes')); return
        try:
            extracted = _extract_points_grib(raw, target_lats, target_lons, var_name)
        except Exception as e:
            errors.append((url, str(e))); return
        for (lat, lon), val in extracted.items():
            key = (lat, lon, vt_str)
            if key not in _sfc_data:
                _sfc_data[key] = {}
            _sfc_data[key][col] = val
        tag = 'PRIOR' if col_suffix else 'TARGET'
        print(f'  ✓ {model} {vt_str}  {var_name} [{tag}]  ({len(extracted)} pts)')

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        await asyncio.gather(
            *[_worker_isob(*t)  for t in tasks],
            *[_worker_sfc(*t)   for t in sfc_tasks],
        )

    return errors


async def run_fetch():
    """Top-level async entry point — replaces Colab's top-level await."""
    global _rdps_run_dt, _gdps_run_dt

    _rdps_run_dt = await _latest_run_verified([0, 6, 12, 18], min_age_h=2.5, is_rdps=True)
    _gdps_run_dt = await _latest_run_verified([0, 12],         min_age_h=6.0, is_rdps=False)

    _MDT_OFFSET   = timedelta(hours=6)
    _base_day_utc = _NOW_UTC.replace(hour=0, minute=0, second=0, microsecond=0)
    _base_day_mdt = (_NOW_UTC - _MDT_OFFSET).replace(
        hour=0, minute=0, second=0, microsecond=0
    ).replace(tzinfo=_tz.utc)

    _target_vts = []
    for _d in range(GDPS_FORECAST_DAYS + 1):
        for _h in sorted(UA_HOURS):
            _vt = _base_day_utc + timedelta(days=_d, hours=_h)
            _target_vts.append(_vt)

    def _fxx(run_dt, valid_dt):
        return int((valid_dt - run_dt).total_seconds() / 3600)

    target_lats = [lat for lat in GEM_LATITUDES for _   in GEM_LONGITUDE]
    target_lons = [lon for _   in GEM_LATITUDES for lon in GEM_LONGITUDE]

    # Build task lists
    tasks = []
    for vt in _target_vts:
        use_rdps = _NOW_UTC < vt < _RDPS_CUTOFF
        for pres in GEM_PRESSURE_LEVELS:
            for var_name in _WXO_VARS:
                if use_rdps:
                    fxx = _fxx(_rdps_run_dt, vt)
                    if 0 <= fxx <= 84:
                        tasks.append(('RDPS', _rdps_url(_rdps_run_dt, fxx, var_name, pres),
                                      var_name, pres, vt))
                else:
                    fxx = _fxx(_gdps_run_dt, vt)
                    if 0 <= fxx <= 240:
                        tasks.append(('GDPS', _gdps_url(_gdps_run_dt, fxx, var_name, pres),
                                      var_name, pres, vt))

    _sfc_target_vts = []
    for _d in range(1, GDPS_FORECAST_DAYS + 2):
        _vt = _base_day_mdt + timedelta(days=_d, hours=6)
        _sfc_target_vts.append(_vt)

    sfc_tasks = []
    for vt in _sfc_target_vts:
        use_rdps  = _NOW_UTC < vt < _RDPS_CUTOFF
        run_dt    = _rdps_run_dt if use_rdps else _gdps_run_dt
        url_fn    = _rdps_sfc_url if use_rdps else _gdps_sfc_url
        max_fxx   = 84 if use_rdps else 240
        fxx_tgt   = _fxx(run_dt, vt)
        fxx_prior = fxx_tgt - 12
        if not (0 <= fxx_tgt <= max_fxx):
            continue
        vt_mslp = vt - timedelta(hours=6)
        vt_pacc = vt
        for var_name in _SFC_VARS:
            if var_name == 'Pressure_MSL':
                fxx_mslp = _fxx(run_dt, vt_mslp)
                if 0 <= fxx_mslp <= max_fxx:
                    sfc_tasks.append((
                        'RDPS' if use_rdps else 'GDPS',
                        url_fn(run_dt, fxx_mslp, var_name),
                        var_name, vt_mslp, ''
                    ))
            elif var_name == 'Precip-Accum':
                sfc_tasks.append((
                    'RDPS' if use_rdps else 'GDPS',
                    url_fn(run_dt, fxx_tgt, var_name),
                    var_name, vt_pacc, ''
                ))
                if fxx_prior >= 0:
                    sfc_tasks.append((
                        'RDPS' if use_rdps else 'GDPS',
                        url_fn(run_dt, fxx_prior, var_name),
                        var_name, vt_pacc, '_PRIOR'
                    ))

    print(f'RDPS run   : {_rdps_run_dt.strftime("%Y-%m-%d %HZ")}')
    print(f'GDPS run   : {_gdps_run_dt.strftime("%Y-%m-%d %HZ")}')
    print(f'Isobaric tasks: {len(tasks)}')
    print(f'Surface tasks : {len(sfc_tasks)}')

    errors = await _fetch_and_extract_all(tasks, sfc_tasks, target_lats, target_lons)

    return _target_vts, _sfc_target_vts, target_lats, target_lons, tasks, sfc_tasks, errors


# ── H/L detection ─────────────────────────────────────────────────────────────
def find_hl_centers(grid, lon_vec, lat_vec, neighborhood=20, min_delta=2.0):
    from scipy.ndimage import gaussian_filter as _gf
    sg    = _gf(grid, sigma=HL_SIGMA)
    max_f = maximum_filter(sg, size=neighborhood)
    min_f = minimum_filter(sg, size=neighborhood)
    is_max = (np.abs(sg - max_f) < 1e-6) & (sg - min_f > min_delta)
    is_min = (np.abs(sg - min_f) < 1e-6) & (max_f - sg > min_delta)
    edge = max(3, int(min(grid.shape) * 0.10))
    centers = []
    for typ, mask in [('H', is_max), ('L', is_min)]:
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
            if r < edge or r > grid.shape[0] - edge: continue
            if c < edge or c > grid.shape[1] - edge: continue
            _grid_val = float(grid[r, c])
            centers.append(dict(
                type=typ,
                lat=float(lat_vec[r]), lon=float(lon_vec[c]),
                val=float(_grid_val)
            ))
    return centers


# ── Grid builder ──────────────────────────────────────────────────────────────
def _gem_build_grid(df, field, N=220, pad=1.5,
                    rbf_smoothing=0.05, sigma=1.5,
                    lon_vec=None, lat_vec=None):
    sub = df[['lat', 'lon', field]].dropna(subset=[field])
    if len(sub) < 8:
        return None, None, None, None, None
    lats = sub['lat'].values
    lons = sub['lon'].values
    vals = sub[field].values.astype(float)
    if lon_vec is None:
        lon_vec = np.linspace(lons.min() - pad, lons.max() + pad, N)
    if lat_vec is None:
        lat_vec = np.linspace(lats.min() - pad, lats.max() + pad, N)
    glon, glat = np.meshgrid(lon_vec, lat_vec)
    obs_xy = np.column_stack([lons, lats])
    try:
        rbf = RBFInterpolator(
            obs_xy, vals,
            kernel='thin_plate_spline',
            smoothing=max(rbf_smoothing * len(sub), 1e-6)
        )
    except np.linalg.LinAlgError:
        rbf = RBFInterpolator(
            obs_xy, vals,
            kernel='linear',
            smoothing=max(rbf_smoothing * len(sub), 1.0)
        )
    grid = rbf(np.column_stack([glon.ravel(), glat.ravel()])).reshape(N, N)
    if sigma > 0:
        grid = gaussian_filter(grid, sigma=sigma)
    return grid, lon_vec, lat_vec, lons, lats


# ── Normal temperature bands ────────────────────────────────────────────────────
from datetime import date as _date

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
_STATIC_GREEN_LO = -6

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

def _make_ua_temp_bands(pressure_level, today=None):
    if today is None:
        today = _date.today()
    normal_hi, normal_lo = _get_normal_band(pressure_level, today)
    shift = normal_lo - _STATIC_GREEN_LO
    return [(bhi + shift, blo + shift, col)
            for (bhi, blo, col) in _UA_TEMP_BANDS_BASE]

_TODAY = _date.today()
_normal_850_hi, _normal_850_lo = _get_normal_band(850, _TODAY)
_normal_500_hi, _normal_500_lo = _get_normal_band(500, _TODAY)
UA_TEMP_BANDS_850 = _make_ua_temp_bands(850, _TODAY)
UA_TEMP_BANDS_500 = _make_ua_temp_bands(500, _TODAY)
UA_TEMP_BANDS     = UA_TEMP_BANDS_850

print(f'850 hPa normal: {_normal_850_lo} to {_normal_850_hi} °C')
print(f'500 hPa normal: {_normal_500_lo} to {_normal_500_hi} °C')

# ── UA processing config ────────────────────────────────────────────────────────
_GRID_N  = 200
_MAX_PTS = 1000
_SIGMA   = {'HGHT': 2.0, 'TEMP': 2.0, 'TTDP': 10, 'SPED': 10}
_INTERVALS = {'HGHT': 6.0, 'TEMP': 2.0, 'TTDP': 2.0, 'SPED': 5.0}
sigmaT700500 = 5.0
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

MSLP_RBF_SMOOTHING = 0.05
MSLP_SIGMA         = 0.5
QPF_RBF_SMOOTHING  = 0.001
QPF_SIGMA          = 0.5
MSLP_INTERVAL      = 4.0
QPF_INTERVAL       = 1.0

SURFACE_STN_SPACING_KM = 500
UA_STN_SPACING_KM      = 1000
SHOW_STATION_SYMBOLS   = True
SHOW_TOOLTIPS          = False

# ── Key height lookup ──────────────────────────────────────────────────────────
_HEIGHT_CONTROL = {
    "Jan 1":  5400, "Apr 3":  5460, "Apr 19": 5520, "May 11": 5580,
    "May 30": 5640, "Jun 27": 5700, "Jul 26": 5760, "Aug 7":  5700,
    "Aug 31": 5640, "Oct 1":  5580, "Oct 17": 5520, "Oct 29": 5460,
    "Nov 17": 5400,
}

def _get_key_hgt_500(today=None):
    if today is None:
        today = _date.today()
    today_doy = today.timetuple().tm_yday
    best_val, best_doy = None, -1
    for label_str, hgt in _HEIGHT_CONTROL.items():
        entry_doy = datetime.strptime(f"{label_str} 2001", "%b %d %Y").timetuple().tm_yday
        if entry_doy <= today_doy and entry_doy > best_doy:
            best_doy = entry_doy
            best_val = hgt
    if best_val is None:
        best_val = list(_HEIGHT_CONTROL.values())[-1]
    return best_val

KEY_HGT_500 = _get_key_hgt_500(_TODAY)
KEY_HGT_850 = KEY_HGT_700 = KEY_HGT_250 = 0
print(f'500 hPa key height: {KEY_HGT_500} m')


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


# ── Assemble gem_ua_df from fetched point data ─────────────────────────────────
def assemble_gem_ua_df(target_vts, sfc_target_vts, target_lats, target_lons):
    _COLS = ['icao','wmo','stn_name','lat','lon','valid_time','hour',
             'PRES','HGHT','TEMP','DWPT','RELH','MIXR','DRCT','SPED',
             'THTA','THTE','THTV','MSLP','QPF12H']

    _target_vt_strs = (
        {vt.strftime('%Y-%m-%d') + f' {vt.hour:02d}Z' for vt in target_vts} |
        {(vt - timedelta(hours=6)).strftime('%Y-%m-%d') + f' {(vt - timedelta(hours=6)).hour:02d}Z'
         for vt in sfc_target_vts}
    )

    rows = []
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
            'icao': _icao(lat, lon, prefix), 'wmo': None,
            'stn_name': f'{prefix} {lat:+.1f}N {abs(lon):.1f}W',
            'lat': float(lat), 'lon': float(lon),
            'valid_time': vt_str, 'hour': int(vt.hour),
            'PRES': float(pres), 'HGHT': hght, 'TEMP': temp,
            'DWPT': dwpt, 'RELH': rh, 'MIXR': mixr,
            'DRCT': drct, 'SPED': sped,
            'THTA': _theta(temp, pres), 'THTE': _theta_e(temp, dwpt, pres),
            'THTV': _theta_v(temp, mixr, pres),
            'MSLP': None, 'QPF12H': None, '_model': prefix,
        })

    # Merge surface data
    _sfc_merged = {}
    for (lat, lon, vt_str), fields in _sfc_data.items():
        vt_key = datetime.strptime(vt_str, '%Y-%m-%d %HZ').replace(tzinfo=_tz.utc)
        if vt_key.hour == 6:
            vt_label = (vt_key - timedelta(hours=6)).strftime('%Y-%m-%d') + f' {(vt_key.hour - 6):02d}Z'
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
                if mslp_raw is not None and not math.isnan(float(mslp_raw))
                else None)
        pacc       = fields.get('_PACC')
        pacc_prior = fields.get('_PACC_PRIOR', 0.0) or 0.0
        if _is_valid(pacc):
            qpf12h = round(max(0.0, float(pacc) - float(pacc_prior)), 2)
        else:
            qpf12h = None
        rows.append({
            'icao': _icao(lat, lon, prefix), 'wmo': None,
            'stn_name': f'{prefix} {lat:+.1f}N {abs(lon):.1f}W',
            'lat': float(lat), 'lon': float(lon),
            'valid_time': vt_str, 'hour': int(vt.hour),
            'PRES': 0.0, 'HGHT': None, 'TEMP': None, 'DWPT': None,
            'RELH': None, 'MIXR': None, 'DRCT': None, 'SPED': None,
            'THTA': None, 'THTE': None, 'THTV': None,
            'MSLP': mslp, 'QPF12H': qpf12h, '_model': prefix,
        })

    gem_ua_df = (pd.DataFrame(rows)[_COLS + ['_model']]
                 if rows else pd.DataFrame(columns=_COLS + ['_model']))
    return gem_ua_df


# ── Build synoptic map HTML ────────────────────────────────────────────────────
def build_synoptic_map(gem_ua_df, ua_summary_df, metar_records, synoptic_times,
                       ts_ua_json_str, ts_ua_stn_json_str,
                       sfc_keys, slp_grids, qpf_grids, lon_vecs, lat_vecs,
                       hl_centers_by_key, rdps_run_dt, gdps_run_dt):
    """Build and save both synoptic HTML maps."""
    # ── Synoptic upper-air map ────────────────────────────────────────────────
    center_lat = 53.3097
    center_lon = -113.5797

    m = folium.Map(location=[center_lat, center_lon], zoom_start=5,
                   tiles=None, prefer_canvas=True)
    folium.TileLayer(tiles='about:blank', attr=' ', name='Blank',
                     max_zoom=19, show=True).add_to(m)
    m.get_root().html.add_child(folium.Element(
        '<style>.leaflet-container{background:#e0f2ff!important;}</style>'
    ))

    # borders JS (same as original)
    borders_js = _build_borders_js()
    m.get_root().html.add_child(folium.Element(borders_js))

    if fire_zones_html:
        m.get_root().html.add_child(folium.Element(fire_zones_html))

    # Build time steps
    _MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    _DOWS   = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    _available_hours = sorted(set(int(_hr) for _, _hr in synoptic_times))
    _use_hour = _available_hours[0]

    time_steps = []
    for (_date_val, _hr) in synoptic_times:
        if int(_hr) != _use_hour:
            continue
        _date_str = pd.Timestamp(_date_val).strftime('%Y%m%d')
        _key      = f'{_date_str}_{int(_hr):02d}'
        _dt       = pd.Timestamp(_date_val).date()
        _dow      = _DOWS[_dt.weekday()]
        _mon      = _MONTHS[_dt.month - 1]
        _label    = f'{_dow} {_mon} {_dt.day} {int(_hr):02d}Z'
        time_steps.append({'key': _key, 'label': _label, 'hour': int(_hr)})

    # Build station data
    ts_data = _build_surface_ts_data(metar_records)
    ts_json_str = json.dumps(ts_data)
    ts_list_str = json.dumps(sorted(ts_data.keys()))

    ua_stn_json_str = _build_ua_stn_data(ua_summary_df)

    # Control bar + JS
    _bar_html = _build_control_bar()
    m.get_root().html.add_child(folium.Element(_bar_html))

    _js_code = _build_main_js(
        json.dumps(time_steps), ua_stn_json_str, ts_ua_json_str,
        KEY_HGT_850, KEY_HGT_700, KEY_HGT_500, KEY_HGT_250,
        SHOW_STATION_SYMBOLS, SHOW_TOOLTIPS
    )
    m.get_root().html.add_child(folium.Element(_js_code))

    ua_out_path = os.path.join(OUTPUT_DIR, 'synoptic_map.html')
    m.save(ua_out_path)
    print(f'✅ Upper-air map saved → {ua_out_path}')

    # ── Surface MSLP/QPF map ──────────────────────────────────────────────────
    _build_surface_map(sfc_keys, slp_grids, qpf_grids, lon_vecs, lat_vecs,
                       hl_centers_by_key, rdps_run_dt, gdps_run_dt)


def _build_borders_js():
    return (
        '<script>\n'
        '(function(){\n'
        '  function loadBorders(){\n'
        '    var keys=Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
        '    if(!keys.length){setTimeout(loadBorders,200);return;}\n'
        '    var MAP=window[keys[0]];\n'
        '    if(!MAP.getPane("landPane")){\n'
        '      MAP.createPane("landPane");\n'
        '      MAP.getPane("landPane").style.zIndex="205";\n'
        '      MAP.getPane("landPane").style.pointerEvents="none";\n'
        '    }\n'
        '    if(!MAP.getPane("bordersPane")){\n'
        '      MAP.createPane("bordersPane");\n'
        '      MAP.getPane("bordersPane").style.zIndex="220";\n'
        '      MAP.getPane("bordersPane").style.pointerEvents="none";\n'
        '    }\n'
        '    if(!MAP.getPane("heightPane")){\n'
        '      MAP.createPane("heightPane");\n'
        '      MAP.getPane("heightPane").style.zIndex="490";\n'
        '      MAP.getPane("heightPane").style.pointerEvents="none";\n'
        '    }\n'
        '    fetch("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_land.geojson")\n'
        '      .then(function(r){return r.json();})\n'
        '      .then(function(gj){\n'
        '        L.geoJSON(gj,{style:function(){return {color:"none",weight:0,fill:true,fillColor:"#e8e8e8",fillOpacity:1.0};},pane:"landPane"}).addTo(MAP);\n'
        '      });\n'
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
        '      });\n'
        '    });\n'
        '    fetch("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_1_states_provinces.geojson")\n'
        '      .then(function(r){return r.json();})\n'
        '      .then(function(gj){\n'
        '        var ab={type:"FeatureCollection",features:gj.features.filter(function(f){return f.properties.name==="Alberta";})};\n'
        '        L.geoJSON(ab,{style:function(){return {color:"#444444",weight:2.5,opacity:1.0,fill:true,fillColor:"#ffffff",fillOpacity:1.0};},pane:"bordersPane"}).addTo(MAP);\n'
        '      });\n'
        '  }\n'
        '  if(document.readyState==="complete"){setTimeout(loadBorders,600);}\n'
        '  else{window.addEventListener("load",function(){setTimeout(loadBorders,600);});}\n'
        '})();\n'
        '</script>'
    )


def _build_surface_ts_data(metar_records):
    ts_data = {}
    _ts_all = sorted(set(d['timestamp'] for d in metar_records if d.get('timestamp')))
    for _ts in _ts_all:
        _entries = []
        _display_set = {id(d) for d in _decimate_stations(
            [d for d in metar_records if d['timestamp'] == _ts],
            spacing_km=SURFACE_STN_SPACING_KM)}
        for _d in metar_records:
            if _d['timestamp'] != _ts: continue
            if id(_d) not in _display_set: continue
            _svg_str, _sw, _sh = station_model_svg(
                {**_d, 'is_surface': True, 'slp_label': '', 'lowest_sig': None}, S=34)
            _entries.append({
                'lat': _d['lat'], 'lon': _d['lon'],
                'popup': f'<b>{_d["icao"]}</b> SLP:{_d.get("slp")} hPa',
                'tip': f'{_d["icao"]}',
                'svg': _svg_str, 'svg_w': int(_sw), 'svg_h': int(_sh),
                'ttd': None, 'tend_color': None,
            })
        ts_data[_ts] = _entries
    return ts_data


def _build_ua_stn_data(ua_summary_df):
    _ua_stn_data = {}
    for (date_val, hr), _grp in ua_summary_df.groupby(
            [ua_summary_df['valid_time'].str[:10], 'hour'], sort=True):
        _date_str = pd.Timestamp(date_val).strftime('%Y%m%d')
        _key      = f'{_date_str}_{int(hr):02d}'
        _stns     = []
        dlat = UA_STN_SPACING_KM / 111.32
        _grp['_cell'] = _grp.apply(
            lambda r: (int(r['lat'] / dlat),
                       int(r['lon'] / (UA_STN_SPACING_KM / (111.32 * np.cos(np.radians(r['lat'])))))),
            axis=1)
        _decimated = _grp.drop_duplicates(subset='_cell')
        for _, _r in _decimated.iterrows():
            _level_svgs = {}
            for _lvl in [850, 500]:
                _lwd = _r.get(f'DRCT_{_lvl}')
                _lws = _r.get(f'SPED_{_lvl}')
                _lh  = _r.get(f'HGHT_{_lvl}')
                _lh_label = ''
                if _lh is not None and not (isinstance(_lh, float) and math.isnan(_lh)):
                    _lh_label = str(int(round(_lh / 10)))[1:]
                _ua_d = {
                    'icao': str(_r['icao']), 'temp': None, 'dew': None,
                    'wind_dir': int(_lwd) if _lwd is not None and not (isinstance(_lwd, float) and math.isnan(_lwd)) else None,
                    'wind_spd': _lws, 'wind_gust': 0,
                    'vis': None, 'weather': '', 'slp_label': _lh_label,
                    'oktas': 8, 'has_sky_obs': True, 'clouds': [], 'lowest_sig': None,
                    'ceiling': 99999, 'flt_cat': 'VFR',
                    'lat': 0, 'lon': 0, 'timestamp': '', 'rh': 0,
                    'tendency': None, 'pressure_change': None,
                }
                _svg_str, _sw, _sh = station_model_svg(_ua_d, S=34)
                _level_svgs[str(_lvl)] = {'svg': _svg_str, 'w': int(_sw), 'h': int(_sh)}
            _stns.append({
                'lat': float(_r['lat']), 'lon': float(_r['lon']),
                'icao': str(_r['icao']), 'name': str(_r['stn_name']),
                'popup': f'<b>{_r["icao"]}</b>', 'tip': str(_r['icao']),
                'svgs': _level_svgs,
            })
        _ua_stn_data[_key] = _stns
    return json.dumps(_ua_stn_data)


def _build_control_bar():
    return '''
<style>
#syn-bar{position:fixed;bottom:0;left:0;right:0;z-index:10000;background:#1a1a2e;
  border-top:2px solid #4a7fc1;padding:8px 16px;display:flex;align-items:center;
  gap:14px;font-family:"Courier New",monospace;font-size:11px;color:#e0e0e0;
  box-shadow:0 -3px 12px rgba(0,0,0,0.5);min-height:52px;}
.syn-lvl-btn{font-size:12px;padding:4px 14px;cursor:pointer;border:1px solid #3a4a6a;
  border-radius:4px;background:#2a2a4a;color:#c0c8e0;font-family:"Courier New",monospace;
  font-weight:bold;}
.syn-lvl-btn:hover{background:#3a4a7a;}
.syn-lvl-btn.active{background:#4a7fc1;color:#fff;border-color:#6a9fe1;}
#syn-time-slider{width:320px;accent-color:#4a7fc1;cursor:pointer;}
#syn-ts-label{color:#c0d0ff;font-size:11px;min-width:200px;}
</style>
<div id="syn-bar">
  <div style="display:flex;align-items:center;gap:6px;border-right:1px solid #3a3a5a;padding-right:14px;">
    <span style="font-size:8px;color:#8888aa;font-weight:bold;text-transform:uppercase;">Level</span>
    <button class="syn-lvl-btn active" id="btn-850" onclick="synSetLevel(\'850\')">850 hPa</button>
    <button class="syn-lvl-btn"        id="btn-500" onclick="synSetLevel(\'500\')">500 hPa</button>
  </div>
  <div style="display:flex;align-items:center;gap:6px;border-right:1px solid #3a3a5a;padding-right:14px;">
    <span style="font-size:8px;color:#8888aa;font-weight:bold;text-transform:uppercase;">Time</span>
    <input type="range" id="syn-time-slider" min="0" value="0" oninput="synSliderChange(this.value)">
  </div>
  <div style="display:flex;align-items:center;gap:6px;">
    <span id="syn-ts-label">—</span>
  </div>
</div>
'''


def _build_main_js(time_steps_str, ua_stn_json_str, ts_ua_json_str,
                   khgt_850, khgt_700, khgt_500, khgt_250,
                   show_stns, show_tooltips):
    return f'''
<script>
var _SYN_TIME_STEPS = {time_steps_str};
var _SYN_UA_STNS    = {ua_stn_json_str};
var _SYN_UA         = {ts_ua_json_str};
var KEY_HGT_DAM     = {{"850":{int(khgt_850/10)},"700":{int(khgt_700/10)},"500":{int(khgt_500/10)},"250":{int(khgt_250/10)}}};
var KEY_HGT_M       = {{"850":{int(khgt_850)},"700":{int(khgt_700)},"500":{int(khgt_500)},"250":{int(khgt_250)}}};
var _synLevel       = "850";
var _synStepIdx     = 0;
var _synUALayer     = null;
var _synStnLayer    = null;
var _synShowStations = {'true' if show_stns else 'false'};
var _synShowTooltips = {'true' if show_tooltips else 'false'};

function _getMap(){{
  var k=Object.keys(window).filter(function(k){{return k.startsWith("map_");}});
  return k.length?window[k[0]]:null;
}}
function synSetLevel(lvl){{
  _synLevel=lvl;
  ["850","500"].forEach(function(l){{
    var b=document.getElementById("btn-"+l);
    if(b) b.classList.toggle("active",l===lvl);
  }});
  synRender();
}}
function synSliderChange(v){{_synStepIdx=parseInt(v);synRender();}}
function synRender(){{
  var MAP=_getMap(); if(!MAP) return;
  var step=_SYN_TIME_STEPS[_synStepIdx]; if(!step) return;
  var lbl=document.getElementById("syn-ts-label");
  if(lbl) lbl.textContent=step.label;
  synRenderUA(step.key,step.label);
}}
function synRenderUA(fullKey,stepLabel){{
  var MAP=_getMap(); if(!MAP) return;
  if(_synUALayer){{MAP.removeLayer(_synUALayer);_synUALayer=null;}}
  if(_synStnLayer){{MAP.removeLayer(_synStnLayer);_synStnLayer=null;}}
  if(!fullKey||!_synLevel) return;
  var uaData=(_SYN_UA[fullKey]||{{levels:{{}}}}).levels[_synLevel]||{{}};
  _synUALayer=L.layerGroup();
  (uaData.hght||[]).forEach(function(ct){{
    var ll=ct.coords.map(function(c){{return [c[1],c[0]];}});
    var isKey=(KEY_HGT_DAM[_synLevel]&&(Math.round(ct.level)===KEY_HGT_DAM[_synLevel]||Math.round(ct.level)===KEY_HGT_M[_synLevel]));
    L.polyline(ll,{{color:"#000000",weight:isKey?4.5:1.5,opacity:isKey?1.0:0.85,pane:"heightPane"}}).addTo(_synUALayer);
  }});
  (uaData.temp||[]).forEach(function(ct){{
    var t=ct.level;
    var col=t>0?"rgb("+(Math.round(180+75*Math.min(t/40,1)))+",0,0)":t<0?"rgb(0,0,"+(Math.round(180+75*Math.min(Math.abs(t)/40,1)))+")":"#00bb00";
    var ll=ct.coords.map(function(c){{return [c[1],c[0]];}});
    L.polyline(ll,{{color:col,weight:0.8,opacity:0.8,dashArray:"6 4"}}).addTo(_synUALayer);
  }});
  _synUALayer.addTo(MAP);
  var stns=_SYN_UA_STNS[fullKey]||[];
  _synStnLayer=L.layerGroup();
  stns.forEach(function(s){{
    var svgInfo=(s.svgs||{{}})[_synLevel]; if(!svgInfo) return;
    var icon=L.divIcon({{html:svgInfo.svg,iconSize:[svgInfo.w,svgInfo.h],iconAnchor:[Math.round(svgInfo.w/2),Math.round(svgInfo.h/2)],className:""}});
    L.marker([s.lat,s.lon],{{icon:icon}}).bindPopup(s.popup,{{maxWidth:320}}).addTo(_synStnLayer);
  }});
  if(_synShowStations) _synStnLayer.addTo(MAP);
}}
function _synInit(){{
  var slider=document.getElementById("syn-time-slider");
  if(slider){{slider.max=String(Math.max(0,_SYN_TIME_STEPS.length-1));slider.value="0";}}
  synSetLevel("850");synRender();
}}
if(document.readyState==="complete"){{setTimeout(_synInit,700);}}
else{{window.addEventListener("load",function(){{setTimeout(_synInit,700);}});}}
</script>
'''


def _build_surface_map(sfc_keys, slp_grids, qpf_grids, lon_vecs, lat_vecs,
                       hl_centers_by_key, rdps_run_dt, gdps_run_dt):
    """Build GEM surface MSLP + QPF map."""
    if not sfc_keys:
        print('No surface keys — skipping surface map')
        return

    _MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    _DOWS   = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

    sfc_time_steps = []
    for _key in sfc_keys:
        _y, _mo, _dd, _h = int(_key[:4]), int(_key[4:6]), int(_key[6:8]), int(_key[9:11])
        _dt = pd.Timestamp(year=_y, month=_mo, day=_dd)
        sfc_time_steps.append({
            'key': _key,
            'label': f'{_DOWS[_dt.dayofweek]} {_MONTHS[_mo-1]} {_dd} {_h:02d}Z',
            'hour': _h
        })

    # Build contour data for each frame
    frame_data = {}
    for _key in sfc_keys:
        slp   = slp_grids.get(_key)
        qpf   = qpf_grids.get(_key)
        lonv  = lon_vecs.get(_key)
        latv  = lat_vecs.get(_key)
        if slp is None or lonv is None:
            continue
        mslp_contours = _extract_contours_sfc(slp, lonv, latv, MSLP_INTERVAL)
        qpf_bands     = _extract_qpf_bands(qpf, latv, lonv) if qpf is not None else []
        frame_data[_key] = {
            'mslp': mslp_contours,
            'qpf':  qpf_bands,
            'hl':   hl_centers_by_key.get(_key, []),
        }

    _center_lat = float(list(lat_vecs.values())[0].mean())
    _center_lon = float(list(lon_vecs.values())[0].mean())

    m = folium.Map(location=[_center_lat, _center_lon], zoom_start=5,
                   tiles=None, prefer_canvas=True)
    folium.TileLayer(tiles='about:blank', attr=' ', name='Blank',
                     max_zoom=19, show=True).add_to(m)
    m.get_root().html.add_child(folium.Element(
        '<style>.leaflet-container{background:#e0f2ff!important;}</style>'
    ))
    m.get_root().html.add_child(folium.Element(_build_borders_js()))
    if fire_zones_html:
        m.get_root().html.add_child(folium.Element(fire_zones_html))

    rdps_str = rdps_run_dt.strftime('%Y-%m-%d %HZ') if rdps_run_dt else 'unknown'
    gdps_str = gdps_run_dt.strftime('%Y-%m-%d %HZ') if gdps_run_dt else 'unknown'

    sfc_bar = '''
<style>
#gem-bar{position:fixed;bottom:0;left:0;right:0;z-index:10000;background:#1a1a2e;
  border-top:2px solid #4a7fc1;padding:8px 16px;display:flex;align-items:center;
  gap:14px;font-family:"Courier New",monospace;font-size:11px;color:#e0e0e0;min-height:52px;}
.gem-layer-btn{font-size:12px;padding:4px 14px;cursor:pointer;border:1px solid #3a4a6a;
  border-radius:4px;background:#2a2a4a;color:#c0c8e0;font-family:"Courier New",monospace;font-weight:bold;}
.gem-layer-btn.active{background:#4a7fc1;color:#fff;}
#gem-time-slider{width:340px;accent-color:#4a7fc1;cursor:pointer;}
#gem-ts-label{color:#c0d0ff;font-size:12px;min-width:220px;font-weight:bold;}
</style>
<div id="gem-bar">
  <div style="display:flex;align-items:center;gap:6px;border-right:1px solid #3a3a5a;padding-right:14px;">
    <button class="gem-layer-btn active" id="btn-mslp" onclick="gemToggle(\'mslp\')">MSLP</button>
    <button class="gem-layer-btn active" id="btn-qpf"  onclick="gemToggle(\'qpf\')">QPF 12h</button>
  </div>
  <div style="display:flex;align-items:center;gap:6px;border-right:1px solid #3a3a5a;padding-right:14px;">
    <input type="range" id="gem-time-slider" min="0" value="0" oninput="gemSliderChange(this.value)">
  </div>
  <span id="gem-ts-label">—</span>
</div>
'''
    m.get_root().html.add_child(folium.Element(sfc_bar))

    sfc_js = f'''
<script>
var _GEM_STEPS  = {json.dumps(sfc_time_steps)};
var _GEM_FRAMES = {json.dumps(frame_data)};
var _gemStepIdx=0;
var _gemShowMslp=true;
var _gemShowQpf=true;
var _gemMslpLayer=null;
var _gemQpfLayer=null;
function _getMap(){{var k=Object.keys(window).filter(function(k){{return k.startsWith("map_");}});return k.length?window[k[0]]:null;}}
function gemToggle(which){{
  if(which==="mslp"){{_gemShowMslp=!_gemShowMslp;document.getElementById("btn-mslp").classList.toggle("active",_gemShowMslp);}}
  else{{_gemShowQpf=!_gemShowQpf;document.getElementById("btn-qpf").classList.toggle("active",_gemShowQpf);}}
  gemRender(_gemStepIdx);
}}
function gemSliderChange(v){{_gemStepIdx=parseInt(v);gemRender(_gemStepIdx);}}
function gemRender(idx){{
  var MAP=_getMap(); if(!MAP) return;
  var step=_GEM_STEPS[idx]; if(!step) return;
  var lbl=document.getElementById("gem-ts-label");
  if(lbl) lbl.textContent=step.label;
  if(_gemQpfLayer){{MAP.removeLayer(_gemQpfLayer);_gemQpfLayer=null;}}
  if(_gemMslpLayer){{MAP.removeLayer(_gemMslpLayer);_gemMslpLayer=null;}}
  var fd=_GEM_FRAMES[step.key]; if(!fd) return;
  if(_gemShowQpf&&fd.qpf&&fd.qpf.length){{
    _gemQpfLayer=L.layerGroup();
    fd.qpf.forEach(function(band){{
      if(!band.coords||band.coords.length<3) return;
      L.polygon([band.coords],{{color:"none",weight:0,fillColor:band.color,fillOpacity:0.45,interactive:false}}).addTo(_gemQpfLayer);
    }});
    _gemQpfLayer.addTo(MAP);
  }}
  if(_gemShowMslp&&fd.mslp&&fd.mslp.length){{
    _gemMslpLayer=L.layerGroup();
    if(!MAP.getPane("heightPane")){{MAP.createPane("heightPane");MAP.getPane("heightPane").style.zIndex=490;MAP.getPane("heightPane").style.pointerEvents="none";}}
    fd.mslp.forEach(function(ct){{
      if(!ct.coords||ct.coords.length<2) return;
      L.polyline(ct.coords,{{color:"#0d2040",weight:ct.weight,opacity:ct.opacity,pane:"heightPane"}}).addTo(_gemMslpLayer);
    }});
    (fd.hl||[]).forEach(function(c){{
      var _isH=c.type==="H";
      var _shadow="1px 1px 0 white,-1px -1px 0 white,1px -1px 0 white,-1px 1px 0 white";
      L.marker([c.lat,c.lon],{{icon:L.divIcon({{html:'<div style="font-size:52px;font-weight:bold;color:'+(_isH?"#cc0000":"#0000cc")+';font-family:Palatino,serif;text-shadow:'+_shadow+';">'+c.type+'<br><span style="font-size:16px;">'+Math.round(c.val)+'</span></div>',iconSize:[60,70],iconAnchor:[30,35],className:""}}))}}).addTo(_gemMslpLayer);
    }});
    _gemMslpLayer.addTo(MAP);
  }}
}}
function _gemInit(){{
  var slider=document.getElementById("gem-time-slider");
  if(slider){{slider.max=String(Math.max(0,_GEM_STEPS.length-1));slider.value="0";}}
  gemRender(0);
}}
if(document.readyState==="complete"){{setTimeout(_gemInit,800);}}
else{{window.addEventListener("load",function(){{setTimeout(_gemInit,800);}});}}
</script>
'''
    m.get_root().html.add_child(folium.Element(sfc_js))

    sfc_out = os.path.join(OUTPUT_DIR, 'gem_surface_map.html')
    m.save(sfc_out)
    print(f'✅ Surface map saved → {sfc_out}')


def _extract_contours_sfc(grid, lon_vec, lat_vec, interval):
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
        for coords in cs.allsegs[li]:
            if len(coords) < 2: continue
            contours.append({
                'level':   round(float(lvl), 1),
                'bold':    is_major,
                'weight':  2.5 if is_major else 2.0,
                'opacity': 1.0,
                'coords':  [[round(float(c[1]),3), round(float(c[0]),3)] for c in coords],
            })
    return contours


def _extract_qpf_bands(grid, lat_vec, lon_vec, n_interp=120):
    from scipy.ndimage import zoom as _zoom
    if grid is None:
        return []
    gq = _zoom(grid, n_interp / grid.shape[0], order=1)
    gq = np.clip(gq, 0, None)
    if gq.max() < 0.6:
        return []
    _QPF_LEVELS = [0.6, 1.5, 3, 5, 10, 20, 30, 40, 50, 60, 80, 100, 120]
    _QPF_COLORS = {
        0.6:'#c8f0a0',1.5:'#78d048',3:'#228b22',5:'#00aaaa',10:'#1a78c2',
        20:'#6a0dad',30:'#cc00cc',40:'#ffff00',50:'#ffaa00',
        60:'#ff4400',80:'#cc0000',100:'#880000',120:'#111111'
    }
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
        color = _QPF_COLORS.get(lvl, '#111111')
        for verts in seg_list:
            verts = np.array(verts)
            if len(verts) < 3: continue
            coords = [[round(float(v[1]),3), round(float(v[0]),3)]
                      for v in verts if not np.isnan(v).any()]
            if len(coords) < 3: continue
            bands.append({'level': float(lvl), 'color': color, 'coords': coords})
    return bands


# ── Main entry point ───────────────────────────────────────────────────────────
def main():
    print('=' * 60)
    print('  GEM Upper Air + Surface Forecast Map Generator')
    print('  GitHub Actions / standalone mode')
    print('=' * 60)

    # Run async fetch
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(run_fetch())
    finally:
        loop.close()

    target_vts, sfc_target_vts, target_lats, target_lons, tasks, sfc_tasks, errors = result

    if errors:
        print(f'\n⚠ {len(errors)} fetch errors:')
        for url, err in errors[:10]:
            print(f'  [{err}] {url[:80]}...')

    # Assemble dataframe
    gem_ua_df = assemble_gem_ua_df(target_vts, sfc_target_vts, target_lats, target_lons)
    print(f'\n✓ gem_ua_df: {len(gem_ua_df)} rows')

    # Build ua_raw_df (just use gem data directly for standalone mode)
    ua_raw_df = gem_ua_df.copy()

    # Build ua_summary_df
    STANDARD_LEVELS = [850, 500]
    LEVEL_TOL       = 25
    FIELDS = ['PRES','HGHT','TEMP','DWPT','RELH','MIXR','DRCT','SPED','THTA','THTE','THTV']

    ua_raw_df['_date'] = ua_raw_df['valid_time'].astype(str).str[:10]
    _ua_isob = ua_raw_df[ua_raw_df['PRES'] != 0.0].copy()
    if 'hour' not in ua_raw_df.columns:
        ua_raw_df['hour'] = pd.to_datetime(ua_raw_df['valid_time'].str[:13], utc=True).dt.hour

    _KEY = ['icao', '_date', 'hour']
    _meta = (_ua_isob.groupby(_KEY, sort=False)
             [['wmo','stn_name','lat','lon','valid_time']]
             .first().reset_index())

    _level_dfs = []
    for lvl in STANDARD_LEVELS:
        _df = _ua_isob.copy()
        _df['_dist'] = (_df['PRES'] - lvl).abs()
        _df = _df[_df['_dist'] <= LEVEL_TOL]
        _df = (_df.sort_values('_dist')
                  .groupby(_KEY, sort=False)[FIELDS]
                  .first().reset_index())
        _df.columns = _KEY + [f'{f}_{lvl}' for f in FIELDS]
        _level_dfs.append(_df)

    ua_summary_df = _meta.rename(columns={'_date': '_date'})
    for _ldf in _level_dfs:
        ua_summary_df = ua_summary_df.merge(_ldf, on=_KEY, how='left')

    ua_raw_df = ua_raw_df.drop(columns=['_date'])
    ua_summary_df['_vt']   = pd.to_datetime(ua_summary_df['valid_time'])
    ua_summary_df['_date'] = ua_summary_df['_vt'].dt.date
    ua_summary_df['_hour'] = ua_summary_df['_vt'].dt.hour

    synoptic_times = sorted(
        ua_summary_df[['_date', '_hour']].drop_duplicates()
        .itertuples(index=False, name=None)
    )

    print(f'✓ ua_summary_df: {len(ua_summary_df)} rows')
    print(f'✓ Synoptic times: {len(synoptic_times)}')

    # Build synthetic metar_records
    _sfc = gem_ua_df[(gem_ua_df['PRES'] == 0.0) & (gem_ua_df['MSLP'].notna())].copy()
    metar_records = []
    for _, r in _sfc.iterrows():
        vt = pd.to_datetime(r['valid_time'].replace('Z', ''), format='%Y-%m-%d %H')
        metar_records.append({
            'time': vt.to_pydatetime(), 'hour': int(vt.hour),
            'timestamp': vt.strftime('%d%H00'),
            'icao': r['icao'], 'name': r['stn_name'],
            'lat': float(r['lat']), 'lon': float(r['lon']),
            'slp': float(r['MSLP']), 'altimeter': None,
            'precip_accum': float(r['QPF12H']) if pd.notna(r.get('QPF12H')) else None,
            'temp': None, 'dew': None, 'wind_dir': None, 'wind_spd': None,
            'wind_gust': None, 'flt_cat': 'VFR', 'rh': None, 'vis': None,
            'clouds': [], 'weather': '', 'tendency': None, 'pressure_change': None,
            '_model': r.get('_model', ''), '_synthetic': True,
        })
    metar_records = sorted(metar_records, key=lambda x: (x['time'], x['icao']))
    print(f'✓ metar_records: {len(metar_records)}')

    # Build surface grids
    _sfc_df = ua_raw_df[ua_raw_df['PRES'] == 0.0].copy()
    _sfc_df['_vt']   = pd.to_datetime(_sfc_df['valid_time'].str.replace('Z', '', regex=False), format='%Y-%m-%d %H')
    _sfc_df['_date'] = _sfc_df['_vt'].dt.date
    _sfc_df['_hour'] = _sfc_df['_vt'].dt.hour
    _sfc_times       = sorted(_sfc_df[['_date', '_hour']].drop_duplicates().itertuples(index=False, name=None))

    slp_grids = {}
    qpf_grids = {}
    lon_vecs  = {}
    lat_vecs  = {}

    for (_date, _hr) in _sfc_times:
        _date_str  = pd.Timestamp(_date).strftime('%Y%m%d')
        _key       = f'{_date_str}_{int(_hr):02d}'
        _sub = _sfc_df[(_sfc_df['_date'] == _date) & (_sfc_df['_hour'] == _hr)]
        slp_grid, lon_vec, lat_vec, _, _ = _gem_build_grid(
            _sub, 'MSLP', rbf_smoothing=MSLP_RBF_SMOOTHING, sigma=MSLP_SIGMA)
        qpf_grid, _, _, _, _ = _gem_build_grid(
            _sub, 'QPF12H', rbf_smoothing=QPF_RBF_SMOOTHING, sigma=QPF_SIGMA,
            lon_vec=lon_vec, lat_vec=lat_vec)
        if slp_grid is not None:
            if qpf_grid is not None:
                qpf_grid = np.clip(qpf_grid, 0.0, None)
            slp_grids[_key] = slp_grid
            qpf_grids[_key] = qpf_grid
            lon_vecs[_key]  = lon_vec
            lat_vecs[_key]  = lat_vec
            print(f'  {_key}: MSLP {slp_grid.min():.1f}–{slp_grid.max():.1f} hPa')

    # H/L detection
    hl_centers_by_key = {}
    for _key, slp_grid in slp_grids.items():
        lon_vec = lon_vecs[_key]
        lat_vec = lat_vecs[_key]
        _centers = find_hl_centers(slp_grid, lon_vec, lat_vec,
                                   neighborhood=HL_NEIGHBORHOOD,
                                   min_delta=HL_MIN_DELTA)
        hl_centers_by_key[_key] = _centers
        _h = [c for c in _centers if c['type']=='H']
        _l = [c for c in _centers if c['type']=='L']
        print(f'  {_key}: {len(_h)} High(s), {len(_l)} Low(s)')

    # Simple UA contour processing (lightweight, no MetPy dependency)
    _ts_ua = {}
    for (_date_val, _hr) in synoptic_times:
        _date_str = pd.Timestamp(_date_val).strftime('%Y%m%d')
        _key      = f'{_date_str}_{int(_hr):02d}'
        _df_hr    = ua_summary_df[
            (ua_summary_df['_date'] == _date_val) &
            (ua_summary_df['_hour'] == _hr)
        ].copy()

        _hr_data = {}
        for _plvl in [850, 500]:
            _lons = _df_hr['lon'].values.astype(float)
            _lats = _df_hr['lat'].values.astype(float)
            _hght_col = f'HGHT_{_plvl}'
            _temp_col = f'TEMP_{_plvl}'
            _hght_segs = []
            _temp_segs = []

            if _hght_col in _df_hr.columns:
                _mask = _df_hr[_hght_col].notna()
                _sub  = _df_hr[_mask]
                if len(_sub) >= 8:
                    from scipy.interpolate import griddata
                    _N = _GRID_N
                    _pad = 1.5
                    _lv = np.linspace(_lons.min()-_pad, _lons.max()+_pad, _N)
                    _ltv = np.linspace(_lats.min()-_pad, _lats.max()+_pad, _N)
                    _glon, _glat = np.meshgrid(_lv, _ltv)
                    _grid = griddata(
                        np.column_stack([_sub['lon'].values, _sub['lat'].values]),
                        _sub[_hght_col].values.astype(float),
                        (_glon, _glat), method='linear')
                    _nan_mask = np.isnan(_grid)
                    if _nan_mask.any():
                        _grid_nn = griddata(
                            np.column_stack([_sub['lon'].values, _sub['lat'].values]),
                            _sub[_hght_col].values.astype(float),
                            (_glon, _glat), method='nearest')
                        _grid[_nan_mask] = _grid_nn[_nan_mask]
                    _grid = gaussian_filter(_grid, sigma=_SIGMA['HGHT'])
                    _fixed = UA_HGHT_LEVELS.get(_plvl)
                    if _fixed is not None:
                        fig, ax = plt.subplots(figsize=(1,1))
                        try:
                            cs = ax.contour(_glon, _glat, _grid, levels=_fixed)
                            for li, lvl in enumerate(cs.levels):
                                for coords in cs.allsegs[li]:
                                    if len(coords) < 2: continue
                                    mid = coords[len(coords)//2]
                                    _hght_segs.append({
                                        'level': float(lvl),
                                        'coords': [[float(c[0]),float(c[1])] for c in coords],
                                        'label_lon': float(mid[0]),
                                        'label_lat': float(mid[1]),
                                    })
                        except Exception:
                            pass
                        plt.close(fig)

            if _temp_col in _df_hr.columns:
                _mask = _df_hr[_temp_col].notna()
                _sub  = _df_hr[_mask]
                if len(_sub) >= 8:
                    from scipy.interpolate import griddata
                    _N = _GRID_N
                    _pad = 1.5
                    _lv = np.linspace(_sub['lon'].values.min()-_pad, _sub['lon'].values.max()+_pad, _N)
                    _ltv = np.linspace(_sub['lat'].values.min()-_pad, _sub['lat'].values.max()+_pad, _N)
                    _glon, _glat = np.meshgrid(_lv, _ltv)
                    _tgrid = griddata(
                        np.column_stack([_sub['lon'].values, _sub['lat'].values]),
                        _sub[_temp_col].values.astype(float),
                        (_glon, _glat), method='linear')
                    _nan_mask = np.isnan(_tgrid)
                    if _nan_mask.any():
                        _tgrid_nn = griddata(
                            np.column_stack([_sub['lon'].values, _sub['lat'].values]),
                            _sub[_temp_col].values.astype(float),
                            (_glon, _glat), method='nearest')
                        _tgrid[_nan_mask] = _tgrid_nn[_nan_mask]
                    _tgrid = gaussian_filter(_tgrid, sigma=_SIGMA['TEMP'])
                    vmin = np.floor(_tgrid.min() / _INTERVALS['TEMP']) * _INTERVALS['TEMP']
                    vmax = np.ceil( _tgrid.max() / _INTERVALS['TEMP']) * _INTERVALS['TEMP']
                    t_levels = np.arange(vmin, vmax + _INTERVALS['TEMP'], _INTERVALS['TEMP'])
                    if len(t_levels) >= 2:
                        fig, ax = plt.subplots(figsize=(1,1))
                        try:
                            cs = ax.contour(_glon, _glat, _tgrid, levels=t_levels)
                            for li, lvl in enumerate(cs.levels):
                                for coords in cs.allsegs[li]:
                                    if len(coords) < 2: continue
                                    mid = coords[len(coords)//2]
                                    _temp_segs.append({
                                        'level': float(lvl),
                                        'coords': [[float(c[0]),float(c[1])] for c in coords],
                                        'label_lon': float(mid[0]),
                                        'label_lat': float(mid[1]),
                                    })
                        except Exception:
                            pass
                        plt.close(fig)

            _hr_data[str(_plvl)] = {
                'hght': _hght_segs,
                'temp': _temp_segs,
                'temp_band_fills': [],
                'ttdp': [],
                'sped': [],
            }

        _ts_ua[_key] = {
            'levels': _hr_data,
            'instab': [],
            'thermal_ridge_850': [], 'thermal_trough_850': [],
            'thermal_ridge_700': [], 'thermal_trough_700': [],
            'thermal_ridge_500': [], 'thermal_trough_500': [],
            'dtdx_zero_pts': [],
            **{f'hl_{pl}': [] for pl in HL_LEVELS},
        }
        print(f'  {_key}: processed')

    ts_ua_json_str     = json.dumps(_ts_ua)
    ts_ua_stn_json_str = '{}'  # built inside build_synoptic_map

    sfc_keys_sorted = sorted(k for k in slp_grids)

    # Get run times
    _rdps_run_dt = globals().get('_rdps_run_dt')
    _gdps_run_dt = globals().get('_gdps_run_dt')

    # Build maps
    build_synoptic_map(
        gem_ua_df      = gem_ua_df,
        ua_summary_df  = ua_summary_df,
        metar_records  = metar_records,
        synoptic_times = synoptic_times,
        ts_ua_json_str         = ts_ua_json_str,
        ts_ua_stn_json_str     = ts_ua_stn_json_str,
        sfc_keys       = sfc_keys_sorted,
        slp_grids      = slp_grids,
        qpf_grids      = qpf_grids,
        lon_vecs       = lon_vecs,
        lat_vecs       = lat_vecs,
        hl_centers_by_key = hl_centers_by_key,
        rdps_run_dt    = _rdps_run_dt,
        gdps_run_dt    = _gdps_run_dt,
    )

    print(f'\n✅ All done — outputs in {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
