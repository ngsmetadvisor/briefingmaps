# @title
# ── Cell UA-2b. Fetch GEM upper-air + surface: RDPS days 0-3, GDPS days 3-7 ──
# ── Source: dd.weather.gc.ca  WXO-DD layout (confirmed May 2026) ─────────────
# - new code, with better runs checking system before run

import subprocess, sys
for _pkg in ['cfgrib', 'eccodes', 'xarray', 'scipy', 'aiohttp']:
    try:
        __import__(_pkg)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', _pkg])
import os
if os.path.exists('/usr/bin/apt-get'):
    subprocess.call(['apt-get', 'install', '-y', '-q', 'libeccodes-dev'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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

RDPS_FORECAST_DAYS = 3
GDPS_FORECAST_DAYS = 7
UA_HOURS  = [12]
SFC_HOURS = [0]
GEM_PRESSURE_LEVELS = [500, 850]
FETCH_SFC           = True
GEM_LAT_MAX  =  80.0
GEM_LAT_MIN  =  30.0
GEM_LON_MIN  = -150.0
GEM_LON_MAX  =  -80.0
GEM_GRID_DEG =   2.0
RDPS_RUN_HOURS    = [0, 6, 12, 18]
GDPS_RUN_HOURS    = [0, 12]
RDPS_MIN_AGE_H    = 2.5
GDPS_MIN_AGE_H    = 6.0
MAX_LOOKBACK_DAYS = 2
PROBE_MODE        = 'representative'
PROBE_RETRIES     = 2
PROBE_RETRY_DELAY = 2.0
PROBE_TIMEOUT     = 10
MAX_CONCURRENT = 3
TIMEOUT_S      = 60


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

_VAR_MAP = {
    'AirTemp':            'TEMP',
    'GeopotentialHeight': 'HGHT',
    'RelativeHumidity':   'RELH',
    'WindU':              '_UGRD',
    'WindV':              '_VGRD',
}
_WXO_VARS = list(_VAR_MAP.keys())
_PROBE_ISOB_VAR = 'AirTemp'

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

async def _probe_run(session, run_dt, is_rdps, target_vts, sfc_target_vts):
    url_isob_fn = _rdps_url    if is_rdps else _gdps_url
    url_sfc_fn  = _rdps_sfc_url if is_rdps else _gdps_sfc_url
    max_fxx     = 84 if is_rdps else 240

    probes = []
    vars_to_probe = [_PROBE_ISOB_VAR] if PROBE_MODE == 'representative' else _WXO_VARS

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

    if FETCH_SFC:
        sfc_vars_to_probe = (
            ['Pressure_MSL', 'Precip-Accum'] if PROBE_MODE == 'representative'
            else _SFC_VARS
        )
        for vt in sorted(sfc_target_vts):
            vt_mslp   = vt - timedelta(hours=6)
            vt_pacc   = vt
            fxx_mslp  = _fxx(run_dt, vt_mslp)
            fxx_tgt   = _fxx(run_dt, vt_pacc)
            fxx_prior = fxx_tgt - 12

            if 'Pressure_MSL' in sfc_vars_to_probe and 0 <= fxx_mslp <= max_fxx:
                url = url_sfc_fn(run_dt, fxx_mslp, 'Pressure_MSL')
                probes.append((f'fxx={fxx_mslp:03d}  Pressure_MSL@Sfc', url, False))

            if 'Precip-Accum' in sfc_vars_to_probe and 0 <= fxx_tgt <= max_fxx:
                url = url_sfc_fn(run_dt, fxx_tgt, 'Precip-Accum')
                probes.append((f'fxx={fxx_tgt:03d}  Precip-Accum@Sfc[target]', url, False))
                if fxx_prior == 0:
                    pass
                elif fxx_prior > 0:
                    url_prior = url_sfc_fn(run_dt, fxx_prior, 'Precip-Accum')
                    probes.append((f'fxx={fxx_prior:03d}  Precip-Accum@Sfc[prior]', url_prior, False))

    results = {}
    all_ok  = True

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
                        last_status = 'missing'
                        last_detail = 'HTTP 404'
                        break
                    else:
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

    fxx_groups = {}
    for label, (status, detail) in results.items():
        fxx_part = label.split('  ')[0]
        if fxx_part not in fxx_groups:
            fxx_groups[fxx_part] = []
        sym = '✓' if status == 'ok' else '✗'
        fxx_groups[fxx_part].append(f'{sym} {label.split("  ", 1)[1]}')

    report_lines = []
    for fxx_key in sorted(fxx_groups.keys()):
        items    = fxx_groups[fxx_key]
        statuses = ' | '.join(items)
        report_lines.append(f'  {fxx_key}: {statuses}')

    return all_ok, report_lines


async def _select_run_verified(run_hours, min_age_h, is_rdps,
                                target_vts, sfc_target_vts):
    model_label = 'RDPS' if is_rdps else 'GDPS'
    now = _NOW_UTC

    candidates = []
    for day_offset in range(MAX_LOOKBACK_DAYS + 1):
        for h in sorted(run_hours, reverse=True):
            cand = (now - timedelta(days=day_offset)).replace(
                hour=h, minute=0, second=0, microsecond=0)
            age_h = (now - cand).total_seconds() / 3600
            if age_h >= min_age_h:
                candidates.append(cand)

    candidates = sorted(set(candidates), reverse=True)

    if not candidates:
        raise RuntimeError(f'{model_label}: no run candidates found within lookback window')

    connector = aiohttp.TCPConnector(ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:

        primary   = candidates[0]
        fallbacks = candidates[1:]

        print(f'\n{model_label}:')
        print(f'  {primary.strftime("%Y-%m-%d %HZ")} run (probing all needed fxx):')
        all_ok, report = await _probe_run(session, primary, is_rdps,
                                          target_vts, sfc_target_vts)
        for line in report:
            print(line)

        if all_ok:
            print(f'  ✓ Verified {model_label} run: {primary.strftime("%Y-%m-%d %HZ")}')
            return primary

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

_target_vts = []
for _d in range(GDPS_FORECAST_DAYS + 1):
    for _h in sorted(UA_HOURS):
        _vt = _base_day_utc + timedelta(days=_d, hours=_h)
        _target_vts.append(_vt)

_sfc_target_vts = []
for _d in range(1, GDPS_FORECAST_DAYS + 2):
    _vt = _base_day_mdt + timedelta(days=_d, hours=6)
    _sfc_target_vts.append(_vt)

_target_lats = [lat for lat in GEM_LATITUDES for _   in GEM_LONGITUDE]
_target_lons = [lon for _   in GEM_LATITUDES for lon in GEM_LONGITUDE]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     SELECT VERIFIED RUNS                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

async def _select_runs():
    global _rdps_run_dt, _gdps_run_dt
    _rdps_run_dt = await _select_run_verified(
        RDPS_RUN_HOURS, RDPS_MIN_AGE_H, is_rdps=True,
        target_vts=_target_vts, sfc_target_vts=_sfc_target_vts
    )
    _gdps_run_dt = await _select_run_verified(
        GDPS_RUN_HOURS, GDPS_MIN_AGE_H, is_rdps=False,
        target_vts=_target_vts, sfc_target_vts=_sfc_target_vts
    )
asyncio.run(_select_runs())

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
        if _math.isnan(val) or abs(val) > 1e8:
            result[(tlat, tlon)] = None
        else:
            result[(tlat, tlon)] = round(float(val), 2) if is_temp else float(val)

    return result


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     BUILD DOWNLOAD TASK LIST                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

_point_data = {}
_sfc_data   = {}

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

_sfc_tasks = []
for vt in _sfc_target_vts:
    use_rdps  = _NOW_UTC < vt < _RDPS_CUTOFF
    run_dt    = _rdps_run_dt if use_rdps else _gdps_run_dt
    url_fn    = _rdps_sfc_url if use_rdps else _gdps_sfc_url
    max_fxx   = 84 if use_rdps else 240
    model_lbl = 'RDPS' if use_rdps else 'GDPS'

    vt_mslp   = vt - timedelta(hours=6)
    vt_pacc   = vt
    fxx_mslp  = _fxx(run_dt, vt_mslp)
    fxx_tgt   = _fxx(run_dt, vt_pacc)
    fxx_prior = fxx_tgt - 12

    if not (0 <= fxx_tgt <= max_fxx):
        continue

    if 0 <= fxx_mslp <= max_fxx:
        _sfc_tasks.append((model_lbl,
                           url_fn(run_dt, fxx_mslp, 'Pressure_MSL'),
                           'Pressure_MSL', vt_mslp, ''))

    _sfc_tasks.append((model_lbl,
                       url_fn(run_dt, fxx_tgt, 'Precip-Accum'),
                       'Precip-Accum', vt_pacc, ''))

    if fxx_prior > 0:
        _sfc_tasks.append((model_lbl,
                           url_fn(run_dt, fxx_prior, 'Precip-Accum'),
                           'Precip-Accum', vt_pacc, '_PRIOR'))

print(f'\nIsobaric tasks  : {len(_tasks)} GRIB2 files')
print(f'Surface tasks   : {len(_sfc_tasks)} GRIB2 files')
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
            extracted = _extract_points_grib(raw, _target_lats, _target_lons, var_name)
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
            extracted = _extract_points_grib(raw, _target_lats, _target_lons, var_name)
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
    print(f'  Fetch summary: {_n_dl} fetched, {_n_mi} missing, {_n_tot} total')

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
                extracted = _extract_points_grib(raw, _target_lats, _target_lons, var_name)
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
                extracted = _extract_points_grib(raw, _target_lats, _target_lons, var_name)
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
        print(f'\n✗ Still missing: {len(_still_isob)} isobaric, {len(_still_sfc)} surface')
        for t in _still_isob: print(f'  ✗ {t[0]}  {t[4].strftime("%Y-%m-%d %HZ")}  {t[2]}@{t[3]}hPa')
        for t in _still_sfc:  print(f'  ✗ {t[0]}  {t[3].strftime("%Y-%m-%d %HZ")}  {t[2]} [{t[4] or "TARGET"}]')
    else:
        print('✓ All retried tasks now populated')

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

if 'ua_raw_df' not in dir():
    print('⚠  ua_raw_df not found — creating empty frame.')
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
