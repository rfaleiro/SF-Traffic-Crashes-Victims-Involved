"""
stats_analysis.py
=================
Statistical analysis of fatal crash concentration near Muni Metro lines.

Null Hypothesis (H0): Fatal crashes occur in proportion to the geographic area
  of the 50m Muni Metro buffer relative to total SF land area.
Alternative (H1): Fatal crashes are more concentrated near Muni Metro than
  expected by chance (one-tailed Binomial test).

Baseline note: The geographic area fraction (~4.1%) is a conservative baseline
  because crashes only happen on roads. Muni Metro routes follow major arterials
  that carry more traffic than their land-area fraction implies. A road-network-
  weighted baseline would be more precise, but is not used here due to data
  availability. The current approach will tend to overstate the Relative Risk
  slightly; the true excess risk is likely real but may be somewhat lower.
"""

import pandas as pd
import geopandas as gpd
from shapely import wkt
from scipy.stats import binomtest, norm
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────
SF_LAND_AREA_SQ_KM = 121.4   # total SF land area in sq km
BUFFER_M = 50                  # buffer radius in metres
ALPHA = 0.05

# ── File paths ─────────────────────────────────────────────────────────────────
muni_path    = 'data/raw/Muni_Simple_Routes_20260319.csv'
crashes_path = 'data/raw/Traffic_Crashes_Resulting_in_Fatality_20260319.csv'

# ── Load & prepare Muni Metro data ─────────────────────────────────────────────
muni_df    = pd.read_csv(muni_path)
muni_metro = muni_df[muni_df['SERVICE_CA'] == 'Muni Metro'].copy()
muni_metro['geometry'] = muni_metro['shape'].apply(wkt.loads)
muni_gdf   = gpd.GeoDataFrame(muni_metro, geometry='geometry', crs="EPSG:4326")

# Project to California UTM zone 10 for accurate metric buffering
muni_gdf_proj = muni_gdf.to_crs("EPSG:32610")
muni_buffered = muni_gdf_proj.copy()
muni_buffered['geometry'] = muni_buffered.geometry.buffer(BUFFER_M)

# Dissolve overlapping buffer polygons to avoid double-counting area (Fix #4)
union_geom     = muni_buffered.geometry.union_all()
muni_area_sq_m  = union_geom.area
muni_area_sq_km = muni_area_sq_m / 1e6
expected_prob   = muni_area_sq_km / SF_LAND_AREA_SQ_KM

# ── Load crash data & perform spatial join (Fix #1 – no more hardcoded counts) ─
crashes_df = pd.read_csv(crashes_path)
crashes_df = crashes_df.dropna(subset=['longitude', 'latitude']).copy()
crashes_gdf = gpd.GeoDataFrame(
    crashes_df,
    geometry=gpd.points_from_xy(crashes_df['longitude'], crashes_df['latitude']),
    crs="EPSG:4326"
).to_crs("EPSG:32610")

# Spatial join: find crashes that fall inside the unioned buffer polygon
crashes_near_muni = gpd.sjoin(crashes_gdf, muni_buffered, how='inner', predicate='intersects')
near_ids = crashes_near_muni['unique_id'].unique()

crashes_gdf_wgs = crashes_gdf.to_crs("EPSG:4326")
crashes_gdf_wgs['near_muni'] = crashes_gdf_wgs['unique_id'].isin(near_ids)

crashes_near   = int(crashes_gdf_wgs['near_muni'].sum())
total_crashes  = len(crashes_gdf_wgs)
crashes_far    = total_crashes - crashes_near
area_far_sq_km = SF_LAND_AREA_SQ_KM - muni_area_sq_km

# ── Derived statistics ─────────────────────────────────────────────────────────
density_near = crashes_near / muni_area_sq_km
density_far  = crashes_far  / area_far_sq_km
rr           = density_near / density_far

# 95% CI for Relative Risk via delta method on log(RR) (Fix #3)
obs_p   = crashes_near / total_crashes
se_log_rr = np.sqrt(
    (1 - obs_p)   / crashes_near                   # near component
    + (1 - expected_prob) / (total_crashes * expected_prob)  # far component
)
z_crit  = norm.ppf(0.975)
rr_lo   = np.exp(np.log(rr) - z_crit * se_log_rr)
rr_hi   = np.exp(np.log(rr) + z_crit * se_log_rr)

# ── Binomial hypothesis test ───────────────────────────────────────────────────
test_result = binomtest(crashes_near, total_crashes, p=expected_prob, alternative='greater')

# ── Output ─────────────────────────────────────────────────────────────────────
print(f"--- Spatial Exposure Setup ---")
print(f"SF Total Land Area: {SF_LAND_AREA_SQ_KM:.2f} sq km")
print(f"Muni Metro {BUFFER_M}m Zone Area: {muni_area_sq_km:.3f} sq km")
print(f"Muni Metro Zone % of SF: {expected_prob*100:.2f}%")
print()
print(f"--- Crash Densities ---")
print(f"Crashes near Muni Metro: {crashes_near} ({crashes_near/total_crashes*100:.2f}%)")
print(f"Total fatal crashes: {total_crashes}")
print(f"Density near Muni Metro: {density_near:.2f} crashes / sq km")
print(f"Density elsewhere: {density_far:.2f} crashes / sq km")
print(f"Relative Risk: {rr:.2f}x (95% CI: {rr_lo:.2f}–{rr_hi:.2f})")
print()
print(f"--- Statistical Significance ---")
print(f"Expected crashes (if random): {total_crashes * expected_prob:.2f}")
p_str = "< 0.0001" if test_result.pvalue < 0.0001 else f"{test_result.pvalue:.4f}"
print(f"P-value: {p_str}")
if test_result.pvalue < ALPHA:
    print("Conclusion: Fatal crashes are STATISTICALLY SIGNIFICANTLY concentrated near Muni Metro lines.")
else:
    print("Conclusion: No significant spatial concentration near Muni Metro lines.")
