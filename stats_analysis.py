"""
stats_analysis.py
=================
Statistical analysis of fatal AND injury crash concentration near Muni Metro lines.

Null Hypothesis (H0): Crashes occur in proportion to the geographic area
  of the 50m Muni Metro buffer relative to total SF land area.
Alternative (H1): Crashes are more concentrated near Muni Metro than
  expected by chance (one-tailed Binomial test).

Baseline note: The geographic area fraction (~4.10%) is a conservative baseline
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
import warnings

warnings.filterwarnings('ignore')

# ── Constants ──────────────────────────────────────────────────────────────────
SF_LAND_AREA_SQ_KM = 121.4   # total SF land area in sq km
BUFFER_M = 50                  # buffer radius in metres
ALPHA = 0.05

# ── File paths ─────────────────────────────────────────────────────────────────
muni_path    = 'data/raw/Muni_Simple_Routes_20260319.csv'
fatal_path   = 'data/raw/Traffic_Crashes_Resulting_in_Fatality_20260319.csv'
injury_path  = 'data/raw/Traffic_Crashes_Resulting_in_Injury_20260319.csv'

# ── Load & prepare Muni Metro data ─────────────────────────────────────────────
muni_df    = pd.read_csv(muni_path)
muni_metro = muni_df[muni_df['SERVICE_CA'] == 'Muni Metro'].copy()
muni_metro['geometry'] = muni_metro['shape'].apply(wkt.loads)
muni_gdf   = gpd.GeoDataFrame(muni_metro, geometry='geometry', crs="EPSG:4326")

muni_gdf_proj = muni_gdf.to_crs("EPSG:32610")
muni_buffered = muni_gdf_proj.copy()
muni_buffered['geometry'] = muni_buffered.geometry.buffer(BUFFER_M)

union_geom      = muni_buffered.geometry.union_all()
muni_area_sq_m  = union_geom.area
muni_area_sq_km = muni_area_sq_m / 1e6
expected_prob   = muni_area_sq_km / SF_LAND_AREA_SQ_KM

print(f"=========================================================")
print(f"  SPATIAL EXPOSURE SETUP")
print(f"=========================================================")
print(f"SF Total Land Area        : {SF_LAND_AREA_SQ_KM:.2f} sq km")
print(f"Muni Metro {BUFFER_M}m Zone Area: {muni_area_sq_km:.3f} sq km")
print(f"Muni Metro Zone % of SF   : {expected_prob*100:.2f}%\n")


def analyze_dataset(name, csv_path, lon_col, lat_col):
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.dropna(subset=[lon_col, lat_col]).copy()
    df = df[(df[lon_col] > -123) & (df[lon_col] < -121)]  # filter errant coords
    
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326"
    ).to_crs("EPSG:32610")

    crashes_near_muni = gpd.sjoin(gdf, muni_buffered, how='inner', predicate='intersects')
    near_ids = crashes_near_muni['unique_id'].unique()
    
    crashes_gdf_wgs = gdf.to_crs("EPSG:4326")
    crashes_gdf_wgs['near_muni'] = crashes_gdf_wgs['unique_id'].isin(near_ids)

    crashes_near   = int(crashes_gdf_wgs['near_muni'].sum())
    total_crashes  = len(crashes_gdf_wgs)
    crashes_far    = total_crashes - crashes_near
    area_far_sq_km = SF_LAND_AREA_SQ_KM - muni_area_sq_km

    density_near = crashes_near / muni_area_sq_km
    density_far  = crashes_far  / area_far_sq_km
    rr           = density_near / density_far

    obs_p   = crashes_near / total_crashes
    se_log_rr = np.sqrt(
        (1 - obs_p)   / crashes_near
        + (1 - expected_prob) / (total_crashes * expected_prob)
    )
    z_crit  = norm.ppf(0.975)
    rr_lo   = np.exp(np.log(rr) - z_crit * se_log_rr)
    rr_hi   = np.exp(np.log(rr) + z_crit * se_log_rr)

    test_result = binomtest(crashes_near, total_crashes, p=expected_prob, alternative='greater')
    p_str = "< 0.0001" if test_result.pvalue < 0.0001 else f"{test_result.pvalue:.4f}"

    print(f"=========================================================")
    print(f"  {name.upper()}")
    print(f"=========================================================")
    print(f"Total mapped crashes      : {total_crashes:,}")
    print(f"Crashes near Muni Metro   : {crashes_near:,} ({crashes_near/total_crashes*100:.2f}%)")
    print(f"Expected crashes (random) : {total_crashes * expected_prob:.2f}")
    print(f"Density near Muni Metro   : {density_near:.2f} crashes / sq km")
    print(f"Density elsewhere         : {density_far:.2f} crashes / sq km")
    print(f"Relative Risk             : {rr:.2f}x (95% CI: {rr_lo:.2f}\u2013{rr_hi:.2f})")
    print(f"P-value                   : {p_str}")
    
    if test_result.pvalue < ALPHA:
        print(f"Conclusion                : {name} are STATISTICALLY SIGNIFICANTLY concentrated near Muni Metro lines.\n")
    else:
        print(f"Conclusion                : No significant spatial concentration.\n")


# Run both analyses
analyze_dataset("Fatal Crashes", fatal_path, 'longitude', 'latitude')
analyze_dataset("Injury Crashes", injury_path, 'tb_longitude', 'tb_latitude')
