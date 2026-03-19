import pandas as pd
import geopandas as gpd
from shapely import wkt
from scipy.stats import binomtest

# SF Land Area in sq km (approx)
SF_LAND_AREA_SQ_KM = 121.4

muni_path = 'data/raw/Muni_Simple_Routes_20260319.csv'
muni_df = pd.read_csv(muni_path)
muni_metro = muni_df[muni_df['SERVICE_CA'] == 'Muni Metro'].copy()
muni_metro['geometry'] = muni_metro['shape'].apply(wkt.loads)
muni_gdf = gpd.GeoDataFrame(muni_metro, geometry='geometry', crs="EPSG:4326")

# Project to California UTM zone 10 for metric properties
muni_gdf_proj = muni_gdf.to_crs("EPSG:32610")
muni_buffered = muni_gdf_proj.copy()
muni_buffered['geometry'] = muni_buffered.geometry.buffer(50)

# Unary union to avoid overlapping buffer areas compounding
unary_union_geom = muni_buffered.geometry.unary_union
muni_area_sq_m = unary_union_geom.area
muni_area_sq_km = muni_area_sq_m / (10**6)

expected_prob = muni_area_sq_km / SF_LAND_AREA_SQ_KM

crashes_near = 47
total_crashes = 351
crashes_far = total_crashes - crashes_near
area_far_sq_km = SF_LAND_AREA_SQ_KM - muni_area_sq_km

density_near = crashes_near / muni_area_sq_km
density_far = crashes_far / area_far_sq_km

rr = density_near / density_far

test_result = binomtest(crashes_near, total_crashes, p=expected_prob, alternative='greater')

print(f"--- Spatial Exposure Setup ---")
print(f"SF Total Land Area: {SF_LAND_AREA_SQ_KM:.2f} sq km")
print(f"Muni Metro 50m Zone Area: {muni_area_sq_km:.2f} sq km")
print(f"Muni Metro Zone % of SF: {expected_prob*100:.2f}%\n")

print(f"--- Crash Densities ---")
print(f"Crashes near Muni Metro: {crashes_near} ({crashes_near/total_crashes*100:.2f}%)")
print(f"Density near Muni Metro: {density_near:.2f} crashes / sq km")
print(f"Density elsewhere: {density_far:.2f} crashes / sq km")
print(f"Relative Risk: {rr:.2f}x\n")

print(f"--- Statistical Significance ---")
print(f"Expected crashes (if random): {total_crashes * expected_prob:.2f}")
print(f"P-value: {test_result.pvalue:.2e}")
if test_result.pvalue < 0.05:
    print("Conclusion: Fatal crashes are STATISTICALLY SIGNIFICANTLY concentrated near Muni Metro lines.")
else:
    print("Conclusion: No significant spatial concentration near Muni Metro lines.")
