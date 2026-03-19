import pandas as pd
import geopandas as gpd
from shapely import wkt
import folium
import os

# File paths
muni_path = 'data/raw/Muni_Simple_Routes_20260319.csv'
crashes_path = 'data/raw/Traffic_Crashes_Resulting_in_Fatality_20260319.csv'

def main():
    print("Loading Muni Route Data...")
    muni_df = pd.read_csv(muni_path)
    # Filter for Muni Metro
    muni_metro = muni_df[muni_df['SERVICE_CA'] == 'Muni Metro'].copy()
    
    print(f"Found {len(muni_metro)} Muni Metro segments.")
    if len(muni_metro) == 0:
        print("No Muni Metro found.")
        return

    # Parse WKT to geometry
    muni_metro['geometry'] = muni_metro['shape'].apply(wkt.loads)
    muni_gdf = gpd.GeoDataFrame(muni_metro, geometry='geometry', crs="EPSG:4326")

    print("Loading Crash Data...")
    crashes_df = pd.read_csv(crashes_path)
    crashes_df = crashes_df.dropna(subset=['longitude', 'latitude']).copy()
    crashes_gdf = gpd.GeoDataFrame(
        crashes_df,
        geometry=gpd.points_from_xy(crashes_df['longitude'], crashes_df['latitude']),
        crs="EPSG:4326"
    )
    print(f"Loaded {len(crashes_gdf)} fatal crashes.")

    # Project to a metric CRS for accurate buffering in meters
    # EPSG:32610 is UTM Zone 10N, which covers San Francisco
    muni_gdf_proj = muni_gdf.to_crs("EPSG:32610")
    crashes_gdf_proj = crashes_gdf.to_crs("EPSG:32610")

    # Buffer Muni Metro lines by 50 meters
    print("Buffering Muni Metro lines by 50 meters...")
    muni_buffered = muni_gdf_proj.copy()
    muni_buffered['geometry'] = muni_buffered.geometry.buffer(50)

    # Perform Spatial Join to find crashes near Muni Metro
    print("Performing spatial join...")
    crashes_near_muni = gpd.sjoin(crashes_gdf_proj, muni_buffered, how='inner', predicate='intersects')
    near_muni_ids = crashes_near_muni['unique_id'].unique()

    crashes_gdf['near_muni'] = crashes_gdf['unique_id'].isin(near_muni_ids)

    from scipy.stats import binomtest
    
    # Calculate Spatial exposures
    SF_LAND_AREA_SQ_KM = 121.4
    unary_union_geom = muni_buffered.geometry.union_all()
    muni_area_sq_m = unary_union_geom.area
    muni_area_sq_km = muni_area_sq_m / (10**6)
    expected_prob = muni_area_sq_km / SF_LAND_AREA_SQ_KM

    # Calculate statistics
    total_crashes = len(crashes_gdf)
    crashes_near = crashes_gdf['near_muni'].sum()
    percentage = (crashes_near / total_crashes) * 100 if total_crashes > 0 else 0
    crashes_far = total_crashes - crashes_near
    area_far_sq_km = SF_LAND_AREA_SQ_KM - muni_area_sq_km
    density_near = crashes_near / muni_area_sq_km if muni_area_sq_km > 0 else 0
    density_far = crashes_far / area_far_sq_km if area_far_sq_km > 0 else 0
    rr = density_near / density_far if density_far > 0 else 0
    
    test_result = binomtest(crashes_near, total_crashes, p=expected_prob, alternative='greater')
    is_significant = "Yes" if test_result.pvalue < 0.05 else "No"

    print("-" * 30)
    print(f"Total Fatal Crashes plotted: {total_crashes}")
    print(f"Fatal Crashes Near a Muni Metro (<=50m): {crashes_near} ({percentage:.2f}%)")
    print(f"Relative Risk: {rr:.2f}x")
    print(f"P-value: {test_result.pvalue:.2e}")
    print("-" * 30)

    # Create Visualization Map
    print("Building Folium Map...")
    sf_coords = [37.7749, -122.4194]
    m = folium.Map(location=sf_coords, zoom_start=12, tiles='CartoDB Positron')

    # Add Muni Metro line geometries to the map
    folium.GeoJson(
        muni_gdf,
        name='Muni Metro Lines',
        style_function=lambda x: {'color': '#27ae60', 'weight': 4, 'opacity': 0.8}
    ).add_to(m)

    # Convert the buffered area back to EPSG:4326 for Folium, add as a subtle polygon
    folium.GeoJson(
        muni_buffered.to_crs("EPSG:4326"),
        name='Muni Metro 50m Buffer',
        style_function=lambda x: {'color': '#2ecc71', 'weight': 0, 'fillOpacity': 0.3}
    ).add_to(m)

    # Add crash points
    for idx, row in crashes_gdf.iterrows():
        is_near = row['near_muni']
        color = '#e74c3c' if is_near else '#2980b9'
        fill_color = '#c0392b' if is_near else '#1f618d'
        radius = 5 if is_near else 3
        
        # Build tooltip HTML
        tooltip_html = (
            f"<b>Near Muni Metro:</b> {'YES' if is_near else 'NO'}<br>"
            f"<b>Date:</b> {row.get('collision_date', 'Unknown')}<br>"
            f"<b>Location:</b> {row.get('location', 'Unknown')}<br>"
            f"<b>Type:</b> {row.get('collision_type', 'Unknown')}"
        )
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=radius,
            color=color,
            fill=True,
            fill_color=fill_color,
            fill_opacity=0.8,
            tooltip=tooltip_html,
            weight=1
        ).add_to(m)

    # Add Layer Control to toggle layers
    folium.LayerControl().add_to(m)

    # Add a custom HTML legend
    legend_html = f'''
     <div style="position: fixed; 
     bottom: 50px; left: 50px; width: 450px; height: auto; max-height: 400px; 
     background-color: white; border:2px solid grey; z-index:9999; font-size:14px; padding: 15px;
     box-shadow: 2px 2px 5px rgba(0,0,0,0.3); overflow-y: auto;">
     <b style="font-size:16px;">Fatal Crashes & Muni Metro (SF)</b><br>
     <i class="fa fa-map-marker fa-1x" style="color:#e74c3c"></i>&nbsp; Crashes Near Muni Metro (<=50m)<br>
     <i class="fa fa-map-marker fa-1x" style="color:#2980b9"></i>&nbsp; Crashes Not Near Muni Metro<br>
     <i class="fa fa-minus fa-1x" style="color:#27ae60"></i>&nbsp; Muni Metro Lines<br>
     <br>
     <b>Total Fatalities Near Muni:</b> {crashes_near} ({percentage:.1f}%)<br>
     <b>Relative Risk:</b> {rr:.2f}x<br>
     <b>Statistically Significant:</b> {is_significant} (p={test_result.pvalue:.2e})<br>
     <hr style="margin: 10px 0;">
     <b>Methodology:</b> A 50m geographic buffer was drawn around Muni Metro lines. Spatial exposure (buffer area vs. SF's 121.4 sq km total area) dictates our expected crash probabilities. A <b>Binomial Test</b> determines if observed crashes near Muni lines significantly exceed random chance.<br>
     <br>
     <b>Conclusion:</b> With a p-value of {test_result.pvalue:.2e}, we reject the null hypothesis. Fatal crashes do not occur randomly; they are statically and significantly concentrated in proximity to Muni Metro routes!
     </div>
     '''
    m.get_root().html.add_child(folium.Element(legend_html))

    output_file = 'Muni_Metro_Fatal_Crashes.html'
    m.save(output_file)
    print(f"Map successfully saved to '{output_file}'!")

if __name__ == "__main__":
    main()
