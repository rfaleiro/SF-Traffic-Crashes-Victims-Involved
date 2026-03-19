"""
analyze_elderly_risk.py
=======================
Analysis of crash probability for parties aged 70+ in San Francisco.

Null Hypothesis (H0): The proportion of crash-involved parties who are >= 70
  is equal to or less than their demographic representation in SF (11.9%).
Alternative (H1): The proportion is greater than expected (one-tailed).

Dataset: Traffic_Crashes_Resulting_in_Injury__Victims_Involved
"""

import pandas as pd
import geopandas as gpd
from scipy.stats import binomtest, norm
import numpy as np
import folium
import os
import warnings

warnings.filterwarnings('ignore')

SF_SENIOR_PCT = 0.119  # 11.91% based on Census ACS
AGE_THRESHOLD = 70

raw_path = '../data/raw/Traffic_Crashes_Resulting_in_Injury__Victims_Involved_20260319.csv'

def main():
    print(f"Loading Injury Victims dataset from {raw_path}...")
    df = pd.read_csv(raw_path, low_memory=False)
    
    # Ensure party_age is numeric
    df['party_age'] = pd.to_numeric(df['party_age'], errors='coerce')
    
    # We analyze risk at the CRASH level.
    # Group by unique_id (crash ID) and find the maximum party age involved in that specific crash.
    crash_max_age = df.groupby('unique_id')['party_age'].max().dropna()
    
    total_crashes_with_age = len(crash_max_age)
    elderly_crashes = (crash_max_age >= AGE_THRESHOLD).sum()
    
    if total_crashes_with_age == 0:
        print("Error: No valid party ages found.")
        return

    expected_crashes = total_crashes_with_age * SF_SENIOR_PCT
    observed_pct = elderly_crashes / total_crashes_with_age
    rr = observed_pct / SF_SENIOR_PCT
    
    # 95% CI for RR via Delta Method
    obs_p = observed_pct
    if elderly_crashes > 0 and obs_p < 1:
        se_log_rr = np.sqrt((1 - obs_p)/elderly_crashes + (1 - SF_SENIOR_PCT)/(total_crashes_with_age * SF_SENIOR_PCT))
        z_crit = norm.ppf(0.975)
        rr_lo = np.exp(np.log(rr) - z_crit * se_log_rr)
        rr_hi = np.exp(np.log(rr) + z_crit * se_log_rr)
    else:
        rr_lo = rr_hi = float('nan')
        
    test_result = binomtest(elderly_crashes, total_crashes_with_age, p=SF_SENIOR_PCT, alternative='two-sided')
    p_str = "< 0.0001" if test_result.pvalue < 0.0001 else f"{test_result.pvalue:.4f}"
    is_significant = "Yes" if test_result.pvalue < 0.05 else "No"
    
    print("-" * 55)
    print(f"Elderly Crash Risk Analysis (Age >= {AGE_THRESHOLD})")
    print("-" * 55)
    print(f"Total crashes with known party age : {total_crashes_with_age:,}")
    print(f"Crashes involving party >= {AGE_THRESHOLD:2}     : {elderly_crashes:,} ({observed_pct*100:.2f}%)")
    print(f"Expected crashes (Census 11.9%)    : {expected_crashes:.2f}")
    print(f"Relative Risk                      : {rr:.2f}x (95% CI: {rr_lo:.2f}-{rr_hi:.2f})")
    print(f"P-value                            : {p_str}")
    print(f"Statistically Significant (≠11.9%) : {is_significant}")
    print("-" * 55)
    
    # Save the text output
    with open('elderly_stats_output.txt', 'w') as f:
        f.write(f"Total crashes with known party age : {total_crashes_with_age:,}\n")
        f.write(f"Crashes involving party >= {AGE_THRESHOLD:2}     : {elderly_crashes:,} ({observed_pct*100:.2f}%)\n")
        f.write(f"Relative Risk                      : {rr:.2f}x (95% CI: {rr_lo:.2f}-{rr_hi:.2f})\n")
        f.write(f"P-value                            : {p_str}\n")
    
    # Generate Map for the crashes involving elderly
    print("\nExtracting geographic data for map...")
    elderly_ids = crash_max_age[crash_max_age >= AGE_THRESHOLD].index
    
    # Filter the original dataframe for just one row per elderly crash to plot
    mappable_df = df[df['unique_id'].isin(elderly_ids)].drop_duplicates(subset=['unique_id'])
    mappable_df = mappable_df.dropna(subset=['tb_latitude', 'tb_longitude'])
    mappable_df = mappable_df[(mappable_df['tb_longitude'] > -123) & (mappable_df['tb_longitude'] < -121)]
    
    if len(mappable_df) > 0:
        print(f"Plotting {len(mappable_df)} elderly-involved crashes to map...")
        sf_coords = [37.7749, -122.4194]
        m = folium.Map(location=sf_coords, zoom_start=12, tiles='CartoDB Positron')
        
        for idx, row in mappable_df.iterrows():
            tooltip_html = (
                f"<b>Crash ID:</b> {row['unique_id']}<br>"
                f"<b>Date:</b> {row.get('collision_date', 'Unknown')}<br>"
                f"<b>Severity:</b> {row.get('collision_severity', 'Unknown')}<br>"
                f"<b>Type:</b> {row.get('type_of_collision', 'Unknown')}"
            )
            folium.CircleMarker(
                location=[row['tb_latitude'], row['tb_longitude']],
                radius=4,
                color='#8e44ad',
                fill=True,
                fill_color='#9b59b6',
                fill_opacity=0.6,
                tooltip=tooltip_html,
                weight=1
            ).add_to(m)
            
        legend_html = f'''
         <div style="position: fixed; 
         bottom: 50px; left: 50px; width: 450px; height: auto; max-height: 400px; 
         background-color: white; border:2px solid grey; z-index:9999; font-size:14px; padding: 15px;
         box-shadow: 2px 2px 5px rgba(0,0,0,0.3); overflow-y: auto;">
         <b style="font-size:16px;">Crashes Involving Parties 70+ (SF)</b><br>
         <i class="fa fa-map-marker fa-1x" style="color:#8e44ad"></i>&nbsp; Elderly-Involved Crash Location<br>
         <br>
         <b>SF Population Rate (70+):</b> {SF_SENIOR_PCT*100:.1f}%<br>
         <b>Observed Crash Rate (70+):</b> {observed_pct*100:.2f}%<br>
         <b>Relative Risk:</b> {rr:.2f}x (95% CI: {rr_lo:.2f}-{rr_hi:.2f})<br>
         <b>Statistically Significant:</b> {is_significant} (p={p_str})<br>
         <hr style="margin: 10px 0;">
         <b>Conclusion:</b> Compared to their {SF_SENIOR_PCT*100:.1f}% demographic representation in San Francisco, parties aged 70 or older are involved in injury crashes at a significantly {'higher' if rr > 1 else 'lower'} rate.
         </div>
         '''
        m.get_root().html.add_child(folium.Element(legend_html))
        m.save('Elderly_Involved_Crashes_Map.html')
        print("Done! Map saved to 'Elderly_Involved_Crashes_Map.html'.")
    else:
        print("No valid mapped coordinates found for elderly crashes.")

if __name__ == "__main__":
    main()
