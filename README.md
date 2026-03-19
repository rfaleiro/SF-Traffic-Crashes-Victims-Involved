# SF Traffic Crashes: Victims Involved near Muni Metro Lines

This project investigates the spatial relationship between traffic crashes (fatalities and injuries) and **Muni Metro** tram public transit lines in San Francisco, California.

By employing a geographic buffer of **50 meters** around the tram network, this analysis maps out specific incidents and runs statistical hypothesis tests (Binomial Tests) to determine if severe crashes disproportionally occur in proximity to these specific transit corridors.

## 🗺️ Interactive Web Maps

The core outputs of this analysis are interactable, layer-driven Folium maps directly hosted via GitHub Pages:

* 🔴 **[Fatal Crashes vs. Muni Metro Lines](https://rfaleiro.github.io/SF-Traffic-Crashes-Victims-Involved/Muni_Metro_Fatal_Crashes.html)**
* 🟠 **[Injury Crashes vs. Muni Metro Lines](https://rfaleiro.github.io/SF-Traffic-Crashes-Victims-Involved/Muni_Metro_Injury_Crashes.html)**

## 📊 Statistical Focus & Results

The objective is to accurately measure **Spatial Exposure** by computing the `m²` land area of the buffered Muni corridors versus the rest of San Francisco (121.4 sq km). Using `scipy`, a **Binomial hypothesis test** was conducted to test the observed crash volumes versus the expected crash proportions if crashes scattered completely randomly.

**Methodology & Conclusion:**

* **The Baseline (Null Hypothesis):** If all injury/fatal crashes happened completely randomly across San Francisco's 121.4 sq km, the probability of a crash landing inside the 50m Muni line buffer would be equal to the percentage of land the buffer takes up.
* **The Findings:** Out of practically all datasets mapped, a massive amount of crashes happened within 50 meters of a Muni line, leading to an extremely high Relative Risk multiplier.
* **P-Value & Significance:** The Binomial test returns an infinitesimally small P-value (e.g., `< 0.05`), thereby giving overwhelming mathematical proof to reject the null hypothesis. Crashes are definitively highly concentrated around Muni lines rather than dispersed randomly across the city!

### 📈 Direct Results (Within a 50-meter buffer)
*The Muni Metro 50m proximity zone makes up roughly **4.10%** of San Francisco's total land area.*

| Dataset | Total SF Crashes | Near Muni (<50m) | Relative Risk | 95% CI | P-Value | Significant? |
|---|---|---|---|---|---|---|
| **Fatalities** | 351 | 47 *(13.39%)* | **3.62x** | 2.04 — 6.41 | `< 0.0001` | ✅ YES |
| **Injuries** | 5,914 | 771 *(13.04%)* | **3.51x** | 3.05 — 4.04 | `< 0.0001` | ✅ YES |

### 📝 Raw Statistical Output

```text
=========================================================
  SPATIAL EXPOSURE SETUP
=========================================================
SF Total Land Area        : 121.40 sq km
Muni Metro 50m Zone Area: 4.972 sq km
Muni Metro Zone % of SF   : 4.10%

=========================================================
  FATAL CRASHES
=========================================================
Total mapped crashes      : 351
Crashes near Muni Metro   : 47 (13.39%)
Expected crashes (random) : 14.37
Density near Muni Metro   : 9.45 crashes / sq km
Density elsewhere         : 2.61 crashes / sq km
Relative Risk             : 3.62x (95% CI: 2.04–6.41)
P-value                   : < 0.0001
Conclusion                : Fatal Crashes are STATISTICALLY SIGNIFICANTLY concentrated near Muni Metro lines.

=========================================================
  INJURY CRASHES
=========================================================
Total mapped crashes      : 5,914
Crashes near Muni Metro   : 771 (13.04%)
Expected crashes (random) : 242.19
Density near Muni Metro   : 155.08 crashes / sq km
Density elsewhere         : 44.17 crashes / sq km
Relative Risk             : 3.51x (95% CI: 3.05–4.04)
P-value                   : < 0.0001
Conclusion                : Injury Crashes are STATISTICALLY SIGNIFICANTLY concentrated near Muni Metro lines.
```

## 📂 Official Data Sources

All datasets used in this geospatial analysis are provided by **DataSF**, the official open data portal for the City and County of San Francisco:

* [**Traffic Crashes Resulting in Injury**](https://data.sfgov.org/Public-Safety/Traffic-Crashes-Resulting-in-Injury-Victims-Involv/nwes-mmgh/about_data)
* [**Traffic Crashes Resulting in Fatality**](https://data.sfgov.org/Public-Safety/Traffic-Crashes-Resulting-in-Fatality/dau3-4s8f/about_data)
* [**Muni Simple Routes**](https://data.sfgov.org/Transportation/Muni-Simple-Routes/9exe-acju/about_data)

## 💻 Tech Stack

* **Geospatial & Data:** `pandas`, `geopandas`, `shapely` for vector processing natively mapped to UTM Zone 10N (`EPSG:32610`).
* **Analysis:** Statistical computation via `scipy.stats` (Binomial Testing / Relative Risk).
* **Visualization:** Interactive HTML CartoDB tiles plotted using `folium`.
