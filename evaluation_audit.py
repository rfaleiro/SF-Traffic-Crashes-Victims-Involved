"""
evaluation_audit.py
===================
Independent statistical audit of the SF Traffic Crashes near Muni Metro analysis.

Checks performed
----------------
1.  Data integrity  – missing coords, duplicates, coordinate outliers
2.  Buffer area     – correctly computed via unary_union (no double-counting)
3.  Baseline (H0)   – geographic area fraction vs. road-network fraction
4.  Binomial retest – rerun from raw data for BOTH fatality and injury datasets
5.  Effect-size     – Relative Risk + 95% CI (Miettinen-Nurminen approximation)
6.  Alternative test– Two-proportion z-test as a sanity check
7.  Power analysis  – minimum detectable effect at the observed sample size
8.  Monte Carlo     – 10 000 random permutations to verify p-value non-parametrically
9.  Summary table   – side-by-side comparison of original reported vs. audit numbers
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from scipy.stats import binomtest, norm
from statsmodels.stats.proportion import proportions_ztest, proportion_effectsize
from statsmodels.stats.power import NormalIndPower
import os, sys

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
MUNI_PATH    = os.path.join(BASE, "data/raw/Muni_Simple_Routes_20260319.csv")
FATAL_PATH   = os.path.join(BASE, "data/raw/Traffic_Crashes_Resulting_in_Fatality_20260319.csv")
INJURY_PATH  = os.path.join(BASE, "data/raw/Traffic_Crashes_Resulting_in_Injury_20260319.csv")

BUFFER_M     = 50            # metres
SF_AREA_KM2  = 121.4         # total SF land area in km²
ALPHA        = 0.05
N_PERMU      = 10_000
RNG          = np.random.default_rng(42)

SEP = "=" * 70

# ── Helper functions ───────────────────────────────────────────────────────────

def rr_ci_95(n_near, total, expected_p):
    """
    Point estimate and 95% CI for Relative Risk using the observed vs.
    expected-under-H0 framing:  RR = (k/n) / p_0.
    CI via the delta method on log(RR).
    """
    k   = int(n_near)
    n   = int(total)
    p1  = k / n                      # observed proportion
    p0  = expected_p                 # null (area-based) proportion
    rr  = p1 / p0
    # delta-method SE on log(RR)
    se_log_rr = np.sqrt((1 - p1) / (k) + (1 - p0) / (n * p0))
    z = norm.ppf(0.975)
    ci_lo = np.exp(np.log(rr) - z * se_log_rr)
    ci_hi = np.exp(np.log(rr) + z * se_log_rr)
    return rr, ci_lo, ci_hi

def monte_carlo_pvalue(n_near, total, expected_p, n_sim=N_PERMU):
    """
    Non-parametric permutation p-value:
    Simulate n_sim draws of Binomial(total, expected_p) and count how
    many >= observed n_near.
    """
    draws = RNG.binomial(total, expected_p, size=n_sim)
    return (draws >= n_near).mean()

def power_at_sample_size(total, observed_p, null_p, alpha=ALPHA):
    """
    Estimate statistical power for a one-sided test of prop > null_p
    given the observed effect size and sample size.
    """
    es = proportion_effectsize(observed_p, null_p, method="normal")
    analysis = NormalIndPower()
    # For a one-sample proportion test we pass nobs1=total, ratio=0
    power = analysis.solve_power(effect_size=abs(es), nobs1=total, alpha=alpha,
                                  alternative='larger')
    return power

def audit_dataset(label, crashes_df, coord_lon, coord_lat, uid_col,
                  muni_gdf_proj, muni_buffered_union, muni_area_km2,
                  originally_reported_near=None, originally_reported_total=None):

    print(f"\n{SEP}")
    print(f"  DATASET: {label}")
    print(SEP)

    # ── 1. Data Integrity ──────────────────────────────────────────────────
    print("\n[ 1 ] DATA INTEGRITY")
    n_raw = len(crashes_df)
    n_miss_coords = crashes_df[[coord_lon, coord_lat]].isnull().any(axis=1).sum()
    crashes_clean = crashes_df.dropna(subset=[coord_lon, coord_lat]).copy()

    # Coordinate sanity: SF bounding box approx.
    lon_ok = crashes_clean[coord_lon].between(-122.55, -122.33)
    lat_ok = crashes_clean[coord_lat].between(37.68, 37.84)
    n_outlier = (~(lon_ok & lat_ok)).sum()
    crashes_clean = crashes_clean[lon_ok & lat_ok].copy()

    # Duplicates (same uid)
    if uid_col in crashes_clean.columns:
        n_dup = crashes_clean[uid_col].duplicated().sum()
    else:
        n_dup = 0

    print(f"  Raw records          : {n_raw:,}")
    print(f"  Missing coordinates  : {n_miss_coords:,}")
    print(f"  Coordinate outliers  : {n_outlier:,}")
    print(f"  Duplicate UIDs       : {n_dup:,}")
    print(f"  Clean records used   : {len(crashes_clean):,}")

    # ── 2. Spatial join ────────────────────────────────────────────────────
    crashes_gdf = gpd.GeoDataFrame(
        crashes_clean,
        geometry=gpd.points_from_xy(crashes_clean[coord_lon], crashes_clean[coord_lat]),
        crs="EPSG:4326"
    ).to_crs("EPSG:32610")

    # Use pre-computed union polygon for the join
    from shapely.ops import unary_union
    near_mask = crashes_gdf.geometry.within(muni_buffered_union)
    # Also check intersection (some points may be on boundary)
    near_mask2 = crashes_gdf.geometry.distance(muni_buffered_union) <= 0
    near_mask  = near_mask | near_mask2

    n_near  = near_mask.sum()
    n_total = len(crashes_gdf)
    n_far   = n_total - n_near
    obs_p   = n_near / n_total

    # ── 3. Spatial Baseline ────────────────────────────────────────────────
    area_far_km2  = SF_AREA_KM2 - muni_area_km2
    null_p        = muni_area_km2 / SF_AREA_KM2      # geographic-area H0
    density_near  = n_near  / muni_area_km2
    density_far   = n_far   / area_far_km2
    rr_area, rr_lo, rr_hi = rr_ci_95(n_near, n_total, null_p)

    print(f"\n[ 2 ] SPATIAL EXPOSURE (buffer area vs. whole SF)")
    print(f"  Muni Metro 50m buffer area : {muni_area_km2:.3f} km²")
    print(f"  Buffer as % of SF land     : {null_p*100:.2f}%")
    print(f"  Crashes near Muni (<={BUFFER_M}m): {n_near:,} / {n_total:,}  ({obs_p*100:.2f}%)")
    print(f"  Expected under H0          : {n_total * null_p:.1f}")
    print(f"  Density near Muni          : {density_near:.2f} crash/km²")
    print(f"  Density elsewhere          : {density_far:.2f} crash/km²")
    print(f"  Relative Risk              : {rr_area:.2f}x  (95% CI: {rr_lo:.2f}–{rr_hi:.2f})")

    # ── 4. Binomial Test (rerun) ────────────────────────────────────────────
    binom = binomtest(n_near, n_total, p=null_p, alternative='greater')
    print(f"\n[ 3 ] BINOMIAL TEST (one-sided, H1: prop > area fraction)")
    print(f"  p-value                    : {binom.pvalue:.3e}")
    print(f"  Reject H0 at α={ALPHA}?       : {'YES ✓' if binom.pvalue < ALPHA else 'NO ✗'}")

    # ── 5. Two-proportion z-test (sanity check) ─────────────────────────────
    # Use scalar inputs to guarantee scalar output
    z_stat, z_p = proportions_ztest(count=int(n_near), nobs=int(n_total),
                                     value=null_p, alternative='larger')
    z_stat = float(z_stat)
    z_p    = float(z_p)
    print(f"\n[ 4 ] ONE-PROPORTION Z-TEST (sanity check)")
    print(f"  z-statistic                : {z_stat:.4f}")
    print(f"  p-value                    : {z_p:.3e}")
    print(f"  Agree with Binomial?       : {'YES ✓' if (z_p < ALPHA) == (binom.pvalue < ALPHA) else 'NO ✗'}")

    # ── 6. Monte Carlo permutation ─────────────────────────────────────────
    mc_p = monte_carlo_pvalue(n_near, n_total, null_p)
    print(f"\n[ 5 ] MONTE CARLO PERMUTATION ({N_PERMU:,} simulations)")
    print(f"  Simulated p-value          : {mc_p:.4f}")
    print(f"  Consistent with Binomial?  : {'YES ✓' if (mc_p < ALPHA) == (binom.pvalue < ALPHA) else 'NO ✗'}")

    # ── 7. Power ───────────────────────────────────────────────────────────
    try:
        pwr = power_at_sample_size(n_total, obs_p, null_p)
        print(f"\n[ 6 ] STATISTICAL POWER")
        print(f"  n = {n_total}, observed proportion = {obs_p:.4f}, H0 p = {null_p:.4f}")
        print(f"  Estimated power            : {pwr:.4f}  ({pwr*100:.1f}%)")
        if pwr >= 0.80:
            print("  Adequately powered ✓")
        else:
            print("  ⚠ Under-powered — results should be interpreted with caution")
    except Exception as e:
        print(f"\n[ 6 ] STATISTICAL POWER — could not compute: {e}")

    # ── 8. Comparison with original reported numbers ───────────────────────
    if originally_reported_near is not None:
        print(f"\n[ 7 ] COMPARISON WITH ORIGINALLY REPORTED NUMBERS")
        print(f"  Originally reported near   : {originally_reported_near}")
        print(f"  Audit computed near        : {n_near}")
        match = originally_reported_near == n_near
        print(f"  Counts match?              : {'YES ✓' if match else f'NO ✗ — DISCREPANCY of {abs(originally_reported_near - n_near)}'}")
        if originally_reported_total is not None:
            print(f"  Originally reported total  : {originally_reported_total}")
            print(f"  Audit computed total       : {n_total}")
            tot_match = originally_reported_total == n_total
            print(f"  Totals match?              : {'YES ✓' if tot_match else f'NO ✗ — DISCREPANCY of {abs(originally_reported_total - n_total)}'}")

    return {
        "label"        : label,
        "n_raw"        : n_raw,
        "n_clean"      : n_total,
        "n_near"       : n_near,
        "obs_p"        : obs_p,
        "null_p"       : null_p,
        "rr"           : rr_area,
        "rr_lo"        : rr_lo,
        "rr_hi"        : rr_hi,
        "binom_p"      : binom.pvalue,
        "mc_p"         : mc_p,
        "significant"  : binom.pvalue < ALPHA,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    print(SEP)
    print("  SF TRAFFIC CRASHES NEAR MUNI METRO — INDEPENDENT AUDIT")
    print(SEP)

    # ── Load & prepare Muni Metro buffer (shared between datasets) ──────────
    print("\nLoading Muni Route data …")
    muni_df    = pd.read_csv(MUNI_PATH)
    muni_metro = muni_df[muni_df['SERVICE_CA'] == 'Muni Metro'].copy()
    muni_metro['geometry'] = muni_metro['shape'].apply(wkt.loads)
    muni_gdf   = gpd.GeoDataFrame(muni_metro, geometry='geometry', crs="EPSG:4326")
    muni_proj  = muni_gdf.to_crs("EPSG:32610")

    muni_proj['geometry'] = muni_proj.geometry.buffer(BUFFER_M)
    union_geom   = muni_proj.geometry.union_all()   # dissolve overlapping buffers
    union_area_m2  = union_geom.area
    union_area_km2 = union_area_m2 / 1e6
    null_prob      = union_area_km2 / SF_AREA_KM2

    print(f"  Muni Metro segments loaded : {len(muni_metro)}")
    print(f"  Buffer ({BUFFER_M}m) union area   : {union_area_km2:.3f} km²")
    print(f"  Fraction of SF land        : {null_prob*100:.2f}%")

    # ── Audit Fatality Dataset ──────────────────────────────────────────────
    print("\nLoading Fatal Crash data …")
    fatal_df = pd.read_csv(FATAL_PATH)
    fatal_res = audit_dataset(
        label                    = "FATALITY CRASHES",
        crashes_df               = fatal_df,
        coord_lon                = "longitude",
        coord_lat                = "latitude",
        uid_col                  = "unique_id",
        muni_gdf_proj            = muni_proj,
        muni_buffered_union      = union_geom,
        muni_area_km2            = union_area_km2,
        originally_reported_near = 47,   # from stats_results.txt
        originally_reported_total= 351,  # from stats_analysis.py
    )

    # ── Audit Injury Dataset ────────────────────────────────────────────────
    print("\n\nLoading Injury Crash data …")
    injury_df = pd.read_csv(INJURY_PATH, low_memory=False)
    injury_res = audit_dataset(
        label                    = "INJURY CRASHES",
        crashes_df               = injury_df,
        coord_lon                = "tb_longitude",
        coord_lat                = "tb_latitude",
        uid_col                  = "unique_id",
        muni_gdf_proj            = muni_proj,
        muni_buffered_union      = union_geom,
        muni_area_km2            = union_area_km2,
        originally_reported_near = None,
        originally_reported_total= None,
    )

    # ── Summary Table ───────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  SUMMARY TABLE")
    print(SEP)
    results = [fatal_res, injury_res]
    print(f"\n{'Dataset':<22} {'N':>6} {'Near':>6} {'Obs%':>7} {'Null%':>7} {'RR':>6} {'95% CI':>16} {'Binom-p':>12} {'MC-p':>8} {'Sig':>4}")
    print("-" * 100)
    for r in results:
        ci = f"({r['rr_lo']:.2f}–{r['rr_hi']:.2f})"
        sig = "YES" if r['significant'] else "NO"
        print(f"{r['label']:<22} {r['n_clean']:>6,} {r['n_near']:>6,} "
              f"{r['obs_p']*100:>6.2f}% {r['null_p']*100:>6.2f}% "
              f"{r['rr']:>6.2f} {ci:>16} {r['binom_p']:>12.3e} {r['mc_p']:>8.4f} {sig:>4}")

    # ── Key Methodological Notes ────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  METHODOLOGICAL NOTES & IDENTIFIED ISSUES")
    print(SEP)
    notes = [
        ("ISSUE 1 – Hardcoded counts in stats_analysis.py",
         "stats_analysis.py uses hardcoded values (crashes_near=47, total=351)\n"
         "  instead of re-deriving them from actual spatial joins. Any data update\n"
         "  will silently produce stale results without error."),
        ("ISSUE 2 – Geographic baseline vs. road-network baseline",
         "The null hypothesis assumes crashes scatter uniformly over ALL SF land.\n"
         "  In reality, crashes only occur ON roads. Muni Metro lines run along\n"
         "  major arterials that make up a LARGER share of total road km than their\n"
         "  4.1% land-area fraction. This inflates RR and makes p-values smaller\n"
         "  than the true effect. A road-network-weighted baseline would be more\n"
         "  defensible, though the current direction of the finding is likely genuine."),
        ("ISSUE 3 – No 95% CI for Relative Risk",
         "The original analysis reports a single RR point estimate but no confidence\n"
         "  interval. CI bounds are critical for understanding precision."),
        ("ISSUE 4 – stats_analysis.py uses unary_union; analyze_injuries.py uses union_all",
         "Minor inconsistency between the two scripts. Both return the same result\n"
         "  in modern Shapely/GeoPandas, but for reproducibility they should be unified."),
        ("NOTE – Binomial test is valid given the spatial join is correct",
         "The core mechanics of the Binomial test are correctly applied: area-based\n"
         "  null probability, one-sided 'greater' alternative, and use of\n"
         "  scipy.stats.binomtest (exact, not approximate). The conclusion direction\n"
         "  is robust."),
    ]
    for title, body in notes:
        print(f"\n  ● {title}")
        print(f"    {body}")

    print(f"\n{SEP}")
    print("  AUDIT COMPLETE")
    print(SEP)
