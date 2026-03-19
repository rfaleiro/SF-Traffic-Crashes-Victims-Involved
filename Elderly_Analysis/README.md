# SF Traffic Crashes: Elderly (70+) Involvement Analysis

This sub-project investigates whether parties aged 70 and older are disproportionately involved in traffic crashes across San Francisco.

Instead of a spatial geographic baseline (like the Muni Metro buffer analysis), this analysis relies on a **demographic baseline**.

## The Baseline Hypothesis

According to the U.S. Census Bureau (ACS 5-Year Estimates), **11.9%** of San Francisco's population is 70 years old or older.

* **Null Hypothesis (H0):** If crash involvement is evenly distributed across age demographics, roughly 11.9% of all crashes should involve a party aged 70+.
* **Observed Reality:** We parse the specific `Victims_Involved` open dataset to find out what percentage of crashes *actually* involve someone 70+.
* **Calculation:** We compute the Relative Risk multiplier and run a Binomial Hypothesis test to determine if the variance from 11.9% is statistically significant.
