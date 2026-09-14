# References & Data Sources

## Methodology fixes (v2) — sources behind each

- **Newey-West (HAC) standard errors**: Newey, W.K. & West, K.D. (1987).
  "A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix." *Econometrica*, 55(3),
  703–708. — Standard reference for robust standard errors under
  autocorrelated/heteroskedastic residuals, applied here via
  `statsmodels`' `cov_type='HAC'`.

- **Variance Inflation Factor (VIF)**: standard regression diagnostic for
  multicollinearity; implemented via
  `statsmodels.stats.outliers_influence.variance_inflation_factor`.

- **Discounted cash flow / present value of a cost stream**: standard
  corporate finance methodology (see any corporate finance textbook,
  e.g. Berk & DeMarzo, *Corporate Finance*) — applied here to discount
  a stream of future carbon costs to present value at a WACC discount
  rate, net of a corporate tax shield.

This project is original code written for this exercise. It is *inspired
by* and *built on top of* publicly available frameworks, datasets, and
academic research listed below. None of the code, text, or figures in
this repository are copied from these sources — they are cited here for
transparency and attribution, and so you can go read the originals
yourself.

## Academic research (concept basis for the "Carbon Beta" model)

- Bolton, P. & Kacperczyk, M. (2021). "Do investors care about carbon
  risk?" *Journal of Financial Economics*, 142(2), 517–549.
  https://doi.org/10.1016/j.jfineco.2021.05.008
  — Introduced the idea of regressing stock returns against a carbon
  emissions factor to test whether markets price carbon risk. This
  project's two-factor regression (market + carbon) is a simplified,
  independently-coded version of that general approach, using a
  different (ETF-based) carbon factor for practicality.

- Bolton, P. & Kacperczyk, M. (2023). "Global Pricing of Carbon-Transition
  Risk." *The Journal of Finance*, 78, 3677–3754.
  https://doi.org/10.1111/jofi.13272
  — Follow-up study extending the carbon premium finding to a global,
  cross-country sample.

## Climate scenario framework

- **NGFS (Network for Greening the Financial System)** Scenario
  Framework — the "Orderly / Disorderly / Hothouse World" scenario
  categories used in `src/scenarios.py` are named after and loosely
  shaped on NGFS's public scenario taxonomy.
  https://www.ngfs.net/en/publications/ngfs-scenarios
  — Note: the actual carbon price numbers in this project are simplified
  and illustrative, not the official NGFS Scenario Explorer output. For
  a rigorous version, download the real data (free) from the NGFS
  Scenario Explorer: https://data.ene.iiasa.ac.at/ngfs/

## Emissions / carbon intensity reference concepts

- **CDP (formerly Carbon Disclosure Project)** — global platform where
  companies disclose emissions data. Referenced as the recommended real
  data source to replace this project's illustrative sector averages.
  https://www.cdp.net/

- **U.S. EPA Greenhouse Gas Reporting Program (GHGRP)** — public U.S.
  facility-level emissions data. Referenced as another recommended real
  data source. https://www.epa.gov/ghgreporting

## Market data

- **Yahoo Finance**, accessed via the open-source `yfinance` Python
  library, for company fundamentals and historical stock prices.
  https://finance.yahoo.com / https://github.com/ranaroussi/yfinance

- **KraneShares Global Carbon Strategy ETF (KRBN)** — used as a real,
  traded proxy for carbon allowance market prices (EU ETS, California
  Cap-and-Trade, RGGI) in the Carbon Beta regression.
  https://kraneshares.com/krbn/

## Software / libraries used

All open-source, used under their respective licenses (MIT/BSD-style):

- Python — https://www.python.org/
- pandas — https://pandas.pydata.org/
- NumPy — https://numpy.org/
- scikit-learn (K-means clustering) — https://scikit-learn.org/
- statsmodels (OLS regression) — https://www.statsmodels.org/
- Streamlit (dashboard framework) — https://streamlit.io/
- Plotly (interactive charts) — https://plotly.com/python/
- yfinance — https://github.com/ranaroussi/yfinance

## A note on academic integrity

If you use this project for a course assignment, cite it as your own
original implementation, and cite the papers/frameworks above the same
way you'd cite any methodology you built on. If you present it in an
interview or on a resume, it's honest and accurate to describe it as:
"a project I built applying concepts from NGFS climate scenarios and
academic carbon-risk-pricing research" — you did write all the code
yourself; you're crediting the ideas, which is exactly what good
attribution looks like.
