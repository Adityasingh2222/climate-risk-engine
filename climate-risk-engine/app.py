"""
Climate Transition Risk Engine - Streamlit Dashboard
------------------------------------------------------
Run with:  streamlit run app.py

Lets a user upload (or use the sample) portfolio, pick an NGFS-style
climate scenario, and see how a rising carbon price would hit portfolio
value across sectors and over time — backed by an NPV/DCF valuation
model, a statistical carbon-beta regression, and ML risk clustering.

ALL core assumptions (carbon prices, sector emissions intensity, WACC,
tax rate) live in /config/*.yaml and can be edited there without
touching this file — Streamlit re-reads config on every run.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.portfolio import Holding, run_scenario_year, run_full_scenario, load_portfolio
from src.scenarios import SCENARIOS
from src.data_loader import list_sample_tickers
from src.market_data import fetch_price_history
from src.carbon_beta import estimate_carbon_betas, compute_factor_vif
from src.clustering import build_feature_matrix, cluster_risk_archetypes
from src.climate_data import SECTOR_CARBON_INTENSITY, get_sector_note
from src.risk_model import DEFAULT_WACC, DEFAULT_TAX_RATE, DEFAULT_VALUATION_YEAR, DEFAULT_HORIZON_YEAR
from src.optimizer import compute_expected_returns_and_cov, minimize_carbon_for_target_return, build_green_efficient_frontier

st.set_page_config(page_title="Climate Transition Risk Engine", layout="wide")

st.title("🌍 Climate Transition Risk Engine")
st.caption(
    "Estimates portfolio value-at-risk from carbon-price shocks under "
    "NGFS-style climate transition scenarios — using a discounted cash "
    "flow valuation model, cross-checked against real market data."
)

# ---------------- Sidebar controls ----------------
st.sidebar.header("Settings")
st.sidebar.caption(
    "💡 All model assumptions below have defaults set in `/config/*.yaml` "
    "— edit those files to change them permanently, or use these controls "
    "to test scenarios interactively."
)

use_live = st.sidebar.toggle(
    "Use live market data (yfinance)", value=True,
    help="If off, or if live data isn't reachable, falls back to a small "
         "built-in sample dataset so the app always works offline.",
)

uploaded = st.sidebar.file_uploader(
    "Upload portfolio CSV (columns: ticker, weight)", type=["csv"]
)

if uploaded is not None:
    portfolio_df = pd.read_csv(uploaded)
else:
    st.sidebar.info(f"Using sample portfolio. Sample tickers available offline: "
                     f"{', '.join(list_sample_tickers())}")
    portfolio_df = pd.read_csv("sample_portfolio.csv")

portfolio_df["weight"] = portfolio_df["weight"] / portfolio_df["weight"].sum()
holdings = [Holding(r.ticker, r.weight) for r in portfolio_df.itertuples()]

scenario_key = st.sidebar.selectbox(
    "Climate scenario",
    options=list(SCENARIOS.keys()),
    format_func=lambda k: SCENARIOS[k]["label"],
)
st.sidebar.caption(SCENARIOS[scenario_key]["description"])

horizon_year = st.sidebar.slider(
    "Horizon year", min_value=2026, max_value=2050, value=2040, step=1,
    help="Value impact = present value of carbon costs from the valuation "
         "year up through this horizon year.",
)

with st.sidebar.expander("Advanced: valuation assumptions"):
    valuation_year = st.number_input("Valuation year (today)", value=DEFAULT_VALUATION_YEAR, step=1)
    wacc = st.slider("WACC (discount rate)", 0.02, 0.15, DEFAULT_WACC, 0.005, format="%.3f")
    tax_rate = st.slider("Corporate tax rate", 0.0, 0.40, DEFAULT_TAX_RATE, 0.01, format="%.2f")
    include_scope3 = st.toggle(
        "Include Scope 3 (supply chain) emissions", value=False,
        help="Uses full lifecycle emissions instead of direct-operations-only. "
             "Changes the risk picture a lot for sectors like Technology and "
             "Financial Services, where direct emissions are low but supply "
             "chain / financed emissions are large.",
    )

pass_through = st.sidebar.slider(
    "Cost pass-through to customers", 0.0, 1.0, 0.0, 0.05,
    help="Fraction of carbon cost companies can pass on via higher prices. "
         "0 = worst case (fully absorbed), 1 = fully passed on.",
)
abatement = st.sidebar.slider(
    "Emissions abatement vs. sector baseline", 0.0, 0.9, 0.0, 0.05,
    help="Fraction of emissions already avoided relative to the sector "
         "average, e.g. from efficiency investments or renewables.",
)

model_kwargs = dict(
    valuation_year=int(valuation_year), wacc=wacc, tax_rate=tax_rate,
    pass_through_rate=pass_through, abatement_rate=abatement,
    include_scope3=include_scope3,
)

# ---------------- Main panel ----------------
st.subheader(f"Portfolio holdings ({len(holdings)})")
st.dataframe(portfolio_df, use_container_width=True, hide_index=True)

with st.spinner("Running DCF risk model..."):
    year_df = run_scenario_year(holdings, scenario_key, horizon_year, use_live=use_live, **model_kwargs)

if year_df.empty:
    st.error("No valid holdings could be priced. Check your tickers.")
    st.stop()

col1, col2, col3 = st.columns(3)
carbon_price = year_df["carbon_price"].iloc[0]
portfolio_impact = year_df["weighted_value_impact_pct"].sum()
total_pv_cost = (year_df["pv_carbon_cost_usd"] * year_df["weight"]).sum()

col1.metric(f"Carbon price in {horizon_year}", f"${carbon_price:,.0f} / tCO2e")
col2.metric(f"Portfolio value impact (NPV through {horizon_year})", f"{portfolio_impact:.1%}")
col3.metric("Weighted PV of carbon costs", f"${total_pv_cost:,.0f}")
st.caption(
    f"Value impact = present value (at {wacc:.1%} WACC, {tax_rate:.0%} tax shield) of carbon "
    f"costs from {int(valuation_year)} through {horizon_year}, as a % of current market cap."
)

st.divider()

left, right = st.columns([3, 2])

with left:
    st.subheader(f"Value impact by holding — through {horizon_year}")
    chart_df = year_df.sort_values("value_impact_pct")
    fig = px.bar(
        chart_df, x="value_impact_pct", y="ticker", color="sector",
        orientation="h", labels={"value_impact_pct": "Value impact (%)", "ticker": ""},
        text=chart_df["value_impact_pct"].map(lambda x: f"{x:.1%}"),
    )
    fig.update_layout(xaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Sector exposure notes")
    for sector in year_df["sector"].unique():
        st.markdown(f"**{sector}**: {get_sector_note(sector)}")

st.divider()

st.subheader(f"Portfolio value impact over time — {SCENARIOS[scenario_key]['label']}")
with st.spinner("Running full scenario path..."):
    path_df = run_full_scenario(holdings, scenario_key, use_live=use_live, **model_kwargs)
fig2 = px.line(
    path_df, x="year", y="portfolio_value_impact_pct", markers=True,
    labels={"portfolio_value_impact_pct": "Cumulative NPV impact (%)", "year": "Horizon year"},
)
fig2.update_layout(yaxis_tickformat=".0%")
st.plotly_chart(fig2, use_container_width=True)
st.caption("Grows over time because it's the *cumulative* discounted cost of carbon from now through each horizon year — not a repeated single-year snapshot.")

st.subheader("Compare all three scenarios")
compare_rows = []
for key in SCENARIOS:
    df = run_full_scenario(holdings, key, use_live=use_live, **model_kwargs)
    df["scenario"] = SCENARIOS[key]["label"]
    compare_rows.append(df)
compare_df = pd.concat(compare_rows, ignore_index=True)
fig3 = px.line(
    compare_df, x="year", y="portfolio_value_impact_pct", color="scenario", markers=True,
    labels={"portfolio_value_impact_pct": "Cumulative NPV impact (%)", "year": "Horizon year"},
)
fig3.update_layout(yaxis_tickformat=".0%")
st.plotly_chart(fig3, use_container_width=True)

st.divider()
st.header("📊 Data Science: Empirical Carbon Beta")
st.caption(
    "A statistical cross-check: does the *market* actually price these "
    "stocks as carbon-exposed? Two-factor regression (market + real "
    "carbon-allowance ETF KRBN) with Newey-West (HAC) robust standard "
    "errors, since daily returns are volatility-clustered."
)

tickers = portfolio_df["ticker"].tolist()
with st.spinner("Fetching price history and running regressions..."):
    price_history = fetch_price_history(tickers, period="2y", use_live=use_live)
    betas_df = estimate_carbon_betas(price_history, tickers)
    vif = compute_factor_vif(price_history)

if price_history.shape[1] <= 2:
    st.info("Using synthetic price data for this demo (live market data unavailable/off).")

vif_carbon = vif.get("vif_carbon", float("nan"))
if vif_carbon == vif_carbon and vif_carbon > 5:  # NaN-safe check
    st.warning(
        f"⚠️ Market and carbon factors show meaningful collinearity (VIF = {vif_carbon:.1f}). "
        f"Individual market-vs-carbon coefficients may be unstable even though the combined "
        f"model fits well — treat beta_market and beta_carbon as less separable than usual."
    )
else:
    st.caption(f"Factor collinearity check: VIF = {vif_carbon:.2f} (< 5, not a concern).")

display_betas = betas_df.reset_index().rename(columns={
    "beta_carbon": "Carbon Beta", "beta_market": "Market Beta",
    "p_value_carbon": "p-value (HAC)", "r_squared": "R²",
})
st.dataframe(
    display_betas[["ticker", "Carbon Beta", "Market Beta", "p-value (HAC)", "R²", "significant_at_10pct"]]
    .round(3),
    use_container_width=True, hide_index=True,
)
st.caption(
    "**Carbon Beta < 0 and significant** → the stock historically underperforms "
    "when carbon prices rise. p-values use Newey-West HAC standard errors, robust "
    "to the autocorrelation/volatility clustering typical of daily financial returns."
)

fig_beta = px.bar(
    display_betas.sort_values("Carbon Beta"), x="Carbon Beta", y="ticker",
    orientation="h", color="Carbon Beta", color_continuous_scale="RdYlGn",
    labels={"ticker": ""},
)
st.plotly_chart(fig_beta, use_container_width=True)

st.divider()
st.header("🤖 Machine Learning: Risk Archetype Clustering")
st.caption(
    "Unsupervised clustering groups holdings into risk archetypes using "
    "sector carbon intensity, empirical carbon beta, margin, and valuation."
)

full_fund_df = load_portfolio(holdings, use_live=use_live)
feature_df = build_feature_matrix(full_fund_df, betas_df, SECTOR_CARBON_INTENSITY)

cc1, cc2 = st.columns(2)
n_clusters = cc1.slider("Number of clusters", 2, min(5, max(2, len(feature_df))), min(3, max(2, len(feature_df))))
method = cc2.selectbox(
    "Clustering method", ["kmeans", "agglomerative"],
    format_func=lambda m: "K-means" if m == "kmeans" else "Agglomerative (hierarchical)",
    help="Agglomerative tends to be more stable on small portfolios.",
)
clustered_df, centroids_df, cluster_warning = cluster_risk_archetypes(
    feature_df, n_clusters=n_clusters, method=method
)
if cluster_warning:
    st.warning(f"⚠️ {cluster_warning}")

col_a, col_b = st.columns([3, 2])
with col_a:
    if not centroids_df.empty:
        fig_cluster = px.scatter(
            clustered_df.reset_index(), x="carbon_intensity", y="beta_carbon",
            color="archetype", text="ticker", size="ebitda_margin",
            labels={"carbon_intensity": "Sector carbon intensity (tCO2e/$M)",
                    "beta_carbon": "Empirical carbon beta"},
        )
        fig_cluster.update_traces(textposition="top center")
        st.plotly_chart(fig_cluster, use_container_width=True)
with col_b:
    if not centroids_df.empty:
        st.markdown("**Cluster centroids**")
        st.dataframe(
            centroids_df[["archetype", "carbon_intensity", "beta_carbon", "ebitda_margin", "pe_ratio"]]
            .round(2),
            use_container_width=True, hide_index=True,
        )

st.dataframe(
    clustered_df.reset_index()[["ticker", "sector", "archetype", "carbon_intensity", "beta_carbon"]]
    .round(3),
    use_container_width=True, hide_index=True,
)

st.divider()
st.header("🎯 Green Efficient Frontier: Carbon-Optimized Portfolio")
st.caption(
    "Starting from your current holdings, what's the best mix that "
    "minimizes carbon transition risk while still hitting a target "
    "return? Solved via constrained optimization (scipy SLSQP): minimize "
    "carbon exposure subject to a minimum expected return, long-only, "
    "fully invested."
)

with st.spinner("Computing expected returns and solving the frontier..."):
    mu, cov = compute_expected_returns_and_cov(price_history, tickers)
    carbon_scores = -year_df.set_index("ticker")["value_impact_pct"]  # positive = more exposed
    frontier_df = build_green_efficient_frontier(tickers, mu, carbon_scores, n_points=12)

current_weights_vec = portfolio_df.set_index("ticker")["weight"].reindex(tickers).fillna(0).values
mu_vec_full = mu.reindex(tickers).fillna(0.0).values
carbon_vec_full = carbon_scores.reindex(tickers).fillna(carbon_scores.mean()).values
current_return = float(np.dot(current_weights_vec, mu_vec_full))
current_carbon = float(np.dot(current_weights_vec, carbon_vec_full))

if frontier_df.empty:
    st.info("Not enough price history to compute an efficient frontier for this portfolio.")
else:
    fig_frontier = go.Figure()
    fig_frontier.add_trace(go.Scatter(
        x=frontier_df["achieved_return"], y=frontier_df["carbon_exposure"],
        mode="lines+markers", name="Green Efficient Frontier",
        line=dict(color="#27ae60", width=2),
    ))
    fig_frontier.add_trace(go.Scatter(
        x=[current_return], y=[current_carbon], mode="markers",
        marker=dict(size=16, color="#c0392b", symbol="star"),
        name="Your current portfolio",
    ))
    fig_frontier.update_layout(
        title="Carbon Exposure vs. Expected Return",
        xaxis_title="Expected annual return", yaxis_title="Carbon risk exposure",
        xaxis_tickformat=".0%",
    )
    st.plotly_chart(fig_frontier, use_container_width=True)
    st.caption(
        "Every point on the green line is the *lowest possible carbon exposure* "
        "achievable at that return level, given your current holdings. Your "
        "actual portfolio (red star) sits above the line — the gap is unnecessary "
        "carbon exposure you aren't being compensated for in return."
    )

    lo, hi = float(frontier_df["achieved_return"].min()), float(frontier_df["achieved_return"].max())
    default_target = min(max(current_return, lo), hi)
    target_return = st.slider(
        "Target annual return for the optimized portfolio",
        lo, hi, default_target, (hi - lo) / 50 if hi > lo else 0.01,
    )

    opt_weights, opt_return, opt_carbon = minimize_carbon_for_target_return(
        tickers, mu, carbon_scores, target_return
    )

    if opt_weights is not None:
        opt_df = opt_weights.reset_index()
        opt_df.columns = ["ticker", "optimized_weight"]
        opt_df = opt_df.merge(
            portfolio_df[["ticker", "weight"]].rename(columns={"weight": "current_weight"}),
            on="ticker",
        )
        opt_df["change"] = opt_df["optimized_weight"] - opt_df["current_weight"]

        colx, coly, colz = st.columns(3)
        carbon_reduction = ((current_carbon - opt_carbon) / abs(current_carbon)) if current_carbon != 0 else 0
        colx.metric("Carbon exposure reduction", f"{carbon_reduction:.1%}")
        coly.metric("Optimized expected return", f"{opt_return:.1%}")
        colz.metric("Current expected return", f"{current_return:.1%}")

        fig_weights = px.bar(
            opt_df.sort_values("change"), x="change", y="ticker", orientation="h",
            title="Weight changes: optimized vs. current",
            labels={"change": "Weight change", "ticker": ""},
            color="change", color_continuous_scale="RdYlGn",
        )
        fig_weights.update_layout(xaxis_tickformat=".0%")
        st.plotly_chart(fig_weights, use_container_width=True)

        st.dataframe(opt_df.round(4), use_container_width=True, hide_index=True)
    else:
        st.warning("That target return isn't achievable from this portfolio's holdings — try a lower value.")

st.divider()

with st.expander("⚠️ Methodology & limitations (read this before quoting numbers)"):
    st.markdown("""
    - Carbon intensities and Scope 3 multipliers are **sector-level averages**
      from `config/sectors.yaml`, not company-specific emissions. Swap in real
      CDP or 10-K disclosed data for rigor.
    - Carbon price paths (`config/scenarios.yaml`) are **simplified,
      illustrative** versions of the NGFS scenario shapes, not the official
      NGFS Scenario Explorer output.
    - **Valuation** is now a proper multi-year DCF: carbon costs are
      discounted at WACC and tax-shielded, summed to a present value, and
      expressed as % of market cap. It still assumes constant revenue/margins
      (isolating the carbon-cost effect) and a single portfolio-wide WACC/tax
      rate rather than company-specific capital structures.
    - **Carbon Beta regression** uses Newey-West (HAC) robust standard errors
      and reports a VIF collinearity check between the market and carbon
      factors. A 2-year daily window is still short for a slow-moving
      structural relationship and won't capture regime shifts — a rolling-
      window beta would be a natural next step.
    - **Clustering** standardizes against the *full portfolio population*
      (not just the cluster centroids — an earlier bug) before ranking
      archetypes. K-means/agglomerative results on small portfolios
      (under ~15 holdings) are a starting hypothesis, not a final verdict —
      the app now warns explicitly when the sample is small.
    - Zero pass-through and zero abatement (the defaults) are a **worst-case**
      assumption. Use the sliders to test more realistic assumptions.
    - **Green Efficient Frontier** uses historical mean returns as the
      expected-return estimate — a well-known weak point of mean-variance
      optimization (Markowitz, 1952 and the literature since): historical
      averages are noisy, backward-looking, and small changes can swing
      optimized weights substantially ("estimation error maximization").
      A production version would use shrinkage estimators or forward-
      looking views (e.g. Black-Litterman) instead.
    """)
