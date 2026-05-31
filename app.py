"""
Causal Out-Strength Centrality Factor Model
Inspired by Stavroglou et al. (2019) — "Causality Networks of Financial Assets"
Data: Yahoo Finance via yfinance Ticker.history() — parallel, works on Streamlit Cloud
"""

import warnings
warnings.filterwarnings("ignore")

import io
import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns

from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from scipy import stats
from scipy.stats import spearmanr

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Causal Centrality Factor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# LIGHT THEME CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #F8F9FA; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E0E0E0;
    }

    /* Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E8ECEF;
        border-radius: 10px;
        padding: 18px 20px;
        margin: 6px 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .metric-label {
        font-size: 12px;
        color: #6B7280;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #111827;
        line-height: 1.2;
    }
    .metric-value.positive { color: #059669; }
    .metric-value.negative { color: #DC2626; }

    /* Section headers */
    .section-header {
        font-size: 18px;
        font-weight: 700;
        color: #1F2937;
        border-left: 4px solid #2C7BB6;
        padding-left: 12px;
        margin: 24px 0 12px 0;
    }

    /* Status badges */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-blue  { background: #DBEAFE; color: #1D4ED8; }
    .badge-green { background: #D1FAE5; color: #065F46; }
    .badge-red   { background: #FEE2E2; color: #991B1B; }
    .badge-gray  { background: #F3F4F6; color: #374151; }

    /* Hide default Streamlit footer */
    footer { visibility: hidden; }

    /* Dataframe styling */
    .dataframe { font-size: 13px; }

    /* Progress bar area */
    .stProgress > div > div { background-color: #2C7BB6; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & COLORS
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "equity":    "#E8534A",
    "bond":      "#4A90D9",
    "commodity": "#7B7B7B",
    "strategy":  "#2C7BB6",
    "benchmark": "#D7191C",
    "neutral":   "#404040",
}

# Yahoo Finance tickers — all confirmed working via yfinance Ticker.history()
ASSET_UNIVERSE = {
    # Equity indices
    "^GSPC":  {"name": "S&P 500",          "class": "equity"},
    "^GDAXI": {"name": "DAX Germany",       "class": "equity"},
    "^FCHI":  {"name": "CAC 40 France",     "class": "equity"},
    "^N225":  {"name": "Nikkei 225",        "class": "equity"},
    "^HSI":   {"name": "Hang Seng HK",      "class": "equity"},
    "^BSESN": {"name": "BSE Sensex India",  "class": "equity"},
    "^NSEI":  {"name": "Nifty 50 India",    "class": "equity"},
    "^BVSP":  {"name": "Bovespa Brazil",    "class": "equity"},
    "^AXJO":  {"name": "ASX 200 Australia", "class": "equity"},
    "^FTSE":  {"name": "FTSE 100 UK",       "class": "equity"},
    "^KS11":  {"name": "KOSPI Korea",       "class": "equity"},
    "^TWII":  {"name": "Taiwan TAIEX",      "class": "equity"},
    # Bond ETFs (more reliable than index tickers on Yahoo)
    "SHY":    {"name": "2Y US Treasury",    "class": "bond"},
    "IEF":    {"name": "10Y US Treasury",   "class": "bond"},
    "TLT":    {"name": "20Y+ US Treasury",  "class": "bond"},
    "LQD":    {"name": "Corp Bond IG",      "class": "bond"},
    "HYG":    {"name": "High Yield Bond",   "class": "bond"},
    # Commodity ETFs
    "GLD":    {"name": "Gold",              "class": "commodity"},
    "USO":    {"name": "Crude Oil WTI",     "class": "commodity"},
    "SLV":    {"name": "Silver",            "class": "commodity"},
}

CRISIS_PERIODS = [
    ("2008-01-01", "2009-12-31", "GFC 2008–09"),
    ("2011-07-01", "2012-06-30", "EU Debt Crisis"),
    ("2020-01-01", "2020-12-31", "COVID-19"),
]

plt.rcParams.update({
    "figure.dpi": 120,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
})


# ─────────────────────────────────────────────────────────────────────────────
# DATA DOWNLOAD  (yfinance Ticker.history — parallel, works on Streamlit Cloud)
# ─────────────────────────────────────────────────────────────────────────────
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf


def _fetch_one_yf(ticker: str, start: str, end: str):
    """
    Fetch one ticker weekly Close via yfinance Ticker.history().
    Uses per-ticker API (not yf.download) to avoid MultiIndex issues.
    Returns (ticker, pd.Series | None, error_str | None).
    """
    try:
        t = yf.Ticker(ticker)
        df = t.history(start=start, end=end, interval="1wk", auto_adjust=True, timeout=20)
        if df is None or df.empty or "Close" not in df.columns:
            return ticker, None, "empty response"
        series = df["Close"].copy()
        series.index = pd.to_datetime(series.index).tz_localize(None)
        series = series.sort_index()
        if series.notna().sum() < 50:
            return ticker, None, "too few rows"
        return ticker, series, None
    except Exception as exc:
        return ticker, None, str(exc)


def download_data_parallel(tickers: tuple, start: str, end: str,
                            status_ph=None, progress_ph=None):
    """Download all tickers in parallel (6 workers) with live progress."""
    n = len(tickers)
    frames = {}
    failed = []
    done = 0

    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(_fetch_one_yf, t, start, end): t for t in tickers}
        for fut in as_completed(futures):
            ticker, series, err = fut.result()
            done += 1
            if series is not None:
                frames[ticker] = series
                msg = f"✅ {ticker}"
            else:
                failed.append(ticker)
                msg = f"⚠️ {ticker} — {err}"
            if status_ph:
                status_ph.text(f"Fetching prices… {done}/{n}   {msg}")
            if progress_ph:
                progress_ph.progress(done / n)

    if not frames:
        raise ValueError(
            "No data downloaded. Yahoo Finance may be temporarily unavailable — "
            "please wait 30 s and try again."
        )

    prices = pd.DataFrame(frames).sort_index()
    prices.index = pd.to_datetime(prices.index)
    prices = prices.resample("W-FRI").last()

    threshold = 0.30
    missing_frac = prices.isna().mean()
    dropped = missing_frac[missing_frac > threshold].index.tolist()
    prices = prices.drop(columns=dropped)
    prices = prices.ffill().dropna(how="all")

    if prices.empty:
        raise ValueError("All tickers had >30% missing data after download.")

    return prices, dropped, failed


@st.cache_data(show_spinner=False, ttl=3600)
def download_data(tickers: tuple, start: str, end: str):
    """Cached wrapper — avoids re-fetching on page interactions."""
    return download_data_parallel(tickers, start, end)


# ─────────────────────────────────────────────────────────────────────────────
# RETURNS & STATIONARITY
# ─────────────────────────────────────────────────────────────────────────────
def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


def adf_test(series: pd.Series) -> bool:
    try:
        result = adfuller(series.dropna(), autolag="AIC")
        return result[1] < 0.05
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CAUSALITY MEASURES
# ─────────────────────────────────────────────────────────────────────────────
def granger_causality_score(x, y, max_lag=4, pval_threshold=0.10):
    try:
        data = np.column_stack([y, x])
        result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        pvals = [result[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1)]
        min_pval = min(pvals)
        return (1.0 - min_pval) if min_pval < pval_threshold else 0.0
    except Exception:
        return 0.0


def discretize(series, n_bins=5):
    ranks = stats.rankdata(series)
    bins = np.linspace(0, len(series), n_bins + 1)
    return np.digitize(ranks, bins, right=True).clip(1, n_bins)


def transfer_entropy(x, y, lag=1, n_bins=5):
    try:
        if len(y) - lag < 20:
            return 0.0
        xd, yd = discretize(x, n_bins), discretize(y, n_bins)
        y_future, y_past, x_past = yd[lag:], yd[:-lag], xd[:-lag]

        def joint_entropy(*arrays):
            combined = np.array(list(zip(*arrays)))
            _, counts = np.unique(combined, axis=0, return_counts=True)
            p = counts / counts.sum()
            return -np.sum(p * np.log2(p + 1e-12))

        def entropy(arr):
            _, counts = np.unique(arr, return_counts=True)
            p = counts / counts.sum()
            return -np.sum(p * np.log2(p + 1e-12))

        H_yf_yp = joint_entropy(y_future, y_past)
        H_yp = entropy(y_past)
        H_yf_yp_xp = joint_entropy(y_future, y_past, x_past)
        H_yp_xp = joint_entropy(y_past, x_past)
        te = H_yf_yp - H_yp - H_yf_yp_xp + H_yp_xp
        H_cond = H_yf_yp - H_yp
        return max(0.0, min(1.0, te / H_cond)) if H_cond > 0 else 0.0
    except Exception:
        return 0.0


def licc_score(x, y, tau=1, pval_threshold=0.10):
    try:
        if len(x) < tau + 10:
            return 0.0
        corr, pval = stats.pearsonr(x[:-tau], y[tau:])
        return corr if pval < pval_threshold else 0.0
    except Exception:
        return 0.0


def build_causality_matrix(window_returns, method="lgc",
                            max_lag=4, licc_lag=1, n_bins=5, pval=0.10):
    assets = window_returns.columns.tolist()
    n = len(assets)
    matrix = np.zeros((n, n))
    for i, ai in enumerate(assets):
        for j, aj in enumerate(assets):
            if i == j:
                continue
            xi, xj = window_returns[ai].values, window_returns[aj].values
            if method == "lgc":
                matrix[i, j] = granger_causality_score(xi, xj, max_lag, pval)
            elif method == "te":
                matrix[i, j] = transfer_entropy(xi, xj, licc_lag, n_bins)
            elif method == "licc":
                matrix[i, j] = abs(licc_score(xi, xj, licc_lag, pval))
    return matrix


def out_strength_centrality(matrix):
    return matrix.sum(axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# ROLLING CAUSALITY ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def run_rolling_causality(returns, window=104, step=4, methods=None,
                           max_lag=4, licc_lag=1, n_bins=5, pval=0.10,
                           progress_cb=None):
    if methods is None:
        methods = ["lgc", "te", "licc"]
    n_assets = len(returns.columns)
    dates = returns.index
    results = {m: [] for m in methods}
    result_dates = []
    total_steps = max(1, (len(dates) - window) // step)

    for idx, start_idx in enumerate(range(0, len(dates) - window, step)):
        end_idx = start_idx + window
        window_ret = returns.iloc[start_idx:end_idx]
        window_date = dates[end_idx - 1]
        for method in methods:
            matrix = build_causality_matrix(
                window_ret, method=method,
                max_lag=max_lag, licc_lag=licc_lag, n_bins=n_bins, pval=pval
            )
            results[method].append(out_strength_centrality(matrix))
        result_dates.append(window_date)
        if progress_cb:
            progress_cb(min(1.0, (idx + 1) / total_steps))

    centrality_dfs = {}
    for method in methods:
        centrality_dfs[method] = pd.DataFrame(
            results[method], index=result_dates, columns=returns.columns
        )
    return centrality_dfs


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE CENTRALITY
# ─────────────────────────────────────────────────────────────────────────────
def rank_aggregate(centrality_dfs):
    methods = list(centrality_dfs.keys())
    first_df = centrality_dfs[methods[0]]
    composite_scores = []
    for date in first_df.index:
        rank_sum = np.zeros(len(first_df.columns))
        for df in centrality_dfs.values():
            if date in df.index:
                rank_sum += stats.rankdata(df.loc[date].values)
        composite_scores.append(rank_sum)
    return pd.DataFrame(composite_scores, index=first_df.index, columns=first_df.columns)


def normalize_cross_sectional(df):
    row_min, row_max = df.min(axis=1), df.max(axis=1)
    return df.sub(row_min, axis=0).div(row_max - row_min + 1e-10, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────
def construct_portfolio(composite_norm, returns, n_long=3, n_short=3, tc=0.001):
    common = composite_norm.index.intersection(returns.index)
    comp_aligned = composite_norm.loc[common]
    results = []
    prev_long, prev_short = set(), set()

    for i in range(len(common) - 1):
        sig_date = common[i]
        trade_date = common[i + 1]
        scores = comp_aligned.loc[sig_date].dropna()
        if len(scores) < n_long + n_short:
            continue
        ranked = scores.sort_values(ascending=False)
        long_assets = set(ranked.iloc[:n_long].index)
        short_assets = set(ranked.iloc[-n_short:].index)
        if trade_date not in returns.index:
            continue
        nr = returns.loc[trade_date]
        long_ret = nr[list(long_assets)].mean()
        short_ret = nr[list(short_assets)].mean()
        strategy_gross = long_ret - short_ret
        turnover = (len(long_assets.symmetric_difference(prev_long)) +
                    len(short_assets.symmetric_difference(prev_short))) / (2 * (n_long + n_short))
        cost = turnover * tc * 2
        results.append({
            "date": trade_date,
            "strategy_gross": strategy_gross,
            "strategy_net": strategy_gross - cost,
            "long_only": long_ret,
            "benchmark": nr.mean(),
            "long_positions": ", ".join(sorted(long_assets)),
            "short_positions": ", ".join(sorted(short_assets)),
            "turnover": turnover,
        })
        prev_long, prev_short = long_assets, short_assets

    return pd.DataFrame(results).set_index("date")


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(r_series, rf=0.02):
    r = r_series.dropna()
    if len(r) < 10:
        return {}
    ann_ret = r.mean() * 52
    ann_vol = r.std() * np.sqrt(52)
    weekly_rf = rf / 52
    excess = r - weekly_rf
    sharpe = (excess.mean() / excess.std()) * np.sqrt(52) if excess.std() > 0 else 0
    downside = r[r < weekly_rf] - weekly_rf
    ds_std = np.sqrt((downside ** 2).mean()) * np.sqrt(52) if len(downside) > 0 else 1e-10
    sortino = (ann_ret - rf) / ds_std if ds_std > 0 else 0
    cum = (1 + r).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    win_rate = (r > 0).mean()
    var95 = r.quantile(0.05)
    cvar95 = r[r <= var95].mean()
    gfc = r[(r.index >= "2008-01-01") & (r.index <= "2009-12-31")].mean() * 52
    covid = r[(r.index >= "2020-01-01") & (r.index <= "2020-12-31")].mean() * 52
    return {
        "Ann. Return":     ann_ret,
        "Ann. Volatility": ann_vol,
        "Sharpe Ratio":    sharpe,
        "Sortino Ratio":   sortino,
        "Calmar Ratio":    calmar,
        "Max Drawdown":    max_dd,
        "Win Rate":        win_rate,
        "Skewness":        r.skew(),
        "VaR (95%)":       var95,
        "CVaR (95%)":      cvar95,
        "GFC Return":      gfc,
        "COVID Return":    covid,
    }


def fmt_metric(val, key):
    pct_keys = {"Ann. Return", "Ann. Volatility", "Max Drawdown",
                "Win Rate", "VaR (95%)", "CVaR (95%)", "GFC Return", "COVID Return"}
    if key in pct_keys:
        return f"{val * 100:.2f}%"
    return f"{val:.3f}"


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def shade_crises(ax, index):
    for s, e, label in CRISIS_PERIODS:
        ts, te = pd.Timestamp(s), pd.Timestamp(e)
        if ts >= index[0] and te <= index[-1]:
            ax.axvspan(ts, te, alpha=0.10, color="#D7191C", zorder=0)


def fig_cumulative(portfolio):
    fig, ax = plt.subplots(figsize=(12, 4))
    for col, label, color, lw, ls in [
        ("strategy_net",   "Strategy (Net)",   COLORS["strategy"],  2.2, "-"),
        ("strategy_gross", "Strategy (Gross)", "#5DA5DA",           1.4, "--"),
        ("long_only",      "Long-Only",        "#FAA43A",           1.6, "-"),
        ("benchmark",      "Equal-Weight BM",  COLORS["benchmark"], 1.6, ":"),
    ]:
        cum = (1 + portfolio[col]).cumprod()
        ax.plot(cum.index, cum.values, label=label, color=color, lw=lw, ls=ls)
    shade_crises(ax, portfolio.index)
    ax.set_title("Cumulative Wealth (Starting $1)", fontweight="bold")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    return fig


def fig_drawdown(portfolio):
    fig, ax = plt.subplots(figsize=(12, 3))
    for col, label, color in [
        ("strategy_net", "Strategy", COLORS["strategy"]),
        ("benchmark",    "Benchmark", COLORS["benchmark"]),
    ]:
        cum = (1 + portfolio[col]).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        ax.fill_between(dd.index, dd.values, 0, alpha=0.35, color=color)
        ax.plot(dd.index, dd.values, color=color, lw=1, label=label)
    shade_crises(ax, portfolio.index)
    ax.set_title("Drawdown Profile", fontweight="bold")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
    ax.legend()
    fig.tight_layout()
    return fig


def fig_rolling_sharpe(portfolio):
    fig, ax = plt.subplots(figsize=(12, 3))
    for col, label, color in [
        ("strategy_net", "Strategy", COLORS["strategy"]),
        ("benchmark",    "Benchmark", COLORS["benchmark"]),
    ]:
        rr = portfolio[col].rolling(52).mean() * 52
        rv = portfolio[col].rolling(52).std() * np.sqrt(52)
        rs = rr / (rv + 1e-10)
        ax.plot(rs.index, rs.values, label=label, color=color, lw=1.5)
    shade_crises(ax, portfolio.index)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title("Rolling 1-Year Sharpe Ratio", fontweight="bold")
    ax.set_ylabel("Sharpe")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_centrality_heatmap(composite_norm):
    heat = composite_norm.resample("ME").mean().T
    if heat.shape[1] > 60:
        heat = heat.iloc[:, -60:]
    fig, ax = plt.subplots(figsize=(14, max(4, len(heat.index) * 0.4)))
    im = ax.imshow(heat.values, aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index.tolist(), fontsize=8)
    step = max(1, heat.shape[1] // 12)
    ax.set_xticks(range(0, heat.shape[1], step))
    ax.set_xticklabels(
        [d.strftime("%Y-%m") for d in heat.columns[::step]],
        rotation=45, ha="right", fontsize=7
    )
    plt.colorbar(im, ax=ax, label="Centrality Score (Normalised)", shrink=0.6)
    ax.set_title("Composite Causal Centrality Heatmap (Monthly)", fontweight="bold")
    fig.tight_layout()
    return fig


def fig_asset_class_centrality(composite_norm, asset_universe):
    fig, ax = plt.subplots(figsize=(12, 3.5))
    for cls, label, color in [
        ("equity",    "Equity",    COLORS["equity"]),
        ("bond",      "Bond",      COLORS["bond"]),
        ("commodity", "Commodity", COLORS["commodity"]),
    ]:
        assets = [a for a in composite_norm.columns
                  if asset_universe.get(a, {}).get("class") == cls]
        if assets:
            avg = composite_norm[assets].mean(axis=1).rolling(8).mean()
            ax.plot(avg.index, avg.values, label=label, color=color, lw=1.8)
    shade_crises(ax, composite_norm.index)
    ax.set_title("Asset Class Average Causal Centrality (8-week rolling)", fontweight="bold")
    ax.set_ylabel("Average Centrality")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_return_distribution(portfolio):
    fig, ax = plt.subplots(figsize=(8, 4))
    m_strat = portfolio["strategy_net"].resample("ME").sum()
    m_bench = portfolio["benchmark"].resample("ME").sum()
    ax.hist(m_strat * 100, bins=28, alpha=0.6, color=COLORS["strategy"],
            label="Strategy", density=True)
    ax.hist(m_bench * 100, bins=28, alpha=0.4, color=COLORS["benchmark"],
            label="Benchmark", density=True)
    for s, c in [(m_strat, COLORS["strategy"]), (m_bench, COLORS["benchmark"])]:
        mu, sd = s.mean() * 100, s.std() * 100
        x = np.linspace(mu - 4*sd, mu + 4*sd, 100)
        ax.plot(x, stats.norm.pdf(x, mu, sd), color=c, lw=1.5, ls="--")
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_title("Monthly Return Distribution", fontweight="bold")
    ax.set_xlabel("Monthly Return (%)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_network(returns, asset_universe, window=104, top_n_edges=12):
    # Use most recent window
    window_ret = returns.iloc[-window:] if len(returns) >= window else returns
    matrix = build_causality_matrix(window_ret, method="lgc")
    assets = window_ret.columns.tolist()
    G = nx.DiGraph()
    for a in assets:
        G.add_node(a,
                   asset_class=asset_universe.get(a, {}).get("class", "equity"),
                   name=asset_universe.get(a, {}).get("name", a))
    flat_edges = []
    for i in range(len(assets)):
        for j in range(len(assets)):
            if i != j and matrix[i, j] > 0:
                flat_edges.append((assets[i], assets[j], matrix[i, j]))
    flat_edges.sort(key=lambda x: -x[2])
    for src, dst, w in flat_edges[:top_n_edges]:
        G.add_edge(src, dst, weight=w)
    fig, ax = plt.subplots(figsize=(10, 7))
    pos = nx.spring_layout(G, k=2.5, seed=42)
    color_map = {"equity": COLORS["equity"], "bond": COLORS["bond"],
                 "commodity": COLORS["commodity"]}
    node_colors = [color_map.get(G.nodes[n].get("asset_class", "equity"),
                                 COLORS["equity"]) for n in G.nodes()]
    node_sizes = []
    for node in G.nodes():
        out_w = sum(d["weight"] for _, _, d in G.out_edges(node, data=True))
        node_sizes.append(300 + out_w * 2000)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                            node_size=node_sizes, alpha=0.85)
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    if edge_weights:
        mw = max(edge_weights)
        nx.draw_networkx_edges(G, pos, ax=ax,
                                width=[1 + 4 * (w/mw) for w in edge_weights],
                                alpha=0.5, arrows=True, arrowsize=12,
                                edge_color="#404040",
                                connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_labels(G, pos, {n: n for n in G.nodes()},
                             ax=ax, font_size=7, font_weight="bold")
    patches = [
        mpatches.Patch(color=COLORS["equity"],    label="Equity"),
        mpatches.Patch(color=COLORS["bond"],      label="Bond"),
        mpatches.Patch(color=COLORS["commodity"], label="Commodity"),
    ]
    ax.legend(handles=patches, loc="upper left", fontsize=8)
    ax.set_title("Causal Network (Most Recent Window — LGC)", fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    return fig


def fig_robustness_n(composite_norm, returns):
    """Sharpe heatmap: N_LONG vs N_SHORT"""
    n_vals = [2, 3, 4, 5]
    sharpe_grid = np.zeros((len(n_vals), len(n_vals)))
    for i, nl in enumerate(n_vals):
        for j, ns in enumerate(n_vals):
            port = construct_portfolio(composite_norm, returns, n_long=nl, n_short=ns)
            m = compute_metrics(port["strategy_net"])
            sharpe_grid[i, j] = m.get("Sharpe Ratio", 0)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sharpe_grid, cmap="RdYlGn", aspect="auto")
    labels = [str(v) for v in n_vals]
    ax.set_xticks(range(4)); ax.set_xticklabels(labels)
    ax.set_yticks(range(4)); ax.set_yticklabels(labels)
    ax.set_xlabel("N Short"); ax.set_ylabel("N Long")
    ax.set_title("Sharpe: N_Long × N_Short", fontweight="bold")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{sharpe_grid[i,j]:.2f}", ha="center", va="center",
                    fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def fig_tc_sensitivity(composite_norm, returns, n_long, n_short):
    costs = [0, 0.001, 0.002, 0.005, 0.010]
    sharpes, ann_rets = [], []
    for c in costs:
        port = construct_portfolio(composite_norm, returns,
                                   n_long=n_long, n_short=n_short, tc=c)
        m = compute_metrics(port["strategy_net"])
        sharpes.append(m.get("Sharpe Ratio", 0))
        ann_rets.append(m.get("Ann. Return", 0) * 100)
    fig, ax = plt.subplots(figsize=(7, 4))
    bps = [c * 10000 for c in costs]
    ax.plot(bps, sharpes, "o-", color=COLORS["strategy"], lw=2, ms=8, label="Sharpe")
    ax.axhline(0, color="red", ls="--", lw=1)
    ax2 = ax.twinx()
    ax2.plot(bps, ann_rets, "s--", color=COLORS["benchmark"], lw=1.5, ms=6, label="Ann Ret %")
    ax2.set_ylabel("Ann. Return (%)", color=COLORS["benchmark"])
    ax.set_xlabel("Transaction Cost (bps)")
    ax.set_ylabel("Sharpe Ratio", color=COLORS["strategy"])
    ax.set_title("Transaction Cost Sensitivity", fontweight="bold")
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2)
    fig.tight_layout()
    return fig


def fig_factor_decay(composite_norm, returns, max_lag=8):
    ics, ic_stds, icirs = [], [], []
    common = composite_norm.index.intersection(returns.index)
    for lag in range(1, max_lag + 1):
        ic_vals = []
        for i in range(len(common) - lag):
            sd = common[i]
            fd = common[i + lag]
            if fd not in returns.index:
                continue
            signal = composite_norm.loc[sd]
            fwd = returns.loc[fd]
            ca = signal.index.intersection(fwd.index)
            if len(ca) < 5:
                continue
            ic, _ = spearmanr(signal[ca].rank(), fwd[ca].rank())
            ic_vals.append(ic)
        mean_ic = np.nanmean(ic_vals)
        ic_std = np.nanstd(ic_vals)
        ics.append(mean_ic)
        ic_stds.append(ic_std)
        icirs.append(mean_ic / (ic_std + 1e-10))
    lags = list(range(1, max_lag + 1))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(lags, ics, color=COLORS["strategy"], alpha=0.7, label="Mean IC")
    ax.errorbar(lags, ics, yerr=ic_stds, fmt="none", color="black", capsize=4, lw=1.2)
    ax.axhline(0, color="red", ls="--", lw=1)
    ax2 = ax.twinx()
    ax2.plot(lags, icirs, "D--", color=COLORS["benchmark"], lw=1.5, ms=6, label="ICIR")
    ax2.set_ylabel("ICIR", color=COLORS["benchmark"])
    ax.set_xlabel("Forward Return Lag (weeks)")
    ax.set_ylabel("Information Coefficient (IC)")
    ax.set_title("Factor Decay Analysis", fontweight="bold")
    ax.set_xticks(lags)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2)
    fig.tight_layout()
    return fig


def buf(fig):
    """Convert matplotlib figure to streamlit-renderable bytes buffer."""
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=120, bbox_inches="tight")
    b.seek(0)
    plt.close(fig)
    return b


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    st.markdown("**📅 Date Range**")
    start_date = st.text_input("Start Date", "2008-01-01")
    end_date   = st.text_input("End Date",   "2024-01-01")

    st.markdown("---")
    st.markdown("**🌍 Asset Universe**")
    all_tickers = list(ASSET_UNIVERSE.keys())
    selected_tickers = st.multiselect(
        "Select tickers (min 6)",
        options=all_tickers,
        default=all_tickers,
        format_func=lambda t: f"{t} — {ASSET_UNIVERSE[t]['name']}",
    )

    st.markdown("---")
    st.markdown("**🔬 Causality Methods**")
    use_lgc  = st.checkbox("Linear Granger (LGC)",    value=True)
    use_te   = st.checkbox("Transfer Entropy (TE)",   value=True)
    use_licc = st.checkbox("Lead-Lag Correlation (LICC)", value=True)

    st.markdown("---")
    st.markdown("**📊 Rolling Window**")
    window_weeks = st.slider("Window (weeks)", 52, 156, 104, step=4,
                              help="2yr=104, 1.5yr=78, 3yr=156")
    step_weeks   = st.slider("Step (weeks)",   1,  26,  4,   step=1,
                              help="Larger step = faster but lower resolution")

    st.markdown("---")
    st.markdown("**💼 Portfolio**")
    n_long   = st.slider("N Long",  1, 7, 3)
    n_short  = st.slider("N Short", 1, 7, 3)
    tc_bps   = st.slider("Transaction Cost (bps)", 0, 50, 10)

    st.markdown("---")
    st.markdown("**⚡ Advanced**")
    max_lag  = st.slider("Granger Max Lag (weeks)", 1, 8, 4)
    pval_thr = st.slider("p-value Threshold", 0.01, 0.20, 0.10, step=0.01)
    n_bins   = st.slider("TE Discretisation Bins", 3, 10, 5)

    st.markdown("---")
    run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style="font-size:28px; font-weight:800; color:#1F2937; margin-bottom:2px;">
    📈 Causal Out-Strength Centrality Factor
</h1>
<p style="color:#6B7280; font-size:14px; margin-top:0;">
    Inspired by Stavroglou et al. (2019) &mdash; <em>Causality Networks of Financial Assets</em>
    &nbsp;|&nbsp; Data: Stooq (free, live) &nbsp;|&nbsp; Methods: LGC · Transfer Entropy · LICC
</p>
<hr style="border:none; border-top:1px solid #E5E7EB; margin: 8px 0 20px 0;">
""", unsafe_allow_html=True)

if not run_btn:
    # ── Landing / Info state ──────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Strategy</div>
            <div class="metric-value">Long–Short</div>
            <div style="font-size:12px;color:#6B7280;margin-top:6px;">
                Top-N vs Bottom-N by causal out-strength
            </div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Universe</div>
            <div class="metric-value">20 Assets</div>
            <div style="font-size:12px;color:#6B7280;margin-top:6px;">
                Global equities, bonds &amp; commodities
            </div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Data Source</div>
            <div class="metric-value">Stooq</div>
            <div style="font-size:12px;color:#6B7280;margin-top:6px;">
                Free · No API key · Works on Streamlit Cloud
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#EFF6FF; border:1px solid #BFDBFE; border-radius:10px;
                padding:16px 20px; margin-top:20px; color:#1E3A5F;">
        <strong>How to use:</strong>
        <ol style="margin:8px 0 0 16px; padding:0; font-size:14px; line-height:1.8;">
            <li>Adjust the date range and asset universe in the sidebar</li>
            <li>Select which causality methods to include</li>
            <li>Tune rolling window, portfolio parameters, and advanced settings</li>
            <li>Click <strong>🚀 Run Analysis</strong> — computation takes 2–10 min depending on settings</li>
        </ol>
        <p style="margin:10px 0 0 0; font-size:13px; color:#374151;">
        <strong>Tip:</strong> For a fast preview, set Step = 12–26 weeks and use LGC only.
        For full quality (paper-replication), Step = 1–4 weeks with all three methods.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
selected_methods = (["lgc"] if use_lgc else []) + \
                   (["te"]   if use_te  else []) + \
                   (["licc"] if use_licc else [])

if len(selected_tickers) < 6:
    st.error("Please select at least 6 tickers.")
    st.stop()
if not selected_methods:
    st.error("Please select at least one causality method.")
    st.stop()

active_universe = {t: ASSET_UNIVERSE[t] for t in selected_tickers if t in ASSET_UNIVERSE}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DATA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Step 1 — Downloading Data</div>', unsafe_allow_html=True)

_status_ph   = st.empty()
_progress_ph = st.progress(0)
_status_ph.text(f"Fetching {len(selected_tickers)} tickers from Stooq in parallel…")

try:
    prices_raw, dropped_tickers, failed_tickers = download_data_parallel(
        tuple(selected_tickers), start_date, end_date,
        status_ph=_status_ph, progress_ph=_progress_ph
    )
except Exception as e:
    _status_ph.empty()
    _progress_ph.empty()
    st.error(f"Data download failed: {e}")
    st.stop()

_progress_ph.progress(1.0)
_status_ph.success(
    f"✅ Downloaded {prices_raw.shape[1]} assets × {prices_raw.shape[0]} weeks  "
    f"({prices_raw.index[0].date()} → {prices_raw.index[-1].date()})"
)

# Keep active_universe in sync with what actually downloaded
active_universe = {t: active_universe[t] for t in prices_raw.columns if t in active_universe}

n_assets = prices_raw.shape[1]
n_weeks  = prices_raw.shape[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Assets Downloaded", n_assets)
c2.metric("Weeks of Data", n_weeks)
c3.metric("Start", str(prices_raw.index[0].date()))
c4.metric("End",   str(prices_raw.index[-1].date()))

if dropped_tickers or failed_tickers:
    all_bad = list(set(dropped_tickers + failed_tickers))
    st.warning(f"Tickers excluded (missing >30% or unavailable): `{', '.join(all_bad)}`")

with st.expander("📋 Price Preview"):
    st.dataframe(prices_raw.tail(10).round(2), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — RETURNS & STATIONARITY
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Step 2 — Log Returns & Stationarity</div>',
            unsafe_allow_html=True)

returns = compute_log_returns(prices_raw)

adf_results = {col: adf_test(returns[col]) for col in returns.columns}
stat_df = pd.DataFrame({
    "Ticker": list(adf_results.keys()),
    "Name":   [active_universe.get(t, {}).get("name", t) for t in adf_results],
    "Class":  [active_universe.get(t, {}).get("class", "?") for t in adf_results],
    "Stationary (ADF p<0.05)": ["✅ Yes" if v else "⚠️ No" for v in adf_results.values()],
})
st.dataframe(stat_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — ROLLING CAUSALITY
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Step 3 — Rolling Causality Engine</div>',
            unsafe_allow_html=True)

total_windows = max(1, (len(returns) - window_weeks) // step_weeks)
est_seconds = total_windows * n_assets * (n_assets - 1) * 0.002 * len(selected_methods)
st.info(
    f"**{total_windows} windows** × {n_assets} assets × {n_assets-1} pairs "
    f"× {len(selected_methods)} method(s) — estimated ~{est_seconds/60:.1f} min"
)

progress_bar = st.progress(0)
status_text  = st.empty()

def update_progress(p):
    progress_bar.progress(p)
    status_text.text(f"Computing causality matrices… {p*100:.0f}%")

centrality_dfs = run_rolling_causality(
    returns,
    window=window_weeks,
    step=step_weeks,
    methods=selected_methods,
    max_lag=max_lag,
    licc_lag=1,
    n_bins=n_bins,
    pval=pval_thr,
    progress_cb=update_progress,
)
progress_bar.progress(1.0)
status_text.text("✅ Rolling causality complete!")

composite_centrality = rank_aggregate(centrality_dfs)
composite_norm = normalize_cross_sectional(composite_centrality)

st.success(f"Composite centrality computed: {composite_norm.shape[0]} dates × "
           f"{composite_norm.shape[1]} assets")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — PORTFOLIO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Step 4 — Portfolio Construction</div>',
            unsafe_allow_html=True)

tc = tc_bps / 10000
portfolio = construct_portfolio(composite_norm, returns,
                                n_long=n_long, n_short=n_short, tc=tc)

st.success(f"Portfolio: {len(portfolio)} weekly observations  |  "
           f"{portfolio.index[0].date()} → {portfolio.index[-1].date()}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Step 5 — Performance Summary</div>',
            unsafe_allow_html=True)

strat_metrics = compute_metrics(portfolio["strategy_net"])
bench_metrics = compute_metrics(portfolio["benchmark"])

# KPI row
k1, k2, k3, k4, k5 = st.columns(5)
def kpi(col, label, val, suffix="", is_pct=False):
    fval = f"{val*100:.2f}%" if is_pct else f"{val:.3f}{suffix}"
    color_class = "positive" if val > 0 else "negative"
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {color_class}">{fval}</div>
    </div>""", unsafe_allow_html=True)

kpi(k1, "Ann. Return (Net)",    strat_metrics.get("Ann. Return", 0),    is_pct=True)
kpi(k2, "Sharpe Ratio",         strat_metrics.get("Sharpe Ratio", 0))
kpi(k3, "Sortino Ratio",        strat_metrics.get("Sortino Ratio", 0))
kpi(k4, "Max Drawdown",         strat_metrics.get("Max Drawdown", 0),   is_pct=True)
kpi(k5, "Win Rate",             strat_metrics.get("Win Rate", 0),       is_pct=True)

# Full table
metric_keys = [
    "Ann. Return", "Ann. Volatility", "Sharpe Ratio", "Sortino Ratio",
    "Calmar Ratio", "Max Drawdown", "Win Rate", "Skewness",
    "VaR (95%)", "CVaR (95%)", "GFC Return", "COVID Return",
]
col_labels = ["Strategy (Net)", "Benchmark EW"]
rows = []
for k in metric_keys:
    sm = strat_metrics.get(k, float("nan"))
    bm = bench_metrics.get(k, float("nan"))
    rows.append({
        "Metric": k,
        "Strategy (Net)": fmt_metric(sm, k),
        "Benchmark EW":   fmt_metric(bm, k),
    })

metrics_df = pd.DataFrame(rows)
st.dataframe(metrics_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS — CHARTS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Charts & Analysis</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Performance", "🌡️ Centrality", "🕸️ Network",
    "🔬 Robustness", "📤 Export"
])

with tab1:
    st.markdown("#### Cumulative Wealth")
    st.image(buf(fig_cumulative(portfolio)), use_container_width=True)

    st.markdown("#### Drawdown Profile")
    st.image(buf(fig_drawdown(portfolio)), use_container_width=True)

    st.markdown("#### Rolling 1-Year Sharpe")
    st.image(buf(fig_rolling_sharpe(portfolio)), use_container_width=True)

    st.markdown("#### Monthly Return Distribution")
    st.image(buf(fig_return_distribution(portfolio)), use_container_width=True)

with tab2:
    st.markdown("#### Composite Centrality Heatmap")
    st.image(buf(fig_centrality_heatmap(composite_norm)), use_container_width=True)

    st.markdown("#### Asset Class Centrality Over Time")
    st.image(buf(fig_asset_class_centrality(composite_norm, active_universe)),
             use_container_width=True)

    st.markdown("#### Current Rankings (Latest Window)")
    latest_scores = composite_norm.iloc[-1].sort_values(ascending=False)
    rank_df = pd.DataFrame({
        "Rank": range(1, len(latest_scores) + 1),
        "Ticker": latest_scores.index,
        "Name": [active_universe.get(t, {}).get("name", t) for t in latest_scores.index],
        "Class": [active_universe.get(t, {}).get("class", "?") for t in latest_scores.index],
        "Centrality Score": latest_scores.values.round(4),
    })
    st.dataframe(rank_df, use_container_width=True, hide_index=True)

with tab3:
    st.markdown("#### Causal Network — Most Recent Window (LGC)")
    top_n_edges = st.slider("Top N edges to show", 5, 30, 12)
    with st.spinner("Building network graph…"):
        st.image(buf(fig_network(returns, active_universe,
                                  window=window_weeks, top_n_edges=top_n_edges)),
                 use_container_width=True)

with tab4:
    st.markdown("#### Sharpe Sensitivity: N_Long × N_Short")
    if len(composite_norm.columns) >= 10:
        with st.spinner("Running N sensitivity…"):
            st.image(buf(fig_robustness_n(composite_norm, returns)),
                     use_container_width=True)
    else:
        st.warning("Need ≥10 assets for robustness grid.")

    st.markdown("#### Transaction Cost Sensitivity")
    with st.spinner("Running TC sensitivity…"):
        st.image(buf(fig_tc_sensitivity(composite_norm, returns, n_long, n_short)),
                 use_container_width=True)

    st.markdown("#### Factor Decay (IC Analysis)")
    max_lag_decay = st.slider("Max lag (weeks)", 4, 12, 8)
    with st.spinner("Computing factor decay…"):
        st.image(buf(fig_factor_decay(composite_norm, returns, max_lag_decay)),
                 use_container_width=True)

    st.markdown("#### Sub-Period Performance")
    periods = {
        "GFC (2008–09)":      ("2008-01-01", "2010-01-01"),
        "Post-GFC (2010–19)": ("2010-01-01", "2020-01-01"),
        "COVID (2020)":        ("2020-01-01", "2021-01-01"),
        "Post-COVID (2021+)": ("2021-01-01", None),
        "Full Period":         (None,          None),
    }
    sub_rows = []
    for label, (ps, pe) in periods.items():
        sub = portfolio["strategy_net"]
        if ps:
            sub = sub[sub.index >= ps]
        if pe:
            sub = sub[sub.index < pe]
        if len(sub) < 10:
            continue
        m = compute_metrics(sub)
        sub_rows.append({
            "Period": label,
            "N Weeks": len(sub),
            "Ann Return": f"{m.get('Ann. Return',0)*100:.2f}%",
            "Sharpe": f"{m.get('Sharpe Ratio',0):.3f}",
            "Max DD": f"{m.get('Max Drawdown',0)*100:.2f}%",
            "Win Rate": f"{m.get('Win Rate',0)*100:.1f}%",
        })
    st.dataframe(pd.DataFrame(sub_rows), use_container_width=True, hide_index=True)

with tab5:
    st.markdown("#### Download Results")

    def df_to_csv(df):
        return df.to_csv().encode("utf-8")

    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button("📥 Portfolio Returns (CSV)",
                           df_to_csv(portfolio),
                           "portfolio_returns.csv", "text/csv")
        st.download_button("📥 Composite Centrality (CSV)",
                           df_to_csv(composite_norm),
                           "composite_centrality.csv", "text/csv")
    with col_b:
        st.download_button("📥 Prices (CSV)",
                           df_to_csv(prices_raw),
                           "prices.csv", "text/csv")
        st.download_button("📥 Log Returns (CSV)",
                           df_to_csv(returns),
                           "log_returns.csv", "text/csv")

    st.markdown("#### Latest Positions")
    if len(portfolio) > 0:
        last_row = portfolio.iloc[-1]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🟢 Long (Most Causal)**")
            for t in last_row["long_positions"].split(", "):
                name = active_universe.get(t.strip(), {}).get("name", t.strip())
                st.markdown(f"- `{t.strip()}` — {name}")
        with col2:
            st.markdown("**🔴 Short (Least Causal)**")
            for t in last_row["short_positions"].split(", "):
                name = active_universe.get(t.strip(), {}).get("name", t.strip())
                st.markdown(f"- `{t.strip()}` — {name}")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border:none;border-top:1px solid #E5E7EB;margin:32px 0 12px 0;">
<p style="text-align:center;color:#9CA3AF;font-size:12px;">
    Causal Centrality Factor Model &nbsp;·&nbsp;
    Based on <em>Stavroglou et al. (2019)</em> &nbsp;·&nbsp;
    Data via Stooq &nbsp;·&nbsp; For research purposes only — not financial advice
</p>
""", unsafe_allow_html=True)