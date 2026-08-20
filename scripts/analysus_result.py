import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re


root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.env import Env
from src.evaluation.precision_metrics import evaluate_vol_forecast



# Helpers


def get_vol_regime_labels(df: pd.DataFrame, index: pd.Index | None = None, col: str = "vol_regime") -> pd.Series | None:
    
    # Return standardized regime labels from df[col].
    
    if col not in df.columns:
        return None

    def norm(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().lower()
        if s == "low":
            return "Low"
        if s in ("mid", "middle", "med", "medium"):
            return "Mid"
        if s == "high":
            return "High"
        return np.nan

    reg = df[col].map(norm)
    reg = pd.Series(reg, index=df.index)
    if index is not None:
        reg = reg.reindex(index)
    return reg

def derive_regime_from_realized_vol(rv: pd.Series) -> pd.Series:
    """
    Fallback regime of realized vol proxy.
    """
    m = rv.dropna()
    q1, q2 = m.quantile([0.33, 0.66]).values
    out = pd.Series(index=rv.index, dtype=object)
    out[rv <= q1] = "Low"
    out[(rv > q1) & (rv <= q2)] = "Mid"
    out[rv > q2] = "High"
    return out


def align_forecast_and_realized(
    df: pd.DataFrame,
    window: int = 21,
    ann_factor: float = 252.0,
    horizon: int = 0,
):
    if "vol_estimate" not in df.columns:
        return None

    under, _ = pick_underlying_returns(df)
    if under is None:
        return None

    rv = realized_vol_proxy(under, window=window, ann_factor=ann_factor)
    vh = pd.Series(df["vol_estimate"], index=df.index).astype(float)

    target = rv.shift(-horizon) if horizon and horizon > 0 else rv
    m = pd.concat(
        [
            vh.rename("vh"),
            rv.rename("rv"),
            target.rename("target_rv"),
        ],
        axis=1,
    ).dropna()

    if m.empty:
        return None

    m["err"] = m["vh"] - m["target_rv"]
    m["abs_err"] = m["err"].abs()
    m["loss"] = qlike_series(m["target_rv"], m["vh"])
    m["vol_jump"] = m["rv"] - m["rv"].shift(1)
    m["vol_of_vol"] = m["vol_jump"].abs()
    return m


def corr_safe(a: pd.Series, b: pd.Series) -> float:
    m = pd.concat([a, b], axis=1).dropna()
    
    if len(m) < 10:
        return float("nan")
    if float(m.iloc[:, 0].std(ddof=1)) <= 0 or float(m.iloc[:, 1].std(ddof=1)) <= 0:
        return float("nan")
    return float(m.iloc[:, 0].corr(m.iloc[:, 1]))

def estimator_key_from_name(strategy_name: str) -> str:
    
    return strategy_name.split("_")[0]

def reduce_to_one_per_estimator(data_map, prefer=("naive", "buy_and_hold")):
    grouped = {}
    for strat_name, df in data_map.items():
        est = estimator_key_from_name(strat_name)
        grouped.setdefault(est, []).append((strat_name, df))

    reduced = {}
    for est, items in grouped.items():
        picked = None
        for tag in prefer:
            for name, df in items:
                if tag in name.lower():
                    picked = (name, df)
                    break
            if picked:
                break
        if picked is None:
            picked = items[0]
        reduced[est] = picked[1]  
    return reduced
def get_data():
    results_dir = Env.path("results")
    files = list(results_dir.glob("*.parquet"))
    data_map = {}
    for f in files:
        try:
            data_map[f.stem] = pd.read_parquet(f)
        except Exception as e:
            print(f"Failed to read {f.name}: {e}")
    return data_map, results_dir


def pick_strategy_returns(df: pd.DataFrame) -> pd.Series:
    """Return strategy PnL series ."""
    if "strategy_returns" in df.columns:
        return df["strategy_returns"]
    if "returns" in df.columns:
        return df["returns"]
    return pd.Series(index=df.index, dtype=float)


def pick_underlying_returns(df: pd.DataFrame) -> tuple[pd.Series | None, str]:
    
    if "asset_returns" in df.columns:
        return df["asset_returns"], "asset_returns"
    if "returns_clean" in df.columns:
        return df["returns_clean"], "returns_clean"
    if "returns_raw" in df.columns:
        return df["returns_raw"], "returns_raw"
    if "returns" in df.columns:
        return df["returns"], "returns"
    return None, "missing"


def safe_mean(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size > 0 else float("nan")


def safe_percentile(x: pd.Series, q: float) -> float:
    arr = x.to_numpy()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))

#benchmark
def realized_vol_proxy(under: pd.Series, window=21, ann_factor=252.0) -> pd.Series:
    """
    Ex-post forward realized volatility.
    """
    r = pd.Series(under).astype(float)

    
    fwd_r2 = (r.shift(-1) ** 2).rolling(window, min_periods=window).sum()

    rv = np.sqrt((ann_factor / window) * fwd_r2)

    return rv.rename("rv_expost")
# def realized_vol_next_from_returns(
#     returns: pd.Series,
#     ann_factor: float = 252.0,
#     eps: float = 1e-12,
# ) -> pd.Series:
#     """
#     Build a simple realized-vol proxy for t+1 using only r_{t+1}:

#         rv_next(t) = sqrt( ann_factor * r_{t+1}^2 )

#     This is a *proxy* (one-step-ahead, single-return), but it matches the
#     "forecast at t for t+1" alignment used in evaluate_vol_forecast.
#     """
#     r_next = pd.Series(returns).shift(-1).astype(float)
#     rv2_next = ann_factor * (r_next**2)
#     rv_next = np.sqrt(np.maximum(rv2_next, eps))
#     rv_next.name = "rv_next"
#     return rv_next

def qlike_series(rv: pd.Series, vol_hat: pd.Series) -> pd.Series:
    r = (rv ** 2).astype(float)
    h = (vol_hat ** 2).astype(float)
    m = pd.concat([r, h], axis=1).dropna()
    r2 = m.iloc[:, 0].clip(lower=1e-18)
    h2 = m.iloc[:, 1].clip(lower=1e-18)
    return np.log(h2) + (r2 / h2)



# Plots

def plot_regime_strategy_metrics_table(
    data_map,
    output_path,
    regime_col="vol_regime",
    min_days_per_regime=60,
):
    def norm_reg(x):
        if pd.isna(x):
            return None
        s = str(x).strip().lower()
        if s in ("low",):
            return "Low"
        if s in ("mid", "middle", "med", "medium"):
            return "Mid"
        if s in ("high",):
            return "High"
        return None

    def max_drawdown_from_returns(rets: pd.Series) -> float:
        r = pd.Series(rets).fillna(0.0).astype(float)
        eq = np.exp(np.cumsum(r.values))
        eq = pd.Series(eq, index=r.index)
        dd = eq / eq.cummax() - 1.0
        return float(dd.min()) if len(dd) else float("nan")

    def regime_blocks_mask(reg: pd.Series, label: str) -> list[pd.Index]:
        is_in = reg.astype(str).str.lower().eq(label.lower()).fillna(False).values
        idx = reg.index
        blocks = []
        start = None
        for i, flag in enumerate(is_in):
            if flag and start is None:
                start = i
            if (not flag) and start is not None:
                blocks.append(idx[start:i])
                start = None
        if start is not None:
            blocks.append(idx[start:])
        return blocks

    rows = []
    for name, df in data_map.items():
        if regime_col not in df.columns:
            continue

        strat = pick_strategy_returns(df).astype(float)
        if strat.dropna().empty:
            continue

        reg = df[regime_col].map(norm_reg)
        reg = pd.Series(reg, index=df.index)

        turn = None
        if "weight" in df.columns:
            turn = pd.Series(df["weight"], index=df.index).diff().abs()

        for rlabel in ["Low", "Mid", "High"]:
            idx_r = df.index[reg.eq(rlabel).fillna(False)]
            if len(idx_r) < min_days_per_regime:
                continue

            r_rets = strat.reindex(idx_r).dropna()
            if len(r_rets) < 20:
                continue

            ann_ret = safe_mean(r_rets.values) * 252
            ann_vol = float(np.nanstd(r_rets.values, ddof=0)) * np.sqrt(252)
            sharpe = ann_ret / ann_vol if np.isfinite(ann_vol) and ann_vol > 1e-12 else float("nan")

            # MaxDD computed within each contiguous regime segment; take the worst segment
            blocks = regime_blocks_mask(reg, rlabel)
            block_mdds = []
            for bidx in blocks:
                b_rets = strat.reindex(bidx).dropna()
                if len(b_rets) >= 10:
                    block_mdds.append(max_drawdown_from_returns(b_rets))
            maxdd = float(np.nanmin(block_mdds)) if len(block_mdds) else float("nan")

            if turn is not None:
                r_turn = turn.reindex(idx_r).dropna()
                avg_turn = float(np.nanmean(r_turn.values)) if len(r_turn) else float("nan")
            else:
                avg_turn = float("nan")

            rows.append({
                "strategy": name,
                "regime": rlabel,
                "n_days": int(len(idx_r)),
                "sharpe": sharpe,
                "max_drawdown": maxdd,
                "avg_turnover": avg_turn,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        print("No regime metrics produced (check vol_regime values and min_days_per_regime).")
        return

    out.to_csv(output_path / "Regime_Strategy_Metrics.csv", index=False)

    piv = out.pivot_table(
        index="strategy",
        columns="regime",
        values=["sharpe", "max_drawdown", "avg_turnover", "n_days"],
        aggfunc="first",
    )
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index()

    disp = piv.copy()
    for c in disp.columns:
        if c == "strategy":
            continue
        if c.startswith("n_days"):
            disp[c] = disp[c].map(lambda x: f"{int(x)}" if pd.notna(x) else "0")
        elif "max_drawdown" in c:
            disp[c] = disp[c].map(lambda x: f"{x:.2%}" if pd.notna(x) and np.isfinite(x) else "nan")
        elif "avg_turnover" in c:
            disp[c] = disp[c].map(lambda x: f"{x:.3%}" if pd.notna(x) and np.isfinite(x) else "nan")
        else:
            disp[c] = disp[c].map(lambda x: f"{x:.3f}" if pd.notna(x) and np.isfinite(x) else "nan")

    fig_w = max(12, 0.75 * (disp.shape[1] + 6))
    fig_h = max(3.5, 0.45 * (disp.shape[0] + 4))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title("Strategy Metrics by vol_regime (Sharpe / MaxDD / Turnover)", fontsize=14, pad=12)

    table = ax.table(
        cellText=disp.values,
        colLabels=disp.columns.tolist(),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    fig.tight_layout()
    fig.savefig(output_path / "Regime_Strategy_Metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path / 'Regime_Strategy_Metrics.csv'}")
    print(f"aved: {output_path / 'Regime_Strategy_Metrics.png'}")
def plot_dim1_returns(data_map, output_path):
    n_strat = len(data_map)
    fig_height = max(8, 4 + 0.6 * n_strat)

    fig, (ax, ax_tbl) = plt.subplots(
        2, 1,
        figsize=(15, fig_height),
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True
    )
    metrics_data = []

    for name, df in data_map.items():
        rets = pick_strategy_returns(df).fillna(0)
        equity = df.get("equity_curve", None)

        if equity is None:
            
            equity = 1000.0 * np.exp(np.cumsum(rets.fillna(0).values))
            equity = pd.Series(equity, index=df.index)

        ann_ret = safe_mean(rets.values) * 252
        ann_vol = float(np.nanstd(rets.values, ddof=0)) * np.sqrt(252)
        sharpe = ann_ret / ann_vol if np.isfinite(ann_vol) and ann_vol > 0 else 0.0

        downside = rets[rets < 0].values
        downside_std = float(np.nanstd(downside, ddof=0)) * np.sqrt(252) if downside.size > 0 else float("nan")
        sortino = ann_ret / downside_std if np.isfinite(downside_std) and downside_std > 0 else 0.0

        dd = (equity / equity.cummax() - 1)
        max_dd = float(np.nanmin(dd.values)) if len(dd) else float("nan")
        calmar = ann_ret / abs(max_dd) if np.isfinite(max_dd) and max_dd != 0 else 0.0

        ax.plot(df.index, equity, label=f"{name}")
        metrics_data.append(
            [name, f"{sharpe:.2f}", f"{sortino:.2f}", f"{calmar:.2f}", f"{(equity.iloc[-1] / 1000 - 1):.2%}"]
        )

    ax.set_title("Dimension 1: Risk-Adjusted Returns (Sharpe / Sortino / Calmar)", fontsize=16)
    ax.legend()

    ax_tbl.axis("off")
    table = ax_tbl.table(
        cellText=metrics_data,
        colLabels=["Strategy Name", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Total Return"],
        loc="center",
        cellLoc="center",
    )
    table.scale(1, 2)
    plt.savefig(output_path / "1_Risk_Adjusted_Returns.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_dim2_vol(data_map, output_path):
    n_strat = len(data_map)
    fig_height = max(8, 4 + 0.6 * n_strat)

    fig, (ax, ax_tbl) = plt.subplots(
        2, 1,
        figsize=(15, fig_height),
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True
    )
    metrics_data = []

    for name, df in data_map.items():
       
        rets = pick_strategy_returns(df)
        rolling_vol = rets.rolling(21).std() * np.sqrt(252)
        realized_mean = float(np.nanmean(rolling_vol.values)) if np.isfinite(rolling_vol.values).any() else float("nan")
        vol_cv = (float(np.nanstd(rolling_vol.values)) / realized_mean) if np.isfinite(realized_mean) and realized_mean > 0 else float("nan")

        if "vol_estimate" in df.columns:
            ax.scatter(df["vol_estimate"], rolling_vol, alpha=0.3, s=10, label=name)

        metrics_data.append([name, f"{realized_mean:.2%}" if np.isfinite(realized_mean) else "nan",
                             f"{vol_cv:.4f}" if np.isfinite(vol_cv) else "nan",
                             f"{(realized_mean - 0.10):.2%}" if np.isfinite(realized_mean) else "nan"])

    ax.plot([0.05, 0.25], [0.05, 0.25], "k--", alpha=0.5, label="Ideal Control")
    ax.set_title("Dimension 2: Volatility Control Ability (Vol CV)", fontsize=16)
    ax.set_xlabel("Estimated Volatility")
    ax.set_ylabel("Realized Volatility")
    ax.legend()

    ax_tbl.axis("off")
    table = ax_tbl.table(
        cellText=metrics_data,
        colLabels=["Strategy Name", "Avg Realized Vol", "Vol CV", "Target Deviation"],
        loc="center",
        cellLoc="center",
    )
    table.scale(1, 2)
    plt.savefig(output_path / "2_Vol_Control.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_dim3_risk(data_map, output_path):
    n_strat = len(data_map)
    fig_height = max(8, 4 + 0.6 * n_strat)

    fig, (ax, ax_tbl) = plt.subplots(
        2, 1,
        figsize=(15, fig_height),
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True
    )
    metrics_data = []

    for name, df in data_map.items():
        equity = df.get("equity_curve", None)
        if equity is None:
            rets = pick_strategy_returns(df).fillna(0)
            equity = 1000.0 * np.exp(np.cumsum(rets.values))
            equity = pd.Series(equity, index=df.index)

        dd = (equity / equity.cummax() - 1)

        # VaR based on STRATEGY returns (extreme loss of strategy)
        strat_rets = pick_strategy_returns(df).fillna(0)
        var_95 = safe_percentile(strat_rets, 5)

        # Ulcer index
        ui = np.sqrt(safe_mean(np.square(dd.values)))

        stress_days = int((dd < -0.05).sum())

        ax.fill_between(df.index, dd, 0, alpha=0.3, label=name)

        metrics_data.append([
            name,
            f"{float(np.nanmin(dd.values)):.2%}" if np.isfinite(dd.values).any() else "nan",
            f"{ui:.4f}" if np.isfinite(ui) else "nan",
            f"{var_95:.2%}" if np.isfinite(var_95) else "nan",
            f"{stress_days} Days"
        ])

    ax.set_title("Dimension 3: Extreme Risk Metrics (MaxDD / Ulcer / VaR)", fontsize=16)
    ax.set_ylabel("Drawdown")
    ax.legend()

    ax_tbl.axis("off")
    table = ax_tbl.table(
        cellText=metrics_data,
        colLabels=["Strategy Name", "Max Drawdown", "Ulcer Index", "VaR (95%)", "Days with DD > 5%"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    plt.savefig(output_path / "3_Extreme_Risk.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_dim4_costs(data_map, output_path):
    n_strat = len(data_map)

    fig_height = max(8, 4 + 0.6 * n_strat)

    fig, (ax, ax_tbl) = plt.subplots(
        2, 1,
        figsize=(15, fig_height),
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True
    )

    metrics_data = []
    names = []
    turnovers = []

    for name, df in data_map.items():
        if "weight" not in df.columns:
            continue

        turnover = float(np.nanmean(df["weight"].diff().abs().values))
        strat_rets = pick_strategy_returns(df)

        pos_rets = strat_rets[strat_rets > 0]
        neg_rets = strat_rets[strat_rets < 0]

        if len(pos_rets) > 0 and len(neg_rets) > 0:
            win_loss = float(pos_rets.mean() / abs(neg_rets.mean()))
        else:
            win_loss = float("nan")

        win_rate = float(len(pos_rets) / len(strat_rets)) if len(strat_rets) > 0 else float("nan")

        names.append(name)
        turnovers.append(turnover)

        metrics_data.append([
            name,
            f"{turnover:.2%}" if np.isfinite(turnover) else "nan",
            f"{win_loss:.2f}" if np.isfinite(win_loss) else "nan",
            f"{win_rate:.2%}" if np.isfinite(win_rate) else "nan"
        ])

    # Bar plot
    x = np.arange(len(names))
    ax.bar(x, turnovers, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_title("Dimension 4: Trading Efficiency (Turnover / Win-Loss Ratio)", fontsize=16)
    ax.set_ylabel("Avg Daily Turnover")

    # Add small margin at top
    ax.margins(y=0.15)

 
    ax_tbl.axis("off")
    table = ax_tbl.table(
        cellText=metrics_data,
        colLabels=["Strategy Name", "Turnover", "Win-Loss Ratio", "Win Rate"],
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    fig.savefig(output_path / "4_Trading_Costs.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def compute_strategy_feature_table(data_map):
    """
    Build a per-strategy feature table using the same signals behind the 4 dimensions.
    Returns a DataFrame indexed by strategy name.
    """
    rows = []
    for name, df in data_map.items():
        rets = pick_strategy_returns(df).fillna(0.0).astype(float)
        if rets.dropna().empty:
            continue

        # Dimension 1: risk-adjusted returns
        ann_ret = safe_mean(rets.values) * 252
        ann_vol = float(np.nanstd(rets.values, ddof=0)) * np.sqrt(252)
        sharpe = ann_ret / ann_vol if np.isfinite(ann_vol) and ann_vol > 0 else float("nan")
        downside = rets[rets < 0].values
        downside_std = float(np.nanstd(downside, ddof=0)) * np.sqrt(252) if downside.size > 0 else float("nan")
        sortino = ann_ret / downside_std if np.isfinite(downside_std) and downside_std > 0 else float("nan")

        # equity curve
        equity = df.get("equity_curve", None)
        if equity is None:
            equity = 1000.0 * np.exp(np.cumsum(rets.values))
            equity = pd.Series(equity, index=df.index)

        dd = equity / equity.cummax() - 1.0
        max_dd = float(np.nanmin(dd.values)) if len(dd) else float("nan")
        calmar = ann_ret / abs(max_dd) if np.isfinite(max_dd) and max_dd != 0 else float("nan")
        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) else float("nan")

        # Dimension 2: vol control
        rolling_vol = rets.rolling(21).std() * np.sqrt(252)
        realized_mean = float(np.nanmean(rolling_vol.values)) if np.isfinite(rolling_vol.values).any() else float("nan")
        vol_cv = (float(np.nanstd(rolling_vol.values)) / realized_mean) if np.isfinite(realized_mean) and realized_mean > 0 else float("nan")
        target_deviation = (realized_mean - 0.10) if np.isfinite(realized_mean) else float("nan")

        # Dimension 3: risk metrics
        var_95 = safe_percentile(rets, 5)
        ulcer = float(np.sqrt(safe_mean(np.square(dd.values)))) if len(dd) else float("nan")
        stress_days = int((dd < -0.05).sum()) if len(dd) else 0

        # Dimension 4: trading costs/efficiency
        turnover = float(np.nanmean(df["weight"].diff().abs().values)) if "weight" in df.columns else float("nan")
        pos_rets = rets[rets > 0]
        neg_rets = rets[rets < 0]
        if len(pos_rets) > 0 and len(neg_rets) > 0:
            win_loss = float(pos_rets.mean() / abs(neg_rets.mean()))
        else:
            win_loss = float("nan")
        win_rate = float(len(pos_rets) / len(rets)) if len(rets) > 0 else float("nan")

        rows.append({
            "strategy": name,
            "total_return": total_return,
            "ann_return": ann_ret,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown": max_dd,
            "realized_vol_mean": realized_mean,
            "vol_cv": vol_cv,
            "target_deviation": target_deviation,
            "var_95": var_95,
            "ulcer_index": ulcer,
            "stress_days": stress_days,
            "turnover": turnover,
            "win_loss": win_loss,
            "win_rate": win_rate,
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).set_index("strategy")
    return out

def plot_strategy_embedding(data_map, output_path, perplexity=30, random_state=42):
    """
    Build a t-SNE embedding from 4-dimension feature table.
    Saves embedding CSV + PNG to output_path.
    """
    feats = compute_strategy_feature_table(data_map)
    if feats.empty or len(feats) < 3:
        print("Not enough strategies for embedding (need >= 3).")
        return

    # Select numeric features
    X = feats.copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    # Impute missing with column medians
    for c in X.columns:
        if X[c].isna().any():
            med = float(X[c].median()) if np.isfinite(X[c].median()) else 0.0
            X[c] = X[c].fillna(med)

    # Standardize
    Xn = (X - X.mean()) / X.std(ddof=0).replace(0.0, 1.0)
    Xn = Xn.to_numpy(dtype=float)

    # Try t-SNE; fallback to PCA if sklearn unavailable
    try:
        from sklearn.manifold import TSNE

        n = Xn.shape[0]
        perpl = min(perplexity, max(5, (n - 1) // 3))
        tsne = TSNE(
            n_components=2,
            perplexity=perpl,
            random_state=random_state,
            init="pca",
            method="exact",
        )
        emb = tsne.fit_transform(Xn)
        method = f"tSNE(perplexity={perpl})"
    except Exception:
        # PCA fallback via SVD
        Xc = Xn - Xn.mean(axis=0, keepdims=True)
        u, s, vt = np.linalg.svd(Xc, full_matrices=False)
        emb = u[:, :2] * s[:2]
        method = "PCA(fallback)"

    emb_df = pd.DataFrame(emb, index=feats.index, columns=["x", "y"])
    emb_df["method"] = method

    emb_df.to_csv(output_path / "strategy_embedding.csv")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(emb_df["x"], emb_df["y"], s=80, alpha=0.8)
    for name, row in emb_df.iterrows():
        ax.text(row["x"], row["y"], name, fontsize=9, ha="center", va="bottom")

    ax.set_title(f"Strategy Capability Embedding ({method})")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path / "strategy_embedding.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path / 'strategy_embedding.csv'}")
    print(f"Saved: {output_path / 'strategy_embedding.png'}")

def plot_strategy_embedding_dim3(data_map, output_path, perplexity=30, random_state=42):
    """
    t-SNE embedding based on three dimensions:
    - risk-adjusted returns (sharpe, sortino, calmar, ann_return, ann_vol, max_drawdown)
    - volatility control (realized_vol_mean, vol_cv, target_deviation)
    - extreme risk (var_95, ulcer_index, stress_days)
    """
    feats = compute_strategy_feature_table(data_map)
    if feats.empty or len(feats) < 3:
        print("Not enough strategies for embedding (need >= 3).")
        return

    use_cols = [
        "ann_return",
        "ann_vol",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "realized_vol_mean",
        "vol_cv",
        "target_deviation",
        "var_95",
        "ulcer_index",
        "stress_days",
    ]
    X = feats[use_cols].replace([np.inf, -np.inf], np.nan)

    for c in X.columns:
        if X[c].isna().any():
            med = float(X[c].median()) if np.isfinite(X[c].median()) else 0.0
            X[c] = X[c].fillna(med)

    Xn = (X - X.mean()) / X.std(ddof=0).replace(0.0, 1.0)
    Xn = Xn.to_numpy(dtype=float)

    try:
        from sklearn.manifold import TSNE
        n = Xn.shape[0]
        perpl = min(perplexity, max(5, (n - 1) // 3))
        tsne = TSNE(
            n_components=2,
            perplexity=perpl,
            random_state=random_state,
            init="pca",
            method="exact",
        )
        emb = tsne.fit_transform(Xn)
        method = f"tSNE(perplexity={perpl})"
    except Exception:
        Xc = Xn - Xn.mean(axis=0, keepdims=True)
        u, s, vt = np.linalg.svd(Xc, full_matrices=False)
        emb = u[:, :2] * s[:2]
        method = "PCA(fallback)"

    emb_df = pd.DataFrame(emb, index=feats.index, columns=["x", "y"])
    emb_df["method"] = method
    emb_df.to_csv(output_path / "strategy_embedding_dim3.csv")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(emb_df["x"], emb_df["y"], s=80, alpha=0.8)
    for name, row in emb_df.iterrows():
        ax.text(row["x"], row["y"], name, fontsize=9, ha="center", va="bottom")
    ax.set_title(f"Strategy Embedding (Risk / Vol Control / Extreme Risk) - {method}")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path / "strategy_embedding_dim3.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path / 'strategy_embedding_dim3.csv'}")
    print(f"Saved: {output_path / 'strategy_embedding_dim3.png'}")



def plot_precision_table(data_map, output_path, ann_factor=252.0):
    rows = []
    warnings = []

    for name, df in data_map.items():
        if "vol_estimate" not in df.columns:
            warnings.append(f"{name}: missing vol_estimate (skip precision metrics)")
            continue

        under, src = pick_underlying_returns(df)
        if under is None:
            warnings.append(f"{name}: missing underlying returns series (skip)")
            continue

        # If we had to fallback to "returns", warn because it might be strategy PnL in old files
        if src == "returns" and "asset_returns" not in df.columns and "returns_clean" not in df.columns:
            warnings.append(f"{name}: using df['returns'] fallback (verify this is UNDERLYING returns)")

        # evaluate_vol_forecast expects finite pairs
        m = evaluate_vol_forecast(
            returns=under,
            vol_hat=df["vol_estimate"],
            ann_factor=ann_factor,
        )

        # if metrics are empty/invalid, skip instead of crashing later
        if m is None or (isinstance(m, dict) and m.get("n", 0) == 0):
            warnings.append(f"{name}: precision metrics empty (n=0), skipped")
            continue

        m["strategy"] = name
        rows.append(m)

    out = pd.DataFrame(rows)
    if out.empty:
        print("No valid strategies for precision table.")
        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(" -", w)
        return

    out = out.set_index("strategy")

    cols = [
        "n",
        "qlike",
        "mse_var",
        "mae_vol",
        "oos_r2_var_vs_const",
        "mz_alpha",
        "mz_beta",
        "mz_r2",
    ]
    cols = [c for c in cols if c in out.columns]

    #qlike lower better
    if "qlike" in out.columns:
        out = out[cols].sort_values("qlike", ascending=True)
    else:
        out = out[cols]

    # save CSV
    out.to_csv(output_path / "precision_metrics.csv")

    # display formatting
    disp = out.copy()
    if "n" in disp.columns:
        disp["n"] = disp["n"].fillna(0).astype(int)

    for c in disp.columns:
        if c != "n":
            disp[c] = disp[c].map(lambda x: f"{x:.4f}" if pd.notna(x) and np.isfinite(x) else "nan")

    nrows, ncols = disp.shape
    fig_w = max(10, 1.2 * (ncols + 3))
    fig_h = max(2.5, 0.6 * (nrows + 3))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title("Volatility Estimation Precision Metrics (Forecast vs Realized Proxy)", fontsize=14, pad=12)

    table = ax.table(
        cellText=disp.values,
        rowLabels=disp.index.tolist(),
        colLabels=disp.columns.tolist(),
        loc="center",
        cellLoc="center",
        rowLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    fig.tight_layout()
    fig.savefig(output_path / "table.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Saved precision table CSV: {output_path / 'precision_metrics.csv'}")
    print(f"✅ Saved precision table PNG : {output_path / 'table.png'}")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(" -", w)



# Meta Analysis 

def plot_meta_analysis(data_map, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    names = []
    sharpes = []
    qlikes = []

    for name, df in data_map.items():
        if "vol_estimate" not in df.columns:
            continue

        # Profitability
        rets = pick_strategy_returns(df).fillna(0).values
        ann_ret = safe_mean(rets) * 252
        ann_vol = float(np.nanstd(rets, ddof=0)) * np.sqrt(252)
        sharpe = ann_ret / ann_vol if np.isfinite(ann_vol) and ann_vol > 0 else 0.0

        # Precision 
        under, _ = pick_underlying_returns(df)
        if under is None:
            continue

        metrics = evaluate_vol_forecast(
            returns=under,
            vol_hat=df["vol_estimate"],
            ann_factor=252.0,
        )
        qlike = metrics.get("qlike", np.nan) if isinstance(metrics, dict) else np.nan
        if not np.isfinite(qlike):
            continue

        names.append(name)
        sharpes.append(sharpe)
        qlikes.append(qlike)

        ax.scatter(qlike, sharpe, s=150, alpha=0.8, label=name)
        ax.text(qlike, sharpe + 0.02, name, fontsize=10, ha="center")

    ax.set_title("Meta-Analysis: Statistical Precision vs. Profitability", fontsize=14, fontweight="bold")
    ax.set_xlabel("Forecast Error (QLIKE Loss - Lower is Better) ⬅️")
    ax.set_ylabel("Profitability (Sharpe Ratio - Higher is Better) ⬆️")
    ax.grid(True, linestyle="--", alpha=0.6)

    # quadrant lines only if non-empty
    if len(sharpes) > 0:
        ax.axhline(y=float(np.mean(sharpes)), color="k", linestyle="--", alpha=0.3)
    if len(qlikes) > 0:
        ax.axvline(x=float(np.mean(qlikes)), color="k", linestyle="--", alpha=0.3)

    plt.savefig(output_path / "5_Meta_Analysis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f" Saved Meta-Analysis plot to: {output_path / '5_Meta_Analysis.png'}")

def plot_precision_timeseries(data_map, output_path, window=21, ann_factor=252.0, top_k=5):
    rows = []
    series = {}

    for name, df in data_map.items():
        if "vol_estimate" not in df.columns:
            continue
        under, _ = pick_underlying_returns(df)
        if under is None:
            continue

        rv = realized_vol_proxy(under, window=window, ann_factor=ann_factor)
        vh = pd.Series(df["vol_estimate"], index=df.index)
        loss = qlike_series(rv, vh)
        score = float(np.nanmean(loss.values)) if len(loss) else np.nan
        rows.append((name, score))
        series[name] = (rv, vh)

    if not rows:
        return

    rows = sorted(rows, key=lambda x: x[1])
    keep = [n for n, _ in rows[:top_k]]

    fig, ax = plt.subplots(figsize=(14, 6))
    # plot realized once (common axis)
    rv0 = series[keep[0]][0]
    ax.plot(rv0.index, rv0.values, linewidth=2, label=f"Realized vol proxy")

    for name in keep:
        rv, vh = series[name]
        ax.plot(vh.index, vh.values, alpha=0.8, label=name)

    ax.set_title("Forecast Vol vs Realized Vol Proxy")
    ax.set_ylabel("Annualized Vol")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path / "P_TS_Forecast_vs_Realized.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
def plot_precision_calibration(data_map, output_path, window=21, ann_factor=252.0, n_bins=10):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot([0, 0.5], [0, 0.5], "k--", alpha=0.5, label="Perfect calibration")

    for name, df in data_map.items():
        if "vol_estimate" not in df.columns:
            continue
        under, _ = pick_underlying_returns(df)
        if under is None:
            continue

        rv = realized_vol_proxy(under, window=window, ann_factor=ann_factor)
        vh = pd.Series(df["vol_estimate"], index=df.index)

        m = pd.concat([vh.rename("vh"), rv.rename("rv")], axis=1).dropna()
        if len(m) < 50:
            continue

        # decile bins by forecast
        m["bin"] = pd.qcut(m["vh"], q=n_bins, duplicates="drop")
        grp = m.groupby("bin").agg(vh_mean=("vh", "mean"), rv_mean=("rv", "mean"))

        ax.plot(grp["vh_mean"].values, grp["rv_mean"].values, marker="o", alpha=0.8, label=name)

    ax.set_title(f"Calibration Plot: Mean Realized Vol vs Mean Forecast Vol")
    ax.set_xlabel("Mean forecast vol")
    ax.set_ylabel("Mean realized vol proxy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path / "P_Calibration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
def plot_precision_by_regime(data_map, output_path, window=21, ann_factor=252.0, regime_col="vol_regime"):
    rows = []
    regimes = ["Low", "Mid", "High"]

    for name, df in data_map.items():
        if "vol_estimate" not in df.columns:
            continue
        under, _ = pick_underlying_returns(df)
        if under is None:
            continue

        rv = realized_vol_proxy(under, window=window, ann_factor=ann_factor)
        vh = pd.Series(df["vol_estimate"], index=df.index)

        loss = qlike_series(rv, vh)
        m = pd.concat([rv.rename("rv"), loss.rename("loss")], axis=1).dropna()
        if len(m) < 100:
            continue

        reg = get_vol_regime_labels(df, index=m.index, col=regime_col)
        if reg is None:
            continue

        m = m.copy()
        m["regime"] = reg
        m = m.dropna(subset=["regime"])

        low = m.loc[m["regime"].eq("Low"), "loss"].mean()
        mid = m.loc[m["regime"].eq("Mid"), "loss"].mean()
        high = m.loc[m["regime"].eq("High"), "loss"].mean()

        rows.append([name, low, mid, high])

    if not rows:
        return

    out = pd.DataFrame(rows, columns=["strategy"] + regimes).set_index("strategy")
    out = out.sort_values("High")  # focus on high-vol regime

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(out.index))
    width = 0.25
    ax.bar(x - width, out["Low"].values, width, label="Low")
    ax.bar(x,         out["Mid"].values, width, label="Mid")
    ax.bar(x + width, out["High"].values, width, label="High")
    ax.set_xticks(x)
    ax.set_xticklabels(out.index, rotation=45, ha="right")
    ax.set_title("Regime Precision: Average QLIKE Loss by vol_regime")
    ax.set_ylabel("Avg QLIKE loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path / "P_Regime_QLIKE.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_regime_explainer_table(
    data_map,
    output_path,
    window=21,
    ann_factor=252.0,
    horizon=0,
    min_n=80,
    regime_col="vol_regime",
):
    rows = []
    for name, df in data_map.items():
        m = align_forecast_and_realized(df, window=window, ann_factor=ann_factor, horizon=horizon)
        if m is None or len(m) < min_n:
            continue

        reg = get_vol_regime_labels(df, index=m.index, col=regime_col)
        if reg is None:
            continue

        m = m.copy()
        m["regime"] = reg
        m = m.dropna(subset=["regime"])

        for r in ["Low", "Mid", "High"]:
            mr = m[m["regime"].eq(r)]
            if len(mr) < 20:
                continue
            rows.append({
                "strategy": name,
                "regime": r,
                "n": int(len(mr)),
                "mean_abs_err": float(mr["abs_err"].mean()),
                "bias_err": float(mr["err"].mean()),
                "corr_vh_vs_target": corr_safe(mr["vh"], mr["target_rv"]),
                "mean_qlike": float(mr["loss"].mean()),
                "mean_vol_of_vol": float(mr["vol_of_vol"].mean()),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        print(" Regime explainer table empty (missing vol_regime or insufficient data).")
        return

    out = out.sort_values(["strategy", "regime"])
    out.to_csv(output_path / "Regime_Explainer_Table.csv", index=False)

    piv = out.pivot_table(
        index="strategy",
        columns="regime",
        values=["mean_abs_err", "bias_err", "corr_vh_vs_target", "mean_qlike", "mean_vol_of_vol"],
        aggfunc="first",
    )
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index()

    disp = piv.copy()
    for c in disp.columns:
        if c == "strategy":
            continue
        disp[c] = disp[c].map(lambda x: f"{x:.4f}" if pd.notna(x) and np.isfinite(x) else "nan")

    fig_w = max(12, 0.8 * (disp.shape[1] + 4))
    fig_h = max(3, 0.5 * (disp.shape[0] + 3))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(f"Regime Explainer Table by vol_regime (horizon={horizon})", fontsize=14, pad=12)

    table = ax.table(
        cellText=disp.values,
        colLabels=disp.columns.tolist(),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    fig.tight_layout()
    fig.savefig(output_path / "Regime_Explainer_Table.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f" Saved: {output_path / 'Regime_Explainer_Table.csv'}")
    print(f" Saved: {output_path / 'Regime_Explainer_Table.png'}")
def plot_abs_error_by_regime(
    data_map,
    output_path,
    window=21,
    ann_factor=252.0,
    horizon=0,
    top_k=7,
    min_n=120,
    regime_col="vol_regime",
):
    scores = []
    aligned = {}

    for name, df in data_map.items():
        m = align_forecast_and_realized(df, window=window, ann_factor=ann_factor, horizon=horizon)
        if m is None or len(m) < min_n:
            continue
        scores.append((name, float(np.nanmean(m["loss"].values))))
        aligned[name] = (df, m)

    if not scores:
        return

    scores = sorted(scores, key=lambda x: x[1])
    keep = [n for n, _ in scores[:top_k]]

    fig, ax = plt.subplots(figsize=(14, 6))
    positions, labels, data = [], [], []
    pos = 1

    for name in keep:
        df, m = aligned[name]

        reg = get_vol_regime_labels(df, index=m.index, col=regime_col)
        if reg is None:
            continue

        tmp = m.copy()
        tmp["regime"] = reg
        tmp = tmp.dropna(subset=["regime"])

        for r in ["Low", "Mid", "High"]:
            arr = tmp.loc[tmp["regime"].eq(r), "abs_err"].dropna().values
            if arr.size < 20:
                continue
            data.append(arr)
            positions.append(pos)
            labels.append(f"{name}\n{r}")
            pos += 1

        pos += 1

    if len(data) == 0:
        return

    ax.boxplot(data, positions=positions, showfliers=False)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(f"Abs Forecast Error by vol_regime (horizon={horizon})")
    ax.set_ylabel("|vol_forecast - vol_realized|")
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path / "R_AbsError_ByRegime.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
def plot_vol_of_vol_by_regime(
    data_map,
    output_path,
    window=21,
    ann_factor=252.0,
    horizon=0,
    top_k=7,
    min_n=120,
    regime_col="vol_regime",
):
    scores = []
    aligned = {}

    for name, df in data_map.items():
        m = align_forecast_and_realized(df, window=window, ann_factor=ann_factor, horizon=horizon)
        if m is None or len(m) < min_n:
            continue
        scores.append((name, float(np.nanmean(m["loss"].values))))
        aligned[name] = (df, m)

    if not scores:
        return

    scores = sorted(scores, key=lambda x: x[1])
    keep = [n for n, _ in scores[:top_k]]

    fig, ax = plt.subplots(figsize=(14, 6))
    positions, labels, data = [], [], []
    pos = 1

    for name in keep:
        df, m = aligned[name]

        reg = get_vol_regime_labels(df, index=m.index, col=regime_col)
        if reg is None:
            continue

        tmp = m.copy()
        tmp["regime"] = reg
        tmp = tmp.dropna(subset=["regime"])

        for r in ["Low", "Mid", "High"]:
            arr = tmp.loc[tmp["regime"].eq(r), "vol_of_vol"].dropna().values
            if arr.size < 20:
                continue
            data.append(arr)
            positions.append(pos)
            labels.append(f"{name}\n{r}")
            pos += 1

        pos += 1

    if len(data) == 0:
        return

    ax.boxplot(data, positions=positions, showfliers=False)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title("Vol-of-Vol (|Δ realized vol|) by vol_regime")
    ax.set_ylabel("|Δ realized vol|")
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path / "R_VolOfVol_ByRegime.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
if __name__ == "__main__":
    data, path = get_data()
    plot_dim1_returns(data, path)
    plot_dim2_vol(data, path)
    plot_dim3_risk(data, path)
    plot_dim4_costs(data, path)
    plot_strategy_embedding(data, path)
    plot_strategy_embedding_dim3(data, path)
    plot_meta_analysis(data, path)
    data_map_est = reduce_to_one_per_estimator(data)
    plot_precision_table(data_map_est, path)
    plot_precision_timeseries(data_map_est, path)
    plot_precision_calibration(data_map_est, path)
    plot_precision_by_regime(data_map_est, path)
    print(f"Analysis plots generated successfully at: {path}")
    horizon = 0

    plot_regime_explainer_table(data_map_est, path, horizon=horizon)
    plot_abs_error_by_regime(data_map_est, path, horizon=horizon, top_k=6)
    plot_vol_of_vol_by_regime(data_map_est, path, horizon=horizon, top_k=6)
    plot_regime_strategy_metrics_table(data, path)
