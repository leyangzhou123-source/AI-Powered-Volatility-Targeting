# src/evaluation/precision_metrics.py
from __future__ import annotations

import numpy as np
import pandas as pd


def _corr_safe(a: pd.Series, b: pd.Series) -> float:
    tmp = pd.concat([pd.Series(a), pd.Series(b)], axis=1).dropna()
    if len(tmp) < 10:
        return np.nan
    if float(tmp.iloc[:, 0].std(ddof=1)) <= 0 or float(tmp.iloc[:, 1].std(ddof=1)) <= 0:
        return np.nan
    return float(tmp.iloc[:, 0].corr(tmp.iloc[:, 1]))


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
def realized_vol_next_from_returns(under: pd.Series, window=21, ann_factor=252.0, eps: float = 1e-12,) -> pd.Series:

    r = pd.Series(under).astype(float)

    # Sum of future squared returns
    fwd_r2 = (r.shift(-1) ** 2).rolling(window, min_periods=window).sum()

    rv_next = np.sqrt((ann_factor / window) * fwd_r2)

    return rv_next.rename("rv_next")


def qlike_loss_var(
    rv2: pd.Series,
    sigma2_hat: pd.Series,
    eps: float = 1e-12,
) -> pd.Series:
    """
    L = log(sigma^2_hat) + rv^2 / sigma^2_hat

    """
    s2 = np.maximum(pd.Series(sigma2_hat).astype(float), eps)
    rv2 = pd.Series(rv2).astype(float)
    return np.log(s2) + rv2 / s2


def oos_r2(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_base: pd.Series,
    eps: float = 1e-12,
) -> float:
    """
    R2 = 1 - SSE_model / SSE_base
    SSE = sum((y - yhat)^2)
    """
    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(y_pred).astype(float)
    y_base = pd.Series(y_base).astype(float)

    tmp = pd.concat(
        [y_true.rename("y"), y_pred.rename("m"), y_base.rename("b")], axis=1
    ).dropna()

    if len(tmp) < 5:
        return np.nan

    sse_m = float(np.sum((tmp["y"] - tmp["m"]) ** 2))
    sse_b = float(np.sum((tmp["y"] - tmp["b"]) ** 2))

    if sse_b <= eps:
        return np.nan

    return 1.0 - sse_m / sse_b


def mincer_zarnowitz(
    y: pd.Series,
    x: pd.Series,
    eps: float = 1e-12,
) -> dict:
    """
    y = alpha + beta * x + e
    """
    y = pd.Series(y).astype(float)
    x = pd.Series(x).astype(float)
    tmp = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()

    n = int(len(tmp))
    if n < 5:
        return {"n": n, "alpha": np.nan, "beta": np.nan, "r2": np.nan}

    yv = tmp["y"].to_numpy()
    xv = tmp["x"].to_numpy()

    x_mean = float(np.mean(xv))
    y_mean = float(np.mean(yv))

    x_dev = xv - x_mean
    y_dev = yv - y_mean

    sxx = float(np.sum(x_dev**2))
    if sxx <= eps:
        return {"n": n, "alpha": np.nan, "beta": np.nan, "r2": np.nan}

    beta = float(np.sum(x_dev * y_dev) / sxx)
    alpha = float(y_mean - beta * x_mean)

    y_hat = alpha + beta * xv
    sse = float(np.sum((yv - y_hat) ** 2))
    sst = float(np.sum((yv - y_mean) ** 2))

    r2 = np.nan if sst <= eps else float(1.0 - sse / sst)
    return {"n": n, "alpha": alpha, "beta": beta, "r2": r2}


def evaluate_vol_forecast(
    returns: pd.Series,
    vol_hat: pd.Series,
    ann_factor: float = 252.0,
    eps: float = 1e-12,
) -> dict:
    
    r = pd.Series(returns).astype(float)
    rv_next = realized_vol_next_from_returns(r, ann_factor=ann_factor, eps=eps)
    rv2_next = (rv_next**2).rename("rv2_next")

    sigma_hat = pd.Series(vol_hat).astype(float).rename("sigma_hat")

    tmp = pd.concat([sigma_hat, rv_next, rv2_next], axis=1).dropna()
    if len(tmp) < 5:
        return {
            "n": int(len(tmp)),
            "mse_var": np.nan,
            "mae_vol": np.nan,
            "qlike": np.nan,
            "corr": np.nan,
            "corr_var": np.nan,
            "mse_var_const": np.nan,
            "rel_mse_var": np.nan,
            "oos_r2_var_vs_const": np.nan,
            "mz_alpha": np.nan,
            "mz_beta": np.nan,
            "mz_r2": np.nan,
        }

    sigma2 = np.maximum(tmp["sigma_hat"] ** 2, eps)

    # primary errors 
    err_var = sigma2 - tmp["rv2_next"]
    mse_var = float(np.mean(err_var**2))

    # baseline
    rv2_mean = float(tmp["rv2_next"].mean())
    sigma2_base = pd.Series(rv2_mean, index=tmp.index, name="sigma2_base")
    err_var_base = sigma2_base - tmp["rv2_next"]
    mse_var_const = float(np.mean(err_var_base**2))

    oos_r2_var_vs_const = float(oos_r2(tmp["rv2_next"], sigma2, sigma2_base, eps=eps))
    rel_mse_var = float(mse_var / mse_var_const) if mse_var_const > eps else np.nan

   
    mae_vol = float(np.mean(np.abs(tmp["sigma_hat"] - tmp["rv_next"])))
    qlike = float(np.mean(qlike_loss_var(tmp["rv2_next"], sigma2, eps=eps)))
    corr = _corr_safe(tmp["sigma_hat"], tmp["rv_next"])
    corr_var = _corr_safe(pd.Series(sigma2, index=tmp.index), tmp["rv2_next"])

    # Mincer–Zarnowitz regression
    mz = mincer_zarnowitz(tmp["rv2_next"], pd.Series(sigma2, index=tmp.index), eps=eps)

    return {
        "n": int(len(tmp)),
        "mse_var": mse_var,
        "mae_vol": mae_vol,
        "qlike": qlike,
        "corr": corr,
        "corr_var": corr_var,
        "mse_var_const": mse_var_const,
        "rel_mse_var": rel_mse_var,
        "oos_r2_var_vs_const": oos_r2_var_vs_const,
        "mz_alpha": float(mz["alpha"]) if pd.notna(mz["alpha"]) else np.nan,
        "mz_beta": float(mz["beta"]) if pd.notna(mz["beta"]) else np.nan,
        "mz_r2": float(mz["r2"]) if pd.notna(mz["r2"]) else np.nan,
    }


def evaluate_multiple_estimators(
    returns: pd.Series,
    vol_forecasts: dict[str, pd.Series],
    ann_factor: float = 252.0,
) -> pd.DataFrame:
    
    rows = []
    for name, vol_hat in vol_forecasts.items():
        m = evaluate_vol_forecast(returns, vol_hat, ann_factor=ann_factor)
        m["model"] = name
        rows.append(m)

    out = pd.DataFrame(rows).set_index("model")

    cols = [
        "n",
        "qlike",
        "mse_var",
        "mae_vol",
        "corr",
        "corr_var",
        "oos_r2_var_vs_const",
        "rel_mse_var",
        "mse_var_const",
        "mz_alpha",
        "mz_beta",
        "mz_r2",
    ]
    cols = [c for c in cols if c in out.columns]
    out = out[cols]

    # sort by qlike (lower is better)
    if "qlike" in out.columns:
        out = out.sort_values("qlike", ascending=True)

    return out


def print_metrics_table(df: pd.DataFrame, title: str | None = None):
    """
    Pretty print
    """
    if title:
        print(f"\n {title}")

    if df.empty:
        print("(empty)")
        return

    df_fmt = df.copy()


    def _fmt(x, f):
        return f.format(x) if pd.notna(x) else "nan"

    if "n" in df_fmt.columns:
        df_fmt["n"] = df_fmt["n"].map(lambda x: str(int(x)) if pd.notna(x) else "nan")

    for c in df_fmt.columns:
        if c != "n":
            df_fmt[c] = df_fmt[c].map(lambda x: _fmt(x, "{:.4f}"))

    print(df_fmt.to_string())
