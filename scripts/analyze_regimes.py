import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from hmmlearn.hmm import GaussianHMM


root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.env import Env

def fit_global_regimes():
    """Fits HMM on prices.parquet rv20 to get global market regimes."""
    print("🌍 Fitting Global HMM on RV20...")
    
    # Path to your prices parquet
    prices_path = Env.path("prices", "databento") / "prices.parquet"
    if not prices_path.exists():
        # Fallback if it's in raw
        prices_path = Env.path("raw") / "prices.parquet"
        
    prices = pd.read_parquet(prices_path)
    prices = prices.dropna(subset=['rv20']).sort_index()
    
    # Fit HMM on log(rv20)
    rv20 = prices['rv20'].astype(float)
    log_rv = np.log(rv20 + 1e-8).values.reshape(-1, 1)
    
    model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000, random_state=42)
    model.fit(log_rv)
    states = model.predict(log_rv)
    

    state_means = model.means_.flatten()
    order = np.argsort(state_means)
    mapping = {order[0]: "Low", order[1]: "Mid", order[2]: "High"}
    
    regime_labels = pd.Series(states).map(mapping)
    regime_labels.index = prices.index
    
    return pd.DataFrame({'rv20': rv20, 'global_regime': regime_labels})

def compute_drawdown(equity):
    cummax = equity.cummax()
    return (equity - cummax) / cummax

def run_strategy_analysis(strat_name, strat_df, global_regimes, dirs):
    """Generates all reports and plots for a single strategy."""
    print(f"  ▶ Processing: {strat_name}")
    
    # Merge strategy data with global regimes
    df = strat_df.join(global_regimes, how='inner').dropna(subset=['global_regime'])
    
    if df.empty:
        print(f"No overlapping data for {strat_name}. Skipping.")
        return

    # Calculate Benchmark  Equity vs Strategy Equity
    strat_rets = df['returns'].astype(float)
    bh_rets = df['asset_returns'].astype(float) # The unscaled raw returns
    
    strat_eq = 1000 * np.exp(np.cumsum(strat_rets))
    bh_eq = 1000 * np.exp(np.cumsum(bh_rets))
    
    strat_dd = compute_drawdown(strat_eq)
    

    # 1. Distribution Comparison (Mean & Variance by Regime)

    dist_data = []
    for regime in ["Low", "Mid", "High"]:
        mask = df['global_regime'] == regime
        if mask.sum() == 0: continue
            
        r_strat = strat_rets[mask]
        r_bh = bh_rets[mask]
        
        dist_data.append({
            "Regime": regime,
            "Days": mask.sum(),
            "Strat Mean (Ann)": r_strat.mean() * 252,
            "Strat Variance (Ann)": r_strat.var() * 252,
            "B&H Mean (Ann)": r_bh.mean() * 252,
            "B&H Variance (Ann)": r_bh.var() * 252
        })
    
    pd.DataFrame(dist_data).to_csv(dirs['dist'] / f"{strat_name}_distribution.csv", index=False)
    

    # 2. 4D Metrics (VaR, MaxDD, Ulcer, Accuracy)
    # ---------------------------------------------------------
    metrics_data = []
    for regime in ["Low", "Mid", "High"]:
        mask = df['global_regime'] == regime
        if mask.sum() == 0: continue
            
        # Risk
        r_strat = strat_rets[mask]
        dd_strat = strat_dd[mask]
        var_95 = np.percentile(r_strat, 5)
        max_dd = dd_strat.min()
        ulcer = np.sqrt(np.mean(dd_strat**2))
        

        vol_est = df.loc[mask, 'vol_estimate']
        rv20 = df.loc[mask, 'rv20']
        
        # QLIKE
        vol_est_sq = np.maximum(vol_est**2, 1e-12)
        rv_sq = rv20**2
        qlike = np.mean(np.log(vol_est_sq) + (rv_sq / vol_est_sq))
        mse = np.mean((vol_est - rv20)**2)
        
        metrics_data.append({
            "Regime": regime,
            "VaR (95%)": var_95,
            "Max DD": max_dd,
            "Ulcer Index": ulcer,
            "QLIKE (Accuracy)": qlike,
            "MSE (Accuracy)": mse
        })
        
    pd.DataFrame(metrics_data).to_csv(dirs['risk'] / f"{strat_name}_4d_metrics.csv", index=False)
    
 
    # 3.Plots
   
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Equity Curve Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(strat_eq.index, strat_eq, label=f"Strategy ({strat_name})", color='#2c3e50', lw=2)
    ax.plot(bh_eq.index, bh_eq, label="Buy & Hold Benchmark", color='#7f8c8d', lw=1.5, alpha=0.8)
    ax.set_title(f"Equity Curve: Strategy vs Buy & Hold\n({strat_name})")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    fig.savefig(dirs['ts'] / f"{strat_name}_equity.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    # Volatility Tracking
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['rv20'], label="Actual RV20", color='grey', alpha=0.5)
    ax.plot(df.index, df['vol_estimate'], label="Estimator Forecast", color='#e74c3c', alpha=0.8)
    ax.set_title(f"Volatility Forecast vs Realized RV20\n({strat_name})")
    ax.set_ylabel("Annualized Volatility")
    ax.legend()
    fig.savefig(dirs['ts'] / f"{strat_name}_vol_tracking.png", dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_global_regimes(global_regimes, output_dir):
    """Plots a scatter overlay of HMM regimes on the RV20 time series."""
    print("📊 Plotting Global Regime Scatter Analysis...")
    
    fig, ax = plt.subplots(figsize=(15, 7))
    

    ax.plot(global_regimes.index, global_regimes['rv20'], 
            color='steelblue', lw=1, alpha=0.4, label="RV20 Path")
    

  
    regime_meta = [
        {"name": "Low", "color": "#1f77b4", "marker": "o"},   # Blue
        {"name": "Mid", "color": "#2ca02c", "marker": "s"},   # Green
        {"name": "High", "color": "#ff7f0e", "marker": "^"}   # Orange
    ]

    # Overlay scatter points for each regime
    for meta in regime_meta:
        mask = global_regimes['global_regime'] == meta['name']
        ax.scatter(global_regimes.index[mask], global_regimes.loc[mask, 'rv20'], 
                   color=meta['color'], 
                   label=f"Regime: {meta['name']}", 
                   s=12,  # point size
                   alpha=0.8,
                   edgecolor='none')

    ax.set_title("HMM Volatility Regime Classification (Scatter Overlay)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Realized Volatility (RV20)")
    ax.set_xlabel("Year")
    ax.legend(loc='upper left', frameon=True)
    
    plt.tight_layout()

    save_path = output_dir / "global_regime_scatter.png"
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f" Scatter plot saved to: {save_path}")

def main():

    base_res = Env.path("results")
    dirs = {
        'dist': base_res / "regime_performance",
        'risk': base_res / "risk_metrics_4d",
        'ts': base_res / "time_series_plots"
    }
    
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
        
    # Fit Global Regimes
    global_regimes = fit_global_regimes()
    plot_global_regimes(global_regimes, dirs['ts'])
    
    # Iterate through strategies
    for parquet_file in base_res.glob("*.parquet"):
        strat_name = parquet_file.stem
        try:
            strat_df = pd.read_parquet(parquet_file)
            run_strategy_analysis(strat_name, strat_df, global_regimes, dirs)
        except Exception as e:
            print(f" Failed on {strat_name}: {e}")

    print("\n All regime analysis completed successfully.")

if __name__ == "__main__":
    main()