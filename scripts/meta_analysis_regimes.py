import sys
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from hmmlearn.hmm import GaussianHMM

# Ensure src can be imported
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.env import Env

# ============================================================
# 1. Global Regime Logic
# ============================================================
def fit_global_regimes():
    """Fits HMM on prices.parquet rv20 to get global market regimes."""
    print("🌍 Fitting Global HMM on RV20 for standardized regime definitions...")
    
    prices_path = Env.path("prices", "databento") / "prices.parquet"
    if not prices_path.exists():
        prices_path = Env.path("raw") / "prices.parquet"
        
    prices = pd.read_parquet(prices_path)
    prices = prices.dropna(subset=['rv20']).sort_index()
    
    rv20 = prices['rv20'].astype(float)
    log_rv = np.log(rv20 + 1e-8).values.reshape(-1, 1)
    
    model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000, random_state=42)
    model.fit(log_rv)
    states = model.predict(log_rv)
    
    # Order regimes: 0=Low, 1=Mid, 2=High
    state_means = model.means_.flatten()
    order = np.argsort(state_means)
    mapping = {order[0]: "Low", order[1]: "Mid", order[2]: "High"}
    
    regime_labels = pd.Series(states).map(mapping)
    regime_labels.index = prices.index
    
    return pd.DataFrame({'rv20': rv20, 'global_regime': regime_labels})

# ============================================================
# 2. Strategy Loader & Controller Filter
# ============================================================
def map_strategies_to_controllers():
    """Reads all YAML files to map strategy names to their controller classes."""
    strat_to_ctrl = {}
    strategies_dir = Env.path("strategies")
    for p in strategies_dir.glob("*.yaml"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                if cfg and "name" in cfg:
                    ctrl_class = cfg.get("controller", {}).get("class", "").lower()
                    strat_to_ctrl[cfg["name"]] = ctrl_class
        except Exception as e:
            print(f"⚠️ Could not read {p.name}: {e}")
    return strat_to_ctrl

def load_filtered_data(target_controller):
    """Loads parquet results, filtering for the specified controller type."""
    data_map = {}
    results_dir = Env.path("results")
    strat_to_ctrl = map_strategies_to_controllers()
    
    target_controller = target_controller.lower()
    
    for p_file in results_dir.glob("*.parquet"):
        strat_name = p_file.stem
        # Check if this strategy uses the requested controller
        ctrl_class = strat_to_ctrl.get(strat_name, "")
        if target_controller in ctrl_class:
            try:
                data_map[strat_name] = pd.read_parquet(p_file)
            except Exception as e:
                print(f"⚠️ Failed to read {p_file.name}: {e}")
                
    return data_map, results_dir

# ============================================================
# 3. Metric Calculations & Plotting
# ============================================================
def calculate_metrics(df_subset):
    """Calculates Sharpe and QLIKE for a subset of data."""
    if df_subset.empty:
        return np.nan, np.nan
        
    # Calculate Sharpe (Profitability)
    rets = df_subset['returns'].astype(float)
    ann_ret = rets.mean() * 252
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    
    # Calculate QLIKE (Precision/Accuracy) based on underlying global RV20
    vol_est = df_subset['vol_estimate'].astype(float)
    rv20 = df_subset['rv20'].astype(float)
    
    vol_est_sq = np.maximum(vol_est**2, 1e-12)
    rv_sq = rv20**2
    qlike = np.mean(np.log(vol_est_sq) + (rv_sq / vol_est_sq))
    
    return sharpe, qlike

def plot_meta_scatter(plot_data, title, output_path):
    """Generates the Meta-Analysis scatter plot with distinct colors and a legend."""
    if not plot_data:
        print(f"⚠️ No valid data for {title}, skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(11, 6))

    names = [d['name'] for d in plot_data]
    sharpes = [d['sharpe'] for d in plot_data]
    qlikes = [d['qlike'] for d in plot_data]

    # Generate a distinct color for each strategy using the tab20 colormap
    colors = cm.get_cmap('tab20', len(names))

    # Plot each point individually
    for i in range(len(names)):
        # Extract base estimator name (e.g., 'ar1_vol_naive_scaling' -> 'AR1')
        short_name = names[i].split("_")[0].upper()
        if short_name == "BUY": 
            short_name = "BUY & HOLD"
            
        ax.scatter(qlikes[i], sharpes[i], 
                   s=150, alpha=0.85, 
                   color=colors(i), 
                   edgecolor='black', linewidth=0.5,
                   label=short_name)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Forecast Error (QLIKE Loss - Lower is Better) ⬅️")
    ax.set_ylabel("Profitability (Sharpe Ratio - Higher is Better) ⬆️")
    ax.grid(True, linestyle="--", alpha=0.6)

    # Add quadrant lines (averages)
    if sharpes:
        ax.axhline(y=float(np.mean(sharpes)), color="k", linestyle="--", alpha=0.3)
    if qlikes:
        ax.axvline(x=float(np.mean(qlikes)), color="k", linestyle="--", alpha=0.3)

    # Place the legend *outside* the plot box
    ax.legend(title="Strategies", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {output_path.name}")

def main():
    parser = argparse.ArgumentParser(description="Meta-Analysis by Controller and Regime")
    parser.add_argument("--controller", type=str, required=True, 
                        help="Target controller (e.g., 'naive', 'variance', 'regime')")
    args = parser.parse_args()

    # 1. Get global market regimes
    global_regimes = fit_global_regimes()

    # 2. Get strategies filtered by controller
    data_map, results_dir = load_filtered_data(args.controller)
    if not data_map:
        print(f"❌ No parquet results found for strategies using controller: '{args.controller}'")
        sys.exit(1)
        
    print(f"🔍 Analyzing {len(data_map)} strategies using '{args.controller}' controller...")

    # Output directory setup
    out_dir = results_dir / f"meta_analysis_{args.controller.replace(' ', '_')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. Structure to hold calculated metrics
    metrics_overall = []
    metrics_low = []
    metrics_mid = []
    metrics_high = []

    # 4. Iterate and calculate
    for name, strat_df in data_map.items():
        if "vol_estimate" not in strat_df.columns:
            continue
            
        # Join strategy data with global regimes
        df = strat_df.join(global_regimes, how='inner').dropna(subset=['global_regime'])
        if df.empty:
            continue

        # Overall
        s_all, q_all = calculate_metrics(df)
        if np.isfinite(s_all) and np.isfinite(q_all):
            metrics_overall.append({'name': name, 'sharpe': s_all, 'qlike': q_all})

        # Low Regime
        df_low = df[df['global_regime'] == 'Low']
        s_low, q_low = calculate_metrics(df_low)
        if np.isfinite(s_low) and np.isfinite(q_low):
            metrics_low.append({'name': name, 'sharpe': s_low, 'qlike': q_low})

        # Mid Regime
        df_mid = df[df['global_regime'] == 'Mid']
        s_mid, q_mid = calculate_metrics(df_mid)
        if np.isfinite(s_mid) and np.isfinite(q_mid):
            metrics_mid.append({'name': name, 'sharpe': s_mid, 'qlike': q_mid})

        # High Regime
        df_high = df[df['global_regime'] == 'High']
        s_high, q_high = calculate_metrics(df_high)
        if np.isfinite(s_high) and np.isfinite(q_high):
            metrics_high.append({'name': name, 'sharpe': s_high, 'qlike': q_high})

    # 5. Generate the 4 Scatter Plots
    plot_meta_scatter(metrics_overall, 
                      f"Overall Meta-Analysis ({args.controller.title()})", 
                      out_dir / f"1_overall_meta_{args.controller}.png")
                      
    plot_meta_scatter(metrics_low, 
                      f"Low Volatility Regime Meta-Analysis ({args.controller.title()})", 
                      out_dir / f"2_low_regime_meta_{args.controller}.png")
                      
    plot_meta_scatter(metrics_mid, 
                      f"Middle Volatility Regime Meta-Analysis ({args.controller.title()})", 
                      out_dir / f"3_mid_regime_meta_{args.controller}.png")
                      
    plot_meta_scatter(metrics_high, 
                      f"High Volatility Regime Meta-Analysis ({args.controller.title()})", 
                      out_dir / f"4_high_regime_meta_{args.controller}.png")

if __name__ == "__main__":
    main()