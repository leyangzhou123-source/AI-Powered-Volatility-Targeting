import sys
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_scatter_regimes(ax, df, vol_col, regime_col, title, colors, labels):
    """
    Plots a faint gray line for the time series, overlaid with colored scatter dots for the regimes.
    """
    # 1. Plot the continuous underlying line (faint gray)
    ax.plot(df.index, df[vol_col], color='gray', linewidth=0.8, alpha=0.5, zorder=1)
    
    # 2. Overlay the scatter points based on the regime
    if regime_col in df.columns:
        unique_states = sorted(df[regime_col].dropna().unique())
        for state in unique_states:
            mask = df[regime_col] == state
            ax.scatter(df.index[mask], 
                       df[vol_col][mask], 
                       color=colors.get(int(state), 'black'), 
                       s=12,          # Dot size
                       alpha=0.8,     # Slight transparency
                       zorder=2, 
                       label=labels.get(int(state), f'Regime {int(state)}'))

    # Formatting
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel("RV20 (%)", fontsize=10)
    
    # Only keep the horizontal grid lines for a cleaner look
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Hide the top and right spines (borders) for a cleaner "Tufte" look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def main():
    print("Loading datasets...")
    data_dir = Path("data/processed")
    results_dir = Path("results/regimes")
    
    master_path = data_dir / "Master_Dataset.parquet"
    signals_path = results_dir / "regime_signals.parquet"
    
    if not master_path.exists() or not signals_path.exists():
        print("Error: Could not find data or regime signals. Run compare_regimes.py first.")
        return

    df_master = pd.read_parquet(master_path)
    df_signals = pd.read_parquet(signals_path)
    
    # Align the data
    df = df_master.join(df_signals, how='inner')
    
    # Select timeframe (2018-2022 captures several distinct vol regimes)
    start_date = '2018-01-01'
    end_date = '2022-12-31'
    
    try:
        plot_df = df.loc[start_date:end_date].copy()
    except KeyError:
        plot_df = df.copy() # Fallback

    # Create the target variable column (RV20 as a percentage)
    plot_df['vol_pct'] = plot_df['rv20'] * 100.0

    # Define clean colors mapping to your reference image style
    state_colors = {0: '#4169E1',  # Royal Blue (Low Vol)
                    1: '#F6A800',  # Orange/Gold (Normal Vol)
                    2: '#D62728'}  # Crimson Red (High Vol / Stress)
    
    state_labels = {0: 'Regime 0 (Low Vol)',
                    1: 'Regime 1 (Normal Vol)',
                    2: 'Regime 2 (High Vol / Stress)'}

    # Set up the 3-panel plotting canvas
    plt.style.use('seaborn-v0_8-white')
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 10), sharex=True)

    # --- Plot the 3 Panels ---
    plot_scatter_regimes(axes[0], plot_df, 'vol_pct', 'hmm_regime', "Gaussian HMM Regime Output", state_colors, state_labels)
    plot_scatter_regimes(axes[1], plot_df, 'vol_pct', 'gmm_regime', "Macro GMM Regime Output", state_colors, state_labels)
    plot_scatter_regimes(axes[2], plot_df, 'vol_pct', 'mqr_discrete', "Macro Quantiles (Discrete) Output", state_colors, state_labels)

    # Standardize Y-axes across all plots so the volatility scale is identical
    y_max = plot_df['vol_pct'].max() * 1.1
    for ax in axes:
        ax.set_ylim(0, y_max)

    # Add the legend ONLY to the top panel, placed inside the chart area just like your reference image
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[0].legend(by_label.values(), by_label.keys(), loc='upper left', frameon=True, edgecolor='lightgray')

    # Formatting the x-axis dates on the bottom-most plot
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.autofmt_xdate()

    plt.tight_layout()
    
    # Save the plot
    out_dir = Path("results/plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "Volatility_Regimes_Clean.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to: {out_path}")
    
    plt.show()

if __name__ == "__main__":
    main()