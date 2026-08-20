import sys
from pathlib import Path

# Fix the import path so Python finds the src folder
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np

# Import all four regime models
from src.regimes.hmm import HMMRegimeModel
from src.regimes.gmm import GMMRegimeModel
from src.regimes.bocpd import BOCPDRegimeModel
from src.regimes.mqr import MacroQuantileRegime
from src.regimes.evaluate import evaluate_regime_model

def main():
    print("Loading Master Dataset...")
    data_path = Path("data/processed/Master_Dataset.parquet") 
    
    if not data_path.exists():
        print(f"Error: Could not find {data_path}. Please check the path.")
        return

    df = pd.read_parquet(data_path)
    
    # 1. Handle Returns
    if 'returns_clean' in df.columns:
        ret_series = df['returns_clean']
    else:
        ret_series = df['returns']

    # 2. Calculate VRP (Variance Risk Premium)
    ann = 252.0
    rv_d = ret_series.pow(2)
    rv_w = rv_d.rolling(5).mean()
    vix_raw = df['vix_close'].astype(float)
    vix_var_daily = (vix_raw / 100.0)**2 / ann
    df['vrp'] = vix_var_daily / (rv_w + 1e-8)
    
    # 3. Calculate 5-day rolling RV20 for the 2D HMM input
    df['rv20_w'] = df['rv20'].rolling(5).mean()
    
    # Ensure required columns exist
    required_cols = ['rv20', 'rv20_w', 't10y2y', 'vrp']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns in dataset: {missing_cols}")
        return

    # --- FIT MODELS ---
    
    print("Fitting Gaussian HMM (Baseline with Persistence)...")
    hmm = HMMRegimeModel(n_components=3, prob_threshold=0.70, hold_days=3)
    hmm_signals = hmm.fit_predict(df[['rv20', 'rv20_w']])
    
    print("Fitting Macro GMM...")
    gmm = GMMRegimeModel(n_components=3)
    gmm_features = df[['t10y2y', 'vrp', 'rv20']] 
    gmm_signals = gmm.fit_predict(gmm_features)
    
    print("Fitting BOCPD (Structural Breaks)...")
    bocpd = BOCPDRegimeModel(hazard_rate=1/252)
    bocpd_continuous_signals = bocpd.fit_predict(ret_series)

    print("Fitting Macro Quantiles (MQR)...")
    mqr = MacroQuantileRegime(lookback_window=756)
    mqr_continuous_signals = mqr.fit_predict(df[['rv20', 'vrp', 't10y2y']])

    # --- DISCRETIZE CONTINUOUS SIGNALS FOR EVALUATION ---
    
    # BOCPD Discretization
    try:
        bocpd_discrete_signals = pd.qcut(
            bocpd_continuous_signals.rank(method='first'), 
            q=3, 
            labels=[0, 1, 2]
        ).astype(float)
    except Exception as e:
        bocpd_discrete_signals = pd.cut(
            bocpd_continuous_signals, 
            bins=[-np.inf, 0.10, 0.50, np.inf], 
            labels=[0, 1, 2]
        ).astype(float)

    # MQR Discretization
    mqr_discrete_signals = pd.cut(
        mqr_continuous_signals, 
        bins=[-np.inf, 0.33, 0.66, np.inf], 
        labels=[0, 1, 2]
    ).astype(float)

    # --- EVALUATE MODELS ---

    print("\nEvaluating Models...")
    hmm_eval = evaluate_regime_model(hmm_signals, ret_series, "Gaussian HMM")
    gmm_eval = evaluate_regime_model(gmm_signals, ret_series, "Macro GMM")
    bocpd_eval = evaluate_regime_model(bocpd_discrete_signals, ret_series, "BOCPD")
    mqr_eval = evaluate_regime_model(mqr_discrete_signals, ret_series, "Macro Quantiles")

    results_df = pd.DataFrame([hmm_eval, gmm_eval, bocpd_eval, mqr_eval]).set_index('Model Name').T
    
    print("\n--- Regime Model Comparison Results ---")
    print(results_df.to_markdown())
    
    # --- SAVE RESULTS ---
    
    out_dir = Path("results/regimes")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(out_dir / "regime_comparison_metrics.csv")
    
    signals_df = pd.DataFrame({
        'hmm_regime': hmm_signals,
        'gmm_regime': gmm_signals,
        'bocpd_prob': bocpd_continuous_signals,
        'bocpd_discrete': bocpd_discrete_signals,
        'mqr_score': mqr_continuous_signals,
        'mqr_discrete': mqr_discrete_signals
    }, index=df.index)
    signals_df.to_parquet(out_dir / "regime_signals.parquet")
    print(f"\nSaved metrics and signals to {out_dir}/")

if __name__ == "__main__":
    main()