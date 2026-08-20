import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

class GMMRegimeModel:
    """
    Gaussian Mixture Model for regime detection using multi-dimensional macro features.
    """
    def __init__(self, n_components=3, random_state=42):
        self.n_components = n_components
        self.model = GaussianMixture(n_components=n_components, covariance_type='full', random_state=random_state)
        self.scaler = StandardScaler()
        self.regime_map = {}

    def fit_predict(self, features_df: pd.DataFrame) -> pd.Series:
        """
        features_df: DataFrame containing columns like ['t10y2y', 'vrp', 'rv20']
        Ensure 'rv20' (or your target vol metric) is included so we can sort properly.
        """
        clean_df = features_df.dropna()
        if len(clean_df) < 50:
            return pd.Series(np.nan, index=features_df.index)

        scaled_features = self.scaler.fit_transform(clean_df)
        
        self.model.fit(scaled_features)
        states = self.model.predict(scaled_features)

        # Find the index of the volatility column to ensure correct low/med/high sorting
        # Fall back to index -1 (last column) if 'rv20' isn't explicitly named
        vol_col_idx = list(features_df.columns).index('rv20') if 'rv20' in features_df.columns else -1
        
        # Sort states STRICTLY by the unscaled mean of the volatility feature
        # Inverse transform the means to get back to real-world units
        unscaled_means = self.scaler.inverse_transform(self.model.means_)
        state_vol_means = unscaled_means[:, vol_col_idx]
        
        order = np.argsort(state_vol_means)
        self.regime_map = {int(order[k]): k for k in range(self.n_components)}
        
        ordered_states = np.vectorize(self.regime_map.get)(states)
        
        out = pd.Series(np.nan, index=features_df.index)
        out.loc[clean_df.index] = ordered_states
        return out