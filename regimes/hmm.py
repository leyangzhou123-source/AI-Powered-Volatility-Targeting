import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM

class HMMRegimeModel:
    """
    Gaussian HMM utilizing 2D inputs (e.g., daily RV and weekly RV) 
    with soft probability persistence filtering and a sticky transition matrix.
    """
    def __init__(self, n_components=3, random_state=42, prob_threshold=0.70, hold_days=3):
        self.n_components = n_components
        self.prob_threshold = prob_threshold
        self.hold_days = hold_days
        
        # Initialize with init_params="cm" (means and covars), leaving "t" (transmat) out 
        # so we can manually set our own sticky prior.
        self.model = GaussianHMM(n_components=n_components, covariance_type="diag", 
                                 n_iter=1000, random_state=random_state, init_params="cm")
        
        # Encode a strong prior that regimes last ~20 days (0.95 diagonal)
        sticky_transmat = np.ones((n_components, n_components)) * ((1.0 - 0.95) / (n_components - 1))
        np.fill_diagonal(sticky_transmat, 0.95)
        self.model.transmat_ = sticky_transmat
        
        self.regime_map = {}

    def fit_predict(self, features_df: pd.DataFrame) -> pd.Series:
        """
        features_df: pd.DataFrame containing e.g., ['rv20', 'rv_w']. 
        Providing a short-term rolling average alongside the raw RV anchors the state.
        """
        clean_df = features_df.dropna()
        if len(clean_df) < 50:
            return pd.Series(np.nan, index=features_df.index)

        # Work in log-variance space for all provided features
        log_features = np.log(clean_df.values + 1e-8)
        
        self.model.fit(log_features)
        
        # Extract soft probabilities instead of hard states
        probs = self.model.predict_proba(log_features)

        # Order regimes based on the emission mean of the FIRST feature (assumed to be raw RV)
        state_means = self.model.means_[:, 0] 
        order = np.argsort(state_means)
        self.regime_map = {int(order[k]): k for k in range(self.n_components)}
        
        # Realign the probability matrix so column 0 is Low Vol, column 2 is High Vol
        ordered_probs = np.zeros_like(probs)
        for i in range(self.n_components):
            ordered_probs[:, self.regime_map[i]] = probs[:, i]

        # Persistence Filter Logic
        raw_argmax = np.argmax(ordered_probs, axis=1)
        smoothed_states = np.copy(raw_argmax)
        current_state = raw_argmax[0]
        consecutive_days = 0

        for t in range(1, len(raw_argmax)):
            target_state = raw_argmax[t]
            target_prob = ordered_probs[t, target_state]

            # If the model wants to change states and is highly confident
            if target_state != current_state and target_prob >= self.prob_threshold:
                consecutive_days += 1
                if consecutive_days >= self.hold_days:
                    current_state = target_state
                    consecutive_days = 0 # Reset counter after successful transition
            else:
                # Reset if confidence drops or it flips back to current state
                consecutive_days = 0 

            smoothed_states[t] = current_state

        out = pd.Series(np.nan, index=features_df.index)
        out.loc[clean_df.index] = smoothed_states
        return out