import pandas as pd
import numpy as np
from scipy import stats

class BOCPDRegimeModel:
    """
    Bayesian Online Changepoint Detection.
    Outputs a continuous probability (0.0 to 1.0) of a structural break at time t.
    """
    def __init__(self, hazard_rate=1/252):
        self.hazard_rate = hazard_rate 

    def fit_predict(self, returns_series: pd.Series) -> pd.Series:
        """
        returns_series: pd.Series of daily returns
        Returns the continuous probability of a changepoint occurring at time t.
        """
        # Scale returns to percentage points to prevent PDF underflow on microscopic variances
        clean_rets = returns_series.dropna().values * 100.0
        T = len(clean_rets)
        
        R = np.zeros((T + 1, T + 1))
        R[0, 0] = 1.0 
        
        alpha0, beta0 = 1.0, 1.0
        kappa0, mu0 = 1.0, 0.0
        
        alphas, betas = np.array([alpha0]), np.array([beta0])
        kappas, mus = np.array([kappa0]), np.array([mu0])
        
        for t in range(1, T + 1):
            x = clean_rets[t - 1]
            
            df = 2 * alphas
            scale = np.sqrt(betas * (kappas + 1) / (alphas * kappas))
            pred_probs = stats.t.pdf(x, df=df, loc=mus, scale=scale)
            
            R[t, 1:t+1] = R[t-1, 0:t] * pred_probs * (1 - self.hazard_rate)
            R[t, 0] = np.sum(R[t-1, 0:t] * pred_probs * self.hazard_rate)
            
            R[t, 0:t+1] /= np.sum(R[t, 0:t+1])
            
            alphas_new = alphas + 0.5
            kappas_new = kappas + 1.0
            mus_new = (kappas * mus + x) / kappas_new
            betas_new = betas + (kappas * (x - mus)**2) / (2 * kappas_new)
            
            alphas = np.append([alpha0], alphas_new)
            betas = np.append([beta0], betas_new)
            kappas = np.append([kappa0], kappas_new)
            mus = np.append([mu0], mus_new)

        # Return the continuous probability of a structural break
        cp_probs = R[1:, 0]
        
        out = pd.Series(np.nan, index=returns_series.index)
        out.loc[returns_series.dropna().index] = cp_probs
        return out