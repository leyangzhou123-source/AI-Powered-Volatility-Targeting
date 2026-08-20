import importlib
import numpy as np
import pandas as pd
from src.estimators.base import Estimator

def _load_class(class_path: str):
    """Dynamically load an estimator class."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

class DynamicPrecisionEnsemble(Estimator):
    def __init__(self, params=None):
        super().__init__(params)
        
        # Load sub-estimators from config
        est_configs = self.params.get("estimators", [])
        if not est_configs:
            raise ValueError("Must provide 'estimators' list in params.")
            
        self.models = []
        for cfg in est_configs:
            cls = _load_class(cfg["class"])
            self.models.append(cls(cfg.get("params", {})))
            
        self.loss_window = int(self.params.get("loss_window", 21))
        self.loss_metric = str(self.params.get("loss_metric", "qlike")).lower()
        self.eps = 1e-8
        
    def _calculate_loss(self, realized_var: float, forecast_var: float) -> float:
        """Calculate loss (QLIKE or MSE) in variance space."""
        r2 = max(realized_var, self.eps)
        f2 = max(forecast_var, self.eps)
        
        if self.loss_metric == "qlike":
            return np.log(f2) + (r2 / f2)
        else: # MSE
            return (r2 - f2) ** 2

    def estimate_window(self, window_returns: pd.Series) -> float:
        r = pd.Series(window_returns).dropna().astype(float)
        
        # We need at least enough data to fit models + calculate the loss window
        if len(r) < self.loss_window + 10:
            return float("nan")
            
        n_models = len(self.models)
        losses = np.zeros(n_models)
        
        # 1. Evaluate historical loss over the last `loss_window` days
        for i in range(self.loss_window):
            # The historical window we are evaluating (up to t - loss_window + i)
            sub_w = r.iloc[: -(self.loss_window - i)]
            # The realized proxy for the *next* day (which is at index -loss_window + i)
            actual_ret = r.iloc[-(self.loss_window - i)]
            actual_var = (actual_ret ** 2) * 252.0
            
            for m_idx, model in enumerate(self.models):
                try:
                    # Some estimators use estimate_window, some use estimate
                    if hasattr(model, "estimate_window"):
                        forecast_vol = model.estimate_window(sub_w)
                    else:
                        forecast_vol = model.estimate(sub_w.index[-1], sub_w)
                        
                    if np.isfinite(forecast_vol) and forecast_vol > 0:
                        forecast_var = forecast_vol ** 2
                        losses[m_idx] += self._calculate_loss(actual_var, forecast_var)
                    else:
                        losses[m_idx] += 1e6 # Heavily penalize invalid forecasts
                except Exception:
                    losses[m_idx] += 1e6

        # 2. Convert cumulative losses to weights (inverse loss)
        # Avoid division by zero
        inv_losses = 1.0 / np.maximum(losses, self.eps)
        weights = inv_losses / np.sum(inv_losses)
        
        # 3. Generate final forecast for tomorrow
        final_forecast = 0.0
        for m_idx, model in enumerate(self.models):
            try:
                if hasattr(model, "estimate_window"):
                    f_vol = model.estimate_window(r)
                else:
                    f_vol = model.estimate(r.index[-1], r)
                    
                if np.isfinite(f_vol) and f_vol > 0:
                    final_forecast += weights[m_idx] * f_vol
                else:
                    # If the forecast is bad today, redistribute weight or fallback
                    pass 
            except Exception:
                pass
                
        return float(final_forecast) if final_forecast > 0 else float("nan")