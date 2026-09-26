import numpy as np
from sklearn.calibration import IsotonicRegression

class ProbabilityCalibrator:
    def __init__(self, method='isotonic'):
        self.method = method
        if method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        else:
            raise ValueError("Only 'isotonic' is currently supported for calibration.")
            
    def fit(self, raw_probs, labels):
        """
        Fits the calibrator. 
        IMPORTANT: raw_probs and labels must come from a dataset that reflects the REAL, 
        imbalanced candidate distribution (e.g., the validation set), NOT the balanced training set.
        """
        self.calibrator.fit(raw_probs, labels)
        
    def predict_proba(self, raw_probs):
        """
        Calibrates raw probabilities.
        """
        return self.calibrator.transform(raw_probs)
