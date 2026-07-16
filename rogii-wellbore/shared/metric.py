"""Shared metric for ROGII Wellbore Geology Prediction.

From the task deck: dTVT = manualTVT - predictedTVT for each predicted point;
quality is the RMSE of all dTVT values. So the competition metric is plain RMSE
on the TVT prediction (units: feet). Both agents score with this so numbers match.
"""
import numpy as np


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# Canonical competition metric (alias for clarity at call sites).
score = rmse
