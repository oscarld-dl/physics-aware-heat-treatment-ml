from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    ElasticNetCV,
    HuberRegressor,
    Lars,
    Lasso,
    LassoCV,
    LassoLars,
    LassoLarsIC,
    Ridge,
    RidgeCV,
)
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


def multioutput_r2(y_true, y_pred) -> float:
    """Mean R² across output columns."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return float(np.mean([
        r2_score(y_true[:, i], y_pred[:, i])
        for i in range(y_true.shape[1])
    ]))


CUSTOM_SCORER = make_scorer(multioutput_r2, greater_is_better=True)

@dataclass
class DatasetSplit:
    X: pd.DataFrame
    Y: pd.DataFrame


def select_xy_from_master(df: pd.DataFrame) -> DatasetSplit:
    """
    Preserve the thesis appendix slicing first.
    Verify later whether [:89] is correct for your actual sheet.
    """
    X = df.iloc[:89, 1:12].copy()
    Y = df.iloc[:89, 12:].copy()
    return DatasetSplit(X=X, Y=Y)


def build_base_models(random_state: int = 42) -> Dict[str, RegressorMixin]:
    return {
        "Random Forest": RandomForestRegressor(random_state=random_state),
        "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
        "AdaBoost": AdaBoostRegressor(random_state=random_state),
        "Decision Tree": DecisionTreeRegressor(random_state=random_state),
        "Ridge": Ridge(),
        "RidgeCV": RidgeCV(),
        "Lasso": Lasso(),
        "LassoCV": LassoCV(),
        "Lars": Lars(),
        "ElasticNet": ElasticNet(),
        "ElasticNetCV": ElasticNetCV(),
        "LassoLars": LassoLars(),
        "LassoLarsIC": LassoLarsIC(),
        "BayesianRidge": BayesianRidge(),
        "HuberRegressor": HuberRegressor(),
        "SVR (RBF Kernel)": SVR(kernel="rbf"),
        "MLP Regressor": MLPRegressor(
            hidden_layer_sizes=(200, 100),
            activation="relu",
            solver="lbfgs",
            max_iter=3500,
            random_state=random_state,
        ),
    }


def wrap_model(model: RegressorMixin) -> RegressorMixin:
    """
    Git/research-ready version:
    - scale X inside the CV pipeline
    - scale Y inside each fold via TransformedTargetRegressor
    - support multi-output through MultiOutputRegressor
    """
    single_target_pipeline = Pipeline(
        steps=[
            ("x_scaler", StandardScaler()),
            ("regressor", model),
        ]
    )

    single_target_with_y_scaling = TransformedTargetRegressor(
        regressor=single_target_pipeline,
        transformer=StandardScaler(),
    )

    return MultiOutputRegressor(single_target_with_y_scaling)


def evaluate_models(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    random_state: int = 42,
    n_splits: int = 5,
    n_repeats: int = 100,
) -> pd.DataFrame:
    base_models = build_base_models(random_state=random_state)
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    results = []
    for name, model in base_models.items():
        wrapped = wrap_model(model)
        scores = cross_val_score(
            wrapped,
            X,
            Y,
            cv=rkf,
            scoring=CUSTOM_SCORER,
            n_jobs=-1,
        )
        results.append(
            {
                "Model": name,
                "Average_R2": float(np.mean(scores)),
                "Std_R2": float(np.std(scores)),
            }
        )

    results_df = pd.DataFrame(results).sort_values(by="Average_R2", ascending=False).reset_index(drop=True)
    return results_df
