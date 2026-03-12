#!/usr/bin/env python3
import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import IsolationForest
from typing import Optional

logger = logging.getLogger("anomaly_app")


class AnomalyDetector:

    def __init__(self, z_threshold: float = 3.0, contamination: float = 0.05):
        self.z_threshold = z_threshold
        self.contamination = contamination

    def zscore_flag(
        self,
        values: pd.Series,
        mean: float,
        std: float
    ) -> pd.Series:
        """
        Flag values more than z_threshold standard deviations from the
        established baseline mean. Returns a Series of z-scores.
        """
        try:
            logger.info(f"Running z-score calculation with mean={mean:.4f}, std={std:.4f}")

            if std == 0:
                logger.info("Standard deviation is 0; returning zero z-scores")
                return pd.Series([0.0] * len(values))

            z_scores = (values - mean).abs() / std
            logger.info(f"Computed z-scores for {len(values)} values")
            return z_scores

        except Exception as e:
            logger.exception(f"Error during z-score calculation: {e}")
            return pd.Series([0.0] * len(values))

    def isolation_forest_flag(self, df: pd.DataFrame, numeric_cols: list[str]) -> np.ndarray:
        """
        Multivariate anomaly detection across all numeric channels simultaneously.
        IsolationForest returns -1 for anomalies, 1 for normal points.
        Scores closer to -1 indicate stronger anomalies.
        """
        try:
            logger.info(f"Running IsolationForest on {len(df)} rows and {len(numeric_cols)} numeric column(s)")

            model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )

            X = df[numeric_cols].fillna(df[numeric_cols].median())
            model.fit(X)

            labels = model.predict(X)
            scores = model.decision_function(X)

            anomaly_count = int((labels == -1).sum())
            logger.info(f"IsolationForest completed with {anomaly_count} anomaly/anomalies detected")

            return labels, scores

        except Exception as e:
            logger.exception(f"Error during IsolationForest calculation: {e}")
            return np.ones(len(df)), np.zeros(len(df))

    def run(
        self,
        df: pd.DataFrame,
        numeric_cols: list[str],
        baseline: dict,
        method: str = "both"
    ) -> pd.DataFrame:
        try:
            logger.info(
                f"Starting anomaly detection run with method={method}, "
                f"rows={len(df)}, numeric_cols={len(numeric_cols)}"
            )

            result = df.copy()

            if method in ("zscore", "both"):
                logger.info("Starting z-score based anomaly detection")
                for col in numeric_cols:
                    stats = baseline.get(col)
                    if stats and stats["count"] >= 30:
                        z_scores = self.zscore_flag(df[col], stats["mean"], stats["std"])
                        result[f"{col}_zscore"] = z_scores.round(4)
                        result[f"{col}_zscore_flag"] = z_scores > self.z_threshold
                        flagged_count = int((result[f"{col}_zscore_flag"] == True).sum())
                        logger.info(f"Z-score calculations complete for {col}; flagged {flagged_count} row(s)")
                    else:
                        result[f"{col}_zscore"] = None
                        result[f"{col}_zscore_flag"] = None
                        logger.info(f"Skipping z-score detection for {col}; insufficient baseline history")

            if method in ("isolation", "both"):
                logger.info("Starting IsolationForest based anomaly detection")
                labels, scores = self.isolation_forest_flag(df, numeric_cols)
                result["if_label"] = labels
                result["if_score"] = scores.round(4)
                result["if_flag"] = labels == -1

            if method == "both":
                logger.info("Combining z-score and IsolationForest results")
                zscore_flags = [
                    result[f"{col}_zscore_flag"]
                    for col in numeric_cols
                    if f"{col}_zscore_flag" in result.columns
                    and result[f"{col}_zscore_flag"].notna().any()
                ]
                if zscore_flags:
                    any_zscore = pd.concat(zscore_flags, axis=1).any(axis=1)
                    result["anomaly"] = any_zscore | result["if_flag"]
                else:
                    result["anomaly"] = result["if_flag"]

                final_anomaly_count = int((result["anomaly"] == True).sum())
                logger.info(f"Final consensus anomaly calculation complete; {final_anomaly_count} row(s) flagged")

            logger.info("Anomaly detection run completed successfully")
            return result

        except Exception as e:
            logger.exception(f"Error during anomaly detection run: {e}")
            return df.copy()