#!/usr/bin/env python3
import json
import math
import boto3
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger("anomaly_app")

s3 = boto3.client("s3")


class BaselineManager:
    """
    Maintains a per-channel running baseline using Welford's online algorithm,
    which computes mean and variance incrementally without storing all past data.
    """

    def __init__(self, bucket: str, baseline_key: str = "state/baseline.json"):
        self.bucket = bucket
        self.baseline_key = baseline_key

    def load(self) -> dict:
        try:
            logger.info("Loading baseline from S3")
            response = s3.get_object(Bucket=self.bucket, Key=self.baseline_key)
            baseline = json.loads(response["Body"].read())
            logger.info("Baseline loaded successfully")
            return baseline
        except s3.exceptions.NoSuchKey:
            logger.info("Baseline does not exist yet, starting fresh")
            return {}
        except Exception as e:
            logger.exception(f"Failed to load baseline: {e}")
            return {}

    def save(self, baseline: dict):
        try:
            logger.info("Saving baseline to S3")
            baseline["last_updated"] = datetime.utcnow().isoformat()

            s3.put_object(
                Bucket=self.bucket,
                Key=self.baseline_key,
                Body=json.dumps(baseline, indent=2),
                ContentType="application/json"
            )

            logger.info("Baseline successfully updated in S3")

        except Exception as e:
            logger.exception(f"Failed to save baseline: {e}")

    def update(self, baseline: dict, channel: str, new_values: list[float]) -> dict:
        """
        Welford's online algorithm for numerically stable mean and variance.
        Each channel tracks: count, mean, M2 (sum of squared deviations).
        Variance = M2 / count, std = sqrt(variance).
        """
        try:
            logger.info(f"Updating baseline for channel {channel} with {len(new_values)} new values")

            if channel not in baseline:
                baseline[channel] = {"count": 0, "mean": 0.0, "M2": 0.0}

            state = baseline[channel]

            for value in new_values:
                state["count"] += 1
                delta = value - state["mean"]
                state["mean"] += delta / state["count"]
                delta2 = value - state["mean"]
                state["M2"] += delta * delta2

            if state["count"] >= 2:
                variance = state["M2"] / state["count"]
                state["std"] = math.sqrt(variance)
            else:
                state["std"] = 0.0

            baseline[channel] = state

            logger.info(
                f"Baseline updated for channel {channel}: "
                f"count={state['count']} mean={state['mean']:.4f} std={state['std']:.4f}"
            )

            return baseline

        except Exception as e:
            logger.exception(f"Error updating baseline for channel {channel}: {e}")
            return baseline

    def get_stats(self, baseline: dict, channel: str) -> Optional[dict]:
        return baseline.get(channel)