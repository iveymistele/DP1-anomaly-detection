#!/usr/bin/env python3
import json
import io
import boto3
import pandas as pd
import logging
import sys
from datetime import datetime

from baseline import BaselineManager
from detector import AnomalyDetector

logger = logging.getLogger("anomaly_app")
if not logger.handlers:
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler("app.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

s3 = boto3.client("s3")

NUMERIC_COLS = ["temperature", "humidity", "pressure", "wind_speed"]

def process_file(bucket: str, key: str):
    try:
        logger.info(f"Starting processing for s3://{bucket}/{key}")

        response = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(io.BytesIO(response["Body"].read()))
        logger.info(f"Loaded file {key} with {len(df)} rows and columns {list(df.columns)}")

        baseline_mgr = BaselineManager(bucket=bucket)
        baseline = baseline_mgr.load()

        for col in NUMERIC_COLS:
            if col in df.columns:
                clean_values = df[col].dropna().tolist()
                if clean_values:
                    logger.info(f"Updating baseline for {col} with {len(clean_values)} value(s)")
                    baseline = baseline_mgr.update(baseline, col, clean_values)
                else:
                    logger.info(f"No non-null values found for {col} in {key}")
            else:
                logger.info(f"Column {col} not found in {key}")

        logger.info("Starting anomaly calculations")
        detector = AnomalyDetector(z_threshold=3.0, contamination=0.05)
        scored_df = detector.run(df, NUMERIC_COLS, baseline, method="both")
        logger.info("Anomaly calculations completed")

        output_key = key.replace("raw/", "processed/")
        csv_buffer = io.StringIO()
        scored_df.to_csv(csv_buffer, index=False)
        s3.put_object(
            Bucket=bucket,
            Key=output_key,
            Body=csv_buffer.getvalue(),
            ContentType="text/csv"
        )
        logger.info(f"Processed CSV written to s3://{bucket}/{output_key}")

        baseline_mgr.save(baseline)
        logger.info("Baseline updated and saved")

        try:
            s3.upload_file("app.log", bucket, "logs/app.log")
            logger.info("Application log synced to s3://%s/logs/app.log", bucket)
        except Exception as e:
            logger.exception(f"Failed to sync app.log to S3: {e}")

        anomaly_count = int(scored_df["anomaly"].sum()) if "anomaly" in scored_df else 0
        summary = {
            "source_key": key,
            "output_key": output_key,
            "processed_at": datetime.utcnow().isoformat(),
            "total_rows": len(df),
            "anomaly_count": anomaly_count,
            "anomaly_rate": round(anomaly_count / len(df), 4) if len(df) > 0 else 0,
            "baseline_observation_counts": {
                col: baseline.get(col, {}).get("count", 0) for col in NUMERIC_COLS
            }
        }

        summary_key = output_key.replace(".csv", "_summary.json")
        s3.put_object(
            Bucket=bucket,
            Key=summary_key,
            Body=json.dumps(summary, indent=2),
            ContentType="application/json"
        )

        logger.info(f"Summary JSON written to s3://{bucket}/{summary_key}")
        logger.info(f"Done processing {key}: {anomaly_count}/{len(df)} anomalies flagged")

        return summary

    except Exception as e:
        logger.exception(f"Error processing file {key}: {e}")
        return {
            "status": "error",
            "source_key": key,
            "message": str(e),
            "processed_at": datetime.utcnow().isoformat()
        }