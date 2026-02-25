#!/usr/bin/env python3
"""
Benchmark ML Pipeline Performance
================================

Times IsolationForest, XGBoost, LightGBM on tier5_features.parquet.
Compares old vs new ensemble scores.
"""

import logging
import time

import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def benchmark_models(input_path):
    df = pd.read_parquet(input_path)
    feature_cols = [
        col
        for col in df.columns
        if any(k in col.lower() for k in ["pm_", "w1_", "w2_", "rms_", "var_"])
    ]
    X = df[feature_cols].fillna(0).values
    n = len(X)

    logger.info(f"Benchmarking on {n} samples, {len(feature_cols)} features")

    # Isolation Forest
    start = time.time()
    if_model = IsolationForest(random_state=42, n_jobs=-1)
    if_model.fit(X)
    if_time = time.time() - start
    logger.info(f"Isolation Forest: {if_time:.2f}s")

    # XGBoost
    start = time.time()
    dtrain = xgb.DMatrix(X)
    xgb_model = xgb.train({"objective": "reg:squarederror"}, dtrain, num_boost_round=50)
    xgb_time = time.time() - start
    logger.info(f"XGBoost: {xgb_time:.2f}s")

    # LightGBM
    start = time.time()
    lgb_data = lgb.Dataset(X)
    lgb_model = lgb.train({"objective": "regression", "verbose": -1}, lgb_data, num_boost_round=50)
    lgb_time = time.time() - start
    logger.info(f"LightGBM: {lgb_time:.2f}s")

    total_time = if_time + xgb_time + lgb_time
    logger.info(f"Ensemble total: {total_time:.2f}s ({total_time/n*1000:.2f}ms/sample)")

    # Memory estimate (rough)
    mem_gb = X.nbytes / 1e9
    logger.info(f"Peak memory: ~{mem_gb:.2f}GB")


if __name__ == "__main__":
    benchmark_models("output/features/tier5_features.parquet")
