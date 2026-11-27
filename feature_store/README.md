# Feature Store (Feast)

Feast sits between feature engineering and training/serving, so both use the **same** feature values.

```
DVC preprocess stage ──► data/processed/store_item_features.parquet   (OFFLINE store: full history)
                                   │
                    python -m feature_store.materialize
                                   ▼
               feature_repo/data/online_store.db               (ONLINE store: latest value per store-item)
                     ▲                                  ▲
   training: get_historical_features()        API: get_online_features()
   (point-in-time correct)                    (millisecond lookup by {store, item})
```

## Files
| File | Purpose |
|---|---|
| `feature_repo/feature_store.yaml` | Feast config: project, registry, SQLite online store, file offline store |
| `feature_repo/entities.py` | Entities `store` and `item` (the lookup keys) |
| `feature_repo/features.py` | Feature fields (names, types, descriptions) and the Parquet data source |
| `feature_repo/feature_views.py` | Feature view `store_item_sales_stats` (entities + schema + source + TTL + owner) |
| `materialize.py` | apply → materialize → verify offline and online values (exit 1 on mismatch) |
| `client.py` | Helper functions for online (API) and offline (training) lookups |

## Feature definitions
**Entity key:** `store` (int64) + `item` (int64) · **Event timestamp:** `event_timestamp` (UTC, one row per day)

| Feature | Type | Transformation (per store-item, previous days only) | Used for |
|---|---|---|---|
| `sales_mean_7d` | float64 | mean of daily sales over the 7 days before the timestamp | model input |
| `sales_mean_28d` | float64 | mean of daily sales over the 28 days before the timestamp | model input |
| `sales_std_28d` | float64 | std. deviation of daily sales over the 28 days before | model input + safety stock |

- **Source:** Kaggle Store Item Demand `train.csv` → validated by Pandera → `build_feature_store_table()`
  in `src/features/preprocessing.py` (the same `add_history_features()` used for training).
- **Freshness:** recomputed every time the pipeline runs on new data; TTL = 30 days (older values are
  treated as missing). The online store holds values "as of" the day after the last sales day.
- **Owner:** Person A (data & ML). **Versioning:** definitions in Git (MLflow tag `feature_repo_version`),
  data in DVC (MLflow tag `data_version_md5`).
- **No sensitive data:** only aggregated sales numbers and numeric ids; no customer data, no secrets.

## Why local SQLite + Parquet?
This is a course-scale project on one laptop: a file-based offline store and a SQLite online store
need no servers and cost nothing. In production we would use S3/Redshift as the offline store and
Redis or DynamoDB as the online store - only `feature_store.yaml` would change, not the feature code.

## Run
```bash
python -m feature_store.materialize        # or: dvc repro (stage "feature_store")
feast -c feature_store/feature_repo feature-views list
feast -c feature_store/feature_repo entities list
```
