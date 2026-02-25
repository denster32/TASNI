# Archived Scripts

These scripts were used during development/experimentation but are not
needed for the production pipeline.

## Why Archived

| File | Reason |
|------|--------|
| filter_anomalies.py | Superseded by filter_anomalies_full.py |
| filter_anomalies_duckdb.py | DuckDB experiment, not production |
| filter_anomalies_polars.py | Polars experiment, not production |
| crossmatch.py | Old version, superseded |
| crossmatch_2mass_api.py | Specialized 2MASS crossmatch |
| crossmatch_nvss.py | Specialized NVSS crossmatch |
| crossmatch_rosat.py | Specialized ROSAT crossmatch |
| download_wise.py | Old download script |
| download_gaia.py | Old download script |
| download_gaia_parallel.py | Parallel experiment |
| download_*.py | Various download experiments |
| process_cdn_xmatch*.py | CDN processing experiments |
| cluster_images.py | CPU version, superseded by XPU |

## To Restore

git checkout HEAD@{1} -- legacy/<filename>

---

*The forest was never dark.*
