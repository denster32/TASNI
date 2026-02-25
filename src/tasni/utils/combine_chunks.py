from pathlib import Path

import duckdb

OUTPUT_DIR = Path(__file__).resolve().parents[3] / "output"
chunks = sorted(OUTPUT_DIR.glob("orphans_chunk_*.parquet"))
print("Found", len(chunks), "chunks")

con = duckdb.connect(":memory:")
con.execute("PRAGMA memory_limit='15GB'")

chunk_list = ",".join(["'" + str(f) + "'" for f in chunks])
print("Combining with DuckDB...")

out_path = str(OUTPUT_DIR) + "/wise_no_gaia_match.parquet"
sql = (
    "COPY (SELECT * FROM read_parquet(["
    + chunk_list
    + "])) TO '"
    + out_path
    + "' (FORMAT PARQUET, COMPRESSION SNAPPY)"
)
con.execute(sql)

print("Done")
con.close()
