"""
Use Gaia Archive's pre-computed crossmatch tables.
"""

from pathlib import Path

from astroquery.gaia import Gaia

OUTPUT_DIR = Path(__file__).resolve().parents[3] / "data" / "gaia_xmatch"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def query_available_tables():
    """Check what crossmatch tables are available"""
    tables = Gaia.load_tables(only_names=True)
    xmatch_tables = [
        str(t) for t in tables if "wise" in str(t).lower() or "allwise" in str(t).lower()
    ]
    print("Available crossmatch tables:")
    for t in xmatch_tables:
        print(f"  - {t}")
    return xmatch_tables


def download_xmatch_sample():
    """Download sample of crossmatch table"""
    print("Querying Gaia archive for crossmatch tables...")
    tables = query_available_tables()

    # Use the first available table
    if tables:
        table = tables[0]
    else:
        print("No WISE x Gaia table found.")
        return

    print(f"Using table: {table}")

    # Get sample
    query = f'SELECT TOP 10000 * FROM "{table}"'
    print(f"Query: {query}")

    job = Gaia.launch_job(query)
    results = job.get_results()

    df = results.to_pandas()
    print(f"Columns: {list(df.columns)}")
    print(df.head())

    output_file = OUTPUT_DIR / "xmatch_sample.parquet"
    df.to_parquet(output_file)
    print(f"Saved {len(df)} rows to {output_file}")

    return df


if __name__ == "__main__":
    download_xmatch_sample()
