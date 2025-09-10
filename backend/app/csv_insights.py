import argparse
import csv
import pandas as pd
from collections import Counter, defaultdict

def small_file_insights(path):
    """Use pandas for small CSVs (<~100k rows) for full stats."""
    df = pd.read_csv(path)
    print(f"Rows: {len(df):,}")
    print("Columns:", list(df.columns))

    print("\nNumeric columns summary:")
    numeric_cols = df.select_dtypes(include=['number'])
    if not numeric_cols.empty:
        print(numeric_cols.describe())

    print("\nTop 5 values per categorical column:")
    for col in df.select_dtypes(include=['object']).columns:
        print(f"\n{col}:")
        print(df[col].value_counts().head(5))


def large_file_insights(path, top_n=5):
    """
    Stream a large CSV (>1M rows), compute numeric stats and top categorical values
    without loading entire file into memory.
    """

    numeric_cols = {"Index"} 
    categorical_cols = set() 

    numeric_sums = defaultdict(float)
    numeric_sumsq = defaultdict(float)
    numeric_counts = defaultdict(int)
    categorical_counts = defaultdict(Counter)

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        categorical_cols = set(fieldnames) - numeric_cols

    total_rows = 0
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            for col in fieldnames:
                val = row[col]
                if val is None or val.strip() == "":
                    continue
                if col in numeric_cols:
                    try:
                        x = float(val)
                        numeric_sums[col] += x
                        numeric_sumsq[col] += x * x
                        numeric_counts[col] += 1
                    except ValueError:
                        pass
                else:
                    categorical_counts[col][val] += 1

    print(f"Rows: {total_rows:,}")
    print("Columns:", fieldnames)

    if numeric_cols:
        print("\nNumeric column stats (mean, variance, count):")
        for col in numeric_cols:
            count = numeric_counts[col]
            if count == 0:
                continue
            mean = numeric_sums[col] / count
            var = (numeric_sumsq[col] / count) - mean ** 2
            print(f"{col}: mean={mean:.2f}, variance={var:.2f}, count={count}")

    print("\nTop categorical values per column:")
    for col in categorical_cols:
        print(f"\n{col}:")
        for val, cnt in categorical_counts[col].most_common(top_n):
            print(f"{val}: {cnt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", help="Path to small CSV (<~100k rows)")
    parser.add_argument("--large", help="Path to large CSV (>1M rows)")
    args = parser.parse_args()

    if args.small:
        small_file_insights(args.small)
    elif args.large:
        large_file_insights(args.large)
    else:
        print("Please provide either --small or --large with a CSV path")
