"""
Village Aggregator
===================
Takes the raw farm-level CSV (many rows per village) and
produces one row per village by averaging all farms in that village.

This gives:
  - One clean row per village for the UI dropdown
  - Average N/P/K/pH/OC across all farms in the village
  - Count of farms averaged (for credibility)

USAGE:
  python village_aggregator.py
  python village_aggregator.py --input data/shc_village_soil.csv --output data/shc_village_avg.csv
"""

import pandas as pd
import argparse
from pathlib import Path


def aggregate(input_csv="data/shc_village_soil.csv",
              output_csv="data/shc_village_avg.csv"):

    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"  Raw records: {len(df)}")
    print(f"  Districts:   {df['district'].nunique()}")
    print(f"  Villages:    {df['village'].nunique()}")

    # Numeric columns to average
    num_cols = ["N", "P", "K", "pH", "OC", "B", "Fe", "Zn", "S"]

    # Group by village + district + state
    agg_dict = {}
    for col in num_cols:
        if col in df.columns:
            agg_dict[col] = "mean"

    # Keep these as-is (same for all farms in same village)
    for col in ["Temperature", "Humidity", "Rainfall_monthly",
                "cycle", "district_code", "state_code"]:
        if col in df.columns:
            agg_dict[col] = "first"

    # Count farms per village
    agg_dict["feature_id"] = "count"

    grouped = df.groupby(
        ["village", "district", "state"],
        as_index=False
    ).agg(agg_dict)

    # Rename feature_id count to farms_count
    grouped = grouped.rename(columns={"feature_id": "farms_averaged"})

    # Round numeric columns to 2 decimal places
    for col in num_cols:
        if col in grouped.columns:
            grouped[col] = grouped[col].round(2)

    # Sort by state, district, village
    grouped = grouped.sort_values(
        ["state", "district", "village"]
    ).reset_index(drop=True)

    # Save
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_csv, index=False)

    print(f"\nAggregated: {len(df)} farm records -> {len(grouped)} village records")
    print(f"Output: {output_csv}")
    print()

    # Show sample
    show_cols = ["village", "district", "state", "N", "P", "K",
                 "pH", "OC", "farms_averaged"]
    print(grouped[show_cols].head(20).to_string(index=False))

    # Summary stats
    print(f"\n--- Summary ---")
    print(f"Total villages: {len(grouped)}")
    print(f"Average farms per village: {grouped['farms_averaged'].mean():.1f}")
    print(f"Max farms in one village:  {grouped['farms_averaged'].max()}")
    print(f"Villages with 1 farm:      "
          f"{(grouped['farms_averaged'] == 1).sum()}")
    print(f"Villages with 5+ farms:    "
          f"{(grouped['farms_averaged'] >= 5).sum()}")

    return grouped


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default="data/shc_village_soil.csv")
    ap.add_argument("--output", default="data/shc_village_avg.csv")
    args = ap.parse_args()

    aggregate(args.input, args.output)
