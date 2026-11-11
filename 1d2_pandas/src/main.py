
import pandas as pd
from pathlib import Path

from .utils import  (
    load_csv,
    preview,
    filter_rows,
    group_agg,
    merge_dfs,
    pivot_table,
    export,
)


# Get the project directory.
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Define raw and processed data folders.
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROC_DIR = PROJECT_DIR / "data" / "processed"

# Input dataset.
DATA_FILE = RAW_DIR / "crop_production.csv"


def main():

    # 1. LOAD DATASET
    print("\n========== LOAD DATASET ==========")

    df = load_csv(DATA_FILE)

    preview(df, n=10)

    # 2. DATA CLEANING
    print("\n========== DATA CLEANING ==========")

    # Remove unwanted spaces from text columns.
    text_columns = [
        "State_Name",
        "District_Name",
        "Season",
        "Crop",
    ]

    for column in text_columns:
        df[column] = df[column].str.strip()

    # Convert numeric columns safely.
    df["Area"] = pd.to_numeric(
        df["Area"], errors="coerce"
    )

    df["Production"] = pd.to_numeric(
        df["Production"], errors="coerce"
    )

    # Remove missing values and duplicate records.
    df = df.dropna(
        subset=[
            "State_Name",
            "District_Name",
            "Crop_Year",
            "Season",
            "Crop",
            "Area",
            "Production",
        ]
    ).drop_duplicates()

    # Keep records with positive area and non-negative production.
    df = df.loc[
        (df["Area"] > 0) &
        (df["Production"] >= 0)
    ].copy()

    print("\nCleaned dataset shape:")
    print(df.shape)

    print("\nCleaned data types:")
    print(df.dtypes)

    # 3. FILTER OPERATION
    print("\n========== FILTER ==========")

    # Select crop production records from Karnataka.
    karnataka_df = filter_rows(
        df,
        lambda data: data["State_Name"] == "Karnataka"
    )

    print("Karnataka dataset shape:")
    print(karnataka_df.shape)

    print(karnataka_df.head(10))

    # 4. GROUPBY OPERATION
    print("\n========== GROUPBY ==========")

    # Calculate total production by state and season.
    season_summary = group_agg(
        df,
        by=["State_Name", "Season"],
        agg_map={"Production": "sum"}
    )

    print("Season summary shape:")
    print(season_summary.shape)

    print(season_summary.head(10))

    # Calculate total production by state.
    state_summary = group_agg(
        df,
        by=["State_Name"],
        agg_map={"Production": "sum"}
    )

    state_summary = state_summary.rename(
        columns={
            "Production": "Total_State_Production"
        }
    )

    print("State summary shape:")
    print(state_summary.shape)

    # 5. MERGE OPERATION
    print("\n========== MERGE ==========")

    # Merge seasonal and state-level production summaries.
    merged_df = merge_dfs(
        season_summary,
        state_summary,
        on="State_Name",
        how="inner"
    )

    print("Merged dataset shape:")
    print(merged_df.shape)

    print(merged_df.head(10))

    print("\nMerged data types:")
    print(merged_df.dtypes)

    # 6. PIVOT TABLE OPERATION
    print("\n========== PIVOT TABLE ==========")

    # Compare crop production across different seasons.
    crop_pivot = pivot_table(
        df,
        index="State_Name",
        columns="Season",
        values="Production",
        aggfunc="sum"
    )

    print("Pivot table shape:")
    print(crop_pivot.shape)

    print(crop_pivot.head(10))

    # 7. EXPORT CSV AND PARQUET
    print("\n========== EXPORT ==========")

    sizes = export(
        df,
        PROC_DIR,
        "cleaned_crop_production"
    )

    print("CSV file size:", sizes["csv_bytes"], "bytes")
    print("Parquet file size:", sizes["parquet_bytes"], "bytes")

    # Compare the file sizes.
    if sizes["csv_bytes"] > sizes["parquet_bytes"]:
        print("Parquet file is smaller than CSV.")
    elif sizes["csv_bytes"] < sizes["parquet_bytes"]:
        print("CSV file is smaller than Parquet.")
    else:
        print("Both files have the same size.")

    print("\n========== WORKFLOW COMPLETED ==========")


if __name__ == "__main__":
    main()
