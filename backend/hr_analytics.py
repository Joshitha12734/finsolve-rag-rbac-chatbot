"""
HR analytics — richer than raw SQL rows: department-level aggregates,
distributions, and summary stats computed with pandas over hr_data.csv.

Gated to `hr` and `c-level` roles at the endpoint level (main.py); this
module itself just computes numbers from whatever CSV(s) it finds under
data/hr/, same separation of concerns as sql_agent.py and admin.py.
"""
from __future__ import annotations

import pandas as pd

from .config import DATA_DIR


def _load_hr_dataframe() -> pd.DataFrame | None:
    hr_dir = DATA_DIR / "hr"
    csv_files = list(hr_dir.glob("*.csv")) if hr_dir.exists() else []
    if not csv_files:
        return None
    return pd.read_csv(csv_files[0])


def get_hr_analytics() -> dict:
    df = _load_hr_dataframe()
    if df is None or df.empty:
        return {"available": False}

    result: dict = {"available": True, "total_employees": int(len(df))}

    if "department" in df.columns:
        result["headcount_by_department"] = df["department"].value_counts().to_dict()

    if "performance_rating" in df.columns:
        result["avg_performance_rating_overall"] = round(float(df["performance_rating"].mean()), 2)
        if "department" in df.columns:
            result["avg_performance_rating_by_department"] = (
                df.groupby("department")["performance_rating"].mean().round(2).to_dict()
            )
        result["performance_rating_distribution"] = df["performance_rating"].value_counts().sort_index().to_dict()

    if "salary" in df.columns:
        result["salary_stats"] = {
            "mean": round(float(df["salary"].mean()), 2),
            "median": round(float(df["salary"].median()), 2),
            "min": round(float(df["salary"].min()), 2),
            "max": round(float(df["salary"].max()), 2),
        }
        if "department" in df.columns:
            result["avg_salary_by_department"] = df.groupby("department")["salary"].mean().round(2).to_dict()

    if "attendance_pct" in df.columns:
        result["avg_attendance_pct"] = round(float(df["attendance_pct"].mean()), 2)
    elif "attendance" in df.columns:
        result["avg_attendance_pct"] = round(float(df["attendance"].mean()), 2)

    if "leaves_taken" in df.columns:
        result["avg_leaves_taken"] = round(float(df["leaves_taken"].mean()), 2)

    if "location" in df.columns:
        result["headcount_by_location"] = df["location"].value_counts().to_dict()

    return result
