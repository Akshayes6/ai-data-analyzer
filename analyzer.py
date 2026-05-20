import pandas as pd
import numpy as np
import pandas.api.types as pat   


def get_basic_info(df):
    """Basic facts: rows, columns, types, memory."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "column_types": df.dtypes.astype(str).to_dict(),
        "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 2),
    }


def get_missing_values(df):
    """Find columns with missing/empty values."""
    missing_counts = df.isnull().sum()
    result = []
    for col in df.columns:
        if missing_counts[col] > 0:
            result.append({
                "column": col,
                "missing_count": int(missing_counts[col]),
                "missing_percent": round(missing_counts[col] / len(df) * 100, 2),
            })
    return result


def get_numeric_stats(df):
    """Mean, median, std, min, max, outlier count for each numeric column."""
    stats = {}
    for col in df.columns:
        if not pat.is_numeric_dtype(df[col]):
            continue
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        outliers = int(((col_data < q1 - 1.5*iqr) | (col_data > q3 + 1.5*iqr)).sum())
        stats[col] = {
            "mean":   round(float(col_data.mean()), 4),
            "median": round(float(col_data.median()), 4),
            "std":    round(float(col_data.std()), 4),
            "min":    round(float(col_data.min()), 4),
            "max":    round(float(col_data.max()), 4),
            "q1":     round(float(q1), 4),
            "q3":     round(float(q3), 4),
            "outlier_count": outliers,
        }
    return stats


def get_text_stats(df):
    """Unique value counts and top values for text/category columns."""
    result = {}
    for col in df.columns:
        if pat.is_numeric_dtype(df[col]):
            continue
        value_counts = df[col].value_counts()
        result[col] = {
            "unique_count": int(df[col].nunique()),
            "top_5": value_counts.head(5).to_dict(),
        }
    return result


def get_correlations(df):
    """Correlation matrix and strongest pairs for numeric columns."""
    numeric_df = df[[c for c in df.columns if pat.is_numeric_dtype(df[c])]]
    if len(numeric_df.columns) < 2:
        return {}
    corr = numeric_df.corr().round(3)
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            val = corr.iloc[i, j]
            if abs(val) > 0.5:
                pairs.append({
                    "col1": cols[i], "col2": cols[j],
                    "correlation": float(val),
                    "relationship": "positive" if val > 0 else "negative",
                })
    pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return {"matrix": corr.to_dict(), "strong_pairs": pairs[:10]}


def get_auto_insights(df):
    """Simple rule-based insights about the data."""
    insights = []
    total_missing = df.isnull().sum().sum()
    if total_missing == 0:
        insights.append("No missing values — the dataset is complete.")
    else:
        pct = round(total_missing / df.size * 100, 1)
        insights.append(f"{total_missing:,} missing values ({pct}% of all cells).")

    numeric_cols = [c for c in df.columns if pat.is_numeric_dtype(df[c])]
    for col in numeric_cols[:3]:
        col_data = df[col].dropna()
        if col_data.mean() != 0:
            cv = col_data.std() / col_data.mean()
            if abs(cv) > 1:
                insights.append(f"'{col}' has high variability — values are very spread out.")

    if len(numeric_cols) >= 2:
        numeric_df = df[numeric_cols]
        corr = numeric_df.corr()
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                val = corr.iloc[i, j]
                if abs(val) > 0.8:
                    d = "positively" if val > 0 else "negatively"
                    insights.append(f"'{corr.columns[i]}' and '{corr.columns[j]}' are strongly {d} correlated ({val:.2f}).")
    return insights


def get_full_summary(df):
    return {
        "basic_info":    get_basic_info(df),
        "missing":       get_missing_values(df),
        "numeric_stats": get_numeric_stats(df),
        "text_stats":    get_text_stats(df),
        "correlations":  get_correlations(df),
        "auto_insights": get_auto_insights(df),
    }


def get_llm_summary(df):
    """Compact text summary to send to the AI for insights."""
    info = get_basic_info(df)
    stats = get_numeric_stats(df)
    text = get_text_stats(df)
    missing = get_missing_values(df)

    lines = [
        f"Dataset: {info['rows']} rows × {info['columns']} columns",
        f"Columns: {', '.join(info['column_names'])}",
        "",
    ]
    if stats:
        lines.append("Numeric columns:")
        for col, s in list(stats.items())[:8]:
            lines.append(f"  {col}: mean={s['mean']}, min={s['min']}, max={s['max']}, outliers={s['outlier_count']}")
    if text:
        lines.append("Text columns:")
        for col, t in list(text.items())[:5]:
            top = list(t["top_5"].keys())[:3]
            lines.append(f"  {col}: {t['unique_count']} unique, top: {', '.join(str(x) for x in top)}")
    if missing:
        lines.append(f"Missing: {len(missing)} column(s) have gaps")
    return "\n".join(lines)
