import pandas as pd


def detect_target_column(df: pd.DataFrame) -> str | None:
    """
    Automatically detect the most likely target column.

    Priority:
    1. Common target names
    2. Last numeric column
    """

    common_targets = [
        "price",
        "salary",
        "revenue",
        "sales",
        "target",
        "label",
        "class",
        "churn",
        "attrition",
        "outcome",
        "profit",
        "income",
        "amount",
    ]

    for column in df.columns:
        if column.lower() in common_targets:
            return column

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

    if numeric_columns:
        return numeric_columns[-1]

    return None


def detect_problem_type(df: pd.DataFrame, target: str) -> str:
    """
    Detect whether the problem is regression or classification.
    """

    if target is None:
        return "unknown"

    if pd.api.types.is_numeric_dtype(df[target]):

        unique_values = df[target].nunique()

        if unique_values <= 10:
            return "classification"

        return "regression"

    return "classification"