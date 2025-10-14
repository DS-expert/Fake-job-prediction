import pandas as pd
from src.preprocessing import handle_missing_values


def test_handle_missing_value_numerical():

    df = pd.DataFrame({
        "age": [25, None, 35, 40, None],
        "salary": [1000, 2000, None, 4000, 5000]
    })

    result = handle_missing_values(df)

    # Assert  no missing values remains

    assert result.isnull().sum().sum() == 0, "There are missing values"

    # Shape should be same or smaller (if dropna removed some rows)

    assert set(result.columns).issuperset({"age", "salary"}), "Missing Original columns"

    # Check missing indicator exits

    indicator = [col for col in result.columns if "missing_indicator" in col]
    if indicator:

        for col in indicator:

            valid_values = result[col].isin([0, 1]).all()
            assert valid_values, f"{col} have more values than 0 and 1."




