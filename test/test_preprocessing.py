import pandas as pd
from src.preprocessing import handle_missing_values


def handle_missing_value_numerical():

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

    assert any("missing_indicator" in col for col in result.columns), "Missing indicator not created for numerical features"



