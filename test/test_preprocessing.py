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


def test_handle_missing_value_categorical():

    df = pd.DataFrame({
        "name": ["Ali", None, "Sara", "Ahmad"],
        "city": ["Lahore", "Karachi", None, None]
    })

    result = handle_missing_values(df)

    assert result.isnull().sum().sum() == 0, "Categorical missing values didn't handle"

    assert "missing" in result["city"].values or "missing" in result["name"].values, "Missing term didn't fill correctly"
    
    indicator = [feature for feature in result.columns if "missing_indicator" in feature]

    if indicator:

        for col in indicator:

            valid_values = result[col].isin([0, 1]).all()

            assert valid_values, "f{col} have more values than 0 and 1"


def test_handle_missing_value_both():

    # For test it is simple flow. 
    # Arrange the setup for test.
    # Action
    # Assertion

    # Arrange the setup

    data = {
        "age": [25, None, 35, 40, None],                
        "salary": [1000, 2000, None, 4000, 5000],         
        "gender": ["Male", "Female", None, "Female", None],  
        "city": ["Lahore", "Karachi", "Lahore", None, None], 
        "experience": [1, 3, 5, None, 7]                  
    }

    df = pd.DataFrame(data)

    # Act 

    result = handle_missing_values(df)

    # Assertion

    # First there should be no missing values remain

    assert result.isnull().sum().sum() == 0, "Missing values remain... Imputation didn't execute properly"


    # There should be "missing" placeholder in categorical 

    if result.shape[0] < df.shape[0]:
        assert True
    else:

        assert "missing" in result["gender"].values or "missing" in result["city"].values, "Placeholder didn't place properly"

    # There should be missing indicator columns

    indicator = [feature for feature in result.columns if "missing_indicator" in feature]

    if indicator:
        for col in indicator:
            valid_values = result[col].isin([0, 1]).all()

            assert valid_values, "Missing indicator columns didn't add to the dataset."
