import pandas as pd
from src.preprocessing import handle_missing_values
from src.preprocessing import text_cleaning
from src.preprocessing import combine_text_features
import re

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

def test_text_cleaning():

    # Arrange
    text = "  HELLO,   world!!  AI---is `AMAZING!!!'  AI---is AMAZING!!!`  "

    # Act
    
    result = text_cleaning(text)

    print(result)

    # Assertion

    # First text should be lower

    assert result == result.lower(), "Text didn't convert into lower case"

    # `' should be not in text.

    assert "`" not in result or "'" not in result, "`' Didn't remove correctly"

    # Have no punctuation in text

    punctuation_found = re.search(r'[^\w\s]', result)

    assert punctuation_found is None, f"Punctuation still present in text: {result}"

    # Have no extra whitespace
    assert not re.search(r'\s{2,}', result), f"Extra Whitespace found in text {result}"

    # Have no trailing or spacing leading 

    assert result == result.strip(), f"Leading or trading spaces remain in text {result}"

def test_combination_text():

    # Arrange 
    df_rich = pd.DataFrame({
    "description": [
        "We are looking for a data scientist with strong Python and SQL skills.",
        "The ideal candidate will have experience in machine learning and AI research.",
        "Join our AI lab and contribute to groundbreaking projects on computer vision."
    ],
    "requirements": [
        "3+ years experience with Python, Pandas, and TensorFlow.",
        "Ability to design scalable ML models and pipelines.",
        "Strong understanding of CNNs and data preprocessing."
    ]
    })

    # Act
    result = combine_text_features(df_rich)

    print(f"df_rich Shape: {df_rich.shape}")
    print(f"Result Shape: {result.shape}")

    # Assertion
    assert "text_all" in result, "'text_all' feature didn't made in process"



def test_combination_small_text():

    # Arrange 
    # Small text feature dataset
    df_small = pd.DataFrame({
    "department": ["IT", "HR", "Finance"],
    "location": ["New York", "San Francisco", "London"],
    "employment_type": ["Full-time", "Part-time", "Contract"]
    })

    # ACT

    result = combine_text_features(df_small)

    # Assertion

    assert not "text_all" in result, "combine rich text feature logic failed! combine_text_feature combine small features"

def test_combination_mix_features():

    # Arrange 
    # Mixed dataset (rich + small)
    df_mixed = pd.DataFrame({
    "description": [
        "We seek a backend developer proficient in Django and REST APIs.",
        "Frontend engineer needed for React-based UI development.",
        "Machine learning engineer to optimize predictive models."
    ],
    "requirements": [
        "Experience with Docker, AWS, and CI/CD pipelines.",
        "Strong knowledge of JavaScript and CSS.",
        "Expertise in Python, Scikit-learn, and data preprocessing."
    ],
    "department": ["Engineering", "Engineering", "AI Lab"],
    "location": ["Berlin", "Toronto", "Tokyo"]
    })

    # ACT
    result = combine_text_features(df_mixed)

    # ASSERTION

    # Check if text_all feature in

    assert "text_all" in result, "combine_text_feature() didn't work properly"

    # Check the shape it should be equal to 5 columns

    assert result.shape[1] == 5, "Shape is misplaced in the process"




