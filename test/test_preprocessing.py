import pandas as pd
from src.preprocessing import handle_missing_values
from src.preprocessing import text_cleaning
from src.preprocessing import combine_text_features
from src.preprocessing import remove_stopwords
from src.preprocessing import lemmatization
from src.preprocessing import encoding
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix, csr_matrix
from src.preprocessing import handle_imbalance_data
from src.preprocessing import save_preprocess_data
from config.config import PROCESSED_DATA_PATH
from nltk.corpus import stopwords
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

def test_remove_stopwords():

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
     
    # ACT

    result = remove_stopwords(df_rich, feature="description")

    # Assertion

    # No Stop word in description function

    stop_words = stopwords.words("english")

    for row in result["description"]:

        for word in row.split():

            assert word not in stop_words, "Stop words still remain in text!"


def test_lemmatization():

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

    # ACT 

    result = lemmatization(df_rich, feature="description")

    # Assertion

    assert "description_lemma" in result.columns, "Description_lemma columns not found in final result"

    assert not "description" in result.columns, "old 'description' Column didn't drop from dataset."

    for original, lemma in zip(df_rich["description"], result["description_lemma"]):
        org_len = len(original.split())
        lemma_len = len(lemma.split())

        assert lemma_len > 0, "Empty lemmatized text found"

        assert abs(org_len - lemma_len) < org_len * 0.5, f"Too much difference in Token Counts Original({org_len}) Vs lemma({lemma_len})"

    
    change_detected = any(
        o_word != l_word
        for original, lemma in zip(df_rich["description"], result["description_lemma"])
        for o_word, l_word in zip(original.lower().split(), lemma.lower().split())
        if len(o_word) > 3
    )

    assert change_detected, "No Lemmatization affect detected - words unchanged"


def test_encoding():

    # Arrange the dataset
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
    "location": ["Berlin", "Toronto", "Tokyo"],

    })

    X_train, X_test = train_test_split(df_mixed, test_size=0.2)

    # Act
    X_train_result, X_test_result = encoding(df_mixed, X_train, X_test)

    # Assertion

    # Test the type

    assert isinstance(X_train_result, coo_matrix), "Encoded result is not sparse matrix"
    assert isinstance(X_test_result, coo_matrix), "Encoded result is not sparse matrix"

    # Test the type of data inside

    assert X_train_result.dtype == "float64", "Encoded matrix data type is not float64"
    assert X_test_result.dtype == "float64", "Encoded matrix data type is not float64"

def test_imbalance_data():

    # Arrange
    data = {
    "Amount": [15, 25, 50, 12, 30, 19, 36],
    "Is_Fraud": [0, 0, 0, 1, 0, 0, 1] # 0 = No Fraud (Majority), 1 = Fraud (Minority)
    }

    df_imbalanced = pd.DataFrame(data)
    
    # Act 
    X_train =  df_imbalanced.drop("Is_Fraud", axis=1)
    y_train = df_imbalanced["Is_Fraud"]

    X_train_result, y_train_result = handle_imbalance_data(X_train, y_train)

    # Assertion

    # Check the Shape of X_train and y_train

    assert X_train_result.shape[0] == 10, "X_train after SMOTE has incorrect number of samples"
    assert y_train_result.shape[0] == 10, "y_train after SMOTE has incorrect number of samples"

    assert X_train_result.shape[0] == y_train_result.shape[0], "X_train and y_train sample counts do not match after SMOTE"

def test_save_data():

    # Arrange
    data = {
    "Amount": [15, 25, 50, 12, 30, 19, 36],
    "Is_Fraud": [0, 0, 0, 1, 0, 0, 1] # 0 = No Fraud (Majority), 1 = Fraud (Minority)
    }

    df= pd.DataFrame(data)

    # Act

    save_preprocess_data(df, output_path=PROCESSED_DATA_PATH)

    # Assertion

    assert PROCESSED_DATA_PATH.exists(), f"File not found at the expected location {PROCESSED_DATA_PATH}"


