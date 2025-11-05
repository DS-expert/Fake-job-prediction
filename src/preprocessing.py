import pandas as pd
import numpy as np
import re
import string
from langdetect import detect, DetectorFactory
from nltk.corpus import stopwords
from tqdm import tqdm
import spacy
from config.config import RANDOM_STATE, TEST_SIZE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from scipy.sparse import coo_matrix
from imblearn.over_sampling import SMOTE

nlp = spacy.load("en_core_web_sm")


# First load the dataset

def load_data(data_path):

    """
    load the dataset into the environment 
    and return the dataframe
    *data_path: Path of dataset

    return: load the dataframe into environment
    """

    data = pd.read_csv(data_path)

    return data


# Handle the Missing values

def handle_missing_values(data):

    """
    Handle the Missing values by using dropna and filling the missing values and return cleaned dataset
    *df: Dataset to be cleaned

    return: cleaned dataset without missing values
    """

    # First DropNa low percentage of null values 


    feature_dropout = [feature for feature in data.columns if data[feature].isnull().mean() < 0.5 and data[feature].isnull().mean() > 0]

    cleaned_data = data.dropna(subset=feature_dropout)

    missing_features = [feature for feature in cleaned_data.columns if cleaned_data[feature].isnull().mean() > 0]

    for feature in missing_features:

        if cleaned_data[feature].dtype == "object":

            # First create missing indicator for those features
            cleaned_data[f"{feature}_missing_indicator"] = cleaned_data[feature].isnull().astype(int)

            # Then handle the missing values by "missing" term

            cleaned_data[feature] = cleaned_data[feature].fillna("missing")
        
        else:
            
            # Create missing indicator for numerical data also
            cleaned_data[f"{feature}_missing_indicator"] = cleaned_data[feature].isnull().astype(int)

            cleaned_data[feature] = cleaned_data[feature].fillna(cleaned_data[feature].mean())
    
    return cleaned_data


def text_cleaning(text):

    """
    text_cleaning clean the all text in dataframe and return clean text
    *text: text in the dataframe row

    return: cleaned text optimized for model
    """

    # First lower the text

    cleaned_text = text.lower()

    # Remove ' from the text according to the EDA

    cleaned_text = re.sub(r"['`]", "", cleaned_text)

    # Remove the punctuaction

    cleaned_text = re.sub(f"[{re.escape(string.punctuation)}]", " ", cleaned_text)

    # Remove extra spaces

    cleaned_text = " ".join(cleaned_text.split())

    return cleaned_text

def apply_text_cleaning(data):

    cat_features = [feature for feature in data.columns if data[feature].dtypes not in [int, float]]
    
    for feature in cat_features:

        data[f"{feature}_cleaned"] = data[feature].apply(lambda x: text_cleaning(x))

    # Drop the uncleaned data 
    data = data.drop(cat_features)

    return data

def combine_text_features(data, threshold_words=5, unique_ratio=0.3):

    """
    Automatically combine text features columns into one rich text feature

    *data: pandas:DataFrame - Input dataset
    *threshold_word: int, optional - Minimum average word count to consider columns as rich text feature
    *unique_ratio: float, optional - Minimum unique ration to consider columns as rich text

    return: df: pandas.DataFrame - Dataframe with new 'text-all' column containing rich text features
    """

    rich_text_feature = []

    for feature in data.columns:

        if data[feature].dtype == "object":

            # Find average word length 
            avg_word_len = data[feature].dropna().apply(lambda x: len(str(x).split())).mean()

            # Find uniquness of feature (Unique Value / total value)
            unique_ratio_col = data[feature].nunique() / len(data)

            if avg_word_len > threshold_words and unique_ratio_col > unique_ratio:

                rich_text_feature.append(feature)
    

    if rich_text_feature:

        data["text_all"] = data[rich_text_feature].fillna("").agg(" ".join, axis=1)

    return data


def detect_language(text):

    """
    detect the language and returned the language count
    *text: string, optional - input text to detect language

    return: returned the language of text
    """

    DetectorFactory.seed = 0

    try:
        return detect(text)
    except:
        return "unknown"

def apply_detect_language(df, feature="text_all", sample=1000):

    """
    Apply detect language function on dataset to detect the language of text
    *df: pandas.DataFrame - Input dataset
    *feature: Input feature in dataframe to detect the language on.
    *sample: int, optional - Sample of dataset to check the language result faster

    return: returned count of language in text or dataset
    """

    df["temp"] = df[feature].sample(n=sample).apply(lambda x: detect_language(x))

    lang_count = df["temp"].value_counts(normalize=True) * 100

    print(f"language count: {lang_count}")

    df = df.drop("temp", axis=1)

def remove_stopwords(data, feature="text_all"):

    """
    Remove stopwords from rich text feature and return cleaned_data
    *data: pd.DataFrame - Input dataset
    *feature = "text_all": remove stopwords from feature which have default value is "text_all"

    return: pd.DataFrame
        return the cleaned dataframe 
    """

    stop_words = set(stopwords.words("english"))
    data[feature] = data[feature].apply(lambda x: " ".join(word for word in x.split() if word not in stop_words))

    return data


def lemmatization(df, feature="text_all", batch_size=100):

    """
    Lemmatize the word to their root word

    *df : pd.Dataframe
        Input dataset
    *feature: 
        Input feature to implement lemmatization technique
    *batch_size: int, optional
        batch size is size into nlp.pipe() to batch the process of lemmatization
    
    return: pd.Dataframe
        return the dataframe with lemmatization feature
    """

    text = df[feature].tolist()

    lemmatized_text = []

    for doc in tqdm(nlp.pipe(text, batch_size=batch_size, disable=["parser", "ner"]), total=len(text)):

        lemmas = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
        lemmatized_text.append(" ".join(lemmas))

    df[f"{feature}_lemma"] = lemmatized_text

    # Drop the feature which was before lemmatized

    df = df.drop(feature, axis=1)

    return df

def train_test_split_fn(input_feature, target_feature, random_state=RANDOM_STATE, test_size=TEST_SIZE):

    """
    Train Test split function split the dataset into train and test dataset

    *input_feature: 
        Input feature which will be in X_train and X_test
    *target_feature:
        Target feature in which will store y_train and y_test
    
    *random_state: int, optional
        Random state for randomness in the train and test data
    
    *test_size:
        Test size is used for how much data will be in test data

    return: 
        Return train and test data
    """

    X_train, X_test, y_train, y_test = train_test_split(input_feature, target_feature, random_state=random_state, test_size=test_size, stratify=target_feature)

    return X_train, X_test, y_train, y_test



def encoding(data, X_train, X_test, threshold_word=5, unique_ratio=0.3):

    """
    Encoding the Categorical features and text features to Vectorizer and One Hot Encoding 
    *data: pd.Dataframe
        Input dataset
    
    *X_train: 
        Input train dataset
    *X_test:
        Input test dataset
    *threshold_word: int, optional
        Minimum average word length to consider as rich text feature
    *unique_ratio: float, optional
        Minimum uniqueness to consider as rich text feature

    return: sparse matrix of vectorizer
    """
    # First extract rich text feature
    text_feature = []

    for feature in data.columns:

        if data[feature].dtype == "object":

            avg_word_len = data[feature].apply(lambda x: len(x.split())).mean()
            uniqueness = data[feature].nunique() / len(data)
        
            if avg_word_len > threshold_word and uniqueness > unique_ratio:
                text_feature.append(feature)
    

    # Now Apply OneHotEncoding on Cat Features

    cat_features = [feature for feature in data.columns if data[feature].dtypes not in [int, float]]
    num_features = [feature for feature in data.columns if data[feature].dtypes in [int, float]]

    # Remove the rich text features from other ones 

    for feature in text_feature:

        if feature in cat_features:

            cat_features.remove(feature)

    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')

    X_train_encoded = encoder.fit_transform(X_train[cat_features])
    X_test_encoded = encoder.transform(X_test[cat_features])

    # Text Feature encoding 

    tfid = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

    X_train_text = None
    X_test_text = None

    for feature in text_feature:
        train_current = tfid.fit_transform(X_train[feature])
        test_current = tfid.transform(X_test[feature])

        if X_train_text is None:
            X_train_text = train_current
            X_test_text = test_current
        else:
            X_train_text = hstack([X_train_text, train_current])
            X_test_text = hstack([X_test_text, test_current])

    X_train_num = X_train[num_features].to_numpy()
    X_test_num = X_test[num_features].to_numpy()

    X_train_num_sparse = coo_matrix(X_train_num)
    X_test_num_sparse = coo_matrix(X_test_num)

    X_train_final  = hstack([X_train_num_sparse, X_train_encoded, X_train_text])
    X_test_final = hstack([X_test_num_sparse, X_test_encoded, X_test_text])


    return X_train_final, X_test_final

def handle_imbalance_data(X_train, y_train):

    """
    Handle imbalanced data and resampled the train labels 
    
    *X_train:
        Input train dataset
    *y_train:
        Input train label dataset
    
    return:
        Return resampled train data and label
    """

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    return X_train_resampled, y_train_resampled

def save_preprocess_data(data, output_path):

    """
    Save preprocessed data to the output path

    *data: pd.Dataframe:
        Input dataset
    *output_path: path
        Input path to store data.
    """

    data.to_csv(output_path, index=False)
    print(f"✅ Processed data saved at: {output_path}")







        
    




    

    









































