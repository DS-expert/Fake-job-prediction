from src.preprocessing import load_data, handle_missing_values, text_cleaning, apply_text_cleaning, remove_stopwords, lemmatization, train_test_split_fn, encoding, handle_imbalance_data, save_preprocess_data, combine_text_features
from config.config import RAW_DATA_PATH, PROCESSED_DATA_DIR

def main():
    # Load data
    df = load_data(RAW_DATA_PATH)

    # Handle missing values
    df_missing_values = handle_missing_values(df)

    # Text cleaning
    df_cleaned_text = apply_text_cleaning(df_missing_values)

    # Combinne text features into one
    df_combine_text = combine_text_features(df_cleaned_text)

    # Remove stopwords
    df_no_stopwords = remove_stopwords(df_combine_text)

    # Lemmatization

    df_lemmatized = lemmatization(df_no_stopwords)

    # Train test split
    input_features = df_lemmatized.drop("fraudulent", axis=1)
    target_features = df_lemmatized["fraudulent"]
    X_train, X_test, y_train, y_test = train_test_split_fn(input_features, target_features)

    # Encoding
    X_train_encoded, X_test_encoded = encoding(df_lemmatized, X_train, X_test)

    # Handle imbalance data
    X_train_balanced, y_train_balanced = handle_imbalance_data(X_train_encoded, y_train)

    # Save data
    save_preprocess_data(X_train_balanced, y_train_balanced, filename="train", output_path=PROCESSED_DATA_DIR)
    save_preprocess_data(X_test_encoded, y_test, filename="test", output_path=PROCESSED_DATA_DIR)

if __name__ == "__main__":
    main()