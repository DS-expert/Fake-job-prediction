# Fake Job Prediction History / Timeline

**Date : 2025-10-06**

- Add Raw data path and Base Dir in `config.py`
- Add `__init__.py` to each folder
- Add BASE_DIR in system path in `notebook_setup.py`
- Add `__pycache__.py` to `.gitignore.py`
- Explore data in `data_exploration.ipynb`

**Date: 2025-10-08**
- Complete `01_data_exploration.ipynb` notebook.
- Write a Summary of data exploration in `README.md`

**Date: 2025-10-11**

- Complete notebook of feature engineering.
- On baseline model, model got 98% accuracy and 92% ROC_AUC_CRUVE.

**Date: 2025-10-13**

```
- Add reusable code in `preprocessing.py`
    - load_data() - load the dataset into enviornment.
    - handle_missing_value() - handle missing values in enviornment.
    - text_cleaning() - clean text from noises and punctuation and normalize the text.
    - apply_text_cleaning() - This function implement `text_cleaning()` on text in dataset / dataframe.
    - combine_text_feature() - This function combine all rich text feature into one text feature.
    - detect_langugage() - detect the language of text and label it
    - apply_detect_language() - Apply `detect_language()` to text in dataset / dataframe.
    - remove_stopwords() - Remove stopwords from text and left all meaningful words.
    - lemmatization() - lemmatization convert word into the root words for example ("running" -> run).
    - train_test_split_fn() - Split the dataset into train and test.
    - encoding() - Encoding function convert text feature into vectorizer or OneHotEncoding which model can understand.
    - handle_imbalance_data() - handle imbalance data using `SMOTE` technique to over_sampling the train label.
```

- Add Train Hyperparameter in config.py