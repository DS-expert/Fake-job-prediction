# Fake Job Prediction 


This project is a **Machine Learning pipeline** designed to predict fake jobs based on survey/text data.  
It is built from scratch without relying only on Jupyter notebooks, focusing instead on a **modular, production-ready structure**.

---

## 🚀 Features
- End-to-end ML pipeline (data preprocessing → model training → evaluation).
- Modular codebase (`preprocessing.py`,  `evaluation.py`, `train.py`).
- Uses **spaCy** for NLP preprocessing.
- Supports multiple ML models (Linear Regression, Decision Trees, Ridge, etc.).
- Logging & metrics tracking for debugging and reproducibility.
- Easy to extend with new datasets or algorithms.

---

## 🗂 Project Structure
```
Fake_job_prediction/
│
├── data/ # Raw or cleaned datasets
├── notebooks/ # Experimentation notebooks (exploration, EDA)
├── src/ # Core source code
│ ├── preprocessing.py # Text & numerical preprocessing
│ ├── evaluation.py # Metrics & evaluation functions
│ ├── train.py # Entry point to run pipeline
│
├── requirements.txt # Project dependencies
├── README.md # Project documentation
```

---

## ⚙️ Installation

1. Clone the repository:

    ```
    git clone https://github.com/DS-expert/Fake-job-prediction.git'
    cd Fake-job-prediction'
    ```

2. Create Virtual Environment

    ```
    python3 -m venv venv
    source venv/bin/activate   # on Linux/Mac
    venv\Scripts\activate      # on Windows
    ```
3. Upgrade code tools

    ```
    pip install -U pip setuptools wheel
    ```

4. Install dependecies

    ```
    pip install -r requirements.txt
    ```

5. Download spacy language model

    ```
    python -m spacy download en_core_web_sm
    ```

## ▶️ Usage

1. Place your dataset in the data/ directory.

2. Run exploratory analysis in notebooks/.

3. Train the pipeline:
    ```
    python src/train.py
    ```


4. Results (metrics, logs, trained model) will be saved in `outputs/`.


### 🧠 Summary of Data Exploration `notebook/01_data_exploration`

In this notebook, we performed an initial overview and exploratory data analysis (EDA) for the Fake Job Prediction dataset.

#### 🧩 EDA in NLP — Basic Flow

1. Check Duplicates and Missing Values
    - Identify incomplete or repeated entries.

2. Analyze Class Distribution
    - Observe label imbalance between real and fake job postings.

3. Perform Basic Text Statistics

    - Word count

    - Sentence count

    - Average word and sentence length

    - Visualize text distributions.

4. Term Frequency Analysis (TF Analysis)
    - Identify the most common words using NLTK and frequency plots.

5. Curiosity-Based Insights
    - Compare common words in fake vs real job descriptions to detect patterns.

#### Key Observations

- The dataset is imbalanced — fake job postings are fewer than real ones.

- The ' character appears frequently and needs cleaning during preprocessing.

- Several features contain missing values.

- Descriptions contain many stopwords, which will need filtering later.

- There are spelling errors and punctuations that should be cleaned.

- Common words in fake job posts:
    ``` time, solutions, within, position, engineering, skills, project ```

- Common words in real job posts:
    ``` sales, client, marketing, design, product, development ```

## 👤 Author

    Ahmad - Student of Machine learning




