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


## 👤 Author

    Ahmad - Student of Machine learning




