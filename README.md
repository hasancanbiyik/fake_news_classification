# Fake News Classification using spaCy Word Vectors

This project demonstrates how to build a robust machine learning model to classify news articles as either **Real** or **Fake**. It leverages spaCy's powerful word embeddings to convert text into meaningful numerical vectors, which are then used to train and evaluate several classification models. The K-Nearest Neighbors (KNN) model emerged as the top performer, achieving an impressive **98% accuracy**.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Installation and Usage](#installation-and-usage)
- [Technologies Used](#technologies-used)

## Project Overview

The goal of this project is to create an effective and reliable fake news detector. The notebook `fake_news_classification.ipynb` provides a step-by-step guide through the entire data science workflow, from initial data exploration to final model comparison. This project highlights the power of semantic vector representations for NLP tasks.

## Dataset

The dataset used in this project is `fake_news.csv`, which contains thousands of news articles. Each article is labeled as either "Real" or "Fake". The dataset is well-balanced, providing an excellent foundation for training an unbiased classifier.

## Methodology

The project follows a structured machine learning pipeline:

1.  **Data Exploration:** The dataset was loaded and analyzed, confirming a balanced class distribution.
2.  **Feature Engineering with spaCy:**
    - The core of this project involves using spaCy's `en_core_web_sm` model to convert the text of each article into a 96-dimension word vector.
    - This process, known as semantic embedding, captures the underlying meaning of the text, providing rich features for our models.
3.  **Model Training:** The vectorized data was split into training and testing sets.
4.  **Model Comparison:** Four different classification algorithms were trained and evaluated to identify the most effective one:
    - K-Nearest Neighbors (KNN)
    - Logistic Regression
    - Support Vector Machine (SVC)
    - Gaussian Naive Bayes

## Results

All models performed exceptionally well, but KNN stood out as the most accurate.

| Model | Accuracy |
| :--- | :---: |
| **K-Nearest Neighbors (KNN)** | **98%** |
| Logistic Regression | 95% |
| Support Vector Machine (SVC) | 95% |
| Gaussian Naive Bayes | 94% |

The outstanding performance of the KNN model suggests that in the high-dimensional vector space created by spaCy, articles of the same class (Real/Fake) are tightly clustered together. This makes a distance-based algorithm like KNN particularly effective.

## Installation and Usage

To run this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/fake-news-classification.git](https://github.com/your-username/fake-news-classification.git)
    cd fake-news-classification
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the spaCy language model:**
    ```bash
    python -m spacy download en_core_web_sm
    ```

5.  **Launch Jupyter Notebook and open the file:**
    ```bash
    jupyter notebook fake_news_classification.ipynb
    ```

## Libraries Used
- pandas
- NumPy
- scikit-learn
- spaCy
