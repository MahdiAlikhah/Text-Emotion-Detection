# Text Emotion Detection

A machine learning project for recognizing emotions in text using traditional NLP techniques and supervised classification.

The project explores how text representations and different classifiers can be used to identify emotional categories in short text messages. It includes a model training notebook, serialized model artifacts, and a small interactive application built with Streamlit.

## Overview

Sentiment analysis typically focuses on whether a piece of text is positive, negative, or neutral. Emotion detection takes this a step further by attempting to identify the specific emotion expressed in the text.

This project approaches the problem as a multiclass text classification task. It uses TF-IDF to represent text and compares several classical machine learning classifiers before selecting a model for inference.

## Emotion Categories

The model recognizes eight emotion categories:

- Anger
- Disgust
- Fear
- Joy
- Neutral
- Sadness
- Shame
- Surprise

## Methodology

The project follows a traditional NLP classification pipeline.

1. **Data preparation** — Load and preprocess the labeled text dataset.
2. **Feature extraction** — Convert text into numerical representations using TF-IDF.
3. **Model training** — Train and compare multiple classification algorithms.
4. **Evaluation** — Compare models using accuracy, precision, recall, and F1-score.
5. **Inference** — Save the selected classifier and vectorizer for use in the application.

### Feature Extraction

TF-IDF (Term Frequency–Inverse Document Frequency) is used to represent text numerically.

Configuration:
- Maximum features: 5,000
- N-gram range: (1, 2)
- Unigrams and bigrams

### Models

Four classifiers were evaluated:

- Logistic Regression
- Multinomial Naive Bayes
- Support Vector Machine (SVM)
- Stochastic Gradient Descent (SGD)

The SVM classifier was selected for the final application based on its performance on the test set.

## Results

The dataset contains 34,792 samples after preprocessing, with an 80/20 train-test split.

| Model | Accuracy |
|---|---:|
| Logistic Regression | 61.30% |
| Naive Bayes | 57.11% |
| SVM | **62.54%** |
| SGD | 58.82% |

The SVM achieved the highest overall accuracy, with a weighted F1-score of 0.62.

These results represent the original experiment and should be interpreted in the context of the dataset's class imbalance. In particular, some emotion categories have considerably fewer examples than others.

## Demo

The project includes a simple Streamlit application for interactive predictions.

Run the application locally:

```bash
git clone https://github.com/MahdiAlikhah/Text-Emotion-Detection.git
cd Text-Emotion-Detection

pip install streamlit numpy pandas scikit-learn joblib altair
streamlit run app.py
```

Enter a text sample to receive its predicted emotion.

## Project Structure

```text
Text-Emotion-Detection/
├── app.py                    # Streamlit application
├── model.ipynb               # Data exploration and model training
├── emotion_dataset_raw.csv   # Original dataset
├── emotion.pkl               # Trained SVM classifier
├── tfidf_vectorizer.pkl      # Fitted TF-IDF vectorizer
└── README.md
```

## Limitations

- Predictions are limited to the emotion categories represented in the training data.
- The dataset has an imbalanced class distribution, which affects performance on less frequent emotions.
- Text-based emotion detection cannot reliably capture context, sarcasm, or the full range of human emotional expression.

## License

MIT
