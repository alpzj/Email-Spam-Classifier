# 📧 Email/SMS Spam Detection Using Machine Learning

This is a machine learning project to classify SMS or Email messages as **Spam** or **Not Spam** using NLP techniques and various classification models. The model is deployed using **Streamlit** and hosted on **Render**.

---

## 🚀 Demo

👉 [Click here to try the live app](https://emailspamclassifier-mfwl.onrender.com)  

---

## 📌 Project Overview

This project processes a dataset of text messages and uses natural language processing (NLP) techniques for cleaning and transforming the text. Various machine learning models are then trained to classify the messages into spam or ham (not spam). The best performing model is deployed in a web application using **Streamlit**.

---

## 📁 Dataset

- Source: [UCI SMS Spam Collection Dataset]
- Records: 5,572
- Columns: `v1` (label - spam or ham), `v2` (message text)

---

## 🧹 Data Cleaning & Preprocessing

- Dropped unnecessary columns
- Renamed columns for clarity (`v1` → `target`, `v2` → `text`)
- Label Encoding (`ham` → 0, `spam` → 1)
- Removed duplicates
- Added additional features: `num_characters`, `num_words`, `num_sentences`
- Text normalization:
  - Lowercasing
  - Tokenization
  - Stopword and punctuation removal
  - Stemming using PorterStemmer

---

## 📊 Exploratory Data Analysis (EDA)

- Visualized class imbalance using a pie chart
- Analyzed distribution of message lengths and word counts
- WordClouds for spam vs ham messages
- Most common words in spam and ham messages

---

## 🧠 Models Trained

Various ML classifiers were trained and evaluated:

| Model              | Accuracy | Precision |
|-------------------|----------|-----------|
| K-Nearest Neighbors | 90.5%    | 100%      |
| Multinomial Naive Bayes | 97.1%    | 100%      |
| Random Forest      | 97.6%    | 98.3%     |
| SVC (Sigmoid)      | 97.6%    | 97.4%     |
| ExtraTrees         | 97.5%    | 97.4%     |
| Logistic Regression| 95.8%    | 97.0%     |
| XGBoost            | 96.7%    | 94.8%     |
| Gradient Boosting  | 94.7%    | 91.9%     |
| Decision Tree      | 92.7%    | 81.2%     |
| AdaBoost           | 92.4%    | 84.9%     |
| Bagging            | 95.8%    | 86.8%     |

### ✅ Final Model Used for Deployment:

- **Multinomial Naive Bayes** (with TF-IDF Vectorizer)
- **Precision: 100%**, **Accuracy: ~97%**

---

## 🧪 Model Evaluation

- Accuracy
- Precision (focused more due to spam classification)
- Confusion Matrix
- Stacking & Voting Classifiers used for performance improvement

---

## 📦 Project Structure

```bash
├── app.py                # Streamlit app
├── model.pkl             # Trained ML model
├── vectorizer.pkl        # TF-IDF vectorizer
├── requirements.txt      # Python dependencies
├── setup.sh              # Render setup script
├── Procfile              # Render deployment file
├── .gitignore
└── README.md             # Project documentation
