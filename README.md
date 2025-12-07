# ✈️ Air France Voice of Customer (VoC) Dashboard

A Data Science portfolio project that transforms raw customer reviews into actionable business insights using Natural Language Processing (NLP).

🔗 **Live Demo:** [Click here to try the app](https://air-france-review-voc-dashboard-by-fauzan-pahlevi.streamlit.app/)

## 📌 Project Overview
This dashboard helps airline management understand passenger sentiment and identify root causes of dissatisfaction. Unlike standard analysis, this project employs a **Hybrid Anchor Strategy** (Title + Text concatenation) to capture high-density sentiment signals often missed in long descriptive text.

## 🛠 Tech Stack
- **Python** (Pandas, NumPy)
- **Visualization:** Streamlit, Matplotlib, Seaborn
- **NLP:** NLTK (VADER Sentiment Analysis), Scikit-Learn (CountVectorizer for Bigram Analysis)

## 🔍 Key Features
1.  **Business Simulation:** Estimates potential revenue loss based on negative sentiment churn rate.
2.  **Hybrid Sentiment Analysis:** Combines "Review Title" (Anchor) and "Review Body" to improve context detection.
3.  **Root Cause Analysis:** Extracts most frequent bigrams (two-word phrases) from negative reviews to pinpoint specific issues (e.g., "lost luggage", "late flight").
4.  **Defensive Programming:** Robust error handling for file uploads and data format mismatches.

## ⚠️ Model Limitations & Known Issues
This project uses **VADER**, a lexicon-based analysis tool. While efficient for real-time dashboards, I have identified specific limitations during the analysis of this dataset:
* **Typo Sensitivity:** Reviews containing typos (e.g., *"Service por"*) may be misclassified as Neutral because the token is out-of-vocabulary.
* **Signal Dilution:** Long, narrative reviews describing bureaucratic issues without using explicit negative adjectives (e.g., "bad", "terrible") can sometimes result in False Positives due to the mathematical aggregation of neutral/polite words.
* **Sarcasm:** Phrases like *"I expected better"* might be scored positively due to the word "better", ignoring the context of disappointment.

## 🚀 How to Run Locally
1. Clone this repository.
2. Install requirements: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`
