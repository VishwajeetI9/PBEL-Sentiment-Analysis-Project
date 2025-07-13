# Sentiment Analysis of Product Reviews using Naive Bayes

This project is a beginner-friendly **NLP + ML web app** that classifies product reviews as **Positive** or **Negative** using a Naive Bayes model. It includes a Streamlit-based interactive GUI and supports both single review analysis and batch CSV uploads.

## Features

- 🔤 Analyze individual product reviews in real-time
- 📁 Upload CSV files to analyze multiple reviews in batch
- 📊 Clean user interface with styled buttons and emojis
- ⬇️ Download sentiment predictions as a new CSV
- ✅ Deployable with Streamlit Cloud


## Technologies Used

- Python
- Natural Language Processing (`nltk`)
- Machine Learning (`scikit-learn`)
- Web Interface (`Streamlit`)
- Data Handling (`pandas`)
- TF-IDF Vectorization


## Dataset

- **Source:** [Amazon Fine Food Reviews on Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- Used `Text` and `Score` columns
- Removed neutral reviews (3 stars)
- Balanced dataset with equal positive and negative reviews


## ⚙️ How to Run Locally

1. **Clone the repository:**
git clone https://github.com/your-username/sentiment-analysis-app.git
cd sentiment-analysis-app

2. **Install the dependencies:**
pip install -r requirements.txt

3. **Run the Streamlit app:**
streamlit run app.py

4. **Open in browser:**
App will open at http://localhost:8501

## Future Improvements

- Add support for neutral reviews 
- Visualize sentiment ratio using charts
- Export results to Excel
- Use BERT or other advanced models

## Author
Vishwajeet Tiwari
