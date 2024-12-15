# Myers-Briggs Personality Prediction Using Deep Learning

## Overview
This project predicts Myers-Briggs Personality Type (MBTI) dimensions based on user-generated text using Natural Language Processing (NLP) and deep learning techniques. By leveraging pre-trained BERT models, the system classifies personality dimensions into binary categories (e.g., Introversion vs. Extraversion) with improved accuracy and robustness.

## Dataset
The dataset is sourced from [Kaggle's Myers-Briggs Personality Type Dataset](https://www.kaggle.com/datasets/datasnaek/mbti-type). It includes approximately 8,600 samples of user social media posts, each labeled with one of 16 MBTI types.

Key steps:
- Analyzed and addressed class imbalance issues.
- Expanded data from ~3,000 entries to ~57,500 samples using text chunking.

## Key Features
- **Binary Classification**: Simplified MBTI classification into four binary dimensions:
  - Introversion (I) vs. Extraversion (E)
  - Intuition (N) vs. Sensing (S)
  - Thinking (T) vs. Feeling (F)
  - Judging (J) vs. Perceiving (P)
- **Fine-Tuned BERT Model**: Customized the pre-trained `bert-base-uncased` model for MBTI personality prediction.
- **Text Chunking**: Split long text posts into 128-token chunks, enhancing data utilization and improving accuracy by 2-3%.
- **Model Optimization**:
  - Early stopping with validation monitoring to prevent overfitting.
  - Majority voting to aggregate chunk predictions into a single output.
- **Baseline Comparison**: Benchmarked performance against logistic regression using BERT embeddings.

## Results
- Fine-tuned BERT achieved higher precision, recall, and F1 scores than baseline models.
- Significant improvements in predicting dimensions such as Introversion vs. Extraversion.
- Identified challenges in predicting certain dimensions (e.g., Sensing vs. Intuition) due to dataset imbalance.

## Technologies Used
- **Programming**: Python
- **Deep Learning Framework**: PyTorch
- **Models**: Pre-trained BERT (`bert-base-uncased`)
- **Tools**: Scikit-learn, Hugging Face Transformers, NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/mbti-prediction.git
