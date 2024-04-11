
# Recommendation Engine with Sentence Transformers and Vector Database

This project implements a recommendation engine using sentence transformers and a Vector Database on the War News dataset, publicly available on Kaggle https://www.kaggle.com/datasets/armitaraz/google-war-news

## Text Similarity Analysis

### Overview
This notebook conducts a comprehensive analysis of text similarity between news headlines and summaries. It employs various methods, including TF-IDF, Word2Vec, GloVe, and Universal Sentence Encoder (USE), to evaluate semantic similarity across different text representation and embedding models.

### Data
The dataset "war-news.csv" serves as the primary data source for this analysis. It comprises news headlines and summaries, which are preprocessed to handle missing values and perform text cleaning.

### Methods
- **TF-IDF Representation**: TF-IDF vectors are computed for headlines and summaries, followed by cosine similarity calculation to quantify their semantic similarity.
- **Word Embedding Models**:
  - **Word2Vec**: Utilizes a pre-trained Word2Vec model to convert text into word embeddings. Cosine similarity is computed to measure similarity between headline and summary embeddings.
  - **GloVe**: Employs a pre-trained GloVe model to generate word embeddings for text data. Similarity between headline and summary embeddings is calculated using cosine similarity.
- **Universal Sentence Encoder (USE)**: Utilizes the Universal Sentence Encoder from TensorFlow Hub to encode headlines and summaries into embeddings. Cosine similarity is then computed to assess semantic similarity.

### Analysis
The notebook presents the top 5 most similar pairs of headlines and summaries for each method. It offers insights into the effectiveness of different text representation and embedding models in capturing semantic similarity between news articles.

### Usage
1. Ensure the required libraries are installed (`pandas`, `scikit-learn`, `NLTK`, `Gensim`, `NumPy`, `TensorFlow`).
2. Download the dataset "war-news.csv" and place it in the appropriate directory.
3. Run the notebook cell by cell to perform text similarity analysis using different methods.

### Author
Saugat Singh

### License
This project is licensed under the MIT License.
