# SBERT Sentiment Analysis (MSML610 - Fall 2025)

This project performs sentiment analysis using Sentence-BERT embeddings on the Financial PhraseBank dataset.

## Dataset
- Source: [Kaggle - Financial PhraseBank](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)
- Download manually and place in `data/raw/`.

## Quickstart
```bash
cd class_project/MSML610/Fall2025/projects/Fall2025_SBert_Sentiment_Analysis_with_Sentence_Embeddings
python src/preprocess.py
python src/sbert_embed.py
