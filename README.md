# Sentiment Analysis for Bali Tourism Reviews

This is a Streamlit web app for performing **sentiment analysis** on Indonesian tourism reviews using **Attention-based BiLSTM models**, including a **LoRA-optimized version**. Users can input a text review and choose which model to use.

---

## **Features**

- Two models:
  - `Attention Model` — standard attention-based BiLSTM.
  - `LoRA Attention Model` — low-rank adaptation for faster/fine-tuned performance.
- Preprocessing:
  - Casefolding
  - Tokenization with spaCy
  - Stopword removal (Indonesian stopwords, keeps negation words)
  - Stemming with Sastrawi
  - Merging of negation phrases and intensifiers
- Real-time sentiment prediction via Streamlit interface.

---
Developed by Tristan Bey Kusuma (NIM : 2008561053) for research "Implementasi Attention-Based BiLSTM dengan LoRA Parameter Tuning untuk Analisis Sentimen Ulasan Destinasi Wisata" published at [https://doi.org/10.62411/tc.v25i1.15089](https://doi.org/10.62411/tc.v25i1.15089)
Model trained in Google Colab [https://colab.research.google.com/drive/1IIklYErR25Jhm48lofkba__suQHhApWJ?usp=sharing](https://colab.research.google.com/drive/1IIklYErR25Jhm48lofkba__suQHhApWJ?usp=sharing)
