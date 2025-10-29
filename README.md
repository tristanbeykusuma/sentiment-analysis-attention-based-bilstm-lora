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
Developed by Tristan Bey Kusuma (NIM : 2008561053)