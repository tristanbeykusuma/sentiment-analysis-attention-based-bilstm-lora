import streamlit as st
import tensorflow as tf
import json
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import spacy
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# -----------------------------
# Custom layers
# -----------------------------
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="random_normal")
        self.b = self.add_weight(name="att_bias", shape=(1,), initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        e = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = inputs * a
        return tf.reduce_sum(output, axis=1)


class ManualLoRA(tf.keras.layers.Layer):
    def __init__(self, rank, original_units, **kwargs):
        super(ManualLoRA, self).__init__(**kwargs)
        self.rank = rank
        self.original_units = original_units

    def build(self, input_shape):
        self.A = self.add_weight(shape=(input_shape[-1], self.rank),
                                 initializer='random_normal', trainable=True)
        self.B = self.add_weight(shape=(self.rank, self.original_units),
                                 initializer='zeros', trainable=True)
        super(ManualLoRA, self).build(input_shape)

    def call(self, inputs):
        low_rank_update = tf.matmul(inputs, self.A)
        low_rank_update = tf.matmul(low_rank_update, self.B)
        return inputs + low_rank_update

# -----------------------------
# Caching resources to optimize
# -----------------------------
@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_max_length():
    with open("max_length.json", "r") as f:
        config = json.load(f)
    return config["max_length"]

@st.cache_resource
def load_models():
    # Load Attention model (.keras format)
    model_attention = tf.keras.models.load_model(
        "attention_model.keras",
        custom_objects={"AttentionLayer": AttentionLayer}
    )

    # Load LoRA Attention model (.keras format)
    model_lora = tf.keras.models.load_model(
        "lora_attention_model.keras",
        custom_objects={"AttentionLayer": AttentionLayer, "ManualLoRA": ManualLoRA}
    )

    return model_attention, model_lora

tokenizer = load_tokenizer()
max_length = load_max_length()
model_attention, model_lora = load_models()

# -----------------------------
# Preprocessing setup
# -----------------------------
stop_words = set(stopwords.words('indonesian'))
stop_words -= {"tidak", "bukan", "jangan", "belum"}

factory = StemmerFactory()
stemmer = factory.create_stemmer()
nlp = spacy.blank("id")

negation_words = {"tidak", "bukan", "jangan", "belum"}
intensifier_words = ["sangat", "benar-benar", "amat", "paling"]

def casefolding(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[-+]?[0-9]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def merge_negations_advanced(tokens):
    merged = []
    i = 0
    while i < len(tokens):
        if tokens[i] in negation_words and i+1 < len(tokens):
            merged.append(f"{tokens[i]}_{tokens[i+1]}")
            i += 2
        elif tokens[i] in intensifier_words and i+1 < len(tokens):
            merged.append(f"{tokens[i]}_{tokens[i+1]}")
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    return merged

def remove_stop_words(tokens):
    return [w for w in tokens if w not in stop_words or "_" in w]

def stem_words(tokens):
    return [stemmer.stem(w) for w in tokens]

def preprocess_text(text):
    t = casefolding(text)
    tokens = [tok.text for tok in nlp(t)]
    tokens = merge_negations_advanced(tokens)
    tokens = remove_stop_words(tokens)
    tokens = stem_words(tokens)
    return " ".join(tokens)

reverse_label_mapping = {0: "positive", 1: "negative", 2: "neutral"}

def predict_sentiment(text, model):
    preprocessed = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([preprocessed])
    pad = pad_sequences(seq, maxlen=max_length, padding='post')
    pred = model.predict(pad)
    label = np.argmax(pred, axis=1)[0]
    return reverse_label_mapping[label]

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Sentiment Analysis Bali Tourism", layout="wide")
st.title("🎭 Sentiment Analysis for Bali Tourism Reviews")

st.markdown("""
**Oleh Tristan Bey Kusuma (NIM 2008561053)**  
Tugas Akhir “Implementasi Attention-Based BiLSTM dengan LORA Parameter Tuning untuk Analisis Sentimen Ulasan Destinasi Wisata”  
Program Studi Informatika  
Fakultas Matematika dan Ilmu Pengetahuan Alam  
Universitas Udayana, 2025
""")

st.write("Masukkan ulasan wisata Anda dan pilih model untuk memprediksi sentimen:")

model_choice = st.selectbox("Pilih model:", ["Attention Model", "LoRA Attention Model"])
model_to_use = model_attention if model_choice == "Attention Model" else model_lora

user_input = st.text_area("📝 Tulis ulasan di sini:")

if st.button("Prediksi Sentimen"):
    if user_input.strip():
        with st.spinner("Memproses..."):
            result = predict_sentiment(user_input, model_to_use)
        st.success(f"Predicted sentiment ({model_choice}): **{result}**")
    else:
        st.warning("Harap masukkan teks terlebih dahulu.")
