import streamlit as st
import pandas as pd
import fitz # PyMuPDF
import spacy
from spacy.lang.es.stop_words import STOP_WORDS
from collections import Counter
import nltk
# Ensure nltk data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.util import ngrams # Import ngrams here
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import os # Import os for file handling

# Define the analysis functions (copying from previous cells)
def read_pdf(pdf_path):
    """Reads text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def process_with_spacy(text):
    """Processes text using spaCy."""
    # Ensure the model is loaded or downloaded if not present
    try:
        nlp = spacy.load("es_core_news_sm")
    except OSError:
        st.error("SpaCy model 'es_core_news_sm' not found. Attempting download...")
        try:
            # This download command might need to be run outside the app context or handled differently in deployment
            # For local testing, running it once before starting the app is recommended.
            # In a deployed app, the model should be part of the deployment package.
            os.system("python -m spacy download es_core_news_sm")
            nlp = spacy.load("es_core_news_sm")
            st.success("SpaCy model downloaded and loaded.")
        except Exception as e:
            st.error(f"Error downloading SpaCy model: {e}")
            return None
    doc = nlp(text)
    return doc

def lemmatize_and_clean(doc):
    """Lemmatizes and cleans tokens from a spaCy doc."""
    lemmas = []
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_space:
            lemmas.append(token.lemma_.lower())
    return lemmas

def calculate_lemma_frequency(lemmas):
    """Calculates and returns the frequency of lemmas."""
    frecuencias = Counter(lemmas)
    df_lemmas = pd.DataFrame(frecuencias.items(), columns=['lema', 'frecuencia'])
    df_lemmas = df_lemmas.sort_values(by='frecuencia', ascending=False)
    return df_lemmas

def calculate_bigrams(text):
    """Calculates and returns the frequency of bigrams."""
    tokens = word_tokenize(text.lower(), language='spanish')
    # Use the STOP_WORDS from spacy loaded globally or within process_with_spacy
    # Ensure STOP_WORDS is accessible here, maybe pass it as an argument or load globally
    # from spacy.lang.es.stop_words import STOP_WORDS # Already imported globally

    tokens = [t for t in tokens if t.isalpha() and t not in STOP_WORDS]
    bigrams = list(ngrams(tokens, 2))
    bigrama_frecuencias = Counter(bigrams)
    df_bigrams = pd.DataFrame(bigrama_frecuencias.items(), columns=['bigrama', 'frecuencia'])
    df_bigrams = df_bigrams.sort_values(by='frecuencia', ascending=False)
    return df_bigrams

def perform_pos_tagging(doc):
    """Performs POS tagging and returns results in a DataFrame."""
    pos = [(token.text, token.pos_) for token in doc if not token.is_punct and not token.is_space]
    df_pos = pd.DataFrame(pos, columns=['token', 'POS'])
    return df_pos

def perform_ner(doc):
    """Performs Named Entity Recognition and returns results in a DataFrame."""
    entidades = [(ent.text, ent.label_) for ent in doc.ents]
    df_ner = pd.DataFrame(entidades, columns=['entidad', 'tipo'])
    return df_ner

def analyze_sentiment(doc): # Analyze sentiment based on spaCy doc for sentence splitting
    """Analyzes sentiment of the text using VADER."""
    analyzer = SentimentIntensityAnalyzer()

    oraciones = [sent.text for sent in doc.sents]

    resultados_sentimiento = []
    for oracion in oraciones:
        vs = analyzer.polarity_scores(oracion)
        resultados_sentimiento.append({'oracion': oracion, 'sentimiento': vs})

    if resultados_sentimiento: # Avoid division by zero if no sentences found
      puntuaciones_compuestas = [res['sentimiento']['compound'] for res in resultados_sentimiento]
      sentimiento_promedio = np.mean(puntuaciones_compuestas)
    else:
      sentimiento_promedio = 0 # Default to neutral if no sentences


    return sentimiento_promedio, resultados_sentimiento


# Streamlit App Layout
st.title("Análisis de Texto con Streamlit")

st.write("Sube un archivo PDF para realizar un análisis de lemas, N-gramas, POS tagging, NER y sentimiento.")

uploaded_file = st.file_uploader("Elige un archivo PDF", type="pdf")

if uploaded_file is not None:
    # To read file as string:
    # uploaded_file.getvalue().decode("utf-8")

    # Save the uploaded file temporarily to process it with fitz
    # Use a unique name or handle potential file access issues in deployment
    temp_file_path = "temp.pdf"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("Archivo cargado exitosamente.")

    # Perform the analysis
    text = read_pdf(temp_file_path)
    doc = process_with_spacy(text)

    # Clean up the temporary file
    os.remove(temp_file_path)


    if doc is not None: # Check if spaCy processing was successful
        lemmas = lemmatize_and_clean(doc)
        df_lemmas = calculate_lemma_frequency(lemmas)
        df_bigrams = calculate_bigrams(text) # Bigrams might be better calculated from processed tokens or original text carefully
        df_pos = perform_pos_tagging(doc)
        df_ner = perform_ner(doc)
        sentimiento_promedio, resultados_sentimiento = analyze_sentiment(doc) # Pass doc to sentiment analysis


        # Display the results
        st.header("Resultados del Análisis")

        st.subheader("Frecuencia de Lemas")
        st.dataframe(df_lemmas.head(20)) # Display top 20 lemmas

        st.subheader("Frecuencia de Bigramas")
        st.dataframe(df_bigrams.head(20)) # Display top 20 bigrams

        st.subheader("POS Tagging (Primeros 20)")
        st.dataframe(df_pos.head(20)) # Display top 20 POS tags

        st.subheader("Named Entity Recognition (NER) (Primeros 20)")
        st.dataframe(df_ner.head(20)) # Display top 20 entities

        st.subheader("Análisis de Sentimiento General")
        st.write(f"Puntuación de sentimiento compuesta promedio del texto: {sentimiento_promedio:.4f}")
        if sentimiento_promedio > 0.05:
            st.write("El sentimiento general del texto es **Positivo**.")
        elif sentimiento_promedio < -0.05:
            st.write("El sentimiento general del texto es **Negativo**.")
        else:
            st.write("El sentimiento general del texto es **Neutro**.")

        st.subheader("Resultados de Sentimiento por Oración (Primeras 10)")
        # Display sentiment for first 10 sentences or all if less than 10
        for i in range(min(10, len(resultados_sentimiento))):
            st.write(f"Oración: {resultados_sentimiento[i]['oracion'][:150]}...") # Display first 150 chars
            st.write(f"Sentimiento: {resultados_sentimiento[i]['sentimiento']}")

else:
    st.info("Por favor, sube un archivo PDF para comenzar el análisis.")
