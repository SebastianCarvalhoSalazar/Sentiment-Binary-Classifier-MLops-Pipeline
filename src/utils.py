"""
Utilidades para el proyecto de clasificación de sentimientos.
"""
import os
import re
import pickle
import numpy as np
import pandas as pd
from typing import List, Tuple
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
from pathlib import Path

# Descargar recursos de NLTK si no están disponibles
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Configurar stopwords en inglés
STOPWORDS = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    """
    Limpia y preprocesa el texto para el modelo.

    Args:
        text: Texto a limpiar

    Returns:
        Texto limpio y procesado
    """
    # Convertir a minúsculas
    text = text.lower()

    # Eliminar etiquetas HTML
    text = re.sub(r'<[^>]+>', '', text)

    # Eliminar caracteres especiales y números
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Eliminar espacios múltiples
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenizar
    tokens = word_tokenize(text)

    # Eliminar stopwords
    tokens = [
        token for token in tokens if token not in STOPWORDS and len(token) > 2]

    return ' '.join(tokens)


def load_imdb_data(sample_size: int = None) -> pd.DataFrame:
    """
    Carga el dataset IMDB. Si no existe, crea datos de ejemplo.

    Args:
        sample_size: Número de muestras a cargar (None para todas)

    Returns:
        DataFrame con columnas 'review' y 'sentiment'
    """
    data_path = 'data/IMDB Dataset.csv'

    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
        print("Dataset Cargado con Exito!")
        print(df)
    else:
        # Crear datos de ejemplo si no existe el dataset
        print("Creando dataset de ejemplo...")

        positive_reviews = [
            "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout.",
            "I loved every minute of this film. The cinematography was breathtaking and the performances were outstanding.",
            "An amazing movie with a great story. Highly recommend watching it!",
            "Brilliant film! The director did an excellent job bringing this story to life.",
            "One of the best movies I've seen this year. The cast was perfect for their roles.",
            "Absolutely loved it! The special effects were incredible and the story was compelling.",
            "A masterpiece! This film will be remembered for years to come.",
            "Fantastic movie with great acting and an engaging plot. Worth every minute!",
            "The best film I've watched in a long time. Simply brilliant!",
            "Outstanding performance by the lead actor. The movie was captivating from start to finish."
        ] * 50  # Repetir para tener más datos

        negative_reviews = [
            "This was the worst movie I've ever seen. Complete waste of time.",
            "Terrible acting and a boring plot. I couldn't even finish watching it.",
            "Absolutely horrible. The story made no sense and the acting was awful.",
            "I want my money back. This movie was a complete disaster.",
            "Boring and predictable. The worst film of the year.",
            "Awful movie with terrible special effects and bad acting.",
            "Complete waste of time. The plot was confusing and the dialogue was terrible.",
            "One of the worst movies ever made. Save your money and skip this one.",
            "Terrible film. Poor acting, bad script, and awful direction.",
            "I fell asleep halfway through. Incredibly boring and poorly made."
        ] * 50  # Repetir para tener más datos

        # Crear DataFrame
        reviews = positive_reviews + negative_reviews
        sentiments = [1] * len(positive_reviews) + [0] * len(negative_reviews)

        df = pd.DataFrame({
            'review': reviews,
            'sentiment': sentiments
        })

        # Mezclar los datos
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Guardar para uso futuro
        os.makedirs('data', exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"Dataset de ejemplo creado y guardado en {data_path}")

    return df


def prepare_data(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepara los datos para entrenamiento.

    Args:
        df: DataFrame con los datos
        test_size: Proporción de datos para test

    Returns:
        Tupla con (train_df, test_df)
    """
    # Limpiar texto
    print("Limpiando texto...")
    df['cleaned_review'] = df['review'].apply(clean_text)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    # Eliminar reviews vacías después de la limpieza
    df = df[df['cleaned_review'].str.len() > 0]

    # Dividir en train/test
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=df['sentiment']
    )

    print(f"Datos de entrenamiento: {len(train_df)} muestras")
    print(f"Datos de prueba: {len(test_df)} muestras")
    print(
        f"Balance de clases en train: {train_df['sentiment'].value_counts().to_dict()}")

    return train_df, test_df


def save_model(model, vectorizer, model_path: str) -> None:
    """
    Guarda el modelo y el vectorizador.

    Args:
        model: Modelo entrenado
        vectorizer: Vectorizador TF-IDF
        model_path: Ruta donde guardar el modelo
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    model_dict = {
        'model': model,
        'vectorizer': vectorizer
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_dict, f)

    print(f"Modelo guardado en: {model_path}")


def load_model(model_path: str) -> Tuple:
    """
    Carga el modelo y el vectorizador.

    Args:
        model_path: Ruta del modelo guardado

    Returns:
        Tupla con (model, vectorizer)
    """
    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)

    return model_dict['model'], model_dict['vectorizer']


def predict_sentiment(text: str, model, vectorizer) -> Tuple[int, float]:
    """
    Predice el sentimiento de un texto.

    Args:
        text: Texto a clasificar
        model: Modelo entrenado
        vectorizer: Vectorizador TF-IDF

    Returns:
        Tupla con (predicción, probabilidad)
    """
    # Limpiar texto
    cleaned_text = clean_text(text)

    # Vectorizar
    text_vector = vectorizer.transform([cleaned_text])

    # Predecir
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0].max()

    return int(prediction), float(probability)


class CustomLogCallback:
    """Callback personalizado para logging detallado"""

    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.iteration = 0
        self.log_file = open(self.log_path, 'w')
        self.log_file.write(f"Optimization started at {datetime.now()}\n")
        self.log_file.write("="*60 + "\n")

    def __call__(self, res):
        """Llamado después de cada iteración"""
        self.iteration += 1

        # Información actual
        current_params = res.x_iters[-1]  # Últimos parámetros evaluados
        current_score = res.func_vals[-1]  # Último score
        best_score = min(res.func_vals)    # Mejor score hasta ahora

        # Escribir al log
        self.log_file.write(
            f"\n--- Iteration {self.iteration}/{len(res.x_iters)} ---\n")
        self.log_file.write(
            f"Timestamp: {datetime.now().strftime('%H:%M:%S')}\n")
        self.log_file.write(
            f"Current params: {dict(zip(res.space.dimension_names, current_params))}\n")
        
        # Negativo porque minimiza
        self.log_file.write(f"Current CV score: {-current_score:.6f}\n")
        self.log_file.write(f"Best score so far: {-best_score:.6f}\n")

        # Flush para escritura inmediata
        self.log_file.flush()

        # También imprimir en consola
        print(f"[{self.iteration}] Score: {-current_score:.6f} | Best: {-best_score:.6f}")

    def __del__(self):
        """Cerrar archivo al finalizar"""
        if hasattr(self, 'log_file'):
            self.log_file.write(
                f"\nOptimization completed at {datetime.now()}\n")
            self.log_file.close()
