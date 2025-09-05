"""
Script de entrenamiento para el clasificador de sentimientos.
Utiliza MLflow para tracking de experimentos y registro de modelos.
"""
import os
import mlflow
import mlflow.sklearn
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

from utils import load_imdb_data, prepare_data, save_model

# Cargar variables de entorno
load_dotenv()

# Configurar MLflow
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'file:./mlruns')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def train_model(model_type='logistic_regression', max_features=5000):
    """
    Entrena un modelo de clasificación de sentimientos.
    
    Args:
        model_type: Tipo de modelo ('logistic_regression' o 'random_forest')
        max_features: Número máximo de features para TF-IDF
    """
    # Configurar experimento en MLflow
    experiment_name = "movie_sentiment_classification"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"{model_type}_{datetime.now():%Y%m%d_%H%M%S}"):
        # Log de parámetros
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("max_features", max_features)
        
        print(f"Iniciando entrenamiento con {model_type}...")
        
        # 1. Cargar datos
        print("Cargando datos...")
        df = load_imdb_data()
        mlflow.log_param("total_samples", len(df))
        
        # 2. Preparar datos
        train_df, test_df = prepare_data(df, test_size=0.2)
        mlflow.log_param("train_samples", len(train_df))
        mlflow.log_param("test_samples", len(test_df))
        
        # 3. Vectorización TF-IDF
        print("Creando features TF-IDF...")
        vectorizer = TfidfVectorizer(max_features=max_features)
        X_train = vectorizer.fit_transform(train_df['cleaned_review'])
        X_test = vectorizer.transform(test_df['cleaned_review'])
        y_train = train_df['sentiment']
        y_test = test_df['sentiment']
        
        # Log del vocabulario size
        mlflow.log_param("vocabulary_size", len(vectorizer.vocabulary_))
        
        # 4. Entrenar modelo
        print(f"Entrenando modelo {model_type}...")
        
        if model_type == 'logistic_regression':
            model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                solver='liblinear'
            )
            mlflow.log_param("solver", "liblinear")
            mlflow.log_param("max_iter", 1000)
        
        elif model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            mlflow.log_param("n_estimators", 100)
        
        else:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}")
        
        # Entrenar
        model.fit(X_train, y_train)
        
        # 5. Evaluar modelo
        print("Evaluando modelo...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Log de métricas
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        print(f"\nResultados del modelo {model_type}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # 6. Crear y guardar matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusión - {model_type}')
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Predicción')
        plt.tight_layout()
        
        # Guardar figura
        os.makedirs('models/figures', exist_ok=True)
        fig_path = f'models/figures/confusion_matrix_{model_type}.png'
        plt.savefig(fig_path)
        plt.close()
        
        # Log de la figura en MLflow
        mlflow.log_artifact(fig_path)
        
        # 7. Guardar modelo localmente
        model_path = f'models/{model_type}_model.pkl'
        save_model(model, vectorizer, model_path)
        
        # 8. Registrar modelo en MLflow
        print("Registrando modelo en MLflow...")
        
        # Log del modelo
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=os.getenv('MODEL_NAME', 'movie_sentiment_classifier')
        )
        
        # Log del vectorizador como artifact
        mlflow.log_artifact(model_path, artifact_path="model_artifacts")
        
        # 9. Crear ejemplos de predicción
        example_reviews = [
            "This movie was absolutely amazing! Best film I've seen all year.",
            "Terrible movie. Complete waste of time and money.",
            "It was okay, nothing special but not bad either."
        ]
        
        print("\nEjemplos de predicción:")
        for review in example_reviews:
            # Usar la función predict_sentiment de utils
            from utils import predict_sentiment
            pred, prob = predict_sentiment(review, model, vectorizer)
            sentiment = "Positivo" if pred == 1 else "Negativo"
            print(f"Review: '{review[:50]}...'")
            print(f"Predicción: {sentiment} (Confianza: {prob:.2%})\n")
        
        # Log de tags
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("dataset", "IMDB")
        mlflow.set_tag("task", "sentiment_classification")
        
        print(f"✓ Entrenamiento completado. Run ID: {mlflow.active_run().info.run_id}")
        
        return model, vectorizer, accuracy


def compare_models():
    """
    Entrena y compara múltiples modelos.
    """
    print("=== Comparación de Modelos ===\n")
    
    results = {}
    
    # Entrenar Logistic Regression
    lr_model, lr_vectorizer, lr_accuracy = train_model('logistic_regression')
    results['logistic_regression'] = lr_accuracy
    
    print("\n" + "="*50 + "\n")
    
    # Entrenar Random Forest
    rf_model, rf_vectorizer, rf_accuracy = train_model('random_forest')
    results['random_forest'] = rf_accuracy
    
    # Mostrar comparación
    print("\n=== Resumen de Resultados ===")
    for model_name, accuracy in results.items():
        print(f"{model_name}: {accuracy:.4f}")
    
    # Determinar mejor modelo
    best_model = max(results, key=results.get)
    print(f"\nMejor modelo: {best_model} con accuracy de {results[best_model]:.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Entrenar clasificador de sentimientos')
    parser.add_argument(
        '--model',
        type=str,
        choices=['logistic_regression', 'random_forest', 'compare'],
        default='logistic_regression',
        help='Tipo de modelo a entrenar o "compare" para comparar todos'
    )
    parser.add_argument(
        '--max-features',
        type=int,
        default=5000,
        help='Número máximo de features para TF-IDF'
    )
    
    args = parser.parse_args()
    
    if args.model == 'compare':
        compare_models()
    else:
        train_model(args.model, args.max_features)