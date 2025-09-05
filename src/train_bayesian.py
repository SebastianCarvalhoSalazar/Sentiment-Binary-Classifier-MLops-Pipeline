"""
Script de entrenamiento con optimización bayesiana de hiperparámetros.
Incluye Logistic Regression, Random Forest y XGBoost.
"""
import os
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import joblib
from joblib import Memory
from skopt.callbacks import CheckpointSaver

from utils import load_imdb_data, prepare_data, CustomLogCallback

# Cargar variables de entorno
load_dotenv()

# Configurar MLflow
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'file:./mlruns')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def train_logistic_regression_bayesian(X_train, X_test, y_train, y_test, max_features=5000):
    """
    Entrena Logistic Regression con optimización bayesiana.
    """
    with mlflow.start_run(run_name=f"logistic_regression_bayesian_{datetime.now():%Y%m%d_%H%M%S}"):
        mlflow.log_param("model_type", "logistic_regression_bayesian")
        mlflow.log_param("optimization", "bayesian")

        # Pipeline para clasificación de texto
        memory = Memory('./cache_lr', verbose=1)

        # Pipeline con TF-IDF y Logistic Regression
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))),
            ('classifier', LogisticRegression(random_state=42)),
        ],
            memory=memory)

        # Espacio de búsqueda de hiperparámetros
        param_space = {
            'tfidf__max_features': Integer(1000, 10000),
            'tfidf__min_df': Integer(1, 10),
            'tfidf__max_df': Real(0.7, 1.0),
            'classifier__C': Real(0.001, 100, prior='log-uniform'),
            'classifier__penalty': Categorical(['l1', 'l2']),
            'classifier__solver': Categorical(['liblinear', 'saga']),
            'classifier__max_iter': Integer(100, 1000)
        }

        # Búsqueda bayesiana
        print("Iniciando optimización bayesiana para Logistic Regression...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_callback = CustomLogCallback("logs/optimization_lr_details.txt")
        checkpoint_callback = CheckpointSaver(
            str(f"logs/checkpoint_lr_{timestamp}.pkl"),
            compress=6
        )
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        bayes_search = BayesSearchCV(
            estimator=pipe,
            search_spaces=param_space,
            n_iter=10,  # Número de iteraciones
            scoring='f1',
            cv=skf,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )

        # Entrenar con los datos de texto directamente
        bayes_search.fit(X_train, y_train, callback=[
                         checkpoint_callback, log_callback])

        # Mejores parámetros
        best_params = bayes_search.best_params_
        best_score = bayes_search.best_score_

        print(f"\nMejores parámetros: {best_params}")
        print(f"Mejor score CV (F1): {best_score:.4f}")

        # Log parámetros
        for param, value in best_params.items():
            mlflow.log_param(param, value)
        mlflow.log_metric("best_cv_score", best_score)

        # Evaluar en test
        y_pred = bayes_search.predict(X_test)

        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        print(f"\nResultados en Test:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

        # # Guardar modelo
        # mlflow.sklearn.log_model(
        #     sk_model=bayes_search.best_estimator_,
        #     artifact_path="model",
        #     registered_model_name="movie_sentiment_lr_bayesian"
        # )

        # Primero loguear
        model_info = mlflow.sklearn.log_model(
            sk_model=bayes_search.best_estimator_,
            artifact_path="model"
        )

        # Luego registrar en el Model Registry
        mlflow.register_model(
            model_uri=model_info.model_uri,
            name="movie_sentiment_classifier"  # Nombre en el registry
        )        

        # Guardar localmente
        model_path = 'models/logistic_regression_bayesian.pkl'
        joblib.dump(bayes_search.best_estimator_, model_path)

        return bayes_search.best_estimator_, best_params


def train_random_forest_bayesian(X_train, X_test, y_train, y_test, max_features=5000):
    """
    Entrena Random Forest con optimización bayesiana y calibración isotónica.
    """
    with mlflow.start_run(run_name=f"random_forest_bayesian_{datetime.now():%Y%m%d_%H%M%S}"):
        mlflow.log_param("model_type", "random_forest_bayesian")
        mlflow.log_param("optimization", "bayesian")
        mlflow.log_param("calibration", "isotonic")

        # Pipeline para clasificación de texto
        memory = Memory('./cache_rf', verbose=1)

        # Pipeline con TF-IDF y Random Forest
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))),
            ('classifier', RandomForestClassifier(random_state=42))
        ],
            memory=memory)

        # Espacio de búsqueda de hiperparámetros
        param_space = {
            'tfidf__max_features': Integer(1000, 10000),
            'tfidf__min_df': Integer(1, 5),
            'classifier__n_estimators': Integer(50, 300),
            'classifier__max_depth': Integer(5, 50),
            'classifier__min_samples_split': Integer(2, 20),
            'classifier__min_samples_leaf': Integer(1, 10),
            'classifier__max_features': Categorical(['sqrt', 'log2', None]),
            'classifier__class_weight': Categorical(['balanced', None])
        }

        # Búsqueda bayesiana
        print("Iniciando optimización bayesiana para Random Forest...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_callback = CustomLogCallback("logs/optimization_rf_details.txt")
        checkpoint_callback = CheckpointSaver(
            str(f"logs/checkpoint_rf_{timestamp}.pkl"),
            compress=6
        )
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        bayes_search = BayesSearchCV(
            estimator=pipe,
            search_spaces=param_space,
            n_iter=10,
            scoring='f1',
            cv=skf,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )

        bayes_search.fit(X_train, y_train, callback=[
                         checkpoint_callback, log_callback])

        # Mejores parámetros
        best_params = bayes_search.best_params_
        best_score = bayes_search.best_score_

        print(f"\nMejores parámetros: {best_params}")
        print(f"Mejor score CV (F1): {best_score:.4f}")

        # Aplicar calibración isotónica
        from sklearn.calibration import CalibratedClassifierCV

        print("\nAplicando calibración isotónica...")
        calibrated_model = CalibratedClassifierCV(
            bayes_search.best_estimator_,
            method='isotonic',
            cv=3
        )
        calibrated_model.fit(X_train, y_train)

        # Log parámetros
        for param, value in best_params.items():
            mlflow.log_param(param, value)
        mlflow.log_metric("best_cv_score", best_score)

        # Evaluar en test con el modelo calibrado
        y_pred = calibrated_model.predict(X_test)
        y_pred_proba = calibrated_model.predict_proba(X_test)

        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log de métricas de calibración
        from sklearn.calibration import calibration_curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_pred_proba[:, 1], n_bins=10
        )

        # Calcular ECE (Expected Calibration Error)
        ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        mlflow.log_metric("expected_calibration_error", ece)

        print(f"\nResultados en Test (con calibración):")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Expected Calibration Error: {ece:.4f}")

        # Guardar modelo calibrado
        mlflow.sklearn.log_model(
            sk_model=calibrated_model,
            artifact_path="model",
            registered_model_name="movie_sentiment_rf_bayesian"
        )

        # Guardar localmente
        model_path = 'models/random_forest_bayesian.pkl'
        joblib.dump(calibrated_model, model_path)

        return calibrated_model, best_params


def train_xgboost_bayesian(X_train, X_test, y_train, y_test, max_features=5000):
    """
    Entrena XGBoost con optimización bayesiana y calibración isotónica.
    """
    with mlflow.start_run(run_name=f"xgboost_bayesian_calibrated_{datetime.now():%Y%m%d_%H%M%S}"):
        mlflow.log_param("model_type", "xgboost_bayesian_calibrated")
        mlflow.log_param("optimization", "bayesian")
        mlflow.log_param("calibration", "isotonic")

        # Pipeline para clasificación de texto
        memory = Memory('./cache_xgb', verbose=1)

        # Pipeline con TF-IDF y XGBoost
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))),
            ('classifier', XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=42
            ))
        ],
            memory=memory)

        # Calcular scale_pos_weight para clases desbalanceadas
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count

        # Espacio de búsqueda de hiperparámetros
        param_space = {
            'tfidf__max_features': Integer(1000, 10000),
            'classifier__n_estimators': Integer(50, 300),
            'classifier__max_depth': Integer(3, 10),
            'classifier__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'classifier__subsample': Real(0.6, 1.0),
            'classifier__colsample_bytree': Real(0.6, 1.0),
            'classifier__reg_alpha': Real(0, 10),
            'classifier__reg_lambda': Real(1, 10),
            'classifier__scale_pos_weight': Real(scale_pos_weight * 0.5, scale_pos_weight * 2)
        }

        # Búsqueda bayesiana
        print("Iniciando optimización bayesiana para XGBoost...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_callback = CustomLogCallback("logs/optimization_xgb_details.txt")
        checkpoint_callback = CheckpointSaver(
            str(f"logs/checkpoint_xgb_{timestamp}.pkl"),
            compress=6
        )
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        bayes_search = BayesSearchCV(
            estimator=pipe,
            search_spaces=param_space,
            n_iter=10,
            scoring='f1',
            cv=skf,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )

        bayes_search.fit(X_train, y_train, callback=[
                         checkpoint_callback, log_callback])

        # Mejores parámetros
        best_params = bayes_search.best_params_
        best_score = bayes_search.best_score_

        print(f"\nMejores parámetros: {best_params}")
        print(f"Mejor score CV (F1): {best_score:.4f}")

        # Log parámetros
        for param, value in best_params.items():
            mlflow.log_param(param, value)
        mlflow.log_metric("best_cv_score", best_score)

        # --- CALIBRACIÓN ISOTÓNICA ---
        print("\nAplicando calibración isotónica...")
        from sklearn.calibration import CalibratedClassifierCV

        calibrated_model = CalibratedClassifierCV(
            bayes_search.best_estimator_,
            method='isotonic',
            cv=3
        )
        calibrated_model.fit(X_train, y_train)

        # Evaluar en test con el modelo calibrado
        y_pred = calibrated_model.predict(X_test)
        y_pred_proba = calibrated_model.predict_proba(X_test)

        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log de métricas de calibración
        from sklearn.calibration import calibration_curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_pred_proba[:, 1], n_bins=10
        )

        # Calcular ECE (Expected Calibration Error)
        ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        mlflow.log_metric("expected_calibration_error", ece)

        print(f"\nResultados en Test (con calibración):")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Expected Calibration Error: {ece:.4f}")

        # Guardar modelo calibrado
        mlflow.sklearn.log_model(
            sk_model=calibrated_model,
            artifact_path="model",
            registered_model_name="movie_sentiment_xgb_bayesian_calibrated"
        )

        # Guardar localmente
        model_path = 'models/xgboost_bayesian_calibrated.pkl'
        joblib.dump(calibrated_model, model_path)

        return calibrated_model, best_params


def compare_models_bayesian():
    """
    Entrena y compara todos los modelos con optimización bayesiana.
    """
    print("=== Comparación de Modelos con Optimización Bayesiana ===\n")

    # Configurar experimento
    mlflow.set_experiment("movie_sentiment_bayesian_optimization")

    # Cargar y preparar datos
    print("Cargando datos...")
    df = load_imdb_data()
    train_df, test_df = prepare_data(df, test_size=0.2)

    # Usar el texto original para los pipelines (no el texto limpio)
    X_train = train_df['review']
    y_train = train_df['sentiment']
    X_test = test_df['review']
    y_test = test_df['sentiment']

    results = {}

    # Entrenar Logistic Regression
    print("\n" + "="*50)
    print("LOGISTIC REGRESSION CON BAYESIAN OPTIMIZATION")
    print("="*50)
    lr_model, lr_params = train_logistic_regression_bayesian(
        X_train, X_test, y_train, y_test)

    # Entrenar Random Forest
    print("\n" + "="*50)
    print("RANDOM FOREST CON BAYESIAN OPTIMIZATION")
    print("="*50)
    rf_model, rf_params = train_random_forest_bayesian(
        X_train, X_test, y_train, y_test)

    # Entrenar XGBoost
    print("\n" + "="*50)
    print("XGBOOST CON BAYESIAN OPTIMIZATION")
    print("="*50)
    xgb_model, xgb_params = train_xgboost_bayesian(
        X_train, X_test, y_train, y_test)

    print("\n=== Optimización Bayesiana Completada ===")
    print("Modelos guardados en ./models/")
    print("Resultados disponibles en MLflow UI (http://localhost:5000)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Entrenar modelos con optimización bayesiana')
    parser.add_argument(
        '--model',
        type=str,
        choices=['logistic_regression', 'random_forest', 'xgboost', 'compare'],
        default='compare',
        help='Modelo a entrenar con optimización bayesiana'
    )

    args = parser.parse_args()

    # Cargar y preparar datos
    df = load_imdb_data()
    train_df, test_df = prepare_data(df, test_size=0.2)
    X_train = train_df['review']
    y_train = train_df['sentiment']
    X_test = test_df['review']
    y_test = test_df['sentiment']

    if args.model == 'compare':
        compare_models_bayesian()
    elif args.model == 'logistic_regression':
        mlflow.set_experiment("movie_sentiment_bayesian_optimization")
        train_logistic_regression_bayesian(X_train, X_test, y_train, y_test)
    elif args.model == 'random_forest':
        mlflow.set_experiment("movie_sentiment_bayesian_optimization")
        train_random_forest_bayesian(X_train, X_test, y_train, y_test)
    elif args.model == 'xgboost':
        mlflow.set_experiment("movie_sentiment_bayesian_optimization")
        train_xgboost_bayesian(X_train, X_test, y_train, y_test)
