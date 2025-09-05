"""
Script para configurar el sistema de monitoreo inicial.
"""
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from monitoring.drift_monitor import ModelMonitor
from src.utils import load_imdb_data, prepare_data
import pandas as pd

import warnings

# Ignorar todos los warnings
warnings.filterwarnings("ignore")


def setup_monitoring():
    """Configura el sistema de monitoreo con datos de referencia."""
    print("=== Configurando Sistema de Monitoreo ===\n")
    
    # 1. Verificar que existe un modelo entrenado
    # model_path = "models/random_forest_bayesian.pkl"

    import glob
    model_files = glob.glob("models/*bayesian*.pkl")
    if model_files:
        model_path = model_files[0]  # Usar el primero que encuentre
        print(f" Modelo encontrado: {model_path}")
    else:
        print(" No se encontró ningún modelo bayesiano.")
        print(" Ejecuta primero: python src/train_bayesian.py")
        return
        
    if not os.path.exists(model_path):
        print("❌ No se encontró modelo entrenado.")
        print("   Ejecuta primero: python src/train_bayesian.py")
        return
    
    print("✓ Modelo encontrado")
    
    # 2. Cargar datos de test para crear referencia
    print("\nCargando datos...")
    df = load_imdb_data()
    train_df, test_df = prepare_data(df, test_size=0.2)
    
    # 3. Crear monitor y generar datos de referencia
    print("\nCreando datos de referencia para monitoreo...")
    monitor = ModelMonitor(model_path)
    
    X_test = test_df['review']
    y_test = test_df['sentiment']
    
    reference_df = monitor.create_reference_data(X_test, y_test)
    
    # 4. Simular algunas predicciones de producción para testing
    print("\nSimulando datos de producción...")
    sample_reviews = [
        "This movie was absolutely fantastic! Best film of the year!",
        "Terrible waste of time. Bad acting and boring plot.",
        "Not bad but not great either. Just okay.",
        "Amazing cinematography and great performances!",
        "I fell asleep halfway through. So boring.",
        "One of the worst movies I've ever seen.",
        "Brilliant storytelling and direction!",
        "Completely predictable and uninspiring."
    ]
    
    predictions = monitor.model.predict(sample_reviews)
    probabilities = monitor.model.predict_proba(sample_reviews)
    
    prod_data = monitor.collect_production_data(
        sample_reviews, predictions, probabilities
    )
    
    # 5. Generar reportes iniciales
    print("\nGenerando reportes iniciales...")
    
    # Reporte de drift
    drift_report, drift_metrics = monitor.generate_drift_report(prod_data)
    print(f"\nMétricas de drift iniciales:")
    print(f"  - Dataset drift: {drift_metrics['dataset_drift']:.2%}")
    print(f"  - Prediction drift: {drift_metrics['prediction_drift']}")
    
    # Reporte de performance
    monitor.generate_performance_report(prod_data)
    
    # 6. Configurar directorios
    directories = [
        "monitoring_data",
        "monitoring_data/reports",
        "monitoring_data/metrics",
        "logs/monitoring"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    setup_monitoring()