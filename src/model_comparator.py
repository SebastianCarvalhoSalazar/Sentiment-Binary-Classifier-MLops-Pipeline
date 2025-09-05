"""
Script para comparar el modelo nuevo con el modelo en producción.
Usado por Airflow para decidir si promover el nuevo modelo.
"""
import os
import sys
import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Agregar src al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_imdb_data, prepare_data


def compare_models(new_model_path: str, production_model_path: str):
    """
    Compara un modelo nuevo con el modelo en producción.
    
    Returns:
        dict: Métricas comparativas y decisión de promoción
    """
    print("=== Comparando Modelos ===")
    
    # Cargar modelos
    print(f"Cargando modelo nuevo: {new_model_path}")
    new_model = joblib.load(new_model_path)
    
    if os.path.exists(production_model_path):
        print(f"Cargando modelo en producción: {production_model_path}")
        prod_model = joblib.load(production_model_path)
        has_production = True
    else:
        print("No hay modelo en producción. El nuevo modelo será promovido automáticamente.")
        has_production = False
        prod_model = None
    
    # Cargar datos de test para evaluación
    print("Preparando datos de evaluación...")
    df = load_imdb_data()
    train_df, test_df = prepare_data(df, test_size=0.2)
    
    # Usar solo una muestra para evaluación rápida
    test_sample = test_df.sample(n=min(500, len(test_df)), random_state=42)
    X_test = test_sample['review']
    y_test = test_sample['sentiment']
    
    # Evaluar nuevo modelo
    print("Evaluando modelo nuevo...")
    new_predictions = new_model.predict(X_test)
    new_metrics = {
        'accuracy': accuracy_score(y_test, new_predictions),
        'f1_score': f1_score(y_test, new_predictions),
        'precision': precision_score(y_test, new_predictions),
        'recall': recall_score(y_test, new_predictions)
    }
    
    # Evaluar modelo en producción si existe
    if has_production:
        print("Evaluando modelo en producción...")
        prod_predictions = prod_model.predict(X_test)
        prod_metrics = {
            'accuracy': accuracy_score(y_test, prod_predictions),
            'f1_score': f1_score(y_test, prod_predictions),
            'precision': precision_score(y_test, prod_predictions),
            'recall': recall_score(y_test, prod_predictions)
        }
    else:
        prod_metrics = {
            'accuracy': 0,
            'f1_score': 0,
            'precision': 0,
            'recall': 0
        }
    
    # Comparar métricas (usamos F1 como métrica principal)
    improvement = new_metrics['f1_score'] - prod_metrics['f1_score']
    should_promote = improvement > -0.02  # Promover si no empeora más de 2%
    
    # Preparar resultado
    result = {
        'new_model': {
            'path': new_model_path,
            'metrics': new_metrics
        },
        'production_model': {
            'path': production_model_path if has_production else None,
            'metrics': prod_metrics
        },
        'improvement': {
            'f1_score': improvement,
            'accuracy': new_metrics['accuracy'] - prod_metrics['accuracy']
        },
        'should_promote': should_promote,
        'reason': f"F1 improvement: {improvement:.4f}"
    }
    
    # Imprimir resumen
    print("\n=== Resultados de la Comparación ===")
    print(f"Modelo Nuevo - F1: {new_metrics['f1_score']:.4f}, Accuracy: {new_metrics['accuracy']:.4f}")
    if has_production:
        print(f"Modelo Producción - F1: {prod_metrics['f1_score']:.4f}, Accuracy: {prod_metrics['accuracy']:.4f}")
    print(f"Mejora en F1: {improvement:.4f}")
    print(f"Decisión: {'✅ PROMOVER' if should_promote else '❌ NO PROMOVER'}")
    
    # Guardar resultado
    Path("monitoring_data/airflow").mkdir(parents=True, exist_ok=True)
    with open("monitoring_data/airflow/model_comparison.json", "w") as f:
        json.dump(result, f, indent=2)
    
    return result


def promote_model(new_model_path: str, production_path: str = "models/production.pkl"):
    """
    Promueve un modelo nuevo a producción.
    """
    import shutil
    from datetime import datetime
    
    # Crear backup del modelo actual si existe
    if os.path.exists(production_path):
        backup_path = f"models/backup_{datetime.now():%Y%m%d_%H%M%S}.pkl"
        shutil.copy2(production_path, backup_path)
        print(f"Backup creado: {backup_path}")
    
    # Copiar nuevo modelo a producción
    shutil.copy2(new_model_path, production_path)
    print(f"✅ Modelo promovido a producción: {production_path}")
    
    # Guardar metadata
    metadata = {
        'promoted_from': new_model_path,
        'promoted_at': datetime.now().isoformat(),
        'production_path': production_path
    }
    
    with open("models/production_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Comparar modelos')
    parser.add_argument('--new-model', type=str, required=True,
                      help='Path al nuevo modelo')
    parser.add_argument('--production-model', type=str, 
                      default='models/production.pkl',
                      help='Path al modelo en producción')
    parser.add_argument('--promote', action='store_true',
                      help='Promover automáticamente si es mejor')
    
    args = parser.parse_args()
    
    # Comparar modelos
    result = compare_models(args.new_model, args.production_model)
    
    # Promover si se especifica y es mejor
    if args.promote and result['should_promote']:
        promote_model(args.new_model, args.production_model)
    
    # Exit code para Airflow (0 = éxito, 1 = no promover)
    sys.exit(0 if result['should_promote'] else 1)