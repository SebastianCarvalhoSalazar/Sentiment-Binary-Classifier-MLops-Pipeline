"""
Sistema de monitoreo de drift y performance del modelo usando Evidently.
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import joblib
from pathlib import Path

# from evidently import ColumnMapping
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, ClassificationPreset
from evidently.metrics import *
from evidently.test_suite import TestSuite
from evidently.tests import *

import mlflow
from dotenv import load_dotenv

import warnings
# Ignorar todos los warnings
warnings.filterwarnings("ignore")

# Cargar configuración
load_dotenv()
MONITORING_PATH = Path("monitoring_data")
MONITORING_PATH.mkdir(exist_ok=True)
REFERENCE_DATA_PATH = MONITORING_PATH / "reference_data.parquet"
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.15"))


class ModelMonitor:
    """Monitor de drift y performance para el modelo de sentimientos."""

    def __init__(self, model_path: str, reference_data_path: Optional[str] = None):
        """
        Inicializa el monitor.

        Args:
            model_path: Ruta al modelo entrenado
            reference_data_path: Ruta a los datos de referencia
        """
        self.model = joblib.load(model_path)
        self.reference_data_path = reference_data_path or REFERENCE_DATA_PATH
        print("reference_data_path")
        self.reference_data = None

        if os.path.exists(self.reference_data_path):
            self.reference_data = pd.read_parquet(self.reference_data_path)
            print(
                f"Datos de referencia cargados: {len(self.reference_data)} muestras")
        else:
            print(
                " No se encontraron datos de referencia. Ejecuta create_reference_data() primero.")

    def create_reference_data(self, X_test: pd.Series, y_test: pd.Series):
        """
        Crea el dataset de referencia para comparación.

        Args:
            X_test: Textos de test
            y_test: Etiquetas reales
        """
        print("Creando datos de referencia...")

        # Generar predicciones y características
        predictions = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)

        # Extraer características del texto
        text_features = self._extract_text_features(X_test)

        # Crear DataFrame de referencia
        reference_df = pd.DataFrame({
            'text': X_test,
            'text_length': text_features['length'],
            'word_count': text_features['word_count'],
            'exclamation_count': text_features['exclamation_count'],
            'question_count': text_features['question_count'],
            'uppercase_ratio': text_features['uppercase_ratio'],
            'actual_label': y_test,
            'predicted_label': predictions,
            'positive_probability': probabilities[:, 1],
            'negative_probability': probabilities[:, 0],
            'max_probability': probabilities.max(axis=1),
            'timestamp': datetime.now()
        })

        # Guardar
        reference_df.to_parquet(self.reference_data_path)
        self.reference_data = reference_df
        print(f"✓ Datos de referencia guardados: {len(reference_df)} muestras")

        return reference_df

    def _extract_text_features(self, texts: pd.Series) -> Dict[str, List]:
        """Extrae características del texto para monitoreo."""
        features = {
            'length': [],
            'word_count': [],
            'exclamation_count': [],
            'question_count': [],
            'uppercase_ratio': []
        }

        for text in texts:
            features['length'].append(len(text))
            features['word_count'].append(len(text.split()))
            features['exclamation_count'].append(text.count('!'))
            features['question_count'].append(text.count('?'))

            uppercase_chars = sum(1 for c in text if c.isupper())
            total_chars = len([c for c in text if c.isalpha()])
            features['uppercase_ratio'].append(
                uppercase_chars / total_chars if total_chars > 0 else 0
            )

        return features

    def collect_production_data(self, texts: List[str], predictions: List[int],
                                probabilities: np.ndarray) -> pd.DataFrame:
        """
        Recolecta datos de producción para monitoreo.

        Args:
            texts: Textos procesados
            predictions: Predicciones del modelo
            probabilities: Probabilidades de cada clase

        Returns:
            DataFrame con los datos de producción
        """
        text_features = self._extract_text_features(pd.Series(texts))

        production_df = pd.DataFrame({
            'text': texts,
            'text_length': text_features['length'],
            'word_count': text_features['word_count'],
            'exclamation_count': text_features['exclamation_count'],
            'question_count': text_features['question_count'],
            'uppercase_ratio': text_features['uppercase_ratio'],
            'predicted_label': predictions,
            'positive_probability': probabilities[:, 1],
            'negative_probability': probabilities[:, 0],
            'max_probability': probabilities.max(axis=1),
            'timestamp': datetime.now()
        })

        # Guardar en archivo de producción
        prod_file = MONITORING_PATH / \
            f"production_data_{datetime.now():%Y%m%d_%H%M%S}.parquet"
        production_df.to_parquet(prod_file)

        return production_df

    def generate_drift_report(self, production_data: pd.DataFrame) -> Tuple[Report, Dict[str, float]]:
        """
        Genera reporte de drift comparando con datos de referencia.

        Args:
            production_data: Datos de producción recientes

        Returns:
            Tuple de (Report de Evidently, métricas de drift)
        """
        if self.reference_data is None:
            raise ValueError(
                "No hay datos de referencia. Ejecuta create_reference_data() primero.")

        # Columnas numéricas para análisis
        numerical_features = [
            'text_length', 'word_count', 'exclamation_count',
            'question_count', 'uppercase_ratio', 'max_probability'
        ]

        # Configurar column mapping
        column_mapping = ColumnMapping(
            target='predicted_label',
            prediction='predicted_label',
            numerical_features=numerical_features
        )

        # Crear reporte de drift
        drift_report = Report(metrics=[
            DataDriftPreset(),
            TargetDriftPreset(),
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
            ColumnDriftMetric(column_name='text_length'),
            ColumnDriftMetric(column_name='word_count'),
            ColumnDriftMetric(column_name='max_probability'),
            ColumnDistributionMetric(column_name='predicted_label'),
            ColumnQuantileMetric(column_name='max_probability', quantile=0.5)
        ])

        # Ejecutar reporte
        drift_report.run(
            reference_data=self.reference_data[numerical_features + [
                'predicted_label']],
            current_data=production_data[numerical_features +
                                         ['predicted_label']],
            column_mapping=column_mapping
        )

        # Extraer métricas clave
        report_dict = drift_report.as_dict()

        # ============= PRINTS PARA DEBUG =============
        print("\n" + "="*60)
        print("📊 ANÁLISIS DETALLADO DE DRIFT")
        print("="*60)
        
        # Analizar cada métrica en el reporte
        for i, metric in enumerate(report_dict.get('metrics', [])):
            metric_type = str(metric.get('metric', ''))
            
            # Dataset Drift General
            if 'DatasetDriftMetric' in metric_type:
                result = metric.get('result', {})
                drift_share = result.get('drift_share', 0.0)
                number_of_drifted_columns = result.get('number_of_drifted_columns', 0)
                number_of_columns = result.get('number_of_columns', 0)
                dataset_drift = result.get('dataset_drift', False)
                
                print(f"\n🎯 DATASET DRIFT:")
                print(f"   - Drift detectado: {'SÍ ⚠️' if dataset_drift else 'NO ✅'}")
                print(f"   - Porcentaje de features con drift: {drift_share:.2%}")
                print(f"   - Features con drift: {number_of_drifted_columns}/{number_of_columns}")
                print(f"   - Umbral configurado: {DRIFT_THRESHOLD:.2%}")
                print(f"   - ¿Supera umbral?: {'SÍ ⚠️' if drift_share > DRIFT_THRESHOLD else 'NO ✅'}")
            
            # Drift por columna específica
            elif 'ColumnDriftMetric' in metric_type:
                result = metric.get('result', {})
                column_name = result.get('column_name', '')
                drift_detected = result.get('drift_detected', False)
                drift_score = result.get('drift_score', 0.0)
                stattest_name = result.get('stattest_name', '')
                
                if column_name:  # Solo si hay nombre de columna
                    print(f"\n📈 DRIFT EN '{column_name}':")
                    print(f"   - Drift detectado: {'SÍ ⚠️' if drift_detected else 'NO ✅'}")
                    print(f"   - Score del test: {drift_score:.4f}")
                    print(f"   - Test estadístico usado: {stattest_name}")
                    print(f"   - P-value < 0.05: {'SÍ' if drift_detected else 'NO'}")
            
            # Target/Prediction Drift
            elif 'TargetDriftPreset' in metric_type or 'target_drift' in metric_type.lower():
                # Los presets contienen múltiples métricas
                if 'result' in metric:
                    result = metric['result']
                    if isinstance(result, dict) and 'drift_detected' in result:
                        print(f"\n🎯 PREDICTION DRIFT:")
                        print(f"   - Drift en predicciones: {'SÍ ⚠️' if result['drift_detected'] else 'NO ✅'}")
        
        # Comparación de distribuciones de referencia vs producción
        print(f"\n📊 COMPARACIÓN DE DISTRIBUCIONES:")
        print(f"   Referencia (n={len(self.reference_data)})")
        print(f"   - Predicciones positivas: {(self.reference_data['predicted_label']==1).mean():.2%}")
        print(f"   - Confianza promedio: {self.reference_data['max_probability'].mean():.3f}")
        print(f"   - Longitud promedio texto: {self.reference_data['text_length'].mean():.0f}")
        
        print(f"\n   Producción (n={len(production_data)})")  
        print(f"   - Predicciones positivas: {(production_data['predicted_label']==1).mean():.2%}")
        print(f"   - Confianza promedio: {production_data['max_probability'].mean():.3f}")
        print(f"   - Longitud promedio texto: {production_data['text_length'].mean():.0f}")
        
        # ============= FIN DE PRINTS DEBUG =============        

        # Extraer métricas de forma más robusta
        drift_metrics = {
            'dataset_drift': 0.0,
            'prediction_drift': False,
            'text_length_drift': False,
            'confidence_drift': False,
            'timestamp': datetime.now().isoformat()
        }

        # Buscar las métricas en el reporte sin asumir posiciones
        try:
            for metric in report_dict.get('metrics', []):
                if 'DatasetDriftMetric' in str(metric.get('metric')):
                    drift_metrics['dataset_drift'] = metric.get(
                        'result', {}).get('drift_share', 0.0)
                elif 'ColumnDriftMetric' in str(metric.get('metric')):
                    col_name = metric.get('result', {}).get('column_name', '')
                    if 'text_length' in col_name:
                        drift_metrics['text_length_drift'] = metric.get(
                            'result', {}).get('drift_detected', False)
                    elif 'max_probability' in col_name:
                        drift_metrics['confidence_drift'] = metric.get(
                            'result', {}).get('drift_detected', False)
        except Exception as e:
            print(
                f"Advertencia: No se pudieron extraer todas las métricas de drift: {e}")

        # Guardar reporte HTML
        report_path = MONITORING_PATH / \
            f"drift_report_{datetime.now():%Y%m%d_%H%M%S}.html"
        drift_report.save_html(str(report_path))
        print(f"✓ Reporte de drift guardado en: {report_path}")

        return drift_report, drift_metrics

    def generate_performance_report(self, production_data: pd.DataFrame,
                                    true_labels: Optional[pd.Series] = None) -> Report:
        """
        Genera reporte de performance del modelo.

        Args:
            production_data: Datos de producción
            true_labels: Etiquetas reales (si están disponibles)

        Returns:
            Report de performance
        """
        if true_labels is not None:
            production_data['actual_label'] = true_labels

            performance_report = Report(metrics=[
                ClassificationPreset(),
                ClassificationQualityMetric(),
                ClassificationConfusionMatrix(),
                ClassificationQualityByClass()
            ])

            column_mapping = ColumnMapping(
                target='actual_label',
                prediction='predicted_label',
                numerical_features=[
                    'positive_probability', 'negative_probability']
            )

            performance_report.run(
                reference_data=None,
                current_data=production_data,
                column_mapping=column_mapping
            )
        else:
            # Sin etiquetas reales, solo podemos analizar distribuciones
            performance_report = Report(metrics=[
                ColumnDistributionMetric(column_name='predicted_label'),
                ColumnDistributionMetric(column_name='max_probability'),
                ColumnQuantileMetric(
                    column_name='max_probability', quantile=0.5),
                ColumnQuantileMetric(
                    column_name='max_probability', quantile=0.95)
            ])

            performance_report.run(
                reference_data=None,
                current_data=production_data
            )

        # Guardar reporte
        report_path = MONITORING_PATH / \
            f"performance_report_{datetime.now():%Y%m%d_%H%M%S}.html"
        performance_report.save_html(str(report_path))
        print(f"✓ Reporte de performance guardado en: {report_path}")

        return performance_report

    def should_retrain(self, drift_metrics: Dict[str, float]) -> bool:
        """
        Determina si el modelo necesita reentrenamiento basado en las métricas.

        Args:
            drift_metrics: Métricas de drift

        Returns:
            True si se debe reentrenar
        """
        # Criterios para reentrenamiento
        conditions = [
            drift_metrics['dataset_drift'] > DRIFT_THRESHOLD,
            drift_metrics['prediction_drift'],
            drift_metrics['confidence_drift']
        ]

        should_retrain = any(conditions)

        if should_retrain:
            print(
                "⚠️  ALERTA: Se detectó drift significativo. Reentrenamiento recomendado.")
            print(f"  - Dataset drift: {drift_metrics['dataset_drift']:.2%}")
            print(f"  - Prediction drift: {drift_metrics['prediction_drift']}")
            print(f"  - Confidence drift: {drift_metrics['confidence_drift']}")

        return should_retrain


def main():
    """Función principal para testing del monitor."""
    # Cargar modelo
    # model_path = "models/logistic_regression_bayesian.pkl"

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
        print(" No se encontró el modelo. Entrena primero con train_bayesian.py")
        return

    monitor = ModelMonitor(model_path)

    # Simular datos de producción
    sample_texts = [
        "This movie was terrible!",
        "Amazing film, loved every minute",
        "Boring and predictable plot",
        "Best movie I've seen this year!",
        "Waste of time and money"
    ]

    # Generar predicciones
    predictions = monitor.model.predict(sample_texts)
    probabilities = monitor.model.predict_proba(sample_texts)

    # Recolectar datos
    prod_data = monitor.collect_production_data(
        sample_texts, predictions, probabilities)

    # Si hay datos de referencia, generar reportes
    if monitor.reference_data is not None:
        drift_report, drift_metrics = monitor.generate_drift_report(prod_data)
        monitor.generate_performance_report(prod_data)

        # Verificar si necesita reentrenamiento
        needs_retrain = monitor.should_retrain(drift_metrics)
        print(f"\n¿Necesita reentrenamiento? {needs_retrain}")


def check_drift_for_airflow():
    """Función específica para ser llamada desde Airflow."""
    import json
    from pathlib import Path

    # Usar el modelo más reciente
    import glob
    model_files = glob.glob("models/*bayesian*.pkl")
    if not model_files:
        print("ERROR: No model found")
        return False

    model_path = model_files[0]
    monitor = ModelMonitor(model_path)

    # Simular datos de producción (en producción real, estos vendrían de tu base de datos)
    # sample_texts = [
    #     "This movie was terrible!",
    #     "Amazing film, loved every minute",
    #     "Boring and predictable plot",
    #     "Best movie I've seen this year!",
    #     "Waste of time and money",
    #     "Not bad but could be better",
    #     "Absolutely fantastic cinematography",
    #     "Poor acting ruined the whole film"
    # ]
    
    sample_texts = [
        "This movie was absolutely fantastic! Best film of the year!",
        "Terrible waste of time. Bad acting and boring plot.",
        "Not bad but not great either. Just okay.",
        "Amazing cinematography and great performances!",
        "I fell asleep halfway through. So boring.",
        "One of the worst movies I've ever seen.",
        "Brilliant storytelling and direction!",
        "Completely predictable and uninspiring."
    ]    

    predictions = monitor.model.predict(sample_texts)
    probabilities = monitor.model.predict_proba(sample_texts)

    # Recolectar datos
    prod_data = monitor.collect_production_data(
        sample_texts, predictions, probabilities
    )

    # Verificar si hay datos de referencia
    if monitor.reference_data is None:
        print("WARNING: No reference data found")
        return False
    try:
        drift_report, drift_metrics = monitor.generate_drift_report(prod_data)
        needs_retrain = monitor.should_retrain(drift_metrics)
        
        # Limpiar métricas de valores no serializables
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [clean_for_json(v) for v in obj]
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.floating, float)):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return float(obj)
            else:
                return obj
        
        result = {
            "needs_retrain": bool(needs_retrain),
            "drift_metrics": clean_for_json(drift_metrics),
            "timestamp": datetime.now().isoformat()
        }
        
        # Escribir con manejo robusto
        Path("monitoring_data/airflow").mkdir(parents=True, exist_ok=True)
        
        # Primero escribir a archivo temporal
        temp_file = "monitoring_data/airflow/last_drift_check.tmp"
        with open(temp_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        
        # Luego mover atómicamente
        os.rename(temp_file, "monitoring_data/airflow/last_drift_check.json")
        
        print(f"JSON successfully written: {result}")

        return bool(needs_retrain)
    
    except Exception as e:
        print(f"ERROR writing JSON: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--airflow":
        # Modo Airflow
        print("Use Airflow!")
        needs_retrain = check_drift_for_airflow()
        sys.exit(0 if not needs_retrain else 1)
    else:
        # Modo normal
        print("Not Use Airflow!")
        main()
