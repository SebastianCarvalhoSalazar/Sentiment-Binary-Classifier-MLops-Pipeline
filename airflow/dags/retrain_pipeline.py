"""
DAG de Airflow para el pipeline de reentrenamiento automático.
Este pipeline:
1. Verifica drift en el modelo
2. Si hay drift, reentrena el modelo
3. Compara el nuevo modelo con el actual
4. Promueve el modelo si es mejor
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
import json
import os

# Configuración global del proyecto
# PROJECT_DIR = "/Users/sebatiancarvalho/Desktop/Repositorios/SCSProjects/movie-sentiment-mlops-V3"

from dotenv import load_dotenv
load_dotenv()
PROJECT_DIR = os.environ.get('MLOPS_PROJECT_DIR', '/app')

# Configuración por defecto del DAG
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Definir el DAG
dag = DAG(
    'movie_sentiment_retrain',
    default_args=default_args,
    description='Pipeline de reentrenamiento automático para modelo de sentimientos',
    schedule='@daily',
    catchup=False,
)

def check_drift(**context):
    """
    Verifica si hay drift y decide si reentrenar.
    """
    import subprocess
    import json
    from pathlib import Path
    from dotenv import load_dotenv
    
    # Cargar variables de entorno desde el proyecto
    env_path = os.path.join(PROJECT_DIR, '.env')
    load_dotenv(env_path)
    
    print("="*50)
    print("VERIFICACIÓN DE DRIFT")
    print("="*50)
    print(f"Proyecto en: {PROJECT_DIR}")
    print(f"DRIFT_THRESHOLD: {os.getenv('DRIFT_THRESHOLD')}")
    
    # Ejecutar el monitor de drift con rutas absolutas
    drift_script = os.path.join(PROJECT_DIR, 'monitoring', 'drift_monitor.py')
    
    env = os.environ.copy()
    result = subprocess.run(
        ['python', drift_script, '--airflow'],
        capture_output=True,
        text=True,
        cwd=PROJECT_DIR,  # Ejecutar EN el directorio del proyecto
        env=env
    )
    
    print(f"\n--- Resultado del subprocess ---")
    print(f"Código de retorno: {result.returncode}")
    print(f"Stdout: {result.stdout[:500]}")
    if result.stderr:
        print(f"Stderr: {result.stderr[:500]}")
    
    # Verificar el archivo JSON con ruta absoluta
    json_path = os.path.join(PROJECT_DIR, 'monitoring_data', 'airflow', 'last_drift_check.json')
    
    if not os.path.exists(json_path):
        print(f"\nERROR: No se encontró {json_path}")
        monitoring_dir = os.path.join(PROJECT_DIR, 'monitoring_data', 'airflow')
        Path(monitoring_dir).mkdir(parents=True, exist_ok=True)
        print("Directorio creado. Forzando reentrenamiento.")
        
        context['task_instance'].xcom_push(key='needs_retrain', value=True)
        return 'retrain_model'
    
    # Leer y procesar el JSON
    try:
        with open(json_path, 'r') as f:
            content = f.read()
            print(f"\n--- Contenido del JSON ---")
            print(content[:200])
            
        drift_result = json.loads(content)
        
        needs_retrain = drift_result.get('needs_retrain', False)
        drift_metrics = drift_result.get('drift_metrics', {})
        
        print(f"\n--- Análisis del Drift ---")
        print(f"needs_retrain: {needs_retrain}")
        print(f"dataset_drift: {drift_metrics.get('dataset_drift', 'N/A')}")
        
        # Guardar en XCom
        context['task_instance'].xcom_push(key='needs_retrain', value=needs_retrain)
        context['task_instance'].xcom_push(key='drift_metrics', value=drift_metrics)
        
        if needs_retrain:
            print(f"\n✅ DECISIÓN: Drift detectado - REENTRENAMIENTO")
            return 'retrain_model'
        else:
            print(f"\n⏭️ DECISIÓN: Sin drift - SKIP")
            return 'skip_retrain'
            
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        context['task_instance'].xcom_push(key='needs_retrain', value=True)
        return 'retrain_model'

def retrain_model(**context):
    """
    Reentrena el modelo con los datos más recientes.
    """
    import subprocess
    from datetime import datetime
    
    print(f"Proyecto en: {PROJECT_DIR}")
    print(f"Directorio de trabajo actual: {os.getcwd()}")
    
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # new_model_name = f'logistic_regression_retrained_{timestamp}.pkl'
    new_model_name = f'logistic_regression_bayesian.pkl'
    
    print(f"Iniciando reentrenamiento: {new_model_name}")
    
    # Ejecutar entrenamiento con ruta absoluta
    train_script = os.path.join(PROJECT_DIR, 'src', 'train_bayesian.py')
    
    result = subprocess.run(
        ['python', train_script, '--model', 'logistic_regression'],
        capture_output=True,
        text=True,
        cwd=PROJECT_DIR  # Ejecutar EN el directorio del proyecto
    )
    
    print(f"Código de retorno: {result.returncode}")
    print(f"Salida: {result.stdout[:1000]}")
    
    if result.returncode == 0:
        # Rutas absolutas para los modelos
        old_model = os.path.join(PROJECT_DIR, 'models', 'logistic_regression_bayesian.pkl')
        new_model = os.path.join(PROJECT_DIR, 'models', new_model_name)
        
        if os.path.exists(old_model):
            os.rename(old_model, new_model)
            print(f"✅ Modelo renombrado a: {new_model_name}")
        else:
            print(f"ERROR: No se encontró {old_model}")
            raise FileNotFoundError(f"Modelo no encontrado: {old_model}")
        
        context['task_instance'].xcom_push(key='new_model', value=new_model_name)
        return new_model_name
    else:
        raise Exception(f"Error en reentrenamiento: {result.stderr}")

def compare_and_promote(**context):
    """
    Compara el nuevo modelo con el de producción y lo promueve si es mejor.
    """
    import subprocess
    
    print(f"Proyecto en: {PROJECT_DIR}")
    
    # Obtener el nombre del nuevo modelo de XCom
    new_model = context['task_instance'].xcom_pull(key='new_model')
    
    if not new_model:
        print("No hay nuevo modelo para comparar")
        return False
    
    new_model_path = os.path.join('models', new_model)
    production_path = 'models/production.pkl'
    
    print(f"Comparando modelos:")
    print(f"  Nuevo: {new_model_path}")
    print(f"  Producción: {production_path}")
    
    # Ejecutar comparación con ruta absoluta
    comparator_script = os.path.join(PROJECT_DIR, 'src', 'model_comparator.py')
    
    result = subprocess.run(
        [
            'python', comparator_script,
            '--new-model', new_model_path,
            '--production-model', production_path,
            '--promote'
        ],
        capture_output=True,
        text=True,
        cwd=PROJECT_DIR
    )
    
    print(f"Resultado: {result.stdout[:1000]}")
    
    # Leer resultado con ruta absoluta
    comparison_file = os.path.join(PROJECT_DIR, 'monitoring_data', 'airflow', 'model_comparison.json')
    
    try:
        with open(comparison_file, 'r') as f:
            comparison = json.load(f)
        
        if comparison['should_promote']:
            print("✅ Modelo promovido a producción!")
        else:
            print("⚠️ Modelo actual mantiene mejor performance")
        
        return comparison['should_promote']
    
    except Exception as e:
        print(f"Error leyendo comparación: {e}")
        return False

def notify_results(**context):
    """
    Notifica los resultados del pipeline.
    """
    from datetime import datetime
    
    print("=" * 50)
    print("📊 RESUMEN DEL PIPELINE")
    print("=" * 50)
    print(f"Fecha: {datetime.now()}")
    
    needs_retrain = context['task_instance'].xcom_pull(key='needs_retrain')
    drift_metrics = context['task_instance'].xcom_pull(key='drift_metrics')
    
    print(f"Drift detectado: {'Sí' if needs_retrain else 'No'}")
    
    if drift_metrics:
        print(f"  - Dataset drift: {drift_metrics.get('dataset_drift', 'N/A')}")
    
    if needs_retrain:
        new_model = context['task_instance'].xcom_pull(key='new_model')
        
        if new_model:
            print(f"\n📦 Nuevo modelo: {new_model}")
            
            comparison_file = os.path.join(PROJECT_DIR, 'monitoring_data', 'airflow', 'model_comparison.json')
            if os.path.exists(comparison_file):
                try:
                    with open(comparison_file, 'r') as f:
                        comparison = json.load(f)
                    
                    if comparison.get('should_promote'):
                        print("✅ Modelo PROMOVIDO")
                        print(f"  - Mejora F1: {comparison['improvement']['f1_score']:.4f}")
                    else:
                        print("⚠️ Modelo NO promovido")
                except Exception as e:
                    print(f"⚠️ Error leyendo comparación: {e}")
            else:
                print("⚠️ No se encontró archivo de comparación")
        else:
            print("⚠️ No se completó el reentrenamiento")
    else:
        print("\n✅ Sistema estable - No requiere reentrenamiento")
    
    print("=" * 50)
    return True

# Definir las tareas del DAG
check_drift_task = BranchPythonOperator(
    task_id='check_drift',
    python_callable=check_drift,
    dag=dag,
)

retrain_task = PythonOperator(
    task_id='retrain_model',
    python_callable=retrain_model,
    dag=dag,
)

compare_task = PythonOperator(
    task_id='compare_and_promote',
    python_callable=compare_and_promote,
    dag=dag,
)

skip_task = EmptyOperator(
    task_id='skip_retrain',
    dag=dag,
)

notify_task = PythonOperator(
    task_id='notify_results',
    python_callable=notify_results,
    trigger_rule='none_failed_min_one_success',
    dag=dag,
)

# Definir el flujo
check_drift_task >> [retrain_task, skip_task]
retrain_task >> compare_task >> notify_task
skip_task >> notify_task