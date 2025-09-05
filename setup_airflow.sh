#!/bin/bash
# Script para configurar Airflow en el proyecto

echo " Configurando Apache Airflow para Movie Sentiment MLOps"
echo "=========================================="

# 1. Establecer variables de entorno
export AIRFLOW_HOME=$(pwd)/airflow
export PROJECT_HOME=$(pwd)

echo " AIRFLOW_HOME: $AIRFLOW_HOME"
echo " PROJECT_HOME: $PROJECT_HOME"

# 2. Crear directorios necesarios
echo ""
echo " Creando directorios..."
mkdir -p $AIRFLOW_HOME
mkdir -p $AIRFLOW_HOME/dags
mkdir -p $AIRFLOW_HOME/logs
mkdir -p $AIRFLOW_HOME/plugins
mkdir -p monitoring_data/airflow

# 3. Inicializar la base de datos de Airflow
echo ""
echo " Inicializando base de datos de Airflow..."
airflow db init

# 4. Copiar el DAG al directorio de Airflow
echo ""
echo " Copiando DAG al directorio de Airflow..."
cp dags/retrain_pipeline.py $AIRFLOW_HOME/dags/

# 5. Crear usuario admin (opcional, para acceder a la UI)
echo ""
echo " Creando usuario admin..."
airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

# 6. Actualizar configuración de Airflow para el proyecto
echo ""
echo " Configurando Airflow..."

# Crear archivo de configuración personalizado
cat > $AIRFLOW_HOME/airflow.cfg.custom << EOF
# Configuración personalizada para Movie Sentiment MLOps

# Desactivar ejemplos
load_examples = False

# Configurar executor (para desarrollo usamos Sequential)
executor = SequentialExecutor

# Configurar paths
dags_folder = $AIRFLOW_HOME/dags
base_log_folder = $AIRFLOW_HOME/logs

# Configurar scheduler
scheduler_heartbeat_sec = 30
EOF

echo ""
echo " Configuración completada!"
echo ""
echo " INSTRUCCIONES DE USO:"
echo "========================"
echo ""
echo "1. Para iniciar Airflow en modo standalone (desarrollo):"
echo "   export AIRFLOW_HOME=$AIRFLOW_HOME"
echo "   airflow standalone"
echo ""
echo "2. O iniciar servicios por separado:"
echo "   # Terminal 1 - Scheduler:"
echo "   export AIRFLOW_HOME=$AIRFLOW_HOME"
echo "   airflow scheduler"
echo ""
echo "   # Terminal 2 - Webserver:"
echo "   export AIRFLOW_HOME=$AIRFLOW_HOME"
echo "   airflow webserver --port 8080"
echo ""
echo "3. Acceder a la interfaz web:"
echo "   http://localhost:8080"
echo "   Usuario: admin"
echo "   Password: admin"
echo ""
echo "4. Para probar el DAG manualmente:"
echo "   export AIRFLOW_HOME=$AIRFLOW_HOME"
echo "   airflow dags test movie_sentiment_retrain $(date +%Y-%m-%d)"
echo ""