#!/bin/bash

echo " Configurando Movie Sentiment MLOps..."

# Crear directorios necesarios
echo " Creando estructura de directorios..."
mkdir -p data models mlruns notebooks/figures

# Copiar .env.example a .env si no existe
if [ ! -f .env ]; then
    echo " Creando archivo .env..."
    cat > .env << EOF
# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# API
API_HOST=0.0.0.0
API_PORT=8000

# Model
MODEL_NAME=movie_sentiment_classifier
MODEL_STAGE=Production
EOF
fi

# Crear ambiente virtual si no existe
if [ ! -d "venv" ]; then
    echo " Creando ambiente virtual..."
    python -m venv venv
fi

# Activar ambiente virtual
echo " Activando ambiente virtual..."
source venv/bin/activate || source venv/Scripts/activate

# Instalar dependencias
echo " Instalando dependencias..."
pip install --upgrade pip
pip install -r requirements.txt

# Descargar datos NLTK
echo " Descargando recursos NLTK..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Entrenar modelo inicial
echo " Entrenando modelo inicial..."
python src/train.py --model logistic_regression

echo " Setup completado!"
echo ""
echo " Próximos pasos:"
echo "1. Iniciar MLflow: mlflow ui"
echo "2. Iniciar API: python app/api.py"
echo "3. Iniciar UI: streamlit run app/streamlit_app.py"
echo ""
echo "O usar Docker: docker-compose up --build"