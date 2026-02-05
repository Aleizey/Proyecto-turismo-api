import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Cargar datos (Asegúrate de tener el CSV en la carpeta o cárgalo igual que antes)
# Para este ejemplo, simulamos que 'df' ya está cargado y procesado
df = pd.read_csv('data/PRECIOS_ALOJAMIENTOS_HOTELEROS.csv') 

# --- PASO PREVIO: Procesamiento rápido ---
le = LabelEncoder()
cols_cat = ['CCAA', 'PROVINCIA', 'CATEGORIA_ALOJAMIENTO', 'PERIODO_ANTELACION']
for col in cols_cat:
    df[col + '_CODE'] = le.fit_transform(df[col].astype(str))

features = ['AÑO', 'MES', 'PERIODO_ANTELACION_CODE', 'CCAA_CODE', 'PROVINCIA_CODE', 'CATEGORIA_ALOJAMIENTO_CODE']
X = df[features]
y = (df['PRECIO_CHECK-IN_ENTRE_SEMANA'] > df['PRECIO_CHECK-IN_ENTRE_SEMANA'].median()).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- MLFLOW ---
mlflow.set_experiment("Auditoria_Hoteles")

with mlflow.start_run() as run:
    # Parámetros del modelo para evitar overfitting
    params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}
    rf = RandomForestClassifier(**params)
    rf.fit(X_train, y_train)
    
    # Métricas
    accuracy = rf.score(X_test, y_test)
    
    # Registro en MLflow
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", accuracy)
    
    # Guardamos el modelo con una "firma" (input example) para que la API sepa qué recibe
    mlflow.sklearn.log_model(rf, "modelo_final")
    
    print(f"Entrenamiento completado.")
    print(f"ID de ejecución (RUN_ID): {run.info.run_id}")
    print(f"Accuracy: {accuracy:.4f}")