import os
import dagshub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import mlflow
import mlflow.sklearn

# Inisialisasi DagsHub (Ganti dengan username DagsHub Anda)
dagshub.init(repo_owner="svsf36-dev", repo_name="Eksperimen_SML_Nurul-Amanda", mlflow=True)

data_path = "dataset_clean.csv"
df = pd.read_csv(data_path)

X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Konfigurasi Hyperparameter Tuning Manual
n_estimators = 250
max_depth = 15
random_state = 42

mlflow.set_experiment("Heart_Disease_Manual_Tuning")

with mlflow.start_run():
    # Model Training
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # Kalkulasi Metrik
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # 1. Manual Logging: Parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", random_state)
    
    # 2. Manual Logging: Metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    
    # 3. Pembuatan Artefak Tambahan 1: Grafik Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Sehat', 'Sakit'], yticklabels=['Sehat', 'Sakit'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    confusion_matrix_path = "confusion_matrix.png"
    plt.savefig(confusion_matrix_path)
    plt.close()
    
    # 4. Pembuatan Artefak Tambahan 2: Text Report Hasil Klasifikasi
    report_text = classification_report(y_test, y_pred)
    report_path = "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
        
    # 5. Manual Logging: Unggah Artefak Tambahan ke DagsHub
    mlflow.log_artifact(confusion_matrix_path)
    mlflow.log_artifact(report_path)
    
    # 6. Logging Model Utama
    mlflow.sklearn.log_model(sk_model=model, artifact_path="model_tuning")
    
    # Hapus file lokal sementara setelah diunggah ke MLflow
    if os.path.exists(confusion_matrix_path):
        os.remove(confusion_matrix_path)
    if os.path.exists(report_path):
        os.remove(report_path)
        
    print("Pelatihan selesai. Parameter, metrik, dan 2 artefak berhasil diunggah ke DagsHub!")