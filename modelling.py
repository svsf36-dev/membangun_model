import dagshub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

dagshub.init(repo_owner="svsf36-dev", repo_name="Eksperimen_SML_Nurul-Amanda", mlflow=True)

data_path = "dataset_clean.csv"
df = pd.read_csv(data_path)

X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

input_example = X_train.iloc[[0]]

n_estimators = 505
max_depth = 37

mlflow.set_experiment("Heart_Disease_Autolog")

with mlflow.start_run():
    mlflow.autolog()
    
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )
    
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    print("Pelatihan model dan logging autolog ke DagsHub selesai!")
    print(f"Akurasi Model: {accuracy:.4f}")