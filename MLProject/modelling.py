import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import os

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.abspath(os.path.join(base_dir, "..", "dataset_preprocessing.csv"))
    
    print(f"Mencoba membaca dataset dari jalur absolut: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Waduh, file tidak ditemukan di jalur: {csv_path}")

    df = pd.read_csv(csv_path)
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        mlflow.sklearn.log_model(model, "outputs")
        print("Model training via MLproject sukses dijalankan!")
