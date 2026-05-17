import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

if __name__ == "__main__":
    df = pd.read_csv('../dataset_preprocessing.csv')
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        mlflow.sklearn.log_model(model, "outputs")
        print("Model training via MLproject sukses dijalankan!")
