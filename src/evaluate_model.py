import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from src.data_preprocessing import load_dataset, FEATURES

def main():
    df = load_dataset()
    X, y = df[FEATURES], df["species"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    bundle = joblib.load("models/classification_model.pkl")
    X_test_s = bundle["scaler"].transform(X_test)
    pred = bundle["model"].predict(X_test_s)
    acc = accuracy_score(y_test, pred)
    report = classification_report(y_test, pred, zero_division=0)
    os.makedirs("results", exist_ok=True)
    with open("results/model_results.txt", "w", encoding="utf-8") as f:
        f.write(f"Best K: {bundle['best_k']}\nTest accuracy: {acc:.4f}\n\n{report}")
    cm = confusion_matrix(y_test, pred, labels=bundle["model"].classes_)
    disp = ConfusionMatrixDisplay(cm, display_labels=bundle["model"].classes_)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("KNN Classification - Confusion Matrix")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=160)
    plt.close()
    comparison = pd.DataFrame([{"model": "KNN", "best_k": bundle["best_k"], "accuracy": acc}])
    comparison.to_csv("results/model_comparison.csv", index=False)
    print(f"Test accuracy: {acc:.4f}")
    print(report)
    print("Saved evaluation outputs in results/")

if __name__ == "__main__":
    main()
