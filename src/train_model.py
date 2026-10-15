import os
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from src.data_preprocessing import load_dataset, split_and_scale, FEATURES

def main():
    df = load_dataset()
    X_train, X_test, y_train, y_test, X_train_s, X_test_s, scaler = split_and_scale(df)
    candidates = range(1, 16)
    scores = {}
    for k in candidates:
        model = KNeighborsClassifier(n_neighbors=k)
        cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring="accuracy")
        scores[k] = cv_scores.mean()
    best_k = max(scores, key=scores.get)
    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train_s, y_train)
    os.makedirs("models", exist_ok=True)
    joblib.dump(
        {"model": model, "scaler": scaler, "features": FEATURES, "best_k": best_k},
        "models/classification_model.pkl"
    )
    print(f"Best K: {best_k}")
    print(f"Best 5-fold CV accuracy: {scores[best_k]:.4f}")
    print("Saved: models/classification_model.pkl")

if __name__ == "__main__":
    main()
