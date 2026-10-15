# AI Data Classification Project — KNN

A supervised machine-learning project based on the Iris dataset. The project notebook preserves the user's classification workflow: data inspection, visualization, an 80/20 stratified split, feature scaling, KNN tuning, evaluation, and comparison with Logistic Regression.

## Repository structure

```text
knn_classification_project/
├── dataset/
│   └── iris.csv
├── models/
│   └── classification_model.pkl
├── notebooks/
│   └── data_classification.ipynb
├── results/
│   ├── confusion_matrix.png
│   ├── model_comparison.csv
│   └── model_results.txt
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── evaluate_model.py
├── app.py
├── iris_classification.csv
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Setup

Use Python 3.10 or newer, then run from the project root:

```bash
pip install -r requirements.txt
python -m src.train_model
python -m src.evaluate_model
streamlit run app.py
```

## Dataset

The CSV contains 150 Iris samples, four numerical features measured in centimeters, and the `species` target (`setosa`, `versicolor`, or `virginica`).

## Notebook

Open `notebooks/data_classification.ipynb` in Jupyter or upload it to Google Colab. In Colab, upload `iris_classification.csv` if it is not already available. The notebook includes a fallback to the repository's `dataset/iris.csv`.

## Outputs

Training saves the model to `models/classification_model.pkl`. Evaluation writes a confusion-matrix image, model comparison CSV, and text classification report to `results/`.
