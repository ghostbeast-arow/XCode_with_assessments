import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_XGBOOST = False

SPEND_COLUMNS = [
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
]

CATEGORICAL_COLUMNS = [
    "HomePlanet",
    "CryoSleep",
    "Destination",
    "VIP",
    "Deck",
    "Side",
]

NUMERIC_COLUMNS = [
    "Age",
    "CabinNum",
    "NameLength",
    "GroupSize",
    "TotalSpend",
] + SPEND_COLUMNS


def split_cabin(df: pd.DataFrame) -> pd.DataFrame:
    cabin = df["Cabin"].astype("string")
    deck = cabin.str.split("/", expand=True)[0]
    num = cabin.str.split("/", expand=True)[1]
    side = cabin.str.split("/", expand=True)[2]
    df = df.copy()
    df["Deck"] = deck
    df["CabinNum"] = pd.to_numeric(num, errors="coerce")
    df["Side"] = side
    return df


def add_group_features(df: pd.DataFrame) -> pd.DataFrame:
    group_id = df["PassengerId"].astype("string").str.split("_", expand=True)[0]
    df = df.copy()
    df["GroupId"] = group_id
    group_sizes = df.groupby("GroupId")["PassengerId"].transform("count")
    df["GroupSize"] = group_sizes
    return df


def add_name_length(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["NameLength"] = df["Name"].astype("string").str.len()
    return df


def add_spend_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[SPEND_COLUMNS] = df[SPEND_COLUMNS].fillna(0)
    df["TotalSpend"] = df[SPEND_COLUMNS].sum(axis=1)
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = split_cabin(df)
    df = add_group_features(df)
    df = add_name_length(df)
    df = add_spend_features(df)
    df = df.drop(columns=["Cabin", "Name"])
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLUMNS),
            ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
        ]
    )


def build_models(random_state: int = 42):
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            solver="saga",
            n_jobs=-1,
            class_weight="balanced",
            random_state=random_state,
        )
    }
    if HAS_XGBOOST:
        models["xgboost"] = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
    return models


def evaluate_model(name, pipeline, X, y, cv, output_dir: Path):
    probas = cross_val_predict(
        pipeline,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    preds = (probas >= 0.5).astype(int)
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds),
        "recall": recall_score(y, preds),
        "f1": f1_score(y, preds),
        "roc_auc": roc_auc_score(y, probas),
    }

    fpr, tpr, _ = roc_curve(y, probas)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"{name} (AUC={metrics['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / f"roc_curve_{name}.png", dpi=150)
    plt.close()

    cm = confusion_matrix(y, preds)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"confusion_matrix_{name}.png", dpi=150)
    plt.close()

    return metrics


def plot_missing_values(df: pd.DataFrame, output_dir: Path) -> None:
    missing = df.isna().sum().sort_values(ascending=False)
    missing_ratio = (missing / len(df)).rename("missing_ratio")
    missing_df = pd.concat([missing.rename("missing_count"), missing_ratio], axis=1)
    missing_df.to_csv(output_dir / "missing_values.csv")

    plt.figure(figsize=(8, 4))
    missing_ratio.head(15).plot(kind="bar")
    plt.ylabel("Missing Ratio")
    plt.title("Top Missing Value Ratios")
    plt.tight_layout()
    plt.savefig(output_dir / "missing_values.png", dpi=150)
    plt.close()


def plot_target_distribution(y: pd.Series, output_dir: Path) -> None:
    plt.figure(figsize=(4, 4))
    sns.countplot(x=y)
    plt.title("Target Distribution")
    plt.xlabel("Transported")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "target_distribution.png", dpi=150)
    plt.close()


def save_metrics(metrics_list, output_dir: Path) -> None:
    metrics_df = pd.DataFrame(metrics_list).sort_values(by="f1", ascending=False)
    metrics_df.to_csv(output_dir / "cv_metrics.csv", index=False)


def train_best_model(best_name, pipeline, X, y, test_df, output_dir: Path):
    pipeline.fit(X, y)
    if test_df is None:
        return
    test_features = prepare_features(test_df)
    predictions = pipeline.predict(test_features)
    submission = pd.DataFrame(
        {
            "PassengerId": test_df["PassengerId"],
            "Transported": predictions.astype(bool),
        }
    )
    submission.to_csv(output_dir / "submission.csv", index=False)
    with open(output_dir / "best_model.txt", "w", encoding="utf-8") as f:
        f.write(f"Best model: {best_name}\n")


def load_data(data_dir: Path):
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if test_path.exists() else None
    return train_df, test_df


def main():
    parser = argparse.ArgumentParser(description="Spaceship Titanic training pipeline")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing train.csv and test.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to save plots and metrics",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_data(args.data_dir)
    plot_missing_values(train_df, args.output_dir)

    features = prepare_features(train_df)
    target = train_df["Transported"].astype(int)

    plot_target_distribution(target, args.output_dir)

    preprocessor = build_preprocessor()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    metrics_list = []
    best_name = None
    best_score = -np.inf
    best_pipeline = None

    for name, model in build_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )
        metrics = evaluate_model(name, pipeline, features, target, cv, args.output_dir)
        metrics_list.append(metrics)
        if metrics["f1"] > best_score:
            best_score = metrics["f1"]
            best_name = name
            best_pipeline = pipeline

    save_metrics(metrics_list, args.output_dir)
    if best_pipeline is not None:
        train_best_model(best_name, best_pipeline, features, target, test_df, args.output_dir)


if __name__ == "__main__":
    main()
