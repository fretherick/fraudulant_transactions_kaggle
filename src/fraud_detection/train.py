"""Train, evaluate, and document fraud detection models."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve

from fraud_detection.data import FLAG_COLUMN, TARGET, load_transactions, temporal_split
from fraud_detection.evaluation import classification_metrics, select_threshold
from fraud_detection.features import build_features
from fraud_detection.modeling import build_models, sample_training_data

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PRIMARY_MODEL = "hist_gradient_boosting"
DISPLAY_NAMES = {
    "business_rule": "Existing rule",
    "logistic_regression": "Logistic regression",
    PRIMARY_MODEL: "Histogram gradient boosting",
}


def _fit(model: object, x: pd.DataFrame, y: pd.Series, weights: np.ndarray) -> None:
    if hasattr(model, "named_steps"):
        model.fit(x, y, logisticregression__sample_weight=weights)
    else:
        model.fit(x, y, sample_weight=weights)


def _importance_sample(
    x: pd.DataFrame,
    y: pd.Series,
    max_rows: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if len(x) <= max_rows:
        return x, y
    positive_index = y.index[y == 1].to_numpy()
    negative_index = y.index[y == 0].to_numpy()
    negative_size = max_rows - len(positive_index)
    rng = np.random.default_rng(random_state)
    if negative_size <= 0:
        chosen = rng.choice(positive_index, size=max_rows, replace=False)
    else:
        chosen_negatives = rng.choice(negative_index, size=negative_size, replace=False)
        chosen = np.concatenate([positive_index, chosen_negatives])
    return x.loc[chosen], y.loc[chosen]


def _save_plots(
    frame: pd.DataFrame,
    y_test: pd.Series,
    test_scores: dict[str, np.ndarray],
    thresholds: dict[str, float],
    metrics: dict[str, dict[str, float | int]],
    importance: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    counts = frame[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(["Legitimate", "Fraud"], counts.values, color=["#4C78A8", "#E45756"])
    ax.set_yscale("log")
    ax.set_ylabel("Transactions (log scale)")
    ax.set_title("Severe class imbalance")
    ax.bar_label(bars, labels=[f"{value:,}" for value in counts.values], padding=4)
    fig.tight_layout()
    fig.savefig(figure_dir / "class_distribution.png", dpi=160)
    plt.close(fig)

    fraud_by_type = (
        frame.groupby("type", observed=True)[TARGET]
        .agg(["sum", "count"])
        .assign(rate=lambda values: values["sum"] / values["count"])
        .sort_values("rate", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(
        fraud_by_type.index.astype(str),
        fraud_by_type["rate"] * 100,
        color="#E45756",
    )
    ax.set_ylabel("Fraud rate (%)")
    ax.set_title("Fraud rate by transaction type")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "fraud_rate_by_type.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.2))
    for model_name, scores in test_scores.items():
        if model_name == "business_rule":
            ax.scatter(
                metrics[model_name]["recall"],
                metrics[model_name]["precision"],
                marker="x",
                s=80,
                label=f"{DISPLAY_NAMES[model_name]} operating point",
            )
            continue
        precision, recall, _ = precision_recall_curve(y_test, scores)
        ap = metrics[model_name]["average_precision"]
        ax.plot(recall, precision, label=f"{DISPLAY_NAMES[model_name]} (AP={ap:.3f})")
    prevalence = float(y_test.mean())
    ax.axhline(prevalence, color="#777777", linestyle="--", label=f"Random ({prevalence:.4f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve on temporal test set")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(figure_dir / "precision_recall_curve.png", dpi=160)
    plt.close(fig)

    primary_predictions = (test_scores[PRIMARY_MODEL] >= thresholds[PRIMARY_MODEL]).astype("uint8")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        primary_predictions,
        display_labels=["Legitimate", "Fraud"],
        cmap="Blues",
        colorbar=False,
        values_format=",d",
        ax=ax,
    )
    ax.set_title("Primary model confusion matrix")
    fig.tight_layout()
    fig.savefig(figure_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    top = importance.head(12).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#59A14F")
    ax.set_xlabel("Decrease in average precision after permutation")
    ax.set_title("Primary model permutation importance")
    fig.tight_layout()
    fig.savefig(figure_dir / "feature_importance.png", dpi=160)
    plt.close(fig)


def _write_model_card(
    output_path: Path,
    summary: dict[str, object],
    comparison: pd.DataFrame,
    importance: pd.DataFrame,
) -> None:
    primary = summary["models"][PRIMARY_MODEL]
    dataset = summary["dataset"]
    rows = [
        "# Model Card",
        "",
        "## Intended use",
        "",
        "This model ranks simulated financial transactions by fraud risk using only information",
        "available before transaction completion. It is a research and portfolio artifact, not a",
        "production authorization system.",
        "",
        "## Data and validation",
        "",
        f"- Dataset rows: {dataset['rows']:,}",
        f"- Fraud prevalence: {dataset['fraud_prevalence']:.4%}",
        f"- Temporal split: steps 1-{summary['split']['train_end_step']} train, "
        f"{summary['split']['train_end_step'] + 1}-{summary['split']['validation_end_step']} "
        f"validation, later steps test",
        "- Account identifiers are excluded to reduce memorization and privacy risk.",
        "- `isFlaggedFraud` is evaluated as a baseline but excluded from model features.",
        "- Post-transaction sender and recipient balances are excluded from model features.",
        "",
        "## Test performance",
        "",
        comparison.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"The operating threshold ({primary['threshold']:.6f}) was selected on the validation set",
        "by maximizing F2, which weights recall more heavily than precision.",
        "",
        "## Most influential features",
        "",
        importance.head(10).to_markdown(index=False, floatfmt=".5f"),
        "",
        "Permutation importance describes predictive contribution, not causality.",
        "",
        "## Limitations",
        "",
        "- PaySim is simulated data and does not represent every real fraud pattern.",
        "- Production integration must verify that every input is available at authorization time.",
        "- Financial loss and investigation cost are not provided. The F2 threshold is therefore a",
        "  transparent proxy, not a fully optimized business decision.",
        "- Performance should be monitored over time for drift and subgroup failures before",
        "  deployment.",
        "",
    ]
    output_path.write_text("\n".join(rows), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = args.output_dir.resolve()
    figure_dir = output_dir / "figures"
    model_dir = output_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.data} ...")
    frame = load_transactions(args.data, args.max_data_rows)
    split = temporal_split(frame)

    print("Engineering features ...")
    x_train_full = build_features(split.train)
    x_validation = build_features(split.validation)
    x_test = build_features(split.test)
    y_train_full = split.train[TARGET]
    y_validation = split.validation[TARGET]
    y_test = split.test[TARGET]
    x_train, y_train, weights = sample_training_data(
        x_train_full,
        y_train_full,
        max_rows=args.max_train_rows,
        random_state=args.random_state,
    )
    del x_train_full

    metrics: dict[str, dict[str, float | int]] = {}
    thresholds: dict[str, float] = {"business_rule": 0.5}
    test_scores: dict[str, np.ndarray] = {
        "business_rule": split.test[FLAG_COLUMN].to_numpy(dtype="float64")
    }
    metrics["business_rule"] = classification_metrics(
        y_test.to_numpy(), test_scores["business_rule"], 0.5
    )

    trained_models = {}
    for model_name, model in build_models(args.random_state).items():
        print(f"Training {DISPLAY_NAMES[model_name]} on {len(x_train):,} rows ...")
        _fit(model, x_train, y_train, weights)
        validation_scores = model.predict_proba(x_validation)[:, 1]
        threshold = select_threshold(y_validation.to_numpy(), validation_scores, beta=2)
        scores = model.predict_proba(x_test)[:, 1]
        thresholds[model_name] = threshold
        test_scores[model_name] = scores
        metrics[model_name] = classification_metrics(y_test.to_numpy(), scores, threshold)
        trained_models[model_name] = model

    print("Computing permutation importance ...")
    importance_x, importance_y = _importance_sample(
        x_test, y_test, args.importance_rows, args.random_state
    )
    permutation = permutation_importance(
        trained_models[PRIMARY_MODEL],
        importance_x,
        importance_y,
        scoring="average_precision",
        n_repeats=3,
        random_state=args.random_state,
        n_jobs=args.jobs,
    )
    importance = pd.DataFrame(
        {
            "feature": x_test.columns,
            "importance_mean": permutation.importances_mean,
            "importance_std": permutation.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    comparison = pd.DataFrame(
        [{"model": DISPLAY_NAMES[name], **values} for name, values in metrics.items()]
    )
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)

    summary: dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset": {
            "rows": int(len(frame)),
            "fraud_count": int(frame[TARGET].sum()),
            "fraud_prevalence": float(frame[TARGET].mean()),
            "time_steps": int(frame["step"].nunique()),
        },
        "split": {
            "train_end_step": split.train_end_step,
            "validation_end_step": split.validation_end_step,
            "train_rows": int(len(split.train)),
            "validation_rows": int(len(split.validation)),
            "test_rows": int(len(split.test)),
            "sampled_training_rows": int(len(x_train)),
            "test_fraud_count": int(y_test.sum()),
        },
        "features": list(x_test.columns),
        "models": metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _save_plots(frame, y_test, test_scores, thresholds, metrics, importance, figure_dir)
    _write_model_card(output_dir / "model_card.md", summary, comparison, importance)

    bundle = {
        "model": trained_models[PRIMARY_MODEL],
        "threshold": thresholds[PRIMARY_MODEL],
        "feature_names": list(x_test.columns),
        "model_name": PRIMARY_MODEL,
    }
    joblib.dump(bundle, model_dir / "fraud_model.joblib")
    print(f"Reports written to {output_dir}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/Fraud.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--max-train-rows", type=int, default=1_000_000)
    parser.add_argument("--importance-rows", type=int, default=50_000)
    parser.add_argument("--max-data-rows", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
