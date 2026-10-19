
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


FEATURE_NAMES = [
    "heart_rate_bpm",
    "hrv_ms",
    "stress_index",
]


def _generate_demo_data(sample_count=1500, seed=42):
    """Generate synthetic teaching data. Not clinical patient data."""
    rng = np.random.default_rng(seed)

    heart_rate = rng.normal(75, 12, sample_count).clip(45, 140)
    hrv = rng.normal(45, 12, sample_count).clip(10, 100)
    stress_index = rng.integers(0, 101, sample_count)

    # Demonstration label only; not a medical event definition.
    elevated_pattern = (
        (heart_rate >= 100) & (stress_index >= 60)
    ).astype(int)

    features = np.column_stack(
        [heart_rate, hrv, stress_index]
    )

    return features, elevated_pattern


def train_demo_model():
    """Train a demonstration classifier on synthetic data only."""
    features, labels = _generate_demo_data()

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    # Keep evaluation separate from the training data.
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=[
                "Pattern not detected",
                "Elevated demo pattern",
            ],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        ).tolist(),
        "test_samples": int(len(y_test)),
        "data_type": "synthetic",
    }


    return model, metrics


# Train once when this module is first imported.
_DEMO_MODEL, _DEMO_METRICS = train_demo_model()


def predict_demo_event(heart_rate, hrv, stress_index):
    """
    Return a demo-pattern score between 0 and 1.

    This is NOT a clinical risk probability.
    """
    features = np.array(
        [[heart_rate, hrv, stress_index]],
        dtype=float,
    )

    if not np.isfinite(features).all():
        raise ValueError("All input measurements must be finite.")

    if not (30 <= heart_rate <= 250):
        raise ValueError("Heart rate must be between 30 and 250 bpm.")

    if not (0 <= hrv <= 250):
        raise ValueError("HRV must be between 0 and 250 ms.")

    if not (0 <= stress_index <= 100):
        raise ValueError("Stress index must be between 0 and 100.")

    probabilities = _DEMO_MODEL.predict_proba(features)[0]
    class_index = list(_DEMO_MODEL.classes_).index(1)

    return float(probabilities[class_index])



def get_demo_model_metrics():
    """Return held-out metrics for the synthetic demonstration."""
    return _DEMO_METRICS



def get_demo_feature_importance():
    """Return feature importance for the synthetic demo model."""
    return {
        name: float(importance)
        for name, importance in zip(
            FEATURE_NAMES,
            _DEMO_MODEL.feature_importances_,
        )
    }
