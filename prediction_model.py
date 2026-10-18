import numpy as np
from sklearn.ensemble import RandomForestClassifier


def train_demo_model():
    """Train a demonstration classifier on synthetic data only."""
    rng = np.random.default_rng(42)
    sample_count = 1500

    heart_rate = rng.normal(75, 12, sample_count).clip(45, 140)
    hrv = rng.normal(45, 12, sample_count).clip(10, 100)
    stress_index = rng.integers(0, 101, sample_count)

    # Synthetic teaching label, not a medical definition.
    elevated_event = (
        (heart_rate >= 100) & (stress_index >= 60)
    ).astype(int)

    features = np.column_stack(
        [heart_rate, hrv, stress_index]
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(features, elevated_event)

    return model


def predict_demo_event(heart_rate, hrv, stress_index):
    """Return a synthetic model score for demonstration only."""
    model = train_demo_model()
    features = np.array([[heart_rate, hrv, stress_index]])

    probability = model.predict_proba(features)[0]
    class_index = list(model.classes_).index(1)

    return float(probability[class_index])