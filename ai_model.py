"""Random Forest parking forecast trained only on generated campus histories."""

from __future__ import annotations

import math
from functools import lru_cache
from datetime import datetime

import numpy as np
from sklearn.ensemble import RandomForestRegressor


@lru_cache(maxsize=1)
def _model() -> RandomForestRegressor:
    rng = np.random.default_rng(34)
    samples = 5000
    capacity = 20
    occupied = rng.integers(0, capacity + 1, samples)
    hour = rng.uniform(6, 22, samples)
    people = rng.integers(10, 160, samples)
    meeting = rng.integers(0, 2, samples)
    horizon = rng.choice([10, 20, 30], samples)
    arrivals_rate = 0.003 + 0.009 * np.exp(-((hour - 9) / 2) ** 2) + 0.006 * np.exp(-((hour - 14) / 3) ** 2)
    departures_rate = 0.004 + 0.009 * np.exp(-((hour - 17) / 2.5) ** 2)
    arrivals = rng.binomial(capacity - occupied, np.clip(arrivals_rate * horizon + meeting * 0.002, 0, 0.8))
    departures = rng.binomial(occupied, np.clip(departures_rate * horizon, 0, 0.8))
    future = np.clip(occupied + arrivals - departures, 0, capacity)
    angle = hour * (2 * math.pi / 24)
    features = np.column_stack((occupied, capacity - occupied, people, meeting, horizon, np.sin(angle), np.cos(angle)))
    model = RandomForestRegressor(n_estimators=60, max_depth=11, min_samples_leaf=3, random_state=34, n_jobs=-1)
    model.fit(features, future)
    return model


def predict_parking(occupied: int, people_count: int, when: datetime, meeting_active: bool, horizon: int = 10) -> dict[str, int]:
    horizon = min((10, 20, 30), key=lambda value: abs(value - horizon))
    hour = when.hour + when.minute / 60
    angle = hour * (2 * math.pi / 24)
    features = np.array([[occupied, 20 - occupied, people_count, int(meeting_active), horizon, math.sin(angle), math.cos(angle)]])
    model = _model()
    predictions = np.array([tree.predict(features)[0] for tree in model.estimators_])
    predicted_occupied = int(np.clip(round(float(predictions.mean())), 0, 20))
    low_occupied, high_occupied = np.percentile(predictions, [15, 85])
    return {
        "horizon_minutes": horizon,
        "available": 20 - predicted_occupied,
        "available_low": max(0, 20 - math.ceil(high_occupied)),
        "available_high": min(20, 20 - math.floor(low_occupied)),
    }
