#!/usr/bin/env python3
"""LSTM 로또 실험 학습/예측 스크립트.

lotto-data.json 을 읽어, 직전 W회차 시퀀스로 다음 회차의 번호별 확률을 학습하고,
최신 시퀀스로 다음 회차를 예측해 lstm-prediction.json 으로 저장한다.

주의: 로또는 독립시행(IID)이라 학습 가능한 신호가 없다. 모델 출력은 사실상 과거
빈도 통계로 수렴하며, 기존 통계 추천과 통계적으로 구분되지 않는다. 이 결과는
"딥러닝이 무엇을 출력하는가"를 보여주는 실험/시연용이며 당첨 확률 상승을 의미하지
않는다.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SEED = 42
NUM_RANGE = 45          # 번호 1..45
PICK = 6                # 한 세트 번호 개수
DEFAULT_WINDOW = 10     # 입력 시퀀스 길이 (직전 W회차)
DEFAULT_EPOCHS = 100
DEFAULT_BATCH = 16
MIN_TRAIN_SAMPLES = 50  # 이보다 적으면 학습 의미가 없어 중단

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "lotto-data.json"
OUT_PATH = ROOT / "lstm-prediction.json"
HISTORY_PATH = ROOT / "lstm-prediction-history.json"
HISTORY_LIMIT = 200  # 이력 최대 보관 회차 수

MODEL_NAME = "keras-lstm-multihot-v2"
WARNING = (
    "로또는 독립시행이므로 이 결과는 실험용이며 당첨 확률 상승을 의미하지 않습니다. "
    "딥러닝 출력은 과거 빈도 통계와 통계적으로 구분되지 않습니다."
)


def load_draws(path: Path) -> list[dict]:
    """lotto-data.json 을 읽어 검증된 회차를 오름차순(과거→최신)으로 반환."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schemaVersion") != 1 or not isinstance(raw.get("draws"), list):
        raise ValueError("lotto-data.json schema invalid")

    draws: list[dict] = []
    for d in raw["draws"]:
        nums = [int(n) for n in d.get("numbers", [])]
        if len(nums) != PICK or len(set(nums)) != PICK:
            continue
        if any(n < 1 or n > NUM_RANGE for n in nums):
            continue
        bonus = d.get("bonus")
        draws.append(
            {
                "round": int(d["round"]),
                "numbers": sorted(nums),
                "date": str(d.get("date", "")),
                "bonus": int(bonus) if isinstance(bonus, (int, float)) and 1 <= int(bonus) <= NUM_RANGE else None,
            }
        )

    if not draws:
        raise ValueError("no valid draws in lotto-data.json")

    # 원본은 newest-first 이므로 반드시 회차 오름차순으로 정렬한다.
    draws.sort(key=lambda x: x["round"])
    return draws


def to_multihot(numbers: list[int]) -> np.ndarray:
    vec = np.zeros(NUM_RANGE, dtype=np.float32)
    for n in numbers:
        vec[n - 1] = 1.0
    return vec


def build_dataset(vectors: np.ndarray, window: int):
    """(samples, window, 45) -> (samples, 45) 시퀀스 데이터셋."""
    x, y = [], []
    for i in range(len(vectors) - window):
        x.append(vectors[i : i + window])
        y.append(vectors[i + window])
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)


def top_probability_set(probs: np.ndarray) -> list[int]:
    idx = np.argsort(probs)[::-1][:PICK]
    return sorted(int(i) + 1 for i in idx)


def weighted_sampling_set(probs: np.ndarray, rng: np.random.Generator) -> list[int]:
    """확률 가중 비복원 추출로 중복 없는 6개를 보장."""
    total = float(probs.sum())
    p = probs / total if total > 0 else np.full(NUM_RANGE, 1.0 / NUM_RANGE)
    picks = rng.choice(NUM_RANGE, size=PICK, replace=False, p=p)
    return sorted(int(i) + 1 for i in picks)


# --- 예측 이력(성적표) 관리 ---------------------------------------------------
# lstm-prediction.json 은 매주 덮어써지므로, 회차별 예측과 실제 당첨 결과 대조를
# lstm-prediction-history.json 에 누적한다. 학습과 같은 커밋으로 CI가 관리한다.


def load_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if raw.get("schemaVersion") == 1 and isinstance(raw.get("entries"), list):
        return [e for e in raw["entries"] if isinstance(e.get("targetRound"), int)]
    return []


def prediction_to_entry(pred: dict) -> dict | None:
    if not isinstance(pred.get("targetRound"), int):
        return None
    recs = pred.get("recommendations")
    if not isinstance(recs, list) or not recs:
        return None
    return {
        "targetRound": pred["targetRound"],
        "sourceLatestRound": pred.get("sourceLatestRound"),
        "trainedAt": pred.get("trainedAt"),
        "recommendations": recs,
        "result": None,
    }


def upsert_entry(entries: list[dict], entry: dict, replace: bool) -> list[dict]:
    """같은 targetRound 항목이 있으면 replace 여부에 따라 교체하거나 유지."""
    exists = any(e["targetRound"] == entry["targetRound"] for e in entries)
    if exists and not replace:
        return entries
    kept = [e for e in entries if e["targetRound"] != entry["targetRound"]]
    return kept + [entry]


def reconcile_history(entries: list[dict], draws: list[dict]) -> list[dict]:
    """결과 미확정 항목을 실제 당첨번호와 대조해 적중 개수를 기록."""
    by_round = {d["round"]: d for d in draws}
    for entry in entries:
        if entry.get("result"):
            continue
        draw = by_round.get(entry["targetRound"])
        if not draw:
            continue
        winning = set(draw["numbers"])
        bonus = draw.get("bonus")
        entry["result"] = {
            "date": draw.get("date", ""),
            "winningNumbers": draw["numbers"],
            "bonus": bonus,
            "matches": [
                {
                    "method": rec.get("method", ""),
                    "matchCount": len(winning & set(rec.get("numbers", []))),
                    "bonusMatched": bonus is not None
                    and bonus in rec.get("numbers", [])
                    and len(winning & set(rec.get("numbers", []))) < PICK,
                }
                for rec in entry.get("recommendations", [])
            ],
        }
    return entries


def save_history(path: Path, entries: list[dict]) -> None:
    entries = sorted(entries, key=lambda e: e["targetRound"], reverse=True)[:HISTORY_LIMIT]
    path.write_text(
        json.dumps({"schemaVersion": 1, "entries": entries}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def update_history(new_prediction: dict, draws: list[dict]) -> None:
    entries = load_history(HISTORY_PATH)

    # 덮어쓰기 전의 기존 예측이 이력에 없으면 보존 (최초 부트스트랩용)
    if OUT_PATH.exists():
        try:
            prev = prediction_to_entry(json.loads(OUT_PATH.read_text(encoding="utf-8")))
            if prev:
                entries = upsert_entry(entries, prev, replace=False)
        except (json.JSONDecodeError, OSError):
            pass

    entries = upsert_entry(entries, prediction_to_entry(new_prediction), replace=True)
    entries = reconcile_history(entries, draws)
    save_history(HISTORY_PATH, entries)


def selftest() -> int:
    """TF 없이 이력 로직만 검증하는 자체 점검."""
    draws = [
        {"round": 100, "numbers": [1, 2, 3, 4, 5, 6], "date": "2026-01-03", "bonus": 7},
    ]
    pred_99 = {
        "targetRound": 100,
        "sourceLatestRound": 99,
        "trainedAt": "t0",
        "recommendations": [{"method": "m", "numbers": [1, 2, 3, 10, 11, 7]}],
    }
    entries = upsert_entry([], prediction_to_entry(pred_99), replace=True)
    assert len(entries) == 1

    # 같은 회차 재학습 시 교체, replace=False 면 유지
    entries = upsert_entry(entries, prediction_to_entry({**pred_99, "trainedAt": "t1"}), replace=True)
    assert len(entries) == 1 and entries[0]["trainedAt"] == "t1"
    entries = upsert_entry(entries, prediction_to_entry({**pred_99, "trainedAt": "t2"}), replace=False)
    assert entries[0]["trainedAt"] == "t1"

    entries = reconcile_history(entries, draws)
    match = entries[0]["result"]["matches"][0]
    assert match["matchCount"] == 3, match
    assert match["bonusMatched"] is True, match

    # 이미 채점된 항목은 다시 채점하지 않음 / 미추첨 회차는 result 없음
    entries = reconcile_history(entries, draws)
    assert entries[0]["result"]["matches"][0]["matchCount"] == 3
    future = upsert_entry(entries, prediction_to_entry({**pred_99, "targetRound": 101}), replace=True)
    future = reconcile_history(future, draws)
    assert next(e for e in future if e["targetRound"] == 101)["result"] is None

    print("selftest ok")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="LSTM 로또 실험 학습/예측")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--selftest", action="store_true", help="이력 로직만 검증 (TF 불필요)")
    args = parser.parse_args()

    if args.selftest:
        return selftest()

    # --- 재현성: numpy / python / tensorflow 시드 고정 ---
    import tensorflow as tf

    tf.keras.utils.set_random_seed(SEED)
    rng = np.random.default_rng(SEED)

    draws = load_draws(DATA_PATH)
    source_latest = draws[-1]["round"]
    target_round = source_latest + 1

    vectors = np.stack([to_multihot(d["numbers"]) for d in draws])
    x, y = build_dataset(vectors, args.window)
    if len(x) < MIN_TRAIN_SAMPLES:
        raise SystemExit(
            f"학습 샘플 부족: {len(x)} < {MIN_TRAIN_SAMPLES} (window={args.window})"
        )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(args.window, NUM_RANGE)),
            tf.keras.layers.LSTM(128),
            tf.keras.layers.Dense(NUM_RANGE, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy")
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )
    history = model.fit(
        x,
        y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.1,
        callbacks=[early],
        verbose=2,
    )

    # 최신 W회차 시퀀스로 다음 회차 예측
    last_window = vectors[-args.window :][np.newaxis, ...]
    probs = model.predict(last_window, verbose=0)[0].astype(float)

    order = np.argsort(probs)[::-1]
    top_numbers = [
        {"number": int(i) + 1, "probability": round(float(probs[i]), 4)}
        for i in order[:10]
    ]

    result = {
        "schemaVersion": 1,
        "model": MODEL_NAME,
        "sourceLatestRound": source_latest,
        "targetRound": target_round,
        "trainedAt": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "window": args.window,
        "epochs": int(len(history.history["loss"])),  # 실제 실행된 에폭 수
        "trainSampleCount": int(len(x)),
        "topNumbers": top_numbers,
        "recommendations": [
            {"method": "lstm-top-probability", "numbers": top_probability_set(probs)},
            {
                "method": "lstm-weighted-sampling",
                "numbers": weighted_sampling_set(probs, rng),
            },
        ],
        "warning": WARNING,
    }

    update_history(result, draws)  # OUT_PATH 덮어쓰기 전에 기존 예측을 이력에 보존
    OUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"saved {OUT_PATH.name}: source={source_latest} target={target_round} "
        f"samples={len(x)} epochs={result['epochs']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
