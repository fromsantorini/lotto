#!/usr/bin/env python3
"""LSTM 로또 실험 학습/예측 스크립트.

lotto-data.json 을 읽어 직전 W회차 시퀀스와 후보 조합의 적합도를 학습하고,
유효 후보를 가중 선택해 lstm-prediction.json 으로 저장한다.

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

NEGATIVES_PER_POSITIVE = 5
INFERENCE_CANDIDATES = 50_000
VALIDATION_FRACTION = 0.1
MAX_CANDIDATE_ATTEMPT_FACTOR = 100

MODEL_NAME = "keras-lstm-combination-scorer-v1"
WARNING = (
    "모델 점수는 후보 간 상대 비교값이며 당첨 확률이 아닙니다. "
    "로또는 독립시행이므로 이 결과는 실험용입니다."
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


# --- 통계 가중 추천 -----------------------------------------------------------
# 번호 형태는 강제하지 않는다. 역대 1등 조합만 제외하고, 기존 통계 지표는
# 후보의 선택 가중치에만 반영한다. 모든 유효 후보의 선택 가중치는 0보다 크다.

STAT_WINDOW = 100   # 강세/소외 판정에 쓰는 최근 회차 수
STAT_SETS = 3
WINDOW_RECENT = 4   # 보정 창: 최근 4회 + 이번 추천 = 5회
STAT_POOL = 10_000  # 통계 가중 선택용 무작위 후보 풀 크기
NUMBER_ZONES = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 45)]


def consecutive_runs(nums: list[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    current = [nums[0]]
    for prev, cur in zip(nums, nums[1:]):
        if cur == prev + 1:
            current.append(cur)
        else:
            if len(current) >= 2:
                runs.append(current)
            current = [cur]
    if len(current) >= 2:
        runs.append(current)
    return runs


def matches_legacy_balance(nums: list[int]) -> bool:
    """과거 강제 조건이 현재 후보를 거르지 않는지 확인하기 위한 자체 테스트용."""
    odd = sum(1 for n in nums if n % 2)
    low = sum(1 for n in nums if n <= 22)
    total = sum(nums)
    digits = [n % 10 for n in nums]
    max_same_ending = max(digits.count(d) for d in digits)
    return (
        2 <= odd <= 4
        and 2 <= low <= 4
        and 90 <= total <= 180
        and max_same_ending <= 2
        and all(len(run) <= 2 for run in consecutive_runs(nums))
    )


def multihot_to_numbers(vector: np.ndarray) -> list[int]:
    return [int(i) + 1 for i in np.flatnonzero(vector)]


def generate_candidates(
    rng: np.random.Generator,
    count: int,
    forbidden: set[tuple[int, ...]] | frozenset[tuple[int, ...]] = frozenset(),
) -> list[list[int]]:
    candidates: set[tuple[int, ...]] = set()
    for _ in range(count * MAX_CANDIDATE_ATTEMPT_FACTOR):
        numbers = tuple(
            sorted(int(i) + 1 for i in rng.choice(NUM_RANGE, PICK, replace=False))
        )
        if numbers not in forbidden:
            candidates.add(numbers)
            if len(candidates) == count:
                return [list(candidate) for candidate in sorted(candidates)]
    raise RuntimeError(f"candidate shortage: {len(candidates)} < {count}")


def expand_scorer_examples(
    sequences: np.ndarray,
    positives: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sequence_examples: list[np.ndarray] = []
    candidate_examples: list[np.ndarray] = []
    labels: list[float] = []

    for sequence, positive in zip(sequences, positives):
        positive_numbers = multihot_to_numbers(positive)
        negatives = generate_candidates(
            rng,
            NEGATIVES_PER_POSITIVE,
            {tuple(positive_numbers)},
        )
        for numbers, label in [
            (positive_numbers, 1.0),
            *((numbers, 0.0) for numbers in negatives),
        ]:
            sequence_examples.append(sequence)
            candidate_examples.append(to_multihot(numbers))
            labels.append(label)

    return (
        np.asarray(sequence_examples, dtype=np.float32),
        np.asarray(candidate_examples, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
    )


def overall_profile(draws: list[dict]) -> dict:
    """전체 누적 통계를 번호 칸 비율로 요약 (끝수·홀짝·저고·구간·연번율)."""
    nums = [n for d in draws for n in d["numbers"]]
    total = len(nums)
    return {
        "digitShare": [sum(1 for n in nums if n % 10 == d) / total for d in range(10)],
        "oddShare": sum(1 for n in nums if n % 2) / total,
        "lowShare": sum(1 for n in nums if n <= 22) / total,
        "zoneShare": [sum(1 for n in nums if lo <= n <= hi) / total for lo, hi in NUMBER_ZONES],
        "runRate": sum(bool(consecutive_runs(d["numbers"])) for d in draws) / len(draws),
    }


def window_deviation(profile: dict, window_nums: list[int]) -> float:
    """(최근 4회 + 후보) 창의 끝수·홀짝·저고·구간 비율과 전체 누적 비율의 편차 합."""
    total = len(window_nums)
    dev = sum(
        abs(sum(1 for n in window_nums if n % 10 == d) / total - profile["digitShare"][d])
        for d in range(10)
    )
    dev += abs(sum(1 for n in window_nums if n % 2) / total - profile["oddShare"])
    dev += abs(sum(1 for n in window_nums if n <= 22) / total - profile["lowShare"])
    dev += sum(
        abs(sum(1 for n in window_nums if lo <= n <= hi) / total - profile["zoneShare"][z])
        for z, (lo, hi) in enumerate(NUMBER_ZONES)
    )
    return dev


def selection_weights(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """유한 점수를 순서 보존·양수 확률로 변환한다."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not len(values) or not np.all(np.isfinite(values)) or temperature <= 0:
        raise ValueError("finite scores and positive temperature required")
    scaled = (values - np.max(values)) / temperature
    scaled = np.maximum(scaled, np.log(np.finfo(np.float64).tiny))
    weights = np.exp(scaled)
    return weights / weights.sum()


def stat_recommendations(
    draws: list[dict],
    rng: np.random.Generator,
    count: int = STAT_SETS,
    forbidden: set[tuple[int, ...]] | frozenset[tuple[int, ...]] = frozenset(),
) -> list[dict]:
    """최근 통계 점수를 선택 가중치로만 쓰는 추천 count세트 (draws는 오름차순)."""
    recent = draws[-STAT_WINDOW:]
    counts = {n: 0 for n in range(1, NUM_RANGE + 1)}
    for d in recent:
        for n in d["numbers"]:
            counts[n] += 1
    hot = sorted(counts, key=lambda n: (-counts[n], n))[:12]
    cold = sorted(counts, key=lambda n: (counts[n], n))[:12]
    historical = {tuple(d["numbers"]) for d in draws} | set(forbidden)
    carry_pool = set(draws[-1]["numbers"])
    profile = overall_profile(draws)
    recent4 = [n for d in draws[-WINDOW_RECENT:] for n in d["numbers"]]
    # 연번: 5회 창 기대치 대비 최근 4회가 부족하면 이번 세트에 2연번 포함
    runs4 = sum(1 for d in draws[-WINDOW_RECENT:] if consecutive_runs(d["numbers"]))
    want_run = profile["runRate"] * (WINDOW_RECENT + 1) - runs4 >= 0.5
    # 끝수 보정 표시용: 5회 창 목표 대비 최근 4회에 1개 이상 부족한 끝수
    slots = len(recent4) + PICK
    deficit_digits = {
        d for d in range(10)
        if profile["digitShare"][d] * slots - sum(1 for n in recent4 if n % 10 == d) >= 1
    }

    candidates = generate_candidates(rng, max(STAT_POOL, count), historical)
    average_sum = float(np.mean([sum(d["numbers"]) for d in draws]))
    sum_std = max(float(np.std([sum(d["numbers"]) for d in draws])), 1.0)
    scores = []
    for cand in candidates:
        hot_count = sum(n in hot for n in cand)
        cold_count = sum(n in cold for n in cand)
        carry_count = sum(n in carry_pool for n in cand)
        preference_penalty = (
            abs(hot_count - 2)
            + abs(cold_count - 2)
            + abs(carry_count - 1)
            + int(bool(consecutive_runs(cand)) != want_run)
        )
        scores.append(
            -window_deviation(profile, recent4 + cand)
            - abs(sum(cand) - average_sum) / sum_std
            - 0.25 * preference_penalty
        )
    selected_indices = rng.choice(
        len(candidates), size=count, replace=False, p=selection_weights(np.asarray(scores))
    )
    recs: list[dict] = []
    for index in selected_indices:
        cand = candidates[int(index)]
        odd = sum(1 for n in cand if n % 2)
        low = sum(1 for n in cand if n <= 22)
        recs.append(
            {
                "method": "weighted-statistical",
                "numbers": cand,
                "reason": {
                    "oddEven": f"{odd}:{PICK - odd}",
                    "lowHigh": f"{low}:{PICK - low}",
                    "sum": sum(cand),
                    "frequentNumbers": [n for n in cand if n in hot],
                    "coldNumbers": [n for n in cand if n in cold],
                    "carryOverNumbers": [n for n in cand if n in carry_pool],
                    "endingNumbers": [n for n in cand if n % 10 in deficit_digits],
                    "consecutiveRuns": consecutive_runs(cand),
                },
            }
        )
    return recs


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


def build_combination_scorer(tf, window: int):
    sequence_input = tf.keras.layers.Input(
        shape=(window, NUM_RANGE), name="sequence"
    )
    candidate_input = tf.keras.layers.Input(shape=(NUM_RANGE,), name="candidate")
    context = tf.keras.layers.LSTM(128)(sequence_input)
    merged = tf.keras.layers.Concatenate()([context, candidate_input])
    hidden = tf.keras.layers.Dense(64, activation="relu")(merged)
    score = tf.keras.layers.Dense(1, activation="sigmoid", name="score")(hidden)
    model = tf.keras.Model([sequence_input, candidate_input], score)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model


def select_scored_recommendations(
    candidates: list[list[int]], scores: np.ndarray, rng: np.random.Generator
) -> list[dict]:
    flat_scores = np.asarray(scores).reshape(-1)
    indices = rng.choice(
        len(candidates), size=2, replace=False, p=selection_weights(flat_scores, temperature=0.1)
    )
    return [
        {
            "method": "lstm-weighted-selection",
            "numbers": candidates[int(index)],
            "modelScore": round(float(flat_scores[int(index)]), 6),
        }
        for index in indices
    ]


def selftest() -> int:
    """TF 없이 추천/이력 핵심 로직을 검증하는 자체 점검."""
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

    # 통계 추천: 3세트, 유효성, 재현성, 역대/번들 중복 제외
    rng_a = np.random.default_rng(SEED)
    fake_draws = [
        {"round": r, "numbers": sorted(int(n) + 1 for n in rng_a.choice(45, size=6, replace=False))}
        for r in range(1, 121)
    ]
    extra_forbidden = {(1, 2, 3, 4, 5, 6)}
    recs_a = stat_recommendations(fake_draws, np.random.default_rng(7), forbidden=extra_forbidden)
    recs_b = stat_recommendations(fake_draws, np.random.default_rng(7), forbidden=extra_forbidden)
    assert len(recs_a) == STAT_SETS
    profile = overall_profile(fake_draws)
    recent4 = [n for d in fake_draws[-WINDOW_RECENT:] for n in d["numbers"]]
    slots = len(recent4) + 6
    deficit_digits = {
        d for d in range(10)
        if profile["digitShare"][d] * slots - sum(1 for n in recent4 if n % 10 == d) >= 1
    }
    for rec in recs_a:
        nums = rec["numbers"]
        assert len(nums) == 6 and len(set(nums)) == 6 and all(1 <= n <= 45 for n in nums), nums
        assert tuple(nums) not in {tuple(d["numbers"]) for d in fake_draws} | extra_forbidden
        reason = rec["reason"]
        assert reason["endingNumbers"] == [n for n in nums if n % 10 in deficit_digits], reason
        assert reason["consecutiveRuns"] == consecutive_runs(nums), reason

    # 창 편차 검증: 창 비율이 전체 비율과 같으면 편차 0
    uniform_draws = [{"round": r, "numbers": [1, 2, 13, 24, 35, 41]} for r in range(1, 11)]
    uprofile = overall_profile(uniform_draws)
    urecent = [n for d in uniform_draws[-WINDOW_RECENT:] for n in d["numbers"]]
    assert window_deviation(uprofile, urecent + [1, 2, 13, 24, 35, 41]) < 1e-9
    assert recs_a == recs_b, "stat recommendations must be reproducible with same rng"
    assert len({tuple(r["numbers"]) for r in recs_a}) == STAT_SETS, "sets must be unique"

    first_candidate = generate_candidates(np.random.default_rng(SEED), 1)[0]
    forbidden = {tuple(first_candidate)}
    candidates_a = generate_candidates(
        np.random.default_rng(SEED), 20, forbidden
    )
    candidates_b = generate_candidates(
        np.random.default_rng(SEED), 20, forbidden
    )
    assert candidates_a == candidates_b
    assert len(candidates_a) == len({tuple(c) for c in candidates_a}) == 20
    assert any(not matches_legacy_balance(c) for c in candidates_a), "shape rules must not filter candidates"
    assert all(tuple(c) not in forbidden for c in candidates_a)

    sequences = np.stack(
        [np.stack([to_multihot(d["numbers"]) for d in fake_draws[:10]])]
    )
    positives = np.stack([to_multihot([1, 8, 15, 22, 29, 36])])
    ex_seq, ex_cand, ex_label = expand_scorer_examples(
        sequences, positives, np.random.default_rng(SEED)
    )
    assert ex_seq.shape == (NEGATIVES_PER_POSITIVE + 1, 10, 45)
    assert ex_cand.shape == (NEGATIVES_PER_POSITIVE + 1, 45)
    assert ex_label.tolist() == [1.0] + [0.0] * NEGATIVES_PER_POSITIVE
    assert all(len(multihot_to_numbers(v)) == PICK for v in ex_cand)

    weights = selection_weights(np.asarray([0.9, 0.0, -10.0]))
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights > 0), weights
    assert weights[0] > weights[1] > weights[2], weights

    scored_candidates = [
        [1, 8, 15, 22, 29, 36],
        [2, 9, 16, 23, 30, 37],
        [1, 8, 17, 24, 31, 38],
    ]
    selected = select_scored_recommendations(
        scored_candidates,
        np.asarray([0.9, 0.8, 0.7]),
        np.random.default_rng(9),
    )
    assert [rec["method"] for rec in selected] == [
        "lstm-weighted-selection",
        "lstm-weighted-selection",
    ]
    assert len({tuple(rec["numbers"]) for rec in selected}) == 2

    print("selftest ok")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="LSTM 로또 실험 학습/예측")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--selftest", action="store_true", help="추천/이력 핵심 로직 검증 (TF 불필요)")
    args = parser.parse_args()

    if args.selftest:
        return selftest()

    # --- 재현성: numpy / python / tensorflow 시드 고정 ---
    import tensorflow as tf

    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    draws = load_draws(DATA_PATH)
    source_latest = draws[-1]["round"]
    target_round = source_latest + 1

    vectors = np.stack([to_multihot(d["numbers"]) for d in draws])
    x, y = build_dataset(vectors, args.window)
    if len(x) < MIN_TRAIN_SAMPLES:
        raise SystemExit(
            f"학습 샘플 부족: {len(x)} < {MIN_TRAIN_SAMPLES} (window={args.window})"
        )

    split = int(len(x) * (1.0 - VALIDATION_FRACTION))
    if split <= 0 or split >= len(x):
        raise SystemExit("시간순 검증 구간을 만들 수 없습니다")
    train_seq, train_pos = x[:split], y[:split]
    val_seq, val_pos = x[split:], y[split:]
    train_x, train_candidates, train_labels = expand_scorer_examples(
        train_seq, train_pos, np.random.default_rng(SEED)
    )
    val_x, val_candidates, val_labels = expand_scorer_examples(
        val_seq, val_pos, np.random.default_rng(SEED + 1)
    )

    model = build_combination_scorer(tf, args.window)
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )
    history = model.fit(
        [train_x, train_candidates],
        train_labels,
        validation_data=([val_x, val_candidates], val_labels),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early],
        verbose=2,
    )
    validation_auc = float(
        model.evaluate(
            [val_x, val_candidates], val_labels, verbose=0, return_dict=True
        )["auc"]
    )

    historical = {tuple(d["numbers"]) for d in draws}
    # ponytail: 50k 후보로 CI 비용을 제한하며, 백테스트가 불안정하면 수를 늘린다.
    candidates = generate_candidates(
        np.random.default_rng(SEED + 2),
        INFERENCE_CANDIDATES,
        historical,
    )
    candidate_vectors = np.stack([to_multihot(candidate) for candidate in candidates])
    sequence_batch = np.repeat(
        vectors[-args.window :][np.newaxis, ...], len(candidates), axis=0
    )
    scores = model.predict(
        [sequence_batch, candidate_vectors], batch_size=1024, verbose=0
    ).reshape(-1)
    lstm_recommendations = select_scored_recommendations(
        candidates, scores, np.random.default_rng(SEED + 3)
    )
    selected_lstm = {tuple(rec["numbers"]) for rec in lstm_recommendations}

    result = {
        "schemaVersion": 2,
        "model": MODEL_NAME,
        "sourceLatestRound": source_latest,
        "targetRound": target_round,
        "trainedAt": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "window": args.window,
        "epochs": int(len(history.history["loss"])),  # 실제 실행된 에폭 수
        "trainSampleCount": int(len(x)),
        "validationAuc": round(validation_auc, 4),
        "candidateCount": len(candidates),
        "recommendations": [
            *lstm_recommendations,
            *stat_recommendations(
                draws, np.random.default_rng(SEED + 4), forbidden=selected_lstm
            ),
        ],
        "warning": WARNING,
    }

    update_history(result, draws)  # OUT_PATH 덮어쓰기 전에 기존 예측을 이력에 보존
    OUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"saved {OUT_PATH.name}: source={source_latest} target={target_round} "
        f"samples={len(x)} epochs={result['epochs']} "
        f"validation_auc={validation_auc:.4f} candidates={len(candidates)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
