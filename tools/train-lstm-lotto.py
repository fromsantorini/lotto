#!/usr/bin/env python3
"""LSTM 로또 실험 학습/예측 스크립트.

lotto-data.json 을 읽어 직전 W회차 시퀀스와 후보 조합의 적합도를 학습하고,
균형 후보를 평가해 lstm-prediction.json 으로 저장한다.

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


# --- 통계 균형 추천 (기존 index.html 클라이언트 로직 포팅) ---------------------
# 홀짝 2~4, 저고 2~4, 합계 90~180, 같은 끝수 최대 2개, 3연번 이상 금지,
# 역대 1등 조합 제외. 강세 2 + 소외 2 + 중간 2 혼합, 이월수 1개 이상.
# 끝수·홀짝·저고·구간·연번은 "최근 4회 + 이번 추천" 5회 창의 비율이
# 전체 누적 비율과 비슷해지도록 보정(평균 회귀)한다.

STAT_WINDOW = 100   # 강세/소외 판정에 쓰는 최근 회차 수
STAT_SETS = 3
WINDOW_RECENT = 4   # 보정 창: 최근 4회 + 이번 추천 = 5회
STAT_POOL = 40      # 창 편차 비교용 후보 풀 크기
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


def is_balanced(nums: list[int]) -> bool:
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


def generate_balanced_candidates(
    rng: np.random.Generator,
    count: int,
    forbidden: set[tuple[int, ...]] | frozenset[tuple[int, ...]] = frozenset(),
) -> list[list[int]]:
    candidates: set[tuple[int, ...]] = set()
    for _ in range(count * MAX_CANDIDATE_ATTEMPT_FACTOR):
        numbers = tuple(
            sorted(int(i) + 1 for i in rng.choice(NUM_RANGE, PICK, replace=False))
        )
        if numbers not in forbidden and is_balanced(list(numbers)):
            candidates.add(numbers)
            if len(candidates) == count:
                return [list(candidate) for candidate in sorted(candidates)]
    raise RuntimeError(f"balanced candidate shortage: {len(candidates)} < {count}")


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
        negatives = generate_balanced_candidates(
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


def stat_recommendations(draws: list[dict], rng: np.random.Generator, count: int = STAT_SETS) -> list[dict]:
    """최근 STAT_WINDOW회차 빈도 기반 통계 균형 추천 count세트 생성 (draws는 오름차순)."""
    recent = draws[-STAT_WINDOW:]
    counts = {n: 0 for n in range(1, NUM_RANGE + 1)}
    for d in recent:
        for n in d["numbers"]:
            counts[n] += 1
    hot = sorted(counts, key=lambda n: (-counts[n], n))[:12]
    cold = sorted(counts, key=lambda n: (counts[n], n))[:12]
    middle = [n for n in range(1, NUM_RANGE + 1) if n not in hot and n not in cold]
    historical = {tuple(d["numbers"]) for d in draws}
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

    def pick_candidate(seen: set[tuple[int, ...]]) -> list[int]:
        pool: list[list[int]] = []
        pooled: set[tuple[int, ...]] = set()
        for _ in range(500):
            cand = sorted(
                int(n)
                for group, k in ((hot, 2), (cold, 2), (middle, 2))
                for n in rng.choice(group, size=k, replace=False)
            )
            key = tuple(cand)
            if (
                key not in pooled
                and key not in seen
                and key not in historical
                and is_balanced(cand)
                and carry_pool & set(cand)
                and bool(consecutive_runs(cand)) == want_run
            ):
                pool.append(cand)
                pooled.add(key)
                if len(pool) >= STAT_POOL:
                    break
        if pool:  # 5회 창 비율이 전체 비율에 가장 가까운 후보 선택
            return min(pool, key=lambda c: (window_deviation(profile, recent4 + c), tuple(c)))
        for _ in range(500):  # 균형 조건을 만족 못 하면 역대 조합만 피한 대체 조합
            cand = sorted(int(n) + 1 for n in rng.choice(NUM_RANGE, size=PICK, replace=False))
            if tuple(cand) not in historical and tuple(cand) not in seen:
                return cand
        raise RuntimeError("no unique statistical recommendation available")

    recs: list[dict] = []
    seen: set[tuple[int, ...]] = set()
    for _ in range(count):
        cand = pick_candidate(seen)
        seen.add(tuple(cand))
        odd = sum(1 for n in cand if n % 2)
        low = sum(1 for n in cand if n <= 22)
        recs.append(
            {
                "method": "balanced-statistical",
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
    candidates: list[list[int]], scores: np.ndarray
) -> list[dict]:
    flat_scores = np.asarray(scores).reshape(-1)
    order = np.argsort(flat_scores)[::-1]
    best_index = int(order[0])
    best = candidates[best_index]
    diverse_index = next(
        (
            int(i)
            for i in order[1:]
            if len(set(best) & set(candidates[int(i)])) <= 2
        ),
        None,
    )
    if diverse_index is None:
        raise RuntimeError("no diverse scored candidate available")
    return [
        {
            "method": "lstm-combination-best",
            "numbers": best,
            "modelScore": round(float(flat_scores[best_index]), 6),
        },
        {
            "method": "lstm-combination-diverse",
            "numbers": candidates[diverse_index],
            "modelScore": round(float(flat_scores[diverse_index]), 6),
        },
    ]


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

    # 통계 추천: 3세트, 유효성, 재현성
    rng_a = np.random.default_rng(SEED)
    fake_draws = [
        {"round": r, "numbers": sorted(int(n) + 1 for n in rng_a.choice(45, size=6, replace=False))}
        for r in range(1, 121)
    ]
    recs_a = stat_recommendations(fake_draws, np.random.default_rng(7))
    recs_b = stat_recommendations(fake_draws, np.random.default_rng(7))
    assert len(recs_a) == STAT_SETS
    latest_numbers = set(fake_draws[-1]["numbers"])
    profile = overall_profile(fake_draws)
    recent4 = [n for d in fake_draws[-WINDOW_RECENT:] for n in d["numbers"]]
    runs4 = sum(1 for d in fake_draws[-WINDOW_RECENT:] if consecutive_runs(d["numbers"]))
    want_run = profile["runRate"] * (WINDOW_RECENT + 1) - runs4 >= 0.5
    slots = len(recent4) + 6
    deficit_digits = {
        d for d in range(10)
        if profile["digitShare"][d] * slots - sum(1 for n in recent4 if n % 10 == d) >= 1
    }
    for rec in recs_a:
        nums = rec["numbers"]
        assert len(nums) == 6 and len(set(nums)) == 6 and all(1 <= n <= 45 for n in nums), nums
        reason = rec["reason"]
        assert reason["carryOverNumbers"] and set(reason["carryOverNumbers"]) <= latest_numbers, reason
        assert reason["endingNumbers"] == [n for n in nums if n % 10 in deficit_digits], reason
        assert reason["consecutiveRuns"] == consecutive_runs(nums), reason
        assert bool(consecutive_runs(nums)) == want_run, (nums, want_run)

    # 창 편차 검증: 창 비율이 전체 비율과 같으면 편차 0
    uniform_draws = [{"round": r, "numbers": [1, 2, 13, 24, 35, 41]} for r in range(1, 11)]
    uprofile = overall_profile(uniform_draws)
    urecent = [n for d in uniform_draws[-WINDOW_RECENT:] for n in d["numbers"]]
    assert window_deviation(uprofile, urecent + [1, 2, 13, 24, 35, 41]) < 1e-9
    assert recs_a == recs_b, "stat recommendations must be reproducible with same rng"
    assert len({tuple(r["numbers"]) for r in recs_a}) == STAT_SETS, "sets must be unique"

    forbidden = {tuple(fake_draws[-1]["numbers"])}
    candidates_a = generate_balanced_candidates(
        np.random.default_rng(SEED), 20, forbidden
    )
    candidates_b = generate_balanced_candidates(
        np.random.default_rng(SEED), 20, forbidden
    )
    assert candidates_a == candidates_b
    assert len(candidates_a) == len({tuple(c) for c in candidates_a}) == 20
    assert all(is_balanced(c) for c in candidates_a)
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
    assert all(is_balanced(multihot_to_numbers(v)) for v in ex_cand)

    scored_candidates = [
        [1, 8, 15, 22, 29, 36],
        [2, 9, 16, 23, 30, 37],
        [1, 8, 17, 24, 31, 38],
    ]
    selected = select_scored_recommendations(
        scored_candidates,
        np.asarray([0.9, 0.8, 0.7]),
    )
    assert [rec["method"] for rec in selected] == [
        "lstm-combination-best",
        "lstm-combination-diverse",
    ]
    assert selected[0]["numbers"] == scored_candidates[0]
    assert selected[1]["numbers"] == scored_candidates[1]
    assert len(set(selected[0]["numbers"]) & set(selected[1]["numbers"])) <= 2
    assert selected[0]["modelScore"] == 0.9

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

    balanced_mask = np.asarray(
        [is_balanced(multihot_to_numbers(target)) for target in y]
    )
    balanced_x, balanced_y = x[balanced_mask], y[balanced_mask]
    if len(balanced_x) < MIN_TRAIN_SAMPLES:
        raise SystemExit(
            f"균형 학습 샘플 부족: {len(balanced_x)} < {MIN_TRAIN_SAMPLES}"
        )

    split = int(len(balanced_x) * (1.0 - VALIDATION_FRACTION))
    if split <= 0 or split >= len(balanced_x):
        raise SystemExit("시간순 검증 구간을 만들 수 없습니다")
    train_seq, train_pos = balanced_x[:split], balanced_y[:split]
    val_seq, val_pos = balanced_x[split:], balanced_y[split:]
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
    candidates = generate_balanced_candidates(
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
    lstm_recommendations = select_scored_recommendations(candidates, scores)

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
        "trainSampleCount": int(len(balanced_x)),
        "validationAuc": round(validation_auc, 4),
        "candidateCount": len(candidates),
        "recommendations": [
            *lstm_recommendations,
            *stat_recommendations(draws, np.random.default_rng(SEED + 3)),
        ],
        "warning": WARNING,
    }

    update_history(result, draws)  # OUT_PATH 덮어쓰기 전에 기존 예측을 이력에 보존
    OUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"saved {OUT_PATH.name}: source={source_latest} target={target_round} "
        f"balanced_samples={len(balanced_x)} epochs={result['epochs']} "
        f"validation_auc={validation_auc:.4f} candidates={len(candidates)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
