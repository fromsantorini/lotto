#!/usr/bin/env python3
"""LSTM 로또 실험 학습/예측 스크립트.

lotto-data.json 을 읽어 회차 순서와 LSTM 상태를 유지하며 번호별 출현 점수를 학습하고,
중복 없는 추천과 시간순 평가를 lstm-prediction.json 으로 저장한다.

주의: 공정한 독립 추첨에서는 과거 번호만으로 다음 당첨번호의 예측 우위를
기대할 수 없다. 모델과 통계 추천은 실험용이며 당첨 확률 상승을 의미하지 않는다.
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
DEFAULT_WINDOW = 1      # 회차당 한 스텝, 이전 회차 정보는 LSTM 상태로 전달
DEFAULT_EPOCHS = 100
DEFAULT_BATCH = 1
MIN_TRAIN_SAMPLES = 50  # 이보다 적으면 학습 의미가 없어 중단

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "lotto-data.json"
OUT_PATH = ROOT / "lstm-prediction.json"
HISTORY_PATH = ROOT / "lstm-prediction-history.json"
HISTORY_LIMIT = 200  # 이력 최대 보관 회차 수

VALIDATION_FRACTION = 0.1
MAX_CANDIDATE_ATTEMPT_FACTOR = 100

MODEL_NAME = "keras-stateful-lstm-tykimos-v3"
LSTM_METHOD = "lstm-stateful-ball-weighted-v3"
REFIT_METHOD = "lstm-refit-25-v1"
REFIT_INTERVAL = 25  # 학습 마감 회차를 25의 배수로 고정해 실전과 평가가 같은 주기로 재학습
REFERENCE_URL = "https://tykimos.github.io/2020/01/25/keras_lstm_lotto_v895/"
WARNING = (
    "번호별 모델 점수는 추천 가중치이며 조합의 당첨 확률이 아닙니다. "
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
    """이전 회차 한 개를 입력하고 다음 회차를 정답으로 사용한다."""
    if window != 1:
        raise ValueError("stateful LSTM requires window=1")
    return vectors[:-1, np.newaxis, :], vectors[1:]


# --- 통계 가중 추천 -----------------------------------------------------------
# 번호 형태는 강제하지 않는다. 역대 1등 조합만 제외하고, 기존 통계 지표는
# 후보의 선택 가중치에만 반영한다. 모든 유효 후보의 선택 가중치는 0보다 크다.

STAT_WINDOW = 100   # 강세/소외 판정에 쓰는 최근 회차 수
STAT_SETS = 3
PATTERN_METHOD = "walk-forward-backtest-pattern-v1"
PATTERN_POOL_SIZE = 1024
PATTERN_LOOKBACK = 50
PATTERN_TRAIN_ROUNDS = 200
SELECTION_WARMUP = 50
EVALUATION_SEEDS = (42, 137, 2026)
COMPARISON_SETS = 2


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


def stat_recommendations(
    draws: list[dict],
    rng: np.random.Generator,
    count: int = STAT_SETS,
    forbidden: set[tuple[int, ...]] | frozenset[tuple[int, ...]] = frozenset(),
) -> list[dict]:
    """최근 100회 빈도에 번호당 1을 더한 가중 추출. 부족한 패턴은 보정하지 않는다."""
    weights = np.ones(NUM_RANGE)
    for draw in draws[-STAT_WINDOW:]:
        weights[np.asarray(draw["numbers"]) - 1] += 1
    hot = set(np.argsort(-weights, kind="stable")[:12] + 1)
    cold = set(np.argsort(weights, kind="stable")[:12] + 1)
    historical = {tuple(d["numbers"]) for d in draws} | set(forbidden)
    carry_pool = set(draws[-1]["numbers"])
    return [
        {
            "method": "weighted-statistical-v2",
            "numbers": cand,
            "reason": {
                "oddEven": f"{sum(n % 2 for n in cand)}:{sum(n % 2 == 0 for n in cand)}",
                "lowHigh": f"{sum(n <= 22 for n in cand)}:{sum(n > 22 for n in cand)}",
                "sum": sum(cand),
                "frequentNumbers": [n for n in cand if n in hot],
                "coldNumbers": [n for n in cand if n in cold],
                "carryOverNumbers": [n for n in cand if n in carry_pool],
                "endingNumbers": [],
                "consecutiveRuns": consecutive_runs(cand),
            },
        }
        for cand in generate_weighted_sets(weights, rng, count, historical)
    ]


def generate_weighted_sets(
    weights: np.ndarray,
    rng: np.random.Generator,
    count: int,
    forbidden: set[tuple[int, ...]] | frozenset[tuple[int, ...]] = frozenset(),
) -> list[list[int]]:
    probabilities = np.asarray(weights, dtype=np.float64)
    if (probabilities.shape != (NUM_RANGE,) or not np.all(np.isfinite(probabilities))
            or np.any(probabilities <= 0) or count < 1):
        raise ValueError("45 positive number weights required")
    probabilities = probabilities / probabilities.max()
    probabilities = probabilities / probabilities.sum()
    selected: set[tuple[int, ...]] = set()
    for _ in range(count * MAX_CANDIDATE_ATTEMPT_FACTOR):
        numbers = tuple(
            sorted(
                int(i) + 1
                for i in rng.choice(
                    NUM_RANGE, PICK, replace=False, p=probabilities
                )
            )
        )
        if numbers not in forbidden:
            selected.add(numbers)
            if len(selected) == count:
                return [list(numbers) for numbers in sorted(selected)]
    raise RuntimeError(f"weighted recommendation shortage: {len(selected)} < {count}")


def match_summary(matches: np.ndarray, baseline: np.ndarray, *, compare_three_plus: bool = False) -> dict:
    """같은 회차의 세트/시드를 묶어 재표집하며 독립 표본으로 과대 계산하지 않는다."""
    count = len(matches)
    if not count:
        return {"meanMatches": None, "threePlusRate": None,
                "differenceVsRandom": None, "differenceCI95": None, "evidence": "insufficient-data"}
    if compare_three_plus:
        differences = (matches >= 3).astype(int) - (baseline >= 3).astype(int)
    else:
        differences = matches - baseline
    differences = differences.reshape(count, -1).mean(axis=1)
    interval = None
    if count >= 2:
        # ponytail: 회차 단위 IID bootstrap; 회차 의존성을 분석할 때 block bootstrap으로 교체.
        indices = np.random.default_rng(SEED).integers(count, size=(2000, count))
        interval = [round(float(v), 4) for v in np.quantile(differences[indices].mean(axis=1), [0.025, 0.975])]
    return {
        "meanMatches": round(float(np.mean(matches)), 4),
        "threePlusRate": round(float(np.mean(matches >= 3)), 4),
        "differenceVsRandom": round(float(np.mean(differences)), 4),
        "differenceCI95": interval,
        "evidence": "above-random" if count >= SELECTION_WARMUP and interval[0] > 0 else "not-established",
    }


def build_backtest_recommendations(draws: list[dict]) -> tuple[dict, list[dict]]:
    """각 회차에 같은 후보를 채점한다. 해당 회차의 당첨번호는 선택 후에만 학습한다."""
    rng = np.random.default_rng(SEED)
    candidates = {}
    while len(candidates) < PATTERN_POOL_SIZE:
        ticket = tuple(sorted(int(n) for n in rng.choice(NUM_RANGE, PICK, replace=False)))
        candidates.setdefault(ticket, len(candidates))
    tickets = np.asarray(list(candidates))
    frequency_ties = rng.permutation(NUM_RANGE)
    baseline_rng = np.random.default_rng(SEED + 1)
    eligible = np.ones(len(tickets), dtype=bool)

    zones = np.arange(NUM_RANGE) // 10
    zone_share = np.bincount(zones) / NUM_RANGE
    zone_std = np.sqrt(20 * PICK * zone_share * (1 - zone_share) * (NUM_RANGE - PICK) / (NUM_RANGE - 1))
    ticket_zones = zones[tickets]
    repeated_endings = [PICK - len(set((row + 1) % 10)) for row in tickets]
    adjacent_pairs = (np.diff(tickets, axis=1) == 1).sum(axis=1)
    largest_zone = np.max([(ticket_zones == z).sum(axis=1) for z in range(len(zone_share))], axis=0)

    def indicators(counts):
        # 0개를 기준으로 1개, 2개, 3개 이상에 서로 다른 가중치를 학습한다.
        return np.minimum(counts, 3)[:, None] == np.arange(1, 4)

    static = np.column_stack([indicators(counts) for counts in (repeated_endings, adjacent_pairs, largest_zone)])
    vectors = np.array([to_multihot(draw["numbers"]) for draw in draws], dtype=int)
    gaps = np.zeros(NUM_RANGE, dtype=int)
    feature_count = static.shape[1] + 5 * 3
    gram = np.zeros((feature_count, feature_count))
    hit_products = np.zeros(feature_count)
    matches, baseline = [], []
    recommendation = []
    first_test = PATTERN_LOOKBACK + PATTERN_TRAIN_ROUNDS
    for index in range(len(draws) + 1):
        if index >= PATTERN_LOOKBACK:
            frequency = vectors[index - PATTERN_LOOKBACK:index].sum(axis=0)
            cold = np.zeros(NUM_RANGE, dtype=bool)
            cold[np.lexsort((frequency_ties, frequency))[:9]] = True
            zone_counts = np.bincount(zones, weights=vectors[index - 20:index].sum(axis=0))
            zone_order = np.argsort((zone_counts - 20 * PICK * zone_share) / zone_std, kind="stable")
            features = np.column_stack((
                static,
                indicators(vectors[index - 1, tickets].sum(axis=1)),
                indicators(cold[tickets].sum(axis=1)),
                indicators((gaps[tickets] >= 15).sum(axis=1)),
                indicators((ticket_zones == zone_order[0]).sum(axis=1)),
                indicators((ticket_zones == zone_order[-1]).sum(axis=1)),
            )).astype(float)
            features -= features.mean(axis=0)
            if index >= first_test:
                weights = np.linalg.solve(gram + np.diag(0.1 * gram.diagonal() + 1e-6), hit_products)
                if not eligible.any():
                    raise ValueError("no eligible pattern candidates")
                chosen = np.argmax(np.where(eligible, features @ weights, -np.inf))
                if index == len(draws):
                    recommendation = [{"method": PATTERN_METHOD, "numbers": (tickets[chosen] + 1).tolist()}]
                    break
                random_ticket = baseline_rng.choice(np.flatnonzero(eligible))
                matches.append(int(vectors[index, tickets[chosen]].sum()))
                baseline.append(int(vectors[index, tickets[random_ticket]].sum()))
            if index < len(draws):
                hits = (vectors[index, tickets].sum(axis=1) >= 3).astype(float)
                hit_products += features.T @ (hits - hits.mean())
                gram += features.T @ features
        if index < len(draws):
            gaps = np.where(vectors[index], 0, gaps + 1)
            prior_winner = candidates.get(tuple(np.flatnonzero(vectors[index])))
            if prior_winner is not None:
                eligible[prior_winner] = False

    matches, baseline = np.asarray(matches), np.asarray(baseline)
    summary = match_summary(matches, baseline, compare_three_plus=True)
    summary.update({
        "metric": "three-plus-rate-per-set",
        "evaluationMode": "prequential-pattern-v1",
        "selectedMethod": "pattern-score",
        "candidateCount": len(tickets),
        "warmupRounds": first_test,
        "testedRounds": len(matches),
        "setsPerMethod": 1,
        "threePlusHits": int(np.count_nonzero(matches >= 3)),
        "randomThreePlusHits": int(np.count_nonzero(baseline >= 3)),
        "randomBaseline": round(float(np.mean(baseline >= 3)), 4) if baseline.size else None,
    })
    return summary, recommendation


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
        "model": pred.get("model"),
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


def build_number_model(tf, window: int):
    if window != 1:
        raise ValueError("stateful LSTM requires window=1")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(batch_shape=(1, 1, NUM_RANGE)),
        tf.keras.layers.LSTM(128, stateful=True),
        tf.keras.layers.Dense(NUM_RANGE, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model


def fit_stateful(tf, model, x: np.ndarray, y: np.ndarray, epochs: int):
    class ResetState(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.model.layers[0].reset_states()

    return model.fit(x, y, epochs=epochs, batch_size=1, shuffle=False,
                     callbacks=[ResetState()], verbose=2)


def predict_stateful(model, inputs: np.ndarray) -> np.ndarray:
    """고정 가중치로 처음부터 재생해 학습 상태나 이전 추론 상태를 제거한다."""
    model.layers[0].reset_states()
    return model.predict(inputs, batch_size=1, verbose=0)


def ball_counts(scores: np.ndarray) -> np.ndarray:
    """원문의 공 개수 int(score * 100 + 1). 점수 0인 번호에도 공 1개."""
    values = np.asarray(scores, dtype=np.float64)
    if (values.shape != (NUM_RANGE,) or not np.all(np.isfinite(values))
            or np.any((values < 0) | (values > 1))):
        raise ValueError("45 finite scores in [0, 1] required")
    return (values * 100 + 1).astype(np.int64)


def lstm_recommendations(
    scores: np.ndarray, rng: np.random.Generator,
    count: int = 2, forbidden: set[tuple[int, ...]] | frozenset[tuple[int, ...]] = frozenset(),
) -> list[dict]:
    # 중복 번호를 다시 뽑는 원문의 공 추출과 동일한 분포의 가중 비복원 추출.
    return [{"method": LSTM_METHOD, "numbers": numbers}
            for numbers in generate_weighted_sets(ball_counts(scores), rng, count, forbidden)]


def number_metrics(tf, targets: np.ndarray, scores: np.ndarray) -> dict:
    auc = tf.keras.metrics.AUC(multi_label=True, num_labels=NUM_RANGE)
    auc.update_state(targets, scores)
    return {"auc": float(auc.result().numpy()),
            "loss": float(tf.reduce_mean(tf.keras.losses.binary_crossentropy(targets, scores)).numpy())}


def comparison_matches(draws: list[dict], target_index: int, weights: np.ndarray) -> np.ndarray:
    history = draws[:target_index]
    forbidden = {tuple(draw["numbers"]) for draw in history}
    winning = set(draws[target_index]["numbers"])
    result = np.zeros((3, len(EVALUATION_SEEDS), COMPARISON_SETS))
    for seed_index, seed in enumerate(EVALUATION_SEEDS):
        seed = seed + draws[target_index]["round"] * 100
        groups = [
            generate_weighted_sets(np.ones(NUM_RANGE), np.random.default_rng(seed), COMPARISON_SETS, forbidden),
            [r["numbers"] for r in stat_recommendations(history, np.random.default_rng(seed), COMPARISON_SETS)],
            [r["numbers"] for r in lstm_recommendations(weights, np.random.default_rng(seed), COMPARISON_SETS, forbidden)],
        ]
        for method, group in enumerate(groups):
            result[method, seed_index] = [len(winning & set(numbers)) for numbers in group]
    return result


def chronological_split(sample_count: int) -> tuple[int, int]:
    block = max(1, int(sample_count * VALIDATION_FRACTION))
    train_end, test_start = sample_count - 2 * block, sample_count - block
    if train_end < MIN_TRAIN_SAMPLES:
        raise ValueError("학습/검증/평가 구간을 만들기에 회차가 부족합니다")
    return train_end, test_start


def refit_cutoff(target_round: int) -> int:
    """target_round 예측에 쓰는 모델의 학습 마감 회차. target_round 이전의 25의 배수."""
    return (target_round - 1) // REFIT_INTERVAL * REFIT_INTERVAL


def refit_train_count(rounds: np.ndarray, cutoff: int) -> int:
    """정답 회차가 cutoff 이하인 학습 샘플 수. 샘플 i의 정답은 rounds[i + 1]."""
    return int(np.searchsorted(rounds, cutoff, side="right")) - 1


def round_rng(target_round: int, offset: int) -> np.random.Generator:
    """회차별 난수열. 고정 시드는 가중치가 비슷한 주마다 같은 조합을 반복한다."""
    return np.random.default_rng(SEED + offset + target_round * 100)


def selftest() -> int:
    """TF 없이 시간순 평가, 추천 중복, 이력 보존을 검증한다."""
    draws = [
        {"round": r, "numbers": sorted(int(n) + 1 for n in
         np.random.default_rng(SEED + r).choice(NUM_RANGE, PICK, replace=False))}
        for r in range(1, 261)
    ]
    vectors = np.stack([to_multihot(d["numbers"]) for d in draws])
    x, y = build_dataset(vectors, DEFAULT_WINDOW)
    assert np.array_equal(x[0], vectors[:DEFAULT_WINDOW])
    assert np.array_equal(y[0], vectors[DEFAULT_WINDOW])
    assert chronological_split(len(x)) == (209, 234)
    assert np.array_equal(x[-1, 0], vectors[-2]) and np.array_equal(y[-1], vectors[-1])
    for bad in (np.full(NUM_RANGE, np.nan), np.full(NUM_RANGE, np.inf), np.zeros(NUM_RANGE)):
        try:
            generate_weighted_sets(bad, np.random.default_rng(SEED), 2)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid weights accepted")
    assert len(generate_weighted_sets(np.full(NUM_RANGE, 1e308), np.random.default_rng(SEED), 2)) == 2
    forbidden = {tuple(d["numbers"]) for d in draws}
    lstm = lstm_recommendations(np.ones(NUM_RANGE), np.random.default_rng(SEED), forbidden=forbidden)
    forbidden.update(tuple(r["numbers"]) for r in lstm)
    stats = stat_recommendations(draws, np.random.default_rng(SEED), forbidden=forbidden)
    assert stats == stat_recommendations(draws, np.random.default_rng(SEED), forbidden=forbidden)
    assert all(r["reason"]["endingNumbers"] == [] for r in stats)
    # 빈도 가중치가 거의 같은 연속 두 주에도 같은 조합을 반복하지 않는다.
    weekly = [{tuple(r["numbers"]) for r in stat_recommendations(draws[:n], round_rng(n + 1, 4))}
              for n in (259, 260)]
    assert not weekly[0] & weekly[1]
    samples = generate_weighted_sets(np.ones(NUM_RANGE), np.random.default_rng(SEED), 30)
    assert any(not matches_legacy_balance(c) for c in samples)
    forbidden.update(tuple(r["numbers"]) for r in stats)
    summary, backtest = build_backtest_recommendations(draws)
    bundle = lstm + stats + backtest
    assert len(bundle) == len({tuple(r["numbers"]) for r in bundle}) == 6
    for rec in bundle:
        assert len(rec["numbers"]) == len(set(rec["numbers"])) == PICK
        assert all(1 <= n <= NUM_RANGE for n in rec["numbers"])
        assert tuple(rec["numbers"]) not in {tuple(d["numbers"]) for d in draws}
    assert summary["testedRounds"] == 10 and summary["setsPerMethod"] == 1
    assert len(backtest) == 1 and backtest[0]["method"] == PATTERN_METHOD
    # 직전까지의 실전 추천을 다음 회차로 채점하면 누적 백테스트 결과와 같아야 한다.
    previous, picks = build_backtest_recommendations(draws[:-1])
    hits = len(set(picks[0]["numbers"]) & set(draws[-1]["numbers"]))
    assert summary["threePlusHits"] == previous["threePlusHits"] + (hits >= 3)
    assert np.isclose(summary["meanMatches"] * 10, previous["meanMatches"] * 9 + hits, atol=0.001)
    weights = np.linspace(0, 1, NUM_RANGE)
    assert np.array_equal(comparison_matches(draws, 205, weights),
                          comparison_matches(draws[:206], 205, weights))
    comparison = comparison_matches(draws, 205, np.ones(NUM_RANGE))
    assert np.array_equal(comparison[0], comparison[2])
    matches = np.asarray([[[0, 3]], [[1, 4]]])
    metrics = match_summary(matches, np.zeros_like(matches))
    assert metrics["meanMatches"] == 2 and metrics["threePlusRate"] == 0.5
    assert metrics["differenceCI95"] == [1.5, 2.5]
    rate_metrics = match_summary(matches, np.zeros_like(matches), compare_three_plus=True)
    assert rate_metrics["differenceCI95"] == [0.5, 0.5]
    assert match_summary(matches, matches)["differenceCI95"] == [0, 0]
    fallback, picks = build_backtest_recommendations(draws[:249])
    assert not picks and fallback["selectedMethod"] == "pattern-score"
    assert fallback["meanMatches"] is None and fallback["testedRounds"] == 0
    draw = {"round": 100, "numbers": [1, 2, 3, 4, 5, 6], "bonus": 7}
    pred = {"targetRound": 100, "sourceLatestRound": 99, "model": MODEL_NAME,
            "recommendations": [{"method": "m", "numbers": [1, 2, 3, 10, 11, 7]}]}
    entries = reconcile_history(upsert_entry([], prediction_to_entry(pred), replace=True), [draw])
    match = entries[0]["result"]["matches"][0]
    assert match["matchCount"] == 3 and match["bonusMatched"]
    assert entries[0]["model"] == MODEL_NAME
    assert upsert_entry(entries, prediction_to_entry(pred), replace=False) == entries
    entries = upsert_entry(entries, prediction_to_entry({**pred, "targetRound": 101}), replace=True)
    assert reconcile_history(entries, [draw])[-1]["result"] is None
    scores = np.zeros(NUM_RANGE)
    scores[:4] = [0, 0.2, 0.99, 1]
    assert ball_counts(scores)[:4].tolist() == [1, 21, 100, 101]
    assert np.all(ball_counts(np.zeros(NUM_RANGE)) == 1)
    for bad in (np.full(NUM_RANGE, np.nan), np.full(NUM_RANGE, -0.1), np.full(NUM_RANGE, 1.1)):
        try:
            ball_counts(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid model scores accepted")
    recs = lstm_recommendations(scores, np.random.default_rng(7))
    expected = generate_weighted_sets(ball_counts(scores), np.random.default_rng(7), 2)
    assert [rec["numbers"] for rec in recs] == expected
    # 원문의 공 박스와 가중치 변환이 동일한지 확인한다.
    box = np.repeat(np.arange(NUM_RANGE), ball_counts(scores))
    assert np.array_equal(np.bincount(box, minlength=NUM_RANGE), ball_counts(scores))
    # 25회 재학습: 예측 대상 회차와 그 이후 회차는 학습 샘플에 들어가지 않는다.
    assert [refit_cutoff(r) for r in (1250, 1251, 1252, 1276)] == [1225, 1250, 1250, 1275]
    rounds = np.array([d["round"] for d in draws])
    count = refit_train_count(rounds, refit_cutoff(251))
    assert count == 249 and np.array_equal(y[count - 1], vectors[rounds.tolist().index(250)])
    print("selftest ok")
    return 0


def model_selftest() -> int:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    model = build_number_model(tf, 1)
    assert model.input_shape == (1, 1, NUM_RANGE)
    assert model.layers[0].stateful and model.layers[0].units == 128
    vectors = np.stack([to_multihot([r, r + 1, r + 2, r + 3, r + 4, r + 5]) for r in range(1, 10)])
    x, y = build_dataset(vectors, 1)
    initial = model.get_weights()
    fit_stateful(tf, model, x, y, 2)
    assert any(not np.array_equal(a, b) for a, b in zip(initial, model.get_weights()))
    scores = predict_stateful(model, vectors[:, np.newaxis, :])
    assert scores.shape == (len(vectors), NUM_RANGE)
    assert np.all(np.isfinite(scores)) and np.all((scores >= 0) & (scores <= 1))
    prefix = predict_stateful(model, vectors[:-1, np.newaxis, :])
    assert np.allclose(prefix, scores[:-1], atol=1e-6), "future input changed an earlier prediction"
    assert np.allclose(predict_stateful(model, vectors[:, np.newaxis, :]), scores, atol=1e-6)
    last_only = predict_stateful(model, vectors[-1:, np.newaxis, :])
    assert not np.allclose(last_only[0], scores[-1]), "history state was not carried"
    # 에포크 시작 시 오염된 상태를 초기화하는지 실제 학습으로 확인한다.
    clone = build_number_model(tf, 1)
    clone.set_weights(initial)
    model.set_weights(initial)
    model.optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=model.optimizer, loss="binary_crossentropy")
    predict_stateful(model, vectors[:, np.newaxis, :])
    fit_stateful(tf, model, x, y, 1)
    fit_stateful(tf, clone, x, y, 1)
    assert all(np.allclose(a, b, atol=1e-6) for a, b in zip(model.get_weights(), clone.get_weights()))
    print("stateful model selftest ok")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="LSTM 로또 실험 학습/예측")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--selftest", action="store_true", help="추천/이력 핵심 로직 검증 (TF 불필요)")
    parser.add_argument("--model-selftest", action="store_true", help="TensorFlow 상태 전달/초기화 검증")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    if args.model_selftest:
        return model_selftest()
    if args.window != 1 or args.batch_size != 1 or args.epochs < 1:
        parser.error("stateful LSTM requires window=1, batch-size=1 and positive epochs")

    import tensorflow as tf

    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    draws = load_draws(DATA_PATH)
    source_latest = draws[-1]["round"]
    vectors = np.stack([to_multihot(d["numbers"]) for d in draws])
    x, y = build_dataset(vectors, args.window)
    train_end, test_start = chronological_split(len(x))
    model = build_number_model(tf, args.window)
    # 원문의 반복 학습 실험: epoch 수를 사전에 고정하고 검증/시험 결과로 고르지 않는다.
    fit_stateful(tf, model, x[:train_end], y[:train_end], args.epochs)
    # 학습 때 누적된 상태를 버리고 고정된 최종 가중치로 과거부터 순서대로 재생.
    evaluation_scores = predict_stateful(model, x)
    validation = number_metrics(tf, y[train_end:test_start], evaluation_scores[train_end:test_start])
    test_scores = evaluation_scores[test_start:]
    test_metrics = number_metrics(tf, y[test_start:], test_scores)
    matches = np.asarray([
        comparison_matches(draws, args.window + test_start + offset, weights)
        for offset, weights in enumerate(test_scores)
    ])
    # 25회 재학습: 회차마다 마감 회차까지 처음부터 학습한 모델로 예측한다. 실전 추천도 같은 규칙.
    # ponytail: 매주 마감 회차별 모델을 다시 학습(약 6회). 느려지면 마감별 가중치를 캐시.
    rounds = np.array([d["round"] for d in draws])
    test_targets = [int(r) for r in rounds[args.window + test_start:]]
    live_cutoff = refit_cutoff(source_latest + 1)
    refit_test = np.zeros_like(test_scores)
    for cutoff in sorted({refit_cutoff(r) for r in test_targets} | {live_cutoff}):
        count = refit_train_count(rounds, cutoff)
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)
        refit_model = build_number_model(tf, args.window)
        fit_stateful(tf, refit_model, x[:count], y[:count], args.epochs)
        replay = predict_stateful(refit_model, vectors[:, np.newaxis, :])
        for offset, target in enumerate(test_targets):
            if refit_cutoff(target) == cutoff:
                refit_test[offset] = replay[test_start + offset]
        if cutoff == live_cutoff:
            refit_weights = replay[-1]
    refit_matches = np.asarray([
        comparison_matches(draws, args.window + test_start + offset, weights)[2]
        for offset, weights in enumerate(refit_test)
    ])
    comparison = {
        "evaluationMode": "chronological-stateful-holdout-v3",
        "model": MODEL_NAME,
        "trainingThroughRound": draws[args.window + train_end - 1]["round"],
        "validationThroughRound": draws[args.window + test_start - 1]["round"],
        "testStartRound": draws[args.window + test_start]["round"],
        "testEndRound": source_latest,
        "testedRounds": len(test_scores),
        "setsPerMethod": COMPARISON_SETS,
        "seedCount": len(EVALUATION_SEEDS),
        "lstmRefitDuringTest": False,
        "testAuc": round(float(test_metrics["auc"]), 4),
        "brierScore": round(float(np.mean((test_scores - y[test_start:]) ** 2)), 6),
        "uniformBrierScore": round(float(np.mean((PICK / NUM_RANGE - y[test_start:]) ** 2)), 6),
        "methods": [
            {"method": method, **match_summary(matches[:, index], matches[:, 0])}
            for index, method in enumerate(("uniform", "weighted-statistical-v2", LSTM_METHOD))
        ] + [{"method": REFIT_METHOD, **match_summary(refit_matches, matches[:, 0])}],
        "refitInterval": REFIT_INTERVAL,
    }
    # 최종 추천은 전체 회차로 동일한 epoch 수만큼 새로 학습한다.
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    model = build_number_model(tf, args.window)
    fit_stateful(tf, model, x, y, args.epochs)
    # 최신 회차도 마지막 입력에 포함해야 다음 회차를 예측한다.
    weights = predict_stateful(model, vectors[:, np.newaxis, :])[-1]
    historical = {tuple(d["numbers"]) for d in draws}
    backtest, backtest_recs = build_backtest_recommendations(draws)
    reserved = {tuple(rec["numbers"]) for rec in backtest_recs}
    lstm = lstm_recommendations(weights, round_rng(source_latest + 1, 3), forbidden=historical | reserved)
    reserved |= {tuple(rec["numbers"]) for rec in lstm}
    refit = [{"method": REFIT_METHOD, "numbers": numbers} for numbers in generate_weighted_sets(
        ball_counts(refit_weights), round_rng(source_latest + 1, 5), 2, historical | reserved)]
    selected = reserved | {tuple(rec["numbers"]) for rec in refit}
    stats = stat_recommendations(draws, round_rng(source_latest + 1, 4), forbidden=selected)
    recommendations = lstm + refit + stats + backtest_recs
    number_sets = [set(rec["numbers"]) for rec in recommendations]
    if len({tuple(rec["numbers"]) for rec in recommendations}) != len(recommendations):
        raise ValueError("duplicate recommendation in bundle")
    result = {
        "schemaVersion": 2,
        "model": MODEL_NAME,
        "sourceLatestRound": source_latest,
        "targetRound": source_latest + 1,
        "trainedAt": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "window": args.window,
        "epochs": args.epochs,
        "stateful": True,
        "lstmUnits": 128,
        "batchSize": 1,
        "referenceUrl": REFERENCE_URL,
        "sampling": "int(score * 100 + 1)",
        "numberScores": [float(score) for score in weights],
        "ballCounts": ball_counts(weights).tolist(),
        "refitTrainedThroughRound": live_cutoff,
        "trainSampleCount": len(x),
        "validationAuc": round(float(validation["auc"]), 4),
        "comparison": comparison,
        "backtest": backtest,
        "diversity": {
            "uniqueNumbers": len(set.union(*number_sets)),
            "maxSharedNumbers": max(len(a & b) for i, a in enumerate(number_sets) for b in number_sets[i + 1:]),
        },
        "recommendations": recommendations,
        "warning": WARNING,
    }
    update_history(result, draws)
    OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved {OUT_PATH.name}: source={source_latest} target={source_latest + 1} "
          f"samples={len(x)} epochs={args.epochs} test_auc={comparison['testAuc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
