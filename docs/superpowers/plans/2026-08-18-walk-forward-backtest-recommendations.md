# Walk-Forward Backtest Recommendations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the existing two LSTM and three statistical recommendations while adding five separately displayed recommendations selected by past walk-forward mean match count.

**Architecture:** Add small NumPy-only weighting, sampling, and walk-forward functions to the existing training script, then append their five recommendations to the existing output array so history reconciliation keeps working unchanged. Extend the existing single-file browser renderer to split the new method into its own section and scoreboard category.

**Tech Stack:** Python 3.11, NumPy, existing TensorFlow training flow, static HTML/browser JavaScript, Node.js built-in `assert`.

## Global Constraints

- Keep the existing LSTM two sets and statistical three sets unchanged.
- Add exactly five `walk-forward-backtest` recommendation sets.
- Optimize `mean-matches-per-set` using only draws earlier than each evaluated draw.
- Compare uniform, all-history frequency, frequency windows `10`, `25`, `50`, `100`, `200`, and decay half-lives `10`, `25`, `50`, `100`, `200`.
- Fall back to uniform unless the winner beats both the uniform backtest mean and `0.8`.
- Apply no odd/even, sum, consecutive, ending, zone, hot/cold, or carry-over rule.
- Require six unique integers in `1..45`, five distinct sets, and no exact historical first-prize combination.
- Keep schema version `2`, existing history storage, and existing dependencies.

---

### Task 1: NumPy walk-forward recommendation generator

**Files:**
- Modify: `tools/train-lstm-lotto.py:25-560`

**Interfaces:**
- Consumes: ascending `draws: list[dict]`, existing `SEED`, `NUM_RANGE`, `PICK`, and historical `numbers` arrays.
- Produces: `backtest_method_weights(draws, kind, parameter) -> np.ndarray`, `generate_weighted_sets(weights, rng, count, forbidden) -> list[list[int]]`, `evaluate_backtest_round(draws, target_index) -> np.ndarray`, `choose_backtest_method(means) -> int`, and `build_backtest_recommendations(draws) -> tuple[dict, list[dict]]`.
- Adds `result["backtest"]` and five appended `result["recommendations"]` entries without changing the existing five entries.

- [ ] **Step 1: Add failing self-checks for selection, fallback, validity, historical exclusion, and reproducibility**

Extend `selftest()` after the existing scored-candidate assertions:

```python
    assert choose_backtest_method(np.asarray([0.79, 0.81, 0.9])) == 2
    assert choose_backtest_method(np.asarray([0.81, 0.80, 0.79])) == 0
    assert choose_backtest_method(np.asarray([0.8, 0.8, 0.8])) == 0

    backtest_draws = [
        {
            "round": r,
            "numbers": sorted(
                int(n) + 1
                for n in np.random.default_rng(SEED + r).choice(45, size=6, replace=False)
            ),
        }
        for r in range(1, 221)
    ]
    summary_a, backtest_a = build_backtest_recommendations(backtest_draws)
    summary_b, backtest_b = build_backtest_recommendations(backtest_draws)
    historical_backtest = {tuple(d["numbers"]) for d in backtest_draws}
    assert summary_a == summary_b
    assert backtest_a == backtest_b
    assert summary_a["metric"] == "mean-matches-per-set"
    assert summary_a["testedRounds"] == 20
    assert len(backtest_a) == BACKTEST_SETS
    assert len({tuple(rec["numbers"]) for rec in backtest_a}) == BACKTEST_SETS
    assert all(rec["method"] == "walk-forward-backtest" for rec in backtest_a)
    assert all(tuple(rec["numbers"]) not in historical_backtest for rec in backtest_a)
    assert all(
        len(rec["numbers"]) == PICK
        and len(set(rec["numbers"])) == PICK
        and all(1 <= n <= NUM_RANGE for n in rec["numbers"])
        for rec in backtest_a
    )

    scores_with_future = evaluate_backtest_round(backtest_draws, 205)
    scores_without_future = evaluate_backtest_round(backtest_draws[:206], 205)
    assert np.array_equal(scores_with_future, scores_without_future)

    fallback_summary, fallback_recs = build_backtest_recommendations(backtest_draws[:100])
    assert fallback_summary["selectedMethod"] == "uniform"
    assert fallback_summary["testedRounds"] == 0
    assert len(fallback_recs) == BACKTEST_SETS
```

- [ ] **Step 2: Run the self-check and confirm the new API is missing**

Run: `python tools/train-lstm-lotto.py --selftest`

Expected: FAIL with `NameError: name 'choose_backtest_method' is not defined`.

- [ ] **Step 3: Add constants and the minimal backtest implementation**

Add beside the existing statistical constants:

```python
BACKTEST_SETS = 5
BACKTEST_MIN_HISTORY = 200
BACKTEST_WINDOWS = (10, 25, 50, 100, 200)
BACKTEST_HALF_LIVES = (10, 25, 50, 100, 200)
BACKTEST_METHODS = (
    ("uniform", None),
    ("frequency-all", None),
    *(("frequency-window", window) for window in BACKTEST_WINDOWS),
    *(("frequency-decay", half_life) for half_life in BACKTEST_HALF_LIVES),
)
```

Add before the history-management section:

```python
def backtest_method_weights(
    draws: list[dict], kind: str, parameter: int | None
) -> np.ndarray:
    weights = np.ones(NUM_RANGE, dtype=np.float64)
    if kind == "uniform":
        return weights
    if kind == "frequency-window":
        selected = draws[-int(parameter):]
        for draw in selected:
            for number in draw["numbers"]:
                weights[number - 1] += 1.0
        return weights
    if kind == "frequency-decay":
        for age, draw in enumerate(reversed(draws)):
            contribution = 2.0 ** (-age / int(parameter))
            for number in draw["numbers"]:
                weights[number - 1] += contribution
        return weights
    if kind == "frequency-all":
        for draw in draws:
            for number in draw["numbers"]:
                weights[number - 1] += 1.0
        return weights
    raise ValueError(f"unknown backtest method: {kind}")


def generate_weighted_sets(
    weights: np.ndarray,
    rng: np.random.Generator,
    count: int,
    forbidden: set[tuple[int, ...]] | frozenset[tuple[int, ...]] = frozenset(),
) -> list[list[int]]:
    probabilities = np.asarray(weights, dtype=np.float64)
    if probabilities.shape != (NUM_RANGE,) or np.any(probabilities <= 0):
        raise ValueError("45 positive number weights required")
    probabilities = probabilities / probabilities.sum()
    selected: set[tuple[int, ...]] = set()
    for _ in range(count * MAX_CANDIDATE_ATTEMPT_FACTOR):
        numbers = tuple(
            sorted(int(i) + 1 for i in rng.choice(
                NUM_RANGE, PICK, replace=False, p=probabilities
            ))
        )
        if numbers not in forbidden:
            selected.add(numbers)
            if len(selected) == count:
                return [list(numbers) for numbers in sorted(selected)]
    raise RuntimeError(f"weighted recommendation shortage: {len(selected)} < {count}")


def choose_backtest_method(means: np.ndarray) -> int:
    values = np.asarray(means, dtype=np.float64)
    winner = int(np.argmax(values))
    return winner if values[winner] > values[0] and values[winner] > 0.8 else 0


def evaluate_backtest_round(draws: list[dict], target_index: int) -> np.ndarray:
    history = draws[:target_index]
    winning = set(draws[target_index]["numbers"])
    forbidden = {tuple(draw["numbers"]) for draw in history}
    scores = np.zeros(len(BACKTEST_METHODS), dtype=np.float64)
    for method_index, (kind, parameter) in enumerate(BACKTEST_METHODS):
        sets = generate_weighted_sets(
            backtest_method_weights(history, kind, parameter),
            np.random.default_rng(SEED + draws[target_index]["round"] * 100 + method_index),
            BACKTEST_SETS,
            forbidden,
        )
        scores[method_index] = sum(len(winning & set(numbers)) for numbers in sets)
    return scores


def build_backtest_recommendations(draws: list[dict]) -> tuple[dict, list[dict]]:
    totals = np.zeros(len(BACKTEST_METHODS), dtype=np.float64)
    tested_rounds = max(0, len(draws) - BACKTEST_MIN_HISTORY)
    for target_index in range(BACKTEST_MIN_HISTORY, len(draws)):
        totals += evaluate_backtest_round(draws, target_index)

    means = totals / (tested_rounds * BACKTEST_SETS) if tested_rounds else totals
    selected_index = choose_backtest_method(means) if tested_rounds else 0
    selected_kind, selected_parameter = BACKTEST_METHODS[selected_index]
    historical = {tuple(draw["numbers"]) for draw in draws}
    numbers = generate_weighted_sets(
        backtest_method_weights(draws, selected_kind, selected_parameter),
        np.random.default_rng(SEED + (draws[-1]["round"] + 1) * 100 + selected_index),
        BACKTEST_SETS,
        historical,
    )
    summary = {
        "metric": "mean-matches-per-set",
        "testedRounds": tested_rounds,
        "selectedMethod": selected_kind,
        "selectedWindow": selected_parameter if selected_kind == "frequency-window" else None,
        "selectedHalfLife": selected_parameter if selected_kind == "frequency-decay" else None,
        "meanMatches": round(float(means[selected_index]), 4) if tested_rounds else None,
        "randomBaseline": round(float(means[0]), 4) if tested_rounds else None,
    }
    return summary, [
        {"method": "walk-forward-backtest", "numbers": recommendation}
        for recommendation in numbers
    ]
```

- [ ] **Step 4: Append the separate recommendations in `main()`**

Immediately before building `result`, compute:

```python
    backtest, backtest_recommendations = build_backtest_recommendations(draws)
```

Add the summary and append the new sets without changing the existing order:

```python
        "backtest": backtest,
        "recommendations": [
            *lstm_recommendations,
            *stat_recommendations(
                draws, np.random.default_rng(SEED + 4), forbidden=selected_lstm
            ),
            *backtest_recommendations,
        ],
```

- [ ] **Step 5: Run the lightweight check**

Run: `python tools/train-lstm-lotto.py --selftest`

Expected: `selftest ok` and exit code `0`.

- [ ] **Step 6: Commit the generator**

```bash
git add tools/train-lstm-lotto.py
git commit -m "feat: add walk-forward lotto recommendations"
```

---

### Task 2: Separate browser section and scoreboard category

**Files:**
- Create: `tools/test-backtest-render.mjs`
- Modify: `index.html:606-655`
- Modify: `index.html:1260-1518`

**Interfaces:**
- Consumes: `prediction.backtest`, recommendations whose method is `walk-forward-backtest`, existing `renderBalls`, and existing history entries.
- Produces: `renderBacktestRecommendations(prediction) -> string`, separate statistical/backtest filtering, and a separate backtest scoreboard average.

- [ ] **Step 1: Write a failing static renderer contract check**

Create `tools/test-backtest-render.mjs`:

```js
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");
const script = html.match(/<script>([\s\S]*)<\/script>/)?.[1];
assert.ok(script, "page script missing");
new Function(script);

assert.match(html, /function renderBacktestRecommendations\(prediction\)/);
assert.match(html, /rec\.method === "walk-forward-backtest"/);
assert.match(html, /rec\.method !== "walk-forward-backtest"/);
assert.match(html, /백테스트 추천/);
assert.match(html, /백테스트 평균/);
assert.match(html, /미래 당첨 성과를 보장하지 않습니다/);
console.log("backtest render check passed");
```

- [ ] **Step 2: Run the renderer check and confirm it fails**

Run: `node tools/test-backtest-render.mjs`

Expected: FAIL on the missing `renderBacktestRecommendations` assertion.

- [ ] **Step 3: Parse the backtest summary in `loadLstmPrediction()`**

Add `backtest` to the returned prediction object:

```js
          backtest: data.backtest && typeof data.backtest === "object" ? {
            metric: String(data.backtest.metric || ""),
            testedRounds: Number(data.backtest.testedRounds || 0),
            selectedMethod: String(data.backtest.selectedMethod || "uniform"),
            selectedWindow: data.backtest.selectedWindow == null
              ? null : Number(data.backtest.selectedWindow),
            selectedHalfLife: data.backtest.selectedHalfLife == null
              ? null : Number(data.backtest.selectedHalfLife),
            meanMatches: data.backtest.meanMatches === null ? null : Number(data.backtest.meanMatches),
            randomBaseline: data.backtest.randomBaseline === null ? null : Number(data.backtest.randomBaseline)
          } : null,
```

- [ ] **Step 4: Add method labels and the separate renderer**

Add to `methodLabel()`:

```js
      if (method === "walk-forward-backtest") return "시간순 백테스트 선택";
```

Add after `renderLstmRecommendations()`:

```js
    function renderBacktestRecommendations(prediction) {
      const recommendations = prediction
        ? prediction.recommendations.filter((rec) => rec.method === "walk-forward-backtest")
        : [];
      if (!recommendations.length) {
        return `<h3 style="margin:20px 0 8px;">백테스트 추천 <span class="stat-pill">대기 중</span></h3>
          <p class="muted">다음 데이터 갱신 때 별도 추천 5세트가 생성됩니다.</p>`;
      }

      const summary = prediction.backtest;
      const selected = !summary || summary.selectedMethod === "uniform"
        ? "균등 무작위"
        : summary.selectedMethod === "frequency-all"
          ? "전체 누적 빈도"
          : summary.selectedMethod === "frequency-window"
            ? `최근 ${summary.selectedWindow}회 빈도`
            : `반감기 ${summary.selectedHalfLife}회 시간감쇠 빈도`;
      const mean = summary && summary.meanMatches !== null
        ? summary.meanMatches.toFixed(3) : "-";
      const baseline = summary && summary.randomBaseline !== null
        ? summary.randomBaseline.toFixed(3) : "-";
      return `
        <h3 style="margin:20px 0 8px;">백테스트 추천 <span class="stat-pill">${recommendations.length}세트</span></h3>
        <p class="muted">과거 ${summary ? summary.testedRounds : 0}회차 시간순 평가에서 세트당 평균 적중 수가 가장 높았던 방식: ${selected}</p>
        <div class="stat-pills">
          <span class="stat-pill">평균 적중 ${mean}개</span>
          <span class="stat-pill">무작위 기준 ${baseline}개</span>
          <span class="stat-pill">이론 기대값 0.800개</span>
        </div>
        <div class="recommendation-list" style="margin-top:12px;">
          ${recommendations.map((recommendation) => `
            <article class="recommendation-card">
              ${renderBalls(recommendation.numbers)}
              <div class="stat-pills" style="margin-top:10px;">
                <span class="stat-pill">${methodLabel(recommendation.method)}</span>
              </div>
            </article>
          `).join("")}
        </div>
        <p class="warning" style="margin-top:12px;">과거 백테스트 결과이며 미래 당첨 성과를 보장하지 않습니다.</p>`;
    }
```

- [ ] **Step 5: Keep the statistical section separate and render the new block**

Change the statistical filter in `renderRecommendations()`:

```js
      const statRecs = lstmPrediction
        ? lstmPrediction.recommendations.filter((rec) =>
            !rec.method.startsWith("lstm")
            && rec.method !== "walk-forward-backtest"
          )
        : [];
```

After the statistical recommendation list, add:

```js
        ${renderBacktestRecommendations(lstmPrediction)}
```

Replace the introductory count paragraph with:

```html
<p class="muted">기존 딥러닝 2세트와 통계 3세트에 별도 백테스트 추천 5세트를 추가해 총 10세트가 매주 CI에서 생성됩니다. 어느 기기에서 보든 같은 번호입니다.</p>
```

- [ ] **Step 6: Split the scoreboard averages**

Replace the current non-LSTM grouping with:

```js
      const backtestMatches = allMatches.filter((match) =>
        String(match.method) === "walk-forward-backtest"
      );
      const statMatches = allMatches.filter((match) =>
        !String(match.method).startsWith("lstm")
        && String(match.method) !== "walk-forward-backtest"
      );
```

Add the separate badge beside the existing statistical badge:

```js
          ${backtestMatches.length ? `<span class="stat-pill">백테스트 평균 ${averageOf(backtestMatches).toFixed(2)}개</span>` : ""}
```

Replace the scoreboard explanation with:

```html
<p class="muted">매주 CI가 생성한 기존 추천 5세트와 별도 백테스트 추천 5세트를 회차별로 누적하고 실제 당첨번호와 자동 대조합니다. 무작위 6개 선택의 이론 기대 적중은 세트당 0.80개입니다.</p>
```

- [ ] **Step 7: Run browser and Python checks**

Run: `node tools/test-backtest-render.mjs`

Expected: `backtest render check passed` and exit code `0`.

Run: `python tools/train-lstm-lotto.py --selftest`

Expected: `selftest ok` and exit code `0`.

- [ ] **Step 8: Commit the separate UI**

```bash
git add index.html tools/test-backtest-render.mjs
git commit -m "feat: show separate backtest recommendations"
```

---

### Task 3: Generate and validate the published artifacts

**Files:**
- Modify: `lstm-prediction.json`
- Modify: `lstm-prediction-history.json`

**Interfaces:**
- Consumes: the completed training script and current `lotto-data.json`.
- Produces: a current schema-version-2 prediction containing the unchanged five existing recommendations, five new backtest recommendations, and the backtest summary; preserves prior history while replacing the current target-round entry.

- [ ] **Step 1: Run the full existing training command**

Run: `python tools/train-lstm-lotto.py`

Expected: a final line beginning `saved lstm-prediction.json:` and exit code `0`.

- [ ] **Step 2: Validate the generated contract**

Run:

```powershell
python -c "import json; p=json.load(open('lstm-prediction.json',encoding='utf-8')); old=[r for r in p['recommendations'] if r['method']!='walk-forward-backtest']; new=[r for r in p['recommendations'] if r['method']=='walk-forward-backtest']; hist={tuple(d['numbers']) for d in json.load(open('lotto-data.json',encoding='utf-8'))['draws']}; assert p['schemaVersion']==2 and len(old)==5 and len(new)==5 and len({tuple(r['numbers']) for r in new})==5 and all(tuple(r['numbers']) not in hist for r in new) and p['backtest']['testedRounds']>0; print('generated backtest contract ok')"
```

Expected: `generated backtest contract ok` and exit code `0`.

- [ ] **Step 3: Run all lightweight regression checks**

Run: `python tools/train-lstm-lotto.py --selftest`

Expected: `selftest ok`.

Run: `node tools/test-backtest-render.mjs`

Expected: `backtest render check passed`.

Run: `git diff --check`

Expected: no whitespace errors.

- [ ] **Step 4: Commit generated predictions**

```bash
git add lstm-prediction.json lstm-prediction-history.json
git commit -m "data: publish backtest recommendations"
```
