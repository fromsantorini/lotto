# LSTM Combination Scorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace independent-number LSTM recommendations with two balanced six-number combinations ranked as whole candidates by an LSTM-based scorer.

**Architecture:** Keep the existing 10-draw multihot sequence and encode it with `LSTM(128)`. Concatenate that context with a candidate combination's 45-value multihot vector, score it through `Dense(64, relu) -> Dense(1, sigmoid)`, and rank 50,000 deterministic balanced candidates. Train only on balanced actual draws and balanced hard negatives so train and inference candidate domains match.

**Tech Stack:** Python 3, NumPy 1.26.x, TensorFlow CPU 2.16.x, HTML/CSS/vanilla JavaScript, JSON

## Global Constraints

- Keep `window=10`, `LSTM(128)`, `epochs=100`, `batch_size=16`, and early stopping patience `8`.
- Use five balanced negative candidates per positive sequence.
- Use 50,000 unique balanced inference candidates and deterministic seeds derived from `SEED=42`.
- Select a best combination and a second combination sharing at most two numbers with the first.
- Treat `modelScore` as a relative ranking score, never a winning probability or percentage.
- Emit `lstm-prediction.json` schema version `2`; keep `lstm-prediction-history.json` schema version `1` readable.
- Preserve the three existing `balanced-statistical` recommendations, data update flow, and GitHub Actions workflow.
- Add no dependencies and no new source modules.

---

### Task 1: Deterministic balanced candidate and scorer datasets

**Files:**
- Modify: `tools/train-lstm-lotto.py`

**Interfaces:**
- Consumes: existing `is_balanced(nums)`, `to_multihot(numbers)`, `NUM_RANGE`, `PICK`, and `SEED`
- Produces: `multihot_to_numbers(vector) -> list[int]`, `generate_balanced_candidates(rng, count, forbidden=()) -> list[list[int]]`, `expand_scorer_examples(sequences, positives, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]`

- [ ] **Step 1: Extend `selftest()` with failing candidate and dataset assertions**

Add these checks before `print("selftest ok")`:

```python
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
```

- [ ] **Step 2: Run the self-test and verify the new names are missing**

Run:

```powershell
python tools/train-lstm-lotto.py --selftest
```

Expected: failure naming `generate_balanced_candidates`, `expand_scorer_examples`, or `NEGATIVES_PER_POSITIVE`.

- [ ] **Step 3: Add constants and minimal candidate helpers**

Add near the existing constants:

```python
NEGATIVES_PER_POSITIVE = 5
INFERENCE_CANDIDATES = 50_000
VALIDATION_FRACTION = 0.1
MAX_CANDIDATE_ATTEMPT_FACTOR = 100
```

Add after `is_balanced`:

```python
def multihot_to_numbers(vector: np.ndarray) -> list[int]:
    return [int(i) + 1 for i in np.flatnonzero(vector)]


def generate_balanced_candidates(
    rng: np.random.Generator,
    count: int,
    forbidden: set[tuple[int, ...]] | frozenset[tuple[int, ...]] = frozenset(),
) -> list[list[int]]:
    candidates: set[tuple[int, ...]] = set()
    for _ in range(count * MAX_CANDIDATE_ATTEMPT_FACTOR):
        numbers = tuple(sorted(int(i) + 1 for i in rng.choice(NUM_RANGE, PICK, replace=False)))
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
        for numbers, label in [(positive_numbers, 1.0), *((n, 0.0) for n in negatives)]:
            sequence_examples.append(sequence)
            candidate_examples.append(to_multihot(numbers))
            labels.append(label)

    return (
        np.asarray(sequence_examples, dtype=np.float32),
        np.asarray(candidate_examples, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
    )
```

- [ ] **Step 4: Run the self-test and verify deterministic candidate generation passes**

Run:

```powershell
python tools/train-lstm-lotto.py --selftest
```

Expected: `selftest ok`.

- [ ] **Step 5: Commit the candidate primitives**

```powershell
git add tools/train-lstm-lotto.py
git commit -m "feat: add balanced combination candidates"
```

---

### Task 2: Train and infer with the LSTM combination scorer

**Files:**
- Modify: `tools/train-lstm-lotto.py`

**Interfaces:**
- Consumes: Task 1's `generate_balanced_candidates`, `expand_scorer_examples`, and existing `build_dataset`
- Produces: `build_combination_scorer(tf, window)`, `select_scored_recommendations(candidates, scores) -> list[dict]`, schema version 2 prediction output

- [ ] **Step 1: Add failing scorer-selection checks to `selftest()`**

```python
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
```

- [ ] **Step 2: Run the self-test and verify selection is undefined**

Run `python tools/train-lstm-lotto.py --selftest`.

Expected: failure naming `select_scored_recommendations`.

- [ ] **Step 3: Replace the model name and add scorer helpers**

Replace the model name and add these functions before `main()`:

```python
MODEL_NAME = "keras-lstm-combination-scorer-v1"


def build_combination_scorer(tf, window: int):
    sequence_input = tf.keras.layers.Input(shape=(window, NUM_RANGE), name="sequence")
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
    order = np.argsort(np.asarray(scores).reshape(-1))[::-1]
    best_index = int(order[0])
    best = candidates[best_index]
    diverse_index = next(
        (int(i) for i in order[1:] if len(set(best) & set(candidates[int(i)])) <= 2),
        None,
    )
    if diverse_index is None:
        raise RuntimeError("no diverse scored candidate available")
    return [
        {
            "method": "lstm-combination-best",
            "numbers": best,
            "modelScore": round(float(scores[best_index]), 6),
        },
        {
            "method": "lstm-combination-diverse",
            "numbers": candidates[diverse_index],
            "modelScore": round(float(scores[diverse_index]), 6),
        },
    ]
```

- [ ] **Step 4: Replace independent-probability training with chronological scorer training**

After `x, y = build_dataset(vectors, args.window)`, filter to the candidate domain, split before negative expansion, and train:

```python
    balanced_mask = np.asarray([is_balanced(multihot_to_numbers(target)) for target in y])
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
        model.evaluate([val_x, val_candidates], val_labels, verbose=0, return_dict=True)["auc"]
    )
```

Delete the old `Sequential` model, 45-sigmoid prediction, probability ordering, `top_probability_set`, and `weighted_sampling_set` calls.

- [ ] **Step 5: Score deterministic candidates and emit schema version 2**

Replace the old recommendation construction with:

```python
    historical = {tuple(d["numbers"]) for d in draws}
    candidates = generate_balanced_candidates(
        np.random.default_rng(SEED + 2),
        INFERENCE_CANDIDATES,
        historical,
    )
    candidate_vectors = np.stack([to_multihot(candidate) for candidate in candidates])
    sequence_batch = np.repeat(vectors[-args.window:][np.newaxis, ...], len(candidates), axis=0)
    scores = model.predict([sequence_batch, candidate_vectors], batch_size=1024, verbose=0).reshape(-1)
    lstm_recommendations = select_scored_recommendations(candidates, scores)

    result = {
        "schemaVersion": 2,
        "model": MODEL_NAME,
        "sourceLatestRound": source_latest,
        "targetRound": target_round,
        "trainedAt": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "window": args.window,
        "epochs": int(len(history.history["loss"])),
        "trainSampleCount": int(len(balanced_x)),
        "validationAuc": round(validation_auc, 4),
        "candidateCount": len(candidates),
        "recommendations": [
            *lstm_recommendations,
            *stat_recommendations(draws, np.random.default_rng(SEED + 3)),
        ],
        "warning": (
            "모델 점수는 후보 간 상대 비교값이며 당첨 확률이 아닙니다. "
            "로또는 독립시행이므로 이 결과는 실험용입니다."
        ),
    }
```

Update the final log to print `balanced_samples`, `validation_auc`, and `candidate_count` rather than the removed probability output.

- [ ] **Step 6: Run cheap checks, then a one-epoch training smoke test**

Run:

```powershell
python tools/train-lstm-lotto.py --selftest
python tools/train-lstm-lotto.py --epochs 1
```

Expected: `selftest ok`; then `lstm-prediction.json` is saved with `schemaVersion: 2`, two `lstm-combination-*` entries, three `balanced-statistical` entries, and no `topNumbers`.

- [ ] **Step 7: Commit the scorer**

```powershell
git add tools/train-lstm-lotto.py
git commit -m "feat: score whole LSTM lottery combinations"
```

---

### Task 3: Render schema version 2 combination scores

**Files:**
- Modify: `index.html`

**Interfaces:**
- Consumes: schema version 2 fields `validationAuc`, `candidateCount`, `recommendations[].modelScore`
- Produces: validated `lstmPrediction` browser state and combination-score cards without individual number probabilities

- [ ] **Step 1: Add a failing static contract check**

Run before editing:

```powershell
node -e "const s=require('fs').readFileSync('index.html','utf8'); if(!s.includes('lstm-combination-best') || s.includes('topNums = (prediction.topNumbers')) process.exit(1)"
```

Expected: exit code `1`.

- [ ] **Step 2: Update `loadLstmPrediction()` for schema version 2**

Require `data.schemaVersion === 2`. Preserve `reason` and add a finite optional model score:

```javascript
        if (data.schemaVersion !== 2 || !Array.isArray(data.recommendations)) return null;

        const recommendations = data.recommendations
          .map((rec) => ({
            method: String(rec.method || "lstm"),
            numbers: Array.isArray(rec.numbers) ? rec.numbers.map(Number) : [],
            reason: rec.reason && typeof rec.reason === "object" ? rec.reason : null,
            modelScore: Number.isFinite(Number(rec.modelScore)) ? Number(rec.modelScore) : null
          }))
```

Replace `topNumbers` in the returned object with:

```javascript
          validationAuc: Number.isFinite(Number(data.validationAuc)) ? Number(data.validationAuc) : null,
          candidateCount: Number.isInteger(Number(data.candidateCount)) ? Number(data.candidateCount) : 0,
```

- [ ] **Step 3: Replace method labels and probability rendering**

Update `methodLabel()`:

```javascript
    function methodLabel(method) {
      if (method === "lstm-combination-best") return "LSTM 최고 점수 조합";
      if (method === "lstm-combination-diverse") return "LSTM 다양성 조합";
      if (method === "balanced-statistical") return "통계 균형 조합";
      return method;
    }
```

In `renderLstmRecommendations()`, delete the `topNums` calculation and its paragraph. Add model metadata after the training metadata:

```javascript
          <p class="muted">검증 AUC ${prediction.validationAuc === null ? "-" : prediction.validationAuc.toFixed(3)} · 평가 후보 ${prediction.candidateCount.toLocaleString()}개</p>
```

Add the score pill inside each LSTM recommendation card:

```javascript
                <div class="stat-pills" style="margin-top:10px;">
                  <span class="stat-pill">딥러닝</span>
                  <span class="stat-pill">${methodLabel(rec.method)}</span>
                  ${rec.modelScore === null ? "" : `<span class="stat-pill">상대 점수 ${rec.modelScore.toFixed(4)}</span>`}
                </div>
```

- [ ] **Step 4: Run static HTML contract checks**

```powershell
node -e "const s=require('fs').readFileSync('index.html','utf8'); const required=['schemaVersion !== 2','lstm-combination-best','lstm-combination-diverse','validationAuc','modelScore','상대 점수']; const forbidden=['topNums = (prediction.topNumbers','LSTM 상위 확률','LSTM 확률 가중 샘플']; if(required.some(x=>!s.includes(x)) || forbidden.some(x=>s.includes(x))) process.exit(1); console.log('html contract ok')"
```

Expected: `html contract ok`.

- [ ] **Step 5: Commit the UI update**

```powershell
git add index.html
git commit -m "feat: show LSTM combination scores"
```

---

### Task 4: Regenerate artifacts and verify the complete flow

**Files:**
- Modify: `lstm-prediction.json`
- Modify: `lstm-prediction-history.json`

**Interfaces:**
- Consumes: Task 2 trainer and Task 3 schema version 2 loader
- Produces: current schema version 2 prediction artifact and preserved score history

- [ ] **Step 1: Run the normal deterministic training command**

```powershell
python tools/train-lstm-lotto.py
```

Expected: training finishes with a saved prediction log containing source round, target round, balanced sample count, validation AUC, and 50,000 candidates.

- [ ] **Step 2: Validate the generated artifacts with one runnable check**

```powershell
python -c "import json; p=json.load(open('lstm-prediction.json',encoding='utf-8')); r=p['recommendations']; dl=[x for x in r if x['method'].startswith('lstm')]; st=[x for x in r if not x['method'].startswith('lstm')]; assert p['schemaVersion']==2 and 'topNumbers' not in p and len(dl)==2 and len(st)==3 and p['candidateCount']==50000; assert all(len(x['numbers'])==len(set(x['numbers']))==6 for x in r); assert len(set(dl[0]['numbers']) & set(dl[1]['numbers']))<=2; assert all('modelScore' in x for x in dl); print('prediction contract ok')"
```

Expected: `prediction contract ok`.

- [ ] **Step 3: Verify history preservation and frontend references**

```powershell
python tools/train-lstm-lotto.py --selftest
node -e "const p=require('./lstm-prediction.json'); const h=require('./lstm-prediction-history.json'); if(p.schemaVersion!==2 || h.schemaVersion!==1 || !h.entries.some(e=>e.targetRound===p.targetRound)) process.exit(1); console.log('history contract ok')"
```

Expected: `selftest ok` and `history contract ok`.

- [ ] **Step 4: Review only the intended files**

```powershell
git status --short
git diff --check
```

Expected: only `lstm-prediction.json` and `lstm-prediction-history.json` remain modified; `git diff --check` prints nothing.

- [ ] **Step 5: Commit generated artifacts**

```powershell
git add lstm-prediction.json lstm-prediction-history.json
git commit -m "data: publish combination score prediction"
```

---

## Final Verification

Run:

```powershell
python tools/train-lstm-lotto.py --selftest
node -e "const p=require('./lstm-prediction.json'); const dl=p.recommendations.filter(x=>x.method.startsWith('lstm')); if(p.schemaVersion!==2 || dl.length!==2 || dl.some(x=>typeof x.modelScore!=='number')) process.exit(1); console.log('schema ok')"
git status --short
```

Expected:

- `selftest ok`
- `schema ok`
- working tree contains no changes from this feature
