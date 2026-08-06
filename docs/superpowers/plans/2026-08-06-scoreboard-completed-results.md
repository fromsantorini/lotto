# Completed Scoreboard Results Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show only completed recommendation results and place each actual draw beside its round heading.

**Architecture:** Keep the existing single-file renderer and data schema. Reuse the already-computed `scored` array and `renderBalls`; remove the duplicate lower result block.

**Tech Stack:** Static HTML, browser JavaScript, Node.js built-in `vm` and `assert`.

## Global Constraints

- Modify only scoreboard rendering behavior and its focused regression check.
- Do not change prediction generation, history storage, or dependencies.
- Preserve recommendation match counts, prize labels, and summary statistics.

---

### Task 1: Completed-only scoreboard

**Files:**
- Create: `tools/test-scoreboard-render.mjs`
- Modify: `index.html:1280-1360`

**Interfaces:**
- Consumes: `renderLstmScoreboard(entries)`, existing history entry schema, and `renderBalls(numbers, bonus)`.
- Produces: scoreboard HTML containing only scored rounds, with actual balls before recommendation cards.

- [x] **Step 1: Write the failing test**

Create a Node check that evaluates the page script with a minimal DOM stub, renders one pending and one completed entry, and asserts:

```js
assert.ok(!html.includes("1236회 예측"));
assert.ok(html.includes("1235회 예측"));
assert.ok(html.indexOf("<span class=\"ball n1\">1</span>") < html.indexOf("<div class=\"recommendation-list\">"));
assert.ok(!html.includes("실제 당첨 번호"));
```

- [x] **Step 2: Run test to verify it fails**

Run: `node tools/test-scoreboard-render.mjs`

Expected: FAIL because the pending `1236회 예측` is still rendered.

- [x] **Step 3: Write minimal implementation**

In `renderLstmScoreboard`:

```js
${scored.slice(0, 20).map((entry) => {
  const result = entry.result;
```

Insert `${renderBalls(result.winningNumbers, result.bonus)}` in `.history-batch-head` immediately after the round metadata, and delete the lower `history-draw` result block.

- [x] **Step 4: Run test to verify it passes**

Run: `node tools/test-scoreboard-render.mjs`

Expected: `scoreboard render check passed` and exit code 0.

- [x] **Step 5: Run existing lightweight checks**

Run: `python tools/train-lstm-lotto.py --selftest`

Expected: `selftest passed` and exit code 0.

- [x] **Step 6: Commit**

```bash
git add index.html tools/test-scoreboard-render.mjs docs/superpowers/plans/2026-08-06-scoreboard-completed-results.md
git commit -m "ui: compact completed recommendation scores"
```
