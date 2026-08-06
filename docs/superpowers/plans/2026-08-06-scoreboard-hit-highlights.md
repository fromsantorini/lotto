# Scoreboard Hit Highlights Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make matched main and bonus numbers visually distinct inside completed recommendation cards.

**Architecture:** Extend the existing `renderBalls` helper with one optional highlight object so all current callers remain unchanged. The scoreboard recommendation call supplies the completed draw; CSS classes provide the visual treatment.

**Tech Stack:** Static HTML, CSS, browser JavaScript, Node.js built-in regression check.

## Global Constraints

- Main-number hits use a gold outline and glow.
- Bonus-only hits use a purple outline and glow.
- Main-number matching takes priority over bonus matching.
- Unmatched balls, actual draw rows, match counts, and prize labels remain unchanged.
- Add no dependency or component.

---

### Task 1: Highlight matched recommendation balls

**Files:**
- Modify: `tools/test-scoreboard-render.mjs`
- Modify: `index.html:140-150, 1143-1150, 1325-1340`

**Interfaces:**
- Consumes: `renderBalls(numbers, bonus, highlights)` where `highlights` is optional and contains `winningNumbers` and `bonus`.
- Produces: `hit` or `bonus-hit` classes only on matching recommendation balls.

- [x] **Step 1: Extend the failing regression check**

Use a recommendation containing main hit `1`, bonus hit `7`, and unmatched `8`, then assert:

```js
assert.ok(panel.innerHTML.includes('<span class="ball n1 hit">1</span>'));
assert.ok(panel.innerHTML.includes('<span class="ball n1 bonus-hit">7</span>'));
assert.ok(panel.innerHTML.includes('<span class="ball n1">8</span>'));
```

- [x] **Step 2: Verify RED**

Run: `node tools/test-scoreboard-render.mjs`

Expected: FAIL because no `hit` class exists.

- [x] **Step 3: Implement the optional highlight classes**

Change the helper signature and class selection:

```js
function renderBalls(numbers, bonus = null, highlights = null) {
  const winningNumbers = new Set(highlights?.winningNumbers || []);
  const highlightClass = winningNumbers.has(number)
    ? " hit"
    : number === highlights?.bonus ? " bonus-hit" : "";
}
```

Pass `{ winningNumbers: result.winningNumbers, bonus: result.bonus }` only when rendering each scored recommendation. Add `.ball.hit` and `.ball.bonus-hit` CSS using gold and purple outlines respectively.

- [x] **Step 4: Verify GREEN**

Run: `node tools/test-scoreboard-render.mjs`

Expected: `scoreboard render check passed` and exit code 0.

- [x] **Step 5: Run existing check**

Run: `python tools/train-lstm-lotto.py --selftest`

Expected: `selftest ok` and exit code 0.

- [x] **Step 6: Commit**

```bash
git add index.html tools/test-scoreboard-render.mjs docs/superpowers/plans/2026-08-06-scoreboard-hit-highlights.md
git commit -m "ui: highlight matched recommendation numbers"
```
