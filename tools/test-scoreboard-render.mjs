import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

const htmlSource = fs.readFileSync(new URL("../index.html", import.meta.url), "utf8");
const script = htmlSource.match(/<script>([\s\S]*)<\/script>/)?.[1];
assert.ok(script, "index.html script not found");

const panel = { innerHTML: "" };
const inertElement = {
  addEventListener() {},
  classList: { toggle() {} },
  setAttribute() {},
  dataset: {},
};
const document = {
  querySelector(selector) {
    return selector === "#lstmScorePanel" ? panel : inertElement;
  },
  querySelectorAll() { return []; },
  addEventListener() {},
};
const localStorage = {
  getItem() { return null; },
  setItem() {},
};

const entries = [
  {
    targetRound: 1236,
    sourceLatestRound: 1235,
    recommendations: [{ method: "lstm-combination-best", numbers: [8, 13, 19, 26, 37, 42] }],
    result: null,
  },
  {
    targetRound: 1235,
    sourceLatestRound: 1234,
    recommendations: [{ method: "lstm-combination-best", numbers: [1, 7, 8, 26, 37, 42] }],
    result: {
      date: "2026-08-01",
      winningNumbers: [1, 2, 3, 4, 5, 6],
      bonus: 7,
      matches: [{ method: "lstm-combination-best", matchCount: 1, bonusMatched: true }],
    },
  },
];

const context = vm.createContext({ document, localStorage, console, Intl, Date, setTimeout, clearTimeout });
vm.runInContext(`${script}\nrenderLstmScoreboard(${JSON.stringify(entries)});`, context);

assert.ok(!panel.innerHTML.includes("1236회 예측"), "pending round must be hidden");
assert.ok(panel.innerHTML.includes("1235회 예측"), "completed round must be shown");
assert.ok(
  panel.innerHTML.indexOf('<span class="ball n1">1</span>')
    < panel.innerHTML.indexOf('<div class="recommendation-list">'),
  "actual draw must appear beside the round before recommendations",
);
assert.ok(!panel.innerHTML.includes("실제 당첨 번호"), "duplicate actual draw block must be removed");
assert.ok(panel.innerHTML.includes('<span class="ball n1 hit">1</span>'), "main hit must be highlighted");
assert.ok(panel.innerHTML.includes('<span class="ball n1 bonus-hit">7</span>'), "bonus hit must be highlighted");
assert.ok(panel.innerHTML.includes('<span class="ball n1">8</span>'), "unmatched ball must keep its normal class");

console.log("scoreboard render check passed");
