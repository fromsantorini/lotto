import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";

const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");
const script = html.match(/<script>([\s\S]*)<\/script>/)?.[1];
assert.ok(script, "page script missing");
const panels = new Map();
let data;
const api = runInNewContext(`${script}\n({ loadLstmPrediction, renderBacktestRecommendations,
  renderMethodComparison, renderLstmScoreboard, renderRecommendations,
  buildRecommendationReasonSummary, finiteNumber, predictTransitionSet,
  setPrediction(value) { lstmPrediction = value; } })`, {
  localStorage: { getItem: () => null },
  document: {
    querySelector(selector) {
      if (!panels.has(selector)) panels.set(selector, { addEventListener() {} });
      return panels.get(selector);
    },
    addEventListener() {},
  },
  fetch: async () => ({ ok: true, json: async () => data }),
  console,
});
assert.equal(api.finiteNumber(null), null);
assert.equal(api.finiteNumber(""), null);
assert.equal(api.finiteNumber(Infinity), null);
assert.equal(api.finiteNumber(0), 0);
assert.match(api.renderBacktestRecommendations(null), /대기 중/);
assert.match(api.renderMethodComparison(null), /재학습 후/);

// Independent one-step prediction: A -> B -> A should select B, never the bonus.
const groupA = [1, 2, 3, 4, 5, 6];
const groupB = [11, 12, 13, 14, 15, 16];
const transitionDraws = [groupA, groupB, groupA].map((numbers, index) => ({
  round: index + 1, numbers, bonus: 45,
}));
const originalDraws = JSON.stringify(transitionDraws);
const transition = api.predictTransitionSet(transitionDraws);
assert.deepEqual(Array.from(transition.numbers), groupB);
assert.equal(transition.targetRound, 4);
assert.equal(transition.transitions, 2);
assert.equal(JSON.stringify(transitionDraws), originalDraws);
assert.equal(JSON.stringify(api.predictTransitionSet([...transitionDraws].reverse())), JSON.stringify(transition));
assert.deepEqual(Array.from(api.predictTransitionSet([
  ...transitionDraws, { round: 4, numbers: groupB, bonus: 45 },
]).numbers), groupA);
assert.equal(api.predictTransitionSet([]), null);
assert.equal(api.predictTransitionSet([transitionDraws[0]]), null);
assert.equal(api.predictTransitionSet([transitionDraws[0], transitionDraws[2]]), null);
assert.equal(api.predictTransitionSet([...transitionDraws, transitionDraws[0]]), null);
assert.equal(api.predictTransitionSet([transitionDraws[0], { round: 2, numbers: [0, 1, 2, 3, 4, 5], bonus: 45 }]), null);
assert.equal(api.predictTransitionSet([{}, {}]), null);
assert.equal(api.predictTransitionSet([...transitionDraws, { round: 5, numbers: groupB, bonus: 45 }]).transitions, 2);

const drawData = JSON.parse(readFileSync(new URL("../lotto-data.json", import.meta.url), "utf8"));
const liveTransition = api.predictTransitionSet(drawData.draws);
assert.equal(liveTransition.targetRound, drawData.latestRound + 1);
assert.equal(new Set(liveTransition.numbers).size, 6);
assert.ok(liveTransition.numbers.every((number) => Number.isInteger(number) && number >= 1 && number <= 45));
api.renderRecommendations({}, drawData);
const independentPanel = panels.get("#recommendPanel").innerHTML;
assert.match(independentPanel, /전체 1세트/);
assert.match(independentPanel, new RegExp(`${liveTransition.targetRound}회 대상`));
assert.equal((independentPanel.match(/class="recommendation-card"/g) || []).length, 1);
assert.doesNotMatch(independentPanel, /NaN|undefined/);
console.log(`Transition prediction for round ${liveTransition.targetRound}: ${liveTransition.numbers.join(", ")}`);

const current = JSON.parse(readFileSync(new URL("../lstm-prediction.json", import.meta.url), "utf8"));
data = structuredClone(current);
delete data.comparison;
data.validationAuc = null;
data.backtest = { selectedMethod: "uniform", meanMatches: null };
data.recommendations[0].modelScore = null;
let prediction = await api.loadLstmPrediction();
assert.equal(prediction.validationAuc, null);
assert.equal(prediction.backtest.randomBaseline, null);
assert.equal(prediction.recommendations[0].modelScore, null);
assert.match(api.renderBacktestRecommendations(prediction), /별도 검증 성적이 아닙니다/);
assert.doesNotMatch(api.renderBacktestRecommendations(prediction), /NaN|undefined/);

data.comparison = {
  evaluationMode: "chronological-holdout-v2", testStartRound: 1118, testEndRound: 1240,
  testedRounds: 123, setsPerMethod: 2, seedCount: 3,
  methods: [
    { method: "uniform", meanMatches: 0.8, threePlusRate: 0.02, differenceVsRandom: 0, differenceCI95: [0, 0] },
    { method: "lstm-number-weighted-v2", meanMatches: 0.81, threePlusRate: 0.03, differenceVsRandom: 0.01, differenceCI95: [-0.02, 0.04] },
    { method: "weighted-statistical-v2", meanMatches: 0.82, threePlusRate: 0.03, differenceVsRandom: 0.02, differenceCI95: [0.01, 0.03] },
    { method: "<img src=x onerror=alert(1)>" }, null,
  ],
};
data.backtest = {
  evaluationMode: "prequential-selection-v2", selectedMethod: "uniform", testedRounds: 990,
  warmupRounds: 50, seedCount: 3, meanMatches: 0.8, randomBaseline: 0.8,
  threePlusRate: 0.02, differenceCI95: [-0.01, 0.01],
};
prediction = await api.loadLstmPrediction();
const comparison = api.renderMethodComparison(prediction);
assert.match(comparison, /1118~1240회/);
assert.match(comparison, /0\.810개/);
assert.match(comparison, /-0\.020 ~ 0\.040/);
assert.match(comparison, /우위 미확인/);
assert.match(comparison, /이 구간에서 양의 차이 관측/);
assert.match(comparison, /가중치는 고정/);
assert.doesNotMatch(comparison, /NaN|undefined|<img/);
const backtest = api.renderBacktestRecommendations(prediction);
assert.match(backtest, /이전 성적으로 방식을 선택한 뒤 다음 회차/);
assert.match(backtest, /선택 전략 평균/);
assert.match(backtest, /미래 당첨 성과를 보장하지 않습니다/);
assert.doesNotMatch(backtest, /NaN|undefined/);
data.comparison.testedRounds = 1;
data.comparison.methods[2].differenceCI95 = [null, 0.03];
assert.doesNotMatch(api.renderMethodComparison(await api.loadLstmPrediction()), /이 구간에서 양의 차이 관측/);

data = current;
prediction = await api.loadLstmPrediction();
assert.ok(prediction);
api.setPrediction(prediction);
api.renderRecommendations({}, drawData);
assert.match(panels.get("#recommendPanel").innerHTML, new RegExp(`전체 ${prediction.recommendations.length + 1}세트`));
assert.equal(JSON.stringify(api.predictTransitionSet(drawData.draws)), JSON.stringify(liveTransition));
api.renderRecommendations({}, { latestRound: current.sourceLatestRound, draws: [{ numbers: [1, 2, 3, 4, 5, 6] }] });
assert.doesNotMatch(panels.get("#recommendPanel").innerHTML, /NaN|undefined/);
assert.match(panels.get("#recommendPanel").innerHTML, /최대 겹침/);
const history = JSON.parse(readFileSync(new URL("../lstm-prediction-history.json", import.meta.url), "utf8"));
api.renderLstmScoreboard(history.entries);
assert.doesNotMatch(panels.get("#lstmScorePanel").innerHTML, /NaN|undefined/);
assert.match(api.buildRecommendationReasonSummary({ method: "weighted-statistical-v2" }), /번호당 1/);

// 기존 JSON과 상태유지 모델 JSON 모두 표시하며 누락/오염 점수는 숨긴다.
data = structuredClone(current);
data.stateful = true;
data.epochs = 100;
data.numberScores = Array(45).fill(0.2);
data.numberScores[0] = 0.99;
data.recommendations[0].method = "lstm-stateful-ball-weighted-v3";
data.comparison = {
  evaluationMode: "chronological-stateful-holdout-v3", testedRounds: 123,
  methods: [{ method: "lstm-stateful-ball-weighted-v3", meanMatches: 0.8 }],
};
prediction = await api.loadLstmPrediction();
api.setPrediction(prediction);
api.renderRecommendations({}, { latestRound: current.sourceLatestRound, draws: [{}] });
const statefulPanel = panels.get("#recommendPanel").innerHTML;
assert.match(statefulPanel, /128유닛 상태유지 LSTM/);
assert.match(statefulPanel, /100에포크 학습/);
assert.match(statefulPanel, /원문 방식 v3/);
assert.match(statefulPanel, /<td>0\.990000<\/td><td>100<\/td>/);
assert.match(statefulPanel, /<td>0\.200000<\/td><td>21<\/td>/);
assert.match(api.renderMethodComparison(prediction), /원문 방식 v3/);
assert.doesNotMatch(statefulPanel, /NaN|undefined/);
data.numberScores[1] = null;
assert.equal((await api.loadLstmPrediction()).numberScores, null);
console.log("recommendation render checks passed");
