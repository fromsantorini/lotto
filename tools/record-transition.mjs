import { readFileSync, writeFileSync } from "node:fs";
import { runInNewContext } from "node:vm";

// 회차 전이 번호를 이력에 고정한다. 이미 기록된 회차는 알고리즘이 바뀌어도 다시 계산하지 않는다.
const METHOD = "round-transition";
const fileUrl = (name) => new URL(`../${name}`, import.meta.url);
const readJson = (name) => JSON.parse(readFileSync(fileUrl(name), "utf8"));

// 화면과 같은 predictTransitionSet을 그대로 사용한다.
const script = readFileSync(fileUrl("index.html"), "utf8").match(/<script>([\s\S]*)<\/script>/)[1];
const { predictTransitionSet } = runInNewContext(`${script}\n({ predictTransitionSet })`, {
  localStorage: { getItem: () => null },
  document: { querySelector: () => ({ addEventListener() {} }), addEventListener() {} },
});

const history = readJson("lstm-prediction-history.json");
const { draws } = readJson("lotto-data.json");
let added = 0;
for (const entry of history.entries) {
  if (entry.recommendations.some((rec) => rec.method === METHOD)) continue;
  const transition = predictTransitionSet(draws.filter((draw) => draw.round < entry.targetRound));
  if (transition?.targetRound !== entry.targetRound) continue;
  const { numbers } = transition;
  entry.recommendations.push({ method: METHOD, numbers });
  if (entry.result) {
    const { winningNumbers, bonus } = entry.result;
    entry.result.matches.push({
      method: METHOD,
      matchCount: numbers.filter((number) => winningNumbers.includes(number)).length,
      bonusMatched: numbers.includes(bonus),
    });
  }
  added++;
}
writeFileSync(fileUrl("lstm-prediction-history.json"), `${JSON.stringify(history, null, 2)}\n`);
console.log(`Recorded round transition for ${added} rounds`);
