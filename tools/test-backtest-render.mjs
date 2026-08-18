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
