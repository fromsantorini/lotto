import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const dataPath = resolve(root, "lotto-data.json");
const OFFICIAL_DRAW_API_URL = "https://www.dhlottery.co.kr/common.do?method=getLottoNumber";
const MAX_BACKFILL_ROUNDS = 20;

function normalizeNumbers(numbers) {
  return numbers.map(Number).sort((a, b) => a - b);
}

function validateDraw(draw) {
  const numbers = normalizeNumbers(draw.numbers || []);
  const round = Number(draw.round);
  const bonus = Number(draw.bonus);

  if (!Number.isInteger(round) || round < 1) throw new Error(`invalid_round_${draw.round}`);
  if (numbers.length !== 6 || new Set(numbers).size !== 6) throw new Error(`invalid_numbers_round_${round}`);
  if (numbers.some((number) => number < 1 || number > 45)) throw new Error(`invalid_number_range_round_${round}`);
  if (!Number.isInteger(bonus) || bonus < 1 || bonus > 45) throw new Error(`invalid_bonus_round_${round}`);

  return {
    round,
    date: String(draw.date || ""),
    numbers,
    bonus,
    firstPrizeAmount: Number(draw.firstPrizeAmount || 0),
    firstWinnerCount: Number(draw.firstWinnerCount || 0),
    totalSellAmount: Number(draw.totalSellAmount || 0)
  };
}

function readCurrentData() {
  const data = JSON.parse(readFileSync(dataPath, "utf8"));
  if (data.schemaVersion !== 1 || !Array.isArray(data.draws)) {
    throw new Error("lotto_data_schema_invalid");
  }
  const draws = data.draws.map(validateDraw).sort((a, b) => b.round - a.round);
  return {
    ...data,
    draws,
    latestRound: draws.reduce((max, draw) => Math.max(max, draw.round), 0)
  };
}

async function fetchOfficialDraw(round) {
  const url = `${OFFICIAL_DRAW_API_URL}&drwNo=${round}`;
  const response = await fetch(url, {
    headers: {
      "user-agent": "Mozilla/5.0 lotto-dashboard-data-updater"
    }
  });
  if (!response.ok) throw new Error(`official_draw_http_${response.status}_round_${round}`);

  const payload = await response.json();
  if (payload.returnValue !== "success") {
    return null;
  }

  return validateDraw({
    round: payload.drwNo,
    date: payload.drwNoDate,
    numbers: [
      payload.drwtNo1,
      payload.drwtNo2,
      payload.drwtNo3,
      payload.drwtNo4,
      payload.drwtNo5,
      payload.drwtNo6
    ],
    bonus: payload.bnusNo,
    firstPrizeAmount: payload.firstWinamnt,
    firstWinnerCount: payload.firstPrzwnerCo,
    totalSellAmount: payload.totSellamnt
  });
}

async function findNewDraws(currentLatestRound) {
  const newDraws = [];
  for (let offset = 1; offset <= MAX_BACKFILL_ROUNDS; offset += 1) {
    const round = currentLatestRound + offset;
    const draw = await fetchOfficialDraw(round);
    if (!draw) break;
    newDraws.push(draw);
  }

  return newDraws;
}

const current = readCurrentData();
const newDraws = await findNewDraws(current.latestRound);

if (newDraws.length === 0) {
  console.log(`No new rounds after ${current.latestRound}.`);
  process.exit(0);
}

const merged = [...newDraws, ...current.draws]
  .map(validateDraw)
  .sort((a, b) => b.round - a.round);

writeFileSync(dataPath, `${JSON.stringify({
  ...current,
  source: "incremental-official-lt645-result",
  updatedAt: new Date().toISOString(),
  latestRound: merged[0].round,
  draws: merged
}, null, 2)}\n`, "utf8");

console.log(`Added ${newDraws.length} round(s). latestRound=${merged[0].round}`);
