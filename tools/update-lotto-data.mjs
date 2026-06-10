import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const dataPath = resolve(root, "lotto-data.json");
const OFFICIAL_DRAW_API_URLS = [
  "https://www.dhlottery.co.kr/common.do?method=getLottoNumber",
  "https://dhlottery.co.kr/common.do?method=getLottoNumber",
  "https://www.nlotto.co.kr/common.do?method=getLottoNumber"
];
const MAX_BACKFILL_ROUNDS = 20;
const REQUEST_RETRY_COUNT = 3;
const REQUEST_RETRY_DELAY_MS = 1000;

function sleep(ms) {
  return new Promise((resolveSleep) => setTimeout(resolveSleep, ms));
}

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
  let lastError = null;

  for (const baseUrl of OFFICIAL_DRAW_API_URLS) {
    const url = `${baseUrl}&drwNo=${round}`;

    for (let attempt = 1; attempt <= REQUEST_RETRY_COUNT; attempt += 1) {
      try {
        const response = await fetch(url, {
          headers: {
            accept: "application/json,text/plain,*/*",
            "accept-language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
            referer: "https://www.dhlottery.co.kr/gameResult.do?method=byWin",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
          }
        });
        const body = await response.text();

        if (!response.ok) {
          throw new Error(`official_draw_http_${response.status}_url_${baseUrl}_round_${round}_${body.slice(0, 120)}`);
        }

        const trimmedBody = body.trim();
        if (!trimmedBody.startsWith("{")) {
          throw new Error(`official_draw_non_json_url_${baseUrl}_round_${round}_${trimmedBody.slice(0, 180).replace(/\s+/g, " ")}`);
        }

        const payload = JSON.parse(trimmedBody);
        if (payload.returnValue !== "success") {
          return null;
        }

        return validateOfficialPayload(payload, round);
      } catch (error) {
        lastError = error;
        console.warn(`official_draw_fetch_attempt_failed round=${round} url=${baseUrl} attempt=${attempt} message=${error.message}`);
        if (attempt < REQUEST_RETRY_COUNT) await sleep(REQUEST_RETRY_DELAY_MS * attempt);
      }
    }
  }

  throw lastError;
}

function validateOfficialPayload(payload, requestedRound) {
  if (payload.returnValue !== "success") {
    return null;
  }

  if (Number(payload.drwNo) !== requestedRound) {
    throw new Error(`official_draw_round_mismatch_requested_${requestedRound}_received_${payload.drwNo}`);
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
console.log(`Current latestRound=${current.latestRound}`);

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
