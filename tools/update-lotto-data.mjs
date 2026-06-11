import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const dataPath = resolve(root, "lotto-data.json");
const OFFICIAL_RESULT_API_URL = "https://www.dhlottery.co.kr/lt645/selectPstLt645InfoNew.do";
const OFFICIAL_RESULT_PAGE_URLS = [
  "https://www.dhlottery.co.kr/lt645/result"
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

function decodeHtml(value) {
  return String(value || "")
    .replace(/&#(\d+);/g, (_, code) => String.fromCharCode(Number(code)))
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'");
}

function stripTags(value) {
  return decodeHtml(String(value || "").replace(/<[^>]+>/g, " ")).replace(/\s+/g, " ").trim();
}

function toMoneyNumber(value) {
  return Number(String(value || "").replace(/[^\d]/g, "")) || 0;
}

function formatYmd(value) {
  const digits = String(value || "").replace(/[^\d]/g, "");
  if (digits.length !== 8) return "";
  return `${digits.slice(0, 4)}-${digits.slice(4, 6)}-${digits.slice(6, 8)}`;
}

function parseResultDate(html) {
  const ymdMatch = html.match(/"ltRflYmd"\s*:\s*"?(\d{8})"?/);
  if (ymdMatch) {
    return formatYmd(ymdMatch[1]);
  }

  const koreanDateMatch = html.match(/\(?\s*(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일\s*추첨\s*\)?/);
  if (koreanDateMatch) {
    return `${koreanDateMatch[1]}-${koreanDateMatch[2].padStart(2, "0")}-${koreanDateMatch[3].padStart(2, "0")}`;
  }

  return "";
}

function parseOfficialResultPayload(html, round) {
  const mainNumbers = [...html.matchAll(/["']?tm[1-6]WnNo["']?\s*[:=]\s*["']?(\d{1,2})["']?/g)]
    .map((match) => Number(match[1]));
  const bonusNumbers = [...html.matchAll(/["']?bnsWnNo["']?\s*[:=]\s*["']?(\d{1,2})["']?/g)]
    .map((match) => Number(match[1]));

  if (mainNumbers.length !== 6 || bonusNumbers.length < 1) {
    throw new Error(`official_result_payload_parse_failed_round_${round}_main=${mainNumbers.length}_bonus=${bonusNumbers.length}`);
  }

  const firstPrizeMatch = html.match(/["']?rnk1WnAmt["']?\s*[:=]\s*["']?([\d,]+)["']?/);
  const firstWinnerMatch = html.match(/["']?rnk1WnNope["']?\s*[:=]\s*["']?([\d,]+)["']?/);
  const firstPrizeAmount = firstPrizeMatch ? toMoneyNumber(firstPrizeMatch[1]) : 0;
  const firstWinnerCount = firstWinnerMatch ? toMoneyNumber(firstWinnerMatch[1]) : 0;
  const date = parseResultDate(html);

  if (!date || firstPrizeAmount <= 0 || firstWinnerCount <= 0) {
    return { notAvailable: true };
  }

  return {
    draw: validateDraw({
    round,
    date,
    numbers: mainNumbers,
    bonus: bonusNumbers[0],
    firstPrizeAmount,
    firstWinnerCount,
    totalSellAmount: 0
    })
  };
}

function getRequestHeaders() {
  return {
    accept: "text/html,application/json,*/*",
    "accept-language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    referer: "https://www.dhlottery.co.kr/lt645/result",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
  };
}

export function mapOfficialApiItemToDraw(item) {
  return validateDraw({
    round: item.ltEpsd,
    date: formatYmd(item.ltRflYmd),
    numbers: [item.tm1WnNo, item.tm2WnNo, item.tm3WnNo, item.tm4WnNo, item.tm5WnNo, item.tm6WnNo],
    bonus: item.bnsWnNo,
    firstPrizeAmount: item.rnk1WnAmt,
    firstWinnerCount: item.rnk1WnNope,
    totalSellAmount: item.rlvtEpsdSumNtslAmt || 0
  });
}

export function parseOfficialApiResponse(payload, round) {
  const list = Array.isArray(payload?.data?.list) ? payload.data.list : [];
  if (list.length === 0) return null;

  const exactMatch = list.find((item) => Number(item?.ltEpsd) === round);
  if (!exactMatch) return null;

  return {
    draw: mapOfficialApiItemToDraw(exactMatch)
  };
}

async function fetchText(url) {
  const response = await fetch(url, {
    headers: getRequestHeaders()
  });
  const body = await response.text();
  if (!response.ok) {
    throw new Error(`official_result_http_${response.status}_url_${url}_${stripTags(body).slice(0, 120)}`);
  }
  return body;
}

async function fetchJson(url) {
  const response = await fetch(url, {
    headers: {
      ...getRequestHeaders(),
      accept: "application/json,text/plain,*/*",
      "x-requested-with": "XMLHttpRequest"
    }
  });
  const body = await response.text();
  if (!response.ok) {
    throw new Error(`official_result_http_${response.status}_url_${url}_${stripTags(body).slice(0, 120)}`);
  }
  try {
    return JSON.parse(body);
  } catch (error) {
    throw new Error(`official_result_json_parse_failed_${error.message}`);
  }
}

async function fetchOfficialDrawViaApi(round) {
  const url = `${OFFICIAL_RESULT_API_URL}?srchDir=center&srchLtEpsd=${round}`;
  const payload = await fetchJson(url);
  return parseOfficialApiResponse(payload, round);
}

async function fetchOfficialDraw(round) {
  let lastError = null;

  for (let attempt = 1; attempt <= REQUEST_RETRY_COUNT; attempt += 1) {
    try {
      return await fetchOfficialDrawViaApi(round);
    } catch (error) {
      lastError = error;
      console.warn(`official_result_api_attempt_failed round=${round} attempt=${attempt} message=${error.message}`);
      if (attempt < REQUEST_RETRY_COUNT) await sleep(REQUEST_RETRY_DELAY_MS * attempt);
    }
  }

  for (const baseUrl of OFFICIAL_RESULT_PAGE_URLS) {
    const url = `${baseUrl}?ltEpsd=${round}`;

    for (let attempt = 1; attempt <= REQUEST_RETRY_COUNT; attempt += 1) {
      try {
        const body = await fetchText(url);
        const trimmedBody = body.trim();

        if (trimmedBody.includes("조회된 결과가 없습니다") || trimmedBody.includes("검색된 결과가 없습니다")) {
          return null;
        }

        return parseOfficialResultPayload(trimmedBody, round);
      } catch (error) {
        lastError = error;
        console.warn(`official_result_fetch_attempt_failed round=${round} url=${baseUrl} attempt=${attempt} message=${error.message}`);
        if (attempt < REQUEST_RETRY_COUNT) await sleep(REQUEST_RETRY_DELAY_MS * attempt);
      }
    }
  }

  throw lastError;
}

async function findNewDraws(currentLatestRound) {
  const newDraws = [];

  for (let offset = 1; offset <= MAX_BACKFILL_ROUNDS; offset += 1) {
    const round = currentLatestRound + offset;
    let result = null;
    try {
      result = await fetchOfficialDraw(round);
    } catch (error) {
      if (newDraws.length > 0) {
        console.warn(`Stopping after ${newDraws.length} new round(s). Round ${round} is not available or could not be parsed yet: ${error.message}`);
        break;
      }
      throw error;
    }

    if (!result || result.notAvailable) break;
    newDraws.push(result.draw);
  }

  return newDraws;
}

export async function main() {
  const current = readCurrentData();
  console.log(`Current latestRound=${current.latestRound}`);

  const newDraws = await findNewDraws(current.latestRound);

  if (newDraws.length === 0) {
    console.log(`No new rounds after ${current.latestRound}.`);
    return;
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
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main();
}
