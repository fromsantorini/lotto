import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const dataPath = resolve(root, "lotto-data.json");
const OFFICIAL_RESULT_PAGE_URLS = [
  "https://www.dhlottery.co.kr/lt645/result",
  "https://dhlottery.co.kr/lt645/result"
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

function firstMatch(text, patterns) {
  for (const pattern of patterns) {
    const match = text.match(pattern);
    if (match) return match;
  }
  return null;
}

function parseResultDate(html) {
  const ymdMatch = html.match(/"ltRflYmd"\s*:\s*"?(\d{8})"?/);
  if (ymdMatch) {
    return `${ymdMatch[1].slice(0, 4)}-${ymdMatch[1].slice(4, 6)}-${ymdMatch[1].slice(6, 8)}`;
  }

  const koreanDateMatch = html.match(/\(?\s*(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일\s*추첨\s*\)?/);
  if (koreanDateMatch) {
    return `${koreanDateMatch[1]}-${koreanDateMatch[2].padStart(2, "0")}-${koreanDateMatch[3].padStart(2, "0")}`;
  }

  return "";
}

function parseOfficialResultPage(html, requestedRound) {
  if (!html.includes("lt645") && !html.includes("당첨결과") && !html.includes("win_result") && !html.includes("ltEpsd")) {
    throw new Error(`official_result_page_unexpected_html_${stripTags(html).slice(0, 160)}`);
  }

  const roundMatch = firstMatch(html, [
    /"ltEpsd"\s*:\s*"?(\d+)"?/,
    /(?:name|id)=["']ltEpsd["'][^>]*value=["'](\d+)["']/i,
    /value=["'](\d+)["'][^>]*(?:name|id)=["']ltEpsd["']/i,
    /ltEpsd[^0-9]{0,80}(\d{3,5})/i,
    /<h4>\s*<strong>\s*(\d+)\s*회\s*<\/strong>\s*당첨결과\s*<\/h4>/,
    /(\d+)\s*회\s*당첨결과/
  ]);
  if (!roundMatch && !requestedRound) {
    throw new Error(`official_result_round_parse_failed_${stripTags(html).slice(0, 160)}`);
  }
  const round = roundMatch ? Number(roundMatch[1]) : requestedRound;

  if (requestedRound && round !== requestedRound) {
    throw new Error(`official_result_round_mismatch_requested_${requestedRound}_received_${round}`);
  }

  const date = parseResultDate(html);

  const quotedEmbeddedNumberMatches = [
    ...html.matchAll(/"tm[1-6]WnNo"\s*:\s*"?(\d{1,2})"?/g),
    ...html.matchAll(/"bnsWnNo"\s*:\s*"?(\d{1,2})"?/g)
  ].map((match) => Number(match[1]));
  const looseEmbeddedNumberMatches = [
    ...html.matchAll(/\btm[1-6]WnNo\b[^0-9]{0,40}(\d{1,2})/g),
    ...html.matchAll(/\bbnsWnNo\b[^0-9]{0,40}(\d{1,2})/g)
  ].map((match) => Number(match[1]));

  const markupNumberMatches = [
    ...html.matchAll(/<span[^>]*class=["'][^"']*ball_645[^"']*["'][^>]*>\s*(\d{1,2})\s*<\/span>/g),
    ...html.matchAll(/<[^>]*class=["'][^"']*(?:ball|number|num|win)[^"']*["'][^>]*>\s*(\d{1,2})\s*<\/[^>]+>/g)
  ]
    .map((match) => Number(match[1]))
    .filter((number) => number >= 1 && number <= 45);
  const ballMatches = quotedEmbeddedNumberMatches.length >= 7
    ? quotedEmbeddedNumberMatches
    : looseEmbeddedNumberMatches.length >= 7
      ? looseEmbeddedNumberMatches
      : markupNumberMatches;

  if (ballMatches.length < 7) {
    throw new Error(`official_result_numbers_parse_failed_round_${round}_quoted=${quotedEmbeddedNumberMatches.length}_loose=${looseEmbeddedNumberMatches.length}_markup=${markupNumberMatches.length}_${stripTags(html).slice(0, 160)}`);
  }

  const firstPrizeMatch = firstMatch(html, [
    /"rnk1WnAmt"\s*:\s*"?([\d,]+)"?/,
    /rnk1WnAmt\s*[:=]\s*"?([\d,]+)"?/
  ]);
  const firstWinnerMatch = firstMatch(html, [
    /"rnk1WnNope"\s*:\s*"?([\d,]+)"?/,
    /rnk1WnNope\s*[:=]\s*"?([\d,]+)"?/
  ]);

  const prizeRows = [...html.matchAll(/<tr[^>]*>([\s\S]*?)<\/tr>/g)]
    .map((match) => match[1])
    .map((rowHtml) => [...rowHtml.matchAll(/<t[hd][^>]*>([\s\S]*?)<\/t[hd]>/g)].map((cell) => stripTags(cell[1])))
    .filter((cells) => cells.length > 0);

  const firstPrizeRow = prizeRows.find((cells) => cells.some((cell) => cell === "1등" || cell.startsWith("1등 ")));
  const firstPrizeAmount = firstPrizeMatch ? toMoneyNumber(firstPrizeMatch[1]) : firstPrizeRow ? toMoneyNumber(firstPrizeRow[2] || firstPrizeRow[1]) : 0;
  const firstWinnerCount = firstWinnerMatch ? toMoneyNumber(firstWinnerMatch[1]) : firstPrizeRow ? toMoneyNumber(firstPrizeRow[3] || firstPrizeRow[2]) : 0;

  return validateDraw({
    round,
    date,
    numbers: ballMatches.slice(0, 6),
    bonus: ballMatches[6],
    firstPrizeAmount,
    firstWinnerCount,
    totalSellAmount: 0
  });
}

async function fetchOfficialDraw(round) {
  let lastError = null;

  for (const baseUrl of OFFICIAL_RESULT_PAGE_URLS) {
    const url = `${baseUrl}?ltEpsd=${round}`;

    for (let attempt = 1; attempt <= REQUEST_RETRY_COUNT; attempt += 1) {
      try {
        const response = await fetch(url, {
          headers: {
            accept: "text/html,*/*",
            "accept-language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
            referer: "https://www.dhlottery.co.kr/lt645/result",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
          }
        });
        const body = await response.text();

        if (!response.ok) {
          throw new Error(`official_result_http_${response.status}_url_${baseUrl}_${stripTags(body).slice(0, 120)}`);
        }

        const trimmedBody = body.trim();
        if (trimmedBody.includes("조회된 결과가 없습니다") || trimmedBody.includes("검색된 결과가 없습니다")) {
          return null;
        }

        return parseOfficialResultPage(trimmedBody, round);
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
