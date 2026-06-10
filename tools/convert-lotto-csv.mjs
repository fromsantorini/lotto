import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const inputPath = resolve(root, "data", "lotto-history.csv");
const outputPath = resolve(root, "lotto-data.json");

function parseCsvLine(line) {
  const cells = [];
  let current = "";
  let inQuotes = false;

  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    const next = line[index + 1];
    if (char === '"' && next === '"') {
      current += '"';
      index += 1;
    } else if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === "," && !inQuotes) {
      cells.push(current.trim());
      current = "";
    } else {
      current += char;
    }
  }

  cells.push(current.trim());
  return cells;
}

function toNumber(value, field, round) {
  const clean = String(value || "").replace(/[^\d.-]/g, "");
  const number = Number(clean);
  if (!Number.isFinite(number)) {
    throw new Error(`invalid_${field}_round_${round}`);
  }
  return number;
}

function normalizeRow(row) {
  const round = toNumber(row.round, "round", "unknown");
  const numbers = [row.n1, row.n2, row.n3, row.n4, row.n5, row.n6]
    .map((value, index) => toNumber(value, `n${index + 1}`, round))
    .sort((a, b) => a - b);
  const bonus = toNumber(row.bonus, "bonus", round);

  if (new Set(numbers).size !== 6) throw new Error(`duplicate_numbers_round_${round}`);
  if (numbers.some((number) => number < 1 || number > 45)) throw new Error(`invalid_number_range_round_${round}`);
  if (bonus < 1 || bonus > 45) throw new Error(`invalid_bonus_range_round_${round}`);

  return {
    round,
    date: String(row.date || ""),
    numbers,
    bonus,
    firstPrizeAmount: toNumber(row.firstPrizeAmount || 0, "firstPrizeAmount", round),
    firstWinnerCount: toNumber(row.firstWinnerCount || 0, "firstWinnerCount", round),
    totalSellAmount: toNumber(row.totalSellAmount || 0, "totalSellAmount", round)
  };
}

const raw = readFileSync(inputPath, "utf8").replace(/^\uFEFF/, "").trim();
const [headerLine, ...lines] = raw.split(/\r?\n/).filter(Boolean);
const headers = parseCsvLine(headerLine);
const rows = lines.map((line) => {
  const values = parseCsvLine(line);
  return Object.fromEntries(headers.map((header, index) => [header, values[index] || ""]));
});

const draws = rows.map(normalizeRow).sort((a, b) => b.round - a.round);
const latestRound = draws.reduce((max, draw) => Math.max(max, draw.round), 0);

writeFileSync(outputPath, `${JSON.stringify({
  schemaVersion: 1,
  source: "data/lotto-history.csv",
  updatedAt: new Date().toISOString(),
  latestRound,
  draws
}, null, 2)}\n`, "utf8");

console.log(`Wrote ${draws.length} draws to ${outputPath}`);
