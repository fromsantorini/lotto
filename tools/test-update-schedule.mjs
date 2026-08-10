import assert from "node:assert/strict";
import fs from "node:fs";

const workflow = fs.readFileSync(
  new URL("../.github/workflows/update-lotto-data.yml", import.meta.url),
  "utf8",
);

assert.match(workflow, /cron:\s*["']0 14 \* \* 6["']/, "update must run at Saturday 23:00 KST");
assert.match(workflow, /23:00.*Saturday.*Korea Standard Time/i, "schedule comment must match the cron");

console.log("update schedule check passed");
