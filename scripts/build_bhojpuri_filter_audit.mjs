import fs from "node:fs/promises";
import path from "node:path";
import zlib from "node:zlib";
import { promisify } from "node:util";

import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const gunzip = promisify(zlib.gunzip);

const driveRoot =
  "/Users/tanishaupadhyay/Library/CloudStorage/GoogleDrive-upad.tanisha@gmail.com/My Drive/LMA_Individual_Project";
const outputDir = path.resolve("outputs/bhojpuri_filter_audit");
const outputPath = path.join(outputDir, "bhojpuri_filter_compact_audit_v1.xlsx");

const sources = [
  {
    decision: "Accepted",
    path: "bhojpuri/interim/anjoria_pre2023_accepted_v1.jsonl.gz",
  },
  {
    decision: "Borderline",
    path: "bhojpuri/interim/anjoria_pre2023_borderline_v1.jsonl.gz",
  },
  {
    decision: "Rejected",
    path: "bhojpuri/interim/anjoria_pre2023_rejected_v1.jsonl.gz",
  },
];

function textSample(text, maxChars = 900) {
  const collapsed = text.replace(/\s+/gu, " ").trim();
  return collapsed.length <= maxChars
    ? collapsed
    : `${collapsed.slice(0, maxChars).trimEnd()}…`;
}

async function loadJsonlGz(relativePath) {
  const compressed = await fs.readFile(path.join(driveRoot, relativePath));
  const content = (await gunzip(compressed)).toString("utf8");
  return content
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line.replace(/:\s*NaN(?=\s*[,}])/gu, ": null")));
}

function evenlySpacedSample(documents, count) {
  const sorted = [...documents].sort((a, b) => {
    const rawScoreA =
      a.metadata.hindi_bhojpuri_discrimination.score_bho_minus_hi;
    const rawScoreB =
      b.metadata.hindi_bhojpuri_discrimination.score_bho_minus_hi;
    const scoreA = Number.isFinite(rawScoreA) ? rawScoreA : Infinity;
    const scoreB = Number.isFinite(rawScoreB) ? rawScoreB : Infinity;
    return scoreA - scoreB || a.document_id.localeCompare(b.document_id);
  });
  const selected = [];
  const used = new Set();
  for (let index = 0; index < count; index += 1) {
    const position = Math.round((index * (sorted.length - 1)) / (count - 1));
    let candidate = position;
    while (used.has(candidate) && candidate + 1 < sorted.length) candidate += 1;
    while (used.has(candidate) && candidate - 1 >= 0) candidate -= 1;
    used.add(candidate);
    selected.push(sorted[candidate]);
  }
  return selected;
}

const rows = [];
for (const source of sources) {
  const documents = await loadJsonlGz(source.path);
  const selected = evenlySpacedSample(documents, 20);
  for (const document of selected) {
    const discrimination = document.metadata.hindi_bhojpuri_discrimination;
    rows.push({
      auditId: `${source.decision.toLowerCase()}_${String(
        rows.filter((row) => row.filterDecision === source.decision).length + 1,
      ).padStart(2, "0")}`,
      filterDecision: source.decision,
      score: discrimination.score_bho_minus_hi,
      title: document.metadata.title ?? "",
      documentId: document.document_id,
      sourceName: document.source_name,
      sourceUrl: document.source_url,
      textSample: textSample(document.text),
    });
  }
}

const workbook = Workbook.create();
const instructions = workbook.worksheets.add("Instructions");
const audit = workbook.worksheets.add("Audit");
instructions.showGridLines = false;
audit.showGridLines = false;

instructions.getRange("A1:E1").merge();
instructions.getRange("A1").values = [["Bhojpuri Filter Audit — Anjoria"]];
instructions.getRange("A1:E1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", size: 16 },
  verticalAlignment: "center",
};
instructions.getRange("A1:E1").format.rowHeight = 34;

instructions.getRange("A3:B9").values = [
  ["Purpose", "Check whether the automatic Hindi–Bhojpuri language filter is making sensible decisions before the final corpus is frozen."],
  ["Sample", "60 Anjoria documents: 20 accepted, 20 borderline, and 20 rejected."],
  ["Selection", "Deterministic, evenly spaced samples across each category's Bhojpuri-minus-Hindi score range."],
  ["What to fill", "Complete the five yellow review columns in the Audit sheet; add your name and notes where useful."],
  ["Language", "Choose Bhojpuri, Hindi, Mixed, or Other/unclear."],
  ["Translation", "Judge the writing itself: Native/original, Likely translated, or Unsure."],
  ["Final action", "Keep if suitable for tokenizer training; Review if uncertain; Remove if unsuitable."],
];
instructions.getRange("A3:A9").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D" },
  verticalAlignment: "top",
};
instructions.getRange("B3:B9").format = {
  fill: "#F7FAFC",
  wrapText: true,
  verticalAlignment: "top",
};
instructions.getRange("A3:B9").format.borders = {
  insideHorizontal: { style: "thin", color: "#D9E2F3" },
  outside: { style: "thin", color: "#9EADBA" },
};
instructions.getRange("A3:A9").format.columnWidth = 18;
instructions.getRange("B3:B9").format.columnWidth = 92;
instructions.getRange("A3:B9").format.autofitRows();

instructions.getRange("D3:E7").values = [
  ["Audit total", null],
  ["Accepted", null],
  ["Borderline", null],
  ["Rejected", null],
  ["Reviewed", null],
];
instructions.getRange("E3:E7").formulas = [
  ["=COUNTA('Audit'!$A$2:$A$61)"],
  ["=COUNTIF('Audit'!$B$2:$B$61,\"Accepted\")"],
  ["=COUNTIF('Audit'!$B$2:$B$61,\"Borderline\")"],
  ["=COUNTIF('Audit'!$B$2:$B$61,\"Rejected\")"],
  ["=COUNTIF('Audit'!$M$2:$M$61,\"Keep\")+COUNTIF('Audit'!$M$2:$M$61,\"Review\")+COUNTIF('Audit'!$M$2:$M$61,\"Remove\")"],
];
instructions.getRange("D3:D7").format = {
  fill: "#E2F0D9",
  font: { bold: true, color: "#385723" },
};
instructions.getRange("E3:E7").format = {
  fill: "#F3F8EF",
  font: { bold: true },
  horizontalAlignment: "right",
  numberFormat: "0",
};
instructions.getRange("D3:E7").format.borders = {
  preset: "outside",
  style: "thin",
  color: "#A9D18E",
};
instructions.getRange("D3:D7").format.columnWidth = 18;
instructions.getRange("E3:E7").format.columnWidth = 12;

const headers = [
  "audit_id",
  "filter_decision",
  "bho_minus_hindi_score",
  "title",
  "document_id",
  "source_name",
  "source_url",
  "text_sample",
  "language_label",
  "translation_label",
  "quality_label",
  "duplicate_label",
  "final_action",
  "reviewer",
  "notes",
];
const values = rows.map((row) => [
  row.auditId,
  row.filterDecision,
  row.score,
  row.title,
  row.documentId,
  row.sourceName,
  row.sourceUrl,
  row.textSample,
  "",
  "",
  "",
  "",
  "",
  "",
  "",
]);

audit.getRange("A1:O1").values = [headers];
audit.getRange("A2:O61").values = values;
audit.getRange("A1:O1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
};
audit.getRange("A1:O1").format.rowHeight = 38;
audit.freezePanes.freezeRows(1);
audit.freezePanes.freezeColumns(3);

audit.getRange("A2:O61").format = {
  verticalAlignment: "top",
};
audit.getRange("C2:C61").format.numberFormat = "0.000";
audit.getRange("C2:C61").format.horizontalAlignment = "right";
audit.getRange("D2:H61").format.wrapText = true;
audit.getRange("I2:O61").format = {
  fill: "#FFF2CC",
  verticalAlignment: "top",
  wrapText: true,
};
audit.getRange("A2:O61").format.borders = {
  insideHorizontal: { style: "thin", color: "#E7E6E6" },
};

const widths = [12, 14, 17, 36, 29, 18, 34, 78, 18, 20, 20, 22, 15, 16, 34];
for (let column = 0; column < widths.length; column += 1) {
  audit.getRangeByIndexes(0, column, 61, 1).format.columnWidth = widths[column];
}
audit.getRange("A2:O61").format.rowHeight = 76;

audit.getRange("I2:I61").dataValidation = {
  rule: { type: "list", values: ["Bhojpuri", "Hindi", "Mixed", "Other/unclear"] },
};
audit.getRange("J2:J61").dataValidation = {
  rule: { type: "list", values: ["Native/original", "Likely translated", "Unsure"] },
};
audit.getRange("K2:K61").dataValidation = {
  rule: { type: "list", values: ["Good", "Usable with cleanup", "Reject"] },
};
audit.getRange("L2:L61").dataValidation = {
  rule: { type: "list", values: ["Unique", "Duplicate/near-duplicate", "Unsure"] },
};
audit.getRange("M2:M61").dataValidation = {
  rule: { type: "list", values: ["Keep", "Review", "Remove"] },
};

audit.getRange("B2:B61").conditionalFormats.add("containsText", {
  text: "Accepted",
  format: { fill: "#E2F0D9", font: { color: "#385723", bold: true } },
});
audit.getRange("B2:B61").conditionalFormats.add("containsText", {
  text: "Borderline",
  format: { fill: "#FFF2CC", font: { color: "#7F6000", bold: true } },
});
audit.getRange("B2:B61").conditionalFormats.add("containsText", {
  text: "Rejected",
  format: { fill: "#FCE4D6", font: { color: "#9C0006", bold: true } },
});

const table = audit.tables.add("A1:O61", true, "BhojpuriFilterAudit");
table.style = "TableStyleMedium2";
table.showBandedRows = true;
table.showFilterButton = true;

await fs.mkdir(outputDir, { recursive: true });

const instructionPreview = await workbook.render({
  sheetName: "Instructions",
  range: "A1:E9",
  scale: 1.5,
  format: "png",
});
await fs.writeFile(
  path.join(outputDir, "instructions_preview.png"),
  new Uint8Array(await instructionPreview.arrayBuffer()),
);

const auditPreview = await workbook.render({
  sheetName: "Audit",
  range: "A1:O8",
  scale: 1,
  format: "png",
});
await fs.writeFile(
  path.join(outputDir, "audit_preview.png"),
  new Uint8Array(await auditPreview.arrayBuffer()),
);

const inspect = await workbook.inspect({
  kind: "table",
  range: "Audit!A1:O8",
  include: "values,formulas",
  tableMaxRows: 8,
  tableMaxCols: 15,
  maxChars: 8000,
});
console.log(inspect.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(JSON.stringify({ outputPath, rows: rows.length }));
