import fs from "node:fs/promises";
import path from "node:path";

import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath =
  "/Users/tanishaupadhyay/Library/CloudStorage/GoogleDrive-upad.tanisha@gmail.com/My Drive/LMA_Individual_Project/manifests/bhojpuri_filter_compact_audit_v1.xlsx";
const outputDir = path.resolve("outputs/bhojpuri_filter_audit_update");
const outputPath = path.join(outputDir, "bhojpuri_filter_compact_audit_v1.xlsx");
const previewOnly = process.argv.includes("--preview-only");

const input = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(input);

await fs.mkdir(outputDir, { recursive: true });

if (process.argv.includes("--dump-review")) {
  const auditSheet = workbook.worksheets.getItem("Audit");
  const borderlineOnly = process.argv.includes("--borderline-only");
  const startRow = borderlineOnly ? 22 : 2;
  const endRow = borderlineOnly ? 41 : 61;
  const sourceRows = auditSheet.getRange(`A${startRow}:H${endRow}`).values;
  sourceRows.forEach((row, index) => {
    console.log(
      JSON.stringify({
        excelRow: index + startRow,
        auditId: row[0],
        filterDecision: row[1],
        score: row[2],
        title: row[3],
        textSample: String(row[7] ?? "").slice(0, 700),
      }),
    );
  });
  process.exit(0);
}

const beforePreview = await workbook.render({
  sheetName: "Audit",
  range: "A1:O8",
  scale: 1,
  format: "png",
});
await fs.writeFile(
  path.join(outputDir, "before_audit_preview.png"),
  new Uint8Array(await beforePreview.arrayBuffer()),
);

for (const range of ["Audit!A1:O14", "Audit!A25:O30", "Audit!A56:O61"]) {
  const inspection = await workbook.inspect({
    kind: "table",
    range,
    include: "values,formulas",
    tableMaxRows: 14,
    tableMaxCols: 15,
    tableMaxCellChars: 180,
    maxChars: 12000,
  });
  console.log(inspection.ndjson);
}

const styleInspection = await workbook.inspect({
  kind: "computedStyle",
  sheetId: "Audit",
  range: "I1:O4",
  maxChars: 4000,
});
console.log(styleInspection.ndjson);

if (previewOnly) {
  console.log(JSON.stringify({ previewOnly: true, inputPath }));
  process.exit(0);
}

const audit = workbook.worksheets.getItem("Audit");
const englishRows = new Set([8, 9, 10]);
const bilingualRows = new Set([13, 28]);
const hindiRows = new Set([
  22, 23, 24, 25, 27,
  42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
]);
const romanBhojpuriRows = new Set([59, 60]);
const romanMixedRows = new Set([61]);
const rejectedRows = new Set([
  ...englishRows,
  ...bilingualRows,
  ...hindiRows,
  ...romanBhojpuriRows,
  ...romanMixedRows,
]);

const reviewValues = [];
for (let excelRow = 2; excelRow <= 61; excelRow += 1) {
  if (rejectedRows.has(excelRow)) {
    let languageLabel = "Other/unclear";
    let translationLabel = "Unsure";
    let note = "Manual review: English text; unsuitable for the Bhojpuri tokenizer corpus.";
    if (bilingualRows.has(excelRow)) {
      languageLabel = "Mixed";
      translationLabel = "Likely translated";
      note = "Manual review: bilingual English–Bhojpuri text; unsuitable for the monolingual Bhojpuri tokenizer corpus.";
    } else if (hindiRows.has(excelRow)) {
      languageLabel = "Hindi";
      translationLabel = "Native/original";
      note = "Manual review: grammatical Hindi, but the wrong language for the Bhojpuri tokenizer corpus.";
    } else if (romanBhojpuriRows.has(excelRow)) {
      languageLabel = "Bhojpuri";
      translationLabel = "Native/original";
      note = "Manual review: Roman-script Bhojpuri; unsuitable for the Devanagari tokenizer corpus.";
    } else if (romanMixedRows.has(excelRow)) {
      languageLabel = "Mixed";
      translationLabel = "Native/original";
      note = "Manual review: Roman-script mixed Hindi/Bhojpuri; unsuitable for the Devanagari tokenizer corpus.";
    }
    reviewValues.push([
      languageLabel,
      translationLabel,
      "Reject",
      "Unsure",
      "Remove",
      "Tanisha Upadhyay",
      note,
    ]);
  } else {
    reviewValues.push([
      "Bhojpuri",
      "Native/original",
      "Good",
      "Unsure",
      "Keep",
      "Tanisha Upadhyay",
      "Manual review: acceptable for tokenizer training.",
    ]);
  }
}
audit.getRange("I2:O61").values = reviewValues;
const actionCounts = reviewValues.reduce(
  (counts, row) => {
    counts[row[4]] = (counts[row[4]] ?? 0) + 1;
    return counts;
  },
  {},
);
console.log(JSON.stringify({ actionCounts }));

const instructionPreview = await workbook.render({
  sheetName: "Instructions",
  range: "A1:E9",
  scale: 1.5,
  format: "png",
});
await fs.writeFile(
  path.join(outputDir, "after_instructions_preview.png"),
  new Uint8Array(await instructionPreview.arrayBuffer()),
);

const afterPreview = await workbook.render({
  sheetName: "Audit",
  range: "A1:O14",
  scale: 1,
  format: "png",
});
await fs.writeFile(
  path.join(outputDir, "after_audit_preview.png"),
  new Uint8Array(await afterPreview.arrayBuffer()),
);

const check = await workbook.inspect({
  kind: "table",
  range: "Audit!A1:O14",
  include: "values,formulas",
  tableMaxRows: 14,
  tableMaxCols: 15,
  maxChars: 12000,
});
console.log(check.ndjson);

for (const range of ["Audit!A22:O28", "Audit!A42:O61"]) {
  const decisionCheck = await workbook.inspect({
    kind: "table",
    range,
    include: "values,formulas",
    tableMaxRows: 20,
    tableMaxCols: 15,
    tableMaxCellChars: 90,
    maxChars: 12000,
  });
  console.log(decisionCheck.ndjson);
}

const summaryCheck = await workbook.inspect({
  kind: "table",
  range: "Instructions!D3:E7",
  include: "values,formulas",
  tableMaxRows: 5,
  tableMaxCols: 2,
  maxChars: 3000,
});
console.log(summaryCheck.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(JSON.stringify({ outputPath, reviewed: 60, rejected: rejectedRows.size }));
