const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const source = fs.readFileSync(__dirname + "/app.js", "utf8");
const context = vm.createContext({
  escapeHtml: value => String(value ?? "").replaceAll("<", "&lt;").replaceAll(">", "&gt;"),
  percentage: value => `${Math.round(value * 100)}%`,
});
vm.runInContext(source.slice(source.indexOf("function gradeDescription("), source.indexOf("function renderWorkflow(")), context);
const ids = [0, 1, 2, 3, 4, 5];
const quality = {metric:"automatic_rubric_pass_rate", evaluatorId:"v1", coverage:1, evaluatedExamples:6, totalExamples:6, estimated:true, judgeModel:"judge", rubricVersion:"v1", score:0.5};
const base = {evaluationIds:ids, qualityEvidence:quality, samples:ids.map(rowId => ({rowId, baseOutput:"answer", grade:{task:"math", verdict:"pass", reason:"<script>"}}))};
const tuned = {...base, qualityEvidence:{...quality, score:1}, samples:ids.map(rowId => ({rowId, tunedOutput:"answer"}))};
const comparison = {evaluationIds:ids, pairs:[{modelId:"m", base, tuned, passed:true}]};
let html = context.workflowComparison(comparison);
assert.match(html, /Estimated correctness/);
assert.match(html, /50% → 100%/);
assert.match(html, /5 of 6 samples/);
assert.equal((html.match(/<summary>Evaluation row/g) || []).length, 5);
assert(!html.includes("<script>"));
tuned.qualityEvidence = {...quality, coverage:0.5, score:null};
html = context.workflowComparison(comparison);
assert.match(html, /Needs review/);
assert(!html.includes("50% →"));
tuned.qualityEvidence = {...quality, evaluatorId:"different"};
assert(!context.workflowComparison(comparison).includes("50% →"));
console.log("Automatic evaluator summary checks passed.");
