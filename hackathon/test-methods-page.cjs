const fs = require("node:fs");
const app = fs.readFileSync("hackathon/app.js", "utf8");
const index = fs.readFileSync("hackathon/index.html", "utf8");
for (const label of ["LoRA", "QLoRA", "Full fine-tuning", "Prompt + RAG"]) {
  if (!app.includes(label)) throw new Error("Missing method: " + label);
}
if (!app.includes('page === "methods"')) throw new Error("Methods route is missing");
if (!index.includes('href="#/methods"')) throw new Error("Methods navigation is missing");
console.log("methods page checks passed");