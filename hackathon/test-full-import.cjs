const assert = require("node:assert/strict");
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || "playwright");
(async () => {
  const browser = await chromium.launch({channel:"msedge", headless:true});
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    let starts = 0, cancelled = 0, analyses = 0, job, oldBackend = false;
    const dataset = {source:"upload", name:"/tmp/full.jsonl", extension:"jsonl", displayName:"Full test split", importedRows:12001, sampleOnly:false, hubConfig:"chat", hubSplit:"train", hubRevision:"pinned"};
    const workflow = {id:"full-ui", status:"waiting", dataset, state:{dataset}, pendingAction:{type:"dataset_ready"}, profile:{facts:{rows:12001, sampledRows:600, validRows:600}, classification:{task:"assistant"}}};
    await page.route("**/api/**", async route => {
      const request = route.request(), path = new URL(request.url()).pathname;
      let data = [], status = 200;
      if (path === "/api/datasets/huggingface") {
        const body = request.postDataJSON();
        assert.equal(body.mode, "all");
        assert.equal(body.configuration, null);
        assert.equal(body.max_rows, undefined);
        if (oldBackend) return route.fulfill({json:{...dataset, sampleOnly:true, importedRows:600}});
        starts++;
        job = {id:"00000000-0000-0000-0000-" + String(starts).padStart(12,"0"), status:"running", importedRows:12000, importedBytes:60000000, totalRows:12001};
        data = job; status = 202;
      } else if (path.endsWith("/cancel")) {
        cancelled++;
        job = {...job, status:"cancelled", error:"Import cancelled"};
        data = job;
      } else if (path.endsWith("/select")) {
        assert.deepEqual(request.postDataJSON().configurations, ["chat","math"]);
        analyses++;
        job.selectedWorkflows = {'"chat"':"full-ui", '"math"':"math-ui"};
        data = {workflows:[{configuration:"chat",id:"full-ui"},{configuration:"math",id:"math-ui"}]};
      } else if (path.startsWith("/api/datasets/imports/")) {
        data = job;
      } else if (path === "/api/workflow/runs" && request.method() === "POST") {
        const body = request.postDataJSON();
        assert.equal(body.dataset.sampleOnly, false);
        assert.equal(body.dataset.importedRows, 12001);
        assert.equal(body.analysis_only, true);
        analyses++; data = workflow;
      } else if (path === "/api/workflow/runs/full-ui") data = workflow;
      else if (path === "/api/hardware") data = {trainerOnline:true, memoryGb:15, name:"Test GPU"};
      await route.fulfill({status, json:data});
    });
    await page.goto("http://127.0.0.1:4173/#/workspace");
    await page.getByRole("button", {name:"Hugging Face", exact:true}).click();
    assert.equal(await page.getByLabel("Import scope", {exact:true}).inputValue(), "all");
    assert(await page.locator("#hf-rows").isDisabled());
    assert(!(await page.getByLabel("Subset / configuration (optional)", {exact:true}).isVisible()));
    await page.getByLabel("Dataset URL or ID", {exact:true}).fill("owner/data");
    await page.getByRole("button", {name:"Import Full Data", exact:true}).click();
    await page.getByRole("button", {name:"Cancel import", exact:true}).waitFor();
    await page.reload();
    await page.getByRole("button", {name:"Cancel import", exact:true}).waitFor();
    assert.equal(starts, 1);
    assert.equal(analyses, 0);
    assert(await page.getByRole("button", {name:"Import Full Data", exact:true}).isDisabled());
    await page.setViewportSize({width:390, height:844});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.getByRole("button", {name:"Cancel import", exact:true}).click();
    await page.getByRole("button", {name:"Import another dataset", exact:true}).click();
    assert.equal(cancelled, 1);
    assert.equal(analyses, 0);
    await page.getByRole("button", {name:"Import Full Data", exact:true}).click();
    await page.getByRole("button", {name:"Cancel import", exact:true}).waitFor();
    job = {...job, status:"complete", importedRows:24002, result:{allSubsets:true, subsets:[
      {configuration:"chat", status:"analyzed", dataset, profile:workflow.profile},
      {configuration:"math", status:"analyzed", dataset:{...dataset,hubConfig:"math"}, profile:workflow.profile},
      {configuration:"missing", status:"failed", error:"No train split"}
    ]}};
    await page.getByRole("heading", {name:"Choose subsets for training", exact:true}).waitFor();
    assert.equal(analyses, 0);
    assert(await page.getByRole("button", {name:"Continue with selected subsets",exact:true}).isDisabled());
    assert(await page.getByRole("checkbox", {name:"Use missing for training",exact:true}).isDisabled());
    await page.getByRole("checkbox", {name:"Use chat for training",exact:true}).check();
    await page.getByRole("checkbox", {name:"Use math for training",exact:true}).check();
    await page.getByRole("button", {name:"Continue with selected subsets",exact:true}).click();
    await page.getByRole("link", {name:"Open math workflow",exact:true}).waitFor();
    await page.reload();
    await page.getByRole("link", {name:"Open chat workflow",exact:true}).click();
    await page.getByRole("heading", {name:"Dataset ready", exact:true}).waitFor();
    await page.getByText("Full Hugging Face split:", {exact:false}).waitFor();
    assert.equal(analyses, 1);
    assert(await page.evaluate(() => !!localStorage.getItem("forgetune-import")));
    oldBackend = true;
    await page.goto("http://127.0.0.1:4173/#/workspace");
    await page.getByRole("button", {name:"Import another dataset",exact:true}).click();
    await page.getByLabel("Dataset URL or ID", {exact:true}).fill("owner/data");
    await page.getByRole("button", {name:"Import Full Data", exact:true}).click();
    await page.getByText("The trainer does not support full imports yet.", {exact:false}).waitFor();
    assert.equal(analyses, 1);
    assert.deepEqual(errors, []);
    console.log("All-subset UI passed: no upfront subset choice, audits before selection, multiple workflows, failed-subset guard, refresh, cancellation, mobile.");
  } finally { await browser.close(); }
})().catch(error => {console.error(error); process.exitCode = 1;});
