const assert = require("node:assert/strict");
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || "playwright");
(async () => {
  const browser = await chromium.launch({channel:"msedge", headless:true});
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on("pageerror", e => errors.push(e.message));
    const workflow = {id:"safeguards-ui", status:"waiting", state:{}, pendingAction:{type:"task_review", analysisLimit:50}};
    let submitted;
    await page.route("**/api/workflow/runs/safeguards-ui**", async route => {
      if (route.request().url().includes("/compass")) return route.fulfill({json:{status:"not_started"}});
      if (route.request().method() === "POST") submitted = route.request().postDataJSON().response;
      await route.fulfill({json:workflow});
    });
    await page.goto("http://127.0.0.1:4173/#/workflow/safeguards-ui");
    await page.getByRole("button", {name:"Cancel", exact:true}).click();
    await page.waitForTimeout(100);
    assert.equal(submitted.action, "abort");
    await page.reload();
    await page.getByLabel("Task description").fill("Classify support tickets");
    await page.getByLabel("Success metric / rubric").fill("Label accuracy");
    await page.getByRole("button", {name:"Confirm task & test models"}).click();
    await page.waitForTimeout(100);
    assert.equal(submitted.task_description, "Classify support tickets");
    assert.equal(submitted.analysis_limit, 50);
    workflow.pendingAction = {type:"model_selection"};
    workflow.state = {task_description:"Classify support tickets", success_metric:"Label accuracy", analysis_limit:50,
      model_benchmark:{developmentIds:Array.from({length:50}, (_,i) => i), results:[], reason:"Awaiting results"}};
    await page.reload();
    await page.getByRole("heading", {name:"Model selection benchmark",exact:true}).waitFor();
    assert.equal(await page.locator("#compass-form").count(), 0);
    await page.setViewportSize({width:390,height:844});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    assert.deepEqual(errors, []);
    console.log("Pipeline UI passed: task checkpoint, configurable sample size, large-benchmark hosted guard, mobile.");
  } finally { await browser.close(); }
})().catch(error => {console.error(error); process.exitCode = 1;});
