const assert=require("node:assert/strict");
const {chromium}=require(process.env.PLAYWRIGHT_MODULE || "playwright");
(async()=>{const browser=await chromium.launch({channel:"msedge",headless:true});try{
const page=await browser.newPage();const errors=[];page.on("pageerror",e=>errors.push(e.message));
const benchmark={developmentIds:[4,8],reason:"Correctness needs grading",prices:{},results:["local-one","local-two"].map(modelId=>({modelId,provider:"local",status:"complete",meanLatencySeconds:.5,samples:[4,8].map(rowId=>({rowId,input:"Question",reference:"Reference",baseOutput:"Answer"}))}))};
const workflow={id:"benchmark-ui",status:"waiting",state:{model_benchmark:benchmark},pendingAction:{type:"model_selection",benchmark}};
let submitted;
await page.route("**/api/workflow/runs/benchmark-ui**",async route=>{
 if(route.request().url().includes("/compass")) return route.fulfill({json:{status:"not_started"}});
 if(route.request().method()==="POST") submitted=route.request().postDataJSON().response;
 await route.fulfill({json:workflow});
});
await page.goto("http://127.0.0.1:4173/#/workflow/benchmark-ui");
await page.getByRole("heading",{name:"Model selection benchmark",exact:true}).waitFor();
assert.equal(await page.locator("#compass-limit").inputValue(),"2");assert(await page.locator("#compass-limit").isDisabled());
for(const summary of await page.locator("#benchmark-review details > summary").all()) await summary.click();
for(const input of await page.locator("[data-grade-model]").all()) await input.selectOption("1");
await page.locator("#benchmark-hourly").fill("0.5");
await page.getByRole("button",{name:"Choose best from results",exact:true}).click();
await page.waitForTimeout(200);
assert.equal(submitted.action,"select_best");assert.equal(submitted.hourlyCostUsd,.5);assert.equal(Object.keys(submitted.grades).length,2);
assert.deepEqual(submitted.grades["local-one"],{"4":1,"8":1});
await page.setViewportSize({width:390,height:844});assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
assert.deepEqual(errors,[]);console.log("Model selection UI passed: identical split, rubric grades, rates, submission, mobile.");
}finally{await browser.close();}})().catch(e=>{console.error(e);process.exitCode=1;});
