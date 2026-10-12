const assert = require("node:assert/strict");
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || "playwright");
(async () => {
 const browser = await chromium.launch({channel:"msedge",headless:true});
 try {
  const page = await browser.newPage(); const errors=[]; page.on("pageerror",e=>errors.push(e.message));
  const profile={facts:{rows:600,sampledRows:600,validRows:600,schema:"chat"},classification:{task:"assistant",source:"local"},findings:[]};
  let record={id:"integrated",status:"waiting",dataset:{name:"sample.jsonl",sampleOnly:true,importedRows:600,hubConfig:"chat",hubSplit:"train"},profile,pendingAction:{type:"dataset_ready",profile}};
  const trial=instruction=>({instruction,results:[{modelId:"local-model",status:"complete",meanTokenOverlap:.6,samples:[{rowId:3,input:"Dataset question",baseOutput:instruction?"Improved actual output":"Base actual output"}]}]});
  const trials=[trial("")];
  await page.route("**/api/workflow/runs/integrated**",async route=>{
   if(route.request().method()==="POST"){
    const answer=route.request().postDataJSON().response;
    if(answer.action==="continue_training") record.pendingAction={type:"prompt_review",message:"Review dataset prompts",trials};
    else if(answer.action==="auto_analyze") {trials.push(trial("Automatic instructions"));record.pendingAction={type:"prompt_review",trials,recommendation:{title:"Try the improved prompt first",reason:"Measured improvement",rag:"Not tested: no knowledge source",selectedTrial:1}};}
    else if(answer.action==="try_prompt") {assert.equal(answer.instruction,"Be concise");trials.push(trial(answer.instruction));record.pendingAction={type:"prompt_review",message:"Review dataset prompts",trials};}
    else if(answer.action==="keep_baseline") {assert.equal(answer.trial_index,1);record.status="complete";record.pendingAction=null;record.comparison={decision:"keep_baseline",pairs:[],instruction:"Be concise",decisionReason:"Chosen after review"};}
   }
   await route.fulfill({contentType:"application/json",body:JSON.stringify(record)});
  });
  await page.goto("http://127.0.0.1:4173/#/workflow/integrated");
  await page.getByRole("button",{name:"Test base model & prompts",exact:true}).click();
  await page.getByRole("heading",{name:"Automatic model analysis",exact:true}).waitFor();
  assert.equal(await page.getByLabel("Improved instructions",{exact:true}).isVisible(),false);
  await page.getByRole("button",{name:"Run automatic analysis",exact:true}).click();
  await page.getByRole("heading",{name:"Try the improved prompt first",exact:true}).waitFor();
  assert.equal(await page.getByLabel("Selected test",{exact:true}).inputValue(),"1");
  await page.getByRole("button",{name:"Keep base model",exact:true}).click();
  await page.getByText("Base model retained",{exact:true}).waitFor();
  assert.equal(new URL(page.url()).hash,"#/workflow/integrated");
  await page.setViewportSize({width:390,height:844});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  assert.deepEqual(errors,[]);
  console.log("Integrated prompt browser checks passed: dataset samples, trials, selected version, retained baseline, same workflow.");
 } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
