const assert = require("node:assert/strict");
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || "playwright");
(async () => {
  const browser = await chromium.launch({channel:"msedge", headless:true});
  try {
    const page = await browser.newPage();
    const errors = []; page.on("pageerror", error => errors.push(error.message));
    const records = []; let uploads = 0;
    const profile = {facts:{rows:120,sampledRows:120,validRows:120,duplicateRate:0,schema:"text"}, classification:{task:"writing",source:"local",confidence:.74},findings:[],_samples:[{text:"Support article example"}]};
    await page.route("**/api/datasets/upload", route => {uploads++; return route.fulfill({contentType:"application/json",body:JSON.stringify({source:"upload",name:`/tmp/${uploads}.jsonl`,displayName:`sample-${uploads}.jsonl`,extension:"jsonl"})});});
    let hfRequests = 0;
    await page.route("**/api/datasets/huggingface", route => {
      const request=route.request().postDataJSON(); hfRequests++;
      assert.equal(request.url,"https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data");
      if(hfRequests===1) return route.fulfill({contentType:"application/json",body:JSON.stringify({requiresConfiguration:true,configurations:["chat","math"]})});
      assert.equal(request.configuration,"chat"); assert.equal(request.max_rows,600);
      return route.fulfill({contentType:"application/json",body:JSON.stringify({source:"upload",name:"/tmp/hf.jsonl",displayName:"Nemotron sample",extension:"jsonl",sampleOnly:true,importedRows:600,hubConfig:"chat",hubRevision:"abc"})});
    });
    await page.route("**/api/datasets/github", route => {
      assert.equal(route.request().postDataJSON().url, "https://github.com/acme/data/blob/main/train.csv");
      return route.fulfill({contentType:"application/json",body:JSON.stringify({source:"upload",name:"/tmp/github.csv",displayName:"train.csv",extension:"csv",sourceUrl:"https://raw.githubusercontent.com/acme/data/main/train.csv"})});
    });
    await page.route("**/api/workflow/runs**", async route => {
      const request = route.request(); const pathname = new URL(request.url()).pathname;
      let result;
      if (pathname.endsWith("/resume")) {
        const answer = request.postDataJSON().response;
        assert.equal(answer.task,"assistant"); assert.equal(answer.notes,"Answer support questions");
        const record = records[0]; record.profile = {...profile,classification:{task:answer.task,source:"user-confirmed"}};
        record.pendingAction = {type:"dataset_ready", profile:record.profile}; result = record;
      } else if (pathname.endsWith("/runs") && request.method() === "POST") {
        const body = request.postDataJSON(); assert.equal(body.analysis_only,true); assert.equal(body.candidate_limit,1);
        if (records.length === 1) assert.equal(body.previous_workflow_id,records[0].id);
        result = {id:`version-${records.length+1}`,status:"waiting",createdAt:"2026-09-08",state:{dataset:body.dataset,previous_workflow_id:body.previous_workflow_id},dataset:body.dataset,profile,pendingAction:{type:"data_review",message:"What should the model learn?",profile}};
        records.push(result);
      } else if (pathname.endsWith("/runs")) result = records;
      else result = records.find(record => pathname.endsWith(record.id));
      await route.fulfill({contentType:"application/json",body:JSON.stringify(result)});
    });
    await page.goto("http://127.0.0.1:4173");
    await page.getByRole("heading",{name:"Upload a dataset",exact:true}).waitFor();
    const file = {name:"sample.jsonl",mimeType:"application/json",buffer:Buffer.from('{"text":"example"}')};
    await page.getByLabel("Dataset file",{exact:true}).setInputFiles(file);
    await page.getByRole("button",{name:"Upload & auto-analyze",exact:true}).click();
    await page.getByRole("heading",{name:"Clarify the intended task",exact:true}).waitFor();
    await page.getByLabel("What should the model do?",{exact:true}).selectOption("assistant");
    await page.getByLabel("Describe the expected result",{exact:true}).fill("Answer support questions");
    await page.waitForTimeout(2800);
    assert.equal(await page.getByLabel("Describe the expected result",{exact:true}).inputValue(),"Answer support questions");
    await page.getByRole("button",{name:"Confirm task & continue analysis",exact:true}).click();
    await page.getByRole("heading",{name:"Dataset ready",exact:true}).waitFor();
    await page.getByRole("button",{name:"Update dataset",exact:true}).click();
    await page.getByText("The previous file, answers, and analysis will remain available.",{exact:false}).waitFor();
    await page.getByRole("button",{name:"GitHub",exact:false}).click();
    await page.getByLabel("GitHub file URL",{exact:true}).fill("https://github.com/acme/data/blob/main/train.csv");
    await page.getByRole("button",{name:"Upload new version & analyze",exact:true}).click();
    await page.getByRole("heading",{name:"Clarify the intended task",exact:true}).waitFor();
    assert.equal(records.length,2); assert.equal(records[0].pendingAction.type,"dataset_ready");
    await page.setViewportSize({width:390,height:844});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),true);
    await page.goto("http://127.0.0.1:4173");
    await page.getByRole("button",{name:"GitHub",exact:true}).click();
    await page.getByLabel("GitHub file URL",{exact:true}).waitFor();
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),true);
    await page.getByRole("button",{name:"Hugging Face",exact:true}).click();
    await page.getByLabel("Import scope",{exact:true}).selectOption("sample");
    await page.getByLabel("Dataset URL or ID",{exact:true}).fill("https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data");
    await page.getByRole("button",{name:"Import Hugging Face sample & analyze",exact:true}).click();
    await page.getByLabel("Choose a dataset subset",{exact:true}).selectOption("chat");
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),true);
    await page.getByRole("button",{name:"Import Hugging Face sample & analyze",exact:true}).click();
    await page.getByRole("heading",{name:"Clarify the intended task",exact:true}).waitFor();
    assert.equal(records.length,3);
    assert.deepEqual(errors,[]);
    console.log("Dataset browser checks passed: upload, clarification, stable form, ready checkpoint, GitHub version update, mobile layout.");
  } finally { await browser.close(); }
})().catch(error => {console.error(error);process.exitCode=1;});
