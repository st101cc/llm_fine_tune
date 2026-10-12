const assert=require("node:assert/strict");
const {chromium}=require(process.env.PLAYWRIGHT_MODULE || "playwright");
(async()=>{
 const browser=await chromium.launch({channel:"msedge",headless:true});
 try{
  const page=await browser.newPage();const errors=[];page.on("pageerror",e=>errors.push(e.message));
  const id="00000000-0000-0000-0000-000000000002";
  const record={id,status:"waiting",state:{split_manifest_path:"split.json"},pendingAction:{type:"prompt_review",trials:[],recommendation:null},trainingRecords:[{id:"run1",status:"running",config:{base_model:"Qwen/Qwen2.5-3B-Instruct"},trainingContext:{rows:170,truncatedRows:0,retainedTokenFraction:1}}]};
  let submitted=false;let rag=false;
  await page.route("**/api/**",async route=>{
   const url=new URL(route.request().url());let data={};
   if(url.pathname.endsWith("/compass/models"))data={models:["gpt-4.1-mini","Qwen2.5-32B-Instruct"]};
   else if(url.pathname.endsWith("/compass")){
    if(route.request().method()==="POST"){const body=route.request().postDataJSON();assert.deepEqual(body.models,["gpt-4.1-mini"]);submitted=true;}
    data=submitted?{status:"complete",models:["gpt-4.1-mini"],phase:"development",cases:[{rowId:1,messages:[{role:"user",content:"Question"}]}],results:[{model:"gpt-4.1-mini",rowId:1,output:"Actual answer <script>unsafe</script>",finishReason:"stop"}]}:{status:"not_started",results:[]};
   }else if(url.pathname.endsWith("/resume")){const body=route.request().postDataJSON().response;assert.equal(body.action,"test_rag");assert.equal(body.documents[0].text,"Independent friction reference");rag=true;data=record;}
   else if(url.pathname.endsWith("/logs"))data={lines:["Training progress"]};
   else if(url.pathname.includes("/workflow/runs/"))data=record;
   else if(url.pathname.endsWith("/hardware"))data={trainerOnline:true,memoryGb:15,name:"Tesla T4"};
   else data=[];
   await route.fulfill({contentType:"application/json",body:JSON.stringify(data)});
  });
  await page.goto("http://127.0.0.1:4173/#/workflow/"+id);
  await page.getByRole("heading",{name:"Compare with Compass",exact:true}).waitFor({timeout:3000});
  await page.getByRole("button",{name:"Load available models",exact:true}).click();
  await page.getByRole("button",{name:"Run comparison",exact:true}).click();
  await page.locator("#compass-results summary").first().click();
  await page.getByText("Actual answer <script>unsafe</script>",{exact:true}).waitFor();
  assert(submitted);assert.equal(await page.locator("#compass-results script").count(),0);
  assert(await page.getByText("100% of training tokens retained",{exact:false}).isVisible());
  await page.getByLabel("Independent reference files").setInputFiles({name:"reference.txt",mimeType:"text/plain",buffer:Buffer.from("Independent friction reference")});
  await page.getByRole("button",{name:"Test with these references",exact:true}).click();
  await page.waitForTimeout(100);assert(rag);
  await page.setViewportSize({width:390,height:844});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  assert.deepEqual(errors,[]);console.log("Integrated website controls passed: Compass, RAG, progress, escaping, mobile.");
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
