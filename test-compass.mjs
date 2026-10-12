import test from "node:test";
import assert from "node:assert/strict";
import {runComparison, prepareMessages, handleCompass} from "./compass.mjs";

test("Compass omits empty system messages without changing question context", () => {
  assert.deepEqual(prepareMessages([{role:"system",content:""},{role:"user",content:"Question"}]),[{role:"user",content:"Question"}]);
  assert.throws(()=>prepareMessages([{role:"user",content:""}]),/input/i);
});

test("A resumed comparison never regenerates completed answers and uses its token cap", async () => {
  const job={models:["a","b"],cases:[{rowId:1,messages:[{role:"user",content:"Question"}]}],results:[{model:"a",rowId:1,output:"Already saved"}]};
  const calls=[];
  await runComparison(job,async()=>{},async(path,body)=>{calls.push(body);return {choices:[{message:{content:"New answer"},finish_reason:"stop"}],usage:{completion_tokens:2}};});
  assert.equal(calls.length,1);assert.equal(calls[0].model,"b");assert.equal(calls[0].max_tokens,256);
  assert.equal(job.status,"complete");assert.equal(job.results[0].output,"Already saved");
});

test("Rate limits persist a resumable state, never a fabricated answer", async () => {
  const job={models:["a"],cases:[{rowId:1,messages:[{role:"user",content:"Question"}]}],results:[]};
  await runComparison(job,async()=>{},async()=>{throw Object.assign(new Error("Rate limited"),{status:429});});
  assert.equal(job.status,"rate_limited");assert.equal(job.results.length,0);assert.equal(job.error,"Rate limited");
});


test("Website routes keep the key server-side and persist completed comparisons", async t => {
  const {createServer}=await import("node:http");
  const {randomUUID}=await import("node:crypto");
  const {unlink}=await import("node:fs/promises");
  const originalFetch=globalThis.fetch, id=randomUUID(), savedKey=process.env.ADVISOR_API_KEY, savedBase=process.env.ADVISOR_BASE_URL;
  process.env.ADVISOR_API_KEY="unit-test-credential";
  process.env.ADVISOR_BASE_URL="https://compass.llm.shopee.io/compass-api/v1";
  let generations=0;
  t.mock.method(globalThis,"fetch",async(url,options={})=>{
    if(String(url).endsWith("/models"))return Response.json({data:[{id:"test-chat-model"}]});
    if(String(url).includes("/comparison-inputs"))return Response.json({workflowId:id,phase:"development",splitManifestId:"fixed",cases:[{rowId:1,messages:[{role:"system",content:""},{role:"user",content:"Public question"}]}]});
    assert(String(url).endsWith("/chat/completions"));generations++;
    assert.equal(options.headers.Authorization,"Bearer unit-test-credential");
    const body=JSON.parse(options.body);assert.equal(body.max_tokens,256);assert.equal(body.messages.length,1);
    return Response.json({choices:[{message:{content:"Saved answer"},finish_reason:"stop"}]});
  });
  const server=createServer((req,res)=>handleCompass(req,res,new URL(req.url,"http://"+req.headers.host),"http://trainer.test"));
  await new Promise(resolve=>server.listen(0,"127.0.0.1",resolve));
  const url=`http://127.0.0.1:${server.address().port}/api/workflow/runs/${id}/compass`;
  const options={method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({models:["test-chat-model"],limit:1})};
  try {
    assert.equal((await originalFetch(url,options)).status,202);
    let job;
    for(let i=0;i<50;i++){job=await (await originalFetch(url)).json();if(job.status==="complete")break;await new Promise(r=>setTimeout(r,10));}
    assert.equal(job.status,"complete");assert.equal(job.results[0].output,"Saved answer");assert(!JSON.stringify(job).includes("unit-test-credential"));
    assert.equal((await originalFetch(url,options)).status,200);assert.equal(generations,1);
    assert.equal((await originalFetch(url,{...options,headers:{...options.headers,origin:"https://unrelated.test"}})).status,403);
  } finally {
    await new Promise(resolve=>server.close(resolve));
    if(savedKey===undefined)delete process.env.ADVISOR_API_KEY;else process.env.ADVISOR_API_KEY=savedKey;
    if(savedBase===undefined)delete process.env.ADVISOR_BASE_URL;else process.env.ADVISOR_BASE_URL=savedBase;
    await unlink(new URL(`../.forge-data/compass/${id}-development.json`,import.meta.url)).catch(error=>{if(error.code!=="ENOENT")throw error;});
  }
});
