import {readFile, writeFile, mkdir, rename} from "node:fs/promises";
import {fileURLToPath} from "node:url";
import {join} from "node:path";
import {createHash} from "node:crypto";
import {setTimeout as delay} from "node:timers/promises";

const storage = fileURLToPath(new URL("../.forge-data/compass/", import.meta.url));
const active = new Map();
const endpoint = "https://compass.llm.shopee.io/compass-api/v1";
const failure = (status, message) => Object.assign(new Error(message), {status});
const safeError = error => String(error.message || "Compass request failed").replaceAll(process.env.ADVISOR_API_KEY || "\0", "[redacted]").slice(0, 500);

export function prepareMessages(messages) {
  if (!Array.isArray(messages) || messages.length > 100 || messages.some(m => !["system","user","assistant"].includes(m.role) || typeof m.content !== "string")) throw failure(422, "Invalid conversation input.");
  const result = messages.filter(m => !(m.role === "system" && !m.content.trim()));
  if (!result.length || !result.at(-1).content.trim()) throw failure(422, "Conversation input is empty.");
  return result.map(({role,content})=>({role,content}));
}

async function requestCompass(path, body) {
  if (!process.env.ADVISOR_API_KEY || process.env.ADVISOR_BASE_URL?.replace(/\/$/, "") !== endpoint) throw failure(503, "Set ADVISOR_API_KEY and the Compass ADVISOR_BASE_URL in the project's root .env, then restart the website.");
  for (let attempt=0; attempt<3; attempt++) {
    const response = await fetch(endpoint + path, {method:body ? "POST" : "GET", headers:{Authorization:"Bearer " + process.env.ADVISOR_API_KEY, "Content-Type":"application/json"}, body:body ? JSON.stringify(body) : undefined, signal:AbortSignal.timeout(120000)});
    const data = await response.json().catch(()=>({message:"Compass returned an unreadable response."}));
    if (response.ok) return data;
    if (response.status===429 && attempt<2) {await delay(30000); continue;}
    throw failure(response.status, safeError(new Error(data.message || "Compass request failed")));
  }
}

export async function runComparison(job, save, request=requestCompass) {
  job.status="running"; delete job.error; await save(job);
  for (const model of job.models) for (const example of job.cases) {
    if (job.results.some(r=>r.model===model && r.rowId===example.rowId)) continue;
    try {
      const started=Date.now();
      const data=await request("/chat/completions",{model,messages:prepareMessages(example.messages),temperature:0,max_tokens:256});
      const choice=data.choices?.[0];
      if (typeof choice?.message?.content !== "string" || !choice.message.content.trim()) throw failure(502,"Compass returned no answer text.");
      job.results.push({model,rowId:example.rowId,output:choice.message.content,finishReason:choice.finish_reason,usage:data.usage,seconds:(Date.now()-started)/1000});
      await save(job);
    } catch (error) {
      job.status=error.status===429 ? "rate_limited" : "failed";
      job.error=safeError(error); await save(job); return;
    }
  }
  job.status="complete"; job.completedAt=new Date().toISOString(); await save(job);
}

async function readJob(path) {
  try {return JSON.parse(await readFile(path,"utf8"));} catch(error) {if(error.code==="ENOENT") return null; throw error;}
}

export async function handleCompass(request, response, url, trainerUrl) {
  const match=url.pathname.match(/^\/api\/workflow\/runs\/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\/compass$/i);
  if (!match && url.pathname!=="/api/compass/models") return false;
  const reply=(status,data)=>{response.writeHead(status,{"content-type":"application/json","cache-control":"no-store"});response.end(JSON.stringify(data));};
  try {
    if (!match) {
      if(request.method!=="GET") throw failure(405,"Use GET to list models.");
      const data=await requestCompass("/models");
      reply(200,{models:(data.data || []).map(m=>m.id).filter(id=>typeof id==="string"),configured:true});return true;
    }
    let phase=url.searchParams.get("phase") || "development";
    if (!["development","heldout"].includes(phase)) throw failure(422,"Choose development or heldout examples.");
    const path=join(storage, match[1]+"-"+phase+".json");
    const save=async job=>{await mkdir(storage,{recursive:true});await writeFile(path+".tmp",JSON.stringify(job,null,2));await rename(path+".tmp",path);};
    if(request.method==="GET") {
      const job=await readJob(path);
      if(job?.status==="running" && !active.has(path)) job.status="interrupted";
      reply(200,job || {status:"not_started",results:[]});return true;
    }
    if(request.method!=="POST") throw failure(405,"Use GET or POST.");
    if(request.headers.origin && request.headers.origin!=="http://"+request.headers.host) throw failure(403,"Start this comparison from ForgeTune.");
    if(!request.headers["content-type"]?.startsWith("application/json")) throw failure(415,"Send JSON.");
    if(active.has(path)) {reply(202,await readJob(path));return true;}
    // Reserve the job before awaiting input so simultaneous clicks cannot double-charge.
    active.set(path,true);
    let launched=false;
    try {
      let text="";
      for await(const chunk of request) {text+=chunk; if(text.length>8192) throw failure(413,"Comparison request is too large.");}
      let body;try{body=JSON.parse(text);}catch{throw failure(422,"Invalid JSON.");}
      const models=body.models;
      if(!Array.isArray(models) || models.length<1 || models.length>2 || new Set(models).size!==models.length || models.some(m=>typeof m!=="string" || m.length>200)) throw failure(422,"Select one or two different Compass models.");
      const limit=body.limit ?? 3;
      if(!Number.isInteger(limit) || limit<1 || limit>9) throw failure(422,"Compare between 1 and 9 examples.");
      const listed=await requestCompass("/models");
      if(models.some(id=>!listed.data?.some(m=>m.id===id))) throw failure(422,"Choose model IDs returned by Compass.");
      const allowSensitiveData=body.allowSensitiveData===true;
      const source=await fetch(trainerUrl+"/workflow/runs/"+match[1]+"/comparison-inputs?phase="+phase+"&limit="+limit+"&allow_sensitive_data="+(allowSensitiveData ? "true" : "false"),{signal:AbortSignal.timeout(30000)});
      const inputs=await source.json();
      if(!source.ok) throw failure(source.status,typeof inputs.detail==="string" ? inputs.detail : "Comparison inputs unavailable.");
      if(!inputs.cases?.length) throw failure(422,"No examples are available in this split.");
      const fingerprint=createHash("sha256").update(JSON.stringify({inputs,models})).digest("hex");
      const previous=await readJob(path);
      const job=previous?.fingerprint===fingerprint ? previous : {...inputs,models,fingerprint,results:[],createdAt:new Date().toISOString()};
      if(job.status==="complete") {reply(200,job);return true;}
      job.status="running";await save(job);
      launched=true;
      runComparison(job,save).catch(async error=>{job.status="failed";job.error=safeError(error);await save(job);}).finally(()=>active.delete(path));
      reply(202,job);
    } finally {if(!launched) active.delete(path);}
  } catch(error) {reply(error.status || 502,{detail:safeError(error)});}
  return true;
}
