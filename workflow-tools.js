const compassChoices = new Map();
let compassCatalog = [];
let compassTimer;

function modelBenchmark(workflow) {
  const benchmark=(workflow.state || workflow).model_benchmark;
  if(!benchmark) return workflow.status==="complete" ? `<section class="card body-pad workflow-tools"><h2>Benchmark model choices</h2><p>This saved run used the previous selection rules. Compare eligible models in a new workflow using this dataset and its existing split. Existing final results remain historical evidence.</p><button class="button" id="start-model-benchmark">Benchmark models on this dataset</button></section>` : "";
  const reviewing=workflow.pendingAction?.type==="model_selection";
  const results=benchmark.results || [];
  const needsGrades=results.filter(r=>r.status==="complete").some(r=>r.qualityEvidence?.coverage!==1 || !Number.isFinite(r.qualityEvidence?.score));
  if (benchmark.sampleWarning && !String(benchmark.reason).includes(benchmark.sampleWarning)) benchmark.reason = (benchmark.reason || "") + " " + benchmark.sampleWarning;
  return `<section class="card body-pad workflow-tools" id="model-benchmark"><h2>Model selection benchmark</h2><div class="notice"><strong>${benchmark.winner ? escapeHtml(benchmark.winner) : "More evidence needed"}</strong><p>${escapeHtml(benchmark.error || benchmark.reason || "")}</p></div><p>${benchmark.developmentIds?.length || 0} development questions · 256 output tokens per answer · Final test questions excluded</p><div class="benchmark-table"><table><thead><tr><th>Model</th><th>Correctness</th><th>USD / test set</th><th>Seconds / answer</th></tr></thead><tbody>${results.map(r=>`<tr><td>${escapeHtml(r.modelId)}<br><small>${escapeHtml(r.provider)} · ${escapeHtml(r.status)}</small></td><td>${r.qualityEvidence?.coverage===1 && Number.isFinite(r.qualityEvidence?.score) ? percentage(r.qualityEvidence.score) + (r.qualityEvidence.estimated ? " (estimated)" : "") : "Needs grading"}</td><td>${Number.isFinite(r.costUsd) ? "$"+r.costUsd.toFixed(6) : "Unknown"}</td><td>${Number.isFinite(r.meanLatencySeconds) ? r.meanLatencySeconds.toFixed(2) : "Unavailable"}</td></tr>`).join("")}</tbody></table></div>${reviewing ? `<form id="benchmark-review"><label class="field">VM/GPU cost (USD per hour)<input id="benchmark-hourly" type="number" min="0" step="any" placeholder="Unknown" value="${benchmark.prices?.hourlyCostUsd ?? ""}"></label>${results.filter(r=>r.provider==="compass").map((r,i)=>`<fieldset data-price-model="${escapeHtml(r.modelId)}"><legend>${escapeHtml(r.modelId)} · USD per million tokens</legend><label class="field">Input tokens<input data-price="input" type="number" min="0" step="any" value="${benchmark.prices?.tokenPrices?.[r.modelId]?.input ?? ""}"></label><label class="field">Output tokens<input data-price="output" type="number" min="0" step="any" value="${benchmark.prices?.tokenPrices?.[r.modelId]?.output ?? ""}"></label></fieldset>`).join("")}<label><input id="benchmark-unknown-cost" type="checkbox"> Continue using correctness and latency if prices are unknown</label>${needsGrades ? `<p>Automatic grading could not resolve every answer. Review the uncertain grades using a shared correctness rubric. A correct answer must satisfy the rubric fully; mark partial, invented, or incomplete answers incorrect.</p><label class="field">Correctness rubric<textarea id="benchmark-rubric" maxlength="2000" required>Answers the question correctly, follows its requirements, and contains no unsupported claims.</textarea></label>${results.filter(r=>r.status==="complete").map(r=>`<details class="experiment-example"><summary>Grade ${escapeHtml(r.modelId)} (${r.samples?.length || 0} answers)</summary>${(r.samples || []).map(sample=>`<div class="experiment-example"><strong>Question ${escapeHtml(sample.rowId)}</strong><pre class="code">${escapeHtml(sample.input || "")}</pre><strong>Reference — check it against your rubric</strong><pre class="code">${escapeHtml(sample.reference || "")}</pre><strong>Model answer</strong><pre class="code">${escapeHtml(sample.baseOutput || "")}</pre><label class="field">Correctness<select data-grade-model="${escapeHtml(r.modelId)}" data-grade-row="${sample.rowId}"><option value="">Not graded</option><option value="1">Correct</option><option value="0">Incorrect / incomplete</option></select></label></div>`).join("")}</details>`).join("")}` : ""}<p>Choose the highest correctness score; within 2 percentage points, prefer lower cost, then lower latency. This small benchmark gives a provisional choice.</p><button class="button primary" type="submit">Choose best from results</button></form>` : ""}</section>`;
}

function workflowTools(workflow) {
  const current = workflow.state || workflow;
  const settings = compassChoices.get(workflow.id) || {models:["gpt-4.1-mini",""], phase:workflow.status === "complete" ? "heldout" : "development", limit:3};
  if(workflow.pendingAction?.type==="model_selection") {settings.phase="development";settings.limit=current.model_benchmark?.developmentIds?.length || 3;}
  compassChoices.set(workflow.id, settings);
  const options = (selected, optional) => (optional ? '<option value="">No second model</option>' : "") + [...new Set([...compassCatalog, selected].filter(Boolean))].map(id=>`<option value="${escapeHtml(id)}" ${id===selected ? "selected" : ""}>${escapeHtml(id)}</option>`).join("");
  const runs = workflow.trainingRecords || [];
  const training = runs.length ? `<section class="card body-pad workflow-tools"><h2>Training progress</h2>${runs.map(run=>{
    const metrics=run.metrics || {}, context=run.trainingContext || metrics.trainingContext;
    const hp=run.config?.hyperparameters || {}, steps=context ? Math.ceil(context.rows/((hp.batch_size || 1)*(hp.gradient_accumulation || 1)))*(hp.epochs || 1) : 0;
    return `<div class="training-record"><h3>${escapeHtml(run.config?.base_model || "Training run")}</h3><p><span class="pill">${escapeHtml(run.status)}</span> ${escapeHtml(run.config?.hyperparameters?.epochs ?? "—")} epochs · ${escapeHtml(run.config?.hyperparameters?.max_sequence_length ?? "—")} token window</p>${context ? `<p>${Math.round(context.retainedTokenFraction*100)}% of training tokens retained · ${escapeHtml(context.truncatedRows)} conversations truncated</p>` : "<p>Preparing training data…</p>"}${metrics.optimizerSteps ? `<p>${escapeHtml(metrics.optimizerSteps)} optimizer steps · ${escapeHtml(metrics.trainRows)} training rows · ${escapeHtml(metrics.evalRows)} development rows</p><p>Development loss: ${escapeHtml(metrics.eval_loss)}. Loss measures prediction fit; it is not answer accuracy.</p>` : ""}${run.status==="running" ? `<p data-training-progress="${escapeHtml(run.id)}">Waiting for the latest training step…</p><details><summary>Live training log</summary><pre class="code" data-training-steps="${steps}" data-training-log="${escapeHtml(run.id)}">Loading progress…</pre></details>` : ""}</div>`;
  }).join("")}</section>` : "";
  const trials=current.prompt_trials || [];
  const ragSamples=trials.filter(trial=>trial.usesRag).flatMap(trial=>trial.results || []).flatMap(result=>result.samples || []);
  const coverage=ragSamples.length ? `<p><strong>Relevant sources found for ${ragSamples.filter(sample=>sample.retrieved?.length).length} of ${ragSamples.length} RAG cases.</strong></p>` : "";
  const history=workflow.pendingAction?.type!=="prompt_review" && trials.length ? `<section class="card body-pad workflow-tools"><h2>Prompt and RAG results</h2><p>Recorded development checks from before training. Final test examples are evaluated separately.</p>${trials.map((trial,index)=>`<details class="experiment-example"><summary>Test ${index+1} · ${trial.usesRag ? "Prompt + retrieved references" : trial.instruction ? "Improved instructions" : "Baseline"}</summary><pre class="code">${escapeHtml(trial.instruction || "No extra instructions")}</pre>${(trial.results || []).map(result=>`<h3>${escapeHtml(result.modelId || "Local model")}</h3><p>${escapeHtml(result.status)} · ${result.qualityEvidence?.estimated ? "estimated correctness" : "task correctness"} ${result.qualityEvidence?.score==null ? "needs review" : percentage(result.qualityEvidence.score)}</p>${gradingSummary(result)}${recordedSamples({...result, samples:(result.samples || []).slice(0, 5)})}`).join("")}</details>`).join("")}</section>` : "";
  const references = `<section class="card body-pad workflow-tools"><h2>Check external knowledge</h2><p>${current.knowledge_documents?.length ? `${current.knowledge_documents.length} reference documents connected. Each RAG trial retrieves relevant documents separately for each question.` : "Connect independent reference documents to test whether retrieval helps before fine-tuning."}</p>${coverage}${workflow.pendingAction?.type === "prompt_review" ? `<form id="workflow-rag-form"><label class="field" for="workflow-reference-files">Independent reference files</label><input id="workflow-reference-files" type="file" accept=".txt,.md" multiple required><p>Use source documents, not the evaluation answers. Up to 50 text or Markdown files; 20,000 characters per file.</p><button class="button" type="submit">Test with these references</button></form>` : ""}<p>Missing or irrelevant sources do not prove that RAG is unnecessary.</p></section>`;
  const privacy = current.privacy_review?.status === "review" ? '<div class="notice dataset-error"><strong>Privacy agent flagged sensitive values.</strong> Review the sample before sending it to a hosted model.</div><label class="field"><input id="compass-privacy-confirm" type="checkbox"> I confirm these examples may be sent to Compass.</label>' : "";
  const compass = `<section class="card body-pad workflow-tools" id="workflow-compass" data-workflow-id="${escapeHtml(workflow.id)}"><div class="section-head"><div><h2>Compare with Compass</h2><p>Compare one or two hosted text models on this dataset. Selected questions are sent to Compass; reference answers and your API key stay out of the browser request.</p></div><button class="button" id="compass-load" type="button">Load available models</button></div><form id="compass-form">${privacy}<div class="compass-fields"><label class="field">First model<select id="compass-first" required>${options(settings.models[0], false)}</select></label><label class="field">Second model (optional)<select id="compass-second">${options(settings.models[1], true)}</select></label><label class="field">Examples<select id="compass-phase"><option value="development" ${settings.phase==="development" ? "selected" : ""}>Development questions</option><option value="heldout" ${settings.phase==="heldout" ? "selected" : ""} ${workflow.status!=="complete" ? "disabled" : ""}>Final test questions</option></select></label><label class="field">Number of questions<input id="compass-limit" type="number" min="1" max="9" value="${settings.limit}" required></label></div><button class="button primary" type="submit">Run comparison</button><p>Up to 256 output tokens per answer. Retry resumes unanswered questions. A small sample cannot establish production accuracy.</p></form><div id="compass-results" aria-live="polite">Loading saved comparison…</div></section>`;
  const brief = current.task_description ? `<section class="card body-pad"><h2>Task and success criteria</h2><p>${escapeHtml(current.task_description)}</p><p>${escapeHtml(current.success_metric)}</p><p>Up to ${current.analysis_limit || 50} examples per phase, limited by available rows. Fewer than 30 examples is exploratory; larger counts alone do not establish representativeness.</p></section>` : "";
  const hosted = workflow.pendingAction?.type === "model_selection" && settings.limit > 9 ? "<section class='card body-pad'><p>Compass comparisons are limited to 9 questions and cannot join this larger local benchmark. Use a separate exploratory comparison after model selection.</p></section>" : compass;
  return brief + modelBenchmark(workflow) + training + references + history + hosted;
}

function renderCompassResults(job) {
  if (!job || job.status === "not_started") return "<p>No Compass comparison has run for this split.</p>";
  const total=(job.models?.length || 0)*(job.cases?.length || 0);
  return `<p><strong>${escapeHtml(job.status)}</strong> · ${job.results?.length || 0}/${total} answers saved</p>${job.error ? `<div class="notice">${escapeHtml(job.error)}. Run the same comparison to resume.</div>` : ""}${(job.cases || []).map(example=>`<details class="experiment-example"><summary>Question ${escapeHtml(example.rowId)}</summary><pre class="code">${escapeHtml(example.messages?.at(-1)?.content || "")}</pre><div class="compass-answers">${(job.models || []).map(model=>{
    const result=(job.results || []).find(r=>r.model===model && r.rowId===example.rowId);
    return `<div><h3>${escapeHtml(model)}</h3>${result ? `<pre class="code">${escapeHtml(result.output)}</pre>${result.finishReason==="length" ? '<p class="notice">Answer reached the token limit.</p>' : ""}` : "<p>No answer saved yet.</p>"}</div>`;
  }).join("")}</div></details>`).join("")}`;
}

function bindWorkflowTools(workflow) {
  clearTimeout(compassTimer);
  const settings=compassChoices.get(workflow.id);
  document.querySelector("#start-model-benchmark")?.addEventListener("click",async event=>{
    event.target.disabled=true;
    try {
      const state=workflow.state || workflow;
      const response=await fetch("/api/workflow/runs",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({dataset:state.dataset,previous_workflow_id:workflow.id,split_seed:state.split_seed ?? 42,test_dataset:state.test_dataset || null,auto_analysis:true,goal:state.goal || "balanced"})});
      const created=await response.json();
      if(!response.ok) throw new Error(typeof created.detail==="string" ? created.detail : "Could not start the model benchmark.");
      location.hash="#/workflow/"+created.id;
    }catch(error){toast("Benchmark unavailable",error.message);event.target.disabled=false;}
  });
  document.querySelector("#benchmark-review")?.addEventListener("submit",event=>{
    event.preventDefault();
    const grades={},tokenPrices={};
    for(const input of document.querySelectorAll("[data-grade-model]")) {
      if(!input.value) {toast("Grade every answer","Apply the same rubric to every completed model before choosing.");return;}
      (grades[input.dataset.gradeModel] ||= {})[input.dataset.gradeRow]=Number(input.value);
    }
    for(const field of document.querySelectorAll("[data-price-model]")) {
      const input=field.querySelector('[data-price="input"]').value,output=field.querySelector('[data-price="output"]').value;
      if(input || output) {
        if(!input || !output) {toast("Complete token prices","Provide both input and output rates, or leave both blank.");return;}
        tokenPrices[field.dataset.priceModel]={input:Number(input),output:Number(output)};
      }
    }
    const hourly=document.querySelector("#benchmark-hourly").value;
    workflowAction(workflow.id,{action:"select_best",grades,rubric:document.querySelector("#benchmark-rubric")?.value || "",hourlyCostUsd:hourly==="" ? null : Number(hourly),tokenPrices,allowUnknownCost:document.querySelector("#benchmark-unknown-cost").checked},"Selecting from measured development results.");
  });
  const panel=()=>document.querySelector('#workflow-compass[data-workflow-id="'+workflow.id+'"]');
  async function loadJob() {
    if(!panel()) return;
    const requestedPhase=settings.phase;
    try {
      const response=await fetch(`/api/workflow/runs/${encodeURIComponent(workflow.id)}/compass?phase=${settings.phase}`);
      const job=await response.json();
      if(!response.ok) throw new Error(job.detail || "Comparison unavailable");
      if(!panel() || settings.phase!==requestedPhase) return;
      document.querySelector("#compass-results").innerHTML=renderCompassResults(job);
      if(workflow.pendingAction?.type==="model_selection" && job?.status==="complete" && settings.phase==="development") {
        const button=document.createElement("button");button.className="button primary";button.textContent="Use these results in model selection";
        button.onclick=()=>workflowAction(workflow.id,{action:"add_compass",job},"Adding Compass answers to the same development benchmark.");
        document.querySelector("#compass-results").prepend(button);
      }
      if(job?.status==="running") compassTimer=setTimeout(loadJob,5000);
    } catch(error) {if(panel()) document.querySelector("#compass-results").textContent=error.message;}
  }
  if (panel()) {
  document.querySelector("#compass-load").onclick=async event=>{
    event.target.disabled=true;
    try {
      const response=await fetch("/api/compass/models");const data=await response.json();
      if(!response.ok) throw new Error(data.detail || "Could not list models.");
      compassCatalog=data.models || [];
      if(!panel()) return;
      for(const [index,id] of ["compass-first","compass-second"].entries()){
        const select=document.getElementById(id);
        select.innerHTML=(index ? '<option value="">No second model</option>' : "")+compassCatalog.map(model=>`<option value="${escapeHtml(model)}">${escapeHtml(model)}</option>`).join("");
        select.value=compassCatalog.includes(settings.models[index]) ? settings.models[index] : index ? "" : compassCatalog[0] || "";
        settings.models[index]=select.value;
      }
      toast("Compass connected",`${compassCatalog.length} model IDs returned. Choose text-generation models.`);
    }catch(error){toast("Compass unavailable",error.message);}finally{if(panel()) document.querySelector("#compass-load").disabled=false;}
  };
  ["compass-first","compass-second"].forEach((id,index)=>document.getElementById(id).onchange=event=>{settings.models[index]=event.target.value;});
  document.querySelector("#compass-phase").onchange=event=>{settings.phase=event.target.value;loadJob();};
  if(workflow.pendingAction?.type==="model_selection") {document.querySelector("#compass-phase").disabled=true;document.querySelector("#compass-limit").disabled=true;}
  document.querySelector("#compass-limit").onchange=event=>{settings.limit=Number(event.target.value);};
  document.querySelector("#compass-form").onsubmit=async event=>{
    event.preventDefault();const button=event.submitter;button.disabled=true;
    try{
      const privacyConfirm=document.querySelector("#compass-privacy-confirm"); if(privacyConfirm && !privacyConfirm.checked) throw new Error("Confirm the privacy warning before sending examples.");
      const response=await fetch(`/api/workflow/runs/${encodeURIComponent(workflow.id)}/compass?phase=${settings.phase}`,{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({models:settings.models.filter(Boolean),limit:settings.limit,allowSensitiveData:Boolean(privacyConfirm?.checked)})});
      const data=await response.json();if(!response.ok) throw new Error(data.detail || "Comparison could not start.");
      await loadJob();
    }catch(error){toast("Comparison unavailable",error.message);}finally{button.disabled=false;}
  };
  }
  document.querySelector("#workflow-rag-form")?.addEventListener("submit",async event=>{
    event.preventDefault();const button=event.submitter;button.disabled=true;
    try {
      const files=[...document.querySelector("#workflow-reference-files").files];
      if(!files.length || files.length>50 || files.some(file=>file.size>80000)) throw new Error("Choose 1–50 small text or Markdown files.");
      const documents=await Promise.all(files.map(async(file,index)=>({id:String(index+1),title:file.name,text:await file.text(),source:file.name})));
      if(documents.some(doc=>!doc.text.trim() || doc.text.length>20000) || documents.reduce((sum,doc)=>sum+doc.text.length,0)>200000) throw new Error("Keep each reference within 20,000 characters and the total within 200,000.");
      workflowAction(workflow.id,{action:"test_rag",documents},"Testing prompts with retrieved source passages.");
    } catch(error) {toast("Reference files need attention",error.message);button.disabled=false;}
  });
  for(const element of document.querySelectorAll("[data-training-log]")) fetch("/api/runs/"+encodeURIComponent(element.dataset.trainingLog)+"/logs").then(r=>r.json()).then(data=>{
    if(!element.isConnected) return;
    const lines=data.lines || [];element.textContent=lines.slice(-15).join("\n");
    const total=Number(element.dataset.trainingSteps);
    const line=[...lines].reverse().find(text=>new RegExp("\\b[0-9]+/"+total+"\\s*\\[").test(text));
    const step=line?.match(new RegExp("\\b([0-9]+)/"+total+"\\s*\\["))?.[1];
    const progress=document.querySelector('[data-training-progress="'+element.dataset.trainingLog+'"]');
    if(progress)progress.textContent=step ? `${step} / ${total} optimizer steps (${Math.round(Number(step)/total*100)}%)` : "Training is running; waiting for the first logged step.";
  }).catch(()=>{if(element.isConnected)element.textContent="Progress log temporarily unavailable.";});
  loadJob();
}
