/* Local quality workspace. Suggestions are never applied without a review decision. */
let qualityTimer;
const qualityEscape = value => escapeHtml(value ?? '');
async function qualityRequest(path, body) {
  const response = await fetch('/api/quality' + path, body === undefined ? {} : {method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});
  const result = await response.json();
  if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'The quality request could not be completed.');
  return result;
}
function qualityError(error) {
  const node = document.querySelector('#quality-error');
  if (node) node.textContent = error.message;
}
function qualityHeader(title, description) {
  clearTimeout(qualityTimer);
  app.innerHTML = shell('Traditional Chinese pilot', title, description, '<a class="button" href="#/workspace">Workspace</a>') + '<section class="card body-pad quality-panel"><p class="notice">Local processing only · No cloud clarification or hosted judging · Taiwan residency has not been verified.</p><nav class="quality-actions"><a class="button" href="#/quality">Dataset quality</a><a class="button" href="#/quality-evaluations">Evaluations</a><a class="button" href="#/quality-comparisons">Comparisons</a><a class="button" href="#/quality-panels">Judge panels</a></nav><p id="quality-error" role="alert"></p><div id="quality-content">Loading…</div></section></div>';
}
async function qualityPage(id) {
  qualityHeader(id ? 'Review changes' : 'Dataset quality', 'Import → audit → review corrections → create a version. Your original data stays unchanged.');
  try {
    if (id) return await qualityReview(id);
    const [audits, versions, sources] = await Promise.all([qualityRequest('/audits'),qualityRequest('/versions'),qualityRequest('/sources')]);
    if (route() !== 'quality') return;
    document.querySelector('#quality-content').innerHTML = `<form id="quality-audit-form"><h2>Audit a local dataset</h2><p>No training minimum and no GPU required. OpenCC suggests conversions; it does not prove a passage is wrong. Names need reviewer judgment.</p><label class="field" for="quality-source">Previously imported dataset</label><select id="quality-source"><option value="">Upload a file below</option>${sources.map((s,i)=>`<option value="${i}">${qualityEscape(s.label)}</option>`).join('')}</select><label class="field" for="quality-file">Dataset file</label><input id="quality-file" type="file" accept=".jsonl,.csv,.parquet"><label class="field" for="quality-mode">Scan scope</label><select id="quality-mode"><option value="full">Background full scan</option><option value="sample">Quick sample — at most 600 rows, not a full audit</option></select><label><input id="quality-raw" type="checkbox"> Allow corrections to raw text field (never labels)</label><label class="field" for="quality-terms">Customer terminology mappings (JSON object)</label><textarea id="quality-terms" rows="3" placeholder='{"鼠标":"滑鼠"}'>{}</textarea><label class="field" for="quality-exclusions">Protected names or phrases (one per line)</label><textarea id="quality-exclusions" rows="3"></textarea><p>Prompts, message roles, labels, metadata, code, URLs and quotations stay unchanged. Default editable scope: completion or final assistant answer.</p><button class="button primary" type="submit">Start audit</button></form><h2>Saved audits</h2>${audits.map(a=>`<p><a href="#/quality/${encodeURIComponent(a.id)}">${qualityEscape(a.dataset?.displayName || a.id.slice(0,8))}</a> · ${qualityEscape(a.status)} · ${qualityEscape(a.coverage)} · ${a.scannedRows || 0} rows scanned</p>`).join('') || '<p>No audits yet.</p>'}<h2>Reviewed dataset versions</h2>${versions.map(v=>`<p><a href="/api/quality/versions/${encodeURIComponent(v.id)}/export">${qualityEscape(v.dataset.displayName)}</a> · ${v.rows} rows · ${v.edits} edits · <a href="#/quality-comparisons/new/${encodeURIComponent(v.id)}">Compare original vs reviewed</a></p>`).join('') || '<p>No corrected versions yet.</p>'}`;
    document.querySelector('#quality-audit-form').onsubmit = async event => {
      event.preventDefault();
      const button = event.submitter; button.disabled = true;
      try {
        const selected = document.querySelector('#quality-source').value;
        let dataset = selected !== '' ? sources[Number(selected)].dataset : null;
        if (!dataset) {
          const file = document.querySelector('#quality-file').files[0];
          if (!file) throw new Error('Select an imported dataset or upload a file.');
          const form = new FormData(); form.append('file',file);
          const response = await fetch('/api/datasets/upload',{method:'POST',body:form});
          dataset = await response.json();
          if (!response.ok) throw new Error(dataset.detail || 'Upload failed.');
        }
        const audit = await qualityRequest('/audits',{dataset,mode:document.querySelector('#quality-mode').value,policy:{terms:JSON.parse(document.querySelector('#quality-terms').value),fields:document.querySelector('#quality-raw').checked ? ['text'] : [],exclusions:document.querySelector('#quality-exclusions').value.split('\n').map(x=>x.trim()).filter(Boolean)}});
        location.hash = '#/quality/' + audit.id;
      } catch(error) {qualityError(error); button.disabled = false;}
    };
  } catch(error) {qualityError(error);}
}
async function qualityReview(id, offset=0) {
  const audit = await qualityRequest('/audits/' + encodeURIComponent(id));
  if (route() !== 'quality/' + id) return;
  const content = document.querySelector('#quality-content');
  const running = ['queued','running','cancelling'].includes(audit.status);
  const reviewable = audit.status === 'complete' && audit.coverage === 'full';
  content.innerHTML = `<h2>${qualityEscape(audit.dataset?.displayName || 'Dataset audit')}</h2><p role="status">${qualityEscape(audit.status)} · ${audit.scannedRows} / ${audit.totalRows ?? '?'} rows · ${qualityEscape(audit.coverage)} coverage · ${audit.findingCount || 0} findings</p><p>${qualityEscape(audit.message)}</p><p>Policy: ${qualityEscape(audit.policy?.version)}. ${reviewable ? 'Full scan completed; this is not a correctness certification.' : 'Not a full audit pass. A completed full scan is required before corrections.'}</p>${running ? '<button class="button" id="quality-cancel">Cancel scan</button>' : ''}${!running && !reviewable ? '<button class="button" id="quality-full">Start a new full scan</button>' : ''}<div id="quality-findings"></div>${reviewable && Object.keys(audit.policy?.terms || {}).length ? `<p>This audit used a ${Object.keys(audit.policy.terms).length}-term customer list. Accepting it applies every pending suggestion from that list at once; OpenCC suggestions and findings you already accepted or ignored are unchanged.</p><button class="button" id="quality-accept-terms">Accept all term-list suggestions</button>` : ''}${reviewable ? '<label class="field" for="quality-new-exceptions">Save terminology exceptions for future audits (one per line)</label><textarea id="quality-new-exceptions" rows="2"></textarea><button class="button" id="quality-save-exceptions">Save exceptions</button><p>Only accepted findings are applied. Ignored and pending findings remain unchanged.</p><button class="button primary" id="quality-version">Create dataset version</button><div id="quality-version-result"></div>' : ''}`;
  document.querySelector('#quality-cancel')?.addEventListener('click',async()=>{try{await qualityRequest(`/audits/${id}/cancel`,{});await qualityReview(id,offset);}catch(e){qualityError(e);}});
  document.querySelector('#quality-full')?.addEventListener('click',async()=>{try{const next=await qualityRequest('/audits',{dataset:audit.dataset,policy:audit.policy,mode:'full'});location.hash='#/quality/'+next.id;}catch(e){qualityError(e);}});
  const page = await qualityRequest(`/audits/${id}/findings?offset=${offset}&limit=50`);
  if (route() !== 'quality/' + id) return;
  const findings = document.querySelector('#quality-findings');
  findings.innerHTML = `<h2>Findings</h2>${page.items.map((f,i)=>`<article class="quality-finding"><p>Row ${f.rowId + 1} · ${qualityEscape(f.field?.join('.'))} · offsets ${f.start}–${f.end} · ${qualityEscape(f.decision)}</p><div class="quality-diff"><div><strong>Original</strong><pre>${qualityEscape(f.original)}</pre></div><div><strong>Suggested replacement</strong><pre>${qualityEscape(f.replacement ?? 'No automatic edit')}</pre></div></div><p>${qualityEscape(f.reason)}</p><div class="quality-actions"><button class="button" data-quality-decision="accepted" data-index="${i}" ${!reviewable || !f.editable ? 'disabled' : ''}>Accept</button><button class="button" data-quality-decision="ignored" data-index="${i}" ${!reviewable ? 'disabled' : ''}>Ignore</button><button class="button" data-quality-decision="pending" data-index="${i}" ${!reviewable ? 'disabled' : ''}>Reset</button></div></article>`).join('') || '<p>No findings on this page.</p>'}<p>${offset + (page.items.length ? 1 : 0)}–${offset + page.items.length} of ${page.total} findings</p><button class="button" id="quality-previous" ${offset ? '' : 'disabled'}>Previous</button><button class="button" id="quality-next" ${offset + 50 < page.total ? '' : 'disabled'}>Next</button>`;
  findings.querySelectorAll('[data-quality-decision]').forEach(button=>button.onclick=async()=>{button.disabled=true;try{await qualityRequest(`/audits/${id}/review`,{fingerprint:audit.fingerprint,decisions:[{id:page.items[Number(button.dataset.index)].id,decision:button.dataset.qualityDecision}]});await qualityReview(id,offset);}catch(e){qualityError(e);button.disabled=false;}});
  document.querySelector('#quality-previous').onclick=()=>qualityReview(id,Math.max(0,offset-50)).catch(qualityError);
  document.querySelector('#quality-next').onclick=()=>qualityReview(id,offset+50).catch(qualityError);
  document.querySelector('#quality-accept-terms')?.addEventListener('click',async event=>{
    if(!confirm(`Accept every pending suggestion from the ${Object.keys(audit.policy.terms).length}-term list? Each one is saved as an accepted decision in the review history.`)) return;
    event.target.disabled=true;
    try{const result=await qualityRequest(`/audits/${id}/review`,{fingerprint:audit.fingerprint,acceptCustomerTerms:true});toast('Term list applied',result.message);await qualityReview(id,offset);}catch(e){qualityError(e);event.target.disabled=false;}
  });
  document.querySelector('#quality-save-exceptions')?.addEventListener('click',async()=>{try{const result=await qualityRequest(`/audits/${id}/review`,{fingerprint:audit.fingerprint,exclusions:document.querySelector('#quality-new-exceptions').value.split('\n').map(x=>x.trim()).filter(Boolean)});toast('Exceptions saved',result.message);}catch(e){qualityError(e);}});
  document.querySelector('#quality-version')?.addEventListener('click',async event=>{
    event.target.disabled=true;
    try {
      const version=await qualityRequest(`/audits/${id}/versions`,{fingerprint:audit.fingerprint});
      document.querySelector('#quality-version-result').innerHTML=`<p>New version saved: ${version.rows} rows, ${version.edits} approved edits. Original preserved.</p><div class="quality-actions"><a class="button" href="/api/quality/versions/${version.id}/export">Export dataset</a><a class="button" href="/api/quality/versions/${version.id}/export?kind=report">Export audit report</a><a class="button" href="#/quality-evaluations">Evaluate</a><a class="button" href="#/quality-comparisons/new/${encodeURIComponent(version.id)}">Compare original vs reviewed</a><button class="button" id="quality-train">Fine-tune</button></div><p>Fine-tune opens the existing analysis and approval workflow. It does not start training.</p>`;
      document.querySelector('#quality-train').onclick=async()=>{try{await analyzeImportedDataset(version.dataset,null,false);}catch(e){qualityError(e);}};
    } catch(e){qualityError(e);} finally{event.target.disabled=false;}
  });
  if(running) qualityTimer=setTimeout(()=>qualityReview(id,offset).catch(qualityError),1500);
}
async function qualityJSONL(file) {
  if(!file) throw new Error('Choose the required JSONL file.');
  if(file.size > 20 * 1024 * 1024) throw new Error('Evaluation uploads are limited to 20 MiB per file.');
  return (await file.text()).split(/\r?\n/).filter(x=>x.trim()).map((line,i)=>{try{return JSON.parse(line);}catch{throw new Error(`Invalid JSON on line ${i+1}.`);}});
}
async function qualityEvaluations(id) {
  qualityHeader('Evaluations','Evaluate saved answers independently of training. Freeze final cases before development; never clean the test set.');
  try {
    if(id) return await qualityEvaluationResult(id);
    const records=await qualityRequest('/evaluations');
    if(route() !== 'quality-evaluations') return;
    document.querySelector('#quality-content').innerHTML=`<form id="quality-evaluation-form"><h2>Evaluate saved outputs</h2><p>Cases: id, input, task (classification, json, writing or mcq), optional reference and subject.<br>Answers: candidate, caseId, output; optional human judgment (pass, fail or needs_review). Export existing ForgeTune outputs to this same format.</p><label class="field" for="quality-cases">Frozen evaluation cases (JSONL)</label><input id="quality-cases" type="file" accept=".jsonl" required><label class="field" for="quality-answers">Candidate answers (JSONL)</label><input id="quality-answers" type="file" accept=".jsonl" required><p>Exact task scores and language findings remain separate. Free-form correctness needs human review. Comparisons require matching cases, policies and declared scoring protocols.</p><button class="button primary" type="submit">Evaluate answers</button></form><h2>TAIDE and TMMLU+</h2><p id="quality-model-status">Checking local assets…</p><p>Public asset preparation is a separate, explicit step. No license terms are accepted here. TMMLU+ smoke results are not comparable to the official leaderboard.</p><h2>Saved evaluations</h2>${records.map(r=>`<p><a href="#/quality-evaluations/${r.id}">${qualityEscape(r.id.slice(0,8))}</a> · ${qualityEscape(r.status)}</p>`).join('') || '<p>No evaluations yet.</p>'}`;
    qualityRequest('/models').then(status=>{const node=document.querySelector('#quality-model-status');if(node)node.textContent=(status.modelId || 'TAIDE')+': '+status.status+' — '+status.message;}).catch(qualityError);
    qualityGenerationPanel().catch(qualityError);
    document.querySelector('#quality-evaluation-form').onsubmit=async event=>{
      event.preventDefault();event.submitter.disabled=true;
      try{const cases=await qualityJSONL(document.querySelector('#quality-cases').files[0]);const answers=await qualityJSONL(document.querySelector('#quality-answers').files[0]);const record=await qualityRequest('/evaluations',{cases,answers});location.hash='#/quality-evaluations/'+record.id;}catch(e){qualityError(e);event.submitter.disabled=false;}
    };
  }catch(e){qualityError(e);}
}
async function qualityGenerationPanel() {
  const [benchmarks, budget, hardware] = await Promise.all([qualityRequest('/benchmarks'),qualityRequest('/budget'),fetch('/api/hardware').then(r=>r.json()).catch(()=>({}))]);
  const small = !(hardware.memoryGb >= 24);
  if(route() !== 'quality-evaluations') return;
  const host=document.querySelector('#quality-content');
  host.insertAdjacentHTML('beforeend',`<h2>Local model evaluation</h2><p>TAIDE only · explicit generation · no downloads or fallback model. Shared pilot GPU budget: ${Math.round(budget.remainingGpuSeconds || 0)} seconds remaining. All model loading, failed attempts, training and inference count.</p><form id="quality-generation-form"><label class="field" for="quality-benchmark">Evaluation cases</label><select id="quality-benchmark"><option value="">Upload frozen case JSONL below</option>${benchmarks.filter(b=>b.status!=='invalid').map((b,i)=>`<option value="${i}">TMMLU+ smoke subset · ${b.allSubjects?'all subjects':'30 questions'} · ${qualityEscape(b.revision.slice(0,12))}</option>`).join('')}</select><input id="quality-generation-cases" type="file" accept=".jsonl"><label class="field" for="quality-adapter">Completed ForgeTune adapter run ID (optional; blank = base model)</label><input id="quality-adapter" type="text" placeholder="Server-issued run ID"><label class="field" for="quality-load-mode">Model precision</label><select id="quality-load-mode"><option value="bnb4" ${small?'selected':''}>4-bit (bnb4) · needs an idle GPU with 12 GB free</option><option value="fp16" ${small?'':'selected'}>FP16 · needs an idle GPU with 24 GB free</option></select><p>Detected GPU: ${qualityEscape(hardware.name || 'unknown')}${hardware.memoryGb ? ' · ' + hardware.memoryGb + ' GB' : ''}. 4-bit and FP16 results are not directly comparable.</p><label class="field" for="quality-sample-tokens">Writing answer length in tokens (optional)</label><input id="quality-sample-tokens" type="number" min="1" max="256" placeholder="Blank: full answers; any answer over 256 tokens stops the run"><p>Set a length (e.g. 128) to cut every writing answer to an equal-length sample, marked truncated. Useful for terminology-rate comparisons; multiple-choice answers are never cut.</p><label class="field" for="quality-seconds">Maximum GPU seconds for this evaluation</label><input id="quality-seconds" type="number" min="1" max="1800" value="300"><button class="button" type="submit">Generate locally & evaluate</button></form><h2>Use existing ForgeTune outputs</h2><form id="quality-saved-form"><label class="field" for="quality-workflow-id">Saved workflow ID</label><input id="quality-workflow-id" type="text" required><p>Legacy outputs keep a legacy label. Missing protocols are not invented to enable comparisons.</p><button class="button" type="submit">Evaluate saved workflow answers</button></form>`);
  document.querySelector('#quality-generation-form').onsubmit=async event=>{
    event.preventDefault();event.submitter.disabled=true;
    try {
      const selected=document.querySelector('#quality-benchmark').value;
      const b=selected===''?null:benchmarks.filter(x=>x.status!=='invalid')[Number(selected)];
      const body={generate:true,loadMode:document.querySelector('#quality-load-mode').value,maxGpuSeconds:Number(document.querySelector('#quality-seconds').value)};
      const sample=document.querySelector('#quality-sample-tokens').value;if(sample)body.writingSampleTokens=Number(sample);
      if(b){body.benchmarkRevision=b.revision;body.allSubjects=b.allSubjects;}
      else body.cases=await qualityJSONL(document.querySelector('#quality-generation-cases').files[0]);
      const adapter=document.querySelector('#quality-adapter').value.trim();if(adapter)body.adapterRunId=adapter;
      const record=await qualityRequest('/evaluations',body);location.hash='#/quality-evaluations/'+record.id;
    }catch(e){qualityError(e);event.submitter.disabled=false;}
  };
  document.querySelector('#quality-saved-form').onsubmit=async event=>{
    event.preventDefault();event.submitter.disabled=true;
    try{const id=document.querySelector('#quality-workflow-id').value.trim();const saved=await qualityRequest(`/workflows/${encodeURIComponent(id)}/answers`);const record=await qualityRequest('/evaluations',{cases:saved.cases,answers:saved.answers});location.hash='#/quality-evaluations/'+record.id;}catch(e){qualityError(e);event.submitter.disabled=false;}
  };
}
async function qualityEvaluationResult(id) {
  const record=await qualityRequest('/evaluations/'+encodeURIComponent(id));
  if(route() !== 'quality-evaluations/'+id) return;
  const result=record.result;
  const benchmarkNote=record.benchmark ? `<p class="notice">TMMLU+ smoke subset · ${qualityEscape(record.benchmark.version)} · immutable revision ${qualityEscape(record.benchmark.revision)}. MCQ accuracy is a per-question aggregate; subjects remain separate. This is not a full benchmark or an official leaderboard score.</p>` : '';
  document.querySelector('#quality-content').innerHTML=`<p role="status">${qualityEscape(record.status)} · ${qualityEscape(record.message)}</p>${['queued','running','cancelling'].includes(record.status)?'<button class="button" id="quality-eval-cancel">Cancel evaluation</button>':''}${result?Object.entries(result.candidates).map(([candidate,data])=>`<article class="quality-finding"><h2>${qualityEscape(candidate)}</h2><p>Coverage: ${data.coverage.answered}/${data.coverage.expected} · Missing: ${qualityEscape(data.coverage.missingCaseIds.join(', ') || 'none')}</p><h3>Separate task measures</h3><pre>${qualityEscape(JSON.stringify(data.metrics,null,2))}</pre><h3>Per-subject measures</h3><pre>${qualityEscape(JSON.stringify(data.subjects,null,2))}</pre><details><summary>Answers, language findings and review judgments</summary><pre>${qualityEscape(JSON.stringify(data.results,null,2))}</pre></details></article>`).join('')+`<h2>Paired differences</h2><pre>${qualityEscape(JSON.stringify(result.comparisons,null,2))}</pre><h3>Comparisons not supported</h3><pre>${qualityEscape(JSON.stringify(result.unpaired,null,2))}</pre>`:''}<a class="button" href="/api/quality/evaluations/${id}/export">Export evaluation report</a>${record.status==='complete'&&result?`<a class="button" href="#/quality-panels/new/evaluation/${encodeURIComponent(id)}">Grade with judge panel</a>`:''}`;
  document.querySelector('#quality-content').insertAdjacentHTML('afterbegin',benchmarkNote);
  document.querySelector('#quality-eval-cancel')?.addEventListener('click',async()=>{try{await qualityRequest(`/evaluations/${id}/cancel`,{});await qualityEvaluationResult(id);}catch(e){qualityError(e);}});
  if(['queued','running','cancelling'].includes(record.status)) qualityTimer=setTimeout(()=>qualityEvaluationResult(id).catch(qualityError),1500);
}
const qualityComparisonModels = [['base','Base'],['original','Original-trained'],['reviewed','Reviewed-trained']];
const qualityComparisonSteps = {prepare:'Prepare split and files','train-original':'Train on original data','train-reviewed':'Train on reviewed data','evaluate-base':'Evaluate base model','evaluate-original':'Evaluate original-trained','evaluate-reviewed':'Evaluate reviewed-trained',report:'Build report'};
const qualityActive = status => ['queued','running','cancelling'].includes(status);
const qualityNumber = (value, digits=1) => value == null || !Number.isFinite(Number(value)) ? '—' : Number(value).toFixed(digits);
const qualityDuration = seconds => { const s = Math.max(0, Math.round(Number(seconds) || 0)); return s >= 60 ? `${Math.floor(s/60)} min ${s%60} s` : `${s} s`; };
function qualityMark(text, terms) {
  const unique = [...new Set((terms || []).filter(Boolean))].sort((a,b)=>b.length-a.length);
  if (!unique.length) return qualityEscape(text);
  const pattern = new RegExp('(' + unique.map(t=>t.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')).join('|') + ')');
  return String(text ?? '').split(pattern).map((part,i)=>i % 2 ? `<mark>${qualityEscape(part)}</mark>` : qualityEscape(part)).join('');
}
async function qualityComparisons(id, versionId) {
  qualityHeader('Comparisons','Train one TAIDE QLoRA adapter on the original rows and one on the reviewed rows, then measure listed-term usage against the base model.');
  try {
    if (id === 'new') return await qualityComparisonStart(versionId);
    if (id) return await qualityComparisonView(id);
    const items = await qualityRequest('/comparisons');
    if (route() !== 'quality-comparisons') return;
    document.querySelector('#quality-content').innerHTML = `<h2>Saved comparisons</h2>${items.map(c=>`<article class="quality-finding"><p><a href="#/quality-comparisons/${encodeURIComponent(c.id)}"><strong>${qualityEscape(c.displayName || c.id.slice(0,8))}</strong></a> · <span class="pill ${c.status === 'complete' ? 'complete' : ''}">${qualityEscape(c.status)}</span> · ${qualityEscape(new Date(c.createdAt).toLocaleString())}</p>${c.headline ? `<p>${qualityEscape(c.headline)}</p>` : ''}</article>`).join('') || '<p>No comparisons yet.</p>'}<p>Start a comparison from a reviewed version in <a href="#/quality">Dataset quality</a> (“Compare original vs reviewed”).</p>`;
  } catch(e) {qualityError(e);}
}
async function qualityComparisonStart(versionId) {
  const versions = await qualityRequest('/versions');
  const here = 'quality-comparisons/new/' + versionId;
  if (route() !== here) return;
  const version = versions.find(v=>v.id === versionId);
  document.querySelector('#quality-content').innerHTML = `<h2>Compare original vs reviewed</h2><p>${version ? `${qualityEscape(version.dataset.displayName)} · ${version.rows} rows · ${version.edits} edits` : 'This reviewed version was not found in the saved versions list.'}</p><form id="quality-comparison-form" novalidate><div class="form-row"><label class="field">Training rows per run (blank = all available, capped at 500)<input id="qc-train-rows" type="number" min="100" max="5000" step="1" placeholder="All, up to 500"></label><label class="field">Share of training rows the review edited (0–1)<input id="qc-edited-share" type="number" min="0" max="1" step="0.05" value="0.5" required></label><label class="field">Epochs<select id="qc-epochs"><option>1</option><option selected>2</option><option>3</option></select></label><label class="field">Evaluation prompts (20–200)<input id="qc-eval-prompts" type="number" min="20" max="200" step="1" value="60" required></label></div><div id="quality-estimate" class="quality-estimate" role="status" aria-live="polite">Estimating…</div><button class="button primary" id="qc-start" type="submit" disabled>Start comparison</button></form>`;
  const form = document.querySelector('#quality-comparison-form'), start = document.querySelector('#qc-start'), panel = document.querySelector('#quality-estimate');
  let body = null, ticket = 0;
  const read = () => {
    const rows = document.querySelector('#qc-train-rows').value;
    const next = {versionId, editedShare:Number(document.querySelector('#qc-edited-share').value), epochs:Number(document.querySelector('#qc-epochs').value), evalPrompts:Number(document.querySelector('#qc-eval-prompts').value)};
    if (rows) next.trainRows = Number(rows);
    return next;
  };
  const show = (reasons, facts='') => {
    panel.innerHTML = facts + (reasons.length ? `<p><strong>This comparison can’t start yet:</strong></p><ul class="quality-reasons">${reasons.map(r=>`<li>${qualityEscape(r)}</li>`).join('')}</ul>` : '<p><strong>Ready to start.</strong> Both runs and all three evaluations fit the per-run cap and the GPU ledger.</p>');
  };
  const estimate = async () => {
    const mine = ++ticket;
    start.disabled = true; body = null;
    const invalid = [...form.querySelectorAll('input')].filter(input=>!input.checkValidity());
    if (invalid.length) return show(invalid.map(input=>`${input.closest('label').firstChild.textContent.trim()}: ${input.validationMessage}`));
    const request = read();
    panel.textContent = 'Estimating…';
    try {
      const result = await qualityRequest('/comparisons/estimate', request);
      if (mine !== ticket || route() !== here) return;
      const s = result.split || {}, e = result.estimate || {};
      const fact = (label, value) => `<div><dt>${label}</dt><dd>${value}</dd></div>`;
      show(e.reasons || s.reasons || [], `<dl class="quality-facts">${fact('Rows in version', s.rows ?? '—')}${fact('Rows the review edited', s.edited ?? '—')}${fact('Eligible eval prompts', s.eligibleEval ?? '—')}${fact('Edited rows in training', s.trainEdited ?? '—')}${fact('Rows skipped (too long)', s.skippedLong ?? '—')}${fact('GPU-seconds (total)', e.totalGpuSeconds ?? '—')}${fact('Wall-clock', e.wallClockSeconds == null ? '—' : `≈ ${Math.ceil(e.wallClockSeconds/60)} min`)}${fact('Ledger remaining', e.ledgerRemaining == null ? '—' : `${Math.floor(e.ledgerRemaining)} s`)}</dl><p>Per training run ≈ ${e.trainSecondsPerRun ?? '—'} s · per evaluation ≈ ${e.evalSecondsBase ?? '—'} s base, ${e.evalSecondsAdapter ?? '—'} s per adapter · per-run cap ${e.perRunCap ?? '—'} s.</p>`);
      if (result.feasible) { body = request; start.disabled = false; }
    } catch(error) { if (mine === ticket) show([error.message]); }
  };
  form.oninput = form.onchange = () => { body = null; start.disabled = true; ticket++; clearTimeout(qualityTimer); qualityTimer = setTimeout(estimate, 400); };
  form.onsubmit = async event => {
    event.preventDefault();
    if (!body || !confirm('Start the comparison? It trains two adapters and runs three evaluations on the GPU, charged to the shared ledger.')) return;
    start.disabled = true;
    try { const record = await qualityRequest('/comparisons', body); location.hash = '#/quality-comparisons/' + record.id; }
    catch(e) { qualityError(e); start.disabled = false; }
  };
  await estimate();
}
function qualityStepTracker(record) {
  return `<ol class="quality-steps">${(record.steps || []).map(step=>{
    const elapsed = step.startedAt ? ((step.completedAt ? Date.parse(step.completedAt) : Date.now()) - Date.parse(step.startedAt)) / 1000 : null;
    const meta = [elapsed == null ? 'not started' : `elapsed ${qualityDuration(elapsed)}`, `${Math.round(step.gpuSeconds || 0)} GPU-s`];
    if (step.attempts > 1) meta.push(`attempt ${step.attempts}`);
    return `<li class="${qualityEscape(step.status)}"><strong>${qualityEscape(qualityComparisonSteps[step.name] || step.name)}</strong><span class="pill ${step.status === 'complete' ? 'complete' : ''}">${qualityEscape(step.status)}</span><span class="quality-step-meta">${meta.join(' · ')}${step.runId ? ` · <a href="#/runs/${encodeURIComponent(step.runId)}">View run</a>` : ''}</span>${step.message ? `<span class="quality-step-meta quality-reasons">${qualityEscape(step.message)}</span>` : ''}</li>`;
  }).join('')}</ol>`;
}
function qualityComparisonResult(record) {
  const report = record.report, models = report.models || {}, split = record.split || {}, settings = record.settings || {};
  const present = qualityComparisonModels.filter(([key])=>models[key]);
  const cell = fn => present.map(([key])=>`<td>${fn(models[key])}</td>`).join('');
  const count = (n, total) => `${n ?? '—'} / ${total ?? '—'}`;
  const mcq = m => m.tmmluplusAccuracy == null ? '—' : `${Math.round(m.tmmluplusAccuracy * m.tmmluplusAnswered)} / ${m.tmmluplusAnswered} (${Math.round(m.tmmluplusAccuracy * 100)}%)`;
  const table = `<div class="quality-table-wrap"><table class="quality-table"><thead><tr><th scope="col">Measure</th>${present.map(([,label])=>`<th scope="col">${label}</th>`).join('')}</tr></thead><tbody><tr><th scope="row">Listed terms per 1,000 chars</th>${cell(m=>qualityNumber(m.termsPer1k))}</tr><tr><th scope="row">OpenCC suggestions per 1,000 chars</th>${cell(m=>qualityNumber(m.openccPer1k))}</tr><tr><th scope="row">Answers with a listed term</th>${cell(m=>count(m.answersWithTerm, m.writingAnswered))}</tr><tr><th scope="row">Answers with an OpenCC suggestion</th>${cell(m=>count(m.answersWithOpencc, m.writingAnswered))}</tr><tr><th scope="row">TMMLU+ correct</th>${cell(mcq)}</tr><tr><th scope="row">Mean training loss</th>${cell(m=>qualityNumber(m.trainLoss, 2))}</tr></tbody></table></div>`;
  const max = Math.max(0, ...present.map(([key])=>Number(models[key].termsPer1k) || 0));
  const bars = `<div class="quality-bars">${present.map(([key,label])=>{const v = models[key].termsPer1k, w = max && v != null ? v / max * 100 : 0; return `<div class="quality-bar" title="${label}: ${qualityNumber(v)} listed terms per 1,000 characters (${count(models[key].answersWithTerm, models[key].writingAnswered)} answers)"><span>${label}</span><span class="quality-bar-track" aria-hidden="true"><i style="width:${w.toFixed(1)}%"></i></span><strong>${qualityNumber(v)}</strong></div>`;}).join('')}</div>`;
  const p = report.paired || {}, pairs = (p.originalOnly || 0) + (p.reviewedOnly || 0) + (p.both || 0) + (p.neither || 0);
  const tile = (n, text) => `<div><dt>${text}</dt><dd>${n ?? 0} <small>of ${pairs} prompts</small></dd></div>`;
  const paired = `<dl class="quality-facts">${tile(p.originalOnly,'Original-trained used a listed term, reviewed-trained did not')}${tile(p.reviewedOnly,'Reviewed-trained used a listed term, original-trained did not')}${tile(p.both,'Both used a listed term')}${tile(p.neither,'Neither used a listed term')}</dl>`;
  const examples = (report.examples || []).slice(0,5).map(x=>`<article class="quality-example"><p class="quality-example-prompt"><strong>Prompt</strong> ${qualityEscape(x.prompt)}</p><div class="quality-answers">${qualityComparisonModels.filter(([key])=>x.answers?.[key] != null).map(([key,label])=>{const hits = x.hits?.[key] || []; return `<div><h3>${label}</h3><pre>${qualityMark(x.answers[key], hits)}</pre><p class="quality-step-meta">${hits.length ? `Listed terms: ${hits.map(qualityEscape).join(', ')}` : 'No listed terms'}</p></div>`;}).join('')}</div></article>`).join('');
  const trainSize = Array.isArray(split.train) ? split.train.length : split.settings?.trainRows;
  const evalSize = Array.isArray(split.eval) ? split.eval.length : settings.evalPrompts;
  const gpu = (record.steps || []).reduce((sum, s)=>sum + (Number(s.gpuSeconds) || 0), 0);
  const runs = report.runs || {};
  const measured = `<aside class="quality-measured"><h2>How this was measured</h2><ul><li>Edited share: ${split.counts?.trainEdited ?? '—'} of ${trainSize ?? '—'} training rows per run were edited by the review (requested ${Math.round((settings.editedShare ?? 0.5) * 100)}%). Train size ${trainSize ?? '—'} rows, ${settings.epochs ?? 2} epochs; both runs use the same rows and differ only in the review’s edits.</li><li>Eval prompts were chosen from rows the review changed, not at random.</li><li>Sample size: ${evalSize ?? '—'} writing prompts and ${models.base?.tmmluplusAnswered ?? '—'} TMMLU+ questions per model; split seed ${split.settings?.seed ?? 42}. Writing answers are 128-token samples.</li><li>Truncated samples: ${present.map(([key,label])=>`${label} ${models[key].truncated ?? 0} of ${models[key].writingAnswered ?? 0}`).join(' · ')}.</li><li>Model: TAIDE, 4-bit QLoRA, revision ${qualityEscape(record.modelRevision || 'not recorded')}; TMMLU+ revision ${qualityEscape(record.benchmarkRevision || 'not recorded')}.</li><li>GPU-seconds used: ${Math.round(gpu)} s across all steps.</li></ul></aside>`;
  return `<p class="quality-headline">${qualityEscape(report.headline)}</p><h2>Results by model</h2>${table}<h2>Listed terms per 1,000 characters</h2><p>Lower is better: fewer terms from the customer list appear in the answers.</p>${bars}<h2>Paired prompts</h2><p>The same prompt answered by both adapters.</p>${paired}${examples ? `<h2>Examples where the adapters differ</h2>${examples}` : ''}${measured}<div class="quality-actions"><a class="button primary" href="/api/quality/comparisons/${encodeURIComponent(record.id)}/export">Export comparison (JSON)</a><a class="button" href="#/quality-panels/new/comparison/${encodeURIComponent(record.id)}">Grade with judge panel</a>${['original','reviewed'].filter(v=>runs[v]).map(v=>`<a class="button" href="#/runs/${encodeURIComponent(runs[v])}">${v === 'original' ? 'Original' : 'Reviewed'}-trained run</a>`).join('')}</div><details><summary>Steps</summary>${qualityStepTracker(record)}</details>`;
}
async function qualityComparisonView(id) {
  const record = await qualityRequest('/comparisons/' + encodeURIComponent(id));
  if (route() !== 'quality-comparisons/' + id) return;
  const active = qualityActive(record.status);
  const done = record.status === 'complete' && record.report;
  document.querySelector('#quality-content').innerHTML = `<h2>${qualityEscape(record.displayName || 'Comparison')}</h2><p role="status"><span class="pill ${record.status === 'complete' ? 'complete' : ''}">${qualityEscape(record.status)}</span> ${qualityEscape(record.message || '')}</p>${done ? qualityComparisonResult(record) : `${qualityStepTracker(record)}${active && record.status !== 'cancelling' ? '<button class="button danger" id="quality-comparison-cancel">Cancel comparison</button>' : ''}${record.status === 'cancelling' ? '<p>Cancelling — the running step stops at its next checkpoint.</p>' : ''}`}`;
  document.querySelector('#quality-comparison-cancel')?.addEventListener('click', async event => {
    if (!confirm('Cancel this comparison? Finished steps are kept; the running step stops.')) return;
    event.target.disabled = true;
    try { await qualityRequest(`/comparisons/${encodeURIComponent(id)}/cancel`, {}); await qualityComparisonView(id); } catch(e) { qualityError(e); event.target.disabled = false; }
  });
  if (active) qualityTimer = setTimeout(()=>qualityComparisonView(id).catch(qualityError), 3000);
}
const qualityJudges = [['breeze','Breeze'],['qwen','Qwen'],['yi','Yi']];
const qualityPanelSteps = {prepare:'Prepare items','judge-breeze':'Breeze judge','judge-qwen':'Qwen judge','judge-yi':'Yi judge',report:'Build report'};
const qualityRate = rate => rate == null ? '—' : `${Math.round(rate * 100)}%`;
const qualityGraded = t => t.pass == null ? '—' : `${t.pass + t.fail} of ${t.pass + t.fail + t.needsReview} graded`;
const qualityPanelStartHash = source => source.evaluationId ? '#/quality-panels/new/evaluation/' + encodeURIComponent(source.evaluationId) : '#/quality-panels/new/comparison/' + encodeURIComponent(source.comparisonId);
const qualitySelfGrading = map => Object.entries(map || {}).map(([candidate, judge])=>`<li>${qualityEscape(candidate)} looks like the ${qualityEscape(judge)} judge’s own model family; the ${qualityEscape(judge)} vote on it may be biased.</li>`).join('');
async function qualityPanels(id, parts) {
  qualityHeader('Judge panels','Three local judge models grade saved writing answers for correctness and Traditional Chinese quality. Results are estimates, not verified.');
  try {
    if (id === 'new') return await qualityPanelStart(parts[2], parts[3]);
    if (id) return await qualityPanelView(id);
    const items = await qualityRequest('/panels');
    if (route() !== 'quality-panels') return;
    document.querySelector('#quality-content').innerHTML = `<h2>Saved judge panels</h2>${items.map(p=>`<article class="quality-finding"><p><a href="#/quality-panels/${encodeURIComponent(p.id)}"><strong>${qualityEscape(p.displayName || p.id.slice(0,8))}</strong></a> · <span class="pill ${p.status === 'complete' ? 'complete' : ''}">${qualityEscape(p.status)}</span> · ${qualityEscape(new Date(p.createdAt).toLocaleString())}</p></article>`).join('') || '<p>No judge panels yet.</p>'}<p>Start a panel with “Grade with judge panel” on a saved <a href="#/quality-evaluations">evaluation</a> or a complete <a href="#/quality-comparisons">comparison</a>.</p>`;
  } catch(e) {qualityError(e);}
}
async function qualityPanelStart(kind, sourceId) {
  if (!['evaluation','comparison'].includes(kind) || !sourceId) throw new Error('Choose a saved evaluation or comparison to grade.');
  const here = `quality-panels/new/${kind}/${sourceId}`, source = {[kind + 'Id']: sourceId};
  const name = kind === 'comparison' ? await qualityRequest('/comparisons/' + encodeURIComponent(sourceId)).then(r=>r.displayName, ()=>null) : null;
  if (route() !== here) return;
  document.querySelector('#quality-content').innerHTML = `<h2>Grade with judge panel</h2><p>${qualityEscape(name || `${kind === 'comparison' ? 'Comparison' : 'Evaluation'} ${sourceId.slice(0,8)}`)} · writing answers only</p><form id="quality-panel-form" novalidate><div class="form-row"><label class="field">Prompts to grade<input id="qp-prompts" type="number" min="1" step="1" placeholder="20"></label></div><div id="quality-estimate" class="quality-estimate" role="status" aria-live="polite">Estimating…</div><button class="button primary" id="qp-start" type="submit" disabled>Start judge panel</button></form>`;
  const form = document.querySelector('#quality-panel-form'), start = document.querySelector('#qp-start'), panel = document.querySelector('#quality-estimate'), input = document.querySelector('#qp-prompts');
  let body = null, ticket = 0;
  const show = (reasons, facts='') => {
    panel.innerHTML = facts + (reasons.length ? `<p><strong>This panel can’t start yet:</strong></p><ul class="quality-reasons">${reasons.map(r=>`<li>${qualityEscape(r)}</li>`).join('')}</ul>` : '<p><strong>Ready to start.</strong> All three judges are prepared and the panel fits the GPU ledger.</p>');
  };
  const estimate = async () => {
    const mine = ++ticket;
    start.disabled = true; body = null;
    if (!input.checkValidity()) return show([`Prompts to grade: ${input.validationMessage}`]);
    const request = {source};
    if (input.value) request.prompts = Number(input.value);
    panel.textContent = 'Estimating…';
    try {
      const result = await qualityRequest('/panels/estimate', request);
      if (mine !== ticket || route() !== here) return;
      const e = result.estimate || {};
      input.max = result.maxPrompts;
      if (!input.value) input.value = result.prompts;
      const fact = (label, value) => `<div><dt>${label}</dt><dd>${value}</dd></div>`;
      const judges = `<ul>${qualityJudges.map(([key,label])=>{const j = result.judges?.[key]; return `<li>${label} · ${qualityEscape(j?.modelId || '—')} · ${j?.status === 'ready' ? 'ready' : '<span class="quality-reasons">not prepared: on the backend, run <code>FORGETUNE_DATA_ROOT=$PWD/data HF_HOME=$PWD/data/huggingface .venv/bin/python trainer/quality_models.py prepare judges</code> from the repo root</span>'}</li>`;}).join('')}</ul>`;
      const notes = [];
      if (kind === 'comparison') notes.push('Comparison answers were cut at 128 tokens; judges grade only what is visible.');
      if (result.truncatedUnknown) notes.push('Uploaded answers are treated as complete.');
      if (result.referenceMode === 'none') notes.push('No reference shown to judges for comparisons.');
      const self = qualitySelfGrading(result.selfGrading);
      show(result.reasons || [], `<dl class="quality-facts">${fact('Answers to grade', result.items ?? '—')}${fact('Candidates', (result.candidates || []).length)}${fact('Prompts', `${result.prompts ?? '—'} of ${result.maxPrompts ?? '—'}`)}${fact('Grades per judge', e.gradesPerJudge ?? '—')}${fact('GPU-seconds needed', e.gpuSeconds ?? '—')}${fact('Ledger remaining', e.ledgerRemaining == null ? '—' : `${Math.floor(e.ledgerRemaining)} s`)}</dl><h3>Judges</h3>${judges}${notes.length || self ? `<ul>${notes.map(n=>`<li>${qualityEscape(n)}</li>`).join('')}${self}</ul>` : ''}`);
      if (result.feasible) { body = request; start.disabled = false; }
    } catch(error) { if (mine === ticket) show([error.message]); }
  };
  form.oninput = () => { body = null; start.disabled = true; ticket++; clearTimeout(qualityTimer); qualityTimer = setTimeout(estimate, 400); };
  form.onsubmit = async event => {
    event.preventDefault();
    if (!body || !confirm('Start the judge panel? It runs three local judge models on the GPU, charged to the shared ledger.')) return;
    start.disabled = true;
    try { const record = await qualityRequest('/panels', body); location.hash = '#/quality-panels/' + record.id; }
    catch(e) { qualityError(e); start.disabled = false; }
  };
  await estimate();
}
function qualityPanelTracker(record) {
  return `<ol class="quality-steps">${(record.steps || []).map(step=>{
    const meta = [step.name.startsWith('judge-') && step.total != null ? `${step.graded ?? 0} of ${step.total} graded` : null, step.gpuSeconds ? `${Math.round(step.gpuSeconds)} GPU-s` : null].filter(Boolean);
    return `<li class="${qualityEscape(step.status)}"><strong>${qualityEscape(qualityPanelSteps[step.name] || step.name)}</strong><span class="pill ${step.status === 'complete' ? 'complete' : ''}">${qualityEscape(step.status)}</span>${meta.length ? `<span class="quality-step-meta">${meta.join(' · ')}</span>` : ''}${step.message ? `<span class="quality-step-meta quality-reasons">${qualityEscape(step.message)}</span>` : ''}</li>`;
  }).join('')}</ol>`;
}
function qualityPanelVote(vote) {
  if (!vote || vote.error) return `<p class="quality-step-meta">No grade (${qualityEscape(vote?.error || 'missing')}).</p>`;
  return ['correctness','language'].map(key=>`<p><strong>${key === 'correctness' ? 'Correctness' : 'Language'}: ${qualityEscape(vote[key]?.verdict)}</strong><br>${qualityEscape(vote[key]?.reason)}</p>`).join('');
}
function qualityPanelResult(record) {
  const report = record.report, names = Object.keys(report.candidates || {});
  const table = `<div class="quality-table-wrap"><table class="quality-table"><thead><tr><th scope="col">Candidate</th><th scope="col">Correctness</th><th scope="col">Correctness coverage</th><th scope="col">Language</th><th scope="col">Language coverage</th></tr></thead><tbody>${names.map(n=>{const c = report.candidates[n]; return `<tr><th scope="row">${qualityEscape(n)}</th><td>${qualityRate(c.correctness.rate)}</td><td>${qualityGraded(c.correctness)}</td><td>${qualityRate(c.language.rate)}</td><td>${qualityGraded(c.language)}</td></tr>`;}).join('')}</tbody></table></div>`;
  const bars = (key, title) => `<h3>${title}</h3><div class="quality-bars quality-bars-coverage">${names.map(n=>{const t = report.candidates[n][key], v = t.rate; return `<div class="quality-bar" title="${qualityEscape(n)}: ${qualityRate(v)} ${title.toLowerCase()} (${qualityGraded(t)})"><span>${qualityEscape(n)}</span><span class="quality-bar-track" aria-hidden="true"><i style="width:${v == null ? 0 : (v * 100).toFixed(1)}%"></i></span><span class="quality-bar-value"><strong>${qualityRate(v)}</strong> <small>${qualityGraded(t)}</small></span></div>`;}).join('')}</div>`;
  const flagged = (report.flagged || []).map(f=>`<article class="quality-example"><p><strong>${qualityEscape(f.candidate)}</strong> · case ${qualityEscape(f.caseId)} · correctness ${qualityEscape(f.correctness)} · language ${qualityEscape(f.language)}</p><p class="quality-example-prompt"><strong>Prompt</strong> ${qualityEscape(f.question)}</p><pre>${qualityEscape(f.answer)}</pre>${f.truncated ? '<p class="quality-step-meta">Answer was cut off at a fixed length.</p>' : ''}<div class="quality-answers">${qualityJudges.map(([key,label])=>`<div><h3>${label}</h3>${qualityPanelVote(f.votes?.[key])}</div>`).join('')}</div></article>`).join('');
  const agreement = report.humanAgreement && report.humanAgreement.compared > 0 ? `<p>Human agreement: the panel matched ${report.humanAgreement.agreed} of ${report.humanAgreement.compared} human judgments (pass or fail only).</p>` : '';
  const self = qualitySelfGrading(report.selfGrading || record.selfGrading);
  const protocol = record.protocol || {};
  return `<p class="quality-headline">Estimated by local models, not verified</p>${report.incomplete ? '<p class="notice" role="alert">Incomplete: fewer than three judges finished. Correctness is not reported; language is reported only if Breeze finished.</p>' : ''}${self ? `<ul>${self}</ul>` : ''}<h2>Results by candidate</h2>${table}${bars('correctness','Correctness rate')}${bars('language','Language rate')}${agreement}${flagged ? `<h2>Flagged disagreements</h2><p>Items where the judges split on correctness or Breeze’s language verdict was not shared.</p>${flagged}` : ''}<aside class="quality-measured"><h2>How this was graded</h2><ul><li>${record.prompts ?? '—'} prompts, ${(report.items || []).length} answers, graded by Breeze, Qwen and Yi with model names hidden.</li><li>Reference: ${record.referenceMode === 'case' ? 'case reference shown to judges' : 'none shown to judges'}.</li>${record.truncatedItems > 0 ? `<li>${record.truncatedItems} answers were cut off at a fixed length; judges graded only the visible text.</li>` : ''}${record.truncatedUnknown ? '<li>Uploaded answers are treated as complete.</li>' : ''}<li>${qualityEscape(protocol.aggregationRule || 'Aggregation rule not recorded')}</li><li>Evaluator ${qualityEscape(record.evaluatorId || 'not recorded')}, ${qualityEscape(protocol.panelVersion || 'version not recorded')}.</li></ul></aside><div class="quality-actions"><a class="button primary" href="/api/quality/panels/${encodeURIComponent(record.id)}/export">Export panel (JSON)</a></div>`;
}
async function qualityPanelView(id) {
  const record = await qualityRequest('/panels/' + encodeURIComponent(id));
  if (route() !== 'quality-panels/' + id) return;
  const active = qualityActive(record.status);
  const retry = ['failed','cancelled'].includes(record.status) && record.source;
  document.querySelector('#quality-content').innerHTML = `<h2>${qualityEscape(record.displayName || 'Judge panel')}</h2><p role="status"><span class="pill ${record.status === 'complete' ? 'complete' : ''}">${qualityEscape(record.status)}</span> ${qualityEscape(record.message || '')}</p>${record.report ? qualityPanelResult(record) + `<details><summary>Steps</summary>${qualityPanelTracker(record)}</details>` : qualityPanelTracker(record)}${active && record.status !== 'cancelling' ? '<button class="button danger" id="quality-panel-cancel">Cancel panel</button>' : ''}${record.status === 'cancelling' ? '<p>Cancelling — the running step stops at its next checkpoint.</p>' : ''}${retry ? `<p>This panel can’t be resumed.</p><a class="button primary" href="${qualityPanelStartHash(record.source)}">Start a new panel</a>` : ''}`;
  document.querySelector('#quality-panel-cancel')?.addEventListener('click', async event => {
    if (!confirm('Cancel this judge panel? Finished grades are kept; the running step stops.')) return;
    event.target.disabled = true;
    try { await qualityRequest(`/panels/${encodeURIComponent(id)}/cancel`, {}); await qualityPanelView(id); } catch(e) { qualityError(e); event.target.disabled = false; }
  });
  if (active) qualityTimer = setTimeout(()=>qualityPanelView(id).catch(qualityError), 3000);
}
