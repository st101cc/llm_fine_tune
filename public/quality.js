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
  app.innerHTML = shell('Traditional Chinese pilot', title, description, '<a class="button" href="#/workspace">Workspace</a>') + '<section class="card body-pad quality-panel"><p class="notice">Local processing only · No cloud clarification or hosted judging · Taiwan residency has not been verified.</p><nav class="quality-actions"><a class="button" href="#/quality">Dataset quality</a><a class="button" href="#/quality-evaluations">Evaluations</a></nav><p id="quality-error" role="alert"></p><div id="quality-content">Loading…</div></section></div>';
}
async function qualityPage(id) {
  qualityHeader(id ? 'Review changes' : 'Dataset quality', 'Import → audit → review corrections → create a version. Your original data stays unchanged.');
  try {
    if (id) return await qualityReview(id);
    const [audits, versions, sources] = await Promise.all([qualityRequest('/audits'),qualityRequest('/versions'),qualityRequest('/sources')]);
    if (route() !== 'quality') return;
    document.querySelector('#quality-content').innerHTML = `<form id="quality-audit-form"><h2>Audit a local dataset</h2><p>No training minimum and no GPU required. OpenCC suggests conversions; it does not prove a passage is wrong. Names need reviewer judgment.</p><label class="field" for="quality-source">Previously imported dataset</label><select id="quality-source"><option value="">Upload a file below</option>${sources.map((s,i)=>`<option value="${i}">${qualityEscape(s.label)}</option>`).join('')}</select><label class="field" for="quality-file">Dataset file</label><input id="quality-file" type="file" accept=".jsonl,.csv,.parquet"><label class="field" for="quality-mode">Scan scope</label><select id="quality-mode"><option value="full">Background full scan</option><option value="sample">Quick sample — at most 600 rows, not a full audit</option></select><label><input id="quality-raw" type="checkbox"> Allow corrections to raw text field (never labels)</label><label class="field" for="quality-terms">Customer terminology mappings (JSON object)</label><textarea id="quality-terms" rows="3" placeholder='{"鼠标":"滑鼠"}'>{}</textarea><label class="field" for="quality-exclusions">Protected names or phrases (one per line)</label><textarea id="quality-exclusions" rows="3"></textarea><p>Prompts, message roles, labels, metadata, code, URLs and quotations stay unchanged. Default editable scope: completion or final assistant answer.</p><button class="button primary" type="submit">Start audit</button></form><h2>Saved audits</h2>${audits.map(a=>`<p><a href="#/quality/${encodeURIComponent(a.id)}">${qualityEscape(a.dataset?.displayName || a.id.slice(0,8))}</a> · ${qualityEscape(a.status)} · ${qualityEscape(a.coverage)} · ${a.scannedRows || 0} rows scanned</p>`).join('') || '<p>No audits yet.</p>'}<h2>Reviewed dataset versions</h2>${versions.map(v=>`<p><a href="/api/quality/versions/${encodeURIComponent(v.id)}/export">${qualityEscape(v.dataset.displayName)}</a> · ${v.rows} rows · ${v.edits} edits</p>`).join('') || '<p>No corrected versions yet.</p>'}`;
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
  content.innerHTML = `<h2>${qualityEscape(audit.dataset?.displayName || 'Dataset audit')}</h2><p role="status">${qualityEscape(audit.status)} · ${audit.scannedRows} / ${audit.totalRows ?? '?'} rows · ${qualityEscape(audit.coverage)} coverage · ${audit.findingCount || 0} findings</p><p>${qualityEscape(audit.message)}</p><p>Policy: ${qualityEscape(audit.policy?.version)}. ${reviewable ? 'Full scan completed; this is not a correctness certification.' : 'Not a full audit pass. A completed full scan is required before corrections.'}</p>${running ? '<button class="button" id="quality-cancel">Cancel scan</button>' : ''}${!running && !reviewable ? '<button class="button" id="quality-full">Start a new full scan</button>' : ''}<div id="quality-findings"></div>${reviewable ? '<label class="field" for="quality-new-exceptions">Save terminology exceptions for future audits (one per line)</label><textarea id="quality-new-exceptions" rows="2"></textarea><button class="button" id="quality-save-exceptions">Save exceptions</button><p>Only accepted findings are applied. Ignored and pending findings remain unchanged.</p><button class="button primary" id="quality-version">Create dataset version</button><div id="quality-version-result"></div>' : ''}`;
  document.querySelector('#quality-cancel')?.addEventListener('click',async()=>{try{await qualityRequest(`/audits/${id}/cancel`,{});await qualityReview(id,offset);}catch(e){qualityError(e);}});
  document.querySelector('#quality-full')?.addEventListener('click',async()=>{try{const next=await qualityRequest('/audits',{dataset:audit.dataset,policy:audit.policy,mode:'full'});location.hash='#/quality/'+next.id;}catch(e){qualityError(e);}});
  const page = await qualityRequest(`/audits/${id}/findings?offset=${offset}&limit=50`);
  if (route() !== 'quality/' + id) return;
  const findings = document.querySelector('#quality-findings');
  findings.innerHTML = `<h2>Findings</h2>${page.items.map((f,i)=>`<article class="quality-finding"><p>Row ${f.rowId + 1} · ${qualityEscape(f.field?.join('.'))} · offsets ${f.start}–${f.end} · ${qualityEscape(f.decision)}</p><div class="quality-diff"><div><strong>Original</strong><pre>${qualityEscape(f.original)}</pre></div><div><strong>Suggested replacement</strong><pre>${qualityEscape(f.replacement ?? 'No automatic edit')}</pre></div></div><p>${qualityEscape(f.reason)}</p><div class="quality-actions"><button class="button" data-quality-decision="accepted" data-index="${i}" ${!reviewable || !f.editable ? 'disabled' : ''}>Accept</button><button class="button" data-quality-decision="ignored" data-index="${i}" ${!reviewable ? 'disabled' : ''}>Ignore</button><button class="button" data-quality-decision="pending" data-index="${i}" ${!reviewable ? 'disabled' : ''}>Reset</button></div></article>`).join('') || '<p>No findings on this page.</p>'}<p>${offset + (page.items.length ? 1 : 0)}–${offset + page.items.length} of ${page.total} findings</p><button class="button" id="quality-previous" ${offset ? '' : 'disabled'}>Previous</button><button class="button" id="quality-next" ${offset + 50 < page.total ? '' : 'disabled'}>Next</button>`;
  findings.querySelectorAll('[data-quality-decision]').forEach(button=>button.onclick=async()=>{button.disabled=true;try{await qualityRequest(`/audits/${id}/review`,{fingerprint:audit.fingerprint,decisions:[{id:page.items[Number(button.dataset.index)].id,decision:button.dataset.qualityDecision}]});await qualityReview(id,offset);}catch(e){qualityError(e);button.disabled=false;}});
  document.querySelector('#quality-previous').onclick=()=>qualityReview(id,Math.max(0,offset-50)).catch(qualityError);
  document.querySelector('#quality-next').onclick=()=>qualityReview(id,offset+50).catch(qualityError);
  document.querySelector('#quality-save-exceptions')?.addEventListener('click',async()=>{try{const result=await qualityRequest(`/audits/${id}/review`,{fingerprint:audit.fingerprint,exclusions:document.querySelector('#quality-new-exceptions').value.split('\n').map(x=>x.trim()).filter(Boolean)});toast('Exceptions saved',result.message);}catch(e){qualityError(e);}});
  document.querySelector('#quality-version')?.addEventListener('click',async event=>{
    event.target.disabled=true;
    try {
      const version=await qualityRequest(`/audits/${id}/versions`,{fingerprint:audit.fingerprint});
      document.querySelector('#quality-version-result').innerHTML=`<p>New version saved: ${version.rows} rows, ${version.edits} approved edits. Original preserved.</p><div class="quality-actions"><a class="button" href="/api/quality/versions/${version.id}/export">Export dataset</a><a class="button" href="/api/quality/versions/${version.id}/export?kind=report">Export audit report</a><a class="button" href="#/quality-evaluations">Evaluate</a><button class="button" id="quality-train">Fine-tune</button></div><p>Fine-tune opens the existing analysis and approval workflow. It does not start training.</p>`;
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
  const [benchmarks, budget] = await Promise.all([qualityRequest('/benchmarks'),qualityRequest('/budget')]);
  if(route() !== 'quality-evaluations') return;
  const host=document.querySelector('#quality-content');
  host.insertAdjacentHTML('beforeend',`<h2>Local model evaluation</h2><p>TAIDE only · explicit generation · no downloads or fallback model. Shared pilot GPU budget: ${Math.round(budget.remainingGpuSeconds || 0)} seconds remaining. All model loading, failed attempts, training and inference count.</p><form id="quality-generation-form"><label class="field" for="quality-benchmark">Evaluation cases</label><select id="quality-benchmark"><option value="">Upload frozen case JSONL below</option>${benchmarks.filter(b=>b.status!=='invalid').map((b,i)=>`<option value="${i}">TMMLU+ smoke subset · ${b.allSubjects?'all subjects':'30 questions'} · ${qualityEscape(b.revision.slice(0,12))}</option>`).join('')}</select><input id="quality-generation-cases" type="file" accept=".jsonl"><label class="field" for="quality-adapter">Completed ForgeTune adapter run ID (optional; blank = base model)</label><input id="quality-adapter" type="text" placeholder="Server-issued run ID"><label class="field" for="quality-seconds">Maximum GPU seconds for this evaluation</label><input id="quality-seconds" type="number" min="1" max="1800" value="300"><button class="button" type="submit">Generate locally & evaluate</button></form><h2>Use existing ForgeTune outputs</h2><form id="quality-saved-form"><label class="field" for="quality-workflow-id">Saved workflow ID</label><input id="quality-workflow-id" type="text" required><p>Legacy outputs keep a legacy label. Missing protocols are not invented to enable comparisons.</p><button class="button" type="submit">Evaluate saved workflow answers</button></form>`);
  document.querySelector('#quality-generation-form').onsubmit=async event=>{
    event.preventDefault();event.submitter.disabled=true;
    try {
      const selected=document.querySelector('#quality-benchmark').value;
      const b=selected===''?null:benchmarks.filter(x=>x.status!=='invalid')[Number(selected)];
      const body={generate:true,maxGpuSeconds:Number(document.querySelector('#quality-seconds').value)};
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
  document.querySelector('#quality-content').innerHTML=`<p role="status">${qualityEscape(record.status)} · ${qualityEscape(record.message)}</p>${['queued','running','cancelling'].includes(record.status)?'<button class="button" id="quality-eval-cancel">Cancel evaluation</button>':''}${result?Object.entries(result.candidates).map(([candidate,data])=>`<article class="quality-finding"><h2>${qualityEscape(candidate)}</h2><p>Coverage: ${data.coverage.answered}/${data.coverage.expected} · Missing: ${qualityEscape(data.coverage.missingCaseIds.join(', ') || 'none')}</p><h3>Separate task measures</h3><pre>${qualityEscape(JSON.stringify(data.metrics,null,2))}</pre><h3>Per-subject measures</h3><pre>${qualityEscape(JSON.stringify(data.subjects,null,2))}</pre><details><summary>Answers, language findings and review judgments</summary><pre>${qualityEscape(JSON.stringify(data.results,null,2))}</pre></details></article>`).join('')+`<h2>Paired differences</h2><pre>${qualityEscape(JSON.stringify(result.comparisons,null,2))}</pre><h3>Comparisons not supported</h3><pre>${qualityEscape(JSON.stringify(result.unpaired,null,2))}</pre>`:''}<a class="button" href="/api/quality/evaluations/${id}/export">Export evaluation report</a>`;
  document.querySelector('#quality-content').insertAdjacentHTML('afterbegin',benchmarkNote);
  document.querySelector('#quality-eval-cancel')?.addEventListener('click',async()=>{try{await qualityRequest(`/evaluations/${id}/cancel`,{});await qualityEvaluationResult(id);}catch(e){qualityError(e);}});
  if(['queued','running','cancelling'].includes(record.status)) qualityTimer=setTimeout(()=>qualityEvaluationResult(id).catch(qualityError),1500);
}
