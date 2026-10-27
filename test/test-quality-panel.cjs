const assert = require('node:assert/strict');
const fs = require('node:fs');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base = process.env.FORGETUNE_URL || 'http://127.0.0.1:4173';
const shot = process.env.PANEL_SCREENSHOT;
const tally = (pass, fail, needsReview) => ({pass, fail, needsReview, rate: pass + fail ? pass / (pass + fail) : null, coverage: (pass + fail) / (pass + fail + needsReview)});
const empty = {pass: null, fail: null, needsReview: null, rate: null, coverage: null};
const vote = (c, l) => ({correctness: {verdict: c, reason: `${c} because correct`}, language: {verdict: l, reason: l === 'fail' ? 'Uses 软件 where Taiwan usage is 軟體' : 'Natural Taiwan usage'}});
const flagged = {itemId: 'w1|reviewed', caseId: 'w1', candidate: 'reviewed', question: 'Explain software updates', answer: '請更新软件 <img src=x onerror="window.pwned=1">', truncated: true, correctness: 'pass', language: 'fail',
  votes: {breeze: vote('pass', 'fail'), qwen: vote('pass', 'pass'), yi: {error: 'invalid_output'}}};
const steps = (done, graded) => ['prepare', 'judge-breeze', 'judge-qwen', 'judge-yi', 'report'].map(name => ({name, status: done ? 'complete' : name === 'prepare' ? 'complete' : name === 'judge-breeze' ? 'running' : 'pending', graded: name.startsWith('judge') ? graded : 0, total: name.startsWith('judge') ? 20 : null, message: null, gpuSeconds: 30}));
const record = {id: 'p1', displayName: 'Comparison c1', status: 'running', source: {comparisonId: 'c1'}, prompts: 20, referenceMode: 'none', steps: steps(false, 12), report: null,
  protocol: {panelVersion: 'judge-panel-v1', aggregationRule: 'majority of 3'}, evaluatorId: 'abc123'};
const report = {candidates: {base: {correctness: tally(18, 0, 2), language: tally(15, 5, 0)}, reviewed: {correctness: tally(10, 6, 4), language: tally(19, 1, 0)}}, flagged: [flagged], items: Array.from({length: 40}, (_, i) => ({itemId: 'i' + i})), incomplete: false, humanAgreement: {compared: 0, agreed: 0}, selfGrading: {}};
const incomplete = {...report, candidates: {base: {correctness: empty, language: tally(15, 5, 0)}, reviewed: {correctness: empty, language: empty}}, flagged: [], incomplete: true};
(async () => {
  const browser = await chromium.launch({channel: 'msedge', headless: true});
  try {
    const page = await browser.newPage({viewport: {width: 1280, height: 900}});
    const errors = []; page.on('pageerror', e => errors.push(e.message));
    page.on('dialog', d => d.accept());
    const estimates = []; let ready = false, current = record;
    const judge = status => ({modelId: 'm', revision: 'r', status, path: '/x'});
    await page.route('**/api/**', async route => {
      const request = route.request(), path = new URL(request.url()).pathname.replace('/api/quality', '');
      let status = 200, result = [];
      if (path === '/panels/estimate') {
        const body = request.postDataJSON(); estimates.push(body);
        if (body.source.evaluationId === 'bad') { status = 422; result = {detail: 'this source has no writing answers to grade'}; }
        else result = {feasible: ready, reasons: ready ? [] : ['breeze: missing_weights', 'qwen: missing_weights', 'yi: missing_weights'], items: 60, candidates: ['base', 'original', 'reviewed'], prompts: body.prompts || 20, maxPrompts: 60,
          judges: {breeze: judge(ready ? 'ready' : 'missing_weights'), qwen: judge(ready ? 'ready' : 'missing_weights'), yi: judge(ready ? 'ready' : 'missing_weights')},
          estimate: {gpuSeconds: 3000, ledgerRemaining: 9000, gradesPerJudge: 60, chunksPerJudge: 1}, selfGrading: {original: 'qwen'}, referenceMode: 'none', truncatedUnknown: body.source.evaluationId === 'e1'};
      } else if (path === '/panels' && request.method() === 'POST') { status = 202; result = record; }
      else if (path === '/panels/p1') result = current;
      else if (path === '/panels/p2') result = {...record, id: 'p2', status: 'failed', message: 'judge-yi: gpu_busy', steps: steps(false, 3).map(s => s.name === 'judge-yi' ? {...s, status: 'failed', message: 'gpu_busy'} : s), report: incomplete};
      else if (path === '/panels/p3') result = {...record, id: 'p3', status: 'complete', source: {evaluationId: 'e1'}, referenceMode: 'case', steps: steps(true, 20), report: {...report, humanAgreement: {compared: 8, agreed: 6}}};
      else if (path === '/panels') result = [{id: 'p1', displayName: 'Comparison c1', status: 'complete', createdAt: new Date().toISOString(), source: {comparisonId: 'c1'}}];
      else if (path === '/comparisons/c1') result = {id: 'c1', displayName: 'Style comparison', status: 'complete', report: {headline: 'done', models: {}}, steps: []};
      else if (path === '/evaluations/e1') result = {id: 'e1', status: 'complete', message: '', cases: [], result: {candidates: {}, comparisons: [], unpaired: []}};
      else if (path === '/hardware') result = {trainerOnline: true};
      await route.fulfill({status, contentType: 'application/json', body: JSON.stringify(result)});
    });

    await page.goto(base + '/#/quality-panels/new/comparison/c1');
    await page.getByRole('heading', {name: 'Grade with judge panel', exact: true}).waitFor();
    await page.getByText('Style comparison').waitFor();
    await page.getByText('This panel can’t start yet:').waitFor();
    assert.equal(await page.getByText('not prepared: run').count(), 3);
    assert(await page.getByText('python trainer/quality_models.py prepare judges').first().isVisible());
    assert(await page.getByRole('button', {name: 'Start judge panel'}).isDisabled());
    await page.getByText('Comparison answers were cut at 128 tokens; judges grade only what is visible.').waitFor();
    await page.getByText('No reference shown to judges for comparisons.').waitFor();
    await page.getByText('original looks like the qwen judge').waitFor();
    assert.equal(await page.getByText('Uploaded answers are treated as complete.').count(), 0);
    assert.equal(await page.locator('#qp-prompts').inputValue(), '20');

    ready = true;
    await page.locator('#qp-prompts').fill('40');
    await page.getByText('Ready to start.').waitFor();
    assert(await page.getByRole('button', {name: 'Start judge panel'}).isEnabled());
    assert.deepEqual(estimates.at(-1), {source: {comparisonId: 'c1'}, prompts: 40});
    await page.getByText('3000', {exact: true}).waitFor();

    await page.getByRole('button', {name: 'Start judge panel'}).click();
    await page.getByRole('button', {name: 'Cancel panel'}).waitFor();
    await page.getByText('12 of 20 graded').first().waitFor();
    current = {...record, status: 'complete', steps: steps(true, 20), report};
    await page.getByText('Estimated by local models, not verified').waitFor({timeout: 10000});
    await page.getByText('18 of 20 graded').waitFor();
    await page.getByText('16 of 20 graded').waitFor();
    await page.getByText('63%').first().waitFor();
    await page.getByText('Uses 软件 where Taiwan usage is 軟體').waitFor();
    await page.getByText('No grade (invalid_output)').waitFor();
    assert.equal(await page.getByText('Human agreement').count(), 0);
    assert.equal(await page.getByText('Incomplete: fewer than three judges finished').count(), 0);
    assert.equal(await page.evaluate(() => window.pwned), undefined);
    assert.equal(await page.locator('.quality-example pre img').count(), 0);
    assert((await page.getByRole('link', {name: 'Export panel (JSON)'}).getAttribute('href')).endsWith('/api/quality/panels/p1/export'));
    if (shot) { fs.mkdirSync(require('node:path').dirname(shot), {recursive: true}); await page.screenshot({path: shot, fullPage: true}); }

    await page.goto(base + '/#/quality-panels/p3');
    await page.getByText('Human agreement: the panel matched 6 of 8').waitFor();

    await page.goto(base + '/#/quality-panels/p2');
    await page.getByText('Incomplete: fewer than three judges finished').waitFor();
    assert.equal(await page.getByText('0%', {exact: true}).count(), 0);
    assert.equal(await page.locator('tbody tr').nth(0).locator('td').nth(0).innerText(), '—');
    assert.equal(await page.locator('tbody tr').nth(1).locator('td').nth(2).innerText(), '—');
    await page.getByText('judge-yi: gpu_busy').first().waitFor();
    assert.equal(await page.getByRole('link', {name: 'Start a new panel'}).getAttribute('href'), '#/quality-panels/new/comparison/c1');
    assert.equal(await page.getByText('start the panel again').count(), 0);

    await page.goto(base + '/#/quality-panels/new/evaluation/bad');
    await page.getByText('this source has no writing answers to grade').waitFor();
    assert(await page.getByRole('button', {name: 'Start judge panel'}).isDisabled());
    await page.goto(base + '/#/quality-panels/new/evaluation/e1');
    await page.getByText('Uploaded answers are treated as complete.').waitFor();
    assert.equal(await page.getByText('Comparison answers were cut').count(), 0);

    await page.goto(base + '/#/quality-panels');
    await page.getByRole('link', {name: 'Comparison c1'}).waitFor();
    await page.goto(base + '/#/quality-evaluations/e1');
    assert.equal(await page.getByRole('link', {name: 'Grade with judge panel'}).getAttribute('href'), '#/quality-panels/new/evaluation/e1');
    await page.goto(base + '/#/quality-comparisons/c1');
    assert.equal(await page.getByRole('link', {name: 'Grade with judge panel'}).getAttribute('href'), '#/quality-panels/new/comparison/c1');
    assert(await page.locator('nav a[data-route="quality-panels"]').isVisible());
    assert.deepEqual(errors, []);
    console.log('Quality judge panel UI passed');
  } finally {await browser.close();}
})().catch(e => {console.error(e); process.exitCode = 1;});
