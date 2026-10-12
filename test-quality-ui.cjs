const assert = require('node:assert/strict');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
(async () => {
  const browser = await chromium.launch({channel:'msedge',headless:true});
  try {
    const page = await browser.newPage();
    const errors = []; page.on('pageerror', e => errors.push(e.message));
    const audit = {id:'a',status:'complete',coverage:'full',mode:'full',scannedRows:1,totalRows:1,findingCount:1,fingerprint:'f'.repeat(64),policy:{version:'zh-TW-review-v1'},dataset:{source:'upload',name:'fixture.jsonl'}};
    let accepted = false;
    await page.route('**/api/**', async route => {
      const path = new URL(route.request().url()).pathname;
      let result = [];
      if (path.endsWith('/hardware')) result = {trainerOnline:true};
      else if(path.endsWith('/upload')) result = audit.dataset;
      else if(path.endsWith('/audits')) result = route.request().method() === 'POST' ? audit : [audit];
      else if(path.endsWith('/audits/a')) result = audit;
      else if(path.endsWith('/findings')) result = {total:1,offset:0,items:[{id:'f',rowId:0,field:['completion'],start:0,end:2,original:'软件',replacement:'軟體',reason:'Regional suggestion',editable:true,decision:accepted?'accepted':'pending'}]};
      else if(path.endsWith('/review')) {accepted = route.request().postDataJSON().decisions[0]?.decision === 'accepted'; result={saved:1};}
      else if(path.endsWith('/audits/a/versions')) result={id:'v',dataset:audit.dataset,rows:1,edits:1};
      await route.fulfill({contentType:'application/json',body:JSON.stringify(result)});
    });
    await page.goto((process.env.FORGETUNE_URL || 'http://127.0.0.1:4173') + '/#/quality/a');
    await page.getByRole('heading',{name:'Review changes',exact:true}).waitFor();
    await page.getByRole('button',{name:'Accept',exact:true}).click();
    await page.getByRole('button',{name:'Create dataset version',exact:true}).click();
    await page.getByRole('link',{name:'Export dataset',exact:true}).waitFor();
    assert(accepted);
    await page.goto((process.env.FORGETUNE_URL || 'http://127.0.0.1:4173') + '/#/quality-evaluations');
    await page.getByRole('heading',{name:'Evaluations',exact:true}).waitFor();
    assert.deepEqual(errors, []);
    console.log('Quality review and evaluation UI passed');
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
