const assert = require('node:assert/strict');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base = process.env.FORGETUNE_URL || 'http://127.0.0.1:4174';
(async () => {
  const browser = await chromium.launch({channel:'msedge',headless:true});
  try {
    const page = await browser.newPage();
    const errors=[], external=[];
    page.on('pageerror',e=>errors.push(e.message));
    await page.route('**/*', route=>{
      const url=new URL(route.request().url());
      if(url.origin!==base){external.push(url.origin);return route.abort();}
      return route.continue();
    });
    await page.goto(base+'/#/quality');
    const source=JSON.stringify({prompt:'保留原始问题',completion:'请使用软件',metadata:{id:'synthetic-only'}})+'\n';
    await page.locator('#quality-file').setInputFiles({name:'synthetic-zh-TW.jsonl',mimeType:'application/x-ndjson',buffer:Buffer.from(source)});
    await page.getByRole('button',{name:'Start audit',exact:true}).click();
    await page.getByRole('heading',{name:'Review changes',exact:true}).waitFor();
    await page.getByRole('button',{name:'Create dataset version',exact:true}).waitFor({timeout:20000});
    // Accept every editable completion finding separately; prompts remain disabled.
    while(await page.locator('.quality-finding').filter({hasText:'pending'}).getByRole('button',{name:'Accept',exact:true}).filter({visible:true}).count()) {
      const pending=page.locator('.quality-finding').filter({hasText:'pending'}).locator('button[data-quality-decision="accepted"]:enabled');
      if(!await pending.count())break;
      const saved=page.waitForResponse(r=>r.url().endsWith('/review')&&r.request().method()==='POST');
      await pending.first().click();await saved;
      await page.waitForFunction(()=>!document.querySelector('#quality-content')?.textContent.includes('Loading'));
      await page.waitForTimeout(150);
    }
    await page.getByRole('button',{name:'Create dataset version',exact:true}).click();
    const link=page.getByRole('link',{name:'Export dataset',exact:true});
    await link.waitFor();
    const exported=await page.request.get(new URL(await link.getAttribute('href'),base).href);
    assert.equal(exported.status(),200);
    const row=JSON.parse(await exported.text());
    assert.equal(row.prompt,'保留原始问题');
    assert.equal(row.completion,'請使用軟體');
    assert.deepEqual(row.metadata,{id:'synthetic-only'});
    await page.goto(base+'/#/quality-evaluations');
    await page.locator('#quality-cases').setInputFiles({name:'cases.jsonl',mimeType:'application/x-ndjson',buffer:Buffer.from(JSON.stringify({id:'case-1',input:'訂單TW1，共2件。',reference:{order_id:'TW1',quantity:2},task:'json'})+'\n')});
    await page.locator('#quality-answers').setInputFiles({name:'answers.jsonl',mimeType:'application/x-ndjson',buffer:Buffer.from(JSON.stringify({candidate:'synthetic-saved-output',caseId:'case-1',output:'{"order_id":"TW1","quantity":2}'})+'\n')});
    await page.getByRole('button',{name:'Evaluate answers',exact:true}).click();
    await page.getByRole('heading',{name:'synthetic-saved-output',exact:true}).waitFor({timeout:20000});
    assert.match(await page.locator('#quality-content').innerText(),/Coverage: 1\/1/);
    assert.match(await page.locator('#quality-content').innerText(),/"structuralExact": 1/);
    await page.setViewportSize({width:390,height:844});
    assert(await page.evaluate(()=>document.documentElement.scrollWidth <= window.innerWidth), 'Mobile page overflows');
    assert.deepEqual(errors,[]);assert.deepEqual(external,[]);
    console.log('Live browser → Node proxy → FastAPI → SQLite audit/review/export/evaluation passed; no external browser calls.');
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
