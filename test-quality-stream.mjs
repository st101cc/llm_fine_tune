import test from 'node:test';
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {once} from 'node:events';

test('quality exports stream before upstream completes and retain download name', async () => {
  const {streamTrainer} = await import('./quality-proxy.mjs');
  let release;
  const gate = new Promise(resolve => {release = resolve;});
  const upstream = createServer(async (_req, res) => {
    res.writeHead(200, {'content-type':'application/x-ndjson', 'content-disposition':'attachment; filename="review.jsonl"'});
    res.write('first\n');
    await gate;
    res.end('last\n');
  }).listen(0, '127.0.0.1');
  await once(upstream, 'listening');
  const proxy = createServer((req, res) => streamTrainer(`http://127.0.0.1:${upstream.address().port}`, req, res)).listen(0, '127.0.0.1');
  await once(proxy, 'listening');
  try {
    const response = await fetch(`http://127.0.0.1:${proxy.address().port}`);
    assert.match(response.headers.get('content-disposition'), /review.jsonl/);
    const reader = response.body.getReader();
    const first = await reader.read();
    assert.equal(new TextDecoder().decode(first.value), 'first\n');
    release();
    assert.equal(new TextDecoder().decode((await reader.read()).value), 'last\n');
  } finally {
    release(); proxy.closeAllConnections(); upstream.closeAllConnections(); proxy.close(); upstream.close();
  }
});
