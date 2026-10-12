import {Readable} from 'node:stream';
import {pipeline} from 'node:stream/promises';

export async function streamTrainer(url, request, response) {
  const controller = new AbortController();
  const closed = () => {if (!response.writableFinished) controller.abort();};
  response.on('close', closed);
  try {
    const upstream = await fetch(url, {signal:controller.signal});
    const headers = {'content-type':upstream.headers.get('content-type') || 'application/octet-stream'};
    for (const name of ['content-disposition', 'content-length']) {
      if (upstream.headers.has(name)) headers[name] = upstream.headers.get(name);
    }
    response.writeHead(upstream.status, headers);
    if (upstream.body) await pipeline(Readable.fromWeb(upstream.body), response);
    else response.end();
  } catch (error) {
    if (!response.headersSent) {
      response.writeHead(503, {'content-type':'application/json'});
      response.end(JSON.stringify({detail:'Local trainer export unavailable.'}));
    } else response.destroy(error);
  } finally {response.off('close', closed);}
}
