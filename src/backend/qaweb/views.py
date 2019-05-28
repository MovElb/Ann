import logging
import sys
import time
from typing import Dict

import ujson
from aiohttp import web

from qaweb.connectors import SaaSConnector, NetConnector
from qaweb.custom_prepro import CustomPrepro

logger = logging.getLogger(__name__)


async def search_handler(request: web.Request) -> web.Response:
    try:
        request_body = await request.json(loads=ujson.loads)
        query = request_body['query']
    except ValueError:
        raise web.HTTPBadRequest(text="JSON is malformed")

    saas: SaaSConnector = request.app['saas']
    if request_body.get('text'):
        documents = [request_body['text']]
    else:
        st = time.time()
        documents = await saas.get_documents(query)
        print('Time in Google', time.time() - st, flush=True, file=sys.stdout)

    st = time.time()
    texts = []
    for doc in documents:
        try:
            text = await doc.summary()
            if len(text) > 1:
                texts.append(text)
        except Exception:
            pass
    print('Time in wikipedia', time.time() - st, flush=True, file=sys.stdout)

    st = time.time()
    prepro: CustomPrepro = request.app['prepro']
    preprocessed_data = []
    for text in texts:
        preprocessed_data.append(prepro.prepro(text, query))
    print('Time in CustomPrepro', time.time() - st, flush=True, file=sys.stdout)

    st = time.time()
    net: NetConnector = request.app['net']
    answers: Dict = await net.get_answer(preprocessed_data)
    print('Time in Net', time.time() - st, flush=True, file=sys.stdout)

    st = time.time()
    answers_packed = []
    for i in range(len(texts)):
        answ = {'text': texts[i]}
        for key in answers.keys():
            answ[key] = answers[key][i]
        answers_packed.append(answ)
    print('Time in packing', time.time() - st, flush=True, file=sys.stdout)

    # answers_packed.sort(key=lambda a: a['score'], reverse=True)

    return web.json_response({'answers': answers_packed}, dumps=ujson.dumps)
