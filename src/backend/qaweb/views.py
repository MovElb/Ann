import logging
import sys
import time
from typing import Dict

import ujson
from aiohttp import web

from qaweb.connectors import SaaSConnector, NetConnector, RedisConnector
from qaweb.custom_prepro import CustomPrepro

logger = logging.getLogger(__name__)


async def search_handler(request: web.Request) -> web.Response:
    try:
        request_body = await request.json(loads=ujson.loads)
        query = request_body['query']
    except ValueError:
        raise web.HTTPBadRequest(text="JSON is malformed")

    redis: RedisConnector = request.app['redis']
    await redis.connect()
    saas: SaaSConnector = request.app['saas']

    if request_body.get('text'):
        texts = [request_body['text']]
        title = [""]
    else:
        st = time.time()
        title = await saas.get_documents(query)
        print(title, flush=True, file=sys.stdout)
        print('Time in Google', time.time() - st, flush=True, file=sys.stdout)

        st = time.time()
        texts = []
        for t in title:
            try:
                text = await redis.get_text(t)
                if not text:
                    text = await saas.get_wikipage(t).summary()
                    await redis.dump_text(t, text)
                if len(text) > 1:
                    texts.append(text)
            except Exception as e:
                print(e)
        print('Time in wikipedia', time.time() - st, flush=True, file=sys.stdout)

    st = time.time()
    prepro: CustomPrepro = request.app['prepro']
    preprocessed_data = []
    for i, text in enumerate(texts):
        prepro_text = await redis.get_preprocessed(title[i])
        if not prepro_text:
            prepro_text = prepro.prepro_text(text)
            await redis.dump_text(title[i], text)
        prepro_query = prepro.prepro_text(query)
        preprocessed_data.append(prepro.prepro_crossed(prepro_text, prepro_query))
    print('Time in CustomPrepro', time.time() - st, flush=True, file=sys.stdout)

    st = time.time()
    net: NetConnector = request.app['net']
    answers: Dict = await net.get_answer(preprocessed_data)
    print('Time in Net', time.time() - st, flush=True, file=sys.stdout)

    print(answers, flush=True, file=sys.stdout)

    st = time.time()
    answers_packed = []
    for i in range(len(texts)):
        answ = {'text': texts[i], 'title': title[i]}
        for key in answers.keys():
            answ[key] = answers[key][i]
        answers_packed.append(answ)

    srt_answ_top = sorted(answers_packed[:3], key=lambda a: a['has_ans_score'], reverse=True)
    for i in range(len(srt_answ_top)):
        answers_packed[i] = srt_answ_top[i]
    print('Time in packing', time.time() - st, flush=True, file=sys.stdout)

    return web.json_response({'query': query, 'answers': answers_packed}, dumps=ujson.dumps)
