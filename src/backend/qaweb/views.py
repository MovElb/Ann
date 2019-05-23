import logging

import ujson
from aiohttp import web

from qaweb.connectors import SaaSConnector
from qaweb.custom_prepro import CustomPrepro

logger = logging.getLogger(__name__)


async def search_handler(request: web.Request) -> web.Response:
    try:
        request_body = await request.json(loads=ujson.loads)
    except ValueError:
        raise web.HTTPBadRequest(text="JSON is malformed")

    query = request_body['query']

    saas: SaaSConnector = request.app['saas']
    if request_body['text']:
        texts = [request_body['text']]
    else:
        texts = saas.get_documents(query)

    prepro: CustomPrepro = request.app['prepro']
    preprocessed_data = []
    for text in texts:
        preprocessed_data.append(prepro.prepro(text, query))

    return web.json_response({'data': preprocessed_data})
