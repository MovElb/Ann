import logging

import ujson
from aiohttp import web

from qaweb.connectors import SaaSConnector

logger = logging.getLogger(__name__)


async def search_handler(request: web.Request) -> web.Response:
    try:
        request_body = await request.json(loads=ujson.loads)
    except ValueError:
        raise web.HTTPBadRequest(text="JSON is malformed")

    saas: SaaSConnector = request.app['saas']
    if request_body['text']:
        texts = [request_body['text']]
    else:
        query = request_body['query']
        texts = saas.get_documents(query)


    return web.Response(text='hello, world')
