import logging

from aiohttp import web

logger = logging.getLogger(__name__)


async def search_handler(request: web.Request) -> web.Response:
    return web.Response(text='hello, world')
