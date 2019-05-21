import os

import aiohttp
import ujson
from typing import Any, List

import wikipediaapi
from aiohttp import web


class BaseConnector:
    def __init__(self, url: str, read_timeout: int, conn_timeout: int):
        self._url = url
        self._read_timeout = read_timeout
        self._conn_timeout = conn_timeout
        self._connector = aiohttp.TCPConnector(
            use_dns_cache=True, ttl_dns_cache=60 * 60, limit=1024
        )

        self._sess = aiohttp.ClientSession(
            connector=self._connector,
            read_timeout=self._read_timeout,
            conn_timeout=self._conn_timeout,
            json_serialize=ujson.dumps,
        )


class SaaSConnector(BaseConnector):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._url = self._url.format(os.environ['GAPI_KEY'], os.environ['GAPI_CX'])
        self._wiki = wikipediaapi.Wikipedia('en')

    async def get_documents(self, query, limit: int = 10) -> List[str]:
        async with self._sess.get(self._url) as resp:
            resp.raise_for_status()
            urls = [item['link'] for item in await resp.json(loads=ujson.loads)]

            texts = []
            for url in urls:
                texts.append(self._wiki.page(url.split('/')[-1]).text)
            return texts

    async def process_query(self, query) -> str:
        pass


class NetConnector(BaseConnector):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    async def get_answer(self, query, texts) -> str:
        pass


def setup_connectors(app: web.Application) -> None:
    app['saas'] = SaaSConnector(**app['config']['saas'])
    app['net'] = NetConnector(**app['config']['net'])