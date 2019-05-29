import os
import urllib.parse

import aiohttp
import aiowiki
import ujson
from typing import Any, List, Dict

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
        self._wiki = aiowiki.Wiki.wikipedia("en")

    async def get_documents(self, query, limit: int = 10) -> List[aiowiki.Page]:
        quote_encoded = urllib.parse.quote(query)

        async with self._sess.get(self._url + quote_encoded) as resp:
            resp.raise_for_status()

            google_serp: Dict = await resp.json(loads=ujson.loads)
            urls = [item['link'] for item in google_serp.get('items', list())][:limit]

            texts = []
            for url in urls:
                texts.append(self._wiki.get_page(url.split('/')[-1]))
            return texts

    async def process_query(self, query) -> str:
        pass


class NetConnector(BaseConnector):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    async def get_answer(self, preprocessed) -> Dict:
        if len(preprocessed) == 0:
            return {}

        async with self._sess.post(self._url, json={'data': preprocessed}) as resp:
            resp.raise_for_status()
            return await resp.json()


def setup_connectors(app: web.Application) -> None:
    app['saas'] = SaaSConnector(**app['config']['saas'])
    app['net'] = NetConnector(**app['config']['net'])
