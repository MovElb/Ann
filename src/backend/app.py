import logging
from typing import Optional, Dict

import uvloop
import yaml
from aiohttp import web
import click

from src.backend.connectors import setup_connectors
from src.backend.middlewares import error_middleware
from src.backend.schemas import ConfigSchema
from src.backend.sentry import init_sentry
from src.backend.views import search_handler

uvloop.install()

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] PID: %(process)d %(levelname)s @ "
           "%(pathname)s:%(lineno)d ~ %(message)s",
    datefmt="%d/%b/%Y %H:%M:%S",
)


def read_config(config_path: str) -> Dict:
    with open(config_path, "r") as conf_file:
        raw_config = yaml.load(conf_file) or {}

    return ConfigSchema(strict=True).load(raw_config).data


def setup_routes(app: web.Application) -> None:
    app.router.add_post('/search', search_handler)


async def create_app(config_path: str) -> web.Application:
    app = web.Application(middlewares=[error_middleware])

    app["config"] = read_config(config_path)

    init_sentry(app)
    setup_routes(app)
    setup_connectors(app)

    return app


@click.command()
@click.option("--config-path", default='/qaweb/etc/development.yml')
@click.option("--port", default=8080)
@click.option("--socket-path")
def start_app(config_path: str, port: int, socket_path: Optional[str]) -> None:
    web.run_app(
        create_app(config_path), host='0.0.0.0', port=port, access_log=None, path=socket_path
    )


if __name__ == '__main__':
    start_app()
