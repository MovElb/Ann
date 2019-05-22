import logging
from typing import Optional, Dict

import uvloop
import yaml
from aiohttp import web
import click

from qaweb.custom_prepro import CustomPrepro
from .connectors import setup_connectors
from .schemas import ConfigSchema
from .views import search_handler

uvloop.install()

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] PID: %(process)d %(levelname)s @ "
           "%(pathname)s:%(lineno)d ~ %(message)s",
    datefmt="%d/%b/%Y %H:%M:%S",
)


def read_config(config_path: str) -> Dict:
    with open(config_path, "r") as conf_file:
        raw_config = yaml.load(conf_file, Loader=yaml.FullLoader) or {}

    return ConfigSchema(strict=True).load(raw_config).data


def setup_routes(app: web.Application) -> None:
    app.router.add_get('/api/search', search_handler)


async def create_app(config_path: str) -> web.Application:
    app = web.Application()

    app["config"] = read_config(config_path)
    app["prepro"] = CustomPrepro()

    setup_routes(app)
    setup_connectors(app)

    return app


@click.command()
@click.option("--config-path", default='/qaweb/configs/development.yml')
@click.option("--port", default=8080)
@click.option("--socket-path")
def start_app(config_path: str, port: int, socket_path: Optional[str]) -> None:
    web.run_app(
        create_app(config_path), host='0.0.0.0', port=port, access_log=None, path=socket_path
    )


if __name__ == '__main__':
    start_app()
