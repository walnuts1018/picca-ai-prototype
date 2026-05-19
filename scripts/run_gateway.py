from __future__ import annotations

import uvicorn

from picca_search.gateway.config import GatewaySettings
from picca_search.gateway.runtime import create_gateway_app


def main() -> None:
    settings = GatewaySettings.from_env()
    uvicorn.run(create_gateway_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
