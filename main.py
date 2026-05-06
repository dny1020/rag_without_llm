from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router as api_router
from src.core.config import get_settings
from src.db.models import Base
from src.db.session import engine, init_vector_extension
from src.helpers.logger import configure_logging

settings = get_settings()


def create_app() -> FastAPI:
    configure_logging(settings.log_level)
    app = FastAPI(title=settings.app_name)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router, prefix=settings.api_prefix)
    return app


def bootstrap() -> None:
    init_vector_extension()
    Base.metadata.create_all(bind=engine)


bootstrap()
app = create_app()
