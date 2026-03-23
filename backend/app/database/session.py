from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

DATABASE_URL = "sqlite:///./rag_chatbot.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # needed for SQLite + FastAPI
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from app.database import models  # noqa: F401 — registers models
    Base.metadata.create_all(bind=engine)
