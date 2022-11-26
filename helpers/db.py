
import psycopg2
from sqlalchemy import create_engine, exc, text

def _create_engine(uri, pool_recycle=300, **kwargs):
    """
    Private method to create SQLAlchemy connection engine with pooling configs

    Args:
        uri: the database uri string used to establish the connection
        random comment

    Returns:
        engine: SQLAlchemy connection engine
    """

    return create_engine(
        uri,
        pool_size=5,
        max_overflow=2,
        pool_recycle=pool_recycle,
        pool_pre_ping=True,
        pool_use_lifo=True,
        **kwargs,
    )


class DB:
    def __init__(self):
        self.host = "localhost"
        self.user = "postgres"
        self.password = "postgres"
        self.db = "postgres"
        self.port = 5438
        self.driver = "postgresql"
        self.schema = "public"
        self.uri = f"{self.driver}://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"
        self.engine = _create_engine(self.uri, pool_recycle=-1)