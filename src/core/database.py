"""
Database module for the ChatKit RAG integration.
Handles PostgreSQL database connections using SQLAlchemy async.
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import AsyncGenerator

from .config import settings

# Create the async engine
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,  # Set to True for SQL debugging
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10
)

# Create async session maker
AsyncSessionFactory = sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get database session for FastAPI endpoints.
    """
    async with AsyncSessionFactory() as session:
        try:
            yield session
        finally:
            await session.close()


# Base class for all models
Base = declarative_base()


# Function to initialize the database
async def init_db():
    """
    Initialize the database by creating all tables.
    """
    async with engine.begin() as conn:
        # Create all tables defined in the models
        await conn.run_sync(Base.metadata.create_all)


# Function to dispose of the database engine
async def dispose_db():
    """
    Dispose of the database engine connection pool.
    """
    await engine.dispose()