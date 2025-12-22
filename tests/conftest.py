# Test configuration and fixtures
import pytest
from src.main import app
from httpx import AsyncClient


@pytest.fixture
def client():
    return AsyncClient(app=app, base_url="http://testserver")