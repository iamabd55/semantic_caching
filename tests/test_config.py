from app.core.config import Settings, get_settings


def test_settings_load():
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.PROJECT_NAME == "Semantic Caching Gateway"
    assert settings.SIMILARITY_THRESHOLD in (0.75, 0.85)
    assert settings.QDRANT_URL == "http://localhost:6333"
