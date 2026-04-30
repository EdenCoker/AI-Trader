from ai_trader.config import AppSettings


def test_settings_redacts_secrets(monkeypatch):
    monkeypatch.setenv("QUIVER_API_KEY", "secret-quiver")

    settings = AppSettings()

    assert settings.provider_status()["quiver"] is True
    assert settings.redacted()["quiver_api_key"] == "**********"

