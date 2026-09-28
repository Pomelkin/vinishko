"""HTTP-приложение поверх пайплайна: фото → бутылки с выбранной позицией каталога либо причиной отказа."""

__all__ = ["create_app"]


def __getattr__(name: str):
    # Сомелье и whatis не должны импортировать GPU-пайплайн через родительский пакет.
    if name == "create_app":
        from vinishko.app.main import create_app

        return create_app
    raise AttributeError(name)
