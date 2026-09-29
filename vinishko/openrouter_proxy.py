"""Прокси для всех вызовов OpenRouter: второй уровень, сомелье, whatis.

OR_PROXY задана и не пуста — запросы к OpenRouter идут через неё, а check_proxy при старте убеждается, что через прокси OpenRouter
вообще отвечает, иначе падает с OpenRouterProxyError. OR_PROXY не задана или пуста — как без неё: второй уровень берёт прокси из
окружения по правилам urllib (HTTPS_PROXY и т. п.), сомелье и whatis — openrouter_http_proxy либо ходят напрямую.
"""

import os
import urllib.error
import urllib.parse
import urllib.request
from functools import cache

from kostyl.utils import setup_logger


PROXY_ENV = "OR_PROXY"
CHECK_URL = "https://openrouter.ai/api/v1/models"
"""Открытый список моделей: отвечает без ключа, поэтому проверяет только путь через прокси."""
CHECK_TIMEOUT_SECONDS = 15.0

logger = setup_logger(fmt="detailed")


class OpenRouterProxyError(RuntimeError):
    """OR_PROXY задана, но запросы к OpenRouter через неё не проходят."""


def masked(proxy: str) -> str:
    """Адрес прокси для сообщений: пароль заменён звёздочками."""
    parts = urllib.parse.urlsplit(proxy)
    if parts.password is None:
        return proxy
    host = parts.netloc.rsplit("@", 1)[1]
    return urllib.parse.urlunsplit(
        parts._replace(netloc=f"{parts.username}:***@{host}")
    )


def or_proxy() -> str | None:
    """OR_PROXY без пробелов по краям; None, если не задана или пуста."""
    proxy = os.environ.get(PROXY_ENV, "").strip()
    if not proxy:
        return None
    if urllib.parse.urlsplit(proxy).scheme not in ("http", "https"):
        raise OpenRouterProxyError(
            f"{PROXY_ENV}={masked(proxy)}: нужен адрес вида http://[логин:пароль@]хост:порт — вызовы OpenRouter идут через urllib, "
            "а он умеет только HTTP-прокси"
        )
    return proxy


def opener(fallback: dict[str, str] | None = None) -> urllib.request.OpenerDirector:
    """Opener для запроса к OpenRouter: через OR_PROXY, если она задана; иначе с прокси fallback, как было без OR_PROXY.

    fallback None — прокси из окружения, как у urllib.request.urlopen; {} — напрямую; словарь схема → адрес — через него.
    """
    proxy = or_proxy()
    proxies = {"http": proxy, "https": proxy} if proxy is not None else fallback
    return urllib.request.build_opener(urllib.request.ProxyHandler(proxies))


def describe_error(error: BaseException) -> str:
    """Текст ошибки вызова OpenRouter; к сетевой ошибке при заданной OR_PROXY дописан прокси, через который шёл запрос."""
    text = f"{type(error).__name__}: {error}"
    proxy = os.environ.get(PROXY_ENV, "").strip()
    if proxy and isinstance(error, OSError):
        text += f" (запрос шёл через {PROXY_ENV} {masked(proxy)})"
    return text


def check_proxy() -> None:
    """Если OR_PROXY задана — убедиться, что OpenRouter отвечает через неё, иначе OpenRouterProxyError; без OR_PROXY ничего не делает."""
    proxy = or_proxy()
    if proxy is not None:
        verify(proxy)


@cache
def verify(proxy: str) -> None:
    """Один GET списка моделей OpenRouter через прокси; удачная проверка запоминается на процесс, неудачная — нет."""
    handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
    request = urllib.request.Request(CHECK_URL, headers={"User-Agent": "vinishko/1"})
    shown = f"{PROXY_ENV}={masked(proxy)}"
    try:
        with urllib.request.build_opener(handler).open(
            request, timeout=CHECK_TIMEOUT_SECONDS
        ) as response:
            status = response.status
    except urllib.error.HTTPError as error:
        raise OpenRouterProxyError(
            f"{shown}: прокси пропустил запрос, но OpenRouter ответил {error.code} {error.reason} на {CHECK_URL} — "
            "через этот прокси OpenRouter недоступен"
        ) from error
    except OSError as error:
        reason = error.reason if isinstance(error, urllib.error.URLError) else error
        raise OpenRouterProxyError(
            f"{shown}: запрос к OpenRouter через прокси не прошёл за {CHECK_TIMEOUT_SECONDS:.0f} с: {reason}. "
            f"Проверьте адрес, порт, логин и пароль прокси либо уберите {PROXY_ENV}, чтобы ходить без неё"
        ) from error
    logger.info(f"{shown}: OpenRouter отвечает через прокси ({status})")
