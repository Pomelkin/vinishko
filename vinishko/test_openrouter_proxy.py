"""OR_PROXY без сети: разбор переменной, выбор прокси у вызовов OpenRouter и проверка при старте с подменённым ответом."""

import os
import socket
import unittest
import urllib.error
from email.message import Message
from unittest.mock import MagicMock
from unittest.mock import patch

from vinishko import openrouter_proxy
from vinishko.openrouter_proxy import OpenRouterProxyError
from vinishko.openrouter_proxy import check_proxy
from vinishko.openrouter_proxy import describe_error
from vinishko.openrouter_proxy import opener
from vinishko.openrouter_proxy import or_proxy
from vinishko.whatis.solution import predict as whatis_predict


JPEG = b"\xff\xd8\xff\xe0test-image"


def proxies(fallback: dict[str, str] | None) -> dict:
    """Прокси, с которыми opener(fallback) собирает ProxyHandler."""
    with patch("urllib.request.build_opener") as build:
        opener(fallback)
    return build.call_args.args[0].proxies


def free_port() -> int:
    """Порт на 127.0.0.1, где никто не слушает."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class OrProxyTests(unittest.TestCase):
    def setUp(self) -> None:
        openrouter_proxy.verify.cache_clear()

    def test_unset_or_blank_means_no_proxy(self) -> None:
        for value in (None, "", "   "):
            env = {} if value is None else {"OR_PROXY": value}
            with self.subTest(value=value), patch.dict(os.environ, env, clear=True):
                self.assertIsNone(or_proxy())

    def test_only_http_proxies_are_accepted(self) -> None:
        with (
            patch.dict(
                os.environ, {"OR_PROXY": "socks5://u:pw@127.0.0.1:1080"}, clear=True
            ),
            self.assertRaises(OpenRouterProxyError) as caught,
        ):
            or_proxy()
        self.assertIn("http://", str(caught.exception))
        self.assertNotIn("pw", str(caught.exception))

    def test_or_proxy_wins_over_fallback(self) -> None:
        env = {"OR_PROXY": " http://u:pw@proxy:3128 ", "HTTPS_PROXY": "http://env:1"}
        with patch.dict(os.environ, env, clear=True):
            for fallback in (None, {}, {"https": "http://legacy:8080"}):
                with self.subTest(fallback=fallback):
                    self.assertEqual(
                        proxies(fallback),
                        {
                            "http": "http://u:pw@proxy:3128",
                            "https": "http://u:pw@proxy:3128",
                        },
                    )

    def test_without_or_proxy_fallback_is_kept(self) -> None:
        with patch.dict(os.environ, {"HTTPS_PROXY": "http://env:1"}, clear=True):
            self.assertEqual(proxies({}), {})
            self.assertEqual(
                proxies({"https": "http://legacy:8080"}),
                {"https": "http://legacy:8080"},
            )
            self.assertEqual(proxies(None)["https"], "http://env:1")

    def test_whatis_goes_through_or_proxy_instead_of_legacy_variable(self) -> None:
        env = {
            "OPENROUTER_API_KEY": "test-key",
            "OR_PROXY": "http://proxy:3128",
            "openrouter_http_proxy": "http://legacy:8080",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch("urllib.request.build_opener") as build,
        ):
            build.return_value.open.side_effect = urllib.error.URLError("stop")
            result = whatis_predict(JPEG)
        self.assertEqual(
            build.call_args.args[0].proxies,
            {"http": "http://proxy:3128", "https": "http://proxy:3128"},
        )
        self.assertIn("через OR_PROXY http://proxy:3128", result["_error"])

    def test_describe_error_names_proxy_only_for_network_errors(self) -> None:
        with patch.dict(os.environ, {"OR_PROXY": "http://u:pw@proxy:3128"}, clear=True):
            self.assertEqual(
                describe_error(TimeoutError("timed out")),
                "TimeoutError: timed out (запрос шёл через OR_PROXY http://u:***@proxy:3128)",
            )
            self.assertEqual(
                describe_error(ValueError("bad json")), "ValueError: bad json"
            )
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                describe_error(TimeoutError("timed out")), "TimeoutError: timed out"
            )

    def test_check_does_nothing_without_or_proxy(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("urllib.request.OpenerDirector.open") as send,
        ):
            check_proxy()
        send.assert_not_called()

    def test_check_passes_once_when_openrouter_answers(self) -> None:
        response = MagicMock(status=200)
        response.__enter__.return_value = response
        with (
            patch.dict(os.environ, {"OR_PROXY": "http://proxy:3128"}, clear=True),
            patch("urllib.request.OpenerDirector.open", return_value=response) as send,
        ):
            check_proxy()
            check_proxy()
        self.assertEqual(send.call_count, 1)
        self.assertEqual(send.call_args.args[0].full_url, openrouter_proxy.CHECK_URL)

    def test_dead_proxy_fails_with_clear_error(self) -> None:
        dead = f"http://u:pw@127.0.0.1:{free_port()}"
        with (
            patch.dict(os.environ, {"OR_PROXY": dead}, clear=True),
            self.assertRaises(OpenRouterProxyError) as caught,
        ):
            check_proxy()
        message = str(caught.exception)
        self.assertIn("не прошёл", message)
        self.assertIn("u:***@127.0.0.1", message)
        self.assertNotIn("pw", message)

    def test_openrouter_refusal_through_proxy_fails(self) -> None:
        refusal = urllib.error.HTTPError(
            openrouter_proxy.CHECK_URL, 403, "Forbidden", Message(), None
        )
        with (
            patch.dict(os.environ, {"OR_PROXY": "http://proxy:3128"}, clear=True),
            patch("urllib.request.OpenerDirector.open", side_effect=refusal),
            self.assertRaises(OpenRouterProxyError) as caught,
        ):
            check_proxy()
        self.assertIn("OpenRouter ответил 403", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
