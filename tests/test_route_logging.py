import unittest
from types import SimpleNamespace

from aetherstream.upstreams.route_logging import format_account_pool_route


class AccountPoolRouteLoggingTests(unittest.TestCase):
    def test_formats_account_pool_route(self) -> None:
        response = SimpleNamespace(
            headers={
                "x-account-pool-id": "himodels_1",
                "x-account-pool-status": "ok",
            }
        )

        self.assertEqual(
            format_account_pool_route(response),
            "account=himodels_1 pool_status=ok",
        )

    def test_formats_direct_route_without_pool_headers(self) -> None:
        response = SimpleNamespace(headers={})

        self.assertEqual(
            format_account_pool_route(response),
            "account=direct pool_status=-",
        )

    def test_sanitizes_route_header_values(self) -> None:
        response = SimpleNamespace(
            headers={
                "x-account-pool-id": "bad\nroute",
                "x-account-pool-status": "needs review",
            }
        )

        self.assertEqual(
            format_account_pool_route(response),
            "account=bad_route pool_status=needs_review",
        )


if __name__ == "__main__":
    unittest.main()
