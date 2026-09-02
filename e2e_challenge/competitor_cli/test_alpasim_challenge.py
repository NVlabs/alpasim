# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from e2e_challenge.competitor_cli import alpasim_challenge as cli


class FakeClient:
    def __init__(self, responses):
        self.responses = responses
        self.posts = []

    def get(self, path):
        return self.responses[path]

    def post(self, path, payload=None):
        self.posts.append((path, payload))
        return self.responses[path]


class TermsCliTest(unittest.TestCase):
    def terms(self, **overrides):
        value = {
            "terms_version": "placeholder-2026-09-02",
            "terms_sha256": "digest",
            "terms_text": "this will contain the terms and conditions.\n",
            "actor_accepted": False,
            "captain_accepted": False,
            "team_ready": False,
        }
        value.update(overrides)
        return value

    def test_not_ready_explains_both_required_acceptances(self):
        client = FakeClient({"/terms/status": self.terms()})

        with self.assertRaises(SystemExit) as raised:
            cli.require_terms_ready(client)

        self.assertIn("Both you and your team captain", str(raised.exception))

    def test_accept_posts_exact_displayed_version_and_digest(self):
        current = self.terms()
        accepted = self.terms(
            actor_accepted=True, captain_accepted=True, team_ready=True
        )
        client = FakeClient({"/terms/current": current, "/terms/accept": accepted})

        with patch("builtins.input", return_value="ACCEPT placeholder-2026-09-02"):
            with redirect_stdout(io.StringIO()):
                cli.accept_terms(client)

        self.assertEqual(
            client.posts,
            [
                (
                    "/terms/accept",
                    {
                        "accepted": True,
                        "terms_version": "placeholder-2026-09-02",
                        "terms_sha256": "digest",
                    },
                )
            ],
        )

    def test_wrong_confirmation_makes_no_server_change(self):
        client = FakeClient({"/terms/current": self.terms()})

        with patch("builtins.input", return_value="yes"):
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                cli.accept_terms(client)

        self.assertEqual(client.posts, [])

    def test_ready_status_passes(self):
        client = FakeClient(
            {
                "/terms/status": self.terms(
                    actor_accepted=True,
                    captain_accepted=True,
                    team_ready=True,
                )
            }
        )

        cli.require_terms_ready(client)


if __name__ == "__main__":
    unittest.main()
