# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import hashlib
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

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


class ReportsCliTest(unittest.TestCase):
    def test_upload_uses_s3_form_without_bearer_and_finalizes(self):
        data = b"%PDF-1.7\ntest"
        grant = {
            "report": {"report_id": "revision"},
            "upload": {
                "url": "https://private.s3.amazonaws.com",
                "fields": {"key": "team/report.pdf", "policy": "signed"},
            },
        }
        client = FakeClient(
            {
                "/reports": {"max_size_bytes": 20971520},
                "/reports/uploads": grant,
                "/reports/revision/complete": {"status": "ACCEPTED"},
            }
        )
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.pdf"
            path.write_bytes(data)
            with patch.object(
                cli.urllib.request, "urlopen", return_value=response
            ) as transfer:
                result = cli.upload_report(client, path)
        self.assertEqual(result["status"], "ACCEPTED")
        self.assertEqual(client.posts[0][1]["sha256"], hashlib.sha256(data).hexdigest())
        request = transfer.call_args.args[0]
        self.assertIsNone(request.get_header("Authorization"))
        self.assertIn(b'name="file"; filename="report.pdf"', request.data)
        self.assertLess(
            request.data.index(b'name="key"'), request.data.index(b'name="file"')
        )

    def test_bad_pdf_is_rejected_before_upload_grant(self):
        client = FakeClient({"/reports": {"max_size_bytes": 20}})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pdf"
            path.write_bytes(b"not a pdf")
            with self.assertRaises(SystemExit):
                cli.upload_report(client, path)
        self.assertEqual(client.posts, [])

    def test_download_verifies_checksum_and_preserves_existing_files(self):
        receipt = {
            "report_id": "revision",
            "size_bytes": 3,
            "sha256": hashlib.sha256(b"pdf").hexdigest(),
        }
        client = FakeClient(
            {
                "/reports/download": {
                    "report": receipt,
                    "download_url": "https://s3/file",
                }
            }
        )
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b"bad"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.pdf"
            with (
                patch.object(cli.urllib.request, "urlopen", return_value=response),
                self.assertRaises(SystemExit),
            ):
                cli.download_report(client, path)
            self.assertFalse(path.exists())
            response.__enter__.return_value.read.return_value = b"pdf"
            with patch.object(cli.urllib.request, "urlopen", return_value=response):
                cli.download_report(client, path)
            self.assertEqual(path.read_bytes(), b"pdf")
            with self.assertRaises(SystemExit):
                cli.download_report(client, path)
            self.assertEqual(path.read_bytes(), b"pdf")

    def test_admin_download_uses_separate_route(self):
        client = FakeClient(
            {
                "/admin/reports/download?team_id=team-a": {
                    "report": {
                        "report_id": "revision",
                        "size_bytes": 3,
                        "sha256": hashlib.sha256(b"pdf").hexdigest(),
                    },
                    "download_url": "https://s3/file",
                }
            }
        )
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b"pdf"
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(cli.urllib.request, "urlopen", return_value=response):
                cli.download_report(
                    client, Path(directory) / "report.pdf", team_id="team-a"
                )


if __name__ == "__main__":
    unittest.main()
