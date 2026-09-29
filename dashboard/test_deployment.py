from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

from django.test import SimpleTestCase


class ProductionDeploymentTests(SimpleTestCase):
    def test_production_serves_collected_css_and_respects_proxy_configuration(self):
        # A fresh process loads the real DEBUG=false settings, including storage
        # and HTTPS middleware, without affecting the normal development tests.
        script = textwrap.dedent("""
            import gzip
            import json
            from pathlib import Path
            from tempfile import TemporaryDirectory
            import django
            django.setup()
            from django.contrib.staticfiles.storage import staticfiles_storage
            from django.core.management import call_command
            from django.test import Client, override_settings

            with TemporaryDirectory() as static_root:
                with override_settings(STATIC_ROOT=static_root):
                    call_command('collectstatic', interactive=False, verbosity=0)
                    css_url = staticfiles_storage.url('dashboard/styles.css')
                    plotly_url = staticfiles_storage.url('plotly/plotly.min.js')
                    client = Client(HTTP_HOST='localhost')
                    response = client.get(css_url, secure=True, HTTP_ACCEPT_ENCODING='gzip')
                    content = b''.join(response.streaming_content)
                    encoding = response.get('Content-Encoding')
                    if encoding == 'gzip':
                        content = gzip.decompress(content)
                    redirect = client.get(css_url)
                    forwarded = client.get(css_url, HTTP_X_FORWARDED_PROTO='https')
                    plotly = client.get(plotly_url, secure=True)
                    print(json.dumps({
                        'url': css_url,
                        'status': response.status_code,
                        'content_type': response.get('Content-Type'),
                        'encoding': encoding,
                        'cache_control': response.get('Cache-Control', ''),
                        'content_matches': content == Path('dashboard/static/dashboard/styles.css').read_bytes(),
                        'redirect_status': redirect.status_code,
                        'redirect_url': redirect.get('Location'),
                        'forwarded_status': forwarded.status_code,
                        'plotly_url': plotly_url,
                        'plotly_status': plotly.status_code,
                        'plotly_content_type': plotly.get('Content-Type'),
                        'plotly_cache_control': plotly.get('Cache-Control', ''),
                    }))
                    response.close()
                    forwarded.close()
                    plotly.close()
        """)
        for trust_proxy in (False, True):
            with self.subTest(trust_proxy=trust_proxy):
                environment = {
                    **os.environ,
                    "DJANGO_SETTINGS_MODULE": "churn_dashboard.settings",
                    "DJANGO_DEBUG": "false",
                    "DJANGO_SECRET_KEY": "test-only-deployment-secret-with-more-than-fifty-characters-123456789",
                    "DJANGO_ALLOWED_HOSTS": "localhost",
                    "DJANGO_TRUST_PROXY_HEADERS": str(trust_proxy).lower(),
                }
                result = subprocess.run(
                    [sys.executable, "-c", script],
                    cwd=Path(__file__).resolve().parents[1],
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                response = json.loads(result.stdout)
                self.assertEqual(response["status"], 200)
                self.assertRegex(response["url"], r"/static/dashboard/styles\.[a-f0-9]+\.css$")
                self.assertIn("text/css", response["content_type"])
                self.assertEqual(response["encoding"], "gzip")
                self.assertIn("immutable", response["cache_control"])
                self.assertTrue(response["content_matches"])
                self.assertEqual(response["redirect_status"], 301)
                self.assertEqual(response["redirect_url"], "https://localhost" + response["url"])
                self.assertEqual(response["forwarded_status"], 200 if trust_proxy else 301)
                self.assertRegex(response["plotly_url"], r"/static/plotly/plotly\.min\.[a-f0-9]+\.js$")
                self.assertEqual(response["plotly_status"], 200)
                self.assertIn("javascript", response["plotly_content_type"])
                self.assertIn("immutable", response["plotly_cache_control"])
