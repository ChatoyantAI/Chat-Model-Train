import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from src.evaluation.roleplay_benchmarks import bootstrap, sources
from scripts.evaluation.run_public_evaluation import main


class BootstrapTests(unittest.TestCase):
    def test_init_preserves_fixed_settings_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'model.json'
            bootstrap.initialize(path, 'my-model', 'local', ['coser', 'characterbench'])
            config = sources.read_json(path)
            self.assertEqual(config['concurrency']['endpoints'], {'local': 16, 'remote': 96})
            self.assertEqual(config['actor']['sampling'], {'temperature': .7, 'max_tokens': 8192})
            self.assertNotIn('simulator', config)
            self.assertEqual(set(config['benchmark_options']), {'coser', 'characterbench'})
            with self.assertRaises(ValueError):
                bootstrap.initialize(path, 'replacement', 'local', ['coser'])
            self.assertEqual(sources.read_json(path), config)

    def test_env_file_is_data_and_preserves_existing_environment(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'FIXED':'kept'}, clear=True):
            path = Path(temp) / 'keys.env'
            path.write_text("FIXED=changed\nEMPTY=\nQUOTED='$(not executed)'\n", encoding='utf-8')
            bootstrap.load_environment(path)
            self.assertEqual(os.environ['FIXED'], 'kept')
            self.assertNotIn('EMPTY', os.environ)
            self.assertEqual(os.environ['QUOTED'], '$(not executed)')

    def test_new_run_prepares_once_and_changed_resume_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / 'model.json'
            bootstrap.initialize(config, 'model', 'local', ['characterbench'])
            directory = Path(temp) / 'run'
            argv = ['run', '--config', str(config), '--run-dir', str(directory)]
            def prepare(value, output):
                sources.write_json(Path(output) / 'manifest.json', {'config': value})
            with patch('scripts.evaluation.run_public_evaluation.bootstrap.check'), \
                 patch('scripts.evaluation.run_public_evaluation.unified.prepare', side_effect=prepare) as prep, \
                 patch('scripts.evaluation.run_public_evaluation.unified.run', return_value={'dry_run':True}) as run, \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(argv), 0)
                self.assertEqual(main(argv), 0)
                self.assertEqual(prep.call_count, 1)
                changed = sources.read_json(config); changed['actor']['model'] = 'different'
                sources.write_json(config, changed)
                self.assertEqual(main(argv), 2)
                self.assertEqual(run.call_count, 2)

    def test_preflight_uses_get_models_once_and_never_posts(self):
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_GET(self):
                requests.append(('GET', self.path))
                body = b'{"data":[{"id":"fixture"}]}'
                self.send_response(200); self.send_header('Content-Length', str(len(body))); self.end_headers()
                self.wfile.write(body)
            def do_POST(self):
                requests.append(('POST', self.path)); self.send_error(500)
        server = ThreadingHTTPServer(('127.0.0.1',0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp)/'config.json'
                bootstrap.initialize(path, 'fixture', 'remote', ['minimax'], judge_model='fixture')
                config = sources.read_json(path)
                for role in (config['actor'],config['judge'],config['simulator']):
                    role.update(base_url_env='BOOT_URL', api_key_env='BOOT_KEY')
                with patch.dict(os.environ, {'BOOT_URL':f'http://127.0.0.1:{server.server_port}/v1','BOOT_KEY':'fixture'}), \
                     patch.object(bootstrap, 'verify_source'), \
                     patch('src.evaluation.roleplay_benchmarks.frozen_backend.load'):
                    result = bootstrap.check(config, check_api=True)
                self.assertEqual(result['api_completions_sent'], 0)
                self.assertEqual(requests, [('GET','/v1/models')])
        finally:
            server.shutdown(); server.server_close(); thread.join()


if __name__ == '__main__':
    unittest.main()
