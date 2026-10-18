from contextlib import redirect_stdout, redirect_stderr
import io
import unittest
from unittest.mock import patch

from src.evaluation.roleplay_benchmarks import cli


class EndpointWaitTests(unittest.TestCase):
    def invoke(self, extra=()):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return cli.main(['run', '--run-dir', '/fixture', '--execute', *extra])

    def test_waits_for_owner_then_runs_with_original_arguments(self):
        busy = RuntimeError('Another evaluation owns this endpoint; run selected benchmarks in one supervisor')
        with patch.object(cli.unified, 'run', side_effect=[busy, {'status': 'completed'}]) as run, \
                patch.object(cli.time, 'sleep') as sleep:
            self.assertEqual(self.invoke(['--wait-for-endpoints']), 0)
            self.assertEqual(run.call_count, 2)
            self.assertEqual(run.call_args_list[0], run.call_args_list[1])
            sleep.assert_called_once_with(5)

    def test_does_not_retry_other_errors(self):
        with patch.object(cli.unified, 'run', side_effect=RuntimeError('corrupt cache')) as run, \
                patch.object(cli.time, 'sleep') as sleep:
            self.assertEqual(self.invoke(['--wait-for-endpoints']), 2)
            self.assertEqual(run.call_count, 1)
            sleep.assert_not_called()

    def test_independent_mode_is_explicit_and_forwarded(self):
        with patch.object(cli.unified, 'run', return_value={'status': 'completed'}) as run:
            self.assertEqual(self.invoke(['--independent-endpoints']), 0)
            run.assert_called_once_with('/fixture', True, None, independent_endpoints=True)


if __name__ == '__main__':
    unittest.main()
