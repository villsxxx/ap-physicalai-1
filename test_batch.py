"""Tests use temporary fake logs/checkpoints and never start an ML training."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_all as batch


class BatchTests(unittest.TestCase):
    def test_metrics_are_distinct(self):
        self.assertEqual(batch.parse_training('Mean reward: 14.2\nОбучение завершено за 125 с')['mean_reward'], 14.2)
        result = batch.parse_inference('Итого: захватов 10/10, success rate = 8/10, средняя награда 12.3')
        self.assertEqual(result['successes'], 8)
        self.assertEqual(result['grasps'], 10)
        with self.assertRaises(ValueError):
            batch.parse_inference('Итого: захватов 2/2, success rate = 2/2, средняя награда 12.3')

    def test_resume_accounts_for_saved_update(self):
        self.assertEqual(batch.retained_updates({'retained_before': 0, 'start_iteration': 0, 'requested_updates': 1001}, 950), 951)
        self.assertEqual(batch.retained_updates({'retained_before': 951, 'start_iteration': 950, 'requested_updates': 50}, 999), 1001)

    def test_snapshots_have_one_change(self):
        plan = json.loads((batch.ROOT / 'plan.json').read_text(encoding='utf-8'))
        baseline = (batch.ROOT / 'snapshots/baseline/train.py').read_text(encoding='utf-8')
        self.assertIn('torch.manual_seed(args.seed)', baseline)
        self.assertEqual(plan['train_seed'], 7)
        self.assertEqual(plan['eval_seed'], 84)
        for experiment in plan['experiments']:
            source = (batch.ROOT / 'snapshots' / experiment['name'] / 'train.py').read_text(encoding='utf-8')
            if experiment['change']:
                self.assertEqual(source.count('    ' + experiment['change'] + '\n'), 1)
                source = source.replace('    ' + experiment['change'] + '\n', '')
            self.assertEqual(source, baseline)

    def test_nan_recovery_and_skip_completed_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'snapshots/baseline').mkdir(parents=True)
            (root / 'snapshots/baseline/train.py').write_text('# fake test-only script')
            calls = []

            def fake_stream(command, path):
                calls.append(command)
                path.parent.mkdir(parents=True, exist_ok=True)
                if len(calls) == 1:
                    path.write_text('Mean reward: 3.0\nValueError: The rewards contain NaN values.', encoding='utf-8')
                    (path.parent / 'model_950.pt').write_bytes(b'fake, test only')
                    return 1, 10
                path.write_text('Mean reward: 4.0\nОбучение завершено за 2 с', encoding='utf-8')
                (path.parent / 'model_999.pt').write_bytes(b'fake, test only')
                return 0, 2

            def fake_checkpoint(segment):
                number = 950 if len(calls) == 1 else 999
                return root / segment['directory'] / f'model_{number}.pt', number

            experiment = {'name': 'baseline', 'num_envs': 4096}
            plan = {'max_iters': 1001, 'train_seed': 7, 'max_recoveries': 2}
            with patch.object(batch, 'ROOT', root), patch.object(batch, 'stream', fake_stream), patch.object(batch, 'latest_finite', fake_checkpoint), contextlib.redirect_stdout(io.StringIO()):
                state = batch.train(experiment, plan, root / 'state/baseline.json')
                self.assertEqual(state['completed_updates'], 1001)
                self.assertTrue(state['recovered'])
                self.assertFalse(state['reported_time_covers_all_segments'])
                self.assertEqual(calls[1][calls[1].index('--max_iters') + 1], '50')
                with patch.object(batch, 'stream', side_effect=AssertionError('Completed training must not run again')):
                    skipped = batch.train(experiment, plan, root / 'state/baseline.json')
                self.assertEqual(skipped['checkpoint'], state['checkpoint'])


if __name__ == '__main__':
    unittest.main()
