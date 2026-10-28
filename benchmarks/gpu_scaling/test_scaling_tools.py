"""CPU-only checks of scaling arithmetic, failure handling, and binding records."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import check_binding
import summarize_scaling


class ScalingToolsTest(unittest.TestCase):
    def make_sweep(self, root):
        manifest = dict(steps=20,repeat=2,gpu_counts=[1,2,4])
        (root/'sweep.json').write_text(json.dumps(manifest))
        for ranks, wall in [(1,40.0),(2,25.0),(4,20.0)]:
            binding = root/'binding'/f'gpus-{ranks}'
            binding.mkdir(parents=True)
            for rank in range(ranks):
                (binding/f'rank-{rank}.json').write_text(json.dumps(dict(
                    rank=rank,hostname='test-node',visible_count=1,pci_bus_id=f'gpu-{rank}')))
            for repeat in range(2):
                run = root/f'gpus-{ranks}'/f'run-{repeat+1}'
                run.mkdir(parents=True)
                (run/'solver.xml').write_text('<case><steps>20</steps></case>')
                data = dict(ranks=ranks,steps=20,wall_seconds=wall,simulation_seconds=wall-4,
                    returncode=0,completed=True,nonlinear_iterations=1,
                    preconditioner='schur-amg',assembly_mode='kokkos',
                    schur_reuse='transfers',schur_operator='explicit',
                    assembly_stats=[dict(kind='ASSEMBLY_STATS',execution_space='Cuda',total_s='1')],
                    jacobians=[dict(total_s=wall-6,cg_iterations=100,cg_calls=2,
                        cg_failures=0,cg_true_residual_failures=0,ri_success=1,cg_success=1,
                        amg_builds=1,amg_refreshes=0)])
                (run/'measurement.json').write_text(json.dumps(data))

    def summarize(self, root):
        with mock.patch.object(sys,'argv',['summarize_scaling.py',str(root),'--expect-cuda']), \
             mock.patch.object(summarize_scaling.importlib.util,'find_spec',return_value=None), \
             contextlib.redirect_stdout(io.StringIO()):
            summarize_scaling.main()

    def test_speedup_efficiency_and_gpu_cost(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);self.make_sweep(root);self.summarize(root)
            report = json.loads((root/'scaling.json').read_text())
            self.assertEqual(report['failures'],[])
            self.assertAlmostEqual(report['groups']['2']['wall_speedup'],1.6)
            self.assertAlmostEqual(report['groups']['2']['wall_efficiency'],0.8)
            self.assertAlmostEqual(report['groups']['4']['wall_efficiency'],0.5)
            self.assertEqual(report['groups']['4']['gpu_seconds'],80.0)
            self.assertIn('pending',report['field_check_status'])

    def test_bad_residual_incomplete_repeats_and_changed_input_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);self.make_sweep(root)
            path = root/'gpus-2/run-1/measurement.json'
            data = json.loads(path.read_text())
            data['jacobians'][0]['cg_true_residual_failures'] = 1
            path.write_text(json.dumps(data))
            (root/'gpus-4/run-2/measurement.json').unlink()
            (root/'gpus-1/run-2/solver.xml').write_text('<different/>')
            with self.assertRaises(SystemExit):
                self.summarize(root)
            failures = json.loads((root/'scaling.json').read_text())['failures']
            self.assertTrue(any('residual' in failure for failure in failures))
            self.assertTrue(any('missing expected repeats' in failure for failure in failures))
            self.assertTrue(any('Input XML differs' in failure for failure in failures))

    def test_duplicate_gpu_or_multiple_nodes_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);self.make_sweep(root)
            path = root/'binding/gpus-2/rank-1.json'
            original = json.loads(path.read_text())
            duplicate = copy.deepcopy(original);duplicate['pci_bus_id'] = 'gpu-0'
            path.write_text(json.dumps(duplicate))
            with self.assertRaisesRegex(RuntimeError,'distinct GPUs'):
                check_binding.validate(path.parent,2)
            original['hostname'] = 'other-node'
            path.write_text(json.dumps(original))
            with self.assertRaisesRegex(RuntimeError,'one node'):
                check_binding.validate(path.parent,2)


if __name__ == '__main__':
    unittest.main()
