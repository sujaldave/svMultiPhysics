"""CPU-only checks of the Marlowe binding validation and scaling summary."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import check_binding
import summarize_marlowe


def write_probe(directory, ranks, hosts, numa_local=True):
    directory.mkdir(parents=True)
    for rank in range(ranks):
        host = hosts[rank*len(hosts)//ranks]
        (directory/f'rank-{rank}.json').write_text(json.dumps(dict(
            rank=rank, hostname=host, visible_count=1, pci_bus_id=f'0000:{rank % 8:02x}:00.0',
            affinity_within_gpu_numa=numa_local)))


def write_runs(variant, gpus, wall, cg_s, repeat=2):
    for index in range(repeat):
        run = variant/f'gpus-{gpus}'/f'run-{index+1}'
        run.mkdir(parents=True)
        (run/'solver.xml').write_text('<case/>')
        (run/'measurement.json').write_text(json.dumps(dict(
            ranks=gpus, steps=20, wall_seconds=wall+index, simulation_seconds=wall-2,
            returncode=0, completed=True, nonlinear_iterations=1, timed_out=False,
            preconditioner='schur-amg', assembly_mode='kokkos', schur_reuse='transfers',
            schur_operator='explicit', gpu_aware_mpi='auto', communication=dict(UCX_TLS='self,sm,cuda_copy'),
            assembly_stats=[dict(kind='ASSEMBLY_STATS', execution_space='Cuda', total_s='1')],
            jacobians=[dict(total_s=wall-4, cg_s=cg_s, gmres_s=1.0, cg_iterations=1000, cg_calls=2,
                            cg_failures=0, cg_true_residual_failures=0, ri_success=1, cg_success=1)])))


def make_job(root, nodes, counts, hosts):
    root.mkdir(parents=True)
    (root/'sweep.json').write_text(json.dumps(dict(nodes=nodes, cases=['iliac'], steps=dict(iliac=20, pipe=10))))
    for bind in ['core', 'numa']:
        for gpus in counts:
            write_probe(root/'binding'/bind/f'gpus-{gpus}', gpus, hosts)
            comms = ['single'] if gpus == 1 else ['ucx-best', 'gpuaware-ipc']
            for comm in comms:
                write_runs(root/'iliac'/f'{bind}__{comm}', gpus, wall=40.0/gpus**0.5 + (bind == 'core'), cg_s=2.0)


class MarloweToolsTest(unittest.TestCase):
    def summarize(self, *paths):
        with mock.patch.object(sys, 'argv', ['summarize_marlowe.py', *map(str, paths), '--expect-cuda']), \
             mock.patch.object(summarize_marlowe.importlib.util, 'find_spec', return_value=None), \
             contextlib.redirect_stdout(io.StringIO()):
            summarize_marlowe.main()

    def test_multi_node_and_numa_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            probe = Path(temporary)/'probe'
            write_probe(probe, 16, ['node-a', 'node-b'])
            with self.assertRaisesRegex(RuntimeError, 'one node'):
                check_binding.validate(probe, 16)
            check_binding.validate(probe, 16, multi_node=True, require_numa_local=True)
            remote = Path(temporary)/'remote'
            write_probe(remote, 4, ['node-a'], numa_local=False)
            with self.assertRaisesRegex(RuntimeError, 'NUMA-local'):
                check_binding.validate(remote, 4, require_numa_local=True)

    def test_campaign_combines_one_and_two_node_jobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Path(temporary)
            make_job(campaign/'iliac-nodes1', 1, [1, 4, 8], ['node-a'])
            make_job(campaign/'iliac-nodes2', 2, [16], ['node-a', 'node-b'])
            # A partial attempt set aside after preemption must be ignored.
            (campaign/'iliac-nodes1/iliac/core__ucx-best/gpus-4').rename(
                campaign/'iliac-nodes1/iliac/core__ucx-best/gpus-4.incomplete-1-0-2')
            write_runs(campaign/'iliac-nodes1/iliac/core__ucx-best', 4, wall=21.0, cg_s=2.0)
            self.summarize(campaign)
            report = json.loads((campaign/'iliac-nodes1/marlowe_scaling.json').read_text())
            self.assertEqual(report['failures'], [])
            rows = {(r['binding'], r['communication'], r['gpus']): r for r in report['rows']}
            self.assertEqual(len(rows), 2*(1 + 2*3))
            row = rows[('numa', 'ucx-best', 16)]
            self.assertEqual(row['nodes'], [2])
            self.assertAlmostEqual(row['speedup_same_binding'], 40.5/10.5)
            self.assertAlmostEqual(row['cg_ms_per_iteration'], 2.0)
            self.assertAlmostEqual(rows[('core', 'ucx-best', 4)]['median_wall_s'], 21.5)
            # The fastest 1-GPU run (numa, 40.5 s) is the common reference.
            self.assertAlmostEqual(rows[('core', 'single', 1)]['speedup_vs_best_1gpu'], 40.5/41.5)

    def test_failed_configurations_are_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/'job'
            make_job(root, 1, [1, 4], ['node-a'])
            (root/'failed.txt').write_text(f'{root}/iliac/none__ucx-best/gpus-4\n')
            with self.assertRaises(SystemExit):
                self.summarize(root)
            failures = json.loads((root/'marlowe_scaling.json').read_text())['failures']
            self.assertTrue(any('timed out' in failure for failure in failures))


if __name__ == '__main__':
    unittest.main()
