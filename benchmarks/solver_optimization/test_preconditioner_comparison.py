"""Check policy selection and honest reporting of classical inner-solve limits."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import summarize

spec = importlib.util.spec_from_file_location('run_benchmark', Path(__file__).parents[1]/'schur_amg/run_benchmark.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class PreconditionerComparisonTest(unittest.TestCase):
    def test_generated_classical_xml_selects_both_policies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);exe = root/'solver';exe.touch()
            output = root/'results'

            def launch(command, cwd, **kwargs):
                results = cwd/'1-procs';results.mkdir()
                (results/'histor.dat').write_text(' NS 1-1s 1.0\n')
                (results/'result_001.vtu').touch()
                return SimpleNamespace(returncode=0)

            with mock.patch.object(sys,'argv',['benchmark','--exe',str(exe),'--output',str(output),
                    '--preconditioner','diagonal','--gmres-preconditioner','resistance','--steps','1']), \
                 mock.patch.object(benchmark.subprocess,'run',side_effect=launch), \
                 mock.patch.object(benchmark.platform,'platform',return_value='test'), \
                 contextlib.redirect_stdout(io.StringIO()):
                benchmark.main()
            xml = ET.parse(output/'run-1/solver.xml')
            self.assertEqual(xml.findtext('.//GMRES_Preconditioner'),'trilinos-resistance')
            self.assertEqual(xml.findtext('.//CG_Preconditioner'),'trilinos-diagonal')
            self.assertEqual(xml.findtext('.//Assembly'),'trilinos')
            self.assertEqual(xml.findtext('.//Add_equation/Tolerance').strip(),'1e-11')
            record = json.loads((output/'run-1/measurement.json').read_text())
            self.assertEqual(record['gmres_preconditioner'],'resistance')

    def make_results(self, root):
        for label, pressure, inner_failures in [('amg-diagonal','schur-amg',0),('classic-resistance','diagonal',4)]:
            run = root/label/'run-1';run.mkdir(parents=True)
            data = dict(assembly_stats=[],jacobians=[dict(total_s=1,cg_failures=inner_failures,
                cg_true_residual_failures=inner_failures,ri_success=1)],preconditioner=pressure,
                returncode=0,completed=True,assembly_mode='kokkos',wall_seconds=2,
                steps=20,ranks=1)
            (run/'measurement.json').write_text(json.dumps(data))

    def summarize(self, root, allow=True):
        argv = ['summary',str(root),'--reference-group','amg-diagonal']
        if allow: argv.append('--allow-diagonal-inner-nonconvergence')
        with mock.patch.object(sys,'argv',argv), \
             mock.patch.object(summarize.importlib.util,'find_spec',return_value=None), \
             contextlib.redirect_stdout(io.StringIO()):
            summarize.main()

    def test_classical_limits_are_visible_warnings_only_when_requested(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);self.make_results(root)
            with self.assertRaises(SystemExit): self.summarize(root,allow=False)
            self.summarize(root)
            report = json.loads((root/'summary.json').read_text())
            self.assertEqual(report['failures'],[])
            self.assertEqual(len(report['warnings']),2)
            self.assertEqual(report['field_check_status'].split(':')[0],'pending')

    def test_amg_inner_failures_and_classical_outer_failures_stay_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);self.make_results(root)
            path = root/'amg-diagonal/run-1/measurement.json'
            data = json.loads(path.read_text());data['jacobians'][0]['cg_failures'] = 1
            path.write_text(json.dumps(data))
            path = root/'classic-resistance/run-1/measurement.json'
            data = json.loads(path.read_text());data['jacobians'][0]['ri_success'] = 0
            path.write_text(json.dumps(data))
            with self.assertRaises(SystemExit): self.summarize(root)
            report = json.loads((root/'summary.json').read_text())
            self.assertEqual(len(report['failures']),2)


if __name__ == '__main__':
    unittest.main()
