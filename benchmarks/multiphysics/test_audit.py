"""Guard against accepting a completed but unconverged simulation as a speedup."""
import json
from pathlib import Path
import tempfile
import unittest
from summarize import audit, summarize
from run_suite import prepare, variants
import xml.etree.ElementTree as ET

class AuditTests(unittest.TestCase):
    def fixture(self, root, residual='1e-8', linear_failure=False):
        (root/'1-procs').mkdir()
        (root/'solver.xml').write_text('<root><Add_equation type="lElas"><Tolerance>1e-5</Tolerance></Add_equation></root>')
        (root/'1-procs/histor.dat').write_text(f' LE 1-2s 1.2e+0 [ -80 {residual} {residual} 1e-6] [ 12 -60 40]\n')
        return dict(steps=1,completed=True,linear_failures=int(linear_failure))
    def test_cap_is_failure_even_without_solver_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); record=self.fixture(root, '1e-2')
            result=audit(root, record)
            self.assertFalse(result['convergence_passed'])
            self.assertEqual(len(result['nonlinear_unconverged']),1)
    def test_linear_failure_and_missing_steps_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); record=self.fixture(root,linear_failure=True)
            self.assertFalse(audit(root,record)['convergence_passed'])
            record.update(linear_failures=0,steps=2)
            self.assertFalse(audit(root,record)['history_complete'])
    def test_converged_and_iteration_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); result=audit(root,self.fixture(root))
            self.assertTrue(result['convergence_passed'])
            self.assertEqual(result['linear_iterations'],12)
    def test_input_matrix_keeps_physics_identical(self):
        for case in ('scalar', 'solid', 'fsi'):
            normalized = []
            with tempfile.TemporaryDirectory() as tmp:
                for index, variant in enumerate(variants(case)):
                    for assembly in ('trilinos', 'fsils'):
                        run = Path(tmp)/f'{index}-{assembly}'
                        run.mkdir()
                        prepare(case, run, variant, assembly, 20)
                        root = ET.parse(run/'solver.xml').getroot()
                        for eq in root.findall('Add_equation'):
                            eq.remove(eq.find('LS'))
                        normalized.append(ET.tostring(root))
                self.assertEqual(len(set(normalized)), 1)
    def test_missing_groups_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'manifest.json').write_text(json.dumps(dict(repeat=2,expected_groups=[['solid','cg-diagonal','trilinos']])))
            self.assertEqual(len(summarize(root)['failures']),1)

if __name__=='__main__':
    unittest.main()
