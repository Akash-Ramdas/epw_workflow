from pathlib import Path
import sys
import unittest
import numpy as np
from pymatgen.core.structure import Lattice, Structure
from pymatgen.io.vasp import Poscar
sys.path.extend(["/home1/09019/akashr/python_codes/epw_workflow/"])


from qe_utils.input_utils import ph_input, projwfc_input, pw2wannier_input, pw2wannier_nl, pw_file, wann_file
from qe_utils.output_utils import qe_xml, qe_bandinfo, proj_xml
from workflow_utils.base_workflow import workflow_object
from workflow_utils.qe_basic_tasks import scf_task


class input_utils_test(unittest.TestCase):
    def test_scf_file(self):
        # Testing Symmetry module
        loc = Path("/scratch1/09019/akashr/automate")
        fcg_dir = loc/"alt_fcg"
        fcg_dir.mkdir(exist_ok=True)
        scf_dir = fcg_dir/"scf"
        scf_dir.mkdir(exist_ok=True)
        scf = scf_task(prefix="fcg", input_file="fcg.poscar", output_file="tmp.otest", 
                       run_dir=scf_dir, pseudo_dir="/home1/09019/akashr/pseudos/pseudodojo/fr",
                       lsoc = True)
        scf.read_task_input()
        scf.run_task()
        self.assertTrue(True)



if __name__ == '__main__':
    unittest.main()
