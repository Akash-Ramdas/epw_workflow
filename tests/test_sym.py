import sys, os
import unittest
import numpy as np
from pymatgen.core.structure import Lattice, Structure
from pymatgen.io.vasp import Poscar
sys.path.extend(["/home1/09019/akashr/python_codes/epw_workflow/"])


from qe_utils.input_utils import ph_input, projwfc_input, pw2wannier_input, pw2wannier_nl, pw_file, wann_file
from qe_utils.output_utils import qe_xml, qe_bandinfo, proj_xml

class input_utils_test(unittest.TestCase):
    def test_scf_file(self):
        # Testing Symmetry module
        struc: Structure = Poscar.from_file("fcg.poscar").structure
        prefix = "fcg"
        scf = pw_file()
        scf.init_default_scf_file(struc=struc, prefix=prefix, rden=6, 
                              pseudo_dir="/home1/09019/akashr/pseudos/pseudodojo/fr")
        scf.write_input_file(fname="scf.in")
        scf.init_default_bands_file(struc=struc, prefix=prefix, npts_first_seg=100, 
                              pseudo_dir="/home1/09019/akashr/pseudos/pseudodojo/fr")
        scf.write_input_file(fname="bands.in")
        scf.init_default_dos_file(struc=struc, prefix=prefix, rden=6, 
                              pseudo_dir="/home1/09019/akashr/pseudos/pseudodojo/fr")
        scf.write_input_file(fname="dos.in")
        scf.init_default_wan_nscf_file(struc=struc, prefix=prefix, rden=6, 
                              pseudo_dir="/home1/09019/akashr/pseudos/pseudodojo/fr")
        scf.write_input_file(fname="wan_nscf.in")
       #wan = wann_file(n_wfc=20, n_bnd=47, nk=np.array([19, 19, 19], dtype=int), wan_nscf=scf, 
        #                plot_wfs=False, exclude_str="20-47")
        wan = wann_file(n_wfc=220, n_bnd=86, nk=np.array([12, 12, 12], dtype=int), wan_nscf=scf, 
                        plot_wfs=False, exclude_str="95-220")
        wan.write_file(filename="{}.win".format(prefix))
        pwfc = projwfc_input()
        pwfc.init_projwfc_file(calc_prefix=prefix, fermi_level=19.5210)
        pwfc.write_input_file(fname="profwfc.in")
        pw2wan = pw2wannier_input()
        pw2wan_nl = pw2wannier_nl()
        #pw2wan_nl.init_default(prefix, prefix, scdm_mu=21.14925, scdm_sigma=5.67924) #cu
        pw2wan_nl.init_default(prefix, prefix, scdm_mu=26.55933, scdm_sigma=5.88569) #fcg

        pw2wan.init_pw2wannier_file(calc_prefix=prefix,pw2wannier=pw2wan_nl)
        pw2wan.write_input_file(fname="pw2wan.in")
        
        ph = ph_input()
        ph.init_default(calc_prefix=prefix, struc=struc, rden=2)
        ph.write_input_file(fname="ph.in")
        self.assertTrue(True)
    
    def test_qe_xml(self):
        qexml = qe_xml(xml_file="data-file-schema.xml")
        print(qexml.ef)
        qebands = qe_bandinfo(xml_file="data-file-schema.xml")
        band_minimas = np.min(qexml.eigen_vals, axis =1)
        exclude_list = np.argwhere(band_minimas>qexml.ef+0.5)

        print(band_minimas[20:25])
        print("{}-{}".format(np.min(exclude_list)+1, np.max(exclude_list)+1))
        #fig = qebands.plot_bs()
        cwd = os.getcwd()
        os.chdir("/scratch1/09019/akashr/automate/alt_fcg/projwfc/tmp/fcg.save")
        at_proj = proj_xml("atomic_proj.xml")
        os.chdir(cwd)

        mu, sig, nbnd, nwfc, fig = at_proj.get_scdm_params()
        fig.write_image("proj.png")
        print(mu, sig, nbnd, nwfc)



        self.assertTrue(np.abs(qexml.ef - 16.5659) < 1e-4)




if __name__ == '__main__':
    unittest.main()