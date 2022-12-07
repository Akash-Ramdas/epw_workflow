from pathlib import Path
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Poscar
import os


from workflow_utils.task import task, FileType, convertstringtoPath
from qe_utils.input_utils import pw_file, projwfc_input, wann_file, pw2wannier_input,\
                                 pw2wannier_nl, ph_input, kpoints_card
from qe_utils.output_utils import qe_xml, proj_xml, qe_bandinfo
from slurm_job.sbatch import sbatch_info



class scf_task(task):
    def __init__(self, name: str = "scf", prefix: str="qe_pref", rden: float = 6,
                 pseudo_dir: FileType = "./",  outdir: FileType = "./tmp",  lsoc: bool = True,
                 run_dir: FileType = Path.cwd(), input_file: FileType = None, 
                 output_file: FileType = None):
        super().__init__(name, input_file, output_file)
        
        if isinstance(run_dir, str):
            run_dir: Path = convertstringtoPath(run_dir)
        
        self.cwd = Path.cwd()
        self.run_dir = run_dir
        self.struc = None
        self.prefix = prefix
        self.rden = rden
        self.pseudo_dir = pseudo_dir
        self.outdir = outdir
        self.lsoc = lsoc

    def check_task_status(self):
        if self.output_file.exists():
            self.task_status="complete"
        else:
            self.task_status="setup"

    def run_task(self, qe_bin: str = "/home1/09019/akashr/perturbo_local/q-e-qe-7.0/bin"):
        os.chdir(self.run_dir)
        scf: pw_file = pw_file()
        scf.init_default_scf_file(self.struc, self.prefix, self.rden, self.pseudo_dir, 
                                  self.outdir, self.lsoc)
        scf.write_input_file(fname="scf.in")
        num_k = scf.cards["K_POINTS"].nk_irr
        k_par_choice = min(num_k, 112)
        if k_par_choice == 112:
            n_nodes = k_par_choice*(5**2)//56
            n_tasks = n_nodes*56
            sbatch = sbatch_info(jobname="{}_scf".format(self.prefix),  cores_per_node = 56, 
                             n_nodes = n_nodes, mail_id = "akashr@stanford.edu", 
                             partition = "normal", time="1:00:00")
        else:
            n_nodes = k_par_choice
            n_tasks = n_nodes*25
            sbatch = sbatch_info(jobname="{}_scf".format(self.prefix),  cores_per_node = 25, 
                             n_nodes = n_nodes, mail_id = "akashr@stanford.edu", 
                             partition = "normal", time="1:00:00")
        #n_nodes = k_par_choice*(5**2)//56 + 1 # 3x3 processor for G 
        sbatch.add_shell_commands("ibrun {}/pw.x -nk {} < {} | tee {}".format(qe_bin,
                                                                            k_par_choice,
                                                                            "scf.in",
                                                                            "scf.out"))
                    

        with open("submit_job.sh", "w") as file:
            file.write(sbatch.__str__())
        # sbacth here
        os.system("sbatch submit_job.sh")
        os.chdir(self.cwd)

    def read_task_input(self):
        os.chdir(self.input_file.parent)
        self.struc: Structure = Poscar.from_file(self.input_file.name).structure
        os.chdir(self.cwd)

    def write_task_output(self):
        pass

class electroniccalcs_task(task):
    def __init__(self, name: str = "scf", prefix: str="qe_pref", rden: float = 6,
                 pseudo_dir: FileType = "./",  outdir: FileType = "./tmp",  lsoc: bool = True,
                 run_dir: FileType = Path.cwd(), input_file: FileType = None, 
                 output_file: FileType = None, npts_first_seg: int = 100):
        super().__init__(name, input_file, output_file)
        if isinstance(run_dir, str):
            run_dir: Path = convertstringtoPath(run_dir)
        
        self.cwd = Path.cwd()
        self.run_dir = run_dir
        self.struc = None
        self.prefix = prefix
        self.rden = rden
        self.pseudo_dir = pseudo_dir
        self.outdir = outdir
        self.lsoc = lsoc
        self.npts_first_seg = npts_first_seg

    def check_task_status(self):
        if self.output_file.exists():
            self.task_status="complete"
        else:
            self.task_status="setup"
    
    def run_task(self, qe_bin: str = "/home1/09019/akashr/perturbo_local/q-e-qe-7.0/bin"):
        os.chdir(self.run_dir)
        scfdir = self.run_dir / "scf"
        scfdir.mkdir(exist_ok=True)
        bandsdir = self.run_dir / "bands"
        bandsdir.mkdir(exist_ok=True)
        pwfcdir = self.run_dir / "projwfc"
        pwfcdir.mkdir(exist_ok=True)
        wannscfdir = self.run_dir / "wan_nscf"
        wannscfdir.mkdir(exist_ok=True)
        phdir = self.run_dir / "phonon"
        phdir.mkdir(exist_ok=True)

        os.chdir(scfdir)
        scf: pw_file = pw_file()
        scf.init_default_scf_file(self.struc, self.prefix, self.rden, self.pseudo_dir, 
                                  self.outdir, self.lsoc)
        scf.write_input_file(fname="scf.in")
        scfoutdir = scfdir / self.outdir
  

        os.chdir(bandsdir)
        bands: pw_file = pw_file()
        bands.init_default_bands_file(self.struc, self.prefix, self.npts_first_seg, self.pseudo_dir, 
                                  self.outdir, self.lsoc)
        bands.write_input_file(fname="bands.in")
        pwfc = projwfc_input()
        pwfc.init_projwfc_file(calc_prefix=self.prefix)
        pwfc.namelists["PROJWFC"].add_nl_params({"kresolveddos": True})
        pwfc.write_input_file(fname="pwfc.in")

        os.chdir(pwfcdir)
        nscf_tet: pw_file = pw_file()
        nscf_tet.init_default_dos_file(self.struc, self.prefix, self.rden, self.pseudo_dir, 
                                       self.outdir, self.lsoc)
        nscf_tet.write_input_file(fname="nscftet.in")
        pwfc = projwfc_input()
        pwfc.init_projwfc_file(calc_prefix=self.prefix)
        pwfc.write_input_file(fname="pwfc.in")

        os.chdir(wannscfdir)
        nscf_wan: pw_file = pw_file()
        nscf_wan.init_default_wan_nscf_file(self.struc, self.prefix, self.rden, self.pseudo_dir, 
                                       self.outdir, self.lsoc)
        nscf_wan.write_input_file(fname="wannscf.in")

        os.chdir(phdir)
        ph: ph_input = ph_input()
        ph.init_default(self.prefix, self.struc, self.rden)

        kpts_tmp = kpoints_card()
        kpts_tmp.init_automatic_grid(struc=self.struc, rden=self.rden)
        nk_mesh = np.zeros(3, dtype=int)
        # Ensure mesh is even to construct commensurate k grid later
        for imesh in range(0, 3):
            if kpts_tmp.num_k[imesh] == 1:
                nk_mesh[imesh] = 1
            elif kpts_tmp.num_k[imesh]%2 == 1:
                nk_mesh[imesh] = kpts_tmp.num_k[imesh]+1
            else:
                nk_mesh[imesh] = kpts_tmp.num_k[imesh]
        
        ph.namelists["INPUTPH"].update_nl_params({"nq1":nk_mesh[0]//2,
                                                  "nq2":nk_mesh[1]//2,
                                                  "nq3": nk_mesh[2]//2})
        ph.write_input_file(fname="ph-ref.in")
        
        
        num_k = scf.cards["K_POINTS"].nk_irr
        scf_k_par_choice = min(num_k, 112)

        n_nodes = scf_k_par_choice//14 # 2x2 unit for each k point, uses all processor per node
        scf_k_par_choice = n_nodes*14
        n_tasks = n_nodes*56


        sbatch = sbatch_info(jobname="{}_es".format(self.prefix),  cores_per_node=56, 
                             n_nodes = n_nodes, mail_id = "akashr@stanford.edu", 
                             partition = "normal", time="1:00:00")
        
        phsbatch = sbatch_info(jobname="{}_ph".format(self.prefix),  cores_per_node=56, 
                             n_nodes = n_nodes, mail_id = "akashr@stanford.edu", 
                             partition = "normal", time="1:00:00")
        
        shell_commands = "cd {scfdir}\n"\
                         "ibrun {qebin}/pw.x -nk {scfkpar} < scf.in | tee scf.out\n"\
                         "cd {phdir}\n"\
                         "cp -r {scfoutdir} ./\n"\
                         "ibrun {qebin}/ph.x -nk {nscfkpar} < ph-ref.in | tee ph-gam.out\n"\
                         "cd {bandsdir}\n"\
                         "cp -r {scfoutdir} ./\n"\
                         "ibrun {qebin}/pw.x -nk {nscfkpar} < bands.in | tee bands.out\n"\
                         "ibrun {qebin}/projwfc.x -nk {nscfkpar} < pwfc.in | tee pwfc.out\n"\
                         "cd {pwfcdir}\n"\
                         "cp -r {scfoutdir} ./\n"\
                         "ibrun {qebin}/pw.x -nk {scfkpar} < nscftet.in | tee nscftet.out\n"\
                         "ibrun {qebin}/projwfc.x -nk {scfkpar} < pwfc.in | tee pwfc.out\n"\
                         "cd {wannscfdir}\n"\
                         "cp -r {scfoutdir} ./\n"\
                         "ibrun {qebin}/pw.x -nk {nscfkpar} < wannscf.in | tee wannscf.out\n"\
                         "echo \"Done!\"\n".format(scfdir=scfdir.__str__(),
                                                 qebin=qe_bin, scfkpar=scf_k_par_choice,
                                                 phdir=phdir.__str__(),
                                                 bandsdir= bandsdir.__str__(),
                                                 scfoutdir = scfoutdir.__str__(),
                                                 nscfkpar = n_tasks,
                                                 pwfcdir = pwfcdir.__str__(),
                                                 wannscfdir = wannscfdir.__str__()) 
        
        ph_shell_script_fname= "ph-submit.sh"
        os.chdir(phdir)
        phsbatch.add_shell_commands("ibrun {qebin}/ph.x -nk {nscfkpar} < ph.in | tee ph.out\n".format(qebin=qe_bin,
                                                                                                    nscfkpar=n_tasks))
        with open(ph_shell_script_fname, "w") as file:
            write_str = "NQIR=$(awk 'FNR == 2 {{print}}' {prefix}.dyn0 | awk -F'[^0-9]*' '$0=$2')\n"\
                        "QE_BIN={qebin}\n"\
                        "for ((iq=2; iq<=$NQIR; iq++))\n"\
                        "do\n"\
                        "PHNAME=\"ph_\"$iq\n"\
                        "mkdir -p $PHNAME\n"\
                        "cd $PHNAME\n"\
                        "cp -r ../tmp ./\n"\
                        "awk -v iq=\"$iq\" '{{gsub(/_q = 1/, \"_q = \" iq)}}1' ../ph-ref.in | tee ph.in"\
                        "cat > \"submit_job.sh\" << EOF\n"\
                        "{phsbatch}\n"\
                        "EOF\n"\
                        "sbatch submit_job.sh\n"\
                        "cd .."\
                        "done".format(prefix=self.prefix, qebin=qe_bin, 
                                              phsbatch=phsbatch.__str__())
            file.write(write_str)

        sbatch.add_shell_commands(shell_commands)
        
        os.chdir(self.run_dir)
        with open("submit_job.sh", "w") as file:
            file.write(sbatch.__str__())
        # sbacth here
        os.system("sbatch submit_job.sh")
        os.chdir(self.cwd)
        
    def read_task_input(self):
        os.chdir(self.input_file.parent)
        self.struc: Structure = Poscar.from_file(self.input_file.name).structure
        os.chdir(self.cwd)

class postprocecalcs_task(task):
    def __init__(self, name: str = "scf", prefix: str="qe_pref", rden: float = 6,
                 pseudo_dir: FileType = "./",  outdir: FileType = "./tmp",  lsoc: bool = True,
                 run_dir: FileType = Path.cwd(), input_file: FileType = None, 
                 output_file: FileType = None, npts_first_seg: int = 100):
        super().__init__(name, input_file, output_file)
        if isinstance(run_dir, str):
            run_dir: Path = convertstringtoPath(run_dir)
        
        self.cwd = Path.cwd()
        self.run_dir = run_dir
        self.struc = None
        self.prefix = prefix
        self.rden = rden
        self.pseudo_dir = pseudo_dir
        self.outdir = outdir
        self.lsoc = lsoc
        self.npts_first_seg = npts_first_seg

    def read_task_input(self):
        os.chdir(self.input_file.parent)
        self.struc: Structure = Poscar.from_file(self.input_file.name).structure
        os.chdir(self.cwd)
    
    def check_task_status(self):
        if self.output_file.exists():
            self.task_status="complete"
        else:
            self.task_status="setup"
    
    def run_task(self, qe_bin: str = "/home1/09019/akashr/perturbo_local/q-e-qe-7.0/bin"):
        os.chdir(self.run_dir)
        scfdir = self.run_dir / "scf"
        bandsdir = self.run_dir / "bands"
        pwfcdir = self.run_dir / "projwfc"
        wannscfdir = self.run_dir / "wan_nscf"
        wandir = self.run_dir / "wan"

        wandir.mkdir(exist_ok=True)
        bands_out_dir = bandsdir / "tmp" / "{}.save".format(self.prefix)
        os.chdir(bands_out_dir)
        bandsqeout = qe_bandinfo(xml_file="data-file-schema.xml", label_info_file="../../bpath.kpts")
        os.chdir(bandsdir)
        fig = bandsqeout.plot_bs()
        fig.write_image("bands.png")
        pwfc_out_dir = pwfcdir / "tmp" / "{}.save".format(self.prefix)
        os.chdir(pwfc_out_dir)
        at_proj = proj_xml("atomic_proj.xml")
        os.chdir(pwfcdir)
        mu, sig, nbnd, nwfc, fig = at_proj.get_scdm_params()
        fig.write_image("proj.png")
        print(mu, sig, nbnd, nwfc)

        os.chdir(wandir)

        kpts_tmp = kpoints_card()
        kpts_tmp.init_automatic_grid(struc=self.struc, rden=self.rden)
        nk_mesh = np.zeros(3, dtype=int)
        # Ensure mesh is even to construct commensurate k grid later
        for imesh in range(0, 3):
            if kpts_tmp.num_k[imesh] == 1:
                nk_mesh[imesh] = 1
            elif kpts_tmp.num_k[imesh]%2 == 1:
                nk_mesh[imesh] = kpts_tmp.num_k[imesh]+1
            else:
                nk_mesh[imesh] = kpts_tmp.num_k[imesh]
        
        wan = wann_file(n_wfc=nwfc, n_bnd=nbnd, nk=nk_mesh,
                        struc=self.struc, plot_wfs=False)
        wan.write_file(filename="{}.win".format(self.prefix))

        pw2wan = pw2wannier_input()
        pw2wan_nl = pw2wannier_nl()
        pw2wan_nl.init_default(self.prefix, self.prefix, scdm_mu=mu, scdm_sigma=sig) 
        pw2wan.init_pw2wannier_file(calc_prefix=self.prefix,pw2wannier=pw2wan_nl)
        pw2wan.write_input_file(fname="pw2wan.in")

    



class ph_task(task):
    def __init__(self, name: str = "ph", prefix: str="qe_pref", rden: float = 2,
                 pseudo_dir: FileType = "./",  outdir: FileType = "./tmp", 
                 run_dir: FileType = Path.cwd(), scfdir: FileType ="./tmp",
                 input_file: FileType = None,  output_file: FileType = None):
        super().__init__(name, input_file, output_file)
        
        if isinstance(run_dir, str):
            run_dir: Path = convertstringtoPath(run_dir)
        
        if isinstance(scfdir, str):
            scfdir: Path = convertstringtoPath(scfdir)
        
        self.cwd = Path.cwd()
        self.run_dir = run_dir
        self.scfdir = scfdir
        self.struc = None
        self.prefix = prefix
        self.rden = rden
        self.pseudo_dir = pseudo_dir
        self.outdir = outdir

    def check_task_status(self):
        if self.output_file.exists():
            self.task_status="complete"
        else:
            self.task_status="setup"

    def run_task(self, scf_outdir: str, 
                 qe_bin: str = "/home1/09019/akashr/perturbo_local/q-e-qe-7.0/bin"):
        scfoutdir: Path = convertstringtoPath(scf_outdir)
        os.chdir(self.run_dir)
        ph = ph_input()
        ph.init_default(self.prefix, self.struc, self.rden)
        kpts = kpoints_card()
        kpts.init_automatic_grid(self.struc, self.rden)
        dyn0_fname = "{}.dyn0".format(self.prefix)



    def read_task_input(self):
        os.chdir(self.input_file.parent)
        self.struc: Structure = Poscar.from_file(self.input_file.name).structure
        os.chdir(self.cwd)

    def write_task_output(self):
        pass


class relaxation_electronic_tasks(task):
    def __init__(self, name: str = "relax", prefix: str="qe_pref", rden: float = 3,
                 pseudo_dir: FileType = "./",  outdir: FileType = "./tmp",  lsoc: bool = True,
                 run_dir: FileType = Path.cwd(), input_file: FileType = None, 
                 output_file: FileType = None, npts_first_seg: int = 100):
        super().__init__(name, input_file, output_file)
        if isinstance(run_dir, str):
            run_dir: Path = convertstringtoPath(run_dir)
        
        self.cwd = Path.cwd()
        self.run_dir = run_dir
        self.struc = None
        self.prefix = prefix
        self.rden = rden
        self.pseudo_dir = pseudo_dir
        self.outdir = outdir
        self.lsoc = lsoc
        self.npts_first_seg = npts_first_seg

    def read_task_input(self):
        os.chdir(self.input_file.parent)
        self.struc: Structure = Poscar.from_file(self.input_file.name).structure
        os.chdir(self.cwd)

    def write_task_output(self):
        pass
    
    def run_task(self, qe_bin: str = "/home1/09019/akashr/perturbo_local/q-e-qe-7.0/bin"):
        os.chdir(self.run_dir)
        relaxdir = self.run_dir/ "vc-relax"
        relaxdir.mkdir(exist_ok=True)
        scfdir = self.run_dir / "scf"
        scfdir.mkdir(exist_ok=True)
        bandsdir = self.run_dir / "bands"
        bandsdir.mkdir(exist_ok=True)
        pwfcdir = self.run_dir / "projwfc"
        pwfcdir.mkdir(exist_ok=True)
        wannscfdir = self.run_dir / "wan_nscf"
        wannscfdir.mkdir(exist_ok=True)
        phdir = self.run_dir / "phonon"
        phdir.mkdir(exist_ok=True)

        os.chdir(relaxdir)
        relax: pw_file = pw_file()
        relax.init_default_vcrel_file(self.struc, self.prefix, self.rden, self.pseudo_dir, 
                                  self.outdir, self.lsoc)
        relax.namelists["CONTROL"].update_nl_params({"max_seconds": 7000})
        relax.namelists["SYSTEM"].update_nl_params({"ecutwfc": 100})
        relax.write_input_file(fname="relax.in")
        relaxoutdir = relaxdir / self.outdir

        os.chdir(scfdir)
        scf: pw_file = pw_file()
        scf.init_default_scf_file(self.struc, self.prefix, self.rden, self.pseudo_dir, 
                                  self.outdir, self.lsoc)
        scf.namelists["CONTROL"].update_nl_params({"restart": "restart"})
        scf.write_input_file(fname="scf.in")
        scfoutdir = scfdir / self.outdir
  

        os.chdir(bandsdir)
        bands: pw_file = pw_file()
        bands.init_default_bands_file(self.struc, self.prefix, self.npts_first_seg, self.pseudo_dir, 
                                  self.outdir, self.lsoc)
        bands.write_input_file(fname="bands.in")
        pwfc = projwfc_input()
        pwfc.init_projwfc_file(calc_prefix=self.prefix)
        pwfc.namelists["PROJWFC"].add_nl_params({"kresolveddos": True})
        pwfc.write_input_file(fname="pwfc.in")

        os.chdir(pwfcdir)
        nscf_tet: pw_file = pw_file()
        nscf_tet.init_default_dos_file(self.struc, self.prefix, self.rden, self.pseudo_dir, 
                                       self.outdir, self.lsoc)
        nscf_tet.write_input_file(fname="nscftet.in")
        pwfc = projwfc_input()
        pwfc.init_projwfc_file(calc_prefix=self.prefix)
        pwfc.write_input_file(fname="pwfc.in")

        os.chdir(wannscfdir)
        nscf_wan: pw_file = pw_file()
        nscf_wan.init_default_wan_nscf_file(self.struc, self.prefix, self.rden, self.pseudo_dir, 
                                       self.outdir, self.lsoc)
        nscf_wan.write_input_file(fname="wannscf.in")

        os.chdir(phdir)
        ph: ph_input = ph_input()
        ph.init_default(self.prefix, self.struc, self.rden)

        kpts_tmp = kpoints_card()
        kpts_tmp.init_automatic_grid(struc=self.struc, rden=self.rden)
        nk_mesh = np.zeros(3, dtype=int)
        # Ensure mesh is even to construct commensurate k grid later
        for imesh in range(0, 3):
            if kpts_tmp.num_k[imesh] == 1:
                nk_mesh[imesh] = 1
            elif kpts_tmp.num_k[imesh]%2 == 1:
                nk_mesh[imesh] = kpts_tmp.num_k[imesh]+1
            else:
                nk_mesh[imesh] = kpts_tmp.num_k[imesh]
        
        ph.namelists["INPUTPH"].update_nl_params({"nq1":nk_mesh[0]//2,
                                                  "nq2":nk_mesh[1]//2,
                                                  "nq3": nk_mesh[2]//2})
        ph.write_input_file(fname="ph-ref.in")
        
        
        num_k = scf.cards["K_POINTS"].nk_irr
        scf_k_par_choice = min(num_k, 112)

        n_nodes = max(scf_k_par_choice//1, 3) # 2x2 unit for each k point, uses all processor per node
        scf_k_par_choice = n_nodes*14
        scf_k_par_choice = 1
        n_tasks = n_nodes*56


        sbatch = sbatch_info(jobname="{}_es".format(self.prefix),  cores_per_node=56, 
                             n_nodes = n_nodes, mail_id = "akashr@stanford.edu", 
                             partition = "normal", time="2:00:00")
        
        phsbatch = sbatch_info(jobname="{}_ph".format(self.prefix),  cores_per_node=56, 
                             n_nodes = n_nodes, mail_id = "akashr@stanford.edu", 
                             partition = "normal", time="2:00:00")
        
        shell_commands = "cd {relaxdir}\n"\
                         "ibrun {qebin}/pw.x -nk {nscfkpar} < relax.in | tee relax.out\n"\
                         "cd {scfdir}\n"\
                         "cp -r {relaxoutdir} ./\n"\
                         "ibrun {qebin}/pw.x -nk {scfkpar} < scf.in | tee scf.out\n"\
                         "cd {phdir}\n"\
                         "cp -r {scfoutdir} ./\n"\
                         "#ibrun {qebin}/ph.x -nk {nscfkpar} < ph-ref.in | tee ph-gam.out\n"\
                         "cd {bandsdir}\n"\
                         "cp -r {scfoutdir} ./\n"\
                         "ibrun {qebin}/pw.x -nk {nscfkpar} < bands.in | tee bands.out\n"\
                         "ibrun {qebin}/projwfc.x -nk {nscfkpar} < pwfc.in | tee pwfc.out\n"\
                         "cd {pwfcdir}\n"\
                         "cp -r {scfoutdir} ./\n"\
                         "ibrun {qebin}/pw.x -nk {scfkpar} < nscftet.in | tee nscftet.out\n"\
                         "ibrun {qebin}/projwfc.x -nk {scfkpar} < pwfc.in | tee pwfc.out\n"\
                         "cd {wannscfdir}\n"\
                         "cp -r {scfoutdir} ./\n"\
                         "ibrun {qebin}/pw.x -nk {nscfkpar} < wannscf.in | tee wannscf.out\n"\
                         "echo \"Done!\"\n".format(scfdir=scfdir.__str__(), 
                                                 relaxdir=relaxdir.__str__(), 
                                                 relaxoutdir=relaxoutdir.__str__(),
                                                 qebin=qe_bin, scfkpar=scf_k_par_choice,
                                                 phdir=phdir.__str__(),
                                                 bandsdir= bandsdir.__str__(),
                                                 scfoutdir = scfoutdir.__str__(),
                                                 nscfkpar = n_tasks,
                                                 pwfcdir = pwfcdir.__str__(),
                                                 wannscfdir = wannscfdir.__str__()) 
        
        ph_shell_script_fname= "ph-submit.sh"
        os.chdir(phdir)
        phsbatch.add_shell_commands("ibrun {qebin}/ph.x -nk {nscfkpar} < ph.in | tee ph.out\n".format(qebin=qe_bin,
                                                                                                    nscfkpar=n_tasks))
        with open(ph_shell_script_fname, "w") as file:
            write_str = "NQIR=$(awk 'FNR == 2 {{print}}' {prefix}.dyn0 | awk -F'[^0-9]*' '$0=$2')\n"\
                        "QE_BIN={qebin}\n"\
                        "for ((iq=2; iq<=$NQIR; iq++))\n"\
                        "do\n"\
                        "PHNAME=\"ph_\"$iq\n"\
                        "mkdir -p $PHNAME\n"\
                        "cd $PHNAME\n"\
                        "cp -r ../tmp ./\n"\
                        "awk -v iq=\"$iq\" '{{gsub(/_q = 1/, \"_q = \" iq)}}1' ../ph-ref.in | tee ph.in"\
                        "cat > \"submit_job.sh\" << EOF\n"\
                        "{phsbatch}\n"\
                        "EOF\n"\
                        "sbatch submit_job.sh\n"\
                        "cd .."\
                        "done".format(prefix=self.prefix, qebin=qe_bin, 
                                              phsbatch=phsbatch.__str__())
            file.write(write_str)

        sbatch.add_shell_commands(shell_commands)
        
        os.chdir(self.run_dir)
        with open("submit_job.sh", "w") as file:
            file.write(sbatch.__str__())
        # sbacth here
        os.system("sbatch submit_job.sh")
        os.chdir(self.cwd)
        