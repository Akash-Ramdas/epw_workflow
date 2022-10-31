from pathlib import Path
from sys import prefix
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Poscar
import os


from workflow_utils.task import task, FileType, convertstringtoPath
from qe_utils.input_utils import pw_file
from qe_utils.output_utils import qe_xml
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
        
        sbatch = sbatch_info(jobname="{}_scf".format(self.prefix),  cores_per_node = 56, 
                             n_nodes = n_nodes, mail_id = "akashr@stanford.edu", 
                             partition = "normal", time="1:00:00")
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

class ph_task(task):
    def __init__(self, name: str = "scf", prefix: str="qe_pref", rden: float = 2,
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
        
        sbatch = sbatch_info(jobname="{}_scf".format(self.prefix),  cores_per_node = 56, 
                             n_nodes = n_nodes, mail_id = "akashr@stanford.edu", 
                             partition = "normal", time="1:00:00")
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

