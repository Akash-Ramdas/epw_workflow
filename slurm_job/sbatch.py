class sbatch_info:
    def __init__(self, job_location: str = None, jobname: str = "default_job",
                 cores_per_node: int = 56, n_nodes: int = 1,
                 mail_id: str = "akashr@stanford.edu", partition: str = "normal",
                 time: str = "1:00:00" ,n_array: int = 0, n_simul_tasks: int = 0,
                dependencies: str = None):

        self.job_location = job_location
        self.jobname: str = jobname
        self.cores_per_node: int = cores_per_node
        self.n_nodes: int = n_nodes
        self.mail_id: str = mail_id
        self.partition: str = partition
        self.time = time
        self.sh_cmds: str = ""

        self.job_array_bool: bool = False
        self.sh_cmds_bool: bool = False
        self.loc_specified: bool = False

        self.tot_cpu: int = n_nodes * cores_per_node

        if n_array != 0:
            self.job_array_bool = True
            self.n_array = n_array
            if n_simul_tasks == 0:
                self.n_simul_tasks = 1
            else:
                self.n_simul_tasks = n_simul_tasks

        self.dependencies_bool: bool = False
        if dependencies is not None:
            self.dependencies = dependencies
            self.dependencies_bool = True

        if self.job_location is not None:
            self.loc_specified = True

    def add_shell_commands(self, sh_cmds: str):
        self.sh_cmds_bool = True
        self.sh_cmds = sh_cmds

    
    def __str__(self) -> str:
        if self.job_array_bool:
            sbatch_string = "#!/bin/bash\n" \
                            "#SBATCH -p {}\n" \
                            "#SBATCH --job-name={}\n" \
                            "#SBATCH --output=%x.%A_%a.out\n" \
                            "#SBATCH --error=%x.%j.err\n" \
                            "#SBATCH --time={}\n" \
                            "#SBATCH --nodes={}\n" \
                            "#SBATCH --mail-type=ALL\n" \
                            "#SBATCH --mail-user={}\n" \
                            "#SBATCH --ntasks-per-node={}\n" \
                            "#SBATCH --array=[1-{}]%{}\n".format(self.partition, self.jobname, self.time, self.n_nodes,
                                                                 self.mail_id, self.cores_per_node,
                                                                 self.n_array, self.n_simul_tasks)
        else:
            sbatch_string = "#!/bin/bash\n" \
                            "#SBATCH -p {}\n" \
                            "#SBATCH --job-name={}\n" \
                            "#SBATCH --output=%x.%j.out\n" \
                            "#SBATCH --error=%x.%j.err\n" \
                            "#SBATCH --time={}\n" \
                            "#SBATCH --nodes={}\n" \
                            "#SBATCH --mail-type=ALL\n" \
                            "#SBATCH --mail-user={}\n" \
                            "#SBATCH --ntasks-per-node={}\n".format(self.partition, self.jobname, self.time,
                                                                    self.n_nodes, self.mail_id,
                                                                    self.cores_per_node, )
        if self.dependencies_bool:
            sbatch_string = sbatch_string + "#SBATCH --dependency={}\n".format(self.dependencies)

        if self.loc_specified:
            add_str: str = "cd {}\n".format(self.job_location)
            sbatch_string = sbatch_string + add_str

        if self.sh_cmds_bool:
            add_str: str = "start_t=" \
                           "echo \"Date              = $(date)\"\n" \
                           "echo \"Hostname          = $(hostname -s)\"\n" \
                           "echo \"Working Directory = $(pwd)\"\n" \
                           "echo \"Starting Commands\"\n" \
                           "{}\n" \
                           "echo \"Commands Completed at $(date)\"\n".format(self.sh_cmds)
            sbatch_string = sbatch_string + add_str
        return sbatch_string