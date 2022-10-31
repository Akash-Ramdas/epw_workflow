"""
Contains a class object for a task in a workflow
A task will have the following components:
  1) An input file
  2) An output file
  3) A list of commands to execute to complete a task
  4) A way to check if the task is already complete
"""

from pathlib import Path
from typing import Union

FileType = Union[Path, str]

def convertstringtoPath(x: str):
    if "/" in x:
        # Assume that path is specified fully as string
        return Path(x)
    else:
        # Assume file is in current working directory
        return Path.cwd() / x

class task:
    def __init__(self, name: str = "default_task", 
                 input_file: FileType = None, output_file: FileType = None):
        """
        Create a task object
        :param name: Name of task
        :param input_file: Input File -> One File that contains all the necessary information to
        run this task. This can be a list of input file names, but the key idea is one file must be
        the input
        :param output_file: Output File of the task -> Will be used to do things
        """
        if input_file is None:
            self.input_file: Path = Path("{}.in".format(name))
        else:
            if isinstance(input_file, str):
                self.input_file: Path = convertstringtoPath(input_file)
            elif isinstance(input_file, Path):
                self.input_file: Path = input_file
            else:
                #TODO: Error handling through some class
                pass

        if output_file is None:
            self.output_file: Path = Path("{}.in".format(output_file))
        else:
            if isinstance(output_file, str):
                self.output_file: Path = convertstringtoPath(output_file)
            elif isinstance(output_file, Path):
                self.input_file: Path = output_file
            else:
                #TODO: Error handling through some class
                pass

        self.task_status = "setup" # Possible Status: setup, success and failure

    """
    The below methods are to be inherited by the specific task class.
    They should be modified, the default definitions really don't do 
    anything
    """
    def check_task_status(self):
        """
        Meant to update the task_status -> We do a backward propogation to check the
        necessary tasks to be run
        :return: Update the self.task_status variable
        """
        pass

    def read_task_input(self):
        """
        Read the input file accordingly and store commands to run
        :return:
        """
        pass

    def run_task(self):
        """
        Run the task
        :return:
        """
        pass

    def write_task_output(self):
        """
        Write the output file
        :return:
        """
        pass
