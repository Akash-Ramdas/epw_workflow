# Imported python packages
from networkx import DiGraph
from networkx.drawing.nx_agraph import to_agraph
from typing import List

from .task import task

"""
All workflows are nothing but directed graphs. A list of task objects and dependencies
can be used to construct this WF. This is inspired by qe_sci_luigi -> Check backwards, 
obtain a list of tasks to be run then run forwards. The rerun tags can be used to 
force certain or all tasks to be rerun. There must be a final workflow task, which 
can be used to find out if the wf is complete or not!
"""


class workflow_object:
    def __init__(self, wf_graph: DiGraph = None):
        """
        Can use this if you have already constructed the
        wf graph, recommended to initialize empty graph
        and build from there
        """
        if wf_graph is not None:
            self.wf_graph = wf_graph.copy()
        else:
            self.wf_graph = DiGraph()
        self.wf_status = "setup"
    
    def get_workflow_status(self):
        pass # Check last node task to get if job is complete or not

    def add_task_to_wf(self, task: task, dependencies: List[int] = None):
        """
        Adds a task to the workflow. Tasks are indexed startung from 0.
        dependcies is a list of task indices that the current task that
        is added depends on
        """
        n_nodes = len(self.wf_graph.nodes(data=True))
        self.wf_graph.add_node(n_nodes, task=task)
        if dependencies is None:
            pass
        else:
            for dep in dependencies:
                self.wf_graph.add_edge(dep, n_nodes)

    def get_tasks_tbr(self, node_id: int, rerun: bool = False, rerun_all: bool=False):
        """
        Based on the status of the tasks, get the remaining to be run for a particular
        task (node_id) to be run.
        """
        node_task: task = self.wf_graph.nodes[node_id]["task"]
        if node_task.task_status == "complete" and not rerun:
            tasks_tbr = None
        else:
            tasks_tbr = [node_id]
            in_edges = self.wf_graph.in_edges(nbunch=self.wf_graph.nodes[node_id])
            higher_tasks = []
            for u, v in in_edges:
                print(u)
                higher_tasks += self.get_tasks_tbr(u, rerun=rerun_all, 
                                                   rerun_all=rerun_all)
            tasks_tbr += higher_tasks
        return tasks_tbr


    def run_node_task(self, node_id: int, rerun: bool = False, rerun_all: bool = False):
        """
        run a node task, (if it has dependecies those will run first)
        """
        node_task: task = self.wf_graph.nodes[node_id]["task"]
        if node_task.task_status == "complete" and not rerun:
            pass
        else:
            in_edges = self.wf_graph.in_edges(nbunch=self.wf_graph.nodes[node_id])
            all_deps_ran = True
            for u, v in in_edges:
                self.run_node_task(u, rerun=rerun_all, rerun_all=rerun_all)
                if self.wf_graph.nodes[u]["task"].task_status != "complete":
                    all_deps_ran = False
                    ## Some error
            if all_deps_ran:
                node_task.read_task_input()
                node_task.run_task()
                node_task.write_task_output()

    def run_wf(self):
        """
        All workflows need to have a final task, if your workflow has multiple ends
        then just ensure the last taks collects all their outputs. Running this final
        task will make sure all the individual ones are run
        """
        n_nodes = len(self.wf_graph.nodes())
        self.run_node_task(node_id=n_nodes-1)

    def output_wf_graph(self, out_file_name: str = "wf_graph.png"):
        """
        To help visualize the workflow progress: Can be prettier
        Will always output this in cwd
        """
        for u, d in self.wf_graph.nodes(data=True):
            if d["task"].task_status == "complete":
                self.wf_graph.nodes[u]["color"] = "Green"
            elif d["task"].task_status == "complete":
                self.wf_graph.nodes[u]["color"] = "Red"
            else:
                self.wf_graph.nodes[u]["color"] = "Orange"
        A = to_agraph(self.wf_graph)
        A.layout()
        A.draw(out_file_name)

