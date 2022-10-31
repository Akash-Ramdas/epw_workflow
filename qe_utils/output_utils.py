import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from scipy.special import erfc

from xml.etree.ElementTree import parse as xml_parse
from pymatgen.core.structure import Lattice, Structure
from xml.etree.ElementTree import ElementTree, Element
from plotly.graph_objects import Figure, Scatter
from typing import List

from qe_utils.constants import AUTOEV, BOHR_RADIUS_ANGS, pi, RYTOEV
from qe_utils.input_utils import kpoints_card, qe_input

NDArrayFloat = NDArray[np.float_]
NDArrayInt = NDArray[np.int_]

class qe_xml:
    def __init__(self, xml_file: str):

        self.xml_tree: ElementTree = xml_parse(xml_file)
        self.root: Element = self.xml_tree.getroot()
        find_node = "output/band_structure/fermi_energy"
        self.ef = float(self.root.find(find_node).text)*AUTOEV

        find_node = "output/atomic_structure/cell"
        cell_info = self.root.find(find_node)
        self.lat = np.zeros((3, 3))
        self.lat[0] = np.array(cell_info.find("a1").text.split(), dtype=float)*BOHR_RADIUS_ANGS
        self.lat[1] = np.array(cell_info.find("a2").text.split(), dtype=float)*BOHR_RADIUS_ANGS
        self.lat[2] = np.array(cell_info.find("a3").text.split(), dtype=float)*BOHR_RADIUS_ANGS
        self.lat_obj = Lattice(self.lat)

        find_node = "output/atomic_structure/atomic_positions/atom"
        pos_info = self.root.findall(find_node)
        self.species = []
        self.positions = []
        for pos in pos_info:
            self.species.append(pos.attrib["name"])
            self.positions.append(np.array(pos.text.split(), dtype=float)*BOHR_RADIUS_ANGS)

        self.rl_obj = self.lat_obj.reciprocal_lattice
        self.struc_obj = Structure(lattice=self.lat_obj, species=self.species,
                                   coords=self.positions, coords_are_cartesian=True)

        self.eigen_vals, self.k_points, self.k_weights, self.soc_flag = self.get_eigenvalues()

    def get_eigenvalues(self):
        find_node = "output/atomic_structure"
        alat = float(self.root.find(find_node).attrib["alat"])*BOHR_RADIUS_ANGS

        find_node = "output/band_structure/lsda"
        lsda_flag = self.root.find(find_node).text
        find_node = "output/band_structure/spinorbit"
        soc_flag = self.root.find(find_node).text
        if soc_flag == "true":
            soc_flag = True
        if lsda_flag == "true":
            find_node = "output/band_structure/nbnd_up"
            n_bands = int(self.root.find(find_node).text)
            find_node = "output/band_structure/nbnd_dw"
            n_bands = n_bands + int(self.root.find(find_node).text)
        else:
            find_node = "output/band_structure/nbnd"
            n_bands = int(self.root.find(find_node).text)

        find_node = "output/band_structure/nks"
        n_k = int(self.root.find(find_node).text)

        find_node = "output/band_structure/ks_energies/eigenvalues"
        einfo = self.root.findall(find_node)
        find_node = "output/band_structure/ks_energies/k_point"
        kinfo = self.root.findall(find_node)

        eigen_vals = np.zeros((n_bands, n_k))
        k_weights = np.zeros(n_k)
        k_points = np.zeros((n_k, 3))
        for i_k in range(0, n_k):
            k_points[i_k] = self.rl_obj.get_fractional_coords(np.array(kinfo[i_k].text.split(), dtype=float)*2*pi/alat)
            k_weights[i_k] = float(kinfo[i_k].attrib["weight"])
            eigen_vals[:, i_k] = np.array(einfo[i_k].text.split(), dtype=float)*AUTOEV

        return eigen_vals, k_points, k_weights, soc_flag


class qe_bandinfo:
    def __init__(self, xml_file: str, label_info_file: str = "bpath.kpts"):
        qedat = qe_xml(xml_file)
        self.bands = qedat.eigen_vals
        self.nbnd, self.nk = self.bands.shape
        self.kinfo = qedat.k_points
        self.soc_flag = qedat.soc_flag
        self.kinfo = np.array(self.kinfo)
        self.tic_loc, self.tic_text = self.parse_label_info(label_info_file)
        self.kpath_dist = None

    @staticmethod
    def parse_label_info(label_fname: str):
        with open(label_fname, "r") as file:
            data = file.readlines()
        tic_loc = []
        tic_text = []
        for line in data:
            tmp = line.strip().split()
            tic_loc.append(int(tmp[1]))
            tic_text.append(tmp[0])
        print(tic_loc, tic_text)

        return tic_loc, tic_text

    def plot_bs(self, width=1000, height=600, colors=["RoyalBlue", "Purple"]) -> Figure:
        fig = Figure()

        color = colors[0]
        color_dwn = colors[1]
        if False:
            k_path_dist = np.zeros(self.nk)
            for ik in range(0, self.nk):
                if ik < self.nk//2:
                    k_path_dist[ik] = ik
                else:
                    k_path_dist[ik] = self.nk - ik
        else:
            k_path_dist = np.arange(0, self.nk)
        print(self.nk)
        for i_bnd in range(0, self.nbnd):
            sl = False
            if i_bnd == 0:
                sl = True
            if False:
                fig.add_trace(Scatter(y=self.bands[i_bnd, :self.nk//2], x=k_path_dist[:self.nk//2], 
                                      mode="lines", showlegend=sl, legendgroup="qe_bands", name="qe_bands_up",
                                      line=dict(color=color)))
                #fig.add_trace(Scatter(y=self.bands[i_bnd, self.nk//2:], x=k_path_dist[self.nk//2:], 
                #                      mode="lines", showlegend=sl, legendgroup="qe_bands", name="qe_bands_down",
                #                      line=dict(color=color_dwn, dash="dash")))

            else:
                fig.add_trace(Scatter(y=self.bands[i_bnd], x=k_path_dist, mode="lines", showlegend=sl,
                                    legendgroup="qe_bands", name="qe_bands",
                                    line=dict(color=color)))

        fig.update_xaxes(title="High Symmetry Path", ticktext=self.tic_text, tickmode="array",
                         tickvals=self.tic_loc, tickangle=0)
        fig.update_layout(width=width, height=height, showlegend=True)
        return fig



class proj_xml:
    def __init__(self, fname) -> None:
        xml_tree = xml_parse(fname)
        root = xml_tree.getroot()
        num_bands: int = int(root.find("./HEADER").attrib["NUMBER_OF_BANDS"].strip())
        num_kpts: int = int(root.find("./HEADER").attrib["NUMBER_OF_K-POINTS"].strip())
        nspin: int = int(root.find("./HEADER").attrib["NUMBER_OF_SPIN_COMPONENTS"].strip())
        natomwfc: int = int(root.find("./HEADER").attrib["NUMBER_OF_ATOMIC_WFC"].strip())
        num_electrons: float = float(root.find("./HEADER").attrib["NUMBER_OF_ELECTRONS"].strip())

        projections: NDArrayFloat = np.zeros((num_kpts, natomwfc, num_bands), dtype=complex)
        eigen_energies: NDArrayFloat = np.zeros((num_kpts, num_bands))
        real_projections: NDArrayFloat = np.zeros((num_kpts, natomwfc, num_bands), dtype=float)
        k_points = np.zeros((num_kpts, 3))
        kpt_info_xml = root.findall("./EIGENSTATES/K-POINT")
        proj_info_xml = root.findall("./EIGENSTATES/PROJS")
        e_info_xml = root.findall("./EIGENSTATES/E")
        for i_kpt in range(0, num_kpts):
            kpt_info = kpt_info_xml[i_kpt].text.strip()
            e_info = e_info_xml[i_kpt].text.strip()
            atomwfc_xml_info = proj_info_xml[i_kpt].findall("./ATOMIC_WFC")
            k_points[i_kpt] = np.asarray(kpt_info.split(), dtype=float)
            eigen_energies[i_kpt] = np.asarray(e_info.split(), dtype=float)
            for i_wfc in range(0, natomwfc):
                atomicwfc_info = atomwfc_xml_info[i_wfc].text.strip()
                list_tmp: List[str] = atomicwfc_info.split("\n")
                for i_bnd in range(0, num_bands):
                    tmp_var = np.asarray(list_tmp[i_bnd].split(), dtype=float)
                    projections[i_kpt, i_wfc, i_bnd] = tmp_var[0] + tmp_var[1] * 1j
                    real_projections[i_kpt, i_wfc, i_bnd] = tmp_var[0] * tmp_var[0] + tmp_var[1] * tmp_var[1]

        self.num_bands = num_bands
        self.num_kpts = num_kpts
        self.nspin = nspin
        self.natomwfc = natomwfc
        self.num_electrons = num_electrons

        self.projections = projections
        self.real_projections = real_projections
        self.eigen_energies = eigen_energies

    def plot_info(self):
        y_plot = np.zeros((self.num_kpts, self.num_bands))
        for i_kpt in range(0, self.num_kpts):
            for i_bnd in range(0, self.num_bands):
                for i_wfc in range(0, self.natomwfc):
                    y_plot[i_kpt, i_bnd] = y_plot[i_kpt, i_bnd] + self.real_projections[i_kpt, i_wfc, i_bnd]

        y_plot = y_plot.reshape(self.num_kpts * self.num_bands)
        x_plot = self.eigen_energies.copy().reshape(self.num_kpts * self.num_bands)

        fig = Figure()
        fig.add_trace(Scatter(x=x_plot, y=y_plot, name="mu_sigma", mode="markers"))
        
        return fig
    
    def get_scdm_params(self, return_plot: bool = True):
        def erfc_func(e,mu,sigma):
            return 0.5*erfc((e - mu)/sigma)

        y_plot = np.zeros((self.num_kpts, self.num_bands))
        for i_kpt in range(0, self.num_kpts):
            for i_bnd in range(0, self.num_bands):
                for i_wfc in range(0, self.natomwfc):
                    y_plot[i_kpt, i_bnd] = y_plot[i_kpt, i_bnd] + self.real_projections[i_kpt, i_wfc, i_bnd]

        y_plot = y_plot.reshape(self.num_kpts * self.num_bands)
        x_plot = self.eigen_energies.copy().reshape(self.num_kpts * self.num_bands)

        popt, pcov = curve_fit(erfc_func, x_plot, y_plot, method="dogbox")

        if return_plot:
            fig = Figure()
            x_fit = np.arange(np.min(x_plot), np.max(x_plot), 0.01)
            y_fit = erfc_func(x_fit, popt[0], popt[1])
            fig.add_trace(Scatter(x=x_plot, y=y_plot, name="mu_sigma", mode="markers"))
            fig.add_trace(Scatter(x=x_fit, y=y_fit, name="fit", mode="lines"))
        
        return popt[0]*RYTOEV, popt[1]*RYTOEV, self.num_bands, self.natomwfc, fig


        
