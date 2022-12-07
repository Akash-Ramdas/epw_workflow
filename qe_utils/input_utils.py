from typing import List, Dict
from typing import Union
import numpy as np
from pathlib import Path
import os
from numpy.typing import NDArray
from pymatgen.core.structure import Structure, Lattice
from pymatgen.core.periodic_table import Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.kpath import KPathSetyawanCurtarolo
from spglib import get_ir_reciprocal_mesh
from math import ceil

from workflow_utils.task import convertstringtoPath, FileType
from qe_utils.constants import BOHR_RADIUS_ANGS, pi
from qe_utils.readpseudo import upffile

allowed_nl_types = Union[bool, str, int, float]
NDArrayFloat = NDArray[np.float_]
NDArrayInt = NDArray[np.int_]


def getibrav(crys_sys: str, sg_symb: str):
    if crys_sys == "cubic":
        if "P" in sg_symb:
            ibrav = 1
        elif "F" in sg_symb:
            ibrav = 2
        else:
            ibrav = -3
    elif crys_sys == "hexagonal":
        ibrav = 4
    elif crys_sys == "trigonal":
        if "P" in sg_symb:
            ibrav = 4
        else:
            ibrav = 5
    elif crys_sys == "tetragonal":
        if "P" in sg_symb:
            ibrav = 6
        else:
            ibrav = 7
    elif crys_sys == "orthorhombic":
        if "P" in sg_symb:
            ibrav = 8
        elif "C" in sg_symb:
            ibrav = 9
        elif "A" in sg_symb:
            ibrav = 91
        elif "F" in sg_symb:
            ibrav = 10
        else:
            ibrav = 11
    elif crys_sys == "monoclinic":
        if "P" in sg_symb:
            ibrav = 12
        else:
            ibrav = 13
    else:
        ibrav = 14
    return ibrav
            

class qe_input:
    """
    A general class object for all QE inputs. Consists of 2 types of entities namelists and cards
    namelists are the inputs in between & and / in QE input formats. Each namelist is a dictionary
    of its input parameters. Each namelist is stored as a dictionary, and all the namelists are
    stored as a dictionary of these dictionaries.
    For e.g.: {'control': dict_of_control , ...}
    sample dict_of_Namelist_1 = {'prefix': 'test', 'outdir': './tmp/', 'tstress': '.true.', ...}
    Cards are the inputs that are different formats based on the card (e.g.: ATOMIC_SPECIES)
    Hence each card is a Dict[str, object]. The card can be user defined as well. The key to
    ensure compatibility is to ensure card.str() prints the desired output in your qe input file.
    For samples of cards refer <>

    NameListsType = Dict[str, object]
    CardsType = Dict[str, object]
    """

    def __init__(self, namelists=None, cards=None):
        """
        Initializes a qe object given the namelists and cards
        :param namelists: Namelists of the qe_class_object
        :param cards: Cards of the qe_class_object
        """
        if cards is None:
            cards: Dict[str, object] = {}
        if namelists is None:
            namelists: Dict[str, nl_object] = {}
        self.namelists: Dict[str, nl_object] = namelists
        self.cards: Dict[str, object] = cards

    def __str__(self) -> str:
        """
        A string representation of the QE Input object.
        This is just the input file as a string
        :return: QE_Input file contents as string
        """
        # Write namelist information
        namelist_keys: List[str] = list(self.namelists.keys())
        n_nl = len(namelist_keys)
        class_as_str = ""
        for i_nl in range(0, n_nl):
            namelist = self.namelists[namelist_keys[i_nl]]
            class_as_str = class_as_str + namelist.__str__()

        # Write card information
        cards_keys: List[str] = list(self.cards.keys())
        n_c = len(cards_keys)
        for i_nl in range(0, n_c):
            card = self.cards[cards_keys[i_nl]]
            class_as_str = class_as_str + card.__str__()

        return class_as_str

    def write_input_file(self, write_dir: str = "./", fname: str = "default_qe.in"):
        """
        Writes the input file to the file fname in directory write_dir
        :param write_dir: Directory in which to be written, must be created apriori
        :param fname: Filename
        """
        cwd = Path.cwd()
        os.chdir(write_dir)
        with open(fname, "w") as file:
            file.write(self.__str__())
        os.chdir(cwd)

    def run_qe_calc(self, executable: str, input_file_name: str = "default.in",
                    output_file_name: str = "default.out", parallelize_ops: bool = False,
                    parallel_dict: Dict[str, int] = None, auto_parallel: bool = False) -> int:
        self.write_input_file(fname=input_file_name)
        pass #TODO: Setup a good default parallelization scheme


class nl_object:
    """
    An object for general namelists in QE
    """
    def __init__(self, param_dict: Dict[str, allowed_nl_types] = None, 
    nl_key_name="UNNAMED_NAMELIST"):
        """
        Initializes a Dict
        :param param_dict:
        :param nl_key_name:
        """
        if param_dict is None:
            param_dict = {}
        self.nl_dict: Dict[str, object] = param_dict
        self.nl_name: str = "&" + nl_key_name

    def update_nl_params(self, update_dict: Dict[str, allowed_nl_types]):
        """
        Update a namelist with these keys. Only the keys that exist in the namelist
        are updated. See add_nl_params for adding new keys and simulataneously
        updating old ones
        :param update_dict: A dictionary to update the namelist
        """
        nl_keys: List[str] = list(self.nl_dict.keys())
        update_keys = list(update_dict.keys())
        for key in update_keys:
            if key in nl_keys:
                self.nl_dict.update({key: update_dict[key]})

    def add_nl_params(self, add_dict: Dict[str, allowed_nl_types], loverride: bool = False):
        """
        Add keys to a namelist. Use loverride = True to overwrite any existing keys
        :param add_dict: A dictionary to update the namelist
        :param loverride: Boolean to override dictionary
        """
        nl_keys: List[str] = list(self.nl_dict.keys())
        update_keys = list(add_dict.keys())
        for key in update_keys:
            if key not in nl_keys:
                self.nl_dict.update({key: add_dict[key]})
            else:
                if loverride:
                    self.nl_dict.update({key: add_dict[key]})

    def remove_nl_params(self, keys: List[str]):
        """
        Removes keys from namelist
        :param keys: Keys to be removed
        """
        for key in keys:
            self.nl_dict.pop(key)

    def reset_nl_dict_to_empty(self):
        """
        Resets the namelist to an empty one
        :return:
        """
        self.nl_dict = {}.copy()

    def __str__(self) -> str:
        """
        Write namelist to a string in the format it would be in
        a QE Input file.
        :return: String of namelist
        """
        namelist_dict = self.nl_dict.copy()
        n_inputs: int = len(namelist_dict)
        input_keys: List[str] = list(namelist_dict.keys())
        class_as_str = ""
        class_as_str = class_as_str + self.nl_name + "\n"
        for i_input in range(0, n_inputs):
            if type(namelist_dict[input_keys[i_input]]) == bool:
                if namelist_dict[input_keys[i_input]]:
                    tmp_str = " {} = {},\n".format(input_keys[i_input], ".true.")
                else:
                    tmp_str = " {} = {},\n".format(input_keys[i_input], ".false.")
            elif type(namelist_dict[input_keys[i_input]]) == str:
                tmp_str = " {} = \"{}\",\n".format(input_keys[i_input], 
                                                   namelist_dict[input_keys[i_input]])
            else:
                tmp_str = " {} = {},\n".format(input_keys[i_input], 
                                               namelist_dict[input_keys[i_input]])
            class_as_str = class_as_str + tmp_str
        class_as_str = class_as_str + "/\n"
        return class_as_str


"""
List of used namelists with useful defaults
"""


class control_nl(nl_object):
    """
    Object for the control namelist for pw.x
    Mainly used to initialize the namelist
    """
    def __init__(self, param_dict: Dict[str, allowed_nl_types] = None):
        super().__init__(param_dict)
        self.nl_name = "&CONTROL"

    def init_control(self, calc_prefix: str = None, calc_title: str = None, 
                     restart: bool = False, tag: str = "scf",
                     pseudo_dir: FileType = "./",  outdir: FileType = "./tmp",
                     verbose: bool = True, additional_options: Dict[str, allowed_nl_types] = None):
        """
        Function to initialize the control name list
        :param calc_prefix: prefix of calculation
        :param calc_title: title of calculation (no real effect)
        :param restart: whether to start from scratch or restart
        :param tag: type of calculation must be one of allowed_tags
        :param verbose: verbosity of qe output
        :param additional_options: Any other pw.x control tags desired
        """
        self.reset_nl_dict_to_empty()

        if calc_prefix is not None:
            self.nl_dict["prefix"] = calc_prefix
        else:
            self.nl_dict["prefix"] = "default_prefix"
        
        if calc_title is not None:
            self.nl_dict["title"] = calc_title
        else:
            self.nl_dict["title"] = "pw_input_calc"

        if restart:
            self.nl_dict["restart_mode"] = "restart"
        else:
            self.nl_dict["restart_mode"] = "from_scratch"
        if not verbose:
            self.nl_dict["verbosity"] = "low"
        else:
            self.nl_dict["verbosity"] = "high"


        if isinstance(pseudo_dir, str):
            self.nl_dict["pseudo_dir"] = str(convertstringtoPath(pseudo_dir))
        elif isinstance(pseudo_dir, Path):
            self.nl_dict["pseudo_dir"] = str(outdir)
        
        if isinstance(outdir, str):
            self.nl_dict["outdir"] = str(convertstringtoPath(outdir))
        elif isinstance(outdir, Path):
            self.nl_dict["outdir"] = str(outdir)

        self.nl_dict["max_seconds"] = 4000
        self.nl_dict["forc_conv_thr"] = 1e-4
        self.nl_dict["etot_conv_thr"] = 1e-5
        self.nl_dict["tprnfor"] = True
        self.nl_dict["tstress"] = True

        if tag == "scf" or tag == "nscf" or tag == "bands":
            self.nl_dict["calculation"] = tag
        elif tag == "bands":
            self.nl_dict["calculation"] = tag
        elif tag == "relax" or tag == "vc-relax":
            self.nl_dict["calculation"] = tag
            self.nl_dict["nstep"] = 50
        elif tag == "md" or tag == "vc-md":
            self.nl_dict["calculation"] = tag
            self.nl_dict["nstep"] = 50
            self.nl_dict["dt"] = 20.0  # About 1 fs

        if additional_options is not None:
            self.nl_dict.update(additional_options)


"""
Convenience class for system namelist:
Ideally should have a class that combines this
and the other structure related cards
"""
class system_nl(nl_object):
    """
    Object for the system namelist for pw.x
    Mainly used to initialize the namelist
    """

    def __init__(self, param_dict: Dict[str, allowed_nl_types] = None):
        super().__init__(param_dict)
        self.nl_name = "&SYSTEM"

    
    def init_system(self, struc: Structure, pseudo_dir: FileType = "./",
                    psp_fnames: List[str] = None, symtol: float = 1e-2,
                    occupations: str = "smearing", smearing : str ="gauss", 
                    degauss: float = 0.01, lspinpol: bool = False,
                    lsoc: bool = False):
        
        # Step 1: Structure Information
        atomic_nums: List[int] = list(struc.atomic_numbers)  # All atomic numbers of unit cell
        nat: int = np.size(atomic_nums)  # Number of atoms in unit cell
        ntyp: int = np.size(np.unique(atomic_nums))  # Number of atom types in unit cell
        unique_atom_typ: NDArrayInt = np.unique(atomic_nums)

        self.nl_dict["nat"] = nat
        self.nl_dict["ntyp"] = ntyp

        
        lattice: Lattice = struc.lattice
        self.nl_dict.update({"ibrav": 0})
        """
        Defective: Uncler why? TODO: fix
        sga = SpacegroupAnalyzer(structure=struc, symprec=symtol)
        crys_sys = sga.get_crystal_system()
        sg_symb =sga.get_space_group_symbol()
        ibrav = getibrav(crys_sys, sg_symb)
        self.nl_dict.update({"ibrav": ibrav,
                             "A": lattice.a,
                             "B": lattice.b,
                             "C": lattice.c,
                             "cosBC": np.cos(lattice.alpha * pi / 180),
                             "cosAC": np.cos(lattice.beta * pi / 180),
                             "cosAB": np.cos(lattice.gamma * pi / 180)})

        """
        
        # Step 2: Setup smearing
        self.nl_dict["occupations"] = occupations
        if occupations == "smearing":
            self.nl_dict["smearing"] = smearing
            self.nl_dict["degauss"] = degauss

        # Step 3: Setup for SOC and SP calcs
        
        if lspinpol and not lsoc:
            self.nl_dict["nspin"] = 2
        if lsoc:
            self.nl_dict["lspinorb"] = True
            self.nl_dict["noncolin"] = True
        for ityp in range(0, ntyp):
            if lspinpol or lsoc:
                tmp_key: str = "starting_magnetization({})".format(ityp + 1)
                # TODO: Simple if conditions to initialize this better
                self.nl_dict[tmp_key] = 0.6
        
        # Step 4: Read psp files and get information on cutoff energy and nbnd
        if isinstance(pseudo_dir, str):
            pseudo_loc: Path = convertstringtoPath(pseudo_dir)
        elif isinstance(pseudo_dir, Path):
            pseudo_loc: Path = pseudo_dir
        cwd = Path.cwd()
        os.chdir(pseudo_loc)
        
        ecut_rec = 120
        if psp_fnames is None:
            for ityp in range(0, ntyp):
                ps_name = "{}.upf".format(Element.from_Z(unique_atom_typ[ityp]).symbol)
                upf = upffile(ps_name)
                ecut_rec = max(ecut_rec, upf.info["ecut_rec"])
        

        nelec = 0
        if psp_fnames is None:
            for iat in range(0, nat):
                ps_name = "{}.upf".format(Element.from_Z(atomic_nums[ityp]).symbol)
                nelec = nelec + upf.info["z_valence"]

        
        os.chdir(cwd)

        if lspinpol:
            nbnd: int = nelec
        elif lsoc:
            nbnd: int = 2*nelec
        else:
            nbnd: int = nelec // 2 + 1
        
        if nbnd // 2 < 6:
            extra_bnds = 6
        else:
            extra_bnds = nbnd // 4

        nbnd = nbnd + extra_bnds

        self.nl_dict["nbnd"] = int(nbnd)
        self.nl_dict["ecutwfc"] = ecut_rec

        # TODO: Use upf info to determine wheter ultrasoft or not
        self.nl_dict["ecutrho"] = ecut_rec * 5


class electrons_nl(nl_object):
    """
    Convenience class for electrons namelist
    """
    def __init__(self, param_dict: Dict[str, allowed_nl_types] = None):
        super().__init__(param_dict)
        self.nl_name = "&ELECTRONS"

    def init_default(self, additional_options: Dict[str, allowed_nl_types] = None):
        self.nl_dict["diagonalization"] = "david"
        self.nl_dict["mixing_beta"] = 0.7
        self.nl_dict["mixing_ndim"] = 12

        if additional_options is not None:
            self.nl_dict.update(additional_options)


class ions_nl(nl_object):
    """
    Convenience class for ions namelist
    """
    def __init__(self, param_dict: Dict[str, allowed_nl_types] = None):
        super().__init__(param_dict)
        self.nl_name = "&IONS"

    def init_default(self, additional_options: Dict[str, allowed_nl_types] = None):
        if additional_options is not None:
            self.nl_dict.update(additional_options)


class cell_nl(nl_object):
    """
    Convenience class for cell namelist
    """
    def __init__(self, param_dict: Dict[str, allowed_nl_types] = None):
        super().__init__(param_dict)
        self.nl_name = "&CELL"

    def init_default(self, additional_options: Dict[str, allowed_nl_types] = None):
        if additional_options is not None:
            self.nl_dict.update(additional_options)


class atomic_specie:
    """
    Convenience class to initialize the atomic species card
    Not essential, mainly to replace this class if you
    want to replace the pseudo potential choice. Currently does not
    support Co1, Co2 etc.
    """

    def __init__(self, at_num: int, psp_fname: str = None):
        self.at_num: int = at_num

        self.symbol: str = Element.from_Z(at_num).symbol
        self.at_mass: float = float(Element.from_Z(at_num).atomic_mass)

        if psp_fname is None:
            self.psp_fname: str = self.symbol + ".upf"
        else:
            self.psp_fname: str = psp_fname

        self.specie_dict: Dict[str, object] = dict(atomic_number=self.at_num,
                                                   atomic_symbol=self.symbol,
                                                   atomic_mass=self.at_mass,
                                                   pseudopotential=self.psp_fname)

    def __str__(self) -> str:
        """
        Write each atomic specie in its string qe_input format
        :return: String of atomic specie info
        """
        class_as_str = " {:.{width}}  {}  {}\n".format(self.symbol, self.at_mass, 
                                                       self.psp_fname, width=2)
        return class_as_str


class atomic_species_card:
    """
    Atomic Species Card for QE
    """

    def __init__(self, struc: Structure, psp_fnames: List[str] = None):
        """
        Initializes the ATOMIC_SPECIES card from a pymatgen
        structure object
        :param struc: Structure object of the material
        """
        atomic_nums: List[int] = list(struc.atomic_numbers)  # All atomic numbers of unit cell
        self.ntyp: int = np.size(np.unique(atomic_nums))  # Number of atom types in unit cell
        self.unique_atom_typ: NDArrayInt = np.unique(atomic_nums)
        self.species_list: List[atomic_specie] = []
        self.card_name = "ATOMIC_SPECIES\n"
        for i_typ in range(0, self.ntyp):
            if psp_fnames is None:
                specie = atomic_specie(self.unique_atom_typ[i_typ])
            else:
                specie = atomic_specie(self.unique_atom_typ[i_typ], psp_fnames[i_typ])
            self.species_list.append(specie)

    def __str__(self) -> str:
        """
        Write ATOMIC SPECIES card as string.
        String is same as the QE Input format
        :return: ATOMIC species card string
        """
        class_as_str: str = self.card_name
        for i_typ in range(0, self.ntyp):
            class_as_str = class_as_str + self.species_list[i_typ].__str__()
        return class_as_str


class atomic_positions_card:
    """
    ATOMIC_POSITIONS card for QE
    """

    def __init__(self, struc: Structure, unit_type: str = "crystal", precision: int = 7):
        """
        Intializes the atomic positions cards from a pymatgen structure object
        :param struc: Pymatgen Structure object
        :param unit_type: The type to represent the structure in. (Angstroms, Bohr,
        crystal, etc.) We assume struc.cart_coords is in Angstrom
        :param precision: Precision with which to describe each position.
        """
        self.atomic_nums: List[int] = list(struc.atomic_numbers)  # All atomic numbers of unit cell
        self.nat: int = np.size(self.atomic_nums)  # Number of atoms in unit cell
        self.symbols: List[str] = []
        self.fixed_pos: NDArrayInt = np.ones((self.nat, 3), dtype=int)
        self.atoms_fixed: bool = False
        self.precision: int = precision
        self.unit_type: str = unit_type

        for i_at in range(0, self.nat):
            self.symbols.append(Element.from_Z(self.atomic_nums[i_at]).symbol)

        if unit_type.lower() == "angstrom":
            self.coords = struc.cart_coords
        elif unit_type.lower() == "bohr":
            self.coords = struc.cart_coords / BOHR_RADIUS_ANGS
        elif unit_type.lower() == "alat":
            alat = np.linalg.norm(struc.lattice.matrix[0])
            self.coords = struc.cart_coords / alat
        elif unit_type.lower() == "crystal":
            self.coords = struc.frac_coords

    def fix_atoms(self, atom_indices: NDArrayInt):
        """
        Fix atoms in the strucutre for relax/md
        :param atom_indices: THe indices of the atom to be fixed (0 -> n-1)
        """
        self.atoms_fixed = True
        n_indices = atom_indices.shape[0]
        for i_index in range(0, n_indices):
            self.fixed_pos[atom_indices[i_index]] = np.zeros(3)

    def fix_atoms_in_dir(self, atom_indices: NDArrayInt, desired_dir: int = 3):
        """
        Fix atoms in the strucutre for relax/md in the desired dir
        :param atom_indices: Indices to be fixed in the desired direction (0 -> n-1)
        :param desired_dir: 1,2,3 (x, y, z) or (a_1, a_2, a_3) based on the choice
        of type of card
        :return:
        """
        self.atoms_fixed = True
        n_indices = atom_indices.shape[0]
        for i_index in range(0, n_indices):
            self.fixed_pos[atom_indices[i_index], desired_dir - 1] = 0

    def unfix_all_atoms(self):
        """
        Unfixes all atoms
        """
        self.fixed_pos: NDArrayInt = np.ones((self.nat, 3), dtype=int)
        self.atoms_fixed: bool = False

    def __str__(self) -> str:
        """
        :return: ATOMIC_POSITIONS card as string
        """
        class_as_str = "ATOMIC_POSITIONS {}\n".format(self.unit_type)
        if self.atoms_fixed:
            for i_at in range(0, self.nat):
                class_as_str = class_as_str + " {:.{width}}  {:.{prec}f}" \
                                              "  {:.{prec}f}  {:.{prec}f}" \
                                              "  {}  {}  {}\n".format(self.symbols[i_at],
                                                                      self.coords[i_at, 0],
                                                                      self.coords[i_at, 1],
                                                                      self.coords[i_at, 2],
                                                                      self.fixed_pos[i_at, 0],
                                                                      self.fixed_pos[i_at, 1],
                                                                      self.fixed_pos[i_at, 2],
                                                                      width=2,
                                                                      prec=self.precision
                                                                      )
        else:
            for i_at in range(0, self.nat):
                class_as_str = class_as_str + " {:.{width}}  {:.{prec}f}" \
                                              "  {:.{prec}f}  {:.{prec}f}" \
                                              "\n".format(self.symbols[i_at],
                                                          self.coords[i_at, 0],
                                                          self.coords[i_at, 1],
                                                          self.coords[i_at, 2],
                                                          width=2,
                                                          prec=self.precision
                                                          )
        return class_as_str


"""
class symmetry_info:
    def __init__(self, lattice: Lattice) -> None:
        # Adapted from PW/src/symm_base.f90
        sin3  =  0.866025403784438597
        cos3  =  0.50
        msin3 = -0.866025403784438597
        mcos3 = -0.50

        self.s0 = np.zeros((9,32))
        self.s0_names = []

        self.s0[:, 0] = np.array([1.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  1.0])
        self.s0_names.append("identity")

        self.s0[:, 1] = np.array([-1.0,  0.0,  0.0,  0.0,  -1.0,  0.0,  0.0,  0.0,  1.0])
        self.s0_names.append("180 deg rotation - cart. axis [0,0,1]")

        self.s0[:, 2] = np.array([-1.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  -1.0])
        self.s0_names.append("180 deg rotation - cart. axis [0,1,0]")

        self.s0[:, 3] = np.array([1.0,  0.0,  0.0,  0.0,  -1.0,  0.0,  0.0,  0.0,  -1.0])
        self.s0_names.append("180 deg rotation - cart. axis [1,0,0]")

        self.s0[:, 4] = np.array([0.0,  1.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  -1.0])
        self.s0_names.append("180 deg rotation - cart. axis [1,1,0]")

        self.s0[:, 5] = np.array([0.0,  -1.0,  0.0,  0.0,  -1.0,  0.0,  0.0,  0.0,  -1.0])
        self.s0_names.append("180 deg rotation - cart. axis [1,-1,0]")
        
        self.s0[:, 6] = np.array([0.0,  -1.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  1.0])
        self.s0_names.append("90 deg rotation - cart. axis [0,0,-1]")

        self.s0[:, 7] = np.array([0.0,  1.0,  0.0,  -1.0,  0.0,  0.0,  0.0,  0.0,  1.0])
        self.s0_names.append("90 deg rotation - cart. axis [0,0,1]")

        self.s0[:, 8] = np.array([0.0,  0.0,  1.0,  0.0,  -1.0,  0.0,  1.0,  0.0,  0.0])
        self.s0_names.append("180 deg rotation - cart. axis [1,0,1]")

        self.s0[:, 9] = np.array([0.0,  0.0,  -1.0,  0.0,  -1.0,  0.0,  -1.0,  0.0,  0.0])
        self.s0_names.append("180 deg rotation - cart. axis [-1,0,1]")

        self.s0[:, 10] = np.array([0.0,  0.0,  -1.0,  0.0,  1.0,  0.0,  1.0,  0.0,  0.0])
        self.s0_names.append("90 deg rotation - cart. axis [0,1,0]")

        self.s0[:, 11] = np.array([0.0,  0.0,  1.0,  0.0,  1.0,  0.0,  -1.0,  0.0,  0.0])
        self.s0_names.append("90 deg rotation - cart. axis [0,-1,0]")

        self.s0[:, 12] = np.array([-1.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  1.0,  0.0])
        self.s0_names.append("180 deg rotation - cart. axis [0,1,1]")

        self.s0[:, 13] = np.array([-1.0,  0.0,  0.0,  0.0,  0.0,  -1.0,  0.0,  -1.0,  0.0])
        self.s0_names.append("180 deg rotation - cart. axis [0,1,-1]")

        self.s0[:, 14] = np.array([1.0,  0.0,  0.0,  0.0,  0.0,  -1.0,  0.0,  1.0,  0.0])
        self.s0_names.append("90 deg rotation - cart. axis [-1,0,0]")

        self.s0[:, 15] = np.array([1.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  -1.0,  0.0])
        self.s0_names.append("90 deg rotation - cart. axis [1,0,0]")

        self.s0[:, 16] = np.array([0.0,  0.0,  1.0,  1.0,  0.0,  0.0,  0.0,  1.0,  0.0])
        self.s0_names.append("120 deg rotation - cart. axis [-1,-1,-1]")

        self.s0[:, 17] = np.array([0.0,  0.0,  -1.0,  -1.0,  0.0,  0.0,  0.0,  1.0,  0.0])
        self.s0_names.append("120 deg rotation - cart. axis [-1,1,1]")

        self.s0[:, 18] = np.array([0.0,  0.0,  -1.0,  1.0,  0.0,  0.0,  0.0,  -1.0,  0.0])
        self.s0_names.append("120 deg rotation - cart. axis [1,1,-1]")

        self.s0[:, 19] = np.array([0.0,  0.0,  1.0,  -1.0,  0.0,  0.0,  0.0,  -1.0,  0.0])
        self.s0_names.append("120 deg rotation - cart. axis [1,-1,1]")

        self.s0[:, 20] = np.array([0.0,  1.0,  0.0,  0.0,  0.0,  1.0,  1.0,  0.0,  0.0])
        self.s0_names.append("120 deg rotation - cart. axis [1,1,1]")

        self.s0[:, 21] = np.array([0.0,  -1.0,  0.0,  0.0,  0.0, -1.0,  1.0,  0.0,  0.0])
        self.s0_names.append("120 deg rotation - cart. axis [-1,1,-1]")

        self.s0[:, 22] = np.array([0.0,  -1.0,  0.0,  0.0,  0.0, 1.0,  -1.0,  0.0,  0.0])
        self.s0_names.append("120 deg rotation - cart. axis [1,-1,-1]")

        self.s0[:, 23] = np.array([0.0,  -1.0,  0.0,  0.0,  0.0, -1.0,  -1.0,  0.0,  0.0])
        self.s0_names.append("120 deg rotation - cart. axis [-1,-1,1]")

        self.s0[:, 24] = np.array([cos3,  sin3,  0.0,  msin3,  cos3, 0.0, 0.0,  0.0,  1.0])
        self.s0_names.append("60 deg rotation - cryst. axis [0,0,1]")

        self.s0[:, 25] = np.array([cos3,  msin3,  0.0,  sin3,  cos3, 0.0, 0.0,  0.0,  1.0])
        self.s0_names.append("60 deg rotation - cryst. axis [0,0,-1]")

        self.s0[:, 26] = np.array([mcos3,  sin3,  0.0,  msin3,  mcos3, 0.0, 0.0,  0.0,  1.0])
        self.s0_names.append("120 deg rotation - cryst. axis [0,0,1]")

        self.s0[:, 27] = np.array([mcos3,  msin3,  0.0,  sin3,  mcos3, 0.0, 0.0,  0.0,  1.0])
        self.s0_names.append("120 deg rotation - cryst. axis [0,0,-1]")

        self.s0[:, 28] = np.array([cos3,  msin3,  0.0,  msin3,  mcos3, 0.0, 0.0,  0.0,  -1.0])
        self.s0_names.append("180 deg rotation - cryst. axis [1,-1,0]")

        self.s0[:, 29] = np.array([cos3,  sin3,  0.0,  sin3,  mcos3, 0.0, 0.0,  0.0,  -1.0])
        self.s0_names.append("180 deg rotation - cryst. axis [2,1,0]")

        self.s0[:, 30] = np.array([mcos3, msin3,  0.0,  msin3,  cos3, 0.0, 0.0,  0.0,  -1.0])
        self.s0_names.append("180 deg rotation - cryst. axis [0,1,0]")

        self.s0[:, 31] = np.array([mcos3, sin3,  0.0,  sin3,  cos3, 0.0, 0.0,  0.0,  -1.0])
        self.s0_names.append("180 deg rotation - cryst. axis [1,1,0]")

        self.s0 = self.s0.reshape((3, 3, 32))

        self.s = np.zeros((3, 3, 48))
        self.snames = []

        overlap = np.zeros((3,3))
        rot = np.zeros((3,3))
        rat = np.zeros(3)
        at = lattice.matrix

        for i in range(0, 3):
            for j in range(0, 3):
                rot[j, i] = at[0, j]* at[0, i] + at[1, j]* at[1, i] + at[2, j]* at[2, i]
        
        nrot = 0
        overlap = np.linalg.pinv(rot)

        for irot in range(0, 32):
            for i in range(0, 3):
                for j in range(0, 3):
                    rat[j] = self.s0[j, 0, irot] * at[0, i] + \
                             self.s0[j, 1, irot] * at[1, i] + \
                             self.s0[j, 2, irot] * at[2, i]
                for j in range(0, 3):
                    rot[j, i] = at[0, j]*rat[0] + at[1, j]*rat[1] + at[2, j]*rat[2]
            rot_found = True
            for i in range(0, 3):
                for j in range(0, 3):
                    val = overlap[i, 0] * rot[0, j] + \
                          overlap[i, 1] * rot[1, j] + \
                          overlap[i, 2] * rot[2, j] 
                    if (np.abs(int(val)) - val > 1e-6):
                        rot_found = False
                   
            if rot_found:
                self.s[j, i, nrot] = int(val)
                self.snames.append(self.s0_names[irot])
                nrot += 1
        print(nrot)
        if nrot not in [1, 2, 4, 6, 8, 12, 24]:
            print("Weird Lattice - Forcing no symmetries")
            nrot = 1
        
        for irot in range(0, nrot):
            self.snames.append("Inversion + {}".format(self.s0_names[irot]))
            for i in range(0, 3):
                for j in range(0, 3):
                    self.s[i, j, irot+nrot] = -self.s[i, j, irot]
"""


class kpoints_card:
    def __init__(self):
        self.grid_type = None
        self.kpoints_array = None
        self.nk = 0
        self.nk_irr = 0
        self.num_k = None
        self.shift_k = None
        self.symm_mapping = None
        self.structure = None


    def init_automatic_grid(self, struc: Structure, rden: float = 4, shift: bool = False):
        self.grid_type="automatic"
        self.structure = struc
        rl = struc.lattice.reciprocal_lattice
        nk1: int = max(ceil(rden*rl.a), 1)
        nk2: int = max(ceil(rden*rl.b), 1)
        nk3: int = max(ceil(rden*rl.c), 1)
        if shift:
            s1: int = 1
            s2: int = 1
            s3: int = 1
        else:
            s1: int = 0
            s2: int = 0
            s3: int = 0
        mesh = [nk1, nk2, nk3]
        cell = (struc.lattice.matrix.copy(), struc.frac_coords, struc.atomic_numbers)
        self.symm_mapping, self.kpoints_array = get_ir_reciprocal_mesh(mesh, 
                                               cell, is_shift=[s1, s2, s3])
        self.nk_irr = len(np.unique(self.symm_mapping))
        self.nk = nk1*nk2*nk3
        self.num_k = [nk1, nk2, nk3]
        self.shift_k = [s1, s2, s3]


    def init_gamma_point(self):
        self.grid_type="gamma"
        self.nk = 1
        self.nk_irr = 1
    
    def init_klist(self, kpoints_array: NDArrayFloat, struc: Structure = None, 
                   k_weights: NDArrayFloat = None, grid_type: str = "crystal"):
        # Assumes that as a refined individual you'll by default specify kpoints in 
        # crystal coordinates
        self.grid_type = grid_type
        self.kpoints_array = kpoints_array
        self.nk, _ = kpoints_array.shape
        self.nk_irr = self.nk
        self.structure = struc
        if k_weights is None:
            # Asign equal weights
            self.k_weights = np.ones(self.nk)/self.nk
        else:
            self.k_weights = k_weights
    
    def init_bpath_crystal(self, npts_first_seg: int, struc: Structure):
        self.structure = struc
        kpath_obj: KPathSetyawanCurtarolo = KPathSetyawanCurtarolo(structure=struc)
        high_sym_k_info = kpath_obj.kpath["kpoints"]
        high_sym_path_info: List[List[str]] = kpath_obj.kpath["path"]

        n_continous_segments = len(high_sym_path_info)
        k_dist = 0
        index_kpt = 0
        kpt_array =[]
        kpt_file_str: str = "" 

        for i_cont_segment in range(0, n_continous_segments):
            cont_segment = high_sym_path_info[i_cont_segment]
            n_segments = len(cont_segment)
            for i_segment in range(0, n_segments - 1):
                kpt_start_frac = high_sym_k_info[cont_segment[i_segment]]
                kpt_end_frac = high_sym_k_info[cont_segment[i_segment + 1]]
                path_length: float = np.linalg.norm(kpt_end_frac - kpt_start_frac)
                path_dir: NDArrayFloat = (kpt_end_frac - kpt_start_frac)
                path_dir = path_dir / path_length
                if i_segment == 0 and i_cont_segment == 0:
                    n_path_pts = npts_first_seg + 1
                    rden = int(n_path_pts/path_length)
                n_path_pts: int = int(np.floor(path_length * rden))
                path_step: NDArrayFloat = path_dir * path_length / (n_path_pts-1)
                dist_step = path_length/(n_path_pts - 1)
                for i_path_pts in range(0, n_path_pts):
                    
                    tmp_kpt = kpt_start_frac + path_step * i_path_pts
                    kpt_array.append(tmp_kpt)
                    if i_path_pts == 0:
                        kpt_file_str += "{} {} {:.{prec}f} {:.{prec}f} {:.{prec}f} {:.{prec}f}\n".format(cont_segment[i_segment],
                                                       index_kpt+i_path_pts, 
                                                       k_dist+(i_path_pts)*dist_step,
                                                       tmp_kpt[0], tmp_kpt[1], tmp_kpt[2],
                                                       prec=5)
                    if i_segment == n_segments - 2:
                        if i_path_pts == n_path_pts-1:
                            kpt_file_str += "{} {} {:.{prec}f} {:.{prec}f} {:.{prec}f} {:.{prec}f}\n".format(cont_segment[i_segment + 1],
                                                        index_kpt+i_path_pts, 
                                                        k_dist+(i_path_pts)*dist_step,
                                                        tmp_kpt[0], tmp_kpt[1], tmp_kpt[2],
                                                        prec=5)

                        #kpt_array.append(kpt_end_frac)
                k_dist = k_dist + path_length
                index_kpt += n_path_pts
        self.kpoints_array = np.array(kpt_array)
        self.nk = self.kpoints_array.shape[0]
        self.k_weights = np.ones(self.nk)/self.nk
        self.grid_type = "crystal"
        
        with open("bpath.kpts", "w") as file:
            file.write(kpt_file_str)
                  

    """
    Replaced
    def init_bpath_crystal(self, rden: float, struc: Structure):
        kpath_obj: KPathSetyawanCurtarolo = KPathSetyawanCurtarolo(structure=struc)
        high_sym_k_info = kpath_obj.kpath["kpoints"]
        high_sym_path_info = kpath_obj.kpath["path"]
        print(high_sym_path_info)
        n_continous_segments = len(high_sym_path_info)
        kpt_array = []
        kpt_file_str = ""
        n_pts = 0
        lengths_for_file = 0
        for i_cont_segment in range(0, n_continous_segments):
            cont_segment = high_sym_path_info[i_cont_segment]
            n_segments = len(cont_segment)
            for i_segment in range(0, n_segments - 1):
                kpt_start_frac = high_sym_k_info[cont_segment[i_segment]]
                kpt_end_frac = high_sym_k_info[cont_segment[i_segment + 1]]
                path_length: float = np.linalg.norm(kpt_end_frac - kpt_start_frac)
                path_dir: NDArrayFloat = (kpt_end_frac - kpt_start_frac)
                path_dir = path_dir / path_length
                if i_segment == 0 and i_cont_segment == 0:
                    n_path_pts = rden
                    rden = int(n_path_pts/path_length)
                n_path_pts: int = int(np.floor(path_length * rden))
                path_step: NDArrayFloat = path_dir * path_length / n_path_pts
                for i_path_pts in range(0, n_path_pts):
                    if i_path_pts == 0:
                        kpt_file_str += "{} {} {:.{prec}f}\n".format(cont_segment[i_segment],
                                                       n_pts+i_path_pts, 
                                                       lengths_for_file+(i_path_pts+1)*path_length/n_path_pts,
                                                       prec=5)
                    if i_path_pts == n_path_pts - 1:
                        kpt_file_str += "{} {} {:.{prec}f}\n".format(cont_segment[i_segment + 1],
                                                       n_pts+i_path_pts, 
                                                       lengths_for_file+(i_path_pts+1)*np.linalg.norm(path_step),
                                                       prec=5)
                    tmp_kpt = kpt_start_frac + path_step * i_path_pts
                    kpt_array.append(tmp_kpt)
                kpt_array.append(kpt_end_frac)
                n_pts = n_pts + n_path_pts
                lengths_for_file = lengths_for_file + path_length
        self.kpoint_array = np.array(kpt_array)
        self.nk = self.kpoint_array.shape[0]
        self.k_weights = np.ones(self.nk)/self.nk
        self.grid_type = "crystal"
        with open("bpath.kpts", "w") as file:
            file.write(kpt_file_str)
    """

    def init_bpath_wannier_str(self, rden: float, struc: Structure) -> str:
        kpath_obj: KPathSetyawanCurtarolo = KPathSetyawanCurtarolo(structure=struc)
        high_sym_k_info = kpath_obj.kpath["kpoints"]
        high_sym_path_info = kpath_obj.kpath["path"]
        n_continous_segments = len(high_sym_path_info)
        class_as_str = "begin kpoint_path\n"
        for i_cont_segment in range(0, n_continous_segments):
            cont_segment = high_sym_path_info[i_cont_segment]
            n_segments = len(cont_segment)
            for i_segment in range(0, n_segments - 1):
                kpt_start_frac = high_sym_k_info[cont_segment[i_segment]]
                kpt_end_frac = high_sym_k_info[cont_segment[i_segment + 1]]
                class_as_str = class_as_str + "{} {:.{prec}f} {:.{prec}f}  {:.{prec}f} " \
                                              "{} {:.{prec}f} {:.{prec}f}  {:.{prec}f}\n".format(
                    cont_segment[i_segment],
                    kpt_start_frac[0],
                    kpt_start_frac[1],
                    kpt_start_frac[2],
                    cont_segment[i_segment + 1],
                    kpt_end_frac[0],
                    kpt_end_frac[1],
                    kpt_end_frac[2],
                    prec=5)
        class_as_str = class_as_str + "end kpoint_path\n"
        return class_as_str


    def __str__(self) -> str:
        """
        Write KPOINTS card as input string
        :return: KPOINTS card as string
        """
        
        class_as_str = "K_POINTS {}\n".format(self.grid_type)
        if self.grid_type == "gamma":
            class_as_str = class_as_str + "\n"
        elif self.grid_type == "automatic":
            class_as_str = class_as_str + " {} {} {} {} {} {}\n".format(self.num_k[0], self.num_k[1],
                                                                        self.num_k[2], self.shift_k[0],
                                                                        self.shift_k[1],
                                                                        self.shift_k[2])
        else:
            class_as_str = class_as_str + "{}\n".format(self.nk)
            for i_k in range(0, self.nk):
                class_as_str = class_as_str + "{:.{prec}f}  {:.{prec}f}" \
                                              "  {:.{prec}f}  {:.{prec}f} \n".format(self.kpoints_array[i_k, 0],
                                                                             self.kpoints_array[i_k, 1],
                                                                             self.kpoints_array[i_k, 2],
                                                                             self.k_weights[i_k],
                                                                             prec=8
                                                                            )
        return class_as_str


"""
Cell Parameter card excluded to encourage ibrav
"""
class cell_parameters_card:
    """
    Cell Parameters Card for QE
    """

    def __init__(self, lattice: Lattice, unit_type: str = "angstrom", precision: int = 7):
        """
        Initialize cell parameters from pymatgen lattice object
        :param lattice: Pymatgen lattice object
        :param unit_type: Desired unit of output of cell parameters card
        :param precision: Precision of cell parameters
        """
        self.lattice: Lattice = lattice.copy()
        self.cell_params: NDArrayFloat = lattice.matrix
        self.cell_dims = np.zeros(6)
        #self.convert_to_ibrav()

        self.unit_type: str = unit_type
        self.precision: int = precision
        allowed_types = ["angstrom", "bohr", "alat"]

        if unit_type.lower() == "angstrom":
            pass
        elif unit_type.lower() == "bohr":
            self.cell_params = self.cell_params / BOHR_RADIUS_ANGS
        elif unit_type.lower() == "alat":
            alat = np.linalg.norm(self.lattice.matrix[0])
            self.cell_params = self.cell_params / alat
    
    def __str__(self) -> str:
        """
        Write CELL_PARAMETERS card as input string
        :return: CELL_PARAMETERS card as string
        """
        class_as_str: str = "CELL_PARAMETERS {}\n".format(self.unit_type)
        for i_lat in range(0, 3):
            class_as_str = class_as_str + " {:.{prec}f} {:.{prec}f} {:.{prec}f}\n".format(self.cell_params[i_lat, 0],
                                                                                          self.cell_params[i_lat, 1],
                                                                                          self.cell_params[i_lat, 2],
                                                                                          prec=self.precision)
        return class_as_str



class pw_file(qe_input):
    def __init__(self, namelists=None, cards=None):
        super().__init__(namelists, cards)
        self.structure = None
    def init_default_scf_file(self, struc: Structure, prefix: str, rden: float = 6,
                          pseudo_dir: FileType = "./",  outdir: FileType = "./tmp", 
                          lsoc: bool = True):
        self.structure = struc.copy()
        control: control_nl = control_nl()
        control.init_control(calc_prefix=prefix, pseudo_dir=pseudo_dir, outdir=outdir)
        system = system_nl()
        system.init_system(struc, lsoc=lsoc, pseudo_dir=pseudo_dir)
        electrons = electrons_nl()
        electrons.init_default()
        ions = ions_nl()
        ions.init_default()
        cell = cell_nl()
        cell.init_default()
        at_species = atomic_species_card(struc)
        at_pos = atomic_positions_card(struc)
        cell_card = cell_parameters_card(struc.lattice)
        kpts = kpoints_card()
        kpts.init_automatic_grid(struc=struc, rden=rden)
        self.namelists = dict(CONTROL=control, SYSTEM=system, 
        ELECTRONS=electrons, IONS=ions, CELL=cell)

        self.cards = dict(ATOMIC_SPECIES=at_species, ATOMIC_POSITIONS=at_pos,
                          CELL_PARAMETERS=cell_card, K_POINTS=kpts)
    def init_default_bands_file(self, struc: Structure, prefix: str, npts_first_seg: int = 100,
                          pseudo_dir: FileType = "./",  outdir: FileType = "./tmp", 
                          lsoc: bool = True):
        self.structure = struc.copy()
        control: control_nl = control_nl()
        control.init_control(calc_prefix=prefix, pseudo_dir=pseudo_dir, outdir=outdir)
        control.update_nl_params({"calculation": "nscf"})
        system = system_nl()
        system.init_system(struc, lsoc=lsoc, pseudo_dir=pseudo_dir)
        system.add_nl_params({"nosym": True})
        electrons = electrons_nl()
        electrons.init_default()
        ions = ions_nl()
        ions.init_default()
        cell = cell_nl()
        cell.init_default()
        at_species = atomic_species_card(struc)
        at_pos = atomic_positions_card(struc)
        cell_card = cell_parameters_card(struc.lattice)
        kpts = kpoints_card()
        kpts.init_bpath_crystal(struc=struc, npts_first_seg=npts_first_seg)
        self.namelists = dict(CONTROL=control, SYSTEM=system, 
        ELECTRONS=electrons, IONS=ions, CELL=cell)

        self.cards = dict(ATOMIC_SPECIES=at_species, ATOMIC_POSITIONS=at_pos,
                          CELL_PARAMETERS=cell_card, K_POINTS=kpts)
    def init_default_dos_file(self, struc: Structure, prefix: str, rden: float = 6,
                          pseudo_dir: FileType = "./",  outdir: FileType = "./tmp", 
                          lsoc: bool = True):
        self.structure = struc.copy()
        control: control_nl = control_nl()
        control.init_control(calc_prefix=prefix, pseudo_dir=pseudo_dir, outdir=outdir)
        control.update_nl_params({"calculation": "nscf"})
        system = system_nl()
        system.init_system(struc, lsoc=lsoc, pseudo_dir=pseudo_dir, occupations="tetrahedra")
        electrons = electrons_nl()
        electrons.init_default()
        ions = ions_nl()
        ions.init_default()
        cell = cell_nl()
        cell.init_default()
        at_species = atomic_species_card(struc)
        at_pos = atomic_positions_card(struc)
        cell_card = cell_parameters_card(struc.lattice)
        kpts = kpoints_card()
        kpts.init_automatic_grid(struc=struc, rden=rden)
        self.namelists = dict(CONTROL=control, SYSTEM=system, 
        ELECTRONS=electrons, IONS=ions, CELL=cell)

        self.cards = dict(ATOMIC_SPECIES=at_species, ATOMIC_POSITIONS=at_pos,
                          CELL_PARAMETERS=cell_card, K_POINTS=kpts)

    def init_default_wan_nscf_file(self, struc: Structure, prefix: str, rden: float = 6,
                          pseudo_dir: FileType = "./",  outdir: FileType = "./tmp", 
                          lsoc: bool = True):
        self.structure = struc.copy()
        control: control_nl = control_nl()
        control.init_control(calc_prefix=prefix, pseudo_dir=pseudo_dir, outdir=outdir)
        control.update_nl_params({"calculation": "nscf"})
        system = system_nl()
        system.init_system(struc, lsoc=lsoc, pseudo_dir=pseudo_dir)
        system.add_nl_params({"nosym": True})
        electrons = electrons_nl()
        electrons.init_default()
        ions = ions_nl()
        ions.init_default()
        cell = cell_nl()
        cell.init_default()
        at_species = atomic_species_card(struc)
        at_pos = atomic_positions_card(struc)
        cell_card = cell_parameters_card(struc.lattice)
        kpts_tmp = kpoints_card()
        kpts_tmp.init_automatic_grid(struc=struc, rden=rden)
        nk_mesh = np.zeros(3, dtype=int)
        # Ensure mesh is even to construct commensurate k grid later
        for imesh in range(0, 3):
            if kpts_tmp.num_k[imesh] == 1:
                nk_mesh[imesh] = 1
            elif kpts_tmp.num_k[imesh]%2 == 1:
                nk_mesh[imesh] = kpts_tmp.num_k[imesh]+1
            else:
                nk_mesh[imesh] = kpts_tmp.num_k[imesh]
        
        kpts = kmesh_pl_pyth(nk1=nk_mesh[0], nk2=nk_mesh[1], nk3=nk_mesh[2])
        self.namelists = dict(CONTROL=control, SYSTEM=system, 
        ELECTRONS=electrons, IONS=ions, CELL=cell)

        self.cards = dict(ATOMIC_SPECIES=at_species, ATOMIC_POSITIONS=at_pos,
                          CELL_PARAMETERS=cell_card, K_POINTS=kpts)
    
    def init_default_vcrel_file(self, struc: Structure, prefix: str, rden: float = 6,
                          pseudo_dir: FileType = "./",  outdir: FileType = "./tmp", 
                          lsoc: bool = True, perturb_cell: bool = False):
        self.structure = struc.copy()
        cart_coords = struc.cart_coords
        lattice_info = struc.lattice.matrix.copy()
        pert_fac = 0
        if perturb_cell:
            pert_fac = 1

        #cart_coords = cart_coords + rng.random(cart_coords.shape) / 10
        seed = 0
        rng = np.random.default_rng(seed=seed)
        lattice_info = lattice_info + pert_fac*rng.random(lattice_info.shape) / 10
        relax_struc = Structure(lattice=lattice_info, species=struc.species,
                                coords=cart_coords, coords_are_cartesian=True)

        control: control_nl = control_nl()
        control.init_control(calc_prefix=prefix, pseudo_dir=pseudo_dir, outdir=outdir)
        control.update_nl_params({"calculation": "vc-relax"})
        system = system_nl()
        system.init_system(struc, lsoc=lsoc, pseudo_dir=pseudo_dir)
        system.add_nl_params({"nosym": True})
        electrons = electrons_nl()
        electrons.init_default()
        ions = ions_nl()
        ions.init_default()
        cell = cell_nl()
        cell.init_default()
        at_species = atomic_species_card(relax_struc)
        at_pos = atomic_positions_card(relax_struc)
        at_pos.fix_atoms(np.array([0]))
        cell_card = cell_parameters_card(relax_struc.lattice)
        kpts = kpoints_card()
        kpts.init_automatic_grid(struc=relax_struc, rden=rden)
        self.namelists = dict(CONTROL=control, SYSTEM=system, 
        ELECTRONS=electrons, IONS=ions, CELL=cell)

        self.cards = dict(ATOMIC_SPECIES=at_species, ATOMIC_POSITIONS=at_pos,
                          CELL_PARAMETERS=cell_card, K_POINTS=kpts)


class projwfc_nl(nl_object):
    """
    Object for the projwfc namelist for projwfc.x
    Mainly used to initialize the namelist
    """
    def __init__(self, param_dict: Dict[str, allowed_nl_types] = None):
        super().__init__(param_dict)
        self.nl_name = "&PROJWFC"

    def init_def_projwfc(self, calc_prefix: str, fermi_level: float = None,
                         additional_options: Dict[str, allowed_nl_types] = None):
        """
        Initializes a default projwfc namelist, with a spacing of 0.0001eV, and
        E_min = E_f - 5 eV, E_max = E_f + 5 eV. Projections are writen to the
        filproj file instead of the default stdout.
        :param calc_prefix: Prefix of calculation
        :param fermi_level: E_f in eV
        :param additional_options: Other parameters as a dictionary
        :return: Updates class object with default parameters
        """
        self.reset_nl_dict_to_empty()
        self.nl_dict["prefix"] = calc_prefix
        self.nl_dict["outdir"] = "./tmp/"
        if fermi_level is not None:
            self.nl_dict["Emin"] = fermi_level - 5
            self.nl_dict["Emax"] = fermi_level + 5
        self.nl_dict["DeltaE"] = 0.0001
        self.nl_dict["filproj"] = "{}_proj".format(calc_prefix)

        if additional_options is not None:
            self.nl_dict.update(additional_options)


class projwfc_input(qe_input):
    """
       Class Object for projwfc INPUT file
    """
    def __init__(self, namelists=None, cards=None):
        super().__init__(namelists, cards)

    def init_projwfc_file(self, calc_prefix: str = None, fermi_level: float = None,
                          projwfc: projwfc_nl = None, additional_options: Dict[str, allowed_nl_types] = None):
        """
        Initializes default projwfc input file
        :param calc_prefix: Prefix of calculation
        :param fermi_level: Fermi Energy in eV
        :param projwfc: If you want to use your won projwfc params, use this,
        the prefix and fermi level params, will be ignored if this is set
        """
        if projwfc is None:
            projwfc = projwfc_nl()
            projwfc.init_def_projwfc(calc_prefix=calc_prefix, fermi_level=fermi_level,
                                     additional_options=additional_options)

        self.namelists = dict(PROJWFC=projwfc)


def kmesh_pl_pyth(nk1: int, nk2: int, nk3: int):
    total_points: int = nk1*nk2*nk3
    kpt_array = np.zeros((total_points, 3))
    kwt = np.zeros(total_points)
    ct = 0
    for i in range(0, nk1):
        for j in range(0, nk2):
            for k in range(0, nk3):
                kpt_array[ct, 0] = np.round(i / nk1, 8)
                kpt_array[ct, 1] = np.round(j / nk2, 8)
                kpt_array[ct, 2] = np.round(k / nk3, 8)
                kwt[ct] = np.round(1/total_points, 12)

                ct = ct + 1
    kpt = kpoints_card()
    kpt.init_klist(kpoints_array=kpt_array, k_weights=kwt)
    return kpt       

class pw2wannier_nl(nl_object):
    """
    Object for the inputpp namelist for pw2wannier.x
    Mainly used to initialize the namelist
    """

    def __init__(self, param_dict: Dict[str, allowed_nl_types] = None):
        super().__init__(param_dict)
        self.nl_name = "&INPUTPP"

    def init_default(self, calc_prefix: str, seedname: str, lscdm: bool = True,
                     scdm_mu: float = 0, scdm_sigma: float = 0, spin: str = "none",
                     additional_options: Dict[str, allowed_nl_types] = None):
        self.nl_dict["prefix"] = calc_prefix
        self.nl_dict["outdir"] = "./tmp/"
        self.nl_dict["seedname"] = seedname
        self.nl_dict["write_amn"] = True
        self.nl_dict["write_mmn"] = True
        self.nl_dict["write_unk"] = True
        self.nl_dict["spin_component"] = spin
        if lscdm:
            self.nl_dict["scdm_proj"] = True
            self.nl_dict["scdm_mu"] = scdm_mu
            self.nl_dict["scdm_sigma"] = scdm_sigma
            self.nl_dict["scdm_entanglement"] = "erfc"
        if additional_options is not None:
            self.nl_dict.update(additional_options)


class pw2wannier_input(qe_input):
    """
    Class Object for pw2wannier INPUT file
    """
    def __init__(self, namelists=None, cards=None):
        super().__init__(namelists, cards)

    def run_pw2wannier_calc(self, input_file_name: str = None, output_file_name: str = None,
                            parallelize_ops: bool = False, parallel_dict: Dict[str, int] = None) -> int:
        if input_file_name is None:
            input_file_name = "{}_pw2wan.in".format(self.namelists["INPUTPP"].nl_dict["prefix"])
        if output_file_name is None:
            output_file_name = "{}_pw2wan.out".format(self.namelists["INPUTPP"].nl_dict["prefix"])

        ret_code = self.run_qe_calc(executable="pw2wannier90.x", input_file_name=input_file_name,
                                    output_file_name=output_file_name, parallelize_ops=parallelize_ops,
                                    parallel_dict=parallel_dict)

        return ret_code

    def init_pw2wannier_file(self, calc_prefix: str = None, pw2wannier: pw2wannier_nl = None):

        if pw2wannier is None:
            pw2wannier = pw2wannier_nl()
            pw2wannier.init_default(calc_prefix=calc_prefix, seedname="{}_wan".format(calc_prefix))

        self.namelists = dict(INPUTPP=pw2wannier)


# PH input classes


class inputph_nl(nl_object):
    """
    Object for the inputph namelist for ph.x
    Mainly used to initialize the namelist
    """

    def __init__(self, param_dict: Dict[str, allowed_nl_types] = None):
        super().__init__(param_dict)
        self.nl_name = "PH_CALC\n&INPUTPH"

    def init_default_grid(self, calc_prefix: str, rden: float, struc: Structure, xml_dyn=True,
                          make_commensurate: bool =True):
        self.reset_nl_dict_to_empty()
        self.nl_dict["prefix"] = calc_prefix
        self.nl_dict["outdir"] = "./tmp/"
        self.nl_dict["tr2_ph"] = 1e-17
        self.nl_dict["verbosity"] = "high"
        self.nl_dict["max_seconds"] = 144000
        if xml_dyn:
            self.nl_dict["fildyn"] = "{}.dyn.xml".format(calc_prefix)
            self.nl_dict["fildvscf"] = "dvscf"
        else:
            self.nl_dict["fildyn"] = "{}.dyn".format(calc_prefix)
            self.nl_dict["fildvscf"] = "dvscf"
        self.nl_dict["ldisp"] = True
        self.nl_dict["nmix_ph"] = 10
        self.nl_dict["reduce_io"] = True
        self.nl_dict["recover"] = False
        #self.nl_dict["electron_phonon"] = "simple"

        num_q = np.zeros(3)
        rl_struc: NDArrayFloat = struc.lattice.reciprocal_lattice.matrix
        for i_lat in range(0, 3):
            num_q[i_lat] = (np.floor(np.linalg.norm(rl_struc[i_lat]) * rden))
            if num_q[i_lat] == 0:
                num_q[i_lat] = 1
        if np.sum(num_q) == 3:
            num_q[0] = 2
            num_q[1] = 2
            num_q[2] = 2
        num_q = num_q.astype(int)

        self.nl_dict["nq1"] = num_q[0]
        self.nl_dict["nq2"] = num_q[1]
        self.nl_dict["nq3"] = num_q[2]
        self.nl_dict["start_q"] = 1
        self.nl_dict["last_q"] = 1
        

    def init_bands(self, calc_prefix):
        self.reset_nl_dict_to_empty()
        self.nl_dict["prefix"] = calc_prefix
        self.nl_dict["outdir"] = "./tmp/"
        self.nl_dict["tr2_ph"] = 1e-17
        self.nl_dict["verbosity"] = "high"
        self.nl_dict["max_seconds"] = 144000
        self.nl_dict["fildyn"] = "{}_dyn".format(calc_prefix)
        self.nl_dict["fildvscf"] = "{}_dvscf".format(calc_prefix)
        self.nl_dict["ldisp"] = True
        self.nl_dict["qplot"] = True
        self.nl_dict["q_in_band_form"] = True
        #self.nl_dict["electron_phonon"] = "simple"


class qPoints(kpoints_card):
    """
    This class doesn't work
    """
    def __init__(self):
        super().__init__()
    def __str__(self) -> str:
        """
        Write KPOINTS card as input string
        :return: KPOINTS card as string
        """
        
        class_as_str = "{}\n".format(self.nk)
        if self.grid_type == "gamma":
            class_as_str = class_as_str + "\n"
        elif self.grid_type == "automatic":
            class_as_str = class_as_str + " {} {} {} {} {} {}\n".format(self.num_k[0], self.num_k[1],
                                                                        self.num_k[2], self.shift_k[0],
                                                                        self.shift_k[1],
                                                                        self.shift_k[2])
        else:
            class_as_str = class_as_str + "{}\n".format(self.nk)
            for i_k in range(0, self.nk):
                class_as_str = class_as_str + "{:.{prec}f}  {:.{prec}f}" \
                                              "  {:.{prec}f}  {:.{prec}f} \n".format(self.kpoints_array[i_k, 0],
                                                                             self.kpoints_array[i_k, 1],
                                                                             self.kpoints_array[i_k, 2],
                                                                             self.k_weights[i_k],
                                                                             prec=8
                                                                            )
        return class_as_str


class ph_input(qe_input):
    def __init__(self, namelists=None, cards=None):
        super().__init__(namelists, cards)
        self.prefix = "default"

    def init_default(self, calc_prefix: str, struc: Structure, rden: float, lgrid: bool = True, lband: bool = False,
                 inputph: inputph_nl = None, qPtSpec: qPoints = None, make_commensurate=True):
        self.prefix = calc_prefix
        if inputph is None:
            if lgrid:
                lband = False
                inputph = inputph_nl()
                inputph.init_default_grid(calc_prefix=calc_prefix, rden=rden, struc=struc, 
                                          make_commensurate=make_commensurate)
            else:
                if lband:
                    if inputph is None:
                        inputph = inputph_nl()
                        inputph.init_bands(calc_prefix=calc_prefix)
                    #if qPtSpec is None:
                        #qPtSpec = ()
                        #qPtSpec.init_band_path(rden=rden, struc=struc)

                    #self.cards = dict(qPointsSpecs=qPtSpec)

            self.namelists = dict(INPUTPH=inputph)


# Begin derived classes for wannier 90 inputs

class cell_parameters_wan_block(cell_parameters_card):
    def __init__(self, lattice: Lattice, precision: int = 7):
        super().__init__(lattice=lattice, unit_type="angstrom", precision=precision)

    def __str__(self) -> str:
        class_as_str = "begin unit_cell_cart\nang\n"
        for i_lat in range(0, 3):
            class_as_str = class_as_str + " {:.{prec}f} {:.{prec}f} {:.{prec}f}\n".format(self.cell_params[i_lat, 0],
                                                                                          self.cell_params[i_lat, 1],
                                                                                          self.cell_params[i_lat, 2],
                                                                                          prec=self.precision)
        class_as_str = class_as_str + "end unit_cell_cart\n"
        return class_as_str


class atomic_positions_wan_block(atomic_positions_card):
    def __init__(self, struc: Structure, precision: int = 7):
        super().__init__(struc=struc, unit_type="crystal", precision=precision)

    def __str__(self) -> str:
        class_as_str = "begin atoms_frac\n"
        for i_at in range(0, self.nat):
            class_as_str = class_as_str + " {:.{width}}  {:.{prec}f}" \
                                          "  {:.{prec}f}  {:.{prec}f}" \
                                          "\n".format(self.symbols[i_at],
                                                      self.coords[i_at, 0],
                                                      self.coords[i_at, 1],
                                                      self.coords[i_at, 2],
                                                      width=2,
                                                      prec=self.precision
                                                      )
        class_as_str = class_as_str + "end atoms_frac\n"
        return class_as_str


class kpoints_wan_block(kpoints_card):
    def __init__(self, precision: int = 8):
        self.precision = precision
    
    def wannier_band_str(self) -> str:
        kpath_obj = KPathSetyawanCurtarolo(structure=self.structure)
        high_sym_k_info = kpath_obj.kpath["kpoints"]
        high_sym_path_info = kpath_obj.kpath["path"]
        n_continous_segments = len(high_sym_path_info)
        class_as_str = "begin kpoint_path\n"
        for i_cont_segment in range(0, n_continous_segments):
            cont_segment = high_sym_path_info[i_cont_segment]
            n_segments = len(cont_segment)
            for i_segment in range(0, n_segments - 1):
                kpt_start_frac = high_sym_k_info[cont_segment[i_segment]]
                kpt_end_frac = high_sym_k_info[cont_segment[i_segment + 1]]
                class_as_str = class_as_str + "{} {:.{prec}f} {:.{prec}f}  {:.{prec}f} " \
                                              "{} {:.{prec}f} {:.{prec}f}  {:.{prec}f}\n".format(
                    cont_segment[i_segment],
                    kpt_start_frac[0],
                    kpt_start_frac[1],
                    kpt_start_frac[2],
                    cont_segment[i_segment + 1],
                    kpt_end_frac[0],
                    kpt_end_frac[1],
                    kpt_end_frac[2],
                    prec=5)

        class_as_str = class_as_str + "end kpoint_path\n"
        return class_as_str

    def __str__(self) -> str:
        class_as_str = "begin kpoints\n"
        for i_kpt in range(0, self.nk):
            class_as_str = class_as_str + " {:.{prec}f} {:.{prec}f}  {:.{prec}f}\n".format(self.kpoints_array[i_kpt, 0],
                                                                                           self.kpoints_array[i_kpt, 1],
                                                                                           self.kpoints_array[i_kpt, 2],
                                                                                           prec=self.precision)

        class_as_str = class_as_str + "end kpoints\n"
        return class_as_str


class wann_file:
    def __init__(self, n_wfc: int, n_bnd: int, nk: NDArrayInt, struc: Structure, wannier_inputs: Dict[str, object] = None,
                 plot_bands: bool = True, plot_wfs: bool = True, exclude_str=None):

        self.wannier_inputs = {"num_wann": n_wfc,
                               "num_bands": n_bnd,
                               "num_iter": 0,
                               "num_print_cycles": 100,
                               "dis_num_iter": 0,
                               "write_xyz": ".true.",
                               "write_u_matrices": ".true.",
                               "auto_projections": ".true.",
                               "kmesh_tol": 0.0001,
                               "mp_grid": "{} {} {}".format(nk[0], nk[1], nk[2]),
                               }
        if exclude_str is not None:
            self.wannier_inputs.update({"exclude_bands": exclude_str})

        if wannier_inputs is not None:
            self.wannier_inputs.update(wannier_inputs)

        if plot_bands:
            self.wannier_inputs.update({"bands_plot": "true",
                                        "bands_plot_format": "gnuplot",
                                        "bands_num_points": 50})
        if plot_wfs:
            self.wannier_inputs.update({"wannier_plot": "true"})
        self.structure = struc.copy()

        self.ap_block = atomic_positions_wan_block(struc=self.structure)

        self.kpoints = kpoints_wan_block()
        self.kpoints.init_klist(kpoints_array=kmesh_pl_pyth(nk[0], nk[1], nk[2]).kpoints_array)
        self.cp_block = cell_parameters_wan_block(lattice=struc.lattice)

        self.kbands = None
        if plot_bands:
            self.kbands = kpoints_wan_block()
            self.kbands.init_bpath_crystal(npts_first_seg = 100, struc=struc)

    def __str__(self) -> str:
        class_as_str = ""
        for key in self.wannier_inputs:
            class_as_str = class_as_str + "{}={}\n".format(key, self.wannier_inputs[key])

        class_as_str = class_as_str + self.cp_block.__str__()
        class_as_str = class_as_str + self.ap_block.__str__()
        class_as_str = class_as_str + self.kpoints.__str__()

        class_as_str = class_as_str + self.kbands.wannier_band_str()

        return class_as_str

    def write_file(self, filename: str):
        with open(filename, "w") as file:
            file.write(self.__str__())