from xml.etree.ElementTree import parse as xml_parse
from xml.etree.ElementTree import ElementTree
from typing import Dict, Union
from pymatgen.core.periodic_table import Element


def isfloat(string: str):
    try:
        float(string)
        return True
    except ValueError:
        return False


def isint(string: str):
    try:
        int(string)
        return True
    except ValueError:
        return False


class upffile:
    """
    Parses all relevant information from the upf file.
    """
    def __init__(self, xml_file: str):
        self.xml_tree: ElementTree = xml_parse(xml_file)
        self.root = self.xml_tree.getroot()
        info_temp: Dict[str, str] = self.root.find("PP_HEADER").attrib
        self.info: Dict[str, Union[str, bool, int, float]] = {"filename": xml_file}
        for key in info_temp:
            if info_temp[key] == "F":
                self.info.update({key: False})
            elif info_temp[key] == "T":
                self.info.update({key: True})
            elif info_temp[key] == "z_valence":
                self.info.update({"z_valence": info_temp[key]})
            elif isint(info_temp[key].strip()):
                self.info.update({key: int(info_temp[key])})
            elif isfloat(info_temp[key].strip()):
                self.info.update({key: float(info_temp[key])})
            elif key == "element":
                self.info.update({key: info_temp[key]})
                self.info.update({"atomic_number": Element(info_temp[key].strip()).Z,
                                  "atomic_mass": Element(info_temp[key].strip()).atomic_mass})
            else:
                self.info.update({key: info_temp[key]})           
        if "ecut_rec" not in self.info.keys():
            self.info.update({"ecut_rec": 150})
        