"""
Taken from QE, using the constants here
should guarantee consistency.
"""

pi: float = 3.14159265358979323846

"""
Physical constants, SI(NIST 2018)
http://physics.nist.gov/constants
Taken from QE/modules/constants.f90
"""

H_PLANCK_SI = 6.62607015E-34  # J s
K_BOLTZMANN_SI = 1.380649E-23  # J K^-1
ELECTRON_SI = 1.602176634E-19  # C
ELECTRONVOLT_SI = 1.602176634E-19  # J
ELECTRONMASS_SI = 9.1093837015E-31  # kg

HARTREE_SI = 4.3597447222071E-18  # J
RYDBERG_SI = HARTREE_SI / 2.0  # J
BOHR_RADIUS_SI = 0.529177210903E-10  # m
AMU_SI = 1.66053906660E-27  # kg
C_SI = 2.99792458E+8  # m s^-1 speed of light
MUNOUGHT_SI = 4 * pi * 1.0E-7  # N A^-2
EPSNOUGHT_SI = 1.0 / (MUNOUGHT_SI * C_SI * C_SI)  # F m^-1

"""
Physical constants, atomic units:
AU for "Hartree" atomic units (e = m = hbar = 1)
RY for "Rydberg" atomic units (e^2=2, m=1/2, hbar=1)
"""
K_BOLTZMANN_AU = K_BOLTZMANN_SI / HARTREE_SI
K_BOLTZMANN_RY = K_BOLTZMANN_SI / RYDBERG_SI
"""
Unit conversion factors: energy and masses
"""
AUTOEV = HARTREE_SI / ELECTRONVOLT_SI
RYTOEV = AUTOEV / 2.0
AMU_AU = AMU_SI / ELECTRONMASS_SI
AMU_RY = AMU_AU / 2.0

AU_SEC = H_PLANCK_SI / (2 * pi * HARTREE_SI)
AU_PS = AU_SEC * 1.0E+12
AU_GPA = HARTREE_SI / (BOHR_RADIUS_SI ** 3 * 1.0E+9)
RY_KBAR = 10.0 * AU_GPA / 2.0

"""
Unit conversion factors: 1 debye = 10^-18 esu*cm 
                                = 3.3356409519*10^-30 C*m 
                                = 0.208194346 e*A
( 1 esu = (0.1/c) Am, c=299792458 m/s)
"""
DEBYE_SI = 3.3356409519 * 1.0E-30  # C*m
AU_DEBYE = ELECTRON_SI * BOHR_RADIUS_SI / DEBYE_SI
eV_to_kelvin = ELECTRONVOLT_SI / K_BOLTZMANN_SI
ry_to_kelvin = RYDBERG_SI / K_BOLTZMANN_SI

EVTONM = 1E+9 * H_PLANCK_SI * C_SI / ELECTRONVOLT_SI
RYTONM = 1E+9 * H_PLANCK_SI * C_SI / RYDBERG_SI

C_AU = C_SI / BOHR_RADIUS_SI * AU_SEC

BOHR_RADIUS_CM = BOHR_RADIUS_SI * 100.0
BOHR_RADIUS_ANGS = BOHR_RADIUS_CM * 1.0E8
ANGSTROM_AU = 1.0 / BOHR_RADIUS_ANGS
DIP_DEBYE = AU_DEBYE
AU_TERAHERTZ = AU_PS
AU_TO_OHMCMM1 = 46000.0  # (ohm cm)^-1
RY_TO_THZ = 1.0 / AU_TERAHERTZ / (4 * pi)
RY_TO_GHZ = RY_TO_THZ * 1000.0
RY_TO_CMM1 = 1.E+10 * RY_TO_THZ / C_SI
AVOGADRO = 6.02214076 * 1.0E+23
