#-------------------------------------------------------------------------------
# definition of functions for make hydrogen .Level file with theretical formulas
#-------------------------------------------------------------------------------
# VERSION
# 0.1.0 
#    2024/02/02   u.k.   
#-------------------------------------------------------------------------------

from spectra_src import Configurations as CFG
from spectra_src import Constants as CST

TEMPLATE = """#--------------------------------------------------------------------------------------------------
Title: Hydrogen atomic model
#--------------------------------------------------------------------------------------------------
Z                    1
Element              H
nLevel               {nLevel}
#
END
#--------------------------------------------------------------------------------------------------
#   conf          term    J          n   L   2S+1  g       stage  E[eV]
prefix    -
    1s            2S      1/2        1   0   2     2       1      0.0000000E+00
{levels}prefix    -
    -             -       -          -   -   -     1       2      1.3598430E+01
END
#--------------------------------------------------------------------------------------------------
"""

def make_hydrogen_levels_(nlevel:int,outfile:str):
    if nlevel < 3:
        nlevel = 3
        print("nlevel < 3, nlevel is set to 3")
    
    template = TEMPLATE.replace("{nLevel}",f"{nlevel}")
    s = ""
    for i in range(2,nlevel): ## exclude n=1 and continuum
        n = conf = i
        g = 2*n*n
        # ionization energy
        Eik = CST.E_Rydberg_ * (1./n**2)
        erg = (1.3598430E+01*CST.eV2erg_ - Eik) / CST.eV2erg_
        s += f"    {conf:<2d}            -       -          {n:<2d}  -   -     {g:<4d}    1      {erg:.7E}\n"
    template = template.replace("{levels}", s)
    with open(outfile, 'w') as f:
        f.write(template)
    print(f"saved as: {outfile}")
    return 0


if __name__ == "__main__":
    import argparse
    # Create the parser
    parser = argparse.ArgumentParser(description='create hydrogen .Level file with theretical formula')
    # Add arguments
    parser.add_argument('outfile', type=str, help='path of output file')
    parser.add_argument('-nl', '--nlevel', type=int, default=8, help='Number of levels including continuum, must >=3')
    # Parse arguments
    args = parser.parse_args()


    make_hydrogen_levels_(args.nlevel, args.outfile)
