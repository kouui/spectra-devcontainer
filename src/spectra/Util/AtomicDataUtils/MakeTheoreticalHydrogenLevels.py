# -------------------------------------------------------------------------------
# definition of functions for make hydrogen .Level file with theretical formulas
# -------------------------------------------------------------------------------

from pathlib import Path

from spectra import Constants as CST

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
    -             -       -          -   -   -     1       2      {continuum}
END
#--------------------------------------------------------------------------------------------------
"""


def make_hydrogen_levels_(nlevel: int, outfile: str):
    if nlevel < 3:
        nlevel = 3
        print("nlevel < 3, nlevel is set to 3")

    template = TEMPLATE.replace("{nLevel}", f"{nlevel}")
    s = ""
    for i in range(2, nlevel):  ## exclude n=1 and continuum
        n = conf = i
        g = 2 * n * n
        # ionization energy
        Ry = CST.E_Rydberg_H_  # Rydberg energy unit, proton-mass corrected
        Eik = Ry * (1.0 / n**2)
        erg = (Ry - Eik) / CST.eV2erg_
        s += f"    {conf:<2d}            -       -          {n:<2d}  -   -     {g:<4d}    1      {erg:.7E}\n"
    continuum_erg = CST.E_Rydberg_H_ / CST.eV2erg_  # ionization energy [eV], proton-mass corrected
    template = template.replace("{continuum}", f"{continuum_erg:.7E}")
    template = template.replace("{levels}", s)
    with Path(outfile).open("w") as f:
        f.write(template)
    print(f"saved as: {outfile}")
    return 0


if __name__ == "__main__":
    import argparse

    # Create the parser
    parser = argparse.ArgumentParser(description="create hydrogen .Level file with theretical formula")
    # Add arguments
    parser.add_argument("outfile", type=str, help="path of output file")
    parser.add_argument("-nl", "--nlevel", type=int, default=8, help="Number of levels including continuum, must >=3")
    # Parse arguments
    args = parser.parse_args()

    make_hydrogen_levels_(args.nlevel, args.outfile)
