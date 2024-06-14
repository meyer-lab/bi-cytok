"""
Implementation of a simple multivalent binding model.
"""

import numpy as np
from valentbind import polyc


def getKxStar():
    # Armaan: redefining this as a constant would make more sense.
    return 2.24e-12


def cytBindingModel(
    recCount: np.ndarray, recXaffs: np.ndarray, dose: float, vals: np.ndarray
):
    """Runs binding model for a given mutein, valency, dose, and cell type

    Armaan: I think saying that the function runs the binding model for a
    particular cell type is a bit misleading. It's more that the function runs
    for a specific set of receptor abundances.

    Args:
        Armaan: Be very specific that the function only returns the number of
        bound receptors corresponding to the first element of this `recCount`
        array. 
        recCount: total count of signaling and targeting receptors

        Armaan: this argument's name could be more relevant.
        recXaffs: Ka for monomer ligand to receptors 
        dose: ligand concentration/dose that is being modeled
        
        Armaan: change this variable name to valencies. `vals` is often used as
        an abbreviation for `values`.
        vals: array of valencies of each ligand epitope
    Return:
        Armaan: This output description is outdated. Only the amount of the
        first receptor bound is returned.
        output: amount of receptor bound of each kind of receptor
    """
    Kx = getKxStar()
    # Armaan: Why 1e9? Again, it should be clear why literals are chosen.
    # Additionally, are you using this value elsewhere? If so, define it in a
    # separate file as a constant, and then import it.
    ligandConc = dose / (vals[0][0] * 1e9)

    output = polyc(
        ligandConc,
        Kx,
        recCount,
        vals,
        # Only one complex, so its ratio to the overall number of complexes is
        # 1.
        np.array([1]),
        recXaffs,
    )[1][0, 0]

    return output
