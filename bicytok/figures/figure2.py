from ..distanceMetricFuncs import KL_EMD_1D
from .common import getSetup


def makeFigure():
    """Figure file to generate 1D KL divergence and EMD for given cell type/subset."""
    ax, f = getSetup((8, 8), (3, 2))

    KL_EMD_1D(ax[0:2], "Treg Memory", 10)
    KL_EMD_1D(ax[2:4], "Treg Memory", 10, offTargState=1)
    KL_EMD_1D(ax[4:6], "Treg Memory", 10, offTargState=2)
    ''' filtering data portion 
    CITE_DF = importCITE()
    markerDF = pd.DataFrame(columns=["Marker", "Cell Type", "Amount"])
    for marker in CITE_DF.loc[
        :,
        (
            (CITE_DF.columns != "CellType1")
            & (CITE_DF.columns != "CellType2")
            & (CITE_DF.columns != "CellType3")
            & (CITE_DF.columns != "Cell")
        ),
    ].columns:
    
    '''

    ''' plot portion
    corrsDF = pd.DataFrame()
    for i, distance in enumerate(["Wasserstein Distance", "KL Divergence"]):
        ratioDF = markerDF.sort_values(by=distance)
        posCorrs = ratioDF.tail(numFactors).Marker.values
        corrsDF = pd.concat(
            [corrsDF, pd.DataFrame({"Distance": distance, "Marker": posCorrs})]
        )
        markerDF = markerDF.loc[markerDF["Marker"].isin(posCorrs)]
        sns.barplot(
            data=ratioDF.tail(numFactors), y="Marker", x=distance, ax=ax[i], color="k"
        )
        ax[i].set(xscale="log")
        ax[0].set(title="Wasserstein Distance - Surface Markers")
        ax[1].set(title="KL Divergence - Surface Markers")
    return corrsDF
    
    '''
    return f
