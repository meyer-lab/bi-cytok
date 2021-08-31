"""
This creates Figure 1, response of bispecific IL-2 cytokines at varing valencies and abundances using binding model.
"""
from .figureCommon import getSetup
from os.path import join
from ..imports import importCITE
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    ax, f = getSetup((15, 8), (2, 2))
    CITE_PCA(ax[0:2])
    

    return f

def CITE_PCA(ax):
    """ Plots all surface markers in PCA format"""
    PCA_DF = importCITE()
    cellType = PCA_DF.CellType2.values
    PCA_DF = PCA_DF.loc[:, ((PCA_DF.columns != 'CellType1') & (PCA_DF.columns != 'CellType2') & (PCA_DF.columns != 'CellType3') & (PCA_DF.columns != 'Cell'))]
    factors = PCA_DF.columns
    scaler = StandardScaler()

    pca = PCA(n_components=2)
    PCA_Arr = scaler.fit_transform(X=PCA_DF.values)
    pca.fit(PCA_Arr)
    comps = pca.components_
    loadingsDF = pd.DataFrame({"PC 1": comps[0], "PC 2": comps[1], "Factors": factors})
    loadP = sns.scatterplot(data=loadingsDF, x="PC 1", y="PC 2", ax=ax[1])

    for line in range(0, loadingsDF.shape[0]):
        loadP.text(loadingsDF["PC 1"][line] + 0.003, loadingsDF["PC 2"][line], 
        factors[line], horizontalalignment='left', 
        size='medium', color='black', weight='semibold')

    scores = pca.transform(PCA_Arr)
    scoresDF = pd.DataFrame({"PC 1": scores[:, 0], "PC 2": scores[:, 1], "Cell Type": cellType})
    sns.scatterplot(data=scoresDF, x="PC 1", y="PC 2", ax=ax[0], hue="Cell Type")
