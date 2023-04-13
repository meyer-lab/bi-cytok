"""
Unit test file.
"""
import unittest

from ..selectivityFuncs import getSampleAbundances, optimizeDesign


class TestModel(unittest.TestCase):
    """Here are the unit tests."""

    def test_eg(self):
        """Example."""
        self.assertTrue(3 + 1 > 0)
    
    def test_optimize_design():
        targCell = 'Treg'
        offTCells = np.array(['CD8 Naive', 'NK', 'CD8 TEM', 'CD4 Naive', 'CD4 CTL'])
        cells = np.append(offTCells, targCell)
        
        epitopesList = pd.read_csv(join(path_here, "data/epitopeList.csv"))
        epitopes = list(epitopesList['Epitope'].unique())
        epitopesDF = getSampleAbundances(epitopes, cells)

        result = optimizeDesign(targCell, offTCells, epitopesDF, 'CD122', 0.01, [8.0, 8.0, 8.0])
