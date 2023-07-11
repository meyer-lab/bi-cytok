"""
Unit test file.
"""
from os.path import join
import unittest
import numpy as np
import pandas as pd
from bicytok.selectivityFuncs import getSampleAbundances, optimizeDesign, path_here


class TestModel(unittest.TestCase):
    """Here are the unit tests."""

    def test_eg(self):
        """Example."""
        self.assertTrue(3 + 1 > 0)

    def test_optimize_design(self):
        targCell = 'Treg'
        offTCells = np.array(['CD8 Naive', 'NK', 'CD8 TEM', 'CD4 Naive', 'CD4 CTL'])
        cells = np.append(offTCells, targCell)
        
        epitopesList = pd.read_csv(join(path_here, "bicytok/data/epitopeList.csv"))
        epitopes = list(epitopesList['Epitope'].unique())
        epitopesDF = getSampleAbundances(epitopes, cells)

        result = optimizeDesign(secondary="CD122", epitope="CD25", targCell=targCell, offTCells=offTCells, selectedDF=epitopesDF, dose=0.1, valency=2, prevOptAffs=[8.0, 8.0, 8.0])
