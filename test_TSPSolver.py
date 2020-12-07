import copy
from unittest import TestCase

from PyQt5.QtWidgets import QApplication

from Proj5GUI import Proj5GUI
from TSPClasses import TSPSolution
from TSPSolver import TSPSolver


class TestTSPSolver(TestCase):
    def test_mutate(self):
        app = QApplication([])
        w = Proj5GUI()
        w.generateNetwork()
        sol = TSPSolution(w._scenario.cities)
        ver = TSPSolver(w.view).mutate([copy.deepcopy(sol)])

        # self.assertTrue(True, msg='good')
        same = True
        for i in range(len(sol.route)):
            if sol.route[i] != ver[0].route[i]:
                same = False
        self.assertFalse(same, msg='the mutated route should have a different order')
        # ver[0].route.append('yep')
        self.assertEqual(len(set(sol.route)), len(set(ver[0].route)), msg='it should have the same elements')
