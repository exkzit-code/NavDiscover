import unittest
from types import SimpleNamespace
import numpy as np
from experiments.search import objective_score, simulate


class SearchTests(unittest.TestCase):
    def test_rejected_run_is_never_rewarded(self):
        self.assertEqual(objective_score(SimpleNamespace(completed=False)),np.inf)

    def test_short_search_candidate_respects_fixed_energy(self):
        for parameters in [[1.0,2.5,1.2,1.0],[1.35,3.5,1.8,1.4]]:
            result=simulate(parameters,n=24,nu=.005,final_time=.01,dt_max=.002)
            self.assertTrue(result.completed,result.reason)
            self.assertAlmostEqual(result.history[0]['energy'],.5,places=13)
            self.assertTrue(np.isfinite(objective_score(result)))

    def test_nonfinite_objective_is_not_rewarded(self):
        for value in [np.inf,np.nan,None]:
            result=SimpleNamespace(completed=True,summary=lambda:{'enstrophy_ratio':value})
            self.assertEqual(objective_score(result),np.inf)


if __name__=='__main__':
    unittest.main()
