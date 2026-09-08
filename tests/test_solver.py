import unittest
import numpy as np
from src.solver import NavierStokesSolver, relative_field_difference
from src.initial_conditions import taylor_green_2d, kida, antiparallel_tubes, filtered_fixed_energy


class SolverTests(unittest.TestCase):
    def test_projection_annihilates_gradient_and_is_idempotent(self):
        s=NavierStokesSolver(16)
        rng=np.random.default_rng(42)
        phi=rng.normal(size=(16,16,16))+1j*rng.normal(size=(16,16,16))
        gradient=1j*s.k*phi*s.mask
        self.assertLess(np.linalg.norm(s.project(gradient))/np.linalg.norm(gradient), 1e-14)
        state=rng.normal(size=(3,16,16,16))+1j*rng.normal(size=(3,16,16,16))
        projected=s.project(state)
        self.assertLess(np.linalg.norm(s.project(projected)-projected)/np.linalg.norm(state),1e-14)
        np.testing.assert_array_equal(projected[:,0,0,0],state[:,0,0,0])

    def test_exact_smooth_flow_does_not_report_blowup(self):
        s=NavierStokesSolver(16,nu=.05)
        state=taylor_green_2d(s)
        result=s.run(state,final_time=.2,dt_max=.005)
        self.assertTrue(result.completed,result.reason)
        exact=state*np.exp(-2*s.nu*.2)
        self.assertLess(relative_field_difference(result.final_state,exact),1e-11)
        self.assertLess(result.history[-1]['energy'],result.history[0]['energy'])

    def test_temporal_order_on_exact_diffusing_flow(self):
        s=NavierStokesSolver(16,nu=.5)
        state=taylor_green_2d(s)
        exact=state*np.exp(-2*s.nu*.4)
        errors=[]
        # Direct steps isolate RK4 temporal error from the adaptive timestep cap.
        for dt in [.04,.02]:
            candidate=state.copy()
            for _ in range(round(.4/dt)):
                candidate,_=s.step(candidate,dt)
            errors.append(relative_field_difference(candidate,exact))
        self.assertGreater(errors[0]/errors[1],14)
        self.assertLess(errors[0]/errors[1],18)

    def test_inviscid_nonlinearity_has_zero_energy_transfer(self):
        s=NavierStokesSolver(24,nu=0)
        state=filtered_fixed_energy(s,antiparallel_tubes(s))
        transfer=np.real(np.sum(np.conj(state)*s.rhs(state)))/s.normalization
        self.assertLess(abs(transfer),1e-13)

    def test_kida_energy_budget_and_incompressibility(self):
        s=NavierStokesSolver(24)
        result=s.run(kida(s),final_time=.02,dt_max=.001)
        self.assertTrue(result.completed,result.reason)
        self.assertLess(result.summary()['max_energy_budget_error'],1e-8)
        self.assertLess(result.summary()['max_relative_divergence'],1e-13)

    def test_normalization_and_filter_are_consistent_across_grids(self):
        states=[]
        for n in [24,48]:
            s=NavierStokesSolver(n)
            state=filtered_fixed_energy(s,antiparallel_tubes(s),amplitude=1.3)
            self.assertAlmostEqual(s.energy(state),.5,places=13)
            self.assertLess(s.diagnostics(state)['reality_error'],1e-13)
            states.append(state)
        self.assertLess(relative_field_difference(*states),1e-5)

    def test_enstrophy_is_computed_from_curl_even_for_bad_input(self):
        s=NavierStokesSolver(16)
        gradient=np.zeros((3,16,16,16),complex)
        gradient[0,1,0,0]=1
        gradient[0,-1,0,0]=1
        self.assertEqual(s.diagnostics(gradient)['enstrophy'],0)
        rejected=s.run(gradient)
        self.assertEqual(rejected.reason,'relative_divergence')

    def test_bad_reality_and_unresolved_fields_are_rejected(self):
        s=NavierStokesSolver(16)
        bad=taylor_green_2d(s)*1j
        self.assertEqual(s.run(bad).reason,'reality_error')
        state=np.zeros((3,16,16,16),complex)
        state[1,5,0,0]=state[1,-5,0,0]=1
        self.assertEqual(s.run(state).reason,'spectral_tail_fraction')

    def test_energy_injecting_solver_is_rejected(self):
        class Broken(NavierStokesSolver):
            def step(self,state,dt):
                return state*2,0.0
        s=Broken(16)
        result=s.run(taylor_green_2d(s))
        self.assertEqual(result.reason,'energy_budget_error')

    def test_zero_flow_remains_zero(self):
        s=NavierStokesSolver(16)
        result=s.run(s.prepare(np.zeros((3,16,16,16))),final_time=.01)
        self.assertTrue(result.completed)
        self.assertEqual(result.summary()['final_energy'],0)
        self.assertIsNone(result.summary()['enstrophy_ratio'])

    def test_invalid_parameters_and_input_fail_explicitly(self):
        for n in [3,15,16.5]:
            with self.assertRaises(ValueError): NavierStokesSolver(n)
        with self.assertRaises(ValueError): NavierStokesSolver(16,nu=-1)
        s=NavierStokesSolver(16)
        with self.assertRaises(ValueError): s.run(taylor_green_2d(s),dt_max=0)
        with self.assertRaises(ValueError): s.prepare(np.full((3,16,16,16),np.nan))


if __name__=='__main__':
    unittest.main()
