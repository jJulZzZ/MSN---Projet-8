"""Independent identities and numerical limits for the new reference routines."""
import unittest

import numpy as np
from scipy.integrate import quad

from msn_p8.references import classical_reference, quantum_reference, transfer_reference


class ReferenceChecks(unittest.TestCase):
    def test_classical_against_adaptive_quadrature(self):
        potential = lambda y: (y*y-1)**2
        for theta in [.4, .6]:
            Z = quad(lambda y: np.exp(-potential(y)/theta), -4, 4)[0]
            exact = quad(lambda y: potential(y)*np.exp(-potential(y)/theta), -4, 4)[0]/Z
            self.assertAlmostEqual(classical_reference(theta)['potential_energy'], exact, places=10)

    def test_single_bead_transfer_limit_is_classical(self):
        for C in [.3, .9]:
            reference = transfer_reference(C, .4, 1, include_centroid=False)
            self.assertAlmostEqual(reference['potential_energy'],
                                   classical_reference()['potential_energy'], places=10)

    def test_quantum_density_normalization_symmetry_and_grid(self):
        for C in [.3, .9]:
            fine = quantum_reference(C, .6)
            coarser = quantum_reference(C, .6, points=2001)
            h = fine['y'][1]-fine['y'][0]
            self.assertAlmostEqual(float(fine['density'].sum()*h), 1, places=10)
            np.testing.assert_allclose(fine['density'], fine['density'][::-1], atol=1e-9)
            self.assertLess(abs(fine['potential_energy']-coarser['potential_energy']), 2e-6)
            self.assertLess(fine['last_state_weight'], 1e-12)

    def test_finite_p_energy_approaches_schrodinger_reference(self):
        for C in [.3, .9]:
            limit = quantum_reference(C, .4)['potential_energy']
            errors = [abs(transfer_reference(C, .4, P, include_centroid=False)
                          ['potential_energy']-limit) for P in [15, 30, 60]]
            self.assertGreater(errors[0], errors[1])
            self.assertGreater(errors[1], errors[2])
            self.assertLess(errors[2]/limit, .003)

    def test_centroid_refinement_and_variance_bound(self):
        for C, theta in [(.3, .4), (.9, .6)]:
            base = transfer_reference(C, theta, 15)
            refined = transfer_reference(C, theta, 15, grid_points=281, modes=100)
            self.assertLess(abs(base['barrier']-refined['barrier']), 2e-5)
            self.assertLess(base['fourier_tail'], 1e-10)
            np.testing.assert_allclose(base['centroid_density'],
                                       np.roll(base['centroid_density'][::-1], 1), atol=1e-12)
            dq = base['centroid_y'][1]-base['centroid_y'][0]
            dy = base['y'][1]-base['y'][0]
            self.assertAlmostEqual(float(base['centroid_density'].sum()*dq), 1, places=10)
            centroid_second = np.sum(base['centroid_y']**2*base['centroid_density'])*dq
            bead_second = np.sum(base['y']**2*base['bead_density'])*dy
            self.assertLess(centroid_second, bead_second)  # Jensen's inequality for each path.


if __name__ == '__main__':
    unittest.main()
