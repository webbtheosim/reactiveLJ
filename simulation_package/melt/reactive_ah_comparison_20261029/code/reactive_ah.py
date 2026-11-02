"""Conservative coordination-dependent, shifted Ashbaugh-Hatch interaction.

With u = LJ(r) - LJ(rc), D = epsilon + LJ(rc), and rm = 2**(1/6)*sigma,
U = u + (1-W)*D for r <= rm, W*u for rm < r < rc, and zero otherwise.
W uses the same leave-pair-out coordination and regularization as ReactiveLJ.
The compiled CPU/GPU force includes all derivatives of W with positions.
This module adds a force; it never replaces the installed ReactiveLJ class.
"""
import copy
import hoomd
from hoomd.md.many_body import ReactiveLJ
import _reactive_ah


class ReactiveAH(ReactiveLJ):
    """Reactive Ashbaugh-Hatch force with the study's coordination measure."""

    def __init__(self, nlist, reactive_type, sigma, epsilon, r_cut=None,
                 weakening_inner=None, weakening_outer=None, weakening_exponent=4.0):
        super().__init__(nlist, reactive_type, sigma, epsilon, r_cut,
                         weakening_inner, weakening_outer, weakening_exponent,
                         smooth_elbow=False, smooth_kappa=0.0, smooth_beta=1.0)

    def _effective_parameters(self):
        pars = super()._effective_parameters()
        if pars[2] <= 2**(1/6)*pars[0] or pars[4] > pars[2]:
            raise ValueError("Require rm < r_cut and weakening_outer <= r_cut")
        if self.smooth_elbow or self.smooth_kappa != 0:
            raise ValueError("ReactiveAH does not use a Hermite smoothing window")
        return pars

    def _apply_cpp_parameters(self):
        if self._attached:
            self._cpp_obj.set_params(*self._effective_parameters())

    def _attach_hook(self):
        if self.nlist._attached and self.nlist._simulation != self._simulation:
            self.nlist = copy.deepcopy(self.nlist)
        self.nlist._attach(self._simulation)
        cls = (_reactive_ah.ReactiveAHForceCompute if
               isinstance(self._simulation.device, hoomd.device.CPU) else
               _reactive_ah.ReactiveAHForceComputeGPU)
        self._cpp_obj = cls(self._simulation.state._cpp_sys_def,
                            self.nlist._cpp_obj, self.reactive_type,
                            *self._effective_parameters())
