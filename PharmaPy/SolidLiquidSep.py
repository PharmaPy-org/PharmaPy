# -*- coding: utf-8 -*-
"""
Created on Wed May 13 02:03:18 2020

@author: huri

References
----------
Destro et al. (2021), Chemical Engineering Science, 244, 116803,
https://doi.org/10.1016/j.ces.2021.116803.
See repository-level REFERENCES.md for the full citation.
"""

from typing import Callable, Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from PharmaPy._assimulo import CVode, Explicit_Problem
from PharmaPy.Commons import trapezoidal_rule, series_erfc
from PharmaPy.Phases import classify_phases
from PharmaPy.MixedPhases import Slurry, Cake
from PharmaPy.general_interpolation import define_initial_state

from PharmaPy.Commons import (unpack_states, reorder_pde_outputs,
                              eval_state_events, handle_events,
                              unpack_discretized, TerminateSimulation)
from PharmaPy.Connections import get_inputs_new
from PharmaPy.Results import DynamicResult
from PharmaPy.NameAnalysis import get_dict_states

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
import copy

from scipy.special import erfc, erfcx

eps = np.finfo(float).eps * 1.1
grav = 9.8  # m/s**2


def high_resolution_fvm(f, boundary_cond, limiter_type='Van Leer'):

    # Ghost cells -1, 0 and N + 1 (see LeVeque 2002, Chapter 9)
    f_extrap = 2*f[-1] - f[-2]
    f_aug = np.concatenate(([boundary_cond]*2, f, [f_extrap]))

    f_diff = np.diff(f_aug, axis=0)

    theta = (f_diff[:-1]) / (f_diff[1:] + eps)

    if limiter_type == 'Van Leer':
        limiter = (np.abs(theta) + theta) / (1 + np.abs(theta))
    else:  # TODO: include more limiters
        pass

    fluxes = f_aug[1:-1] + 0.5 * f_diff[1:] * limiter

    return fluxes


def upwind_fvm(f, boundary_cond):
    f_aug = np.concatenate(([boundary_cond], f))

    return f_aug


def get_alpha(solid_phase, porosity: float, sphericity: float,
              rho_sol: float, csd=None) -> float:
    """Estimate the cake specific resistance from the particle-size grid.

    Parameters
    ----------
    solid_phase : SolidPhase
        Solid phase containing ``x_distrib`` [um], ``distrib`` [#/m**3/um],
        and density information [kg/m**3].
    porosity : float
        Cake porosity [-].
    sphericity : float
        Particle sphericity [-]. Accepted for the legacy API; the current
        implementation uses the solid phase's volumetric shape factor.
    rho_sol : float
        Solid density [kg/m**3]. Accepted for the legacy API; the current
        implementation reads density from ``solid_phase``.
    csd : ndarray, optional
        Crystal-size distribution [#/m**3/um]. Accepted for the legacy API;
        the current implementation reads the distribution from ``solid_phase``.

    Returns
    -------
    float
        Cake specific resistance [m/kg].

    Notes
    -----
    ``SolidPhase.x_distrib`` is stored in micrometers [um] and converted to
    meters [m] before the Carman-Kozeny-style resistance is evaluated.
    The coefficient 180 is the Carman (1937), Equation 10 value with Kozeny
    constant 5 for granular beds (doi:10.1016/S0263-8762(97)80003-2).
    The phase-owned scalar ``kv`` [-] cancels from normalized volume weights;
    resistance is independent of it at fixed porosity.
    """
    csd = solid_phase.distrib  # [common number basis/um]
    rho_sol = solid_phase.getDensity()
    x_grid = solid_phase.x_distrib * 1e-6

    kv = solid_phase.kv  # [-], phase-owned volumetric shape factor

    del_x_dist = np.diff(x_grid)
    node_x_dist = (x_grid[:-1] + x_grid[1:]) / 2
    node_CSD = (csd[:-1] + csd[1:]) / 2

    # Volume of crystals in each bin
    vol_cry = node_CSD * del_x_dist * (kv * node_x_dist**3)
    # Normalize volume weights; their common scalar kv cancels exactly.
    vol_frac = vol_cry/ np.sum(vol_cry)
    x_grid = node_x_dist
    alpha_x = 180 * (1 - porosity) / porosity**3 / x_grid**2 / rho_sol
    alpha = np.sum(alpha_x * vol_frac)

    return alpha


def get_sat_inf(x_vec, csd, deltaP: float, porosity: float, height: float,
                mu_zero: float, props):
    """Estimate irreducible cake saturation from a particle-size distribution.

    Parameters
    ----------
    x_vec : ndarray
        Particle diameter grid supplied to the capillary correlation [m].
    csd : ndarray
        Crystal-size distribution on the stored grid [#/m**3/um].
    deltaP : float
        Pressure drop through the cake used in the capillary number [Pa].
    porosity : float
        Cake porosity [-].
    height : float
        Cake height [m].
    mu_zero : float
        Zeroth number-distribution moment [#/m**3]. Accepted for the legacy
        API; this volume-weighted implementation does not use it directly.
    props : tuple
        Surface tension [N/m] and liquid density [kg/m**3].

    Returns
    -------
    float or ndarray
        Irreducible saturation clipped to the physically admissible interval
        ``0 <= S_inf <= 1`` [-]. Scalar property inputs return a scalar;
        array-valued property inputs return one saturation per property node.

    Notes
    -----
    The threshold-pressure and irreducible-saturation correlations follow
    Destro et al. (2021), Equations 16-18
    (doi:10.1016/j.ces.2021.116803), which adapt Wakeman's mono-sized-cake
    correlations to a particle-size distribution with additive
    volume-fraction weighting. The common scalar volume shape factor
    cancels exactly: ``kv*w_i / sum(kv*w) = w_i / sum(w)``. This function
    therefore needs no phase-owned shape factor or change to its callers.
    """
    surf_tens, rho_liq = props

    del_x_dist = np.diff(x_vec)
    node_x_dist = (x_vec[:-1] + x_vec[1:]) / 2
    node_CSD = (csd[:-1] + csd[1:]) / 2

    x_vec = node_x_dist
    if isinstance(surf_tens, float) or isinstance(rho_liq, float):
        capillary_number = porosity**3 * x_vec**2 * \
            (rho_liq*grav*height + deltaP) / (1 - porosity)**2 / height / surf_tens
    else:
        capillary_number = np.outer(
            porosity**3 * x_vec**2,
            (rho_liq*grav*height + deltaP)/(1 - porosity)**2 / height / surf_tens
            )
    # kv is common to all bin volumes: kv*w_i / sum(kv*w) = w_i/sum(w).
    # No phase collaborator or shape-factor handoff is needed here.
    vol_cry = node_CSD * del_x_dist * node_x_dist**3  # [proportional volume]

    s_inf = 0.155 * (1 + 0.031*capillary_number**(-0.49))
    s_inf = np.where(s_inf > 1, 1, s_inf)

    # Calculate irreducible saturation in weighted csd (volume based)
    vol_frac = vol_cry/ np.sum(vol_cry)

    if np.ndim(s_inf) == 1:
        s_inf = np.sum(vol_frac * s_inf)
    else:
        s_inf = np.sum(vol_frac[:, np.newaxis] * s_inf, axis=0)

    s_inf = np.clip(s_inf, 0, 1)  # [-]

    return s_inf


def _validate_irreducible_saturation(s_inf, deltaP, size_grid_m, operation):
    """Validate that reduced-saturation coordinates are well defined.

    Parameters
    ----------
    s_inf : float or ndarray
        Irreducible saturation predicted by ``get_sat_inf`` [-].
    deltaP : float
        Pressure drop used in the irreducible-saturation estimate [Pa].
    size_grid_m : ndarray
        Particle diameter grid supplied to ``get_sat_inf`` [m].
    operation : str
        Unit-operation name included in the error message [-].

    Returns
    -------
    None
        The function returns only when ``s_inf`` is finite and below one.

    Raises
    ------
    ValueError
        If ``s_inf`` is non-finite or greater than or equal to one, making the
        reduced-saturation mapping ``(S - S_inf) / (1 - S_inf)`` undefined.
    """
    s_inf_array = np.asarray(s_inf)  # [-]
    size_grid_m = np.asarray(size_grid_m)  # [m]

    if (
            not np.all(np.isfinite(s_inf_array))
            or np.any(s_inf_array >= 1)):
        size_min = np.min(size_grid_m)  # [m]
        size_max = np.max(size_grid_m)  # [m]
        raise ValueError(
            f"{operation} irreducible saturation s_inf={s_inf} must be "
            "finite and less than 1 for the reduced-saturation mapping; "
            f"deltaP={deltaP} Pa, particle size grid spans "
            f"{size_min} to {size_max} m."
        )


class Carousel:

    def __init__(self, Phases=None, num_chambers=None, cycle_time=None):
        self._Phases = Phases
        self.uo_instances = {}

    def __CollectInstances(self):
        for key, value in self.__dict__.items():
            module = getattr(value, '__module__', None)
            if module == 'SolidLiqSep':
                self.uo_instances[key] = value


class DeliquoringStep:
    def __init__(self, num_nodes, diam_unit=0.01,
                 resist_medium=1e9):

        """

        Parameters
        ----------
        number_nodes : float
            Number of apatial discretization of the cake along the axial
            coordinate.
        diam_unit : float (optional, default=0.01)
            Diameter of the dryer's cross section [m]
        resist_medium : float (optional, default=1e9)
            Mesh resistance of filter in dryer. [m**-1]

        """

        self.num_nodes = num_nodes
        self.diam_unit = diam_unit
        self.area_cross = np.pi/4 * diam_unit**2
        self.resist_medium = resist_medium

        # Phases output
        self._Phases = None

        self.nomenclature()
        self.oper_mode = 'Batch'
        self.is_continuous = False

        self.outputs = None

    @property
    def Phases(self):
        return self._Phases

    @Phases.setter
    def Phases(self, phases):
        if isinstance(phases, (list, tuple)):
            self._Phases = phases

            self.CakePhase = Cake(self.num_nodes)
            self.CakePhase.Phases = list(phases)
        elif phases.__class__.__name__ == 'Cake':
            self.CakePhase = phases
            self._Phases = phases.Phases
        else:
            raise RuntimeError('Please provide a list or tuple of phases '
                               'objects')

        classify_phases(self)  # Enumerate phases: Liquid_1,..., Solid_1, ...

        self.cake_height = self.CakePhase.cake_vol / self.area_cross

        z_grid_red = np.linspace(0, 1, self.num_nodes + 1)
        dz = z_grid_red[1] - z_grid_red[0]

        z_centers = (z_grid_red[1:] + z_grid_red[:-1]) / 2

        self.z_centers = z_centers
        self.z_grid = z_grid_red
        self.delta_z = np.diff(z_grid_red)

        self.__original_phase__ = copy.deepcopy(self.Liquid_1.__dict__)

        self.name_species = self.Liquid_1.name_species

        self.states_di = {
            'saturation' :{# 'index': index_z,
                           'dim': 1,
                           'units': '', 'type': 'diff'},
            'mass_conc' : {'index': self.name_species,
                         'dim':len(self.name_species), 'units': 'kg/m**3', 'type':'diff'} # Order of the states matters
                }

        self.fstates_di = {
            'mean_saturation_value' : {# 'index': index_z,
                     'dim': 1, 'units': '',}}

        self.name_states = list(self.states_di.keys())
        self.dim_states = [a['dim'] for a in self.states_di.values()]

        self.nomenclature()

    def nomenclature(self):
        self.names_states_in = ['mass', 'temp',
                                'mass_frac', 'total_distrib']  #from filter states_out
        self.names_states_out = self.names_states_in

    def get_inputs(self, time):
        pass

    def unit_model(self, theta, states):
        states_reord = states.reshape(-1, self.Liquid_1.num_species + 1)
        sat_red = states_reord[:, 0]
        mass_conc = states_reord[:, 1:]

        model_eqns = self.material_balance(theta, sat_red, mass_conc)

        return model_eqns

    def material_balance(self, theta, sat_star, conc_star):

        """

        Calculate material balance for a non-dimensional version of the
        governing equations. Based on Wakeman

        """

        lambd = 5
        sat_star = np.where(sat_star<0, eps, sat_star) #Changing the negative element for numerical issue

        sat_aug = np.append(sat_star, sat_star[-1])
        p_liq = (self.p_gas - self.p_thresh*sat_aug**(-1/lambd))/self.p_thresh

        dpliq_dz = np.diff(p_liq)

        k_rl = sat_star**3.4

        q_liq = -k_rl * dpliq_dz  # Non-dimensional liquid flux

        sinf = self.sat_inf
        sat_fun = (1 - sinf) / (sat_star*(1 - sinf) + sinf)
        advection_vel = q_liq * sat_fun

        conc_bound = conc_star[0]  # dC/dt|_{z=0} = 0
        # conc_bound = np.zeros(conc_star.shape[1])  # F_{1 - 1/2} = 0
        # conc_bound = (np.zeros(conc_star.shape[1]) - self.conc_mean_init) / \
        #     (self.rho_j - self.conc_mean_init)

        flux_sat = upwind_fvm(q_liq, boundary_cond=0)
        # flux_conc = upwind_fvm((advection_vel * conc_star.T).T,
        #                         boundary_cond=conc_bound)
        flux_conc = upwind_fvm(conc_star, boundary_cond=conc_bound)

        numerical_fluxes = np.column_stack((flux_sat, flux_conc))

        dstates_dtheta = -np.diff(numerical_fluxes, axis=0).T / self.delta_z
        dstates_dtheta[1:] = dstates_dtheta[1:] * advection_vel

        return dstates_dtheta.T.ravel()

    def solve_unit(self, deltaP, runtime, p_atm=101325,
                   verbose=True):
        """Initialize and integrate the deliquoring model.

        Parameters
        ----------
        deltaP : float
            Applied pressure drop across the cake and filter medium [Pa].
        runtime : float
            Physical duration to simulate [s].
        p_atm : float, optional
            Ambient pressure at the outlet face [Pa].
        verbose : bool, optional
            If False, suppress solver output [-].

        Returns
        -------
        theta : ndarray
            Non-dimensional solver time grid [-].
        states : ndarray
            Flattened reduced saturation and concentration state history [-].

        Notes
        -----
        ``Solid_1.x_distrib`` is stored on the PharmaPy solid grid in
        micrometers [um]. The capillary and threshold-pressure correlations use
        particle diameters in meters [m], while CSD-weighted averages integrate
        over the stored number distribution ``Solid_1.distrib`` [#/m**3/um] on
        the micrometer grid.
        """

        # Solid properties
        size_grid_um = self.Solid_1.x_distrib  # [um]
        diam_i = size_grid_um * 1e-6  # [m]
        csd = self.Solid_1.distrib  # [#/m**3/um]
        mom_zero = self.Solid_1.moments[0]  # [#/m**3]
        alpha = self.CakePhase.alpha  # [m/kg]

        # Irreducible saturation
        epsilon = self.Solid_1.getPorosity()

        rho_liq_per_node = self.Liquid_1.getDensity()
        rho_liq = np.mean(rho_liq_per_node)
        self.visc_liq_per_node = self.Liquid_1.getViscosity()
        self.visc_liq = np.mean(self.visc_liq_per_node)
        surf_tens_per_node = self.Liquid_1.getSurfTension()
        surf_tens = np.mean(surf_tens_per_node)
        s_inf = get_sat_inf(diam_i, csd, deltaP, epsilon, self.cake_height,
                            mom_zero, (surf_tens, rho_liq))
        _validate_irreducible_saturation(
            s_inf, deltaP, diam_i, "DeliquoringStep"
        )

        self.sat_inf = s_inf

        # Threshold pressure
        # Destro et al. (2021), Eq. 16, reports the Wakeman threshold
        # coefficient for completely wettable cakes.
        p_thresh_coeff = 4.6  # [-]
        p_thresh_i = (
            p_thresh_coeff * (1 - epsilon) * surf_tens / epsilon / diam_i
        )  # [Pa]
        p_thresh = (
            trapezoidal_rule(size_grid_um, p_thresh_i*csd) / mom_zero
        )  # [Pa]

        self.p_thresh = p_thresh

        rho_s = self.Solid_1.getDensity()
        k_perm = 1 / alpha / rho_s / (1 - epsilon)

        # Gas pressure
        deltaP_media = deltaP*self.resist_medium / \
            (alpha*rho_s*self.cake_height*(1 - epsilon) + self.resist_medium)

        pgas_out = p_atm + deltaP - deltaP_media

        self.p_gas = np.linspace(pgas_out, p_atm, self.num_nodes + 1)

        # ---------- Initial states
        # Saturation # TODO: read from cake nodes
        sat_initial = np.ones(self.num_nodes) * self.CakePhase.saturation[0]
        sat_red_init = (sat_initial - s_inf) / (1 - s_inf)

        # Concentration
        self.rho_j = self.Liquid_1.getDensityPure()[0]
        # self.rho_j = np.ones_like(self.rho_j)
        conc_upstream = self.CakePhase.Liquid_1.mass_conc

        z_dim = self.z_centers * self.cake_height
        conc_init = define_initial_state(state=conc_upstream, z_after=z_dim,
                     z_before=self.CakePhase.z_external, indexed_state=True)

        # if conc_upstream.ndim == 1:     # also for saturation : Daniel
        #     z_dim = self.z_centers * self.cake_height
        #     conc_liq = self.Liquid_1.mass_conc  # Lets check how the mass conc is calculated: Daniel
        #     conc_init = np.tile(conc_liq, (self.num_nodes, 1))
        # elif conc_upstream.ndim == 2:
        #     z_dim = self.z_centers * self.cake_height
        #     interp = SplineInterpolation(self.CakePhase.z_external, conc_upstream)
        #     conc_init = interp.evalSpline(z_dim)

        self.conc_mean_init = np.zeros_like(conc_init)
        for i in range(len(self.Liquid_1.name_species)):
            self.conc_mean_init[:,i] = trapezoidal_rule(z_dim, conc_init[:,i]) / \
            self.cake_height

        conc_star_init = (conc_init - self.conc_mean_init) / \
            (self.rho_j - self.conc_mean_init)

        y_zero = np.column_stack((sat_red_init, conc_star_init)).ravel()

        model = Explicit_Problem(self.unit_model, y0=y_zero, t0=0)

        # Solve model
        model.name = 'Deliquoring PDE'
        sim = CVode(model)
        sim.linear_solver = 'SPGMR'

        self.theta_conv = k_perm*p_thresh / \
            self.visc_liq/self.cake_height**2/epsilon/(1 - s_inf)

        t_final = runtime * self.theta_conv
        
        if not verbose:
          sim.verbosity = 50
        
        if t_final < eps:
            t_final = 1
            
        theta, states = sim.simulate(t_final)

        self.rho_s = rho_s
        self.retrieve_results(theta, states)

        return theta, states

    def flatten_states(self):
        pass

    def retrieve_results(self, theta, states):
        num_species = self.Liquid_1.num_species

        time = theta/ self.theta_conv

        indexes = {key: self.states_di[key].get('index', None)
                   for key in self.name_states}

        dp = {}

        dp_reduced= unpack_discretized(states, self.dim_states, self.name_states,
                                indexes=indexes)

        s_red = dp_reduced['saturation']

        # s_red = states[:, ::num_species + 1]
        satProf = s_red * (1 - self.sat_inf) + self.sat_inf

        conc_diff = self.rho_j - self.conc_mean_init

        concPerSpecies = {}
        mass_j = {}
        mass_bar_j = {}

        porosity = self.CakePhase.porosity

        for ind, name in enumerate(indexes['mass_conc']):
            conc_sp = dp_reduced['mass_conc'][name]

            conc_sp = conc_sp * conc_diff[:,ind] + self.conc_mean_init[:,ind]
            mass_sp = porosity * satProf * conc_sp
            massbar = porosity * satProf * conc_sp / \
                ((1 - porosity)*self.rho_s + porosity*satProf*self.rho_j[ind])

            concPerSpecies[name] = conc_sp
            mass_j[name] = mass_sp
            mass_bar_j[name] = massbar

        self.timeProf = time
        self.satProf = satProf

        dp['time'] = time
        dp['z'] = self.z_centers
        dp['saturation'] = self.satProf
        dp['mass_conc'] =  concPerSpecies

        self.result = DynamicResult(self.states_di, self.fstates_di, **dp)

        self.mean_sat = trapezoidal_rule(self.z_centers, s_red.T) * \
            (1 - self.sat_inf) + self.sat_inf

        # dp['mean_saturation_value'] = self.mean_sat


        concPerVolElement = {}
        concPerVolElement = dp_reduced['mass_conc']

        for name in indexes['mass_conc']:
            concPerVolElement[name] = concPerSpecies[name] * conc_diff[:, ind] \
                + self.conc_mean_init[:,ind]

        self.concPerSpecies = concPerSpecies
        self.massCompPerCakeUnitVolume = mass_j
        self.massjPerMassCake = mass_bar_j
        self.concPerVolElement = concPerVolElement

        last_state = {}
        for name in self.concPerSpecies.keys():
            last_state[name] = self.concPerSpecies[name][-1][-1]

        self.mass_conc= list(last_state.values())

        self.Liquid_1.updatePhase(mass_conc=self.mass_conc)


        liquid_out = copy.deepcopy(self.Liquid_1)
        solid_out = copy.deepcopy(self.Solid_1)

        self.Outlet = self.CakePhase
        self.CakePhase.saturation = self.satProf[-1]
        self.CakePhase.z_external = self.z_centers
        self.Outlet.Phases = (liquid_out, solid_out)
        self.outputs = states

    def plot_profiles(self, fig_size=None, mean_sat=True,
                      time=None, z_star=None, jump=20, pick_comp=None):
        """

        Parameters
        ----------
        fig_size : tuple (optional, default = None)
            Size of the figure to be populated.
        mean_sat : bool (optional, default = True)
            Boolean value indicating whether the
            averaged saturation value is plotted over the time.
        time : float (optional, default = None)
            Integer value indicating the time in which
            axial saturation value is calculated.
        z_star : float (optional, default = None)
            The axial coordinate of cake of interest to be calculated.
        jump : int (optional, default = 20)
            The number of sample to be skipped on the plot.
        pick_comp : list (optional, default = None)
            List contains the index of compounds of interest
            in the liquid phase to be calculated.

        """
        fig, axis = plt.subplots(2, 1, figsize=fig_size, sharex=True)

        if pick_comp is None:
            pick_comp = np.arange(len(self.Liquid_1.name_species))

        if mean_sat:
            axis[0].plot(self.timeProf, self.mean_sat)
            axis[0].set_xlabel('time (s)')
            axis[0].set_ylabel(r'$\bar{S}$')

            # axis[1].plot()

        elif time is None and z_star is None:
            # Saturation
            sat_plot = self.satProf.T[:, ::jump]
            num_lines = sat_plot.shape[1]
            axis[0].plot(self.z_centers, sat_plot)

            scale = np.linspace(0, 1, num_lines)
            colors = plt.cm.YlGnBu(scale)

            [line.set_color(color) for line, color in zip(axis[0].lines, colors)]

            axis[0].set_ylabel('$S$')

            # Concentration
            my_colors = plt.rcParams['axes.prop_cycle']()
            alphas = np.linspace(0.1, 1, num_lines)
            for ind in pick_comp:
                conc = self.concPerSpecies[ind]
                color = next(my_colors)
                axis[1].plot(self.z_centers, conc[::jump].T, **color)

                [line.set_alpha(alphas[idx])
                 for idx, line in enumerate(axis[1].lines[-num_lines:])]

                axis[1].lines[-1].set_label(self.Liquid_1.name_species[ind])

            axis[1].legend(loc='best')
            axis[1].set_ylabel('$C$ $(\mathregular{kg \ m^{-3}})$')
            axis[1].set_xlabel('$z/L$')

        elif time is not None:

            idx_time = np.argmin(abs(self.timeProf - time))
            profiles = [self.concPerSpecies[ind][idx_time]
                        for ind in range(self.Liquid_1.num_species)]

            axis[1].plot(self.z_centers, np.column_stack(profiles))

            axis[1].legend(self.Liquid_1.name_species, loc='best')

            axis[1].set_xlabel('$z/L$')
            axis[1].set_ylabel('$C_j$ ($\mathregular{kg \ m^{-3}}$)')
            axis[1].text(1, 1.04, '$t = %.1f$ s' % self.timeProf[idx_time],
                      ha='right', transform=axis[1].transAxes)

        elif z_star is not None:
            idx_z = np.argmin(abs(self.z_centers - z_star))
            profiles = self.concPerVolElement[idx_z]

            axis[1].plot(self.timeProf, profiles)
            axis[1].set_xlabel('time (s)')
            axis[1].set_ylabel('$C_j$ ($\mathregular{kg \ m^{-3}}$)')
            axis[1].text(1, 1.04, '$z^* = %.1f$' % self.z_centers[idx_z],
                         ha='right', transform=axis.transAxes)

        for ax in axis:
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        return fig, axis

    def animate_unit(self, filename=None, title=None, pick_idx=None,
                     step_data=2, fps=5):
        if self.timeProf is None:
            raise RuntimeError

        if pick_idx is None:
            pick_idx = np.arange(self.Liquid_1.num_species)
            names = self.name_species
        else:
            pick_idx = pick_idx
            names = [self.name_species[ind] for ind in pick_idx]

        fig_anim, (ax_anim, ax_sat) = plt.subplots(2, 1, figsize=(4, 5))
        ax_anim.set_xlim(0, self.z_grid.max())
        fig_anim.suptitle(title)

        fig_anim.subplots_adjust(left=0, bottom=0, right=1, top=1,
                                 wspace=None, hspace=None)

        conc_min = np.vstack(self.concPerVolElement)[:, pick_idx].min()
        conc_max = np.vstack(self.concPerVolElement)[:, pick_idx].max()

        conc_diff = conc_max - conc_min

        ax_anim.set_ylim(conc_min - 0.03*conc_diff, conc_max + conc_diff*0.03)

        ax_anim.set_xlabel('$z/L$')
        ax_anim.set_ylabel('$C_j$ (mol/L)')

        sat = self.satProf
        sat_diff = sat.max() - sat.min()
        ax_sat.set_xlim(0, self.z_grid.max())
        ax_sat.set_ylim(sat.min() - sat_diff*0.03, sat.max() + sat_diff*0.03)

        ax_sat.set_xlabel('$z/L$')
        ax_sat.set_ylabel('$S$')

        def func_data(ind):
            conc_species = []
            for comp in pick_idx:
                conc_species.append(self.concPerSpecies[comp][ind])

            conc_species = np.column_stack(conc_species)
            return conc_species

        lines_conc = ax_anim.plot(self.z_centers, func_data(0))
        line_temp, = ax_sat.plot(self.z_centers, sat[0])

        time_tag = ax_anim.text(
            1, 1.04, '$time = {:.2f}$ s'.format(self.timeProf[0]),
            horizontalalignment='right',
            transform=ax_anim.transAxes)

        def func_anim(ind):
            f_vals = func_data(ind)
            for comp, line in enumerate(lines_conc):
                line.set_ydata(f_vals[:, comp])
                line.set_label(names[comp])

            line_temp.set_ydata(sat[ind])

            ax_anim.legend()
            fig_anim.tight_layout()

            time_tag.set_text('$t = {:.2f}$ s'.format(self.timeProf[ind]))

        frames = np.arange(0, len(self.timeProf), step_data)
        animation = FuncAnimation(fig_anim, func_anim, frames=frames,
                                  repeat=True)

        writer = FFMpegWriter(fps=fps, metadata=dict(artist='Me'),
                              bitrate=-1)

        suff = '.mp4'

        animation.save(filename + suff, writer=writer)

        return animation, fig_anim, (ax_anim, ax_sat)


class Filter:
    def __init__(self, station_diam: float,
                 alpha: Optional[Union[float, Callable[[float], float]]] = None,
                 resist_medium: Union[float, Callable[[float], float]] = 1e9,
                 log_params: bool = False) -> None:
        """Configure a batch filter with physical resistance parameters.

        Parameters
        ----------

        station_diam : float
            Diameter of the filter's cross section [m]
        alpha : float or callable, optional
            Specific cake resistance [m/kg], or a function of pressure drop
            [Pa] returning [m/kg]. None estimates resistance from the solid.
        resist_medium : float or callable, optional
            Medium resistance [1/m], or a function of pressure drop [Pa]
            returning [1/m]. Default 1e9 [1/m] is the existing medium value.
        log_params : bool, optional
            Interpret only estimation model_params as natural logarithms of
            SI numerical values. Constructor values and callable results
            always use physical SI units, including when this is True.

        Raises
        ------
        ValueError
            If numeric alpha is nonfinite or nonpositive, or numeric medium
            resistance is nonfinite or negative. A zero medium resistance is
            valid for simulation, but cannot seed logarithmic estimation.
        """
        if alpha is not None and not callable(alpha) and (not np.isfinite(alpha) or alpha <= 0):
            raise ValueError("Cake resistance alpha [m/kg] must be finite and strictly positive.")
        if not callable(resist_medium) and (not np.isfinite(resist_medium) or resist_medium < 0):
            raise ValueError("Medium resistance [1/m] must be finite and nonnegative.")
        self._Phases = None
        self.material_from_upstream = False

        self.r_medium = resist_medium
        self.station_diam = station_diam
        self.area_filt = station_diam**2 * np.pi/4  # [m^2]

        self.nomenclature()

        self.oper_mode = 'Batch'
        self.is_continuous = False

        self.alpha = alpha

        self.log_params = log_params

        self.elapsed_time = 0

        self.name_params = ['alpha', 'Rm']
        self.mask_params = [True, True]
        self.states_uo = ['mass_filtrate', 'mass_retained']

        self.deltaP = None
        self.outputs = None

    @property
    def Phases(self):
        return self._Phases

    @Phases.setter
    def Phases(self, phases) -> None:
        """Attach the slurry and retain physical filtration parameters.

        Parameters
        ----------
        phases : Slurry or sequence of phase objects
            Slurry or liquid/solid phases with attached inventories [kg].

        Notes
        -----
        params retains alpha [m/kg] and medium resistance [1/m], or their
        pressure-dependent callables, until solve_unit resolves them. The
        log_params option transforms only supplied estimation parameters.

        Raises
        ------
        RuntimeError
            If phases is neither a Slurry nor a phase sequence.
        """
        if isinstance(phases, (list, tuple)):
            self._Phases = phases

            self.SlurryPhase = Slurry()
            self.SlurryPhase.Phases = list(phases)
        elif phases.__class__.__name__ == 'Slurry':
            self.SlurryPhase = phases
            self._Phases = phases.Phases
        else:
            raise RuntimeError('Please provide a list or tuple of phases '
                               'objects')
        classify_phases(self)  # Enumerate phases: Liquid_1,..., Solid_1, ...

        epsilon = self.Solid_1.getPorosity(diam_filter=self.station_diam)
        dens_sol = self.Solid_1.getDensity()
        if self.alpha is None:
            self.alpha = get_alpha(self.Solid_1, sphericity=1,
                                    porosity=epsilon,
                                    rho_sol=dens_sol)

        self.params = (self.alpha, self.r_medium)  # [m/kg, 1/m] or callables

        self.states_di = {
            'mass_filtrate': {'dim': 1, 'units': 'kg', 'type': 'diff'},
            'mass_liquid': {'dim': 1, 'units': 'kg', 'type': 'diff'},
            }

        self.fstates_di = {
            'mass_cake_dry': {'dim': 1, 'units': 'kg', 'type': 'alg'},
            'mass_cake_wet': {'dim': 1, 'units': 'kg', 'type': 'alg'}

            }

        self.name_states = list(self.states_di.keys())
        self.dim_states = [a['dim'] for a in self.states_di.values()]

        self.nomenclature()

    def nomenclature(self):
        self.names_states_in = ['mass', 'temp', 'mass_frac', 'total_distrib']
        self.names_states_out = self.names_states_in

    def get_inputs(self):
        self.Inlet

    def unit_model(self, time, states, sw):
        if self.oper_mode == 'Batch':
            mass_liquid = states
            material_balance = self.material_balance(time, mass_liquid,
                                                     self.deltaP)

        # print(material_balance)
        # print(time)
        # print(mass_liquid)

        return material_balance

    def material_balance(self, time, masses, dP):
        mass_filtr, mass_up = masses
        rho, visc, tens, rho_s = self.physical_props
        c_solids = self.c_solids

        alpha, resist = self.params

        cake_term = alpha * c_solids * mass_filtr / (self.area_filt * rho)**2
        filt_term = resist / (self.area_filt * rho)

        deriv = dP / visc / (cake_term + filt_term)

        dmass_dt = [deriv, -deriv]

        return dmass_dt

    def __state_event(self, time, states, switch):
        events = []
        if switch[0]:
            event_mass = self.mass_crit - states[0]
            events.append(event_mass)

        return np.array(events)

    def __handle_event(self, solver, event_info):
        state_event = event_info[0]

        if state_event:
            raise TerminateSimulation

    @property
    def param_seed(self) -> np.ndarray:
        """Return numeric resistances in the optimizer's parameter space.

        Returns
        -------
        ndarray
            Seed with shape (2,), ordered as alpha and medium resistance.
            With log_params=False, units are [m/kg] and [1/m]. With
            log_params=True, entries are natural logarithms of the SI
            numerical values [-]. Uses configured alpha and r_medium,
            independently of params from prior optimizer evaluations.

        Raises
        ------
        ValueError
            If a configured resistance is callable or nonfinite, alpha is
            nonpositive, or medium resistance is negative. A zero medium
            resistance is allowed only when log_params=False.

        Notes
        -----
        A prior solve may have resolved a callable into numeric params;
        the underlying callable still makes that resistance non-estimable.
        """
        if any(callable(value) for value in (self.alpha, self.r_medium)):
            raise ValueError(
                "Callable resistances are not estimable; configure numeric "
                "alpha [m/kg] and medium resistance [1/m] before estimation.")
        physical_params = np.array((self.alpha, self.r_medium), dtype=float)  # [m/kg, 1/m]
        if (not np.all(np.isfinite(physical_params)) or physical_params[0] <= 0
                or physical_params[1] < 0 or (self.log_params and physical_params[1] == 0)):
            raise ValueError(
                "Estimation seeds require finite alpha [m/kg] (strictly positive) "
                "and medium resistance [1/m] (nonnegative); medium resistance "
                "must be strictly positive when log_params=True; "
                f"received {physical_params}.")
        return np.log(physical_params) if self.log_params else physical_params

    def reset(self) -> None:
        """Reset the batch clock while preserving physical parameters.

        Notes
        -----
        log_params affects only the estimation-parameter transform in
        solve_unit; resetting does not logarithmically transform params.
        """
        self.elapsed_time = 0  # [s]

    def solve_unit(self, runtime: Optional[float] = None,
                   time_grid: Optional[ArrayLike] = None, deltaP: float = 1e5,
                   slurry_div: float = 1, verbose: bool = True,
                   model_params: Optional[ArrayLike] = None,
                   sundials_opts: Optional[dict] = None) -> tuple:
        """Integrate a batch with pressure-resolved filtration parameters.

        Parameters
        ----------
        runtime : float, optional
            Batch duration [s]. Without model_params, an omitted duration
            integrates to cake completion. Estimation requires a bounded
            runtime or time_grid; times beyond completion hold the plateau.
        time_grid : array-like, optional
            Output times relative to this batch's start [s], shape (num_times,).
            Takes precedence over runtime; caller data is not modified. With
            model_params, return exactly these samples or raise RuntimeError
            if CVode omits any. The initial condition is returned only if requested.
        deltaP : float, optional
            Pressure drop [Pa], default 1e5 Pa (one bar).
        slurry_div : float, optional
            Number of equal feed portions [-], default one (the full batch).
        verbose : bool, optional
            Enable solver output.
        model_params : array-like, optional
            Override (alpha [m/kg], medium resistance [1/m]). When log_params
            is True, these are natural logarithms of the SI numerical values.
            Overrides take precedence over the constructor's parameters.
        sundials_opts : dict, optional
            CVode options, including relative [-] and absolute [kg] tolerances.

        Returns
        -------
        time : ndarray
            Absolute output times [s], shape (num_times,).
        states : ndarray
            Filtrate and remaining liquid masses [kg], shape (num_times, 2).

        Raises
        ------
        ValueError
            If the feed has no positive solid mass, or lacks enough liquid
            to saturate the packed cake and leave positive filtrate volume.
            These feed checks precede changes to deltaP and params. Also
            raised before any mutation if estimation supplies neither runtime
            nor time_grid.
        RuntimeError
            If CVode omits requested estimation samples. Tighten rtol/atol
            in sundials_opts; output-grid coverage depends on solver tolerances.

        Notes
        -----
        Each call filters a fresh portion of the attached slurry, starting at
        elapsed_time. Callable resistances receive deltaP [Pa] and return SI
        values even when log_params is True; they are retained for later
        batches at different pressures. Only model_params supplied for
        estimation are exponentiated when log_params is True.
        The numeric tuple in params is the one used by the balance and the
        analytical completion-time calculation. Estimation integrates only
        up to completion and holds later requested samples at mass_crit [kg]
        of filtrate and the remaining cake liquid [kg]. This permits optimizer
        trials that finish before the last observation. Relative durations
        come directly from the input grid, avoiding cancellation against
        elapsed_time. No warning is emitted for the physical plateau. CVode
        can omit samples inside its final step, especially on dense grids at
        default tolerances. Estimation checks exact time coverage before
        returning; callers may need tighter rtol/atol in sundials_opts.
        """
        if model_params is not None and runtime is None and time_grid is None:
            raise ValueError("Filter estimation requires runtime or time_grid; "
                             "an unbounded estimation solve is not supported.")
        if self.Solid_1.mass <= 0:
            raise ValueError(
                "Filter requires positive solid mass [kg]; "
                "use a liquid transfer for a zero-solid feed.")
        epsilon = self.Solid_1.getPorosity(diam_filter=self.station_diam)  # [-]
        dens_sol = self.Solid_1.getDensity()  # [kg/m**3]

        solid_conc = self.SlurryPhase.getSolidsConcentr()
        solid_conc = max(0, solid_conc)

        # Initial state
        vol_slurry = self.SlurryPhase.vol / slurry_div
        frac_liq = self.SlurryPhase.getFractions()[0]

        vol_liq_cake = vol_slurry * solid_conc/dens_sol * epsilon/(1 - epsilon)
        vol_liq_slur = vol_slurry * frac_liq

        vol_filtrate = vol_liq_slur - vol_liq_cake  # [m**3]
        if vol_filtrate <= 0:
            raise ValueError(
                "Slurry liquid must saturate the packed cake and leave "
                "positive filtrate volume; "
                f"slurry liquid volume={vol_liq_slur} m**3, "
                f"cake liquid volume={vol_liq_cake} m**3. "
                "Add liquid or reduce solids.")

        if model_params is not None:
            resolved_params = (np.exp(model_params) if self.log_params
                               else model_params)  # [m/kg, 1/m]
        else:
            resolved_params = (
                self.alpha(deltaP) if callable(self.alpha) else self.alpha,
                self.r_medium(deltaP) if callable(self.r_medium) else self.r_medium,
            )  # [m/kg, 1/m]
        self.params = tuple(resolved_params)  # [m/kg, 1/m]
        alpha, resistance = self.params  # [m/kg], [1/m]

        self.deltaP = deltaP  # [Pa]

        self.c_solids = vol_slurry * solid_conc / vol_filtrate

        # Physical properties
        dens_liq = self.Liquid_1.getDensity()
        visc_liq = self.Liquid_1.getViscosityMix()
        surf_liq = self.Liquid_1.getSurfTension()

        self.physical_props = [dens_liq, visc_liq, surf_liq, dens_sol]
        self.mass_crit = vol_filtrate * dens_liq

        # Initial values
        mass_filtr_init = 0
        mass_up_init = vol_liq_slur * dens_liq

        mass_init = [mass_filtr_init, mass_up_init]

        # Solve ODE
        problem = Explicit_Problem(self.unit_model, y0=mass_init,
                                   t0=self.elapsed_time, sw0=[True])

        # State event
        if model_params is None:
            problem.state_events = self.__state_event
            problem.handle_event = self.__handle_event

        solver = CVode(problem)

        if not verbose:
            solver.verbosity = 50

        if sundials_opts is not None:
            for name, val in sundials_opts.items():
                setattr(solver, name, val)

                if name == 'time_limit':
                    solver.report_continuously = True

        relative_grid = None  # [s], caller's batch-relative output times
        if time_grid is not None:
            relative_grid = np.asarray(time_grid, dtype=float)  # [s]
            duration = relative_grid[-1]  # [s], before adding the batch clock
            time_grid = relative_grid + self.elapsed_time  # [s], absolute output times
            final_time = time_grid[-1]  # [s]
        elif runtime is None:
            final_time = 1e10  # [s], existing event-driven integration horizon
        else:
            duration = runtime  # [s]
            final_time = runtime + self.elapsed_time  # [s]

        self.time_filt = visc_liq/self.deltaP * (
            alpha*self.c_solids/2 * (vol_filtrate/self.area_filt)**2 +
            resistance * (vol_filtrate/self.area_filt))  # [s]
        holds_plateau = model_params is not None and duration >= self.time_filt
        integration_grid = time_grid  # [s]
        integration_end = final_time  # [s]
        if holds_plateau:
            integration_end = self.elapsed_time + self.time_filt  # [s]
            if relative_grid is not None:
                before_completion = relative_grid < self.time_filt
                # Only requested observations before completion need integration.
                # Appending a nearby completion point can make CVode omit the
                # preceding observation; the exact plateau needs no ODE sample.
                integration_grid = time_grid[before_completion]  # [s]
                integration_end = (integration_grid[-1] if len(integration_grid)
                                   else self.elapsed_time)  # [s]

        if integration_end == self.elapsed_time:
            time = np.array([self.elapsed_time])  # [s], initial condition only
            states = np.array([mass_init])  # [kg]
        else:
            time, states = solver.simulate(integration_end, ncp_list=integration_grid)
        time = np.asarray(time, dtype=float)  # [s], consistent return type on all routes
        states = np.asarray(states, dtype=float)  # [kg]
        if model_params is not None and relative_grid is not None:
            requested_indices = np.searchsorted(time, integration_grid)
            if (np.any(requested_indices == len(time)) or
                    not np.array_equal(time[requested_indices], integration_grid)):
                raise RuntimeError(
                    "CVode did not return every requested Filter output time; "
                    "set tighter rtol/atol in sundials_opts to cover the grid.")
            # Exclude CVode's unrequested initial row, including an empty
            # pre-completion grid that will be filled entirely by the plateau.
            time = time[requested_indices]  # [s]
            states = states[requested_indices]  # [kg]
        if holds_plateau:
            plateau = np.array([self.mass_crit, mass_up_init - self.mass_crit])  # [kg, kg]
            if relative_grid is not None:
                plateau_times = time_grid[~before_completion]  # [s], preserve caller sampling
                time = np.concatenate((time, plateau_times))  # [s]
                states = np.vstack((states, np.tile(plateau, (len(plateau_times), 1))))  # [kg]
            else:
                states[-1] = plateau
                if duration > self.time_filt:
                    time = np.append(time, final_time)  # [s], bounded runtime endpoint
                    states = np.vstack((states, plateau))  # [kg]
        if model_params is not None:
            # Cap numerical overshoot while preserving total liquid inventory.
            states[:, 0] = np.minimum(states[:, 0], self.mass_crit)  # [kg]
            states[:, 1] = mass_up_init - states[:, 0]  # [kg]

        self.retrieve_results(time, states, dens_liq, dens_sol, epsilon,
                              mass_solids=self.Solid_1.mass / slurry_div)

        return time, states

    def retrieve_results(self, time: ArrayLike, states: np.ndarray,
                         dens_liq: float, dens_sol: float,
                         epsilon: float, mass_solids: float) -> None:
        """Store filtrate histories and the recovered cake inventory.

        Parameters
        ----------
        time : array-like
            Absolute output times [s], shape (num_times,).
        states : ndarray
            Filtrate and remaining liquid masses [kg], shape (num_times, 2).
        dens_liq, dens_sol : float
            Liquid and solid mixture densities [kg/m**3].
        epsilon : float
            Packed-cake porosity [-].
        mass_solids : float
            Solid inventory in this feed portion [kg]. Recovery is the
            filtrate fraction of mass_crit, capped at one.

        Notes
        -----
        Scale the feed's total particle population [#/um] by the recovered
        solid mass fraction. This preserves particle sizes while reconciling
        attached mass and distribution for divided or incomplete batches.
        The attached liquid is the remaining batch liquid; incomplete
        filtration may still leave liquid above the cake. Recovered dry mass
        is capped at the attached solid inventory, so solver-tolerance
        overshoot cannot make the recovered population fraction exceed one.
        """
        self.timeProf = np.array(time)
        self.massProf = states

        dp = unpack_states(states, self.dim_states, self.name_states)
        dp['time'] = np.asarray(time)

        portion_recovery = np.minimum(states[:, 0] / self.mass_crit, 1.)  # [-]
        cake_dry = mass_solids * portion_recovery  # [kg], exact inventory on the plateau
        cake_wet = cake_dry * (1 + epsilon/(1 - epsilon) * dens_liq/dens_sol)

        dp['mass_cake_dry'] = cake_dry
        dp['mass_cake_wet'] = cake_wet

        self.result = DynamicResult(self.states_di, self.fstates_di, **dp)

        solid_cake = copy.deepcopy(self.Solid_1)
        recovered_fraction = cake_dry[-1] / self.Solid_1.mass  # [-]
        cake_distribution = self.Solid_1.distrib * recovered_fraction  # [#/um]
        solid_cake.updatePhase(mass=cake_dry[-1], distrib=cake_distribution)

        liquid_cake = copy.deepcopy(self.Liquid_1)
        # Deferred outlet/inventory work: the filtrate outlet is not constructed;
        # incomplete filtration can leave liquid above the cake.
        liquid_cake.updatePhase(mass=self.massProf[-1, 1])

        self.Outlet = Cake()
        self.Outlet.Phases = (liquid_cake, solid_cake)

        self.outputs = np.concatenate(([states[-1, 1], self.Liquid_1.temp],
                                       self.Liquid_1.mass_frac,
                                       solid_cake.distrib))  # [kg], [K], [-], [#/um]

        self.outputs = np.atleast_2d(self.outputs)

        self.elapsed_time = time[-1]  # [s], absolute batch endpoint

    def flatten_states(self):
        pass

    def paramest_wrapper(self, params: ArrayLike, time_vals: ArrayLike,
                         modify_phase=None, modify_controls=None,
                         run_args: dict = {}) -> np.ndarray:
        """Evaluate filtrate masses using estimation-space parameters.

        Parameters
        ----------
        params : array-like
            Optimizer vector (alpha, medium resistance), shape (2,).
            With log_params=False, units are [m/kg] and [1/m]; otherwise
            supply natural logarithms of the SI numerical values [-].
            param_seed provides this representation from configured resistances.
        time_vals : array-like
            Requested output times relative to batch start [s], shape
            (num_times,). No unrequested initial-condition row is returned.
        modify_phase, modify_controls : dict, optional
            Only None or empty mappings are supported; non-empty modifiers
            are rejected because this wrapper does not apply them.
        run_args : dict, optional
            Additional solve_unit options. May override a resolved deltaP
            [Pa] and default verbose=False, or supply slurry_div [-] and
            sundials_opts. time_grid and model_params are supplied by this
            wrapper. The mapping is not modified. An unresolved or explicitly
            None deltaP is omitted, selecting the solve_unit default (1e5 Pa).

        Returns
        -------
        ndarray
            Filtrate mass history [kg], shape (num_times,).

        Raises
        ------
        ValueError
            If modify_phase or modify_controls is non-empty.
        RuntimeError
            If CVode omits a requested time. Supply tighter rtol/atol via
            run_args['sundials_opts'] (wrapper_kwargs['sundials_opts'] in SimExec).

        Notes
        -----
        Each evaluation resets the batch clock. solve_unit transforms the
        optimizer vector into physical resistances exactly once. Exact
        output-grid coverage depends on CVode tolerances and is verified
        before residual calculation; omitted samples raise an actionable error.
        """
        for name, modifier in (('modify_phase', modify_phase),
                               ('modify_controls', modify_controls)):
            if modifier:
                raise ValueError(
                    f"{name} is not supported by Filter.paramest_wrapper; "
                    "configure the Filter phases directly and pass solve options "
                    "through wrapper_kwargs/run_args instead.")
        self.reset()
        solve_options = {'verbose': False}
        if self.deltaP is not None:
            solve_options['deltaP'] = self.deltaP  # [Pa], previously resolved pressure
        solve_options.update(run_args)
        if solve_options.get('deltaP') is None:
            solve_options.pop('deltaP', None)
        _, states = self.solve_unit(time_grid=time_vals, model_params=params,
                                    **solve_options)
        return states[:, 0]

    def plot_profiles(self, time_div=1, black_white=False, **fig_kwargs):
        """

        Parameters
        ----------

        fig_size : tuple (optional, default = None)
            Size of the figure to be populated.
        time_div : float (optional, default = 1)
            The float value used to scale time value in
            plotting by dividing the simulated time.
        black_white : bool (optional, default = False)
            Boolean value indicating whether the figure
            is presented in black and white style.

        """

        mass_filtr, mass_up = self.massProf.T
        time_plot = self.timeProf / time_div

        fig, ax = plt.subplots(1, 2, **fig_kwargs, sharex=True)

        # Liquid mass
        if black_white:
            ax[0].plot(time_plot, mass_filtr, time_plot, mass_up, '--',
                       color='k')
            ax[1].plot(time_plot, self.cake_dry, time_plot, self.cake_wet,
                       '--', color='k')
        else:
            ax[0].plot(time_plot, mass_filtr, time_plot, mass_up, '--')
            ax[1].plot(time_plot, self.result.mass_cake_dry,
                       time_plot, self.result.mass_cake_wet,
                       '--')

        ax[0].set_ylabel('mass liquid (kg)')
        ax[0].legend(('filtrate', 'hold-up'))

        # Cake
        ax[1].set_ylabel('mass cake (kg)')
        ax[1].legend(('dry cake', 'wet cake'), loc='best')

        for axis in ax:
            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)

            axis.xaxis.set_minor_locator(AutoMinorLocator(2))
            axis.yaxis.set_minor_locator(AutoMinorLocator(2))

        if time_div == 1:
            ax[1].set_xlabel('time (s)')

        return fig, ax


def _exp_erfc(exponent: ArrayLike, argument: ArrayLike) -> np.ndarray:
    """Evaluate exp(exponent)*erfc(argument) without spurious overflow.

    Parameters
    ----------
    exponent, argument : array-like
        Dimensionless exponent and erfc argument [-], broadcastable together.

    Returns
    -------
    ndarray
        Product [-], with the broadcast shape of the inputs.

    Notes
    -----
    For b >= 0, use erfcx(b)*exp(a-b**2), since
    erfcx(b) = exp(b**2)*erfc(b). For b < 0, use the complementary
    identity erfc(b) = 2-erfc(-b), keeping its erfc argument nonnegative.
    This also avoids evaluating erfcx at negative arguments. The two washing
    callers have b >= 0 and a-b**2 <= 0, so no residual exponential overflow
    occurs there; the negative branch is retained only for generality.
    Underflow in vanishing exponential tails is allowed to yield zero or
    subnormal values; overflow of the product itself remains an error when
    enabled by the caller. These are algebraic identities, not fitted limits.
    """
    exponent, argument = np.broadcast_arrays(
        np.asarray(exponent, dtype=float), np.asarray(argument, dtype=float))  # [-]
    product = np.empty(exponent.shape)  # [-]
    nonnegative = argument >= 0
    negative = ~nonnegative
    with np.errstate(under='ignore'):
        product[nonnegative] = erfcx(argument[nonnegative]) * np.exp(
            exponent[nonnegative] - argument[nonnegative]**2)
        product[negative] = np.exp(exponent[negative]) * (2 - erfc(-argument[negative]))
    return product


class DisplacementWashing:
    def __init__(self, solvent_idx, num_nodes, diam_unit=None,
                 resist_medium=1e9, k_ads=0):

        """

        Parameters
        ----------

        solvent_idx : int
            Integer value indicating the index of the compounds
            used as solvent. Index correpond to the coumpounds
            order in the physical properties .json file.
        number_nodes : float
            Number of apatial discretization of the cake along the axial
            coordinate.
        diam_unit : float (optional, default=0.01)
            Diameter of the dryer's cross section [m]
        resist_medium : float (optional, default=2.22e9)
            Mesh resistance of filter in dryer. [m**-1]
        k_ads : float (optional, default = 0)
            Equilibirum coefficient between main flow concentration
            and overall partical concentration.Used to calculate adsoprtion
            factor.The default value '0' means no adsorption occurs on the solid phase.
            (Lapidus and Amundson, 1952)

        """
        self.max_exp = np.log(np.finfo('d').max)
        self.satur = 1
        self.num_nodes = num_nodes

        self.solvent_idx = solvent_idx
        self.k_ads = k_ads
        self.cross_area = np.pi / 4 * diam_unit**2
        self.diam_unit = diam_unit

        self.resist_medium = resist_medium

        self._Phases = None
        self._Inlet = None

        self.oper_mode = 'Batch'
        self.is_continuous = False

        self.nomenclature()

        self.outputs = None

    @property
    def Phases(self):
        return self._Phases

    @Phases.setter
    def Phases(self, phases):
        if isinstance(phases, (list, tuple)):
            self._Phases = phases

            self.CakePhase = Cake()
            self.CakePhase.Phases = list(phases)
        elif phases.__class__.__name__ == 'Cake':
            self.CakePhase = phases
            self._Phases = phases.Phases
        else:
            raise RuntimeError('Please provide a list or tuple of phases '
                               'objects')

        classify_phases(self)  # Enumerate phases: Liquid_1,..., Solid_1, ...

        self.__original_phase__ = copy.deepcopy(self.Liquid_1.__dict__)

        self.name_species = self.Liquid_1.name_species

        self.states_di = {
            'mass_conc' : {'index': self.name_species,
                         'dim':len(self.name_species), 'units': 'kg/m**3', 'type':'algebraic'}} # Order of the states matters

        self.fstates_di = {
            'mean_mass_conc' : {'index': self.name_species,
                     'dim': len(self.name_species), 'units': 'kg/m**3', 'type': 'alg'},
            'washing_time' : {#'index': index_z,
                              'dim':1, 'units': 's', 'type': 'alg'}}

        self.name_states = list(self.states_di.keys())
        self.dim_states = [a['dim'] for a in self.states_di.values()]

        self.nomenclature()
    @property
    def Inlet(self):
        return self._Inlet

    @Inlet.setter
    def Inlet(self, inlet):
        self._Inlet = inlet

    def get_diffusivity(self, vel: float, diff_pure: np.ndarray,
                        cake_height: Optional[float] = None) -> np.ndarray:
        """Evaluate axial dispersion from the volume-weighted Peclet number.

        Parameters
        ----------
        vel : float
            Superficial liquid velocity [m/s].
        diff_pure : ndarray
            Molecular diffusivity for each species [m**2/s].
        cake_height : float, optional
            Test hook overriding cake height [m] to isolate correlation
            branches. Production calls derive thickness from attached cake
            volume divided by the unit cross-sectional area.

        Returns
        -------
        ndarray
            Effective axial diffusivity by species [m**2/s].

        Raises
        ------
        ValueError
            If the particle population has no positive volume to normalize.

        Notes
        -----
        Destro's thesis (2021), section 4.3.5, Eq. 4.27 averages
        bin Peclet numbers ``vel * particle_size / diff_pure`` [-] with
        particle size in metres [m]. Eq. 4.26 is then applied ONCE to that
        average for each species; averaging bin diffusivities would change
        the nonlinear correlation. The corresponding article is Destro
        et al. (2021), doi:10.1016/j.ces.2021.116803, also cited by
        get_alpha and get_sat_inf.

        As in those helpers, midpoint bin volumes approximate
        ``integral(kv * L**3 * n(L) dL)``. Normalization cancels a common
        population scale and the scalar shape factor kv [-], so kv is omitted
        from the proportional-volume weights as in get_sat_inf. The stored
        size grid [um] is converted to particle sizes [m]; bin counts use
        the matching distribution [#/um] and bin widths [um].

        Eq. 4.26 attributes the dimensionless constants 1/sqrt(2), 55.5,
        0.96 and 1.75 to Wakeman and Tarleton (2005). Re*Sc <= 1 uses
        1/sqrt(2), retaining the existing equality convention. For Re*Sc > 1,
        add 55.5*(Re*Sc)**0.96 for cake heights at or below 0.10 m and
        1.75*Re*Sc above 0.10 m. The 10 cm threshold is the source's stated
        limit; assigning exact height equality to the thin-cake branch closes
        the source's unspecified equality case.

        References
        ----------
        Destro, F. (2021). Digitalizing pharmaceutical development and
        manufacturing: advanced mathematical modeling for operation design,
        process monitoring and process control. Ph.D. thesis, University
        of Padova, section 4.3.5, Eqs. 4.26-4.27.
        https://www.research.unipd.it/retrieve/e14fb26f-d5f2-3de1-e053-1705fe0ac030/Dissertation_Destro.pdf
        """
        distr = self.Solid_1.distrib  # [#/um], total population density
        size = self.Solid_1.x_distrib  # [um]
        micrometre_to_metre = 1e-6  # [m/um], exact SI prefix conversion
        bin_sizes = (size[:-1] + size[1:]) / 2 * micrometre_to_metre  # [m]
        bin_counts = (distr[:-1] + distr[1:]) / 2 * np.diff(size)  # [-], particle counts
        bin_volumes = bin_sizes**3 * bin_counts  # [m**3], proportional volume; kv cancels
        total_bin_volume = bin_volumes.sum()  # [m**3], same proportional basis
        if total_bin_volume <= 0:
            raise ValueError("Washing diffusivity requires a nonzero particle "
                             "population with positive particle volume.")
        volume_fractions = bin_volumes / total_bin_volume  # [-]
        bin_peclet = vel * bin_sizes[:, np.newaxis] / diff_pure  # [-], bin/species
        re_sc = np.sum(volume_fractions[:, np.newaxis] * bin_peclet, axis=0)  # [-]
        diff_ratio = np.ones_like(re_sc) * 1/np.sqrt(2)  # [-], Eq. 4.26
        if cake_height is None:
            cake_height = self.CakePhase.cake_vol / self.cross_area  # [m]
        thin_cake_limit = 0.10  # [m], Destro thesis Eq. 4.26: 10 cm

        for i in range(len(re_sc)):
            if re_sc[i] > 1:
                if cake_height > thin_cake_limit:
                    diff_ratio[i] += 1.75 * re_sc[i]  # [-], thick-cake branch, Eq. 4.26
                else:
                    diff_ratio[i] += 55.5 * re_sc[i]**0.96

        diff_eff = diff_ratio * diff_pure  # [m**2/s]

        return diff_eff

    def nomenclature(self):
        self.names_states_in = ['mass_conc', 'temp',
                                'mass_frac', 'total_distrib']  #from filter states_out
        self.names_states_out = self.names_states_in

    def get_inputs(self, time):
        pass

    def material_balance(self, z_pos: ArrayLike, time: ArrayLike, vel: float,
                         diff: np.ndarray, lambd: float) -> np.ndarray:
        """Evaluate the transient analytical washing profile.

        Parameters
        ----------
        z_pos : array-like
            Axial positions [m], shape (num_nodes,).
        time : array-like
            Positive washing times [s], shape (num_times,).
        vel : float
            Superficial liquid velocity [m/s].
        diff : ndarray
            Effective axial diffusivities [m**2/s], shape (num_species,).
        lambd : float
            Adsorption correction [-].

        Returns
        -------
        ndarray
            Normalized concentration [-], shape (num_nodes, num_times,
            num_species), with one representing the initial liquid.

        Notes
        -----
        The Lapidus-Amundson exponential-erfc product is evaluated by
        _exp_erfc to avoid overflow followed by multiplication by zero.
        """
        arg_one = 0.5 * np.sqrt(lambd*vel**2*np.einsum('i,j->ij', time,
                                                       1/diff))  # [-], time/species
        arg_two = 0.5 * np.sqrt(1/lambd) * np.einsum(
            'i,j,k->ijk', z_pos, 1/np.sqrt(time), 1/np.sqrt(diff))  # [-], position/time/species
        arg_exp = vel * np.einsum('i,j->ij', z_pos, 1/diff)  # [-], position/species

        first = erfc(arg_one - arg_two)  # [-]
        second = _exp_erfc(arg_exp[:, None, :], arg_one + arg_two)  # [-]
        with np.errstate(under='ignore'):  # vanishing profile tails
            conc_star = 0.5 * (first - second)  # [-]

        return conc_star

    def material_bce(self, z_pos: ArrayLike, wash_ratio: float, vel: float,
                     l_total: float, diff: np.ndarray, lambd: float) -> np.ndarray:
        """Evaluate the analytical profile at a specified washing ratio.

        Parameters
        ----------
        z_pos : array-like
            Axial positions [m], shape (num_nodes,).
        wash_ratio : float
            Displacement distance divided by cake height [-], positive.
        vel : float
            Superficial liquid velocity [m/s].
        l_total : float
            Cake height [m].
        diff : ndarray
            Effective axial diffusivities [m**2/s], shape (num_species,).
        lambd : float
            Adsorption correction [-].

        Returns
        -------
        ndarray
            Normalized concentration [-], shape (num_nodes, num_species),
            with one representing the initial liquid.

        Notes
        -----
        This is the Lapidus-Amundson solution at time wash_ratio*l_total/vel
        [s]. _exp_erfc evaluates its exponential-erfc product without the
        overflowing intermediate exponential.
        """
        root = np.sqrt(vel*l_total / diff)  # [-]
        z_adim = np.asarray(z_pos) / l_total  # [-]
        lambd_wash = lambd * wash_ratio  # [-]
        arg_one = (z_adim - lambd_wash)/2/np.sqrt(lambd_wash)  # [-]
        arg_two = (z_adim + lambd_wash)/2/np.sqrt(lambd_wash)  # [-]
        arg_exp = vel * np.outer(z_pos, 1/diff)  # [-]

        with np.errstate(under='ignore'):  # vanishing profile tails
            conc_star = 1 - 0.5 * (
                erfc(np.outer(arg_one, root)) +
                _exp_erfc(arg_exp, np.outer(arg_two, root))
                )  # [-]

        return conc_star

    def solve_unit(self, deltaP: float, wash_ratio: float = 1,
                   time_vals: Optional[ArrayLike] = None,
                   dynamic: bool = True, verbose: bool = True) -> tuple:
        """Calculate a displacement-washing profile on the cake's axial grid.

        Parameters
        ----------
        deltaP : float
            Pressure drop through cake and medium [Pa].
        wash_ratio : float, optional
            Displacement distance divided by cake height [-], default one.
        time_vals : array-like, optional
            Dynamic output times [s]; defaults to the full washing interval.
        dynamic : bool, optional
            If True, store the time series. If False, evaluate the final
            analytical profile and store it with one time at washing completion.
        verbose : bool, optional
            Reserved output option; currently unused.

        Returns
        -------
        tuple of ndarray
            Final concentration [kg/m**3], normalized final concentration [-],
            retained concentration [kg/m**3], and effluent concentration
            [kg/m**3], each with shape (num_nodes, num_species).

        Raises
        ------
        RuntimeError
            If the requested final time exceeds the total washing time.
        ValueError
            If time_vals is supplied with dynamic=False, which computes only
            the profile at the washing-completion time.

        Notes
        -----
        Stored concProf has axes (position [m], time [s], species), with
        concentration values [kg/m**3]. Static output has a singleton time axis.
        Vanishing tails may underflow to zero or subnormal values during
        profile assembly; other floating-point errors retain caller settings.
        """
        if not dynamic and time_vals is not None:
            raise ValueError("time_vals cannot be supplied with dynamic=False; "
                             "static washing returns the completion profile.")
        # ---------- Physical properties
        # Liquid
        visc_liq_per_node = self.Liquid_1.getViscosity()
        visc_liq = np.mean(visc_liq_per_node)
        diff_pure = self.Liquid_1.getDiffusivityPure(wrt=self.solvent_idx)
        epsilon = self.Solid_1.getPorosity(diam_filter=self.diam_unit)
        lambd_ads = 1 / (1 - self.k_ads + self.k_ads/epsilon)

        c_zero = np.array(self.CakePhase.Liquid_1.mass_conc)

        # Solid
        epsilon = self.Solid_1.getPorosity(diam_filter=self.diam_unit)
        dens_sol = self.Solid_1.getDensity()
        alpha = self.CakePhase.alpha

        # Cake
        cake_height = self.CakePhase.cake_vol / self.cross_area  # m
        vel_liq = deltaP / visc_liq / (alpha * dens_sol * cake_height *
                                       (1 - epsilon) + self.resist_medium)
        diff = self.get_diffusivity(vel_liq, diff_pure)  # [m**2/s]

        z_vals = np.linspace(0, cake_height, self.num_nodes)
        c_zero = define_initial_state(state=c_zero, z_after=z_vals,
                     z_before=self.CakePhase.z_external, indexed_state=True)
        # if c_zero.ndim == 1:     # also for saturation : Daniel
        #     c_zero = np.tile(c_zero, (self.num_nodes, 1))
        # elif c_zero.ndim == 2:
        #     interp = SplineInterpolation(self.CakePhase.z_external, c_zero)
        #     c_zero = interp.evalSpline(z_vals)

        # self.conc_mean_init = np.zeros_like(conc_init)
        # for i in range(len(self.Liquid_1.name_species)):
        #     self.conc_mean_init[:,i] = trapezoidal_rule(z_dim, conc_init[:,i]) / \
        #     self.cake_height
        # c_zero = self.Liquid_1.mass_conc

        c_inlet = np.zeros(self.Liquid_1.num_species)
        c_inlet[self.solvent_idx] = self.Liquid_1.rho_liq[self.solvent_idx]

        # ---------- Solve
        time_total = wash_ratio * cake_height / vel_liq

        if time_vals is None:
            time_vals = np.linspace(eps, time_total)
        elif time_vals[-1] > time_total:
            raise RuntimeError('Final time higher than total time')

        self.num_z = len(z_vals)
        self.num_t = len(time_vals)

        if dynamic:
            conc_adim = self.material_balance(z_vals, time_vals, vel_liq, diff,
                                              lambd_ads)

            conc_star = conc_adim[:, -1]
        else:
            conc_adim = self.material_bce(z_vals, wash_ratio, vel_liq,
                                          cake_height, diff, lambd_ads)

            conc_star = conc_adim

        with np.errstate(under='ignore'):  # assembly may round vanishing tails to zero
            conc = conc_star * (c_zero - c_inlet) + c_inlet
            if dynamic:
                conc_all = (conc_adim * (c_zero - c_inlet)[:, None, :]
                            + c_inlet)  # [kg/m**3], position/time/species
            else:
                conc_all = conc[:, None, :]  # [kg/m**3], final profile with one time
                time_vals = np.array([time_total])  # [s], actual static output time
                self.num_t = 1

            # Average final concentration and material balance
            integral = trapezoidal_rule(z_vals, conc_star)
            c_cake = (c_zero - c_inlet) / cake_height * integral + c_inlet

            sat_zero = self.satur

            c_effl = (epsilon/wash_ratio * (sat_zero * c_zero - c_cake) + c_inlet) / \
                (1 + epsilon/wash_ratio * (sat_zero - 1))

        self.retrieve_results(z_vals, time_vals, conc_all)
        self.cake_height = cake_height
        self.conc_cake = c_cake
        self.conc_effl = c_effl

        return conc, conc_star, c_cake, c_effl

    def retrieve_results(self, z_coord, time_coord, conc):
        num_species = self.Liquid_1.num_species

        indexes = {key: self.states_di[key].get('index', None)
                   for key in self.name_states}

        conc_T = np.transpose(conc, (1, 0, 2))

        dp= unpack_discretized(conc_T, self.dim_states, self.name_states,
                               indexes=indexes)

        self.zProf = z_coord
        self.timeProf = time_coord
        self.concProf = conc

        dp['time'] = np.asarray(self.timeProf)
        dp['z'] = self.zProf

        self.result = DynamicResult(self.states_di, self.fstates_di, **dp)

        if conc.ndim == 3:
            concPerVolElem = []
            for ind in range(self.num_z):
                concPerVolElem.append(conc[ind])

            concPerSpecies = []
            for ind in range(self.num_t):
                concPerSpecies.append(conc[:, ind])

            self.concPerVolElem = concPerVolElem
            self.concPerSpecies = concPerSpecies

        self.CakePhase.mass_concentr = self.concPerSpecies[-1]  # TODO
        self.CakePhase.z_external = self.zProf

        last_state = self.concPerSpecies[-1]
        self.Liquid_1.updatePhase(mass_conc=last_state)

        liquid_out = copy.deepcopy(self.Liquid_1)
        solid_out = copy.deepcopy(self.Solid_1)

        self.Outlet = self.CakePhase
        self.Outlet.Phases = (liquid_out, solid_out)
        if conc.ndim == 3:
            self.outputs = concPerVolElem[-1]

    def flatten_states(self):
        pass

    def plot_profiles(self, fig_size=None, z_val=None, time=None,
                      pick_idx=None):

        """

        Parameters
        ----------

        fig_size : tuple (optional, default = None)
            Size of the figure to be populated.
        z_val : int (optional, default=None)
            Integer value indicating the axial position of cake coordniate
            at which calculated washing outputs to be plotted.
        time : int (optional, default=None)
            Integer value indicating the time on which calculated washing
            outputs to be plotted.
        pick_idx : tuple of lists (optional, default=None)
            List of index of components to include in plotting.
            Length of tuple is 2. Index 0 and 1 corresponds to index of compounds
            in liquid phase and gas phase respectively.
            If None, all the existing components are plotted

        """

        fig, ax = plt.subplots(figsize=fig_size)

        if pick_idx is None:
            pick_idx = np.arange(self.Liquid_1.num_species)
        else:
            pick_idx = list(pick_idx)

        if self.concProf.ndim == 2:
            t_idx = np.argmin(abs(self.timeProf - time))
            ax.plot(washer.z_grid, self.concProf)
            ax.set_xlabel('$z$ (m)')
            ax.text(1, 1.04, '$t = %.0f$ s' % self.timeProf[t_idx],
                    ha='right', transform=ax.transAxes)
        else:
            if z_val is not None:
                z_idx = np.argmin(abs(self.zProf - z_val))

                ax.plot(self.t_grid, self.concPerVolElem[z_idx])
                ax.set_xlabel('time (s)')
                ax.text(1, 1.04, '$z = %.2e$ m' % self.zProf[z_idx],
                        ha='right', transform=ax.transAxes)

            if time is not None:
                t_idx = np.argmin(abs(self.timeProf - time))
                conc_plot = self.concPerSpecies[t_idx][:, pick_idx]

                ax.plot(self.zProf, conc_plot)
                ax.set_xlabel('$z$ (m)')
                ax.text(1, 1.04, '$t = %.0f$ s' % self.timeProf[t_idx],
                        ha='right', transform=ax.transAxes)

        ax.set_ylabel('$C_i$ $(\mathregular{kg \ m^{-3}})$')

        for ind in pick_idx:
            ax.legend(self.Liquid_1.name_species[ind], loc='best')

        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        return fig, ax


if __name__ == '__main__':
    plt.style.use(
        '../../../publications/2021_CACE/source/cace_palatino.mplstyle')

    case = 1
    washer = DisplacementWashing()
    wratio = 0.2

    if case == 1:

        concentr, cstar, conc_cake, conc_receiver = washer.solve_unit(
            1.01325e5, wash_ratio=wratio, dynamic=False)

        figw, axw = washer.plot_profiles(fig_size=(4, 2.8), time=100)
                                         # z_val=0.05)

        axw.text(0, 1.1, '$C_{0, A} = C_{0, B} = 2$,   '
                 '$W = %.1f$, $C_{in, A} = C_{in, B} = 0$' % wratio,
                 transform=axw.transAxes)

    elif case == 2:  # Based on Lapidus and
        length = 0.05  # m
        vel = 0.001  # m/s
        diff = np.array([1e-5, 1e-5]) * 1e-4**0  # m**2/s
        time_total = wratio*length / vel

        length = 50
        vel = 400
        diff = vel / np.array([1, 2, 10, 100])  # m**2/s
        time_total = 2400 / vel
        times = np.linspace(1e-6, time_total, 100)

        z_vals = np.linspace(0, length, 50)

        cstar = washer.material_balance(z_vals, times, vel, diff)
        c_zero = 2
        c_in = 0

        conc_real = cstar * (c_zero - c_in) + c_in

        figw, axw = plt.subplots(1, 2, figsize=(6, 2))

        volume = vel * times * 1
        axw[0].plot(volume, cstar[-1, :, :-1])

        axw[0].set_xlabel(r'$t$')
        axw[0].set_ylabel('$\dfrac{C - C_0}{C_{in} - C_0}$')

        axw[1].plot(z_vals, conc_real[:, -1, :-1])
        axw[1].set_xlabel('$z$')
        axw[1].set_ylabel('$\dfrac{C - C_0}{C_{in} - C_0}$')
