# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 00:24:18 2020

@author: huri
"""
from typing import Union

import numpy as np
import numpy.matlib
from PharmaPy._assimulo import CVode, Explicit_Problem
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import CubicSpline

from PharmaPy.Phases import classify_phases
from PharmaPy.MixedPhases import Cake
from PharmaPy.SolidLiquidSep import (
    _validate_irreducible_saturation,
    get_sat_inf,
    high_resolution_fvm,
    upwind_fvm,
)
from PharmaPy.NameAnalysis import get_dict_states
# from PharmaPy.Interpolation import SplineInterpolation
from PharmaPy.general_interpolation import define_initial_state
from PharmaPy.Commons import reorder_pde_outputs, eval_state_events, handle_events, unpack_discretized
from PharmaPy.Connections import get_inputs_new
# from pathlib import Path
from PharmaPy.Results import DynamicResult
from PharmaPy.Plotting import plot_distrib

eps = np.finfo(float).eps  # [-]

gas_ct = 8.314  # [J/mol/K]


class Drying:

    def __init__(self, number_nodes, supercrit_names, diam_unit=0.01,
                 resist_medium=2.22e9, eta_fun=None, mass_eta=False,
                 state_events=None):
        """

        Parameters
        ----------
        number_nodes : float
            Number of apatial discretization of the cake along the axial
            coordinate.
        supercrit_names : list/tuple of str
            Names of the species in drying gas medium which doesn't participate
            in the mass transfer phenomenon. Names correspond to species names
            in the physical properties .json file.
        diam_unit : float (optional, default=0.01)
            Diameter of the dryer's cross section [m]
        resist_medium : float (optional, default=2.22e9)
            Mesh resistance of filter in dryer. [m**-1]
        eta_fun : callable
            Function with signature eta_fun(saturation). The default is None.
        mass_eta : bool (optional, default=None)
            If true, drying rate limiting factor is a function of mass fractional
            saturation value.
        state_events : list of dict (optional, default=None)
            Dictionary with keys respresenting properties used to monitor the event.
            Keys are composed of 'state_name', 'state_idx', 'value', 'callable'.
            Refer to 'parameter estimation' module for details.

        """
        self.supercrit_names = supercrit_names

        self.num_nodes = number_nodes  # [-]
        self.station_diameter = diam_unit  # [m]
        self.area_cross = self.station_diameter**2 * np.pi/4  # [m**2]
        self.resist_medium = resist_medium  # [1/m]
        self.T_ambient = 298  # [K]

        # Transfer coefficients
        self.k_y = 1e-3  # [mol/s/m**2] (Seader, Separation process)
        # self.h_T_j = 30  # [W/m**2/K]
        self.h_T_j = 10  # [W/m**2/K]
        self.h_T_loss = 30  # [W/m**2/K]

        self._Phases = None
        self._Inlet = None

        # Limiting factor
        if eta_fun is None:
            eta_fun = lambda sat, mass_frac: np.ones_like(sat)

        self.eta_fun = eta_fun
        self.mass_eta = mass_eta
        self.oper_mode = 'Batch'
        self.is_continuous = False
        self.state_event_list = state_events

        self.outputs = None
        self.elapsed_time = 0  # [s]

    @property
    def Phases(self):
        return self._Phases

    @Phases.setter
    def Phases(self, phases):
        if phases.__class__.__name__ == 'Cake':
            self.CakePhase = phases
            if self._Phases is None:
                self._Phases = phases.Phases
            else:
                self._Phases += phases.Phases
        elif phases.__module__ == 'PharmaPy.Phases':
            if self._Phases is None:
                self._Phases = [phases]
            else:
                self._Phases.append(phases)
        else:
            raise RuntimeError('Please provide a list or tuple of phases '
                               'objects')

        if len(self._Phases) > 1:
            classify_phases(self)  # Enumerate phases: Liquid_1,..., Solid_1, ...
            self.cake_height = self.CakePhase.cake_vol / self.area_cross

            # ---------- Discretization
            self.z_grid = np.linspace(0, self.cake_height, self.num_nodes + 1)
            self.z_centers = (self.z_grid[1:] + self.z_grid[:-1]) / 2

            self.dz = np.diff(self.z_grid)

            names = self.Liquid_1.name_species
            idx_supercrit = [names.index(sup_name)
                             for sup_name in self.supercrit_names]
            self.idx_supercrit = np.atleast_1d(idx_supercrit)

            self.name_species = self.Liquid_1.name_species

            self.nomenclature()

    @property
    def Inlet(self):
        return self._Inlet

    @Inlet.setter
    def Inlet(self, inlet):
        self._Inlet = inlet

    def nomenclature(self):
        """Create drying-model state metadata.

        Returns
        -------
        None
            The method populates inlet, outlet, state, and final-state metadata
            on the instance.

        Notes
        -----
        Saturation and mass fractions are dimensionless [-]. Gas and condensed
        temperatures are tracked in [K].
        """
        self.names_states_in = ['temp', 'mass_frac']
        self.names_states_out = self.names_states_in

        self.name_states = ['saturation', 'y_gas', 'x_liq', 'temp_gas',
                            'temp_cond']

        index_z = list(range(self.num_nodes))

        name_liq = [name for name in self.name_species
                    if name not in self.supercrit_names]
        self.states_di = {
            'saturation': {# 'index': index_z,
                           'dim': 1,
                           'units': '[-]', 'type': 'diff'},
            'y_gas': {'index': self.name_species,
                      'dim': len(self.name_species), 'units': '[-]',
                      'type': 'diff'},
            'x_liq': {'index': name_liq, 'dim': len(name_liq), 'units': '[-]',
                      'type': 'diff'},
            'temp_gas': {# 'index': index_z,
                         'dim': 1, 'units': '[K]',
                         'type': 'diff'},
            'temp_cond': {# 'index': index_z,
                         'dim': 1, 'units': '[K]',
                         'type': 'diff'},
            }

        self.dim_states = [di['dim'] for di in self.states_di.values()]

        self.fstates_di = {}

    def get_inputs(self, time):

        input_dict = get_inputs_new(time, self.Inlet, self.states_in_dict)

        return input_dict

    def _eval_state_events(self, time, states, sw):

        events = eval_state_events(
            time, states, sw, self.len_states,
            self.name_states, self.state_event_list, sdot=self.derivatives,
            discretized_model=True)

        return events

    def get_y_equilib(self, temp_cond, x_liq, p_gas):
        """Calculate equilibrium gas mole fractions above the liquid.

        Parameters
        ----------
        temp_cond : ndarray
            Condensed-phase temperature by spatial node [K].
        x_liq : ndarray
            Liquid mass fractions for volatile components [-].
        p_gas : ndarray or float
            Gas pressure [Pa].

        Returns
        -------
        ndarray
            Equilibrium gas mole fractions for volatile components [-].
        """
        mw_liq = self.Liquid_1.mw[self.idx_volatiles]  # [g/mol]
        x_liq_mole_frac = (x_liq / mw_liq).T / np.dot(1/mw_liq, x_liq.T)
        x_liq_mole_frac = x_liq_mole_frac.T  # [-]

        p_sat = self.Liquid_1.AntoineEquation(temp=temp_cond)  # [Pa]

        gamma = self.Liquid_1.getActivityCoeff(mole_frac=x_liq_mole_frac)  # [-]
        p_partial = (gamma * x_liq_mole_frac * p_sat[:, self.idx_volatiles]).T  # [Pa]
        y_equil = p_partial  / p_gas  # [-]

        return y_equil

    def get_drying_rate(self, x_liq, temp_cond, y_gas, p_gas):
        """Calculate component drying rates on a molar basis.

        Parameters
        ----------
        x_liq : ndarray
            Liquid mass fractions for volatile components [-].
        temp_cond : ndarray
            Condensed-phase temperature [K].
        y_gas : ndarray
            Gas-phase mass fractions for all vapor species [-].
        p_gas : float
            Gas pressure [Pa].

        Returns
        -------
        ndarray
            Component drying rates on a molar basis [mol/m**3/s].
        """

        y_gas_mole_frac = self.Vapor_1.frac_to_frac(mass_frac=y_gas)
        y_equil = self.get_y_equilib(temp_cond, x_liq, p_gas)

        y_volat = y_gas_mole_frac[:, self.idx_volatiles].T  # [-]
        dry_volatiles = self.k_y * self.a_V * (y_equil - y_volat).T  # [mol/m**3/s]
        dry_rates = np.zeros_like(y_gas_mole_frac)  # [mol/m**3/s]
        dry_rates[:, self.idx_volatiles] = dry_volatiles

        dry_rates[dry_rates < 0] = 0

        return dry_rates

    def _drying_rate_mass_basis(self, dry_rate):
        """Convert molar drying rates to mass rates.

        Parameters
        ----------
        dry_rate : ndarray
            Component drying rates on a molar basis [mol/m**3/s].

        Returns
        -------
        ndarray
            Component drying rates on a mass basis [kg/m**3/s].
        """
        mw = np.asarray(self.Liquid_1.mw) / 1000  # [kg/mol]

        return dry_rate * mw

    def _gas_mixture_molar_mass(self, y_gas):
        """Calculate gas mixture molar mass from mass fractions.

        Parameters
        ----------
        y_gas : ndarray
            Gas-phase mass fractions, with species on the final axis [-].

        Returns
        -------
        float or ndarray
            Mixture molar mass for each gas composition [g/mol].

        Notes
        -----
        For nonzero mass-fraction weights ``w_i`` [-] and species molar masses
        ``MW_i`` [g/mol], the mixture molar mass is
        ``sum(w_i) / sum(w_i / MW_i)`` [g/mol]. The numerator makes the result
        invariant to common scaling of the composition state and equals one
        for normalized mass fractions.
        """
        species_molar_mass = np.asarray(self.Vapor_1.mw)  # [g/mol]
        total_mass_fraction = np.sum(y_gas, axis=-1)  # [-]
        reciprocal_molar_mass = np.sum(
            y_gas / species_molar_mass, axis=-1
        )  # [mol/g]

        return total_mass_fraction / reciprocal_molar_mass

    def unit_model(self, time, states, sw=None):
        """Evaluate the drying model residual equations.

        Parameters
        ----------
        time : float
            Current integration time [s].
        states : ndarray
            Flattened state vector ordered by node as
            ``S|w_gas|w_liq|Tg|Ts``; saturation and mass fractions are
            dimensionless [-], and temperatures are [K].
        sw : sequence of bool, optional
            Assimulo event switches [-].

        Returns
        -------
        ndarray
            Flattened state derivatives ordered like ``states``. Saturation
            and mass-fraction derivatives are [1/s]; temperature derivatives
            are [K/s].

        Notes
        -----
        The Darcy gas velocity is a superficial velocity [m/s] throttled by
        relative permeability ``k_ra`` [-]. Its dimensional check is
        ``k_perm`` [m**2] * ``k_ra`` [-] * ``dPg_dz`` [Pa/m] /
        ``visc_gas`` [Pa*s] = [m/s].

        The ``x_liq`` supercritical slot reset preserves the existing state
        layout.
        """

        num_comp = self.Liquid_1.num_species  # [-]
        states_reord = states.reshape(
            -1, 3 + num_comp + self.num_volatiles)  # [-] and [K]

        satur = states_reord[:, 0]  # [-]
        y_gas = states_reord[:, 1:1 + num_comp]  # [-]
        x_liq = states_reord[:, 1 + num_comp:
                             1 + num_comp + self.num_volatiles]  # [-]
        x_liq[:, -2] = 0  # legacy supercritical component slot [-]
        temp_gas = states_reord[:, -2]  # [K]
        temp_sol = states_reord[:, -1]  # [K]

        # ---------- Darcy's equation
        visc_gas = self.Vapor_1.getViscosity(temp=temp_gas,
                                             mass_frac=y_gas)  # [Pa*s]

        sat_red = (satur - self.s_inf) / (1 - self.s_inf)  # [-]
        sat_red = np.clip(sat_red, 0, 1)  # [-]
        k_ra = (1 - sat_red)**2 * (1 - sat_red**1.4)  # [-]
        vel_gas = self.k_perm * k_ra * self.dPg_dz / visc_gas  # [m/s]

        # ---------- Drying rate term
        mw_avg_gas = self._gas_mixture_molar_mass(y_gas)  # [g/mol]
        rho_gas = self.pres_gas / gas_ct / temp_gas * mw_avg_gas / 1000  # [kg/m**3]
        rho_liq_ = self.Liquid_1.rho_liq[self.idx_volatiles]  # [kg/m**3]
        self.rho_liq =  1 / np.sum((x_liq/ rho_liq_), axis=1)  # [kg/m**3]
        # Dry correction
        if self.mass_eta:
            rho_liq = self.rho_liq  # [kg/m**3]

            sat_eta = (self.porosity * satur * rho_liq)/ \
                ((1 - self.porosity) * self.rho_sol + self.porosity * satur *rho_liq)  # [-]
            w_eta = x_liq  # [-]
        else:
            sat_eta = satur  # [-]
            w_eta = x_liq  # [-]

        limiter_factor = self.eta_fun(sat_eta, w_eta)  # [-]

        # Drying rate from get_drying_rate is molar [mol/m**3/s].
        self.dry_rate = self.get_drying_rate(x_liq, temp_sol, y_gas,
                                             self.pres_gas)

        # Balances below consume mass-basis drying rates [kg/m**3/s].
        self.dry_rate = self._drying_rate_mass_basis(self.dry_rate)
        self.dry_rate *= limiter_factor[..., np.newaxis]  # [kg/m**3/s]

        # ---------- Model equations
        inputs = self.get_inputs(time)['Inlet']  # inlet states: [-] and [K]

        material_eqns = self.material_balance(
            time, satur, temp_gas, temp_sol, y_gas, x_liq,
            vel_gas, rho_gas, self.dry_rate, inputs)  # [1/s]

        energy_eqns = self.energy_balance(time, temp_gas, temp_sol,
                                          satur, y_gas, x_liq, vel_gas,
                                          rho_gas, self.dry_rate, inputs)  # [K/s]

        model_eqns = np.column_stack(material_eqns + energy_eqns)  # [1/s, K/s]

        self.derivatives = model_eqns.ravel()  # [1/s, K/s]

        return model_eqns.ravel()

    def material_balance(self, time: float, satur: np.ndarray,
                         temp_gas: np.ndarray, temp_sol: np.ndarray,
                         y_gas: np.ndarray, x_liq: np.ndarray,
                         u_gas: np.ndarray, dens_gas: np.ndarray,
                         dry_rate: np.ndarray, inputs: dict,
                         return_terms: bool = False) -> Union[list, int]:
        """Evaluate the saturation and composition material balances.

        Parameters
        ----------
        time : float
            Current integration time [s].
        satur : ndarray
            Cake saturation by spatial node [-].
        temp_gas : ndarray
            Gas-phase temperature by spatial node [K].
        temp_sol : ndarray
            Condensed-phase temperature by spatial node [K].
        y_gas : ndarray
            Gas-phase mass fractions by node and species [-].
        x_liq : ndarray
            Liquid mass fractions by node and volatile species [-].
        u_gas : ndarray
            Superficial Darcy gas velocity by spatial node [m/s].
        dens_gas : ndarray
            Gas density by spatial node [kg/m**3].
        dry_rate : ndarray
            Component drying rates on a mass basis [kg/m**3/s].
        inputs : dict
            Inlet condition dictionary; ``mass_frac`` entries are gas mass
            fractions [-].
        return_terms : bool, optional
            Return stored diagnostic terms instead of balance derivatives [-].

        Returns
        -------
        list of ndarray or int
            ``dsat_dt`` [1/s], ``dygas_dt`` [1/s], and ``dxliq_dt`` [1/s].
            When ``return_terms`` is True, returns the legacy
            ``masstrans_comp`` diagnostic flag [-].

        Notes
        -----
        The pore gas is an open, constant-pressure control volume with
        holdup ``H = porosity*(1-S)*dens_gas`` [kg/m**3]. Component and total
        balances are ``d(H*y_i)/dt = F_i + r_i - y_i*V`` and
        ``dH/dt = sum(F_j + r_j) - V``, where convection ``F_i``, evaporation
        ``r_i``, and the vent ``V`` have units [kg/m**3/s]. The vent carries
        the bulk composition and cancels identically in the quotient rule:
        ``H*dy_i/dt = F_i + r_i - y_i*sum(F_j + r_j)``.
        The code already expresses convection in non-conservative form,
        ``-u_gas/epsilon_gas * dy_i/dz``, using superficial Darcy velocity;
        its species sum vanishes for normalized inlet and interior fields.
        The total-source subtraction preserves normalization and restores
        small normalization errors when total accumulation is positive:
        ``d(sum(y)-1)/dt = -sum((F+r)/H)*(sum(y)-1)``. A separate
        saturation-only correction would violate this closure.
        """
        
        satur[satur < eps] = eps
        satur[satur >= 1] = 1 - eps
        # ----- Reading inputs
        y_gas_inputs = inputs['mass_frac']  # [-]

        # ----- Liquid phase
        dens_liq = self.rho_liq  # [kg/m**3]

        dry_rate_volatiles = dry_rate[:, self.idx_volatiles]  # [kg/m**3/s]
        sum_dry = dry_rate_volatiles.sum(axis=1)  # [kg/m**3/s]

        dsat_dt = -sum_dry / dens_liq / self.porosity  # [1/s]

        dxliq_dt = -1 / satur* \
            (dry_rate_volatiles.T / dens_liq / self.porosity +
              x_liq.T * dsat_dt)  # [1/s]

        # ----- Gas phase
        # Convective term
        epsilon_gas = self.porosity * (1 - satur)  # [-]
        epsilon_gas[epsilon_gas <= eps] = eps  # [-]

        fluxes_yg = upwind_fvm(y_gas, boundary_cond=y_gas_inputs)  # [-]

        dygas_dz = np.diff(fluxes_yg, axis=0).T / self.dz  # [1/m]

        convection = -u_gas * dygas_dz / epsilon_gas  # [1/s]
        # Transfer term
        # [kg/m**3/s] / ([-] * [kg/m**3]) = [1/s].
        # epsilon_gas already includes porosity*(1 - satur), the gas holdup.
        transfer_gas = dry_rate.T / epsilon_gas / dens_gas  # [1/s]
        component_accumulation = convection + transfer_gas  # [1/s], divided by H
        total_accumulation = component_accumulation.sum(axis=0)  # [1/s], divided by H
        dygas_dt = component_accumulation - y_gas.T * total_accumulation  # [1/s]

        if return_terms:
            self.masstrans_comp = 1

            return self.masstrans_comp

        else:
            return [dsat_dt, dygas_dt.T, dxliq_dt.T]

    def energy_balance(self, time: float, temp_gas: np.ndarray,
                       temp_sol: np.ndarray, satur: np.ndarray,
                       y_gas: np.ndarray, x_liq: np.ndarray,
                       u_gas: np.ndarray, rho_gas: np.ndarray,
                       dry_rate: np.ndarray, inputs: dict,
                       return_terms: bool = False) -> Union[list, tuple]:
        """Evaluate gas and condensed-phase energy balances.

        Parameters
        ----------
        time : float
            Current integration time [s].
        temp_gas : ndarray
            Gas-phase temperature by spatial node [K].
        temp_sol : ndarray
            Condensed-phase temperature by spatial node [K].
        satur : ndarray
            Cake saturation by spatial node [-].
        y_gas : ndarray
            Gas-phase mass fractions by node and species [-].
        x_liq : ndarray
            Liquid mass fractions by node and volatile species [-].
        u_gas : ndarray
            Superficial Darcy gas velocity by spatial node [m/s].
        rho_gas : ndarray
            Gas density by spatial node [kg/m**3].
        dry_rate : ndarray
            Component drying rates on a mass basis [kg/m**3/s].
        inputs : dict
            Inlet condition dictionary; ``temp`` is the gas inlet
            temperature [K].
        return_terms : bool, optional
            Return stored diagnostic terms instead of temperature derivatives [-].

        Returns
        -------
        list of ndarray
            ``[dTg_dt, dTcond_dt]``: gas and condensed temperature derivatives
            by spatial node [K/s] when ``return_terms`` is False.
        tuple of ndarray
            When ``return_terms`` is True, returns the diagnostic terms
            ``(convec_term, drying, heat_cond, heat_loss_emp)`` instead.
            ``convec_term`` is the raw ``u_gas * dTg_dz`` diagnostic
            [kg*K/m**3/s]. ``drying`` is the condensed-temperature latent
            contribution [K/s]; ``heat_cond`` and ``heat_loss_emp`` are
            gas-temperature-rate contributions [K/s].

        Notes
        -----
        ``latent_heat`` is requested on a mass basis [J/kg] and spans every
        species, so its volatile columns are paired with the matching
        ``dry_rate`` columns to give the latent power [J/m**3/s].
        ``u_gas`` is the superficial Darcy velocity supplied by ``unit_model``.
        Convective power per bed volume is ``-u_gas*rho_gas*cp*dT/dz``
        [J/m**3/s]. Dividing by the gas heat capacity per bed volume,
        ``epsilon_gas*rho_gas*cv`` [J/m**3/K], gives the temperature-rate
        contribution ``-(u_gas/epsilon_gas)*(cp/cv)*dT/dz`` [K/s]. Thus
        temperature and composition both convect at the pore velocity
        ``u_gas/epsilon_gas`` [m/s], with density cancelling exactly once.
        """

        mw_avg_gas = self._gas_mixture_molar_mass(y_gas)  # [g/mol]
        # ----- Reading inputs
        temp_gas_inputs = inputs['temp']  # [K]

        # ----- Gas phase equations
        cpg_mix = self.Vapor_1.getCp(temp=temp_gas, mass_frac=y_gas,
                                     basis='mass')  # [J/kg/K]
        cvg_mix = cpg_mix - gas_ct / mw_avg_gas * 1000  # [J/kg/K]

        epsilon_gas = self.porosity * (1 - satur)  # [-]
        denom_gas = cvg_mix * epsilon_gas * rho_gas  # [J/m**3/K]

        latent_heat = self.Vapor_1.getHeatVaporization(temp_sol,
                                                       basis='mass')  # [J/kg]

        xliq_extended = np.column_stack((x_liq, np.zeros((x_liq.shape[0],
                                                          len(self.idx_supercrit)))))  # [-]

        cpl_mix = self.Liquid_1.getCp(temp=temp_sol, mass_frac=xliq_extended,
                                      basis='mass')  # [J/kg/K]
        temp_wb = 22 + 273  # [K]
        sensible_heat = (
            cpg_mix * (temp_gas - temp_wb) * dry_rate.sum(axis=1)
        )  # [J/m**3/s]

        heat_transf = (
            self.h_T_j * self.a_V * (temp_gas - temp_sol)
        )  # [J/m**3/s]
        heat_loss = np.zeros_like(temp_gas)  # [J/m**3/s]
        fluxes_Tg = high_resolution_fvm(temp_gas,
                                        boundary_cond=temp_gas_inputs)  # [K]

        dTg_dz = np.diff(fluxes_Tg) / self.dz * rho_gas  # [kg*K/m**4]

        conv_term = -u_gas * dTg_dz * cpg_mix  # [J/m**3/s]

        dTg_dt = (
            conv_term + sensible_heat - heat_transf - heat_loss
        ) / denom_gas  # [K/s]

        # ----- Condensed phases equations
        dens_liq = self.rho_liq  # [kg/m**3]
        heat_loss_cond = np.zeros_like(temp_sol)  # [J/m**3/s]
        # getHeatVaporization returns one column per species, and drops to
        # 1-D for a single node, so restore the node axis before pairing
        # each volatile's drying rate with that same volatile's latent
        # heat. The non-condensable carrier columns are zero and excluded.
        latent_heat_volatiles = np.atleast_2d(
            latent_heat)[:, self.idx_volatiles]  # [J/kg]
        drying_terms = (
            dry_rate[:, self.idx_volatiles] * latent_heat_volatiles
        ).sum(axis=1)  # [J/m**3/s]
        denom_cond = self.rho_sol * (1 - self.porosity) * self.cp_sol + \
            self.porosity * satur * cpl_mix * dens_liq  # [J/m**3/K]

        dTcond_dt = (-drying_terms + heat_transf - heat_loss_cond) / denom_cond  # [K/s]

        if return_terms:
            self.convec_term = u_gas * dTg_dz  # [kg*K/m**3/s]
            self.drying = drying_terms / denom_cond  # [K/s]
            self.heat_cond = heat_transf/ denom_gas  # [K/s]
            self.heat_loss_emp = heat_loss/ denom_gas  # [K/s]

            return self.convec_term, self.drying, self.heat_cond, self.heat_loss_emp

        else:

            return [dTg_dt, dTcond_dt]

    def solve_unit(self, deltaP, runtime=None, time_grid=None, any_event=True,
                   verbose=True, sundials_opts=None):
        """Initialize and integrate the drying model.

        Parameters
        ----------
        deltaP : float
            Pressure drop across the drying cake and medium [Pa].
        runtime : float, optional
            Duration to simulate from the current elapsed time [s].
        time_grid : ndarray, optional
            Output time grid for the CVode simulation [s].
        any_event : bool, optional
            Whether any configured state event can terminate the solve [-].
        verbose : bool, optional
            If False, suppress solver output [-].
        sundials_opts : dict, optional
            Solver option names and values forwarded to CVode.

        Returns
        -------
        time : ndarray
            Simulated time points [s].
        states : ndarray
            Flattened state history. Saturation and mass-fraction columns are
            dimensionless [-]; temperature columns are [K].

        Notes
        -----
        The initial per-node state vector is assembled as
        ``S`` [-] | ``y_gas`` [-] | ``x_liq`` [-] | ``temp_gas`` [K] |
        ``temp_cond`` [K] for both
        distributed and uniform single-node cake initial conditions.
        Cake permeability is computed as
        ``1 / (alpha * rho_sol * (1 - porosity))`` [m**2].
        ``Solid_1.x_distrib`` is stored in micrometers [um] and is converted to
        meters [m] before evaluating the shared irreducible-saturation
        correlation.
        Liquid density [kg/m**3] computed from the initial liquid composition
        is used for the irreducible-saturation estimate. During integration,
        ``unit_model`` recomputes ``self.rho_liq`` from the current liquid
        state before material and energy balances are evaluated.
        """
        
        p_atm=101325  # [Pa]
        # ---------- Initialization
        # Volatile components
        idx_liquid = np.arange(0, self.Liquid_1.num_species)
        idx_volatiles = [i for i in idx_liquid if i not in self.idx_supercrit]
        self.num_volatiles = len(idx_volatiles)
        self.idx_volatiles = idx_volatiles

        num_y_gas = self.num_volatiles + len(self.idx_supercrit)
        num_x_liq = self.num_volatiles
        self.len_states = [1, num_y_gas, num_x_liq, 1, 1]
        num_comp = self.Liquid_1.num_species

        len_states_in = [1, num_y_gas]
        states_in_dict = dict(zip(self.names_states_in, len_states_in))
        self.states_in_dict = {'Inlet': states_in_dict}

        y_gas_init = self.Vapor_1.mass_frac  # [-]
        x_liq_init = self.CakePhase.Liquid_1.mass_frac  # [-]

        satur_init = self.CakePhase.saturation  # [-]

        temp_cond_init = self.CakePhase.Solid_1.temp  # [K]
        temp_gas_init = self.Vapor_1.temp  # [K]
        z_cake = self.CakePhase.z_external  # [m]

        if x_liq_init.ndim == 1:
            x_liq_init = x_liq_init[idx_volatiles]
            
            if len(satur_init) != 1:
                states_tuple = (y_gas_init, x_liq_init, temp_gas_init, temp_cond_init)

                states_stacked = np.hstack(states_tuple)
                state_tiled = np.tile(states_stacked, (self.num_nodes, 1))
                states_prev = np.column_stack((satur_init, state_tiled))
                
            else:

                states_tuple = (satur_init, y_gas_init, x_liq_init,
                                temp_gas_init, temp_cond_init)
    
                states_stacked = np.hstack(states_tuple)
                states_prev = np.tile(states_stacked, (self.num_nodes, 1))

        else:
            x_liq_init = x_liq_init[:, idx_volatiles]
            if y_gas_init.ndim == 1:
                y_gas_init = np.tile(y_gas_init, (self.num_nodes, 1))
            if isinstance(temp_cond_init, float):
                temp_cond_init = np.ones_like(satur_init) * temp_cond_init
            if isinstance(temp_gas_init, float):
                temp_gas_init = np.ones_like(satur_init) * temp_gas_init
            states_stacked = np.column_stack(
                (satur_init, y_gas_init, x_liq_init, temp_gas_init,
                 temp_cond_init))

            interp_obj = CubicSpline(z_cake, states_stacked)
            states_prev = interp_obj(self.z_centers)

        alpha = self.CakePhase.alpha  # [m/kg]
        rho_sol = self.Solid_1.getDensity()  # [kg/m**3]
        porosity = self.CakePhase.porosity  # [-]

        xliq = states_prev[:, num_comp + 1: num_comp + 1 + self.num_volatiles]

        xliq_init = np.zeros((self.num_nodes, num_comp))
        xliq_init[:, self.idx_volatiles] = xliq
        rho_liq = self.Liquid_1.getDensity(temp=temp_cond_init,
                                           mass_frac=xliq_init, basis='mass')  # [kg/m**3]
        surf_tens = self.Liquid_1.getSurfTension(temp=temp_cond_init,
                                                 mass_frac=xliq_init)  # [N/m]

        self.k_perm = 1 / alpha / rho_sol / (1 - porosity)  # [m**2]
        self.rho_sol = rho_sol
        self.porosity = porosity
        self.cp_sol = self.Solid_1.getCp()  # [J/kg/K]

        moments = self.Solid_1.getMoments(mom_num=[0, 1, 2, 3, 4])

        self.a_V = moments[2] * (1 - porosity) / moments[3]  # [m**2/m**3]

        deltaP_media = deltaP*self.resist_medium / \
            (alpha*rho_sol*(1 - porosity)*self.cake_height +
              self.resist_medium)  # [Pa]
        deltaP -= deltaP_media  # [Pa]
        self.deltaP = deltaP  # [Pa]
        p_top = p_atm + deltaP  # [Pa]

        self.dPg_dz = deltaP / self.cake_height  # [Pa/m]
        self.pres_gas = np.linspace(p_top, p_top - deltaP,
                                    num=self.num_nodes)  # [Pa]

        x_csd = self.Solid_1.x_distrib * 1e-6  # [m]
        csd = self.Solid_1.distrib  # [#/m**3/um]
        mom_zero = self.Solid_1.moments[0]  # [#/m**3]

        self.s_inf = get_sat_inf(x_csd, csd, deltaP, porosity,
                                 self.cake_height, mom_zero,
                                 (np.mean(surf_tens), rho_liq[0]))  # [-]
        _validate_irreducible_saturation(
            self.s_inf, deltaP, x_csd, "Drying"
        )

        # ---------- Solve model
        model = self.unit_model
        if self.state_event_list is None:
            def model(t, y): return self.unit_model(t, y)#, None)
            problem = Explicit_Problem(self.unit_model, y0=states_prev.ravel(),
                                       t0=0)
        else:
            switches = [True] * len(self.state_event_list)
            problem = Explicit_Problem(self.unit_model, y0=states_prev.ravel(),
                                 t0=0, sw0=switches)

            def new_handle(solver, info):
                return handle_events(solver, info, self.state_event_list,
                                     any_event=any_event)

            problem.state_events = self._eval_state_events
            problem.handle_event = new_handle

        self.derivatives = model(0, states_prev.ravel())

        problem.name = 'Drying Model'

        sim = CVode(problem)

        if sundials_opts is not None:
            for key, val in sundials_opts.items():
                setattr(sim, key, val)

        # sim.linear_solver = 'SPGMR'
        
        if runtime is not None:
            final_time = runtime + self.elapsed_time

        if time_grid is not None:
            final_time = time_grid[-1] + self.elapsed_time
            self.elapsed_time = time_grid[0]
        
        if not verbose:
          sim.verbosity = 50
          
        time, states = sim.simulate(final_time,  ncp_list=time_grid)
        
        self.retrieve_results(time, states)

        return time, states

    def retrieve_results(self, time, states):
        """Store drying simulation outputs in a ``DynamicResult``.

        Parameters
        ----------
        time : array_like
            Simulated time points [s].
        states : ndarray
            Flattened state history. Saturation and mass-fraction columns are
            dimensionless [-]; temperature columns are [K].

        Returns
        -------
        None
            The method updates ``result`` and cake-grid metadata in place.
        """
        time = np.array(time)
        self.timeProf = time
        self.elapsed_time += time[-1]
        
        indexes = {key: self.states_di[key].get('index', None)
                   for key in self.name_states}

        inputs = self.get_inputs(time)['Inlet']

        dp = unpack_discretized(states, self.dim_states, self.name_states,
                                indexes=indexes, inputs=inputs)

        dp['time'] = time
        dp['z'] = self.z_centers
        
        self.result = DynamicResult(self.states_di, self.fstates_di, **dp)

        self.CakePhase.z_external = self.z_centers

    def flatten_states(self):
        """Placeholder for future drying-state flattening support.

        Returns
        -------
        None
            This method is not implemented.
        """
        pass

    def plot_profiles(self, times=None, z_pos=None, pick_comp=None, **fig_kw):
        '''

        Parameters
        ----------
        time : int (optional, default=None)
            Integer value indicating the time on which calculated drying
            outputs to be plotted.
        z_pos : int (optional, default=None)
            Integer value indicating the axial position of cake coordniate
            at which calculated drying outputs to be plotted.
        pick_comp : list of lists (optional, default=None)
            Lists containing indexes/names of components to include in the
            x_liq and y_gas plots. First list contains indexes of compounds
            in liquid phase, and second list those of the gas phase.
            If None, all the existing components are plotted.

        '''
        if pick_comp is None:
            pick_liq = np.arange(self.num_liq)
            pick_vap = np.arange(self.num_gas)
        else:
            pick_liq, pick_vap = pick_comp

        if times is not None:

            states_plot = [('x_liq', pick_liq), ('y_gas', pick_vap),
                           'temp_cond', 'temp_gas', 'saturation']

            y_labels = ('x_liq', 'y_gas', 'T_cond', 'T_gas', 'sat')

            fig, axes = plot_distrib(self, states_plot, times=times,
                                     x_name='z', ncols=2, nrows=3,
                                     ylabels=y_labels, **fig_kw)

            fig.tight_layout()
            
        if z_pos is not None:
            
            states_plot = [('x_liq', pick_liq), ('y_gas', pick_vap),
                           'temp_cond', 'temp_gas', 'saturation']

            y_labels = ('x_liq', 'y_gas', 'T_cond', 'T_gas', 'sat')

            fig, axes = plot_distrib(self, states_plot, x_vals=z_pos,
                                     x_name='time', ncols=2, nrows=3,
                                     ylabels=y_labels, **fig_kw)

            fig.tight_layout()

        if z_pos is not None:  # TODO: this needs to be implemented
            fig, axes = plt.subplots(2, sharex=True, **fig_kw)

            idx_node = np.argmin(abs(z_pos - self.z_centers))
            w_liq = [self.xLiqProf[ind] for ind in pick_liq]
            w_vap = [self.yGasProf[ind] for ind in pick_vap]

            xliq_plot = np.vstack(w_liq)[:, idx_node].reshape(-1, len(self.timeProf))
            ygas_plot = np.vstack(w_vap)[:, idx_node].reshape(-1, len(self.timeProf))

            axes[0].plot(self.timeProf, xliq_plot.T)
            axes[1].plot(self.timeProf, ygas_plot.T)
            # axes[1].set_yscale('log')

            axes[0].text(1, 1.04, 'z_pos = %.3e m' % self.z_centers[idx_node],
                         ha='right', transform=axes[0].transAxes)

            axes[1].set_xlabel('$time$ (sec)')

            label_liq = [self.Liquid_1.name_species[ind] for ind in pick_liq]
            label_vap = [self.Liquid_1.name_species[ind] for ind in pick_vap]

            axes[0].legend(label_liq)
            axes[1].legend(label_vap)

            axes[0].set_ylabel('$x_{liq}$')
            axes[1].set_ylabel('$y_{gas}$')

        return fig, axes

    def plot_rates(self, z_pos=0, fig_size=None):
        """Plot drying rates at one axial position.

        Parameters
        ----------
        z_pos : float, optional
            Axial position used to select the nearest drying node [m].
        fig_size : tuple, optional
            Figure size forwarded to matplotlib [-].

        Returns
        -------
        fig : matplotlib.figure.Figure
            Created figure.
        axis : matplotlib.axes.Axes
            Axes containing drying-rate profiles [mol/m**3/s].
        """
        z_idx = np.argmin(abs(z_pos - self.z_centers))
        temp_cond = self.tempLiqProf[:, z_idx]

        x_liq = np.column_stack([array[:, z_idx] for array in self.xLiqProf])
        y_gas = np.column_stack([array[:, z_idx] for array in self.yGasProf])

        gas_pressure = self.pres_gas[z_idx]

        self.dry_rate = self.get_drying_rate(x_liq, temp_cond, y_gas, gas_pressure)

        fig, axis = plt.subplots()

        axis.plot(self.timeProf, self.dry_rate)
        axis.set_xlabel('time (s)')
        axis.set_ylabel('drying rate ($\mathregular{mol_i \ s^{-1}}$)')

        axis.text(1, 1.04, '$z = %.2f$ m' % self.z_centers[z_idx], ha='right',
                  transform=axis.transAxes)

        axis.legend([self.Liquid_1.name_species[ind]
                     for ind in self.idx_volatiles], loc='best')

        return fig, axis
