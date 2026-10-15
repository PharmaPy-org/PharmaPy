# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 14:55:22 2020

@author: dcasasor
"""

import copy

import numpy as np
from scipy.optimize import newton

from PharmaPy.Commons import mid_fn
from PharmaPy.Phases import LiquidPhase
from PharmaPy.ThermoModule import (
    VALID_ACTIVITY_MODELS as VALID_GAMMA_METHODS, validate_activity_model)
from PharmaPy.Streams import LiquidStream
from PharmaPy.Connections import get_inputs_new

from PharmaPy.Results import DynamicResult


def material_setter(instance, oper_mode):
    """Return inlet metadata for extractor amount or flow routing.

    Parameters
    ----------
    instance : LiquidPhase or LiquidStream
        Material object passed to the extractor. Batch phases provide total
        ``moles`` [mol]; continuous streams provide ``mole_flow`` [mol/s].
    oper_mode : {'Batch', 'Continuous'}
        Extractor operating mode. ``Continuous`` selects flow-rate semantics;
        ``Batch`` selects amount semantics.

    Returns
    -------
    out : dict
        Inlet metadata with ``in_flow`` in [mol/s] for continuous operation or
        [mol] for batch operation, ``temp`` [K], ``pres`` [Pa], component
        count, and result-state metadata.
    """
    name_comp = instance.name_species
    num_comp = len(name_comp)
    states_di = {
        'x_heavy': {'dim': num_comp, 'type': 'alg', 'index': name_comp,
                    'units': '[-]'},
        'x_light': {'dim': num_comp, 'type': 'alg', 'index': name_comp,
                    'units': '[-]'},
        'moles_heavy': {'dim': 1, 'type': 'alg', 'units': '[mol]'},
        'moles_light': {'dim': 1, 'type': 'alg', 'units': '[mol]'}}

    if oper_mode == 'Continuous':
        in_flow = instance.mole_flow
    else:
        in_flow = instance.moles

    out = {'in_flow': in_flow, 'temp': instance.temp, 'pres': instance.pres,
           'num_comp': num_comp, 'states_di': states_di}

    return out


def validate_gamma_method(gamma_method: str,
                          param_name: str = 'gamma_method') -> None:
    """Validate an activity-coefficient model selector.

    Parameters
    ----------
    gamma_method : {'ideal', 'UNIFAC', 'UNIQUAC'}
        Activity-coefficient model selector.
    param_name : str, optional
        Parameter name to use in the validation error.

    Raises
    ------
    ValueError
        If ``gamma_method`` is not one of ``VALID_GAMMA_METHODS``.
    """
    validate_activity_model(gamma_method, param_name=param_name)


class ContinuousExtractor:
    def __init__(self, k_fun=None, gamma_method='UNIQUAC'):
        """Create a continuous extractor.

        Parameters
        ----------
        k_fun : callable, optional
            Equilibrium function. It must accept two liquid mole-fraction
            vectors ``x1`` [-] and ``x2`` [-] plus temperature ``temp`` [K],
            and return distribution coefficients ``K_i`` [-].
        gamma_method : {'ideal', 'UNIFAC', 'UNIQUAC'}, optional
            Activity-coefficient method used by the default ``k_fun``.

        Notes
        -----
        The extractor declares the flowsheet destination contract of a
        continuous unit (issue #445): ``oper_mode = 'Continuous'``,
        ``is_continuous = True`` and ``names_states_in``, the stream fields
        it reads from its ``Inlet``: mole fractions [-], molar flow
        [mol/s], temperature [K] and pressure [Pa]. It reads them from the
        stored state of the ``Inlet`` object, so a connected upstream time
        profile is not resolved: the extractor splits the upstream outlet's
        final state. It publishes ``Liquid_2`` and ``Liquid_3`` but no
        ``Outlet``, so it cannot be a flowsheet source.
        """
        validate_gamma_method(gamma_method)

        self._Inlet = None

        self.k_fun = k_fun

        self.gamma_method = gamma_method

        self.oper_mode = 'Continuous'
        self.is_continuous = True
        self.names_states_in = ['mole_frac', 'mole_flow', 'temp', 'pres']

    @property
    def Inlet(self):
        return self._Inlet

    @Inlet.setter
    def Inlet(self, instance):
        fields = material_setter(instance, oper_mode=self.oper_mode)
        self._Inlet = instance

        self.matter = self._Inlet

        for key, val in fields.items():
            setattr(self, key, val)

    def get_inputs(self):
        pass

    def get_reported_outlets(self) -> dict:
        """Return the two liquid products that the stream table reports.

        Returns
        -------
        dict of str to LiquidStream or LiquidPhase
            ``{'heavy': Liquid_2, 'light': Liquid_3}``: the liquid products,
            as flows [mol/s] for ``ContinuousExtractor`` or amounts [mol]
            for ``BatchExtractor``, at the feed temperature [K] and
            pressure [Pa].

        Raises
        ------
        AttributeError
            If the extractor has not been solved, so it has no
            ``Liquid_2``.

        Notes
        -----
        ``heavy`` and ``light`` follow the extractor's current
        classification of ``Liquid_2`` and ``Liquid_3`` in
        ``retrieve_results``, also used by its result fields ``x_heavy``
        and ``x_light``: when two liquid phases form, ``Liquid_2`` is the
        one with the larger molar density [mol/m**3], not necessarily the
        larger mass density [kg/m**3]. Otherwise the products are not
        compared by density: ``Liquid_2`` takes the phase-fraction share of
        the feed and ``Liquid_3`` the rest, so one of them is empty.
        Classifying by mass density is tracked by issue #257
        (https://github.com/PharmaPy-org/PharmaPy/issues/257); fixing it
        there corrects these rows without renaming them.
        ``SimulationResult.GetStreamTable`` reads this hook before
        ``Outlet`` and reports the products as the rows ``'Outlet_heavy'``
        and ``'Outlet_light'`` (issue #451). The extractor deliberately has
        no ``Outlet``, so it cannot be a flowsheet source (issue #445).
        """
        return {'heavy': self.Liquid_2, 'light': self.Liquid_3}

    def material_eqn_based(self, extr, raff, x_extr, x_raff, z_i):

        gamma_extr = self.matter.getActivityCoeff(method=self.gamma_method,
                                                  mole_frac=x_extr,
                                                  temp=self.temp)  # [-]

        gamma_raff = self.matter.getActivityCoeff(method=self.gamma_method,
                                                  mole_frac=x_raff,
                                                  temp=self.temp)  # [-]

        global_bce = 1 - extr - raff  # [-]
        comp_bces = z_i - extr*x_extr - raff*x_raff  # [-]
        equilibria = x_extr * gamma_extr - x_raff * gamma_raff  # [-]

        diff_frac = np.sum(x_extr - x_raff)  # [-]
        args_mid = np.array([raff, diff_frac, raff - 1])  # [-]
        vap_flow = mid_fn(args_mid)  # phase-presence residual [-]

        balance = np.concatenate(
            (np.array([global_bce]),
             comp_bces, equilibria,
             np.array([vap_flow]))
            )
        # All entries are dimensionless material/equilibrium residuals [-].

        return balance

    def material_balance(self, phi_seed, x_1_seed, x_2_seed, z_i, temp):
        """Solve dimensionless liquid split and equilibrium relations.

        ``phi_seed`` is a phase fraction [-], ``x_1_seed``, ``x_2_seed``,
        and ``z_i`` are mole fractions [-], and ``temp`` is temperature [K].
        """

        def get_ki(x1, x2, temp):
            gamma_1 = self.matter.getActivityCoeff(method=self.gamma_method,
                                                   mole_frac=x1,
                                                   temp=temp)  # [-]

            gamma_2 = self.matter.getActivityCoeff(method=self.gamma_method,
                                                   mole_frac=x2,
                                                   temp=temp)  # [-]

            k_i = gamma_1 / gamma_2  # K_i [-]

            return k_i

        if self.k_fun is None:
            k_fun = get_ki
        else:
            k_fun = self.k_fun

        def func_phi(phi, k_i):
            # phi, z_i, and K_i are dimensionless, so f_phi is dimensionless.
            f_phi = z_i * (1 - k_i) / (1 + phi*(k_i - 1))

            return f_phi.sum()

        def deriv_phi(phi, k_i):
            # d(f_phi)/d(phi) is dimensionless because phi is dimensionless.
            deriv = z_i * (1 - k_i)**2 / (1 + phi*(k_i - 1))**2

            return deriv.sum()

        error = 1

        tol = self.solver_options.get('tol', 1e-4)
        max_iter = self.solver_options.get('max_iter', 100)

        count = 0

        while error > tol and count < max_iter:
            k_i = k_fun(x_1_seed, x_2_seed, self.temp)  # K_i [-]
            phi_k = newton(func_phi, phi_seed, args=(k_i, ),
                           fprime=deriv_phi)  # phase fraction [-]

            x_1_k = z_i / (1 + phi_k*(k_i - 1))  # mole fractions [-]
            x_2_k = x_1_k * k_i  # mole fractions [-]

            # Normalize x's
            x_1_k *= 1 / x_1_k.sum()
            x_2_k *= 1 / x_2_k.sum()

            x_k = np.concatenate((x_1_k, x_2_k))  # mole fractions [-]
            x_km1 = np.concatenate((x_1_seed, x_2_seed))  # [-]

            error = np.linalg.norm(x_k - x_km1)  # [-]

            # Update
            x_1_seed = x_1_k
            x_2_seed = x_2_k
            phi_seed = phi_k

            count += 1

        # Retrieve results
        phi_conv = phi_seed
        x1_conv = x_1_seed
        x2_conv = x_2_seed

        info = {'error': error, 'num_iter': count}

        return phi_conv, x1_conv, x2_conv, info

    def energy_balance(self, liq, vap, x_i, y_i, z_i):
        """Evaluate the continuous extractor energy residual.

        Parameters
        ----------
        liq : float
            Liquid split fraction [-].
        vap : float
            Extract phase split fraction [-].
        x_i : ndarray
            Liquid outlet mole fractions [-].
        y_i : ndarray
            Extract outlet mole fractions [-].
        z_i : ndarray
            Feed mole fractions [-].

        Returns
        -------
        float
            Heat duty for the outlet-minus-inlet enthalpy balance [J/s].
        """
        liq_flow = liq * self.in_flow  # [mol/s]
        vap_flow = vap * self.in_flow  # [mol/s]

        tref = self.temp  # [K]

        h_liq = 0  # [J/mol]
        h_vap = self.VaporOut.getEnthalpy(self.temp, temp_ref=tref,
                                          mole_frac=y_i, basis='mole')  # [J/mol]

        h_in = self.matter.getEnthalpy(temp_ref=tref, basis='mole')  # [J/mol]

        heat_duty = liq_flow * h_liq + vap_flow * h_vap - self.in_flow * h_in  # [J/s]

        return heat_duty  # [J/s]

    def unit_model(self, phi, x1, x2, temp, material=True):
        z_i = self.matter.mole_frac

        if material:
            balance = self.material_balance(phi, x1, x2, z_i, temp)
        else:
            balance = self.energy_balance(phi, x1, x2, z_i, temp)

        return balance

    def flatten_states(self):
        pass

    def solve_unit(self, solver_options=None):

        if solver_options is None:
            solver_options = {}

        self.solver_options = solver_options

        # Set seeds
        mol_z = self.in_flow * self.matter.mole_frac

        distr_seed = np.random.random(self.num_comp)
        distr_seed = distr_seed / distr_seed.sum()

        liq1_seed = mol_z * distr_seed
        liq2_seed = mol_z * (1 - distr_seed)

        x1_seed = liq1_seed / liq1_seed.sum()
        x2_seed = liq2_seed / liq2_seed.sum()

        phi_seed = liq2_seed.sum() / self.in_flow

        # Solve material balance
        solution = self.unit_model(phi_seed, x1_seed, x2_seed, self.temp)

        # # Energy balance
        # heat_bce = self.unit_model(solution, material=False)

        self.retrieve_results(solution)

        return solution

    def retrieve_results(self, solution):
        phase_part = solution[0]
        self.info_solver = solution[-1]

        if phase_part > 1 or phase_part < 0:
            phase_part = 1

            print('\nNo phases in equilibrium present at the specified '
                  'conditions', end='\n')

            xa_liq = self.matter.mole_frac
            xb_liq = xa_liq

        else:
            xa_liq = solution[1]
            xb_liq = solution[2]

        liquid_b = phase_part * self.in_flow
        liquid_a = self.in_flow - liquid_b

        path = self.matter.path_data

        # Store in objects
        if self.oper_mode == 'Continuous':
            Liquid_a = LiquidStream(path, mole_frac=xa_liq,
                                    mole_flow=liquid_a,
                                    temp=self.temp, pres=self.pres)
            Liquid_b = LiquidStream(path, mole_frac=xb_liq,
                                    mole_flow=liquid_b,
                                    temp=self.temp, pres=self.pres)
        else:
            Liquid_a = LiquidPhase(path, mole_frac=xa_liq,
                                   moles=liquid_a,
                                   temp=self.temp, pres=self.pres)
            Liquid_b = LiquidPhase(path, mole_frac=xb_liq,
                                   moles=liquid_b,
                                   temp=self.temp, pres=self.pres)

        dens_a = Liquid_a.getDensity(basis='mole')
        dens_b = Liquid_b.getDensity(basis='mole')

        if phase_part > 0 and phase_part < 1:
            if dens_a > dens_b:
                self.Liquid_2 = Liquid_a
                self.Liquid_3 = Liquid_b
            else:
                self.Liquid_2 = Liquid_b
                self.Liquid_3 = Liquid_a

        else:
            self.Liquid_2 = Liquid_b
            self.Liquid_3 = Liquid_a

        # Result object
        if dens_a > dens_b:
            di = {'x_light': xb_liq, 'x_heavy': xa_liq,
                  'mol_heavy': liquid_a, 'mol_light': liquid_b,
                  'rho_light': dens_b, 'rho_heavy': dens_a}
        else:
            di = {'x_light': xa_liq, 'x_heavy': xb_liq,
                  'mol_heavy': liquid_b, 'mol_light': liquid_a,
                  'rho_light': dens_a, 'rho_heavy': dens_b}

        self.result = DynamicResult(self.states_di, **di)


class BatchExtractor(ContinuousExtractor):
    def __init__(self, k_fun=None, gamma_method='UNIQUAC'):
        """Create a batch extractor.

        Parameters
        ----------
        k_fun : callable, optional
            Equilibrium function. It must accept two liquid mole-fraction
            vectors ``x1`` [-] and ``x2`` [-] plus temperature ``temp`` [K],
            and return distribution coefficients ``K_i`` [-].
        gamma_method : {'ideal', 'UNIFAC', 'UNIQUAC'}, optional
            Activity-coefficient method used by the default ``k_fun``.

        Notes
        -----
        A batch destination is charged through ``Phases`` with an inventory
        [mol], so ``is_continuous`` is False. The extractor publishes no
        ``Outlet`` and an untimed result, so it cannot be a flowsheet
        source (issue #445).
        """
        super().__init__(k_fun, gamma_method)

        # ``super`` sets the continuous default; batch extraction must keep
        # amount semantics [mol] for Phases and downstream result objects.
        self.oper_mode = 'Batch'
        self.is_continuous = False
        self._Phases = None

    @property
    def Phases(self):
        """Liquid charge of the extractor (``LiquidPhase``), or None."""
        return self._Phases

    @Phases.setter
    def Phases(self, instance):
        """Charge the extractor with the liquid inventory to separate.

        Parameters
        ----------
        instance : LiquidPhase
            Liquid charge with mole fractions [-], amount [mol],
            temperature [K] and pressure [Pa].

        Notes
        -----
        A deep copy of the charge is kept as ``__original_phase__``, the
        initial-holdup snapshot that ``SimulationExec.get_holdup`` reports
        as a raw material unless the charge was transferred from an
        upstream unit (``transferred_from_uo``), whose source is charged
        instead. Earlier releases kept no snapshot, so a user charge was
        missing from ``GetRawMaterials`` and ``GetOPEX`` (issue #456).
        """
        fields = material_setter(instance, oper_mode=self.oper_mode)
        self._Phases = instance
        self.matter = self._Phases

        for key, val in fields.items():
            setattr(self, key, val)

        self.__original_phase__ = copy.deepcopy(instance)
