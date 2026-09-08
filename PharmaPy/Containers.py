# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 14:26:50 2020

@author: dcasasor
"""

from typing import Mapping, Optional, Sequence, Tuple, Union
from copy import copy

from PharmaPy._assimulo import CVode, Explicit_Problem

from PharmaPy.Phases import LiquidPhase, SolidPhase, classify_phases
from PharmaPy.Streams import LiquidStream, SolidStream
from PharmaPy.MixedPhases import Cake, Slurry, SlurryStream

from PharmaPy.NameAnalysis import get_dict_states
from PharmaPy.Crystallizers import SemibatchCryst
from PharmaPy.Connections import get_inputs_new

from PharmaPy.Commons import unpack_states

from PharmaPy.Results import DynamicResult

from scipy.optimize import newton, fsolve
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

eps = np.finfo(float).eps


class ContinuousHoldup:
    """Adiabatic, perfectly mixed liquid tank with fixed mass inventory.

    Inlet and outlet mass flows are equal. Composition and temperature evolve
    without reaction or external heat transfer; retrieval commits the terminal
    liquid state and absolute time for continuation.
    """

    def __init__(self) -> None:
        """Initialize an adiabatic, constant-mass continuous liquid holdup.

        Notes
        -----
        Assign a liquid phase to set the fixed inventory [kg] and an inlet
        stream supplying mass flow [kg/s], composition [-], and temperature
        [K]. Outlet mass flow equals inlet mass flow at every time [s].
        """
        self.is_continuous = True
        self._Inlet = None
        self._Phases = None

        self.elapsed_time = 0

        self.oper_mode = 'Continuous'

    @property
    def Inlet(self):
        return self._Inlet

    @Inlet.setter
    def Inlet(self, inlet):
        self._Inlet = inlet

    @property
    def Phases(self):
        return self._Phases

    @Phases.setter
    def Phases(self, phase):
        if isinstance(phase, (list, tuple)):
            self._Phases = phase
        elif phase.__module__ == 'PharmaPy.Phases':
            self._Phases = [phase]

        classify_phases(self)
        self.nomenclature()
        self.mass = self.Liquid_1.mass

    def nomenclature(self) -> None:
        """Declare holdup composition, temperature, and outlet-flow metadata.

        Notes
        -----
        Differential states are species mass fractions [-], followed by
        temperature [K]. Outlet mass flow [kg/s] is algebraic and equals the
        inlet flow because the liquid mass inventory [kg] remains constant.
        """
        name_comp = self.Liquid_1.name_species
        num_comp = len(name_comp)

        # Inlets
        dict_in = {'mass_flow': 1, 'mass_frac': num_comp, 'temp': 1}
        self.dict_states_in = dict_in
        self.names_states_in = list(dict_in.keys())
        self.names_states_out = self.names_states_in.copy()

        self.dict_states_in = {'Inlet': self.dict_states_in}

        # Phase
        states_di = {'mass_frac': {'type': 'diff', 'dim': num_comp,
                                   'index': name_comp},
                     'temp': {'type': 'diff', 'dim': 1, 'units': 'K'}}

        self.dim_states = [val['dim'] for val in states_di.values()]
        self.name_states = list(states_di.keys())

        self.states_di = states_di
        self.fstates_di = {'mass_flow': {'type': 'alg', 'dim': 1,
                                         'units': 'kg/s'}}

    def get_inputs(self, time):
        inputs = get_inputs_new(time, self.Inlet, self.dict_states_in)

        return inputs['Inlet']

    def unit_model(self, time, states):
        di_states = unpack_states(states, self.dim_states, self.name_states)

        inputs = self.get_inputs(time)

        material = self.material_balance(di_states['mass_frac'], inputs)
        energy = self.energy_balance(di_states['mass_frac'], di_states['temp'],
                                     inputs)

        balances = np.hstack((material, energy))

        return balances

    def material_balance(self, mass_frac, inputs):

        dw_dt = inputs['mass_flow'] / self.mass * (
            inputs['mass_frac'] - mass_frac)

        return dw_dt

    def energy_balance(self, mass_frac, temp, inputs):
        cp = self.Liquid_1.getCp(mass_frac=mass_frac, temp=temp, basis='mass')

        h_in = self.Inlet.getEnthalpy(temp=inputs['temp'],
                                      mass_frac=inputs['mass_frac'])
        h = self.Inlet.getEnthalpy(temp=temp, mass_frac=mass_frac)

        dtemp_dt = inputs['mass_flow'] / self.mass / cp * \
            (h_in - h)

        return dtemp_dt

    def solve_unit(self, runtime, verbose=True):
        w_init = self.Liquid_1.mass_frac
        temp_init = self.Liquid_1.temp

        states_init = np.hstack((w_init, temp_init))

        problem = Explicit_Problem(self.unit_model, states_init,
                                   t0=self.elapsed_time)
        solver = CVode(problem)

        if not verbose:
            solver.verbosity = 50

        final_time = runtime + self.elapsed_time
        time, states = solver.simulate(final_time)

        self.retrieve_results(time, states)

        return time, states

    def retrieve_results(self, time: Sequence[float],
                         states: np.ndarray) -> None:
        """Publish a holdup segment and commit its state for continuation.

        Parameters
        ----------
        time : sequence of float
            Absolute integration times [s], shape (num_times,).
        states : numpy.ndarray
            State rows of shape (num_times, num_species + 1): liquid mass
            fractions [-] followed by temperature [K].

        Notes
        -----
        ``outputs`` and the retained legacy ``output`` alias contain the
        current segment, including outlet mass flow [kg/s] equal to inlet
        flow. The final liquid state updates the fixed inventory [kg] and a
        separate ``LiquidStream`` outlet. ``elapsed_time`` becomes the
        absolute endpoint [s], so the next solve starts at the terminal state.
        """
        time = np.asarray(time)  # [s]
        di = unpack_states(states, self.dim_states, self.name_states)
        # di: species mass fractions [-] and temperature [K]
        inputs = self.get_inputs(time)
        di['mass_flow'] = np.broadcast_to(inputs['mass_flow'], time.shape).copy()  # [kg/s]
        di['time'] = time
        self.result = DynamicResult(self.states_di, self.fstates_di, **di)
        self.outputs = di
        self.output = self.outputs

        terminal_fraction = np.atleast_1d(di['mass_frac'][-1])  # [-], species vector
        self.Liquid_1.updatePhase(mass_frac=terminal_fraction,
                                  temp=di['temp'][-1], mass=self.mass)
        self.Outlet = LiquidStream(
            self.Liquid_1.path_data, mass_flow=di['mass_flow'][-1],
            mass_frac=terminal_fraction, temp=di['temp'][-1],
            pres=self.Liquid_1.pres)
        self.elapsed_time = time[-1]  # [s]


class Mixer:
    def __init__(self, temp_refer=298.15):

        self._Inlets = []

        self.oper_mode = None

        self.elapsed_time = 0

        self.temp_refer = temp_refer

        self.states_out_dict = {}

        self.nomenclature()
        self.is_continuous = None
        self.timeProf = None

        self.type_out = None
        self.outputs = None

    @property
    def Inlets(self):
        return self._Inlets

    @Inlets.setter
    def Inlets(self, inlets: object) -> None:
        """Append inlets sharing a material-amount basis.

        Parameters
        ----------
        inlets : object or sequence of object
            Liquid or mixed-phase inventories [kg], or streams [kg/s].

        Raises
        ------
        ValueError
            If batch inventories and flow rates are combined without a
            duration contract. Supply batch phases or use only streams.
        """
        incoming = list(inlets) if isinstance(inlets, (list, tuple)) else [inlets]
        flow_flags = [hasattr(inlet, 'mass_flow')
                      for inlet in self._Inlets + incoming]
        if any(flow_flags) and not all(flow_flags):
            raise ValueError(
                'Mixer cannot combine batch inventories [kg] with flow rates '
                '[kg/s] without a duration contract; supply batch phases or '
                'use only streams.')
        self._Inlets += incoming

        flow_flag = hasattr(self.Inlets[-1], 'mass_flow')

        if flow_flag:
            self.oper_mode = 'Continuous'
            self.is_continuous = True
        else:
            self.oper_mode = 'Batch'
            self.is_continuous = False

        if 'flow' in self.names_states_in:
            if flow_flag:
                self.names_states_in = self.names_states_in['flow']
            else:
                self.names_states_in = self.names_states_in['non_flow']

        self.names_upstream.append(None)
        self.bipartite.append(None)

        if flow_flag:
            self.states_di = {
                'mass_flow': {'units': 'kg/s', 'dim': 1, 'type': 'alg'},
                'mass_frac': {'units': 'kg', 'dim': 1,
                              'index': self.Inlets[0].name_species, 'type': 'alg'},
                'temp': {'units': 'K', 'dim': 1, 'type': 'alg'}
                }

        else:
            self.states_di = {
                'mass': {'units': 'kg', 'dim': 1, 'type': 'alg'},
                'mass_frac': {'units': 'kg', 'dim': 1,
                              'index': self.Inlets[0].name_species, 'type': 'alg'},
                'temp': {'units': 'K', 'dim': 1, 'type': 'alg'}
                }

        self.dim_states = [a['dim'] for a in self.states_di.values()]
        self.name_states = list(self.states_di.keys())

    def nomenclature(self):
        self.names_states_in = {
            'flow': ['mass_frac', 'mass_flow', 'temp'],
            'non_flow': ['mass_frac', 'mass', 'temp']}

        self.names_states_out = self.names_states_in
        self.names_upstream = []
        self.bipartite = []

    def get_inputs_new(self, time: Optional[Sequence[float]]) -> dict:
        """Collect liquid inputs with explicit time and species axes.

        Parameters
        ----------
        time : sequence of float or None
            Profile times [s]; None evaluates static inlet attributes.

        Returns
        -------
        dict
            Continuous inputs are per-inlet tuples: mass fractions [-] of
            shape (num_times, num_species), mass flows [kg/s] and temperatures
            [K] of shape (num_times,). Static streams broadcast to the
            connected grid when mixed with profiles; an entirely static mix
            has one time row. Batch inputs are arrays of masses [kg],
            temperatures [K], and fractions [-] with inlet and species axes.
        """
        inlets = self.Inlets

        massfracs = []
        masses = []
        temps = []

        if self.oper_mode == 'Continuous':
            names_in = [name for name in self.names_states_in
                        if name != 'mass']

            dict_list = []
            for ind, inlet in enumerate(self.Inlets):
                di = get_inputs_new(time, inlet, self.states_in_dict)['Inlet']
                # Retain a complete species vector even for one static time.
                di['mass_frac'] = np.asarray(di['mass_frac']).reshape(
                    -1, inlet.num_species)  # [-], time rows and species columns
                di['mass_flow'] = np.atleast_1d(di['mass_flow'])  # [kg/s]
                di['temp'] = np.atleast_1d(di['temp'])  # [K]

                dict_list.append(di)

            dict_inputs = {}

            for key in dict_list[0]:
                dict_inputs[key] = tuple(d[key] for d in dict_list)

            names_out = names_in

        else:
            for inlet in inlets:
                massfracs.append(inlet.mass_frac)
                temps.append(inlet.temp)

                # mass = getattr(inlet, 'mass', getattr(inlet, 'mass_flow'))
                masses.append(inlet.mass)

            masses = np.array(masses)
            massfracs = np.array(massfracs)
            dict_inputs = {'mass': masses, 'mass_frac': massfracs,
                           'temp': temps}

            # if is_mass:
            names_out = [name for name in self.names_states_out
                         if name != 'mass_flow']

            self.timeProf = [0]

        self.names_states_out = names_out

        return dict_inputs

    def get_inputs_solids(self) -> Tuple[dict, int]:
        """Collect static phase amounts on one common population basis.

        Returns
        -------
        dict
            Inlet arrays: temperature [K], liquid mass fractions [-] with
            shape (num_inlets, num_species), liquid/solid amounts [kg] for
            batch or [kg/s] for continuous mixing, and number distributions
            [#/um] or [#/um/s] with shape (num_inlets, num_sizes).
        int
            Index of the first solids-bearing inlet.

        Raises
        ------
        ValueError
            If a solids-bearing inlet differs from the first solid in shape
            factor ``kv`` [-], solid mass fractions [-], or size grid [um].
            Also raised if attached solid mass [kg], or mass flow [kg/s],
            disagrees with ``rho_solid * kv * mu_3`` on the same basis,
            where ``mu_3`` is integrated from the consumed size distribution.
            Solids without either a size distribution or a size grid are
            unsupported.
            The message identifies the differing quantity and inlet index.

        Notes
        -----
        Solids must share exactly the same shape factor and size-grid values.
        Compositions may differ only by normalization roundoff (relative
        tolerance 1e-12, zero absolute tolerance to retain trace species).
        A size distribution on a size grid is required. Its third moment is
        recomputed with ``SolidPhase.getMoments`` rather than read from cached
        ``moments``:
        moment-mode crystallizer outlets do not refresh their distribution.
        Attached solid distributions already represent total populations
        (or number rates for streams), including those attached to a Slurry.
        Cake saturation metadata does not override attached phase inventories.
        Attached solid amounts must be finite and satisfy
        ``abs(attached - moment_derived) <= 1e-9 * attached``. This relative
        tolerance is a numerical consistency allowance for floating-point
        conversions and integration, not a physical uncertainty. It has no
        absolute floor. MixedPhases inventory/enthalpy reconciliation is
        deferred: the mixer keeps attached masses authoritative, but Slurry
        enthalpy weights phases by moment-derived fractions, so inconsistent
        solids cannot be mixed conservatively.
        Profiled multiphase mixing remains outside this static path (#221).
        """

        timeseries_flag = []

        for inlet in self.Inlets:
            if getattr(inlet, 'y_upstream', None) is None:
                timeseries_flag.append(False)
            else:
                timeseries_flag.append(len(inlet.y_upstream) > 1)

        solids_flag = [hasattr(inlet, 'Solid_1') for inlet in self.Inlets]

        mass_solid = []
        mass_liquid = []

        massfrac_liq = []
        distrib_sol = []

        temps = []

        ind_solid = np.argmax(solids_flag)
        reference_solid = self.Inlets[ind_solid].Solid_1
        # Allow normalization roundoff without accepting a different material;
        # zero absolute tolerance preserves distinctions in trace species.
        composition_rtol = 1e-12  # [-], float64 fraction normalization allowance
        # Numerical agreement after floating-point moment integration and
        # amount conversions; this is not a physical modeling tolerance.
        inventory_rtol = 1e-9  # [-], relative consistency allowance; no absolute floor
        amount_name = 'mass_flow' if self.is_continuous else 'mass'
        amount_units = 'kg/s' if self.is_continuous else 'kg'
        for index, inlet in enumerate(self.Inlets):
            if not solids_flag[index]:
                continue
            solid = inlet.Solid_1
            if (getattr(solid, 'distrib', None) is None
                    or getattr(solid, 'x_distrib', None) is None):
                raise ValueError(
                    f'Mixer inlet {index}: the solids mixer requires a '
                    'size-distributed solid on a size grid; moment-only or gridless '
                    'populations are not supported.')
            attached_amount = getattr(solid, amount_name)  # [kg] or [kg/s]
            distribution_moment = np.asarray(solid.getMoments(
                x_distrib=solid.x_distrib, distrib=solid.distrib, mom_num=3)).item()  # [m**3] or [m**3/s]
            moment_amount = solid.getDensity() * solid.kv * distribution_moment  # [kg] or [kg/s]
            if not (np.isfinite(attached_amount) and np.isfinite(moment_amount)
                    and abs(attached_amount - moment_amount)
                    <= inventory_rtol * attached_amount):
                raise ValueError(
                    f'Mixer inlet {index} has attached solid '
                    f'{amount_name}={attached_amount:.12g} {amount_units} but '
                    f'moment-derived {amount_name}={moment_amount:.12g} {amount_units}. '
                    'The mixer keeps attached masses authoritative while slurry '
                    'enthalpy weights phases by moment-derived fractions, so '
                    'inconsistent solids cannot be mixed conservatively; see the '
                    'deferred MixedPhases inventory/enthalpy reconciliation. '
                    'Note that moment-mode crystallizer outlets do not refresh '
                    'their distribution.')
            if index == ind_solid:
                continue
            if solid.kv != reference_solid.kv:
                raise ValueError(
                    f'Mixer inlet {index} has kv different from inlet {ind_solid}.')
            if (solid.mass_frac.shape != reference_solid.mass_frac.shape
                    or not np.allclose(solid.mass_frac, reference_solid.mass_frac,
                                       rtol=composition_rtol, atol=0)):
                raise ValueError(
                    f'Mixer inlet {index} has mass_frac different from inlet {ind_solid}.')
            if not np.array_equal(solid.x_distrib, reference_solid.x_distrib):
                raise ValueError(
                    f'Mixer inlet {index} has x_distrib different from inlet {ind_solid}.')
        num_dist = reference_solid.distrib.shape[0]

        if any(timeseries_flag):
            pass
        else:
            for inlet in self.Inlets:
                if hasattr(inlet, 'Solid_1'):
                    mass_solid.append(getattr(inlet.Solid_1, amount_name))
                    mass_liquid.append(getattr(inlet.Liquid_1, amount_name))

                    massfrac_liq.append(inlet.Liquid_1.mass_frac)

                    distrib_sol.append(inlet.Solid_1.distrib)

                    temps.append(inlet.Liquid_1.temp)
                else:
                    mass_solid.append(0)
                    mass_liquid.append(getattr(inlet, amount_name))

                    massfrac_liq.append(inlet.mass_frac)
                    distrib_sol.append(np.zeros(num_dist))

                    temps.append(inlet.temp)

            massfrac_liq = np.array(massfrac_liq)
            mass_solid = np.array(mass_solid)
            mass_liquid = np.array(mass_liquid)
            temps = np.array(temps)
            distrib_sol = np.array(distrib_sol)

        dict_out = {'temp': temps, 'mass_frac': massfrac_liq,
                    'mass_liq': mass_liquid, 'mass_solid': mass_solid,
                    'num_distrib': distrib_sol}

        return dict_out, ind_solid

    def energy_balance(
            self, u_inputs: Mapping[str, np.ndarray]
            ) -> Union[float, np.floating]:
        """Solve the adiabatic outlet temperature for mixed-phase inlets.

        Parameters
        ----------
        u_inputs : mapping of str to numpy.ndarray
            Inlet states from :meth:`get_inputs_solids`. ``mass_frac`` is the
            liquid composition [-], ``temp`` is temperature [K], and
            ``mass_liq`` and ``mass_solid`` are phase inventories [kg] for
            batch inlets or phase flow rates [kg/s] for continuous inlets.
            ``num_distrib`` is the particle-number distribution [#/um] used
            by cake collaborators.

        Returns
        -------
        float or numpy.floating
            Adiabatic outlet temperature [K].

        Raises
        ------
        TypeError
            If an inlet exposes a solid phase but is neither a supported
            :class:`Slurry` nor :class:`Cake` collaborator.
        RuntimeError
            If the Newton iteration cannot find the energy-balance root.

        Notes
        -----
        Inlet and outlet enthalpies use the total-mixture mass basis [J/kg].
        Weighting them by total slurry mass [kg], or total slurry mass flow
        [kg/s], keeps both sides of the balance on a common energy [J], or
        energy-rate [J/s], basis.
        """
        massfrac_in = u_inputs['mass_frac']  # [-]
        temp_in = u_inputs['temp']  # [K]
        mass_total_in = (
            u_inputs['mass_liq'] + u_inputs['mass_solid']
        )  # [kg] for batch inputs or [kg/s] for continuous inputs

        h_in = []  # [J/kg]
        for ind, inlet in enumerate(self.Inlets):
            if isinstance(inlet, Slurry):
                h_in.append(inlet.getEnthalpy(temp_in[ind],
                                              volumetric=False))
            elif isinstance(inlet, Cake):
                distrib_in = u_inputs['num_distrib']  # [#/um]
                h_in.append(inlet.getEnthalpy(temp_in[ind],
                                              mass_frac=massfrac_in[ind],
                                              distrib=distrib_in[ind]))
            elif hasattr(inlet, 'Solid_1'):
                raise TypeError(
                    "Mixer.energy_balance supports Slurry and Cake "
                    "solids-bearing inlets; got %s."
                    % type(inlet).__name__
                )
            else:
                h_in.append(inlet.getEnthalpy(temp_in[ind],
                                              mass_frac=massfrac_in[ind]))

        def temp_root(temp: float) -> Union[float, np.floating]:
            """Return the total enthalpy residual at an outlet temperature.

            Parameters
            ----------
            temp : float
                Trial outlet temperature [K].

            Returns
            -------
            float or numpy.floating
                Inlet-minus-outlet enthalpy [J] for batch inputs or enthalpy
                flow [J/s] for continuous inputs.
            """
            if isinstance(self.Outlet, Slurry):
                h_out = self.Outlet.getEnthalpy(
                    temp, volumetric=False)  # [J/kg]
            else:
                h_out = self.Outlet.getEnthalpy(temp)  # [J/kg]

            balance = (
                mass_total_in.dot(h_in) - mass_total_in.sum() * h_out
            )  # [J] for batch inputs or [J/s] for continuous inputs

            return balance

        temp_seed = np.mean(temp_in)  # [K]
        temp_bce = newton(temp_root, temp_seed)  # [K]

        return temp_bce

    def balances(self, u_inputs: Mapping[str, np.ndarray]) -> tuple:
        """Close batch liquid material and adiabatic energy balances.

        Parameters
        ----------
        u_inputs : mapping of str to numpy.ndarray
            Masses [kg] and temperatures [K], shape (num_inlets,), and liquid
            mass fractions [-], shape (num_inlets, num_species).

        Returns
        -------
        tuple
            Total mass [kg], mass fractions [-] with shape (num_species,),
            and temperature [K], using ``temp_refer`` [K] on both sides.
        """
        massfrac_in = u_inputs['mass_frac']
        mass_in = u_inputs['mass']
        temp_in = u_inputs['temp']

        # ---------- Material balances
        total_mass = mass_in.sum()
        massfrac = np.dot(mass_in, massfrac_in) / total_mass

        # ---------- Energy balance
        h_in = []
        for temp, mass_frac in zip(temp_in, massfrac_in):
            h_in.append(self.Liquid_1.getEnthalpy(
                temp, mass_frac=mass_frac, temp_ref=self.temp_refer))

        def temp_root(temp: float) -> float:
            """Evaluate the batch enthalpy residual.

            Parameters
            ----------
            temp : float
                Trial outlet temperature [K].

            Returns
            -------
            float
                Inlet minus outlet energy [J].
            """
            h_out = self.Liquid_1.getEnthalpy(temp, temp_ref=self.temp_refer,
                                              mass_frac=massfrac)

            balance = mass_in.dot(h_in) - total_mass * h_out

            return balance

        temp_seed = np.mean(temp_in)
        temp_bce = newton(temp_root, temp_seed)

        return total_mass, massfrac, temp_bce

    def dynamic_balances(self, u_inputs: Mapping[str, Sequence[np.ndarray]]) -> tuple:
        """Mix aligned liquid inlet profiles with a common enthalpy reference.

        Parameters
        ----------
        u_inputs : mapping of str to sequence of numpy.ndarray
            Per-inlet mass flows [kg/s] and temperatures [K], shape
            (num_times,), and mass fractions [-], shape
            (num_times, num_species).

        Returns
        -------
        tuple
            Total mass-flow profile [kg/s], mixed mass fractions [-], and
            temperatures [K], preserving time and species axes.
        """
        massfrac_in = u_inputs['mass_frac']
        mass_in = u_inputs['mass_flow']
        temp_in = u_inputs['temp']

        # ---------- Material balances
        total_mass = sum(mass_in)
        masscomp_in = [(frac.T * mass).T for (frac, mass)
                       in zip(massfrac_in, mass_in)]
        massfrac = sum(masscomp_in) / total_mass[..., np.newaxis]

        # ---------- Energy balance
        h_in = []
        for temp, mass_frac in zip(temp_in, massfrac_in):
            h_in.append(self.Liquid_1.getEnthalpy(
                temp, mass_frac=mass_frac, temp_ref=self.temp_refer))

        energy_in = sum([mass * enth for (mass, enth) in zip(mass_in, h_in)])

        def temp_root(temp: np.ndarray, ind: Optional[int] = None) -> np.ndarray:
            """Evaluate continuous enthalpy-flow residuals.

            Parameters
            ----------
            temp : numpy.ndarray
                Trial outlet temperature [K], shape (1,) or (num_times,).
            ind : int, optional
                Profile row; omitted to evaluate the complete profile.

            Returns
            -------
            numpy.ndarray
                Inlet minus outlet enthalpy flow [J/s].
            """
            if ind is None:
                h_out = self.Liquid_1.getEnthalpy(temp, temp_ref=self.temp_refer,
                                                  mass_frac=massfrac)

                balance = energy_in - total_mass * h_out
            else:
                h_out = self.Liquid_1.getEnthalpy(temp, temp_ref=self.temp_refer,
                                                  mass_frac=massfrac[ind])
                balance = energy_in[ind] - total_mass[ind] * h_out

            return balance

        temp_seed = sum(temp_in) / 2

        temp_seed = temp_seed[0]
        temp_bce = np.zeros(massfrac.shape[0])
        for idx in range(len(temp_bce)):
            temp_bce[idx] = fsolve(temp_root, temp_seed, args=(idx, )).item()  # [K]
            temp_seed = temp_bce[idx]

        return total_mass, massfrac, temp_bce

    def balances_solids(self, u_inputs: Mapping[str, np.ndarray],
                        ind_solids: int) -> tuple:
        """Mix static liquid and crystal amounts without renormalizing counts.

        Parameters
        ----------
        u_inputs : mapping of str to numpy.ndarray
            Inputs from ``get_inputs_solids``: phase amounts [kg] or [kg/s],
            liquid fractions [-], temperatures [K], and total-population
            distributions [#/um] or number-rate distributions [#/um/s].
        ind_solids : int
            Index of the inlet supplying the solid composition, size grid
            [um], and phase-owned volumetric shape factor [-].

        Returns
        -------
        tuple
            Liquid and solid amounts [kg] or [kg/s], mixed liquid fractions
            [-], distribution [#/m**3/um] for slurry or [#/um] for cake,
            and adiabatic temperature [K].

        Notes
        -----
        ``get_inputs_solids`` validates that inlets share solid composition,
        size grid, and shape factor before their populations are summed.
        Each attached solid mass [kg], or flow [kg/s], must also agree with
        ``rho_solid * kv * mu_3`` within relative tolerance 1e-9, with
        no absolute floor. The third moment is integrated from the consumed
        size distribution, not cached moments; moment-only outlets are
        unsupported. This is a numerical consistency check, not a
        physical tolerance: until MixedPhases inventory/enthalpy reconciliation
        is resolved, inconsistent amounts cannot be mixed conservatively.
        A solid constructor with zero mass consumes raw number counts; the
        explicit balanced mass is reconciled afterward. Slurry construction
        then normalizes the total population by the combined phase volume.
        Cake uses attached phase masses, independent of inlet saturation.
        The outlet type is selected from the balanced population's solid
        volume ``kv * mu_3`` [m**3]; zero populations always produce Slurry.
        Saturation defensively uses the returned Cake's own pore-volume
        geometry, identical to the mass basis for validated inputs, and
        assumes uniform filling. An inlet Cake supplies a copied float
        spatial grid, so later changes to its grid cannot affect the outlet.
        Continuous inputs always produce a SlurryStream: Cake has no flow
        or duration contract.
        """
        mass_liquid = u_inputs['mass_liq']  # [kg] or [kg/s]
        mass_solid = u_inputs['mass_solid']  # [kg] or [kg/s]
        massfrac_liq = u_inputs['mass_frac']  # [-]
        total_solid = mass_solid.sum()  # [kg] or [kg/s]
        total_liquid = mass_liquid.sum()  # [kg] or [kg/s]
        massfrac = mass_liquid.dot(massfrac_liq) / total_liquid  # [-]
        total_distrib = u_inputs['num_distrib'].sum(axis=0)  # [#/um] or [#/um/s]

        inlet_solid = self.Inlets[ind_solids].Solid_1
        path = self.Inlets[ind_solids].Liquid_1.path_data
        solid_args = dict(mass_frac=inlet_solid.mass_frac, distrib=total_distrib,
                          x_distrib=inlet_solid.x_distrib, kv=inlet_solid.kv)
        # Zero constructor mass selects the documented raw-number basis.
        if self.is_continuous:
            liquid_out = LiquidStream(path, mass_flow=total_liquid,
                                      mass_frac=massfrac)
            solid_out = SolidStream(path, mass_flow=0, **solid_args)
            solid_out.updatePhase(mass_flow=total_solid)
            self.Outlet = SlurryStream()
        else:
            liquid_out = LiquidPhase(path, mass=total_liquid, mass_frac=massfrac)
            solid_out = SolidPhase(path, mass=0, **solid_args)
            solid_out.updatePhase(mass=total_solid)
            solid_volume = solid_out.kv * solid_out.moments[3]  # [m**3]
            make_cake = False
            if solid_volume > 0:
                porosity = solid_out.getPorosity()  # [-], balanced population packing
                pore_volume = solid_volume * porosity / (1 - porosity)  # [m**3]
                make_cake = liquid_out.vol <= pore_volume
            if make_cake:
                solid_inlet = self.Inlets[ind_solids]
                z_external = (np.asarray(solid_inlet.z_external, dtype=float).copy()
                              if isinstance(solid_inlet, Cake) else None)  # [m]
                self.Outlet = Cake(z_external=z_external)
            else:
                self.Outlet = Slurry()

        self.Outlet.Phases = (liquid_out, solid_out)
        if isinstance(self.Outlet, Cake):
            # Use the returned Cake's own geometry defensively; validation
            # makes the moment and attached-mass volume bases equivalent.
            pore_volume = self.Outlet.cake_vol * self.Outlet.porosity  # [m**3]
            self.Outlet.saturation = np.full_like(
                self.Outlet.z_external, liquid_out.vol / pore_volume,
                dtype=float)  # [-]
        if isinstance(self.Outlet, Slurry):
            self.type_out = 'Slurry'
            distrib = self.Outlet.distrib  # [#/m**3/um]
        else:
            self.type_out = 'Cake'
            distrib = total_distrib  # [#/um]

        temp_out = self.energy_balance(u_inputs)  # [K]
        return total_liquid, total_solid, massfrac, distrib, temp_out

    def solve_unit(self) -> tuple:
        """Solve instantaneous batch or continuous mixing and publish Outlet.

        Returns
        -------
        tuple
            Liquid-only amounts [kg] or flows [kg/s], mass fractions [-],
            and temperatures [K]. Solids mixing returns liquid/solid amounts,
            liquid mass fractions, distribution [#/m**3/um] for slurry or
            [#/um] for cake, and temperature [K].

        Raises
        ------
        ValueError
            If a multi-sample liquid inlet's time window does not overlap the
            chosen grid or starts after that grid begins. The message names
            the zero-based inlet index and both windows [s].

        Notes
        -----
        The first connected liquid inlet with more than one sample supplies
        the evaluation grid [s]. Single-sample inlets are constant feeds: they
        neither select the grid nor restrict its time window. Other connected
        inlets must overlap the grid and cover its beginning. Endpoint
        comparisons allow ``sqrt(machine epsilon) * grid span`` [s] as a
        numerical roundoff allowance for independently propagated clocks,
        not a physical extrapolation allowance. Values past an inlet's last
        time within the grid are held at its endpoint. Static streams
        broadcast to the chosen grid.

        Entirely static liquid mixing publishes a single time [s]
        and contributes no duration. Duration-based flowsheet accounting of
        instantaneous units is a separate SimExec concern.
        """

        # ---------- Read inputs
        solids_flag = [inlet.__module__ == 'PharmaPy.MixedPhases'
                       for inlet in self.Inlets]

        len_in = (self.Inlets[0].num_species, 1, 1)

        states_in_dict = dict(zip(self.names_states_in, len_in))

        time_prof = None

        if any(solids_flag):
            self.states_in_dict = {'Inlet': states_in_dict}  # TODO (solids?)
            u_input, ind_solids = self.get_inputs_solids()

            path = self.Inlets[ind_solids].Liquid_1.path_data
            if isinstance(u_input['mass_frac'], list):
                pass
            else:
                states = self.balances_solids(u_input, ind_solids)
                # Reuse balanced liquid fractions [-] and the appropriate
                # batch inventory [kg] or continuous mass flow [kg/s].
                if self.is_continuous:
                    self.Liquid_1 = LiquidStream(
                        path, mass_frac=states[2], mass_flow=states[0])
                else:
                    self.Liquid_1 = LiquidPhase(
                        path, mass_frac=states[2], mass=states[0])
        else:
            self.states_in_dict = {'Inlet': states_in_dict}
            time_prof = [0]  # [s], instantaneous static mixing

            if self.is_continuous:
                time_prof = np.zeros(1)  # [s], static continuous-source time
                for inlet in self.Inlets:
                    # A single upstream sample is constant for every time.
                    inlet_time = getattr(inlet, 'time_upstream', None)  # [s]
                    if inlet_time is not None and np.size(inlet_time) > 1:
                        time_prof = np.atleast_1d(inlet_time)  # [s]
                        break

                clock_tolerance = np.sqrt(eps) * (
                    time_prof[-1] - time_prof[0])  # [s], clock roundoff allowance
                for index, inlet in enumerate(self.Inlets):
                    inlet_time = getattr(inlet, 'time_upstream', None)  # [s]
                    if inlet_time is None or np.size(inlet_time) == 1:
                        continue
                    inlet_time = np.atleast_1d(inlet_time)  # [s]
                    # Start coverage also guarantees a start before grid end.
                    if (inlet_time[-1] < time_prof[0] - clock_tolerance
                            or inlet_time[0] > time_prof[0] + clock_tolerance):
                        raise ValueError(
                            f'Inlet {index} window [{inlet_time[0]}, '
                            f'{inlet_time[-1]}] s must overlap the mixer grid '
                            f'[{time_prof[0]}, {time_prof[-1]}] s and cover '
                            'its start; supply aligned inlet profiles.')

            u_input = self.get_inputs_new(time_prof)

            # ---------- Create output phase
            path = self.Inlets[0].path_data
            if self.is_continuous:
                self.Liquid_1 = LiquidStream(path_thermo=path,
                                             mass_frac=u_input['mass_frac'][0][0],
                                             mass_flow=eps)
            else:
                self.Liquid_1 = LiquidPhase(path_thermo=path,
                                            mass_frac=u_input['mass_frac'][0],
                                            mass=eps)

            # ---------- Run balances
            if self.is_continuous:
                states = self.dynamic_balances(u_input)
            else:
                states = self.balances(u_input)

        # ---------- Retrieve results
        self.retrieve_results(time_prof, states)

        return states

    def retrieve_results(self, time: Optional[Sequence[float]], states: tuple) -> None:
        """Publish mixer balances and commit final phase states.

        Parameters
        ----------
        time : sequence of float or None
            Profile times [s]; None for static solids mixing.
        states : tuple
            Returned balance quantities: liquid-only amount [kg] or flow
            [kg/s], composition [-], temperature [K]; for solids, liquid and
            solid amounts, liquid composition, slurry volume-specific or cake
            total distribution, and temperature, as in ``balances_solids``.

        Notes
        -----
        Solid populations and phase amounts are already reconciled by
        ``balances_solids``. Retrieval commits temperature without replacing
        a total solid population with a volume-specific slurry distribution.
        For continuous liquid mixing, outlet temperature is committed before
        ``updatePhase`` reconciles volumetric flow at that temperature.
        Entirely static, instantaneous liquid mixing publishes one time [s]
        and contributes no duration. Duration-based accounting for these units
        is a separate SimExec concern.
        """
        solids_flag = [inlet.__module__ == 'PharmaPy.MixedPhases'
                       for inlet in self.Inlets]

        if any(solids_flag):
            mass_liq, mass_sol, massfrac_liq, distrib, temp = states

            if self.type_out == 'Slurry':
                self.names_states_out = ['mass_liq', 'temp', 'num_distrib']
            else:
                self.names_states_out = ['mass_liq', 'temp', 'total_distrib']

            self.outputs = states

            # Amounts and populations were reconciled during construction.
            self.Outlet.Liquid_1.temp = temp
            self.Outlet.Solid_1.temp = temp
            self.Outlet.temp = temp  # [K]

            self.timeProf = [0]

        else:
            mass, massfrac, temp = states

            # ---------- Update phases
            if massfrac.ndim == 1:
                last_massfrac = massfrac
                last_mass = mass

                result = dict(zip(self.name_states, states))
                result['time'] = time

                self.result = DynamicResult(self.states_di, **result)

            else:
                last_massfrac = massfrac[-1]
                last_mass = mass[-1]

                result = dict(zip(self.name_states, states))
                result['time'] = time

                self.result = DynamicResult(self.states_di, **result)

            self.outputs = result

            if self.is_continuous:
                self.Liquid_1.temp = np.asarray(temp).reshape(-1)[-1].item()  # [K]
                self.Liquid_1.updatePhase(mass_frac=last_massfrac,
                                          mass_flow=last_mass)

                self.massFracProf = massfrac
                self.massFlowProf = mass
                self.tempProf = temp
            else:
                self.Liquid_1.temp = temp
                self.Liquid_1.updatePhase(mass_frac=last_massfrac,
                                          mass=last_mass)

            self.Outlet = self.Liquid_1


class DynamicCollector:
    """Dynamic holdup model for liquid and crystallizing inlet streams.

    The inlet phase type selects either a liquid-mixer balance or a delegated
    semibatch crystallizer model. State labels and result retrieval therefore
    follow the selected model's declared state layout.
    """

    def __init__(self, temp_refer: float = 298.15,
                 tau: Optional[float] = None,
                 num_interp_points: int = 3) -> None:
        """Initialize an unconnected dynamic collector.

        Parameters
        ----------
        temp_refer : float, optional
            Reference temperature retained for API compatibility [K].
        tau : float, optional
            Residence-time configuration retained for flowsheet use [s].
        num_interp_points : int, optional
            Number of inlet interpolation points [-].
        """

        self._Inlet = None
        self.num_interp_points = num_interp_points

        self.tau = tau
        self.vol_offset = 0.75

        self.oper_mode = 'Dynamic'

        self._Phases = None

        self.is_continuous = False
        self.has_solids = None

        self.names_upstream = None
        self.bipartite = None

        self.nomenclature()

        # Crystallizer instances
        self.KinCryst = None
        self.CrystInst = None
        self.is_cryst = False

        self.kwargs_cryst = None

        self.elapsed_time = 0
        self.oper_mode = 'Continuous'
        self.oper_mode = 'Semibatch'

        self.outputs = None

    @property
    def Phases(self):
        return self._Phases

    @Phases.setter
    def Phases(self, phases):
        if isinstance(phases, (list, tuple)):
            self._Phases = phases
        elif phases.__module__ == 'PharmaPy.Phases':
            if self._Phases is None:
                self._Phases = [phases]
            else:
                self._Phases.append(phases)

        classify_phases(self)

    @property
    def Inlet(self):
        return self._Inlet

    @Inlet.setter
    def Inlet(self, inlet_object: Union[LiquidStream, SlurryStream]) -> None:
        """Assign an inlet and select its compatible collector model.

        Parameters
        ----------
        inlet_object : LiquidStream or SlurryStream
            Upstream liquid or crystallizing stream. Liquid composition uses
            mass fractions [-]; slurry concentration uses [kg/m**3].
        """
        module = inlet_object.__module__

        if module == 'PharmaPy.MixedPhases':
            self.name_species = inlet_object.Phases[0].name_species

            names_states_in = self.names_states_in['crystallizer']
            self.model_type = 'crystallizer'

            states_in_dict = dict.fromkeys(names_states_in)

        else:
            self.name_species = inlet_object.name_species

            names_states_in = self.names_states_in['liquid_mixer']
            self.model_type = 'liquid_mixer'

            len_in = [len(self.name_species), 1, 1]

            states_in_dict = dict(zip(names_states_in, len_in))

        self.is_cryst = self.model_type == 'crystallizer'
        self.num_species = len(self.name_species)

        self.states_in_dict = {'Inlet': states_in_dict}

        self._Inlet = inlet_object

    def nomenclature(self) -> None:
        """Declare collector input alternatives and output state names.

        Notes
        -----
        Crystallizer inputs accept FVM distributions [#/m**3/um] or SI
        moments [m**n/m**3]. Connection matches the available upstream name;
        solve_unit selects the population representation before initialization.
        Liquid composition is [kg/kg], flow [kg/s], and temperature [K].
        """
        names_liquid = ['mass_frac', 'mass_flow', 'temp']
        names_solids = ['mass_conc', 'vol_flow', 'temp', 'distrib', 'mu_n']

        self.names_states_in = {'liquid_mixer': names_liquid,
                                'crystallizer': names_solids}

        names_out_liquid = ['mass_frac', 'mass', 'temp']
        names_out_solids = ['mass_conc', 'vol', 'temp', 'total_distrib']
        self.names_states_out = {'liquid_mixer': names_out_liquid,
                                 'crystallizer': names_out_solids}

    def get_inputs(self, time):
        all_inputs = self.Inlet.InterpolateInputs(time)

        if hasattr(self.Inlet, 'Solid_1'):
            num_distrib = self.Inlet.Solid_1.num_distrib
        else:
            num_distrib = 0
        inputs = get_dict_states(self.names_upstream, self.num_species,
                                 num_distrib, all_inputs)

        input_dict = {}
        for name in self.names_states_in[self.model_type]:
            input_dict[name] = inputs[self.bipartite[name]]
        return input_dict

    def get_inputs_new(self, time: Union[float, np.ndarray]) -> dict:
        """Resolve collector feed fields without changing the inlet stream.

        Parameters
        ----------
        time : float or numpy.ndarray
            Evaluation time [s], scalar or shape (num_times,).

        Returns
        -------
        dict
            Inlet fields: mass fractions [-], mass flow [kg/s], temperature
            [K] for liquid feeds; concentrations [kg/m**3], volume flow
            [m**3/s], temperature [K], and moments [m**n/m**3] or distribution
            [#/m**3/um] for slurry feeds. Multiple times put time first;
            a scalar or one-element time array retains species/population
            vectors of shape (num_species,) or (num_population,).

        Notes
        -----
        When mu_n is an active input, SlurryStream fallback fields come from
        its own moments and attached liquid concentration. Without an attached
        liquid phase, concentration retains the generic missing-field default.
        Connected upstream and dynamic inlet values take precedence for each
        supplied field. Aliases are installed on a shallow copy so the original
        stream is unchanged.
        """
        inlet = self.Inlet
        if (isinstance(inlet, SlurryStream)
                and 'mu_n' in self.states_in_dict['Inlet']):
            inlet = copy(inlet)
            inlet.mu_n = self.Inlet.moments  # [m**n/m**3], slurry-volume basis
            if hasattr(self.Inlet, 'Liquid_1'):
                inlet.mass_conc = self.Inlet.Liquid_1.mass_conc  # [kg/m**3 liquid]
        return get_inputs_new(time, inlet, self.states_in_dict)

    def unit_model(self, time, states):
        # Calculate inlets
        u_values = self.get_inputs_new(time)['Inlet']

        fracs = states[:self.num_species]

        mass = states[self.num_species]
        temp = states[self.num_species + 1]

        material_balances = self.material_balances(time, fracs, mass, u_values)
        energy_balance = self.energy_balance(time, fracs, mass, temp, u_values)

        balances = np.append(material_balances, energy_balance)

        return balances

    def material_balances(self, time, fracs, mass, u_inputs):
        inlet_flow = u_inputs['mass_flow']
        inlet_fracs = u_inputs['mass_frac']

        dfrac_dt = inlet_flow / mass * (inlet_fracs - fracs)
        dm_dt = inlet_flow

        dmaterial_dt = np.append(dfrac_dt, dm_dt)

        return dmaterial_dt

    def energy_balance(self, time: float, fracs: np.ndarray, mass: float,
                       temp: float, u_inputs: Mapping[str, object]) -> float:
        """Return the adiabatic collector temperature derivative.

        Parameters
        ----------
        time : float
            Current time [s], retained for the balance interface.
        fracs : numpy.ndarray
            Tank liquid mass fractions [-], shape (num_species,).
        mass : float
            Tank liquid inventory [kg].
        temp : float
            Tank temperature [K].
        u_inputs : mapping of str to object
            Feed mass flow [kg/s], mass fractions [-], and temperature [K].

        Returns
        -------
        float
            Temperature derivative [K/s], using mass-basis enthalpy and Cp.
        """
        inlet_flow = u_inputs['mass_flow']
        inlet_fracs = u_inputs['mass_frac']
        inlet_temp = u_inputs['temp']

        h_in = self.Inlet.getEnthalpy(temp=inlet_temp, mass_frac=inlet_fracs)
        h_tank = self.Liquid_1.getEnthalpy(temp=temp, mass_frac=fracs)
        cp_tank = self.Liquid_1.getCp(temp=temp, mass_frac=fracs,
                                       basis='mass')  # [J/kg/K]

        dtemp_dt = inlet_flow / mass / cp_tank * (h_in - h_tank)

        return dtemp_dt

    def solve_unit(self, runtime: Optional[float] = None,
                   time_grid: Optional[Sequence[float]] = None,
                   verbose: bool = True,
                   sundials_opts: Optional[Mapping[str, object]] = None
                   ) -> Tuple[np.ndarray, np.ndarray]:
        """Solve the collector model selected by its inlet phase type.

        Parameters
        ----------
        runtime : float, optional
            Integration duration measured from ``elapsed_time`` [s].
        time_grid : sequence of float, optional
            Requested integration times [s]. If supplied with ``runtime``, its
            final value determines the liquid-mixer integration end time.
        verbose : bool, optional
            Whether the delegated solver should emit its normal progress
            output.
        sundials_opts : mapping of str to object, optional
            CVode option names and values for integration. Supplied options
            are also forwarded to the delegated ``SemibatchCryst`` solve.

        Returns
        -------
        time : numpy.ndarray
            Integration times [s] with shape ``(n_time,)``.
        states : numpy.ndarray
            State trajectory with shape ``(n_time, n_states)``. Liquid-mixer
            columns are mass fractions [-], holdup mass [kg], then
            temperature [K]; crystallizer columns follow ``states_di`` from
            the delegated crystallizer.

        Notes
        -----
        Supply ``runtime`` or ``time_grid``. Crystallizer states are retained
        on the delegated model rather than interpreted as liquid-only states.
        Crystallizer seed liquid volume uses the inlet liquid fraction
        1 - kv*mu_3 [-], with mu_3 on the slurry-volume basis [m**3/m**3].
        The seed solid retains the inlet solid phase's shape factor kv [-].
        Gridless MSMPR outlets use SI inlet mu_n [m**n/m**3] to initialize
        total seed moments [m**n] and delegate a moment-mode SemibatchCryst.
        No distribution is synthesized for moment-mode collection.
        Selector dictionaries and caller-owned ``kwargs_cryst`` are retained
        across solves. ``target_ind`` selects the seed solid species and is
        omitted only from a local copy of crystallizer constructor options;
        the collector's ``num_interp_points`` takes precedence in that copy.

        Raises
        ------
        ValueError
            If a liquid-mixer solve receives neither ``runtime`` nor
            ``time_grid`` and therefore has no integration end time [s].
        """
        if self.model_type == 'crystallizer':

            moment_mode = self.Inlet.distrib is None
            population_name = 'mu_n' if moment_mode else 'distrib'
            unused_name = 'distrib' if moment_mode else 'mu_n'
            # Rebuild active dimensions without consuming selector metadata.
            self.states_in_dict = {'Inlet': dict.fromkeys(
                name for name in self.names_states_in[self.model_type]
                if name != unused_name)}
            if moment_mode:
                self.states_in_dict['Inlet'][population_name] = len(self.Inlet.moments)
            else:
                self.states_in_dict['Inlet'][population_name] = len(self.Inlet.x_distrib)
            self.states_in_dict['Inlet']['mass_conc'] = len(self.Inlet.Liquid_1.mass_conc)
            self.states_in_dict['Inlet']['vol_flow'] = 1
            self.states_in_dict['Inlet']['temp'] = 1

            init_dict = self.get_inputs_new(self.elapsed_time)['Inlet']

            path = self.Inlet.Liquid_1.path_data

            vol_init = np.sqrt(eps)  # [m**3], established small positive seed volume
            conc_init = init_dict['mass_conc']  # [kg/m**3]
            # Total [m**n] for moments or [#/um] for FVM, using slurry volume.
            population_init = init_dict[population_name] * vol_init
            temp_init = init_dict['temp']  # [K]

            kv_inlet = self.Inlet.Solid_1.kv  # [-], phase-owned crystal shape
            if moment_mode:
                vol_init -= kv_inlet * population_init[3]  # [m**3], liquid at initial time
            else:
                vol_init *= (1 - kv_inlet * self.Inlet.moments[3])  # [m**3], liquid

            liquid = LiquidPhase(path, temp=temp_init, mass_conc=conc_init,
                                 vol=vol_init)

            frac_solid = np.zeros_like(conc_init)
            frac_solid[self.kwargs_cryst['target_ind']] = 1
            population_args = ({'moments': population_init} if moment_mode else
                               {'distrib': population_init,
                                'x_distrib': self.Inlet.Solid_1.x_distrib})
            solid = SolidPhase(path, temp=temp_init, mass_frac=frac_solid,
                               kv=kv_inlet, **population_args)

            phases = (liquid, solid)

            kwargs_cryst = dict(self.kwargs_cryst)
            kwargs_cryst.pop('target_ind')
            kwargs_cryst['num_interp_points'] = self.num_interp_points
            method = 'moments' if moment_mode else '1D-FVM'
            SemiCryst = SemibatchCryst(method=method, adiabatic=True,
                                       **kwargs_cryst)
            SemiCryst.Phases = phases
            SemiCryst.Kinetics = self.KinCryst
            SemiCryst.Inlet = self.Inlet

            SemiCryst.names_upstream = self.names_upstream
            SemiCryst.bipartite = self.bipartite

            self.states_di = SemiCryst.states_di

            SemiCryst.elapsed_time = self.elapsed_time

            time, states = SemiCryst.solve_unit(
                runtime, time_grid, verbose=verbose, sundials_opts=sundials_opts)

            # Retrieve crystallizer results
            output_names = ['Outlet', 'outputs']

            for name in output_names:
                setattr(self, name, getattr(SemiCryst, name))

            self.CrystInst = SemiCryst

            self.retrieve_results(time, states)

            vol_phase = self.Outlet.vol
            if isinstance(vol_phase, np.ndarray):
                vol_phase = vol_phase[0]

            self.vol_phase = vol_phase

            self.Phases = phases

        elif self.model_type == 'liquid_mixer':
            path = self.Inlet.path_data

            init_dict = self.get_inputs_new(self.elapsed_time)['Inlet']

            mass_init = init_dict['mass_flow'] / 10
            frac_init = init_dict['mass_frac']
            temp_init = init_dict['temp']

            liquid = LiquidPhase(path, temp=temp_init, mass_frac=frac_init)

            states_init = np.hstack((frac_init, mass_init, temp_init))

            self.Phases = (liquid,)

            self.states_di = {
                'mass_frac': {'units': '', 'dim': self.num_species,
                              'index': self.Liquid_1.name_species,
                              'type': 'diff'},
                'mass': {'units': 'kg', 'dim': 1, 'type': 'diff'},
                'temp': {'units': 'K', 'dim': 1, 'type': 'diff'}
                }

            self.dim_states = [a['dim'] for a in self.states_di.values()]
            self.name_states = list(self.states_di.keys())

            problem = Explicit_Problem(self.unit_model, states_init,
                                       t0=self.elapsed_time)
            solver = CVode(problem)

            if sundials_opts is not None:
                for name, val in sundials_opts.items():
                    setattr(solver, name, val)

                    if name == 'time_limit':
                        solver.report_continuously = True

            if not verbose:
                solver.verbosity = 50

            if time_grid is not None:
                final_time = time_grid[-1]  # [s]
            elif runtime is not None:
                final_time = runtime + self.elapsed_time  # [s]
            else:
                raise ValueError(
                    "DynamicCollector.solve_unit requires 'runtime' [s] or "
                    "'time_grid' [s]; neither was supplied."
                )

            time, states = solver.simulate(final_time, ncp_list=time_grid)

            self.retrieve_results(time, states)

            vol_liq = self.Liquid_1.vol
            if isinstance(vol_liq, np.ndarray):
                vol_liq = vol_liq[0]

            self.vol_phase = vol_liq

        return time, states

    def retrieve_results(self, time: Sequence[float],
                         states: np.ndarray) -> None:
        """Store a solved trajectory using the active model's state layout.

        Parameters
        ----------
        time : sequence of float
            Integration times [s] with shape ``(n_time,)``.
        states : numpy.ndarray
            State trajectory with shape ``(n_time, n_states)``. For a liquid
            mixer the columns are mass fractions [-], holdup mass [kg], and
            temperature [K]. Crystallizer columns follow the delegated
            ``SemibatchCryst.states_di`` layout.

        Notes
        -----
        Crystallizer trajectories begin with distribution states, so their
        results and outputs are delegated without liquid-shaped slicing.
        """
        self.timeProf = np.array(time)  # [s]
        self.elapsed_time = time[-1]  # [s]

        if not self.is_cryst:
            dynamic_result = unpack_states(states, self.dim_states,
                                           self.name_states)
            self.wConcProf = dynamic_result['mass_frac']  # [-]
            self.massProf = dynamic_result['mass']  # [kg]
            self.tempProf = dynamic_result['temp']  # [K]

            self.Liquid_1.updatePhase(mass_frac=self.wConcProf[-1],
                                      mass=self.massProf[-1])

            self.Liquid_1.temp = self.tempProf[-1]

            self.Outlet = self.Liquid_1

            dynamic_result['time'] = np.asarray(time)

            self.result = DynamicResult(self.states_di, **dynamic_result)

            self.outputs = dynamic_result

        else:
            self.Outlet = self.CrystInst.Outlet
            self.outputs = self.CrystInst.outputs
            self.result = self.CrystInst.result

    def plot_profiles(self, fig_size=None, time_div=1, pick_comp=None,
                      kwargs=None):
        if kwargs is None:
            kwargs = {}

        if self.is_cryst:
            fig, axes, ax_right = self.CrystInst.plot_profiles(
                fig_size, time_div=time_div, **kwargs)
        else:
            fig, axes = self.plot_local(fig_size, time_div, pick_comp)

        return fig, axes

    def plot_local(self, fig_size=None, time_div=1, pick_comp=None):

        if pick_comp is None:
            pick_comp = range(self.wConcProf.shape[1])

        leg_comp = [self.name_species[ind] for ind in pick_comp]

        fig, axes = plt.subplots(2, 1, figsize=fig_size)

        # Mass fraction
        axes[0].plot(self.timeProf / time_div, self.wConcProf[:, pick_comp])

        axes[0].set_ylabel('mass frac')
        axes[0].legend(leg_comp)

        # Mass and temperature
        axes[1].plot(self.timeProf / time_div, self.massProf, 'k')
        axes[1].set_ylabel('mass (kg)')

        ax_temp = axes[1].twinx()
        ax_temp.plot(self.timeProf / time_div, self.tempProf, '--')
        ax_temp.set_ylabel('$T$ (K)')

        color = ax_temp.lines[0].get_color()
        ax_temp.spines['right'].set_color(color)
        ax_temp.tick_params(colors=color)
        ax_temp.yaxis.label.set_color(color)
        ax_temp.spines['top'].set_visible(False)

        for axis in axes:
            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)

            axis.xaxis.set_minor_locator(AutoMinorLocator(2))
            axis.yaxis.set_minor_locator(AutoMinorLocator(2))

        if time_div == 1:
            fig.text(0.5, 0, 'time (s)', ha='center')

        return fig, axes
