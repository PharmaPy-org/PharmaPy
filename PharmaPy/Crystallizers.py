# -*- coding: utf-8 -*-


from PharmaPy._assimulo import CVode, Explicit_Problem

from PharmaPy.Phases import classify_phases
from PharmaPy.Streams import LiquidStream, SolidStream
from PharmaPy.MixedPhases import Slurry, SlurryStream
from PharmaPy.Commons import (reorder_sens, plot_sens, trapezoidal_rule,
                              upwind_fvm, high_resolution_fvm,
                              eval_state_events, handle_events,
                              unpack_states, complete_dict_states,
                              flatten_states)

from PharmaPy.ProcessControl import analyze_controls

from PharmaPy.jac_module import numerical_jac, numerical_jac_central, dx_jac_x
from PharmaPy.Connections import get_inputs, get_inputs_new

from PharmaPy.Results import DynamicResult
from PharmaPy.Plotting import plot_function, plot_distrib

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import LightSource

from scipy.optimize import brentq, RootResults

import copy
import inspect
import string
import warnings
from typing import Sequence

import numpy as np

eps = np.finfo(float).eps
gas_ct = 8.314  # J/mol/K

# Design assumption: jacket volume as a fraction of Batch slurry volume or
# MSMPR/Semibatch vessel volume (working vol_tank / vol_offset). Introduced
# by the original author in e1e8164 without a cited source; issue #113 asks
# the author to confirm it. It affects only the jacket thermal transient:
# jacket volume cancels at jacket steady state.
DEFAULT_JACKET_VOLUME_RATIO = 0.14  # [-]


def _caller_stacklevel() -> int:
    """Return the ``warnings.warn`` stacklevel of the nearest external caller.

    A fixed ``stacklevel`` cannot be correct for a class hierarchy of varying
    depth: ``BatchCryst`` and ``MSMPR`` call ``_BaseCryst.__init__`` directly,
    while ``SemibatchCryst`` subclasses ``MSMPR`` and so sits one frame deeper.
    Counting frames instead keeps a warning raised inside this module pointed at
    the user code that constructed the unit operation.

    Returns
    -------
    int
        Stacklevel [-] identifying the first frame executing outside this
        module, counted from the caller of this helper. Falls back to ``1``
        when every frame belongs to this module.

    Notes
    -----
    Frames are matched on the module globals rather than on ``__file__`` so the
    comparison is unaffected by relative or absolute import paths.
    """
    frame = inspect.currentframe()
    if frame is None:  # pragma: no cover - no frame support on this runtime
        return 1

    module_globals = globals()
    level = 0
    frame = frame.f_back  # caller of this helper, i.e. stacklevel 1
    while frame is not None:
        level += 1
        if frame.f_globals is not module_globals:
            return level
        frame = frame.f_back

    return 1


class _BaseCryst:
    """Store shared configuration for crystallizer balance models.

    Subclasses provide batch, semibatch, and continuous material and energy
    balances. The base class retains the selected concentration basis and
    numerical discretization and configures sensitivity callbacks when a
    solver problem is built. Automatic-differentiation callbacks are not
    currently implemented, so requests for them use finite differences.
    """
    np = np
    # Support objects that bypass __init__.
    vol_ht: "float | None" = None  # [m**3], unspecified jacket inventory

    def __init__(self, mask_params,
                 method, target_comp, scale, vol_tank, controls,
                 adiabatic, rad_zero,
                 reset_states,
                 h_conv, vol_ht: "float | None", basis, jac_type,
                 state_events, param_wrapper) -> None:
        """Initialize crystallizer model and numerical configuration.

        Parameters
        ----------
        mask_params : sequence of bool or None
            Mask selecting active kinetic parameters [-].
        method : {'moments', '1D-FVM'}
            Crystal-size-distribution discretization method.
        target_comp : str or sequence of str
            Component name or names represented by the crystallizing solid.
        scale : float
            Crystal-size-distribution scaling factor [-].
        vol_tank : float or None
            Working vessel volume [m**3], or None to infer it at initialization.
            This stored volume is not expanded by a headspace factor.
        controls : dict or None
            Controlled-state callables. Each callable must return the units of
            its controlled state, such as temperature [K].
        adiabatic : bool
            If True, exclude utility heat transfer while retaining the vessel
            energy balance.
        rad_zero : float
            Nucleation size [um], on the same length basis as the CSD grid.
            Zero represents point nuclei.
        reset_states : bool
            If True, reset model states before a subsequent simulation.
        h_conv : float
            Vessel-side convective heat-transfer coefficient [W/m**2/K].
        vol_ht : float or None
            Finite, strictly positive cooling-jacket volume [m**3].
            If None, use the historical 0.14 design ratio: current slurry
            volume for Batch, vessel volume (working vol_tank / vol_offset)
            for MSMPR and Semibatch. See DEFAULT_JACKET_VOLUME_RATIO and #113.
        basis : {'mass_conc', 'mass_frac'}
            Composition basis, respectively [kg/m**3] or [kg/kg].
        jac_type : {'finite_diff', 'analytical', 'AD', None}
            Sensitivity Jacobian mode. ``'AD'`` currently warns and selects
            ``'finite_diff'`` because AD callbacks are unavailable.
        state_events : sequence of dict or None
            State-event specifications; event values use the units of their
            associated model states.
        param_wrapper : callable or None
            Transformation accepting a ``DynamicResult`` and a sensitivity
            dictionary whose arrays have shape ``(num_times, num_params)``.
            Result moments use SI lengths [m**n] or [m**n/m**3]; moment
            sensitivities retain solver lengths [um**n] or [um**n/m**3] per
            parameter unit. A callback differentiating reported SI moments
            must multiply order-n sensitivities by ``(1e-6)**n`` before
            returning transformed states and sensitivities on the same basis.

        Raises
        ------
        ValueError
            If vol_ht is neither None nor a finite, strictly positive real
            scalar volume [m**3].

        Warns
        -----
        RuntimeWarning
            If ``jac_type='AD'`` is requested; finite-difference sensitivity
            Jacobians are configured instead.
        """
        if vol_ht is not None and (
                not isinstance(vol_ht, (int, float, np.integer, np.floating))
                or not np.isfinite(vol_ht) or vol_ht <= 0):
            raise ValueError(
                "vol_ht must be None or a finite, strictly positive volume "
                "in m**3; use None for the default jacket-volume ratio.")

        if jac_type == 'AD':
            warnings.warn(
                "Automatic-differentiation Jacobian callbacks are not "
                "available; using finite-difference sensitivities with "
                "NumPy.",
                RuntimeWarning,
                stacklevel=_caller_stacklevel(),
            )
            jac_type = 'finite_diff'

        self.distributed_uo = False
        self.mask_params = mask_params
        self.basis = basis
        self.adiabatic = adiabatic

        # ---------- Building objects
        self._Phases = None
        self._Kinetics = None
        self._Utility = None
        self.material_from_upstream = False

        self.jac_type = jac_type

        if isinstance(target_comp, str):
            target_comp = [target_comp]

        self.target_comp = target_comp

        self.scale = scale
        self.scale_flag = True
        self.vol_tank = vol_tank

        # Controls
        if controls is None:
            self.controls = {}
        else:
            self.controls = analyze_controls(controls)

        self.method = method
        self.rad = rad_zero  # [um], same length basis as the CSD grid

        self.dx = None
        self.sensit = None

        # ---------- Create jacobians (autodiff)
        self.jac_states_vals = None
        # if method == 'moments':
        #     self.jac_states_fun = autojac(self.unit_model, 1)
        #     self.jac_params_fun = autojac(self.unit_model, 2)
        # elif method == 'fvm':
        #     self.jac_states_fun = make_jvp(self.fvm_method)

        #     # self.jac_params_fun = autojac(self.fvm_method, 1)
        #     self.jac_params_fun = None

        # Outlets
        self.reset_states = reset_states
        self.elapsed_time = 0

        self.profiles_runs = []

        self.__original_prof__ = {
            'tempProf': [], 'concProf': [], 'distribProf': [], 'timeProf': [],
            'elapsed_time': 0, 'scale_flag': True
        }

        # ---------- Names
        self.states_uo = ['mass_conc']
        self.names_states_in = ['mass_conc']

        # if not self.isothermal and 'temp' not in self.controls.keys():
        #     self.states_uo.append('temp')

        self.names_upstream = None
        self.bipartite = None

        # Other parameters
        self.h_conv = h_conv
        self.vol_ht = vol_ht  # [m**3], None selects the historical design ratio

        # Slurry phase
        self.Slurry = None

        # Parameters for optimization
        self.vol_mult = 1

        if state_events is None:
            state_events = []

        self.state_event_list = state_events

        self.param_wrapper = param_wrapper

        self.outputs = None

    @property
    def Phases(self):
        return self._Phases

    @Phases.setter
    def Phases(self, phases):
        if isinstance(phases, (list, tuple)):
            self._Phases = phases
        elif isinstance(phases, Slurry):
            self._Phases = phases.Phases
        elif phases.__module__ == 'PharmaPy.Phases':
            if self._Phases is None:
                self._Phases = [phases]
            else:
                self._Phases.append(phases)
        else:
            raise RuntimeError('Please provide a list or tuple of phases '
                               'objects')

        if isinstance(phases, Slurry):
            self.Slurry = phases
        elif isinstance(self._Phases, (list, tuple)):
            if len(self._Phases) > 1:

                # Mixed phase
                self.Slurry = Slurry()
                self.Slurry.Phases = self._Phases

        if self.Slurry is not None:
            self.__original_phase_dict__ = [
                copy.deepcopy(phase.__dict__) for phase in self.Slurry.Phases]

            self.vol_slurry = copy.copy(self.Slurry.vol)
            if isinstance(self.vol_slurry, np.ndarray):
                self.vol_phase = self.vol_slurry[0]
            else:
                self.vol_phase = self.vol_slurry

            classify_phases(self)  # Solid_1, Liquid_1...

            # Names and target compounds
            self.name_species = self.Liquid_1.name_species

            # Input defaults
            self.input_defaults = {
                'distrib': np.zeros_like(self.Solid_1.distrib)}

            name_bool = [name in self.target_comp for name in self.name_species]
            self.target_ind = np.where(name_bool)[0][0]

            # Save safe copy of original phases
            self.__original_phase__ = [copy.deepcopy(self.Liquid_1),
                                       copy.deepcopy(self.Solid_1)]

            self.__original_phase__ = copy.deepcopy(self.Slurry)

            self.kron_jtg = np.zeros_like(self.Liquid_1.mass_frac)
            self.kron_jtg[self.target_ind] = 1

            # ---------- Names
            # Moments
            if self.method == 'moments':
                name_mom = [r'\mu_{}'.format(ind) for ind
                            in range(self.Solid_1.num_mom)]
                name_mom.append('C')

                self.num_distr = len(self.Solid_1.moments)

            else:
                self.num_distr = len(self.Solid_1.distrib)

            # Species
            if self.name_species is None:
                num_sp = len(self.Liquid_1.mass_frac)
                self.name_species = list(string.ascii_uppercase[:num_sp])

            self.states_in_dict = {
                'Liquid_1': {'mass_conc': len(self.Liquid_1.name_species)},
                'Inlet': {'vol_flow': 1, 'temp': 1}}

            self.nomenclature()

    @property
    def Kinetics(self):
        return self._Kinetics

    @Kinetics.setter
    def Kinetics(self, instance):
        self._Kinetics = instance

        name_params = self._Kinetics.name_params
        if self.mask_params is None:
            self.mask_params = [True] * self._Kinetics.num_params
            self.name_params = name_params

        else:
            self.name_params = [name for ind, name in enumerate(name_params)
                                if self.mask_params[ind]]

        self.mask_params = np.array(self.mask_params)

    @property
    def Utility(self):
        return self._Utility

    @Utility.setter
    def Utility(self, utility):
        self.u_ht = 1 / (1 / self.h_conv + 1 / utility.h_conv)
        self._Utility = utility

    def nomenclature(self) -> None:
        """Declare reported-state metadata and derived result fields.

        Notes
        -----
        Public states_di describes SI reported/phase moments: [m**n] for
        Batch/Semibatch and [m**n/m**3] for MSMPR. Internally, solve_unit
        seeds the raw solver vector in micrometre lengths, and
        retrieve_results converts these moments back to SI for reporting.
        """
        name_class = self.__class__.__name__

        states_di = {
            }

        di_distr = {'dim': self.num_distr,
                    'index': list(range(self.num_distr)), 'type': 'diff',
                    'depends_on': ['time', 'x_cryst']}

        if name_class != 'BatchCryst':
            self.names_states_in += ['vol_flow', 'temp']

            if self.method == 'moments':
                self.states_in_dict['Inlet']['mu_n'] = self.num_distr
            else:
                self.states_in_dict['Inlet']['distrib'] = self.num_distr

        if self.method == 'moments':
            self.names_states_in.insert(0, 'mu_n')

            if name_class == 'MSMPR':
                self.states_uo.append('moments')

                di_distr['units'] = 'm**n/m**3'
                states_di['mu_n'] = di_distr
            else:
                self.states_uo.append('total_moments')

                di_distr['units'] = 'm**n'
                states_di['mu_n'] = di_distr

        elif self.method == '1D-FVM':
            self.names_states_in.insert(0, 'distrib')

            states_di['distrib'] = di_distr

            if name_class == 'MSMPR':
                self.states_uo.insert(0, 'distrib')
                di_distr['units'] = '#/m**3/um'

            else:
                self.states_uo.insert(0, 'total_distrib')
                di_distr['units'] = '#/um'

        states_di['mass_conc'] = {'dim': len(self.name_species),
                                  'index': self.name_species,
                                  'units': 'kg/m**3', 'type': 'diff',
                                  'depends_on': ['time']}

        if name_class != 'MSMPR':
            states_di['vol'] = {'dim': 1, 'units': 'm**3', 'type': 'diff',
                                'depends_on': ['time']}
            self.states_uo.append('vol')

        if self.adiabatic:
            self.states_uo.append('temp')

            states_di['temp'] = {'dim': 1, 'units': 'K', 'type': 'diff',
                                 'depends_on': ['time']}
        elif 'temp' not in self.controls:
            self.states_uo += ['temp', 'temp_ht']

            states_di['temp'] = {'dim': 1, 'units': 'K', 'type': 'diff',
                                 'depends_on': ['time']}
            states_di['temp_ht'] = {'dim': 1, 'units': 'K', 'type': 'diff',
                                    'depends_on': ['time']}

        self.states_in_phaseid = {'mass_conc': 'Liquid_1'}
        self.names_states_out = self.names_states_in

        self.states_di = states_di
        self.dim_states = [di['dim'] for di in self.states_di.values()]
        self.name_states = list(self.states_di.keys())

        self.fstates_di = {
            'supersat': {'dim': 1, 'units': 'kg/m**3'},
            'solubility': {'dim': 1, 'units': 'kg/m**3'}
            }

        if 'temp' in self.controls:
            self.fstates_di['temp'] = {'dim': 1, 'units': 'K'}

        if self.method != 'moments':
            self.fstates_di['mu_n'] = {'dim': 4, 'index': list(range(4)),
                                       'units': 'm**n'}

            self.fstates_di['vol_distrib'] = {
                'dim': self.num_distr,
                'index': list(range(self.num_distr)),
                'units': 'm**3/m**3'}

    def reset(self) -> None:
        """Restore the original phase and slurry inventories for a new run.

        Notes
        -----
        Keeps the attached phase objects and configured working tank volume.
        Restores liquid and solid amounts, compositions, and temperatures,
        then rebuilds the cached slurry population and volume from them.
        Moment-mode populations use total phase moments [m**n], including
        when a retained plotting distribution describes a different population.
        Clears reporting profiles and elapsed time [s].
        """
        copy_dict = copy.deepcopy(self.__original_prof__)
        self.__dict__.update(copy_dict)

        for phase, di in zip(self.Phases, self.__original_phase_dict__):
            phase.__dict__.update(copy.deepcopy(di))

        self._refresh_slurry_inventory()
        self.profiles_runs = []

    def _refresh_slurry_inventory(self) -> None:
        """Synchronize slurry caches after restoring or modifying phase amounts.

        Notes
        -----
        Rebuilds slurry volume [m**3] and volume-specific population from the
        attached phase inventories. Moment models use total solid moments
        [m**n], independently of any retained plotting distribution. Their
        third moment and shape factor determine solid volume, preserving the
        charged liquid volume when moments were modified without a mass update.
        FVM models use the total solid number distribution [#/um]. Phase object
        identities and the configured working tank volume are preserved.
        """
        if self.method == 'moments':
            solid_volume = self.Solid_1.kv * self.Solid_1.moments[3]  # [m**3]
            volume = self.Liquid_1.vol + solid_volume  # [m**3], current slurry
            moments = self.Solid_1.moments / volume  # [m**n/m**3], slurry basis
            self.Slurry = Slurry(vol=volume, moments=moments)
        else:
            self.Slurry = Slurry()
        self.Slurry.Phases = self.Phases
        self.vol_slurry = copy.copy(self.Slurry.vol)  # [m**3], current holdup
        self.vol_phase = (self.vol_slurry[0] if isinstance(self.vol_slurry, np.ndarray)
                          else self.vol_slurry)  # [m**3], scalar holdup convention

    def get_inputs(self, time: "float | np.ndarray") -> dict:
        """Read feed values on the crystallizer's phase and population bases.

        Parameters
        ----------
        time : float or ndarray
            Evaluation times [s], scalar or shape (num_times,).

        Returns
        -------
        dict
            Empty for Batch. Otherwise Inlet contains flow [m**3/s],
            temperature [K], and population [m**n/m**3] or [#/m**3/um];
            Liquid_1 contains species concentrations [kg/m**3]. Array times
            place time on the first axis for interpolated inlet values.

        Raises
        ------
        ValueError
            If the feed provides fewer moments than the downstream population
            model requires, with orders counted from zero on the final axis.

        Notes
        -----
        The dynamic solve_unit path supports a bare LiquidStream as a
        solid-free feed. Its composition belongs to the stream itself;
        missing population fields are zero. solve_steady_state requires an
        inlet with a Liquid_1 phase.
        For SlurryStream feeds, moments supplies the static SI mu_n fallback.
        If a connection omits mu_n from its converted fields, retain the
        upstream SI moment profile (also reported by FVM crystallizers) and
        interpolate it with the other inlet fields. Explicitly converted or
        dynamic mu_n values take precedence; the original stream is unchanged.
        Other inlet fields do not suppress the moment fallback. For moment-mode
        slurry feeds, dynamic concentration overrides belong to Liquid_1;
        omitted dynamic fields retain static moments, liquid concentration,
        flow and temperature without modifying the stream. Multiple evaluation
        times broadcast the fallback to (num_times, num_moments);
        a single evaluation time retains shape (num_moments,). The downstream
        model uses the lowest num_distr moment orders. Extra feed orders are
        omitted without changing the source stream; missing orders are rejected
        rather than reconstructed.
        """
        if self.__class__.__name__ == 'BatchCryst':
            inputs = {}
        elif isinstance(self.Inlet, LiquidStream):
            inlet_states = {**self.states_in_dict['Inlet'],
                            **self.states_in_dict['Liquid_1']}
            inputs = get_inputs_new(time, self.Inlet, {'Inlet': inlet_states})
            inputs['Liquid_1'] = {
                name: inputs['Inlet'].pop(name)
                for name in self.states_in_dict['Liquid_1']}
        elif (isinstance(self.Inlet, SlurryStream)
              and 'mu_n' in self.states_in_dict['Inlet']):
            inlet = copy.copy(self.Inlet)
            inlet.mu_n = self.Inlet.moments  # [m**n/m**3], slurry-volume fallback
            if isinstance(inlet.y_upstream, dict) and 'mu_n' in inlet.y_upstream:
                inlet.y_inlet = {'mu_n': inlet.y_upstream['mu_n'],
                                 **(inlet.y_inlet or {})}
                # [m**n/m**3], retain upstream history and converted-field precedence
            if inlet.DynamicInlet is not None:
                # DynamicInput supplies flat fields; resolve phase fallbacks
                # before restoring the crystallizer's grouped input contract.
                inlet.mass_conc = self.Inlet.Liquid_1.mass_conc  # [kg/m**3]
                inlet_states = {**self.states_in_dict['Inlet'],
                                **self.states_in_dict['Liquid_1']}
                inputs = get_inputs_new(time, inlet, {'Inlet': inlet_states})
                inputs['Liquid_1'] = {
                    name: inputs['Inlet'].pop(name)
                    for name in self.states_in_dict['Liquid_1']}
            else:
                inputs = get_inputs_new(time, inlet, self.states_in_dict)
        else:
            inputs = get_inputs_new(time, self.Inlet, self.states_in_dict)

        if self.method == 'moments' and inputs:
            moments = np.asarray(inputs['Inlet']['mu_n'])  # [m**n/m**3]
            supplied_count = moments.shape[-1] if moments.ndim else 0
            if supplied_count < self.num_distr:
                raise ValueError(
                    f"Inlet mu_n must provide at least {self.num_distr} moments "
                    f"in ascending order from zero; got {supplied_count}.")
            inputs['Inlet']['mu_n'] = moments[..., :self.num_distr]  # [m**n/m**3]

        return inputs

    def _set_active_params(self, params: "np.ndarray | None") -> None:
        """Install an active kinetic vector while retaining fixed parameters.

        Parameters
        ----------
        params : ndarray or None
            One-dimensional active parameter vector in mask_params order,
            with the native units and transformations of concat_params().
            None retains all stored parameters. Fractional values are preserved
            even when stored kinetics were initialized with integers.

        Raises
        ------
        ValueError
            If params is not a vector with one entry per True mask_params item.
        """
        if params is None:
            return
        active_params = np.asarray(params)  # [native active parameter units]
        expected_count = np.count_nonzero(self.mask_params)
        if active_params.shape != (expected_count,):
            raise ValueError(
                f"mask_params selects {expected_count} active parameter(s); "
                f"params must have shape ({expected_count},), "
                f"got {active_params.shape}.")
        merged_params = np.asarray(self.Kinetics.concat_params(), dtype=float)
        # [native parameter units], floating buffer preserves fractional iterates
        merged_params[self.mask_params] = active_params
        self.Kinetics.set_params(merged_params)

    def method_of_moments(self, mu: np.ndarray, conc: np.ndarray,
                          temp: float, params: "np.ndarray | None",
                          rho_cry: float, vol: float = 1) -> tuple:
        """Evaluate crystal moment derivatives and the physical mass source.

        Parameters
        ----------
        mu : ndarray
            Shape (num_moments,), total moments [um**n] for Batch/Semibatch
            or volume-specific moments [um**n/m**3] for MSMPR.
        conc : ndarray
            Liquid species mass concentrations [kg/m**3], shape (num_species,).
        temp : float
            Slurry temperature [K].
        params : ndarray or None
            Active kinetic parameters in mask_params order, using the native
            units and transformations of CrystKinetics.concat_params(). None
            retains the stored parameters. Fixed parameter values are retained.
        rho_cry : float
            Crystal density [kg/m**3].
        vol : float, optional
            Slurry volume [m**3] for total moments; use the default unit volume
            for volume-specific MSMPR moments.

        Returns
        -------
        dmu_dt : ndarray
            Moment derivatives [um**n/s] or [um**n/m**3/s], shape mu.shape.
        mass_transf : ndarray
            Crystal mass source [kg/s] or [kg/m**3/s], shape (1,).

        Raises
        ------
        ValueError
            If params does not have one entry per active mask_params item.

        Notes
        -----
        Supplied active parameters are installed on the kinetics object so its
        cached rates and parameter values agree for subsequent Jacobian calls.
        The nucleus size rad is in [um]. Length conversion uses 1 um = 1e-6 m.
        Nucleation contributes B*vol*rad**n to every total moment, including
        n=0. The crystal mass source is rho_cry*kv times the SI third-moment
        derivative. For MSMPR, the unit-volume convention gives rates per
        slurry volume instead.
        """
        kv = self.Solid_1.kv

        # Kinetics
        if self.basis == 'mass_frac':
            rho_liq = self.Liquid_1.getDensity()
            comp_kin = conc / rho_liq
        else:
            comp_kin = conc

        self._set_active_params(params)

        # Kinetic terms
        mu_susp = mu*(1e-6)**np.arange(self.num_distr) / vol  # m**n/m**3_susp
        nucl, growth, dissol = self.Kinetics.get_kinetics(comp_kin, temp, kv,
                                                          mu_susp)

        growth = growth * self.Kinetics.alpha_fn(conc)

        ind_mom = np.arange(1, len(mu))

        # Model
        dmu_zero_dt = np.atleast_1d(nucl * vol)
        dmu_1on_dt = ind_mom * (growth + dissol) * mu[:-1] + \
            nucl * vol * self.rad**ind_mom
        dmu_dt = np.concatenate((dmu_zero_dt, dmu_1on_dt))

        # Material balance in kg_API/s --> G in um, u_2 in um**2 (or m**2/m**3)
        mass_transf = np.atleast_1d(rho_cry * kv * (
            3*(growth + dissol)*mu[2] + nucl*vol*self.rad**3)) * (1e-6)**3

        return dmu_dt, mass_transf

    def fvm_method(self, csd: np.ndarray, moms: np.ndarray,
                   conc: np.ndarray, temp: float, params: "np.ndarray | None",
                   rho_cry: float, output: str = 'dstates',
                   vol: float = 1) -> "tuple | np.ndarray":
        """Evaluate FVM population fluxes and physical crystal mass generation.

        Parameters
        ----------
        csd : ndarray
            Numerically scaled CSD, shape (num_bins,). Before multiplication
            by scale [-], its basis is [#/um] for Batch/Semibatch or
            [#/m**3/um] for MSMPR. The size grid and dx are in [um].
        moms : ndarray
            Unscaled physical moments, shape (num_moments,), with order n in
            [m**n] for Batch/Semibatch or [m**n/m**3] for MSMPR.
        conc : ndarray
            Liquid species mass concentrations [kg/m**3], shape (num_species,).
        temp : float
            Slurry temperature [K].
        params : ndarray or None
            Active kinetic parameters in mask_params order, using the native
            units and transformations of CrystKinetics.concat_params(). None
            retains the stored parameters. Fixed parameter values are retained.
        rho_cry : float
            Crystal density [kg/m**3].
        output : {'flux', 'dstates'}, optional
            'flux' returns scaled population fluxes; 'dstates' (default)
            returns the CSD derivative and physical mass source.
        vol : float, optional
            Slurry volume [m**3] for total CSDs; use the default unit volume
            for volume-specific MSMPR distributions.

        Returns
        -------
        dcsd_dt, mass_transfer : tuple of ndarray
            For output='dstates', scaled derivatives with shape csd.shape
            in [#/um/s] or [#/m**3/um/s], and a scalar unscaled physical mass
            source [kg/s] or [kg/m**3/s].
        flux : ndarray
            For output='flux', scaled bin-face fluxes [#/s] or [#/m**3/s],
            shape (num_bins + 1,).

        Raises
        ------
        ValueError
            If params does not have one entry per active mask_params item,
            or output is neither 'flux' nor 'dstates'.

        Notes
        -----
        Supplied active parameters are installed on the kinetics object so its
        cached rates and parameter values agree for subsequent Jacobian calls.
        Physical crystal volume generation is 3*kv*(G+D)*mu_2 plus B*vol
        times the nucleus volume kv*rad**3. G and D [um/s] require a linear
        conversion to [m/s]; rad [um] requires a cubic conversion to [m**3].
        Numerical scale affects population fluxes only. Kinetics receive
        slurry-volume-specific SI moments, matching method_of_moments;
        total moments remain available for the physical mass source.
        """

        if output not in ('flux', 'dstates'):
            raise ValueError("output must be 'flux' or 'dstates'.")

        mu_2 = moms[2]

        kv_cry = self.Solid_1.kv

        # Kinetic terms
        if self.basis == 'mass_frac':
            rho_liq = self.Liquid_1.getDensity()
            comp_kin = conc / rho_liq
        else:
            comp_kin = conc

        self._set_active_params(params)

        moments_per_volume = moms / vol  # [m**n/m**3], slurry-volume basis
        nucl, growth, dissol = self.Kinetics.get_kinetics(comp_kin, temp,
                                                          kv_cry, moments_per_volume)

        nucleation_rate = nucl * vol  # [#/s] total, or [#/m**3/s] for MSMPR
        scaled_nucleation = nucleation_rate * self.scale  # [#/s] or [#/m**3/s]

        impurity_factor = self.Kinetics.alpha_fn(conc)
        growth = growth * impurity_factor  # um/s

        boundary_cond = scaled_nucleation / (growth + eps)  # [#/um] or [#/m**3/um]
        f_aug = np.concatenate(([boundary_cond]*2, csd, [csd[-1]]))

        # Flux source terms
        f_diff = np.diff(f_aug)

        if growth > 0:
            theta = f_diff[:-1] / (f_diff[1:] + eps*10)
        else:
            theta = f_diff[1:] / (f_diff[:-1] + eps*10)
        # Van-Leer limiter
        limiter = np.zeros_like(f_diff)
        limiter[:-1] = (np.abs(theta) + theta) / (1 + np.abs(theta))

        growth_term = growth * (f_aug[1:-1] + 0.5 * f_diff[1:] * limiter[:-1])
        dissol_term = dissol * (f_aug[2:] - 0.5 * f_diff[1:] * limiter[1:])

        flux = growth_term + dissol_term

        if output == 'flux':
            return flux
        else:
            dcsd_dt = -np.diff(flux) / self.dx

            # Exact conversion: 1 um = 1e-6 m. mu_2 already uses SI lengths.
            mass_transfer = rho_cry * kv_cry * (  # [kg/s] or [kg/m**3/s]
                3 * (growth + dissol) * mu_2 * 1e-6
                + nucleation_rate * (self.rad * 1e-6)**3)

            return dcsd_dt, np.array(mass_transfer)

    def unit_model(self, time: float, states: np.ndarray, params=None, sw=None,
                   mat_bce: bool = False,
                   enrgy_bce: bool = False) -> np.ndarray:
        """Evaluate balances in the declared raw solver state order.

        Parameters
        ----------
        time : float
            Evaluation time [s].
        states : numpy.ndarray
            Packed state vector, shape (num_states,). Population states use
            [um**n] or scaled [#/um] for Batch/Semibatch and the corresponding
            slurry-volume-specific basis for MSMPR. Remaining states are
            liquid composition on the configured basis ([kg/m**3] for
            mass_conc or [kg/kg] for mass_frac), liquid volume [m**3] when
            applicable, and tank then jacket temperatures [K] when integrated.
        params : array-like or None, optional
            Active kinetic parameters in the kinetics model's native units;
            None retains the configured values.
        sw : sequence of bool or None, optional
            Solver event switches, retained for the callback interface.
        mat_bce : bool, optional
            Return only material derivatives; takes precedence over enrgy_bce.
        enrgy_bce : bool, optional
            Return energy terms for duty retrieval instead of derivatives.

        Returns
        -------
        numpy.ndarray
            Flat derivatives in state units per second, or material-only
            derivatives. With enrgy_bce, return the unit's energy diagnostic
            unchanged. Batch/MSMPR diagnostics contain crystallization and
            utility terms [W], plus feed heat [W] for MSMPR; prescribed-
            temperature utility entries instead contain tank heat capacity
            [J/K]. The diagnostic option is supported by Batch and MSMPR.

        Notes
        -----
        Updates phase composition and temperature for property evaluation.
        Jacket derivatives are flattened individually so scalar and length-one
        thermal rates retain the tank-then-jacket state order.
        """
        di_states = unpack_states(states, self.dim_states, self.name_states)

        # Inputs
        u_input = self.get_inputs(time)

        di_states = complete_dict_states(time, di_states,
                                         ('temp', 'temp_ht', 'vol'),
                                         self.Slurry, self.controls)

        # ---------- Physical properties
        self.Liquid_1.updatePhase(mass_conc=di_states['mass_conc'])
        self.Liquid_1.temp = di_states['temp']
        self.Solid_1.temp = di_states['temp']

        rhos_susp = self.Slurry.getDensity(temp=di_states['temp'])

        name_unit = self.__class__.__name__

        if self.method == 'moments':
            di_states['distrib'] = di_states['mu_n']
            moms = di_states['mu_n'] * \
                (1e-6)**np.arange(self.states_di['mu_n']['dim'])

        else:
            moms = self.Solid_1.getMoments(
                distrib=di_states['distrib']/self.scale)  # m**n

        di_states['mu_n'] = moms

        if name_unit == 'BatchCryst':
            rhos = rhos_susp
            h_in = None
            phis_in = None
        elif name_unit == 'SemibatchCryst' or name_unit == 'MSMPR':
            inlet_temp = u_input['Inlet']['temp']

            if self.Inlet.__module__ == 'PharmaPy.MixedPhases':
                rhos_in = self.Inlet.getDensity(temp=di_states['temp'])

                if 'distrib' in u_input['Inlet']:

                    inlet_distr = u_input['Inlet']['distrib']

                    mom_in = self.Inlet.Solid_1.getMoments(distrib=inlet_distr,
                                                            mom_num=3)
                elif 'mu_n' in u_input['Inlet']:

                    mom_in = np.array([u_input['Inlet']['mu_n'][3]])


                # kv * mu_3 is the solid volume fraction of the inlet
                # slurry, so phi_in is the inlet liquid volume fraction
                phi_in = 1 - self.Inlet.Solid_1.kv * mom_in  # [-]
                # [-], [liquid, solid] inlet volume fractions
                phis_in = np.concatenate([phi_in, 1 - phi_in])

                h_in = self.Inlet.getEnthalpy(inlet_temp, phis_in, rhos_in)
            else:
                rho_liq_in = self.Inlet.getDensity(temp=inlet_temp)
                rho_sol_in = None

                rhos_in = np.array([rho_liq_in, rho_sol_in])
                h_in_mass = self.Inlet.getEnthalpy(temp=inlet_temp)  # [J/kg]
                h_in = h_in_mass * rho_liq_in  # [J/kg]*[kg/m**3] -> [J/m**3]

                phis_in = [1, 0]  # [-], [liq, sol]; solid-free liquid inlet

            rhos = [rhos_susp, rhos_in]

        # Balances
        material_bces, cryst_rate = self.material_balances(
            time, params, u_input, rhos, **di_states, phi_in=phis_in)

        if mat_bce:
            return material_bces
        elif enrgy_bce:
            energy_bce = self.energy_balances(
                time, params, cryst_rate, u_input, rhos, **di_states,
                h_in=h_in, heat_prof=True)

            return energy_bce

        else:

            if 'temp' in self.name_states:
                energy_bce = self.energy_balances(
                    time, params, cryst_rate, u_input, rhos, **di_states,
                    h_in=h_in)

                if isinstance(energy_bce, tuple):
                    energy_bce = np.hstack(energy_bce)  # [K/s], tank then jacket
                balances = np.append(material_bces, energy_bce)
            else:
                balances = material_bces

            self.derivatives = balances

            return balances

    def unit_jacobians(self, time, states, sens, params, fy, v_vector):
        if sens is not None:
            jac_states = self.jac_states_fun(time, states, params)
            jac_params = self.jac_params_fun(time, states, params)

            dsens_dt = np.dot(jac_states, sens) + jac_params

            if not isinstance(dsens_dt, np.ndarray):
                dsens_dt = dsens_dt._value

            return dsens_dt
        elif v_vector is not None:
            _, jac_v = self.jac_states_fun(time, states, params)(v_vector)

            return jac_v
        else:
            jac_states = self.jac_states_fun(time, states, params)

            if not isinstance(jac_states, np.ndarray):
                jac_states = jac_states._value

            return jac_states

    def jac_states_numerical(self, time, states, params, return_only=True):
        if return_only:
            return self.jac_states_vals
        else:
            def wrap_states(st): return self.unit_model(time, st, params)

            abstol = self.sundials_opt['atol']
            reltol = self.sundials_opt['rtol']
            jac_states = numerical_jac_central(wrap_states, states,
                                               dx=dx_jac_x,
                                               abs_tol=abstol, rel_tol=reltol)

            return jac_states

    def jac_params_numerical(self, time, states, params):
        def wrap_params(theta): return self.unit_model(time, states, theta)

        abstol = self.sundials_opt['atol']
        reltol = self.sundials_opt['rtol']
        p_bar = self.sundials_opt['pbar']

        dp = np.abs(p_bar) * np.sqrt(max(reltol, eps))

        jac_params = numerical_jac_central(wrap_params, params,
                                           dx=dp,
                                           abs_tol=abstol, rel_tol=reltol)

        return jac_params

        return jac_params

    def rhs_sensitivity(self, time, states, sens, params):
        jac_params_vals = self.jac_params_fn(time, states, params)

        jac_states_vals = self.jac_states_fn(time, states, params,
                                             return_only=False)

        rhs_sens = np.dot(jac_states_vals, sens) + jac_params_vals

        self.jac_states_vals = jac_states_vals

        return rhs_sens

    def set_ode_problem(self, eval_sens, states_init, params_mergd,
                        jacv_prod):
        """Build the explicit Assimulo problem and sensitivity callbacks.

        Parameters
        ----------
        eval_sens : bool
            If True, configure state and parameter sensitivity callbacks.
        states_init : numpy.ndarray
            Initial model state vector [state-dependent units].
        params_mergd : numpy.ndarray
            Active kinetic parameter vector [parameter-dependent units].
        jacv_prod : bool
            If True, configure the Jacobian-vector product for a finite-volume
            model without sensitivity evaluation.

        Returns
        -------
        assimulo.problem.Explicit_Problem
            ODE problem carrying the selected state and sensitivity callbacks.

        Raises
        ------
        NameError
            If ``jac_type`` is not ``'finite_diff'``, ``'analytical'``, or
            None after constructor normalization.
        """
        if eval_sens:
            problem = Explicit_Problem(self.unit_model, states_init,
                                       t0=self.elapsed_time,
                                       p0=params_mergd)

            if self.jac_type == 'finite_diff':
                self.jac_states_fn = self.jac_states_numerical
                self.jac_params_fn = self.jac_params_numerical

                problem.jac = self.jac_states_fn
                problem.rhs_sens = self.rhs_sensitivity

            elif self.jac_type == 'analytical':
                self.jac_states_fn = self.jac_states
                self.jac_params_fn = self.jac_params

                problem.jac = self.jac_states_fn
                problem.rhs_sens = self.rhs_sensitivity

            elif self.jac_type is None:
                pass
            else:
                raise NameError("Bad string value for the 'jac_type' argument")

        else:
            if self.state_event_list is None:
                def model(time, states, params=params_mergd):
                    return self.unit_model(time, states, params)

                problem = Explicit_Problem(model, states_init,
                                           t0=self.elapsed_time)
            else:
                sw0 = [True] * len(self.state_event_list)
                def model(time, states, sw=None):
                    return self.unit_model(time, states, params_mergd, sw)

                problem = Explicit_Problem(model, states_init,
                                           t0=self.elapsed_time, sw0=sw0)

            # ----- Jacobian callables
            if self.method == 'moments':
                # w.r.t. states
                # problem.jac = lambda time, states: \
                #     self.unit_jacobians(time, states, None, params_mergd,
                #                         None, None)

                pass

            elif self.method == 'fvm':
                # J*v product (AD, slower than the one used by SUNDIALS)
                if jacv_prod:
                    problem.jacv = lambda time, states, fy, v: \
                        self.unit_jacobians(time, states, None, params_mergd,
                                            fy, v)

        return problem

    def _eval_state_events(self, time, states, sw):
        events = eval_state_events(
            time, states, sw, self.len_states,
            self.states_uo, self.state_event_list, sdot=self.derivatives,
            discretized_model=False)

        return events

    def solve_unit(self, runtime=None, time_grid=None,
                   eval_sens: bool = False,
                   jac_v_prod: bool = False, verbose: bool = True, test=False,
                   sundials_opts=None, any_event: bool = True):
        """Initialize and integrate the crystallizer state.

        Parameters
        ----------
        runtime : float, optional
            Integration duration [s] from the elapsed time.
        time_grid : array-like, optional
            Output times [s], shape (num_times,). Its final value overrides
            runtime when both are supplied. One time specification is needed.
        eval_sens : bool, optional
            Integrate parameter sensitivities; default False.
        jac_v_prod : bool, optional
            Use the FVM Jacobian-vector product; default False.
        verbose : bool, optional
            Display solver statistics; default True.
        test : bool, optional
            Legacy unused option.
        sundials_opts : dict, optional
            CVode solver options; units follow the named solver option.
        any_event : bool, optional
            Stop on any state event when True (default), otherwise all events.

        Returns
        -------
        time : ndarray
            Integration times [s], shape (num_times,).
        states : ndarray
            State rows at each time in ``name_states`` order: total crystal
            moments [um**n] or unscaled distribution [#/um], liquid species mass
            concentrations [kg/m**3], liquid volume [m**3] for Batch/Semibatch,
            and optional phase/jacket temperatures [K]. Continuous units use
            volume-normalized crystal states instead. Returned moments retain
            the solver's micrometre basis; ``result.mu_n`` uses SI moments.
        sensit : list of ndarray, optional
            Parameter sensitivities [state unit / parameter unit], returned
            only when eval_sens is True.

        Notes
        -----
        The Batch/Semibatch volume state is initialized from the liquid phase;
        total slurry volume includes crystals and is reserved for geometry.
        Reset, when requested, precedes initial-state capture. vol_tank retains
        its assigned or inferred working volume [m**3] across initializations;
        geometry does not expand that volume by the headspace factor.
        Stored kinetic parameters are restored after integration, before heat
        retrieval, so sensitivity difference-quotient probes cannot persist.
        The raw solver vector carries micrometre moments internally, seeded
        from SI phase moments here. retrieve_results converts them back to SI;
        public states_di describes those reported values. Result retrieval
        updates the attached phases and stores the profiles. Moment solves do
        not require a size grid or slurry grid-spacing metadata.
        """

        if self.__class__.__name__ != 'BatchCryst' and self.method != 'moments':
            x_distr = getattr(self.Solid_1, 'x_distrib', [])
            self.states_in_dict['Inlet']['distrib'] = len(x_distr)

        self.Kinetics.target_idx = self.target_ind

        if self.reset_states:
            self.reset()

        # ---------- Solid phase states
        if 'vol' in self.states_uo:
            if self.method == 'moments':
                init_solid = self.Solid_1.moments * 1e6**np.arange(self.num_distr)
                # SI phase moments converted to [um**n]; exactly 1e6 um per metre

            elif self.method == '1D-FVM':
                x_grid = self.Solid_1.x_distrib
                init_solid = self.Solid_1.distrib * self.scale

        else:
            if self.method == 'moments':
                init_solid = self.Slurry.moments * 1e6**np.arange(self.num_distr)
                # SI slurry moments converted to [um**n/m**3]; exactly 1e6 um/m

            elif self.method == '1D-FVM':
                x_grid = self.Slurry.x_distrib
                init_solid = self.Slurry.distrib * self.scale

        self.dx = self.Slurry.dx if self.method == '1D-FVM' else None  # [um]
        self.x_grid = self.Slurry.x_distrib

        # ---------- Liquid phase states
        init_liquid = self.Liquid_1.mass_conc.copy()

        self.num_species = len(init_liquid)

        self.len_states = [self.num_distr, self.num_species]

        if 'vol' in self.states_uo:  # Batch or semibatch
            vol_init = self.Liquid_1.vol  # [m**3], ODE liquid-volume state
            init_susp = np.append(init_liquid, vol_init)

            self.len_states.append(1)
        else:
            init_susp = init_liquid

        # ---------- Read time
        if runtime is not None:
            final_time = runtime + self.elapsed_time

        if time_grid is not None:
            final_time = time_grid[-1]

        if self.scale_flag:
            self.scale_flag = False

        states_init = np.append(init_solid, init_susp)

        if self.vol_tank is None:
            if isinstance(self, SemibatchCryst):
                time_vec = np.linspace(self.elapsed_time, final_time)
                vol_flow = self.get_inputs(time_vec)['Inlet']['vol_flow']

                self.vol_tank = trapezoidal_rule(time_vec, vol_flow)

            else:
                self.vol_tank = self.Slurry.vol

        # Existing cylindrical geometry assumes working liquid height = diameter.
        self.diam_tank = (4/np.pi * self.vol_tank)**(1/3)  # [m]
        self.area_base = np.pi/4 * self.diam_tank**2  # [m**2]

        if 'temp_ht' in self.states_uo:

            if len(self.profiles_runs) == 0:
                temp_ht = self.Liquid_1.temp
            else:
                temp_ht = self.profiles_runs[-1]['temp_ht'][-1]

            states_init = np.concatenate(
                (states_init, [self.Liquid_1.temp, temp_ht]))

            self.len_states += [1, 1]
        elif 'temp' in self.states_uo:
            states_init = np.append(states_init, self.Liquid_1.temp)
            self.len_states += [1]

        merged_params = self.Kinetics.concat_params()[self.mask_params]

        # ---------- Create problem
        problem = self.set_ode_problem(eval_sens, states_init,
                                       merged_params, jac_v_prod)

        self.derivatives = problem.rhs(self.elapsed_time, states_init,
                                       merged_params)

        if len(self.state_event_list) > 0:
            def new_handle(solver, info):
                return handle_events(solver, info, self.state_event_list,
                                     any_event=any_event)

            problem.state_events = self._eval_state_events
            problem.handle_event = new_handle

        # ---------- Set solver
        # General
        solver = CVode(problem)
        solver.iter = 'Newton'
        solver.discr = 'BDF'

        if sundials_opts is not None:
            for name, val in sundials_opts.items():
                setattr(solver, name, val)

                if name == 'time_limit':
                    solver.report_continuously = True

        self.sundials_opt = solver.get_options()

        if eval_sens:
            solver.sensmethod = 'SIMULTANEOUS'
            solver.suppress_sens = False
            solver.report_continuously = True

        if self.method == '1D-FVM':
            solver.linear_solver = 'SPGMR'  # large, sparse systems

        if not verbose:
            solver.verbosity = 50

        # ---------- Solve model
        nominal_params = self.Kinetics.concat_params().copy()  # [native parameter units]
        try:
            time, states = solver.simulate(final_time, ncp_list=time_grid)
        finally:
            # CVODES sensitivity probes install perturbed values through the RHS.
            self.Kinetics.set_params(nominal_params)

        self.retrieve_results(time, states)

        # ---------- Organize sensitivity
        if eval_sens:
            sensit = []
            for elem in solver.p_sol:
                sens = np.array(elem)
                sens[0] = 0  # correct NaN's at t = 0 for sensitivities
                sensit.append(sens)

            self.sensit = sensit

            return time, states, sensit
        else:
            return time, states

    def paramest_wrapper(self, params, t_vals,
                         modify_phase=None, modify_controls=None,
                         scale_factor=1e-3, run_args={}, reord_sens=True):
        """Reset and evaluate a crystallizer for parameter estimation.

        Parameters
        ----------
        params : dict or array-like
            Kinetic parameters in the attached kinetics model's order and
            units, including any configured parameter transformations.
        t_vals : array-like
            Evaluation times [s], shape ``(num_times,)``.
        modify_phase : dict, optional
            Phase-update dictionaries keyed by ``Liquid`` and/or ``Solid``.
            Liquid fractions are dimensionless, concentrations use [mol/L] or
            [kg/m**3], and amounts use mass [kg], volume [m**3], or moles [mol].
            If the liquid modifier has no amount key, retain its reset charged
            volume [m**3]. Explicit amounts follow liquid phase precedence.
            Solid arguments retain the units of ``SolidPhase.updatePhase``.
        modify_controls : dict, optional
            Updates to each named control specification; units follow the
            controlled state.
        scale_factor : float, optional
            Legacy unused scaling argument [-]; defaults to 1e-3.
        run_args : dict, optional
            Additional keyword arguments passed to ``solve_unit``.
        reord_sens : bool, optional
            Stack time-by-parameter sensitivities by state when True; otherwise
            retain the parameter-by-time-by-state layout. Passed to a configured
            ``param_wrapper`` as well. Defaults to True.

        Returns
        -------
        result : numpy.ndarray, tuple, or object
            With no custom wrapper, return the time-by-state array, plus
            sensitivities for the moments method. State units and ordering
            follow ``solve_unit``; sensitivity units are state units per kinetic
            parameter unit. With a custom moments wrapper, return its result.

        Raises
        ------
        ValueError
            If a nonempty phase modifier names neither ``Liquid`` nor ``Solid``.

        Notes
        -----
        Composition-only liquid modifiers preserve the reset charged volume
        used by geometry, residence time, and initial states, rather than
        conserving liquid mass. Slurry volume and population caches are
        refreshed from the modified phase inventories before initialization.
        Supplied modifier dictionaries are not changed.
        This wrapper owns the estimation reset: solve_unit's reset_states flag
        is temporarily disabled so it preserves these modifiers, then restored
        even if the solve raises an exception.
        A custom moments callback receives SI ``DynamicResult`` moments
        [m**n] or [m**n/m**3], while its sensitivity dictionary retains raw
        solver moments [um**n] or [um**n/m**3] per parameter unit. The callback
        owns any conversion: multiply order-n sensitivities by ``(1e-6)**n``
        when differentiating the reported SI moments. Runtime callback values
        retain this established convention.
        """
        self.reset()

        self.Kinetics.set_params(params)

        self.elapsed_time = 0

        if isinstance(modify_phase, dict) and len(modify_phase) > 0:

            if 'Liquid' not in modify_phase and 'Solid' not in modify_phase:
                raise ValueError(
                    "Phase modifier must specify the targeted phase, i.e. "
                    "must have an additional layer with keys 'Liquid_1' "
                    "and/or 'Solid_1'")

            liquid_mod = modify_phase.get('Liquid', {})
            solid_mod = modify_phase.get('Solid', {})

            if not any(key in liquid_mod for key in ('mass', 'vol', 'moles')):
                liquid_mod = {'vol': self.Liquid_1.vol, **liquid_mod}  # vol [m**3]
            self.Liquid_1.updatePhase(**liquid_mod)
            self.Solid_1.updatePhase(**solid_mod)
            self._refresh_slurry_inventory()

        if isinstance(modify_controls, dict):
            for key, val in modify_controls.items():
                self.controls[key].update(val)

        reset_states = self.reset_states
        self.reset_states = False
        try:
            if self.param_wrapper is None:
                if self.method == 'moments':
                    t_prof, states, sens = self.solve_unit(time_grid=t_vals,
                                                           eval_sens=True,
                                                           verbose=False,
                                                           **run_args)

                    if reord_sens:
                        sens = reorder_sens(sens, separate_sens=False)
                    else:
                        sens = np.stack(sens)

                    result = (states, sens)
                else:
                    t_prof, states_out = self.solve_unit(time_grid=t_vals,
                                                         eval_sens=False,
                                                         verbose=False,
                                                         **run_args)

                    result = states_out

            elif callable(self.param_wrapper):
                if self.method == 'moments':
                    t_prof, states, sens = self.solve_unit(time_grid=t_vals,
                                                           eval_sens=True,
                                                           verbose=False,
                                                           **run_args)

                    # Group parameter sensitivities by state.
                    sens_sep = reorder_sens(sens, separate_sens=True)

                    di_keys = ['mu_%s' % ind for ind in range(self.num_distr)]
                    di_keys += ['w_%s' % name for name in self.name_species]
                    di_keys.append('vol')

                    sens_sep = dict(zip(di_keys, sens_sep))

                    result = self.param_wrapper(self.result, sens_sep,
                                                reord_sens=reord_sens)

        finally:
            self.reset_states = reset_states

        return result

    def _store_heat_duty(self, time: np.ndarray, heat_terms: np.ndarray) -> None:
        """Integrate utility heat for the latest Batch or MSMPR solve segment.

        Parameters
        ----------
        time : ndarray
            Increasing reporting times [s], shape (num_times,).
        heat_terms : ndarray
            Shape (num_times, 2) for Batch or (num_times, 3) for MSMPR.
            Column 0 is signed crystallization source [W], negative for heat
            release. Column 1 is heat removed to the jacket [W], except with
            prescribed temperature and nonadiabatic operation, when it is
            the total tank heat capacitance [J/K]. MSMPR column 2 is net
            advective heat entering the tank [W]; Batch has no flow term.

        Notes
        -----
        Retains these components in heat_prof with the units above. Stores
        heat_duty = [0, integrated utility heat] [J], positive for heat
        removed and negative for heat supplied, with legacy duty_type [0, -2].
        Both heat_prof and heat_duty cover only the latest solve segment;
        each call replaces the previous segment's values, even when stored
        result profiles include earlier segments.
        Q fills the cooling column of SimExec.GetDuties; positive Q means
        heat removed to the utility, opposite to the reactor heating column.
        For prescribed temperature, C*dT/dt = flow - source - Q gives
        Q = flow - source - C*dT/dt. Use the temperature slope on each
        reporting interval and trapezoidal means of C and the heat rates.
        Thus every interval contributes, even for a two-point trajectory;
        capacitance itself is never integrated as a heat rate. Adiabatic
        operation takes precedence over a prescribed-temperature control.
        """
        self.heat_prof = heat_terms  # [W] or [J/K], column contract above
        if 'temp' in self.controls and not self.adiabatic:
            control = self.controls['temp']
            temperatures = np.asarray([
                control['fun'](point, *control['args'], **control['kwargs'])
                for point in time])  # [K]
            intervals = np.diff(time)  # [s]
            temperature_rate = np.diff(temperatures) / intervals  # [K/s]
            mean_terms = (heat_terms[1:] + heat_terms[:-1]) / 2  # column units above
            utility_rate = (-mean_terms[:, 0]
                            - mean_terms[:, 1] * temperature_rate)  # [W]
            if heat_terms.shape[1] == 3:
                utility_rate += mean_terms[:, 2]  # [W], net advective heat
            utility_energy = np.dot(intervals, utility_rate)  # [J]
        else:
            utility_energy = trapezoidal_rule(time, heat_terms[:, 1])  # [J]
        self.heat_duty = np.array([0, utility_energy])  # [J]
        self.duty_type = [0, -2]

    def flatten_states(self):
        out = flatten_states(self.profiles_runs)

        return out

    def plot_profiles(self, **fig_kwargs):
        """

        Parameters
        ----------
        fig_kwargs : keyword arguments
            keyword arguments to be passed to the plot.subplots() method

        Returns
        -------
        fig : TYPE
            DESCRIPTION.
        ax : TYPE
            DESCRIPTION.

        """

        def get_mu_labels(mu_idx, msmpr=False):
            out = []
            for idx in mu_idx:
                name = '$\mu_{%i}$' % idx

                if idx == 0:
                    unit = '#'
                elif idx == 1:
                    unit = 'm'
                else:
                    unit = '$\mathrm{m^{%i}}$' % idx

                if msmpr:
                    unit += ' $\mathrm{m^{-3}}$'

                unit = r' (%s)' % unit

                out.append(name + unit)

            return out

        states = [('mu_n', (0, )), 'temp', ('mass_conc', (self.target_ind,)),
                  'supersat']

        figmap = [0, 4, 5, 5]
        ylabels = ['mu_0', 'T', 'C_j', 'sigma']

        if hasattr(self.result, 'temp_ht'):
            states.append('temp_ht')
            figmap.append(4)
            ylabels.append('T_{ht}')

        fig, ax = plot_function(self, states, fig_map=figmap,
                                nrows=3, ncols=2, ylabels=ylabels,
                                **fig_kwargs)

        ax[0, 0].legend().remove()

        time = self.result.time
        moms = self.result.mu_n

        is_msmpr = self.__class__.__name__ == 'MSMPR'
        labels_moms = get_mu_labels(range(moms.shape[1]), msmpr=is_msmpr)

        for ind, row in enumerate(moms[:, 1:].T):
            ax.flatten()[ind + 1].plot(time, row)

        for ind, lab in enumerate(labels_moms):
            ax.flatten()[ind].set_ylabel(lab)

        # Solubility
        ax[2, 1].plot(time, self.result.solubility)
        ax[2, 1].lines[1].set_color('k')
        ax[2, 1].lines[1].set_alpha(0.4)

        ax[2, 1].legend([self.target_comp[0], 'solubility'])

        fig.tight_layout()
        return fig, ax

    def plot_csd(self, times=(0,), logy=False, vol_based=False, **fig_kw):

        if vol_based:
            state_plot = ['vol_distrib']
            y_lab = ('f_v', )
        else:
            state_plot = ['distrib']
            y_lab = ('f', )

        fig, axis = plot_distrib(self, state_plot, times=times,
                                 x_name='x_cryst', ylabels=y_lab, legend=False,
                                 **fig_kw)

        # axis.set_xlabel('$x$ ($\mathregular{\mu m}$)')
        axis.set_xscale('log')

        fig.texts[0].remove()
        axis.set_xlabel('$x$ ($\mathregular{\mu m}$)')

        return fig, axis

    def plot_csd_heatmap(self, vol_based=False, **fig_kw):
        self.flatten_states()

        if self.method != '1D-FVM':
            raise RuntimeError('No 3D data to show. Run crystallizer with the '
                               'FVM method')

        res = self.result
        x_mesh, t_mesh = np.meshgrid(res.x_cryst, res.time)

        fig, ax = plt.subplots(**fig_kw)

        if vol_based:
            distrib = res.vol_distrib.T
        else:
            distrib = res.distrib.T

        cf = ax.contourf(x_mesh.T, t_mesh.T, distrib, cmap=cm.Blues,
                         levels=150)

        cbar = fig.colorbar(cf)

        if self.scale == 1:
            cbar.ax.set_ylabel(r'$f$ $\left( \frac{\#}{m^3 \mu m} \right)$')
        else:
            exp = int(np.log10(self.scale))
            cbar.ax.set_ylabel(
                r'$f$ $\times 10^{%i}$ $\left( \frac{\#}{m^3 \mu m} \right)$ ' % exp)

        # Edit
        ax.set_xlabel(r'size ($\mu m$)')
        ax.set_ylabel('time (s)')
        ax.invert_yaxis()

        ax.set_xscale('log')

        return fig, ax

    def plot_sens(self, mode='per_parameter'):
        if type(self.timeProf) is list:
            self.flatten_states()

        if self.sensit is None:
            raise AttributeError("No sensitivities detected. Run the unit "
                                 " with 'eval_sens'=True")

        if mode == 'per_parameter':
            sens_data = self.sensit
        elif mode == 'per_state':
            sens_data = reorder_sens(self.sensit, separate_sens=True)

        # Name states
        name_mom = ['\mu_%i' % i for i in range(self.num_distr)]
        name_conc = ["C_{" + self.name_species[ind] + "}"
                     for ind in range(len(self.Liquid_1.name_species))]

        name_others = []
        if 'vol' in self.states_uo:
            name_others.append('vol')

        if 'temp' in self.states_uo:
            name_others.append('temp')

        name_states = name_mom + name_conc + name_others
        name_params = [name for ind, name in
                       enumerate(self.Kinetics.name_params)
                       if self.mask_params[ind]]

        fig, axis = plot_sens(self.result.time, sens_data,
                              name_states=name_states,
                              name_params=name_params,
                              mode=mode)

        return fig, axis

    def animate_cryst(self, filename=None, fps=5, step_data=1):
        from matplotlib.animation import FuncAnimation
        from matplotlib.animation import FFMpegWriter

        if type(self.timeProf) is list:
            self.flatten_states()

        if filename is None:
            filename = 'anim'

        fig_anim, ax_anim = plt.subplots(figsize=(5, 3.125))

        # ax_anim.set_xlim(0, self.x_grid.max())
        ax_anim.set_xlabel(r'crystal size ($\mu m$)')

        ax_anim.set_ylabel('counts')

        def func_data(ind):
            dist = self.distribProf[ind]
            return dist

        line, = ax_anim.plot(self.x_grid, func_data(0), '-o', mfc='None',
                             ms='2')
        time_tag = ax_anim.text(
            1, 1.04, '$time = {:.1f}$ s'.format(self.timeProf[0]),
            horizontalalignment='right',
            transform=ax_anim.transAxes)

        def func_anim(ind):
            f_vals = func_data(ind)
            line.set_ydata(f_vals)
            plt.gca().set_xscale("log")

            if f_vals.max() > f_vals.min():
                ax_anim.set_ylim(f_vals.min()*1.15, f_vals.max()*1.15)

            time_tag.set_text('$time = {:.1f}$ s'.format(self.timeProf[ind]))

            fig_anim.tight_layout()

        frames = np.arange(0, len(self.timeProf), step_data)
        animation = FuncAnimation(fig_anim, func_anim, frames=frames,
                                  repeat=True)

        writer = FFMpegWriter(fps=fps, metadata=dict(artist='Me'),
                              bitrate=1800)

        animation.save(filename + '.mp4', writer=writer)

        return animation, fig_anim, ax_anim


class BatchCryst(_BaseCryst):
    """Construct a Batch Crystallizer object
    
    Parameters
    ----------
    target_comp : str, list of strings
        Name of the crystallizing compound(s) from .json file.
    mask_params : list of bool (optional, default = None)
        Binary list of which parameters to exclude from the kinetics
        computation
    method : str
        Choice of the numerical method. Options are: 'moments', '1D-FVM'
    scale : float
        Scaling factor by which crystal size distribution will be
        multiplied.
    controls : dict of dicts (funcs) (optional, default = None)
        Dictionary with keys representing the state (e.g.'Temp')
        which is controlled and the value indicating the function
        to use while computing the varible. Functions are of the form
        f(time) = state_value
    cfun_solub: callable
        User defined function for the solubility function :
        func(conc)
    adiabatic : bool (optional, default =True)
        Boolean value indicating whether the heat transfer of
        the crystallization is considered.
    rad_zero : float (optional)
        Nucleation size [um], matching the CSD grid; zero gives point nuclei.
    reset_states : bool (optional, default = False)
        Boolean value indicating whether the states should be
        reset before simulation
    basis : str (optional, default = 'mass_conc')
        Options : 'massfrac', 'massconc'
    state_events : lsit of dict(s)
        list of dictionaries, each one containing the specification of a
        state event
    """

    def __init__(self, target_comp, mask_params=None,
                 method='1D-FVM', scale=1, vol_tank=None,
                 controls=None, adiabatic=False,
                 rad_zero=0, reset_states=False,
                 h_conv=1000, vol_ht=None, basis='mass_conc',
                 jac_type=None, state_events=None, param_wrapper=None):


        super().__init__(mask_params, method, target_comp, scale, vol_tank,
                         controls, adiabatic,
                         rad_zero, reset_states, h_conv, vol_ht,
                         basis, jac_type, state_events, param_wrapper)

        self.is_continuous = False
        self.oper_mode = 'Batch'

        self.vol_offset = 0.75

    def jac_states(self, time: float, states: np.ndarray,
                   params: "np.ndarray | None", return_only: bool = True
                   ) -> np.ndarray:
        """Return the BatchCryst state Jacobian.

        Parameters
        ----------
        time : float
            Evaluation time [s].
        states : ndarray
            Solver state vector ordered as total moments [um**n], liquid mass
            concentrations [kg/m**3], and liquid volume [m**3].
        params : ndarray or None
            Kinetic parameter vector; units depend on the kinetic model.
        return_only : bool, optional
            If True, return the cached Jacobian from the solver interface.

        Returns
        -------
        ndarray
            State Jacobian. The concentration-concentration block has units
            [1/s].

        Notes
        -----
        Crystal growth rates are stored in [um/s], so the explicit
        ``(1e-6)**3`` factors convert crystal-volume terms to [m**3].
        The nucleation row differentiates V*(B_prim + B_sec), including
        B_sec proportional to (kv*mu_j*1e-6**j/V)**s_2, where V is slurry
        volume and j is the configured secondary-nucleation moment order.
        Rates must have been cached by unit_model at the same state first.
        This analytical path assumes built-in kinetics, zero nucleation
        radius, alpha_fn = 1, mass-concentration inputs, and prescribed
        temperature. Phase densities and concentration-dependent solubility
        are held fixed. Growth and dissolution are mutually exclusive, so
        the active rate equals G + D; its exponent controls the concentration
        derivatives in each regime.
        """

        if return_only:
            return self.jac_states_vals
        else:
            # Name states
            vol_liq = states[-1]

            num_material = self.num_distr + self.num_species
            w_conc = states[self.num_distr:num_material]

            control = self.controls['temp']
            temp = control['fun'](time, *control['args'], **control['kwargs'])

            num_states = len(states)
            conc_tg = w_conc[self.target_ind]
            c_sat = self.Kinetics.get_solubility(temp, w_conc)

            moms = states[:self.num_distr]
            idx_moms = np.arange(1, len(moms))

            rho_l = self.Liquid_1.getDensity(temp=temp)
            rho_c = self.Solid_1.getDensity(temp=temp)
            kv = self.Solid_1.kv

            # Kinetics
            b_pr = self.Kinetics.prim_nucl
            b_sec = self.Kinetics.sec_nucl

            nucl = b_pr + b_sec
            # Built-in kinetics make the inactive rate zero: selecting the
            # active branch therefore gives the signed sum G + D.
            if conc_tg < c_sat:
                size_rate = self.Kinetics.dissol  # [um/s], negative dissolution
                size_exponent = self.Kinetics.params['dissolution'][-1]  # [-]
            else:
                size_rate = self.Kinetics.growth  # [um/s], nonnegative growth
                size_exponent = self.Kinetics.params['growth'][-1]  # [-]
            bp_exp = self.Kinetics.params['nucl_prim'][-1]
            k_s, _, bs_exp, bs2_exp = self.Kinetics.params['nucl_sec']

            jacobian = np.zeros((num_states, num_states))

            # ----- Moments columns
            wrt_mu = idx_moms * size_rate  # [um/s]

            rng = np.arange(len(wrt_mu))
            jacobian[rng + 1, rng] = wrt_mu

            # V*B depends on V directly and through B_sec ~ V**(-s_2).
            vol_slurry = vol_liq + kv * moms[3] * (1e-6)**3  # [m**3]
            dnucl_dvol = nucl - bs2_exp * b_sec  # [#/m**3/s]
            jacobian[0, 3] = dnucl_dvol * kv * (1e-6)**3  # [#/s/um**3]
            if b_sec != 0 and bs2_exp != 0:
                order = self.Kinetics.mu_sec_nucl
                jacobian[0, order] += (vol_slurry * bs2_exp * b_sec
                                       / moms[order])  # [#/s/um**order]

            # Second moment column (concentration eqns)
            dtr_mu2 = 3 * kv * size_rate * rho_c * \
                (1e-6)**3  # [kg/s/um**2], cubic crystal-volume conversion

            dfconc_dmu2 = -1/vol_liq * dtr_mu2 * (self.kron_jtg - w_conc/rho_l)
            jacobian[self.num_distr:self.num_distr + dfconc_dmu2.shape[0],
                     2] = dfconc_dmu2

            # Volume eqn
            jacobian[-1, 2] = -dtr_mu2 / rho_l  # dfvol/dmu2

            # ----- Concentration columns
            # Moment eqns
            conc_diff = conc_tg - c_sat

            dfmu0_dconc = (bp_exp * b_pr + bs_exp * b_sec) * \
                vol_slurry / conc_diff
            dfmun_dconc = (idx_moms * moms[:-1] * size_exponent
                             / conc_diff * size_rate)  # [um**n*m**3/kg/s]

            jacobian[0, self.num_distr +
                     self.target_ind] = dfmu0_dconc

            jacobian[
                1:1 + dfmun_dconc.shape[0],
                self.num_distr + self.target_ind] = dfmun_dconc

            # Concentration eqns
            # tr is the crystal mass transfer rate [kg/s].
            tr = 3 * kv * size_rate * moms[2] * rho_c * (1e-6)**3

            # dtr_dconc_tg is d(tr)/d(c_target) [m**3/s].
            dtr_dconc_tg = size_exponent * tr / conc_diff

            # first_conc is [-]; second_conc is [m**3/s].
            first_conc = np.outer(self.kron_jtg - w_conc/rho_l, self.kron_jtg)
            second_conc = -tr/rho_l * np.eye(len(w_conc))

            # d(dmass_conc_dt)/d(mass_conc) [1/s].
            dfconc_dconc = -1/vol_liq * \
                (dtr_dconc_tg * first_conc + second_conc)

            jacobian[self.num_distr:-1, self.num_distr:-1] = dfconc_dconc

            # Volume eqn
            jacobian[-1, self.num_distr + self.target_ind] = - \
                dtr_dconc_tg / rho_l

            # ----- Volume column
            # mu_zero eqn
            jacobian[0, -1] = dnucl_dvol  # [#/m**3/s]

            # Concentration eqn
            dfconc_dvol = 1/vol_liq**2 * (self.kron_jtg*tr - w_conc/rho_l * tr)
            jacobian[self.num_distr:self.num_distr + dfconc_dvol.shape[0],
                     -1] = dfconc_dvol

            return jacobian

    def jac_params(self, time: float, states: np.ndarray,
                   params: np.ndarray) -> np.ndarray:
        """Return active kinetic parameter partials of the batch moment RHS.

        Parameters
        ----------
        time : float
            Evaluation time [s].
        states : ndarray
            Total moments [um**n], species concentrations [kg/m**3], then
            liquid volume [m**3], shape (num_states,).
        params : ndarray
            Native kinetic parameters, with units defined by CrystKinetics.
            Cached kinetics must already correspond to these parameters.

        Returns
        -------
        ndarray
            Shape (num_states, num_active_params), with units
            [state unit/s/parameter unit]. Columns follow concat_params and
            mask_params, including all three dissolution columns.

        Notes
        -----
        Call unit_model at the same state first to cache rates. This analytical
        path assumes built-in kinetics, zero nucleation radius, alpha_fn = 1,
        mass-concentration inputs, and prescribed temperature. Secondary
        nucleation uses slurry-normalized SI moments, including its selected
        area or volume moment. The logarithm differentiates the numerical
        moment factor in the configured units of the kinetic prefactor.
        At a nonpositive moment factor, the s_2 partial is set to zero by
        convention. In particular, its derivative is undefined at zero
        moment and s_2 = 0 even though the rate uses 0**0 = 1.
        """
        control = self.controls['temp']
        temp = control['fun'](time, *control['args'], **control['kwargs'])  # [K]
        vol_liq = states[-1]  # [m**3]
        moms = states[:self.num_distr]  # [um**n], total moments
        num_material = self.num_distr + self.num_species
        w_conc = states[self.num_distr:num_material]  # [kg/m**3]
        conc_tg = w_conc[self.target_ind]  # [kg/m**3]
        kv = self.Solid_1.kv  # [-]
        rho_c = self.Solid_1.getDensity(temp=temp)  # [kg/m**3]
        rho_l = self.Liquid_1.getDensity(temp=temp)  # [kg/m**3]
        vol_slurry = vol_liq + kv * moms[3] * 1e-18  # [m**3]
        order = self.Kinetics.mu_sec_nucl
        moment_factor = kv * moms[order] * 1e-6**order / vol_slurry  # [m**order/m**3]
        b_sec = self.Kinetics.sec_nucl  # [#/m**3/s]
        # [rate/parameter]; rates are nucleation [#/m**3/s] or size [um/s].
        dbp, dbs, dg, dd, _ = self.Kinetics.deriv_cryst(conc_tg, w_conc, temp)
        # The zero-moment boundary uses the convention documented above.
        dbs_ds2 = (b_sec * np.log(moment_factor)
                   if moment_factor > 0 else 0)  # [#/m**3/s]
        dbs = np.append(dbs, dbs_ds2)  # [#/m**3/s/parameter]
        size_partials = np.concatenate((dg, dd))  # [um/s/parameter]
        num_nucl = len(dbp) + len(dbs)
        jacobian = np.zeros((len(states), num_nucl + len(size_partials)))
        jacobian[0, :num_nucl] = vol_slurry * np.concatenate((dbp, dbs))
        idx_moms = np.arange(1, self.num_distr)
        jacobian[1:self.num_distr, num_nucl:] = np.outer(
            idx_moms * moms[:-1], size_partials)
        # Cubic conversion: growth [um/s] times total mu_2 [um**2] to [m**3/s].
        transfer_partials = (3 * kv * rho_c * moms[2] * size_partials
                             * 1e-18)  # [kg/s/parameter]
        jacobian[self.num_distr:num_material, num_nucl:] = -np.outer(
            self.kron_jtg - w_conc/rho_l, transfer_partials) / vol_liq
        jacobian[-1, num_nucl:] = -transfer_partials / rho_l
        return jacobian[:, self.mask_params]

    def material_balances(self, time, params, u_inputs, rhos, mu_n,
                          distrib, mass_conc, temp, temp_ht, vol, phi_in=None):
        """
        Material balances for the batch crystallizer.

        Parameters
        ----------
        time : float
            integration time [s].
        params : array-like
            kinetic parameters passed to ``self.Kinetics``.
        u_inputs : dict
            unit inputs at `time`. Unused by the batch unit, kept for
            signature compatibility with the flow-through units.
        rhos : list of float
            [liquid, solid] densities [kg/m**3].
        mu_n : array-like
            crystal size distribution moments [m**n] (total basis).
        distrib : array-like
            distribution state: [#/um] for ``method == '1D-FVM'``, or the
            raw moment vector [um**n] for ``method == 'moments'``.
        mass_conc : array-like
            liquid-phase mass concentrations [kg/m**3].
        temp : float
            liquid temperature [K].
        temp_ht : float or None
            jacket temperature [K]. Unused here, kept for signature
            compatibility.
        vol : float
            liquid volume [m**3].
        phi_in : None
            unused, kept for signature compatibility.

        Returns
        -------
        dmaterial_dt : numpy.ndarray
            stacked derivatives with mixed units, in this order:
            ``ddistr_dt`` ([#/um/s] for '1D-FVM', [um**n/s] for 'moments'),
            ``dcomp_dt`` [kg/m**3/s] and ``dvol_liq`` [m**3/s].
        transf : numpy.ndarray
            crystallization mass rate [kg/s].

        Notes
        -----
        Batch states are declared on a *total* basis in
        :meth:`_BaseCryst.solve_unit` (``mu_n`` in [um**n], ``distrib`` in
        [#/um]), and the kinetics are evaluated with ``vol=vol_slurry``, so
        ``transf`` is a total rate [kg/s] rather than the volumetric
        [kg/m**3/s] rate returned by :meth:`MSMPR.material_balances`.

        ``dcomp_dt`` is documented as [kg/m**3/s] because the
        ``basis == 'mass_frac'`` rescaling below acts on a copy and never
        reaches the returned array (tracked in issue #47).
        """

        # 'vol' represents liquid volume

        rho_liq, rho_s = rhos

        vol_solid = mu_n[3] * self.Solid_1.kv  # mu_3 is total, not by volume
        vol_slurry = vol + vol_solid

        if self.method == 'moments':
            ddistr_dt, transf = self.method_of_moments(distrib, mass_conc, temp,
                                                       params, rho_s,
                                                       vol=vol_slurry)
        elif self.method == '1D-FVM':
            ddistr_dt, transf = self.fvm_method(distrib, mu_n, mass_conc, temp,
                                                params, rho_s, vol=vol_slurry)

        # Balance for target
        self.Liquid_1.updatePhase(mass_conc=mass_conc, vol=vol)

        dvol_liq = -transf/rho_liq  # TODO: results not consistent with mu_3
        dcomp_dt = -transf/vol * (self.kron_jtg - mass_conc/rho_liq)

        dliq_dt = np.append(dcomp_dt, dvol_liq)

        if self.basis == 'mass_frac':
            dcomp_dt *= 1 / rho_liq

        dmaterial_dt = np.concatenate((ddistr_dt, dliq_dt))

        return dmaterial_dt, transf  # transf [kg/s]

    def energy_balances(self, time: float, params, cryst_rate: np.ndarray,
                        u_inputs: dict, rhos: list, mu_n: np.ndarray,
                        distrib, mass_conc: np.ndarray, temp: float,
                        temp_ht: "float | None", vol: float,
                        h_in=None, heat_prof: bool = False
                        ) -> "float | tuple | np.ndarray":
        """
        Energy balances for the batch crystallizer.

        Parameters
        ----------
        time : float
            integration time [s].
        params : array-like
            kinetic parameters. Unused here, kept for signature
            compatibility.
        cryst_rate : numpy.ndarray
            crystallization mass rate [kg/s], as returned by
            :meth:`material_balances`.
        u_inputs : dict
            unit inputs at `time`. Unused by the batch unit.
        rhos : list of float
            [liquid, solid] densities [kg/m**3].
        mu_n : array-like
            crystal size distribution moments [m**n] (total basis).
        distrib : array-like
            distribution state. Unused here, kept for signature
            compatibility.
        mass_conc : array-like
            liquid-phase mass concentrations [kg/m**3].
        temp : float
            liquid temperature [K].
        temp_ht : float or None
            jacket temperature [K]. ``None`` when the unit carries no
            jacket state.
        vol : float
            liquid volume [m**3].
        h_in : None
            unused, kept for signature compatibility.
        heat_prof : bool, optional
            if True, return the individual heat terms instead of the
            temperature derivatives. The default is False.

        Returns
        -------
        ndarray or float or tuple
            If `heat_prof` is True, shape (2,): signed crystallization source [W]
            and jacket heat removed [W], with column 1 replaced by total heat
            capacitance [J/K] for nonadiabatic prescribed temperature. See the
            common contract in _store_heat_duty. Otherwise ``dtemp_dt`` [K/s], or
            the pair
            (``dtemp_dt``, ``dtht_dt``) [K/s] when a jacket state is present.

        Notes
        -----
        When ``'temp'`` is a control, ``ht_term`` carries the heat
        capacitance [J/K] instead of a heat rate, so that the caller can
        back out the required duty. Capacitance per liquid volume is
        multiplied by liquid volume for storage; vessel geometry and jacket
        volume retain the total slurry-volume basis.
        """

        vol_solid = mu_n[3] * self.Solid_1.kv  # mu_3 is total, not by volume
        vol_total = vol + vol_solid  # [m**3], slurry volume

        phi = vol / vol_total  # [-], liquid volume fraction
        phis = [phi, 1 - phi]  # [-], [liq, sol]

        # Suspension properties  TODO: slurry should be updated here
        capacitance = self.Slurry.getCp(temp, phis, rhos,
                                        times_vliq=True)  # [J/m**3 liquid/K]

        # Renaming
        dh_cryst = -1.46e4  # [J/kg]
        # dh_cryst = -self.Liquid_1.delta_fus[self.target_ind] / \
        #     self.Liquid_1.mw[self.target_ind] * 1000  # [J/kg]

        height_liq = vol_total / (np.pi/4 * self.diam_tank**2)  # [m]
        # [m**2], wetted lateral area plus tank base
        area_ht = np.pi * self.diam_tank * height_liq + self.area_base

        source_term = dh_cryst*cryst_rate  # [J/s]

        if self.adiabatic:
            ht_term = 0  # [J/s]
        elif 'temp' in self.controls.keys():
            ht_term = capacitance * vol  # [J/K], return capacitance
        elif 'temp' in self.states_uo:
            ht_term = self.u_ht*area_ht*(temp - temp_ht)  # [J/s]

        if heat_prof:
            heat_components = np.hstack([source_term, ht_term])
            return heat_components
        else:
            # Balance inside the tank
            dtemp_dt = (-source_term - ht_term) / capacitance / vol  # [K/s]

            if temp_ht is not None:
                ht_dict = self.Utility.get_inputs(time)
                tht_in = ht_dict['temp_in']  # [K], Utility inlet temperature
                flow_ht = ht_dict['vol_flow']  # [m**3/s]

                cp_ht = 4180  # [J/kg/K]
                rho_ht = 1000  # [kg/m**3]
                vol_ht = (vol_total * DEFAULT_JACKET_VOLUME_RATIO
                          if self.vol_ht is None else self.vol_ht)  # [m**3]

                dtht_dt = flow_ht / vol_ht * (tht_in - temp_ht) - \
                    self.u_ht*area_ht*(temp_ht - temp) / rho_ht/vol_ht/cp_ht

                return dtemp_dt, dtht_dt

            else:
                return dtemp_dt

    def retrieve_results(self, time: np.ndarray, states: np.ndarray) -> None:
        """Store batch profiles, update inventories, and construct the outlet.

        Parameters
        ----------
        time : ndarray
            Reporting times [s], shape (num_times,).
        states : ndarray
            Solver rows in name_states order: total moments [um**n] or scaled
            total FVM CSD [#/um], concentrations [kg/m**3], liquid volume
            [m**3], and optional temperatures [K]. FVM retrieval removes
            scale in place.

        Notes
        -----
        Raw solver moments are seeded in micrometre lengths by solve_unit.
        Retrieval converts them to total SI moments [m**n] for reported and
        phase values. Public states_di and result metadata both describe SI.
        Each run is converted before storage so continuation preserves SI.
        Final solid mass and volume follow kv*mu_3 of the total moments;
        outlet moments use the final combined liquid and solid volume.
        For the solid population, moment-mode retrieval updates only moments,
        mass, and volume. Any seed distrib/x_distrib is retained for plotting
        consumers and is not refreshed to represent the final population.
        """
        time = np.array(time)
        self.elapsed_time = time[-1]

        # ---------- Create result object
        dp = unpack_states(states, self.dim_states, self.name_states)
        dp['time'] = time

        if self.method == '1D-FVM':
            dp['distrib'] *= 1 / self.scale
            dp['x_cryst'] = self.x_grid

            moms = self.Solid_1.getMoments(distrib=dp['distrib'])
            dp['mu_n'] = moms

            dp['vol_distrib'] = self.Solid_1.convert_distribution(
                num_distr=dp['distrib'])

        if 'temp' in self.controls:
            control = self.controls['temp']
            dp['temp'] = control['fun'](time, *control['args'], **control['kwargs'])

        sat_conc = self.Kinetics.get_solubility(dp['temp'], dp['mass_conc'])

        supersat = dp['mass_conc'][:, self.target_ind] - sat_conc

        dp['solubility'] = sat_conc
        dp['supersat'] = supersat

        if self.method == 'moments':
            dp['mu_n'] = dp['mu_n'] * (1e-6)**np.arange(self.num_distr)  # [m**n]

        self.profiles_runs.append(dp)
        dp = self.flatten_states()

        self.result = DynamicResult(self.states_di, self.fstates_di, **dp)
        # ---------- Update phases
        vol_sol = dp['mu_n'][-1, 3] * self.Solid_1.kv  # [m**3]

        rho_solid = self.Solid_1.getDensity()  # [kg/m**3]
        mass_sol = rho_solid * vol_sol  # [kg]

        vol_slurry = dp['vol'][-1] + vol_sol  # [m**3]

        self.Liquid_1.updatePhase(mass_conc=dp['mass_conc'][-1],
                                  vol=dp['vol'][-1])

        self.Liquid_1.temp = dp['temp'][-1]
        self.Solid_1.temp = dp['temp'][-1]
        if self.method == '1D-FVM':
            slurry = Slurry(vol=vol_slurry)
            self.Solid_1.updatePhase(distrib=dp['distrib'][-1],
                                     mass=mass_sol)

        elif self.method == 'moments':
            self.Solid_1.updatePhase(moments=dp['mu_n'][-1], mass=mass_sol)
            slurry = Slurry(vol=vol_slurry, moments=dp['mu_n'][-1] / vol_slurry)

        # Create outlets
        liquid_out = copy.deepcopy(self.Liquid_1)
        solid_out = copy.deepcopy(self.Solid_1)

        self.Outlet = slurry
        self.Outlet.Phases = (liquid_out, solid_out)

        self.outputs = dp

        # ---------- Calculate heat duty
        self.get_heat_duty(time, states)

    def get_heat_duty(self, time: np.ndarray, states: np.ndarray) -> None:
        """Evaluate and integrate the batch utility heat profile.

        Parameters
        ----------
        time : ndarray
            Reporting times [s], shape (num_times,).
        states : ndarray
            Retrieved state rows: unscaled total FVM distribution [#/um] or
            total moments [um**n], concentrations [kg/m**3], liquid volume
            [m**3], and optional temperatures [K]. FVM retrieval has already
            removed the numerical distribution scale from these rows.

        Notes
        -----
        Moment-mode retrieval requires scale=1 under the existing scaling
        convention. Stores heat_prof and heat_duty using the rate/capacitance
        column contract in _store_heat_duty. Positive duty [J] means heat removed,
        including under prescribed temperature (correcting the former sign).
        Uses the current stored active kinetic parameters for every profile row.
        """
        q_heat = np.zeros((len(time), 2))  # [W] or [J/K], profile column contract
        merged_params = self.Kinetics.concat_params()[self.mask_params]
        # [native active parameter units], current solve rather than an old iterate
        for ind, row in enumerate(states):
            row = row.copy()  # [state units], documented above
            row[:self.num_distr] *= self.scale  # [-], numerical distribution scale
            q_heat[ind] = self.unit_model(time[ind], row, merged_params,
                                          enrgy_bce=True)
        self._store_heat_duty(time, q_heat)


class MSMPR(_BaseCryst):
    """ Construct a MSMPR object.
    
    Parameters
    ----------
    target_comp : str, list of strings
        Name of the crystallizing compound(s) from .json file.
    mask_params : list of bool (optional, default = None)
        Binary list of which parameters to exclude from the kinetics
        computation
    method : str
        Choice of the numerical method. Options are: 'moments', '1D-FVM'
    scale : float
        Scaling factor by which crystal size distribution will be
        multiplied.
    controls : dict of dicts (funcs) (optional, default = None)
        Dictionary with keys representing the state (e.g.'Temp')
        which is controlled and the value indicating the function
        to use while computing the varible. Functions are of the form
        f(time) = state_value
    cfun_solub: callable
        User defined function for the solubility function :
        func(conc)
    adiabatic : bool (optional, default =True)
        Boolean value indicating whether the heat transfer of
        the crystallization is considered.
    reset_states : bool (optional, default = False)
        Boolean value indicating whether the states should be
        reset before simulation
    basis : str (optional, default = 'mass_conc')
        Options : 'massfrac', 'massconc'
    state_events : lsit of dict(s)
        list of dictionaries, each one containing the specification of a
        state event
    """
    def __init__(self, target_comp,
                 mask_params=None,
                 method='1D-FVM', scale=1, vol_tank=None,
                 controls=None, adiabatic=False, rad_zero=0,
                 reset_states=False,
                 h_conv=1000, vol_ht=None, basis='mass_conc',
                 jac_type=None, num_interp_points=3, state_events=None,
                 param_wrapper=None):

        super().__init__(mask_params, method, target_comp, scale, vol_tank,
                         controls, adiabatic, rad_zero,
                         reset_states, h_conv, vol_ht,
                         basis, jac_type, state_events, param_wrapper)

        # self.states_uo.append('conc_j')
        self.is_continuous = True
        self.oper_mode = 'Continuous'
        self._Inlet = None

        # self.nomenclature()

        self.vol_offset = 0.75
        self.num_interp_points = num_interp_points

    @property
    def Inlet(self):
        return self._Inlet

    @Inlet.setter
    def Inlet(self, inlet_object):
        self._Inlet = inlet_object
        self._Inlet.num_interpolation_points = self.num_interp_points

    def _get_tau(self):
        time_upstream = getattr(self.Inlet, 'time_upstream', None)
        if time_upstream is None:
            time_upstream = [0]

        inputs = self.get_inputs(time_upstream[-1])

        volflow_in = inputs['Inlet']['vol_flow']
        tau = self.Liquid_1.vol / volflow_in

        self.tau = tau
        return tau

    def solve_steady_state(self, frac_seed: float, temp: float, *,
                           num_scan: int = 64) -> tuple:
        """Solve the MSMPR moment and target-composition steady balances.

        Parameters
        ----------
        frac_seed : float
            Target liquid composition hint in the unit's basis: [kg/m**3]
            for mass_conc or [kg/kg] for mass_frac. Among accepted roots,
            return the one nearest this finite, positive-growth hint.
            Kinetics must use the same basis.
        temp : float
            Constant vessel temperature [K].
        num_scan : int, optional
            Number of evenly spaced domain samples, at least two. The default
            64 is a numerical resolution choice, not a physical parameter.
            Detected admissibility transitions are refined to concentration
            resolution. Increase this value for narrow admissible intervals
            or closely spaced roots; tangential roots can still be missed.

        Returns
        -------
        x_vec : ndarray
            Crystal size grid [um], shape (num_bins,).
        f_convg : ndarray
            Steady number density [#/m**3/um] on x_vec.
        composition : float
            Target liquid composition in the unit's basis: [kg/m**3] or
            [kg/kg].
        info : scipy.optimize.RootResults
            Mass-concentration root [kg/m**3] convergence information; for
            kv=0, records the exact B/G boundary [#/m**3/um] with zero
            iterations and method='closed-form'.
        final_fn : float
            Dynamic model's target derivative, evaluated by calling
            material_balances at the returned analytical moments, in
            [kg/m**3/s] or [kg/kg/s], according to the unit's basis. Density
            is evaluated at the attached liquid composition, so this residual
            cannot expose error from the constant-density approximation.

        Raises
        ------
        ValueError
            If basis or num_scan is invalid, nuclei have finite radius, the
            inlet concentration is nonpositive or nonfinite, the hint is
            nonfinite or has nonpositive growth, or no scanned root passes
            the positive-population, positive-liquid-holdup, and relative
            population-closure gates, including skipped unusable brackets.
            Also raised for a bare LiquidStream inlet (supported only by the
            dynamic solve_unit path), or the unsupported c* cancellation feed
            with kv>0.
            No-root errors describe the nearest rejected candidate's gate,
            relative closure error, and closure_rtol when available.
        RuntimeError
            If the kv=0 boundary is not positive and finite.

        Warns
        -----
        UserWarning
            If brackets were skipped but an accepted root is returned; the
            warning reports the skipped count and suggests finer num_scan.

        Notes
        -----
        Solves the moment-mode MSMPR.material_balances target equation with
        phi_in=1 and no inlet solids. With D=Q/V [1/s], target concentration
        c [kg/m**3], and phi=1-kv*mu_3, this is
        0=D*(c_in-c*phi)-R*(1-c/rho_l). The derivative divides this expression
        by phi, and also by rho_l for mass_frac. Thus this is the dynamic
        model's solute bookkeeping, not a separate stream mass balance.

        Assumes constant tank liquid density and holdup, no inlet solids,
        zero-size nuclei, and positive size- and population-independent
        growth. The kinetic target index is initialized here, so no prior
        dynamic solve is needed. No Kinetics.alpha_fn impurity factor is
        applied, although the dynamic model applies it. Density is evaluated
        at the attached liquid composition and requested temperature. The feed
        term always uses the inlet liquid's own mass concentration, allowing
        its density to differ from the tank density.

        For n(L)=boundary*exp(-D*L/G), the infinite-domain moment factors
        are a_n=n!*(G/D)**(n+1)*(1e-6)**n, so mu_n=boundary*a_n in
        [m**n/m**3]. These analytical integrals satisfy the moment equations
        without grid truncation error. The returned grid CSD is a sample;
        downstream quadrature needs many G/D decay lengths to reproduce
        these analytical moments.
        With r=3*kv*rho_s*G*a_2*1e-6, R=boundary*r [kg/m**3/s]. At each
        trial c, solve the linear solute equation for boundary:
        boundary=D*(c_in-c)/(r*(1-c/rho_l)-D*c*kv*a_3).
        The rescaled residual is f(c)=D*(c_in-c)-coefficient*B(mu_n)/G.
        Scan the open interval 0<c<min(c_in,c*), where
        c*=rho_s*rho_l/(rho_s+rho_l), retaining positive growth and positive
        amplitude coefficients. Candidates must also have finite positive
        liquid volume fraction phi=1-kv*boundary*a_3. A zero solute residual
        alone does not establish positive liquid holdup. Growth is evaluated
        at each composition, including composition-dependent solubility.
        Each detected sign change between consecutive admissible samples is
        solved with brentq, using xtol=4*machine epsilon*c_in [kg/m**3] as
        concentration resolution and brentq's default rtol. Secondary
        nucleation can give multiple roots; return the accepted root nearest
        frac_seed. Brackets with nonfinite midpoint residuals, failed solves,
        or nonfinite returned roots or residuals are skipped. Warn with the
        skipped count if another root is returned. A finite midpoint does
        not rule out an unsampled inadmissible interval inside the bracket.
        The trivial washout state at exactly c_in and states above c_in
        (possible for feeds above c*) are not returned. Positive populations
        must satisfy the measured closure gate; frac_seed selects among
        accepted roots.

        Accept physically admissible, converged roots only when
        abs(f(c)) <= closure_rtol*D*(c_in-c) [kg/m**3/s], with
        closure_rtol=1e-5 [-]. Since boundary=D*(c_in-c)/coefficient,
        abs(f(c))/(D*(c_in-c)) equals abs(boundary-B/G)/boundary. Thus
        rejection depends on measured relative population-closure error,
        not just a sign change or the eliminated solute residual. The
        tolerance leaves over a decade of margin above the worst resolved
        repository case (2.6e-7, absolute-supersaturation low conversion)
        and below false-root errors of at least 1.9e-2. The residual roundoff
        floor is about eps*D*c_in. Relative depletion (c_in-c)/c_in, or
        relative supersaturation near growth onset, below roughly
        eps/closure_rtol (about 5e4 float64 spacings) cannot reliably pass
        closure. Such roots are rejected when measured error exceeds the
        gate. Increasing num_scan helps resolve narrow intervals or growth
        gaps, but cannot improve root accuracy or physical admissibility.
        Only kv=0 uses the zero-transfer branch: crystal volume vanishes,
        secondary nucleation sees kv*mu=0 at every population, and B/G is
        evaluated directly without iteration at c_in/concentration_scale.
        With kv>0, the cancellation feed c_in=c* is unsupported; use a
        slightly different feed. The final dynamic residual can expose
        omitted impurity effects, but uses the same constant-density
        approximation. Evaluation restores phase temperatures and solid
        moments, including on failure; the usual kinetic caches are refreshed.
        """
        if isinstance(self.Inlet, LiquidStream):
            raise ValueError(
                "solve_steady_state does not accept a bare LiquidStream; "
                "use solve_unit for a dynamic bare-liquid feed, or configure "
                "an inlet with a Liquid_1 phase for solve_steady_state.")
        if self.basis not in {'mass_conc', 'mass_frac'}:
            raise ValueError("basis must be 'mass_conc' or 'mass_frac'")
        if self.rad != 0:
            raise ValueError("finite-radius nuclei are unsupported; rad must be zero")
        self.Kinetics.target_idx = self.target_ind
        flow_v = self.Inlet.vol_flow / self.vol_slurry  # [1/s]
        x_vec = self.Solid_1.x_distrib  # [um]
        kv = self.Solid_1.kv  # [-]
        rho_solid = self.Solid_1.getDensity(temp=temp)  # [kg/m**3]
        rho_liquid = self.Liquid_1.getDensity(temp=temp)  # [kg/m**3]
        concentration_scale = rho_liquid if self.basis == 'mass_frac' else 1
        # [kg/m**3] or [-], configured composition to mass concentration
        concentration_in = self.Inlet.Liquid_1.mass_conc[self.target_ind]
        # [kg/m**3], the feed term consumed by material_balances
        composition_in = concentration_in / concentration_scale
        # [configured composition unit], feed concentration on the tank basis
        if not np.isfinite(concentration_in) or concentration_in <= 0:
            raise ValueError("Inlet target mass concentration must be positive and finite")
        if not np.isfinite(frac_seed):
            raise ValueError("Trial composition hint must be finite")
        if (not isinstance(num_scan, (int, np.integer))
                or isinstance(num_scan, bool) or num_scan < 2):
            raise ValueError("num_scan must be an integer of at least two")
        num_moments = self.Solid_1.num_mom
        orders = np.arange(num_moments)
        factorials = np.cumprod(np.maximum(orders, 1), dtype=float)
        # [-], n! including 0!=1
        metre_per_um = 1e-6  # [m/um], exact SI prefix conversion

        def population_data(composition: float) -> "tuple | None":
            """Evaluate growth and analytical exponential moment factors.

            Parameters
            ----------
            composition : float
                Target liquid composition [kg/m**3] or [kg/kg].

            Returns
            -------
            tuple
                Primary nucleation [#/m**3/s], growth [um/s], moment factors
                [m**n*um], mass-source factor [kg*um/s], and coefficient of
                boundary in the solute equation [kg*um/s].
                Returns None when growth is nonpositive or nonfinite.
            """
            empty_moments = np.zeros(num_moments)  # [m**n/m**3]
            nucl_seed, growth, _ = self.Kinetics.get_kinetics(
                composition, temp, kv, empty_moments)
            # [#/m**3/s], [um/s], [um/s]
            if not np.isfinite(growth) or growth <= 0:
                return None
            factors = (factorials * (growth / flow_v)**(orders + 1)
                       * metre_per_um**orders)  # [m**n*um]
            mass_factor = 3 * kv * rho_solid * growth * factors[2] * metre_per_um
            # [kg*um/s], crystal mass source per unit boundary number density
            concentration = composition * concentration_scale  # [kg/m**3]
            coefficient = (mass_factor * (1 - concentration / rho_liquid)
                           - flow_v * concentration * kv * factors[3])
            # [kg*um/s], coefficient of boundary in the solute equation
            return nucl_seed, growth, factors, mass_factor, coefficient

        def composition_residual(concentration: float) -> float:
            """Evaluate the solute residual without its amplitude pole.

            Parameters
            ----------
            concentration : float
                Trial tank target concentration [kg/m**3].

            Returns
            -------
            float
                Rescaled population-closure residual [kg/m**3/s], or NaN
                for nonpositive/nonfinite growth or amplitude coefficient.
            """
            composition = concentration / concentration_scale  # [basis unit]
            data = population_data(composition)
            if data is None or not np.isfinite(data[-1]) or data[-1] <= 0:
                return np.nan
            _, growth, factors, _, coefficient = data
            feed_difference = flow_v * (concentration_in - concentration)
            # [kg/m**3/s]
            boundary = feed_difference / coefficient  # [#/m**3/um]
            moments = boundary * factors  # [m**n/m**3]
            nucl, _, _ = self.Kinetics.get_kinetics(composition, temp, kv, moments)
            # [#/m**3/s], total population-dependent nucleation
            return feed_difference - coefficient * nucl / growth

        if population_data(frac_seed) is None:
            raise ValueError(
                f"Trial composition {frac_seed!r} has nonpositive growth; "
                "check solubility and the growth kinetics")
        inlet_data = population_data(composition_in)
        critical_concentration = rho_solid * rho_liquid / (rho_solid + rho_liquid)
        # [kg/m**3], c* where the amplitude coefficient cancels for positive kv
        # Four rounded arithmetic steps form the coefficient; use its
        # mass-source scale to recognize exact physical cancellation.
        if (kv > 0 and inlet_data is not None
                and abs(inlet_data[-1]) <= 4 * eps * abs(inlet_data[-2])):
            raise ValueError(
                f"Unsupported feed concentration {float(concentration_in)!r} kg/m**3 "
                f"at c*={float(critical_concentration)!r} kg/m**3 for kv > 0; "
                "use a slightly different feed concentration")
        if kv == 0:
            if inlet_data is None:
                raise ValueError("Inlet composition has nonpositive growth; check solubility")
            nucl_seed, growth, factors, _, _ = inlet_data
            # [#/m**3/s], [um/s], [m**n*um], [kg*um/s], [kg*um/s]
            composition = composition_in  # [configured composition unit]
            boundary = nucl_seed / growth  # [#/m**3/um], kv*mu=0 at every population
            info = RootResults(boundary, iterations=0, function_calls=1,
                               flag=0, method='closed-form')
            # flag=0 is SciPy's success code; one evaluation of B/G, no iteration.
        else:
            upper = min(concentration_in, critical_concentration)  # [kg/m**3]
            concentration_xtol = 4 * eps * concentration_in  # [kg/m**3]
            closure_rtol = 1e-5  # [-], relative population-closure tolerance
            # Resolved repository cases reach 2.6e-7 error; false roots reach
            # at least 1.9e-2. This leaves over a decade of margin each way.
            grid = np.linspace(np.nextafter(0.0, upper),
                               np.nextafter(upper, 0.0), num_scan)  # [kg/m**3]
            scan = []
            previous = None
            for concentration in grid:
                residual = composition_residual(concentration)  # [kg/m**3/s]
                admissible = np.isfinite(residual)
                if previous is not None and admissible != np.isfinite(previous[1]):
                    # Refine detected growth/coefficient boundaries so roots
                    # close to positive-growth onset can also be bracketed.
                    left, right = previous[0], concentration  # [kg/m**3]
                    left_admissible = np.isfinite(previous[1])
                    while right - left > concentration_xtol:
                        midpoint = (left + right) / 2  # [kg/m**3]
                        if midpoint == left or midpoint == right:
                            break
                        if np.isfinite(composition_residual(midpoint)) == left_admissible:
                            left = midpoint  # [kg/m**3]
                        else:
                            right = midpoint  # [kg/m**3]
                    edge = left if left_admissible else right  # [kg/m**3]
                    scan.append((edge, composition_residual(edge)))
                scan.append((concentration, residual))
                previous = concentration, residual
            roots = []
            rejected = []
            skipped_brackets = 0
            for (left, left_fn), (right, right_fn) in zip(scan, scan[1:]):
                # Do not bridge intervals containing inadmissible scan points.
                if (not np.isfinite(left_fn) or not np.isfinite(right_fn)
                        or left == right or left_fn * right_fn > 0):
                    continue
                midpoint = (left + right) / 2  # [kg/m**3]
                if not np.isfinite(composition_residual(midpoint)):
                    skipped_brackets += 1
                    continue
                try:
                    concentration, root_info = brentq(
                        composition_residual, left, right,
                        xtol=concentration_xtol, full_output=True)  # [kg/m**3]
                except (ValueError, RuntimeError):
                    # An unsampled growth gap or solver failure invalidates
                    # this bracket, not roots resolved in other brackets.
                    skipped_brackets += 1
                    continue
                if not np.isfinite(concentration):
                    skipped_brackets += 1
                    continue
                residual = composition_residual(concentration)  # [kg/m**3/s]
                if not np.isfinite(residual):
                    skipped_brackets += 1
                    rejected.append((concentration / concentration_scale,
                                     'finite residual', np.inf))
                    # [basis unit], gate name, unavailable relative closure error [-]
                    continue
                composition = concentration / concentration_scale  # [basis unit]
                _, growth, factors, _, coefficient = population_data(composition)
                # [#/m**3/s], [um/s], [m**n*um], [kg*um/s], [kg*um/s]
                boundary = flow_v * (concentration_in - concentration) / coefficient
                # [#/m**3/um], positive population recovered from solute balance
                liquid_fraction = 1 - kv * boundary * factors[3]  # [-]
                depletion_rate = flow_v * (concentration_in - concentration)
                # [kg/m**3/s], residual/depletion_rate equals relative B/G error
                relative_error = (abs(residual) / depletion_rate
                                  if depletion_rate > 0 else np.inf)  # [-]
                if not root_info.converged:
                    gate = 'solver convergence'
                elif not np.isfinite(boundary) or boundary <= 0:
                    gate = 'positive population'
                elif not np.isfinite(liquid_fraction) or liquid_fraction <= 0:
                    gate = 'positive liquid volume fraction'
                elif relative_error > closure_rtol:
                    gate = 'population closure'
                else:
                    roots.append((composition, boundary, growth, factors, root_info))
                    continue
                rejected.append((composition, gate, relative_error))
                # [basis unit], gate name, relative population-closure error [-]
            if not roots:
                if rejected:
                    candidate, gate, relative_error = min(
                        rejected, key=lambda root: abs(root[0] - frac_seed))
                    # [basis unit], gate name, relative population-closure error [-]
                    diagnosis = (
                        f"Nearest candidate {float(candidate)!r} ({self.basis}) "
                        f"rejected by {gate} gate; relative closure error="
                        f"{relative_error:.6g}; closure_rtol={closure_rtol:g}. ")
                else:
                    diagnosis = (
                        "No finite candidate resolved; rejected by bracket "
                        "admissibility gate; relative closure error=unavailable; "
                        f"closure_rtol={closure_rtol:g}. ")
                raise ValueError(
                    f"No accepted steady root in scanned tank concentration domain "
                    f"(0, {float(upper)!r}) kg/m**3 with positive growth, coefficient, "
                    "finite positive liquid volume fraction, and population closure. "
                    f"{diagnosis}"
                    "Washout at exactly the inlet and states above the inlet "
                    f"concentration are not returned; skipped brackets: {skipped_brackets}. "
                    "Increasing num_scan may resolve narrow root intervals or "
                    "growth gaps, but cannot fix physical inadmissibility or "
                    "insufficient root accuracy; check the feed and kinetics.")
            if skipped_brackets:
                warnings.warn(
                    f"MSMPR steady-state scan skipped brackets: {skipped_brackets}; "
                    "use a finer num_scan to resolve narrow growth gaps or roots.",
                    stacklevel=2)
            composition, boundary, growth, factors, info = min(
                roots, key=lambda root: abs(root[0] - frac_seed))
            # [basis unit], [#/m**3/um], [um/s], [m**n*um], solver information
        if not np.isfinite(boundary) or boundary <= 0:
            raise RuntimeError(
                "Steady-state population boundary must be positive and finite; "
                "check the nucleation kinetics")
        f_convg = boundary * np.exp(-flow_v / growth * x_vec)  # [#/m**3/um]
        moments_si = boundary * factors  # [m**n/m**3]
        concentrations = self.Liquid_1.mass_conc.copy()  # [kg/m**3]
        concentrations[self.target_ind] = composition * concentration_scale
        # [kg/m**3], tank target concentration on the dynamic model's basis
        inputs = {'Inlet': {'vol_flow': self.Inlet.vol_flow},
                  'Liquid_1': {'mass_conc': self.Inlet.Liquid_1.mass_conc.copy()}}
        # [m**3/s], [kg/m**3], solid-free feed as consumed by material_balances
        if self.method == 'moments':
            population_state = moments_si / metre_per_um**orders  # [um**n/m**3]
            inputs['Inlet']['mu_n'] = np.zeros(num_moments)  # [m**n/m**3]
        else:
            population_state = f_convg * self.scale  # scaled [#/m**3/um]
            inputs['Inlet']['distrib'] = np.zeros_like(f_convg)  # [#/m**3/um]
            self.dx = self.Slurry.dx  # [um], same FVM initialization as solve_unit
        rho_inlet = self.Inlet.Liquid_1.getDensity(temp=temp)  # [kg/m**3]
        liquid_temp, solid_temp = self.Liquid_1.temp, self.Solid_1.temp  # [K]
        if self.method != 'moments':
            solid_moments = self.Solid_1.moments.copy()  # [m**n], phase inventory
        try:
            self.Liquid_1.temp = temp  # [K], match the dynamic density evaluation
            self.Solid_1.temp = temp  # [K]
            derivative, _ = self.material_balances(
                0.0, None, inputs, [[rho_liquid, rho_solid], [rho_inlet, None]],
                moments_si, population_state, concentrations, temp, None,
                self.vol_slurry, [1.0, 0.0])
            # [population unit/s], [composition unit/s], actual dynamic balance
        finally:
            self.Liquid_1.temp, self.Solid_1.temp = liquid_temp, solid_temp  # [K]
            if self.method != 'moments':
                self.Solid_1.moments[:] = solid_moments  # [m**n], preserve inventory
        final_fn = derivative[len(population_state) + self.target_ind]
        # [configured composition unit/s], actual dynamic target derivative
        return x_vec, f_convg, composition, info, final_fn

    def material_balances(self, time, params, u_inputs, rhos, mu_n,
                          distrib, mass_conc, temp, temp_ht, vol, phi_in):
        """
        Material balances for the continuous (MSMPR) crystallizer.

        Parameters
        ----------
        time : float
            integration time [s].
        params : array-like
            kinetic parameters passed to ``self.Kinetics``.
        u_inputs : dict
            unit inputs at `time`, holding the inlet volumetric flow
            [m**3/s], inlet distribution and inlet mass concentrations
            [kg/m**3].
        rhos : list
            [[liquid, solid] tank densities, [liquid, solid] inlet
            densities], all in [kg/m**3].
        mu_n : array-like
            crystal size distribution moments [m**n/m**3] (volumetric
            basis).
        distrib : array-like
            distribution state: [#/m**3/um] for ``method == '1D-FVM'``, or
            the raw moment vector [um**n/m**3] for ``method == 'moments'``.
        mass_conc : array-like
            liquid-phase mass concentrations [kg/m**3].
        temp : float
            liquid temperature [K].
        temp_ht : float or None
            jacket temperature [K]. Unused here, kept for signature
            compatibility.
        vol : float
            slurry volume [m**3].
        phi_in : array-like
            inlet [liquid, solid] volume fractions [-].

        Returns
        -------
        dmaterial_dt : numpy.ndarray
            stacked derivatives with mixed units, in this order:
            ``ddistr_dt`` ([#/m**3/um/s] for '1D-FVM', [um**n/m**3/s] for
            'moments') and ``dcomp_dt`` ([kg/m**3/s], or [1/s] when
            ``basis == 'mass_frac'``).
        transf : numpy.ndarray
            crystallization mass rate per unit slurry volume [kg/m**3/s].

        Notes
        -----
        MSMPR states are declared on a *volumetric* basis in
        :meth:`_BaseCryst.solve_unit` (``mu_n`` in [um**n/m**3],
        ``distrib`` in [#/m**3/um]), and the kinetics are evaluated with the
        default ``vol=1``. ``transf`` is therefore an intensive rate
        [kg/m**3/s], unlike the total [kg/s] rate returned by
        :meth:`BatchCryst.material_balances` and
        :meth:`SemibatchCryst.material_balances`, whose states are on a
        total basis. :meth:`energy_balances` multiplies it by ``vol`` to
        obtain a heat rate in [J/s].

        The kinetic mass-transfer rate uses the crystal density [kg/m**3],
        while the liquid-volume correction normalizes ``mass_conc`` by the
        tank liquid density [kg/m**3].
        """

        rho_liq, rho_sol = rhos[0]  # [kg/m**3], liquid and solid tank densities

        input_flow = u_inputs['Inlet']['vol_flow']  # [m**3/s]

        input_conc = u_inputs['Liquid_1']['mass_conc']

        if self.method == 'moments':
            input_distrib = u_inputs['Inlet']['mu_n'] * (1e6)**np.arange(self.num_distr)#* self.scale
            ddistr_dt, transf = self.method_of_moments(distrib, mass_conc, temp,
                                                       params, rho_sol)
        elif self.method == '1D-FVM':
            input_distrib = u_inputs['Inlet']['distrib'] * self.scale
            ddistr_dt, transf = self.fvm_method(distrib, mu_n, mass_conc, temp,
                                                params, rho_sol)

            self.Solid_1.moments[[2, 3]] = mu_n[[2, 3]]

        # ---------- Add flow terms
        # Distribution
        tau_inv = input_flow / vol  # [1/s], inverse residence time
        flow_distrib = tau_inv * (input_distrib - distrib)

        ddistr_dt = ddistr_dt + flow_distrib
        # Liquid phase
        phi = 1 - self.Solid_1.kv * mu_n[3]  # [-], liquid volume fraction

        # [kg/m**3/s] both terms
        flow_term = tau_inv * (input_conc*phi_in[0] - mass_conc*phi)
        transf_term = transf * (self.kron_jtg - mass_conc / rho_liq)
        dcomp_dt = 1 / phi * (flow_term - transf_term)

        if self.basis == 'mass_frac':
            dcomp_dt *= 1 / rho_liq

        dmaterial_dt = np.concatenate((ddistr_dt, dcomp_dt))

        return dmaterial_dt, transf  # transf [kg/m**3/s]

    def energy_balances(self, time: float, params, cryst_rate: np.ndarray,
                        u_inputs: dict, rhos: list, mu_n: np.ndarray,
                        distrib, mass_conc: np.ndarray, temp: float,
                        temp_ht: "float | None", vol: float,
                        h_in: float, heat_prof: bool = False
                        ) -> "float | tuple | np.ndarray":
        """
        Energy balances for the continuous (MSMPR) crystallizer.

        Parameters
        ----------
        time : float
            integration time [s].
        params : array-like
            kinetic parameters. Unused here, kept for signature
            compatibility.
        cryst_rate : numpy.ndarray
            crystallization mass rate per unit slurry volume [kg/m**3/s],
            as returned by :meth:`material_balances`. This is the
            volumetric counterpart of the total [kg/s] rate returned by
            :meth:`BatchCryst.material_balances`, hence the ``* vol``
            factor in the source term below.
        u_inputs : dict
            unit inputs at `time`, holding the inlet volumetric flow
            [m**3/s].
        rhos : list
            [[liquid, solid] tank densities, [liquid, solid] inlet
            densities], all in [kg/m**3].
        mu_n : array-like
            crystal size distribution moments [m**n/m**3].
        distrib : array-like
            distribution state. Unused here, kept for signature
            compatibility.
        mass_conc : array-like
            liquid-phase mass concentrations [kg/m**3].
        temp : float
            liquid temperature [K].
        temp_ht : float or None
            jacket temperature [K]. ``None`` when the unit carries no
            jacket state.
        vol : float
            slurry volume [m**3].
        h_in : float
            volumetric enthalpy of the inlet stream [J/m**3].
        heat_prof : bool, optional
            if True, return the individual heat terms instead of the
            temperature derivatives. The default is False.

        Returns
        -------
        ndarray or float or tuple
            If `heat_prof` is True, shape (3,): signed crystallization source [W],
            jacket heat removed [W], and net flow heat entering [W]. Column 1 is
            replaced by total heat capacitance [J/K] for nonadiabatic prescribed
            temperature; see _store_heat_duty. Otherwise ``dtemp_dt`` [K/s], or
            the pair
            (``dtemp_dt``, ``dtht_dt``) [K/s] when a jacket state is present.

        Notes
        -----
        When ``'temp'`` is a control, ``ht_term`` carries the heat
        capacitance [J/K] instead of a heat rate, so that the caller can
        back out the required duty.
        """

        rho_susp, rho_in = rhos

        input_flow = u_inputs['Inlet']['vol_flow']  # [m**3/s]

        # Thermodynamic properties (basis: slurry volume)
        phi_liq = 1 - self.Solid_1.kv * mu_n[3]  # [-], liquid volume fraction

        phis = [phi_liq, 1 - phi_liq]  # [-], [liq, sol]
        h_sp = self.Slurry.getEnthalpy(temp, phis, rho_susp)  # [J/m**3]
        capacitance = self.Slurry.getCp(temp, phis, rho_susp)  # [J/m**3/K]

        # Renaming
        dh_cryst = -1.46e4  # [J/kg]  # TODO: read this from json file
        # dh_cryst = -self.Liquid_1.delta_fus[self.target_ind] / \
        #     self.Liquid_1.mw[self.target_ind] * 1000  # [J/kg]

        height_liq = vol / (np.pi/4 * self.diam_tank**2)  # [m]
        # [m**2], wetted lateral area plus tank base
        area_ht = np.pi * self.diam_tank * height_liq + self.area_base

        # Energy terms [J/s]
        flow_term = input_flow * (h_in - h_sp)
        source_term = dh_cryst*cryst_rate * vol

        if self.adiabatic:
            ht_term = 0  # [J/s]
        elif 'temp' in self.controls.keys():
            ht_term = capacitance * vol  # [J/K], return capacitance
        elif 'temp' in self.states_uo:
            ht_term = self.u_ht*area_ht*(temp - temp_ht)  # [J/s]
        if heat_prof:
            heat_components = np.hstack([source_term, ht_term, flow_term])
            return heat_components
        else:
            # Balance inside the tank
            dtemp_dt = (flow_term - source_term - ht_term) / \
                vol / capacitance  # [K/s]

            if self.adiabatic or temp_ht is None:
                return dtemp_dt

            # Balance in the jacket
            ht_media = self.Utility.get_inputs(time)
            flow_ht = ht_media['vol_flow']  # [m**3/s]
            tht_in = ht_media['temp_in']  # [K], Utility inlet temperature

            cp_ht = self.Utility.cp  # [J/kg/K]
            rho_ht = self.Utility.rho  # [kg/m**3]

            vol_ht = (self.vol_tank / self.vol_offset * DEFAULT_JACKET_VOLUME_RATIO
                      if self.vol_ht is None else self.vol_ht)  # [m**3]

            dtht_dt = flow_ht / vol_ht * (tht_in - temp_ht) - \
                self.u_ht*area_ht*(temp_ht - temp) / rho_ht/vol_ht/cp_ht

            return dtemp_dt, dtht_dt

    def retrieve_results(self, time: np.ndarray, states: np.ndarray) -> None:
        """Store continuous or semibatch profiles and construct the outlet.

        Parameters
        ----------
        time : ndarray
            Reporting times [s], shape (num_times,).
        states : ndarray
            MSMPR solver rows contain volume-specific moments [um**n/m**3]
            or scaled FVM CSD [#/m**3/um], liquid concentrations [kg/m**3],
            and optional temperatures [K], in name_states order. Semibatch
            rows instead contain total moments [um**n] or CSD [#/um] and
            a liquid volume [m**3]. FVM retrieval removes scale in place.

        Notes
        -----
        solve_unit seeds raw solver moments in micrometre lengths. Retrieval
        converts them to SI [m**n/m**3] for MSMPR profiles or total [m**n]
        for Semibatch profiles. Public states_di and result metadata describe
        these SI reported values; phase moments also retain SI lengths.
        Semibatch solid inventory follows kv*mu_3 of the final total moments.
        Moment outlets retain that inventory and the phase-owned shape factor,
        including when the seed phase also has a size grid.
        For the solid population, moment-mode retrieval updates only moments,
        mass, and volume. Any seed distrib/x_distrib is retained for plotting
        consumers and is not refreshed to represent the final population.
        """
        time = np.array(time)

        # ---------- Create result object
        inputs = self.get_inputs(time)
        volflow = inputs['Inlet']['vol_flow']

        dp = unpack_states(states, self.dim_states, self.name_states)

        dp['time'] = time
        dp['vol_flow'] = volflow
        if self.method == '1D-FVM':
            dp['x_cryst'] = self.x_grid  # [um]

        if 'temp' in self.controls:
            control = self.controls['temp']
            dp['temp'] = control['fun'](time, *control['args'], **control['kwargs'])

        sat_conc = self.Kinetics.get_solubility(dp['temp'], dp['mass_conc'])

        supersat = dp['mass_conc'][:, self.target_ind] - sat_conc

        dp['solubility'] = sat_conc
        dp['supersat'] = supersat

        if self.method == '1D-FVM':
            dp['distrib'] *= 1 / self.scale
            moms = self.Solid_1.getMoments(distrib=dp['distrib'])
            dp['mu_n'] = moms

            dp['vol_distrib'] = self.Solid_1.convert_distribution(
                num_distr=dp['distrib'])

            if type(self) == MSMPR:
                vol_slurry = self.Slurry.vol
                self.Solid_1.updatePhase(distrib=dp['distrib'][-1] * vol_slurry)

        if self.method == 'moments':
            dp['mu_n'] = dp['mu_n'] * (1e-6)**np.arange(self.num_distr)

        if self.__class__.__name__ == 'SemibatchCryst' and self.method == '1D-FVM':
            dp['total_distrib'] = dp['distrib']

        self.profiles_runs.append(dp)
        dp = self.flatten_states()

        self.outputs = dp

        self.result = DynamicResult(self.states_di, self.fstates_di, **dp)

        # ---------- Update phases

        self.Solid_1.temp = dp['temp'][-1]
        self.Liquid_1.temp = dp['temp'][-1]

        if type(self) == MSMPR:
            vol_slurry = self.Slurry.vol
            vol_liq = (1 - self.Solid_1.kv * dp['mu_n'][-1, 3]) * vol_slurry

            self.Liquid_1.updatePhase(vol=vol_liq,
                                      mass_conc=dp['mass_conc'][-1])
            if self.method == '1D-FVM':
                distrib_tilde = dp['distrib'][-1] * vol_slurry
                self.Solid_1.updatePhase(distrib=distrib_tilde)

                self.Slurry = Slurry()

            elif self.method == 'moments':
                self.Slurry = Slurry(moments=dp['mu_n'][-1], vol=vol_slurry)

        else:
            vol_liq = dp['vol'][-1]
            self.Liquid_1.updatePhase(mass_conc=dp['mass_conc'][-1],
                                  vol=dp['vol'][-1])
            
            rho_solid = self.Solid_1.getDensity()  # [kg/m**3]
            vol_solid = dp['mu_n'][-1, 3] * self.Solid_1.kv  # [m**3]
            mass_solid = rho_solid*vol_solid  # [kg]


            vol_slurry = vol_solid + vol_liq

            if self.method == '1D-FVM':
                distrib_tilde = dp['total_distrib'][-1]
                self.Solid_1.updatePhase(distrib=distrib_tilde,
                                         mass= mass_solid)

                self.Slurry = Slurry()

            elif self.method == 'moments':
                self.Solid_1.updatePhase(moments=dp['mu_n'][-1], mass=mass_solid)
                self.Slurry = Slurry(vol=vol_slurry,
                                     moments=dp['mu_n'][-1] / vol_slurry)

        self.Slurry.Phases = (self.Solid_1, self.Liquid_1)
        self.elapsed_time = time[-1]

        # ---------- Create output stream
        path = self.Liquid_1.path_data

        solid_comp = np.zeros(self.num_species)
        solid_comp[self.target_ind] = 1

        if type(self) == MSMPR:
            liquid_out = LiquidStream(path,
                                      mass_conc=dp['mass_conc'][-1],
                                      temp=dp['temp'][-1], check_input=False)

            solid_out = SolidStream(path, mass_frac=solid_comp, kv=self.Solid_1.kv)

            if isinstance(inputs['Inlet']['vol_flow'], float):
                vol_flow = inputs['Inlet']['vol_flow']
            else:
                vol_flow = inputs['Inlet']['vol_flow'][-1]

            if self.method == '1D-FVM':

                self.Outlet = SlurryStream(
                    vol_flow=vol_flow,
                    x_distrib=self.x_grid,
                    distrib=dp['distrib'][-1])

            elif self.method == 'moments':

                self.Outlet = SlurryStream(
                    vol_flow=vol_flow,
                    moments=dp['mu_n'][-1])

            # Duty diagnostics are published for MSMPR; Semibatch has none.
            self.get_heat_duty(time, states)

        else:
            liquid_out = copy.deepcopy(self.Liquid_1)
            solid_out = copy.deepcopy(self.Solid_1)

            self.Outlet = Slurry(
                vol=vol_slurry,
                moments=dp['mu_n'][-1] / vol_slurry if self.method == 'moments' else None)

        # self.outputs = y_outputs
        self.Outlet.Phases = (liquid_out, solid_out)

    def get_heat_duty(self, time: np.ndarray, states: np.ndarray) -> None:
        """Evaluate and integrate the continuous utility heat profile.

        Parameters
        ----------
        time : ndarray
            Reporting times [s], shape (num_times,).
        states : ndarray
            Retrieved state rows: unscaled FVM distribution [#/m**3/um] or
            moments [um**n/m**3], concentrations [kg/m**3], and optional
            temperatures [K]. FVM retrieval has already removed the numerical
            distribution scale from these rows.

        Notes
        -----
        Stores heat_prof and heat_duty using the rate/capacitance column
        contract in _store_heat_duty, including net flow heat. Prescribed
        temperature contributes C*dT/dt [W]; C [J/K] is never integrated alone.
        Uses the current stored active kinetic parameters for every profile row.
        """
        q_heat = np.zeros((len(time), 3))  # [W] or [J/K], profile column contract
        merged_params = self.Kinetics.concat_params()[self.mask_params]
        # [native active parameter units], current solve rather than an old iterate
        for ind, row in enumerate(states):
            row = row.copy()  # [state units], documented above
            row[:self.num_distr] *= self.scale  # [-], numerical distribution scale
            q_heat[ind] = self.unit_model(time[ind], row, merged_params,
                                          enrgy_bce=True)
        self._store_heat_duty(time, q_heat)


class SemibatchCryst(MSMPR):
    """ Construct a Semi-batch Crystallizer object
    
    Parameters
    ----------
    target_comp : str, list of strings
        Name of the crystallizing compound(s) from .json file.
    mask_params : list of bool (optional, default = None)
        Binary list of which parameters to exclude from the kinetics
        computation
    method : str
        Choice of the numerical method. Options are: 'moments', '1D-FVM'
    scale : float
        Scaling factor by which crystal size distribution will be
        multiplied.
    controls : dict of dicts (funcs) (optional, default = None)
        Dictionary with keys representing the state (e.g.'Temp')
        which is controlled and the value indicating the function
        to use while computing the varible. Functions are of the form
        f(time) = state_value
    adiabatic : bool (optional, default =True)
        Boolean value indicating whether the heat transfer of
        the crystallization is considered.
    reset_states : bool (optional, default = False)
        Boolean value indicating whether the states should be
        reset before simulation
    basis : str (optional, default = 'mass_conc')
        Options : 'massfrac', 'massconc'
    state_events : lsit of dict(s)
        list of dictionaries, each one containing the specification of a
        state event
    """
    def __init__(self, target_comp, vol_tank=None, mask_params=None,
                 method='1D-FVM', scale=1, controls=None, adiabatic=False,
                 rad_zero=0, reset_states=False, h_conv=1000, vol_ht=None,
                 basis='mass_conc', jac_type=None, num_interp_points=3,
                 state_events=None, param_wrapper=None):

        super().__init__(target_comp, mask_params,
                         method, scale, vol_tank,
                         controls, adiabatic, rad_zero,
                         reset_states,
                         h_conv, vol_ht, basis,
                         jac_type, num_interp_points, state_events,
                         param_wrapper)

    def material_balances(self, time: float, params: "np.ndarray | None",
                          u_inputs: dict, rhos: "Sequence[Sequence[float | None]]",
                          mu_n: np.ndarray, distrib: np.ndarray,
                          mass_conc: np.ndarray, temp: float,
                          temp_ht: "float | None", vol: float,
                          phi_in: np.ndarray) -> tuple:
        """
        Material balances for the semibatch crystallizer.

        Parameters
        ----------
        time : float
            integration time [s].
        params : ndarray or None
            Active kinetic vector, shape (num_active_params,), in mask_params
            order and the native units and transformations of
            CrystKinetics.concat_params(). None retains stored parameters;
            fixed parameters remain unchanged.
        u_inputs : dict
            unit inputs at `time`, holding the inlet volumetric flow
            [m**3/s], volume-specific inlet mu_n [m**n/m**3] for moments
            or distrib [#/m**3/um] for FVM, and liquid mass concentrations
            [kg/m**3].
        rhos : sequence of sequences
            Density pairs, shape (2, 2): tank then inlet, each ordered liquid
            then solid [kg/m**3]. The unused inlet solid density may be None
            for a solid-free feed.
        mu_n : array-like
            crystal size distribution moments [m**n] (total basis).
        distrib : array-like
            distribution state: [#/um] for ``method == '1D-FVM'``, or the
            raw moment vector [um**n] for ``method == 'moments'``.
        mass_conc : array-like
            liquid-phase mass concentrations [kg/m**3].
        temp : float
            liquid temperature [K].
        temp_ht : float or None
            jacket temperature [K]. Unused here, kept for signature
            compatibility.
        vol : float
            liquid volume [m**3].
        phi_in : array-like
            inlet [liquid, solid] volume fractions [-].

        Returns
        -------
        dmaterial_dt : numpy.ndarray
            stacked derivatives with mixed units, in this order:
            ``ddistr_dt`` ([#/um/s] for '1D-FVM', [um**n/s] for 'moments'),
            ``dcomp_dt`` [kg/m**3/s] and ``dvol_dt`` [m**3/s].
        transf : numpy.ndarray
            crystallization mass rate [kg/s].

        Notes
        -----
        Like :class:`BatchCryst` and unlike :class:`MSMPR`, the semibatch
        states are declared on a *total* basis in
        :meth:`_BaseCryst.solve_unit` (``mu_n`` in [um**n], ``distrib`` in
        [#/um]), and the kinetics are evaluated with ``vol=vol_slurry``, so
        ``transf`` is a total rate [kg/s].

        ``dcomp_dt`` is documented as [kg/m**3/s] because the
        ``basis == 'mass_frac'`` rescaling below acts on a copy and never
        reaches the returned array (tracked in issue #47).
        """

        rho_susp, rho_in = rhos

        rho_liq, rho_sol = rho_susp
        rho_in_liq, _ = rho_in

        input_flow = u_inputs['Inlet']['vol_flow']  # [m**3/s]
        input_flow = np.max([eps, input_flow])

        input_conc = u_inputs['Liquid_1']['mass_conc']  # [kg/m**3]

        vol_solid = mu_n[3] * self.Solid_1.kv  # mu_3 is total, not by volume
        vol_slurry = vol + vol_solid

        self.Liquid_1.updatePhase(mass_conc=mass_conc)

        if self.method == 'moments':
            # [um**n/m**3], exact SI-to-micrometre inlet conversion
            input_distrib = u_inputs['Inlet']['mu_n'] * 1e6**np.arange(self.num_distr)
            ddistr_dt, transf = self.method_of_moments(distrib, mass_conc, temp,
                                                       params, rho_sol,
                                                       vol=vol_slurry)

        elif self.method == '1D-FVM':
            input_distrib = u_inputs['Inlet']['distrib'] * self.scale  # scaled [#/m**3/um]
            ddistr_dt, transf = self.fvm_method(distrib, mu_n, mass_conc, temp,
                                                params, rho_sol,
                                                vol=vol_slurry)

        # ---------- Add flow terms
        # Distribution
        flow_distrib = input_flow * input_distrib

        ddistr_dt = ddistr_dt + flow_distrib

        # Liquid phase
        c_tank = mass_conc  # [kg/m**3]

        flow_term = phi_in[0]*input_flow * (
            input_conc - mass_conc * rho_in_liq/rho_liq)  # [kg/s]
        transf_term = transf * (self.kron_jtg - c_tank/rho_liq)  # [kg/s]

        dcomp_dt = 1/vol * (flow_term - transf_term)  # [kg/m**3/s]
        dvol_dt = (phi_in[0] * input_flow * rho_in_liq
                   - transf) / rho_liq  # [m**3/s]

        dliq_dt = np.append(dcomp_dt, dvol_dt)

        if self.basis == 'mass_frac':
            dcomp_dt *= 1 / rho_liq

        dmaterial_dt = np.concatenate((ddistr_dt, dliq_dt))

        return dmaterial_dt, transf  # transf [kg/s]

    def energy_balances(self, time: float, params, cryst_rate: np.ndarray,
                        u_inputs: dict, rhos: list, distrib,
                        mass_conc: np.ndarray, temp: float,
                        temp_ht: "float | None", vol: float,
                        mu_n: np.ndarray, h_in: float) -> "float | tuple":
        """
        Energy balances for the semibatch crystallizer.

        Parameters
        ----------
        time : float
            integration time [s].
        params : array-like
            kinetic parameters. Unused here, kept for signature
            compatibility.
        cryst_rate : numpy.ndarray
            crystallization mass rate [kg/s], as returned by
            :meth:`material_balances`. Unlike :meth:`MSMPR.energy_balances`,
            it is already a total rate, so the source term below needs no
            ``* vol`` factor.
        u_inputs : dict
            unit inputs at `time`, holding the inlet volumetric flow
            [m**3/s].
        rhos : list
            [[liquid, solid] tank densities, [liquid, solid] inlet
            densities], all in [kg/m**3].
        distrib : array-like
            distribution state. Unused here, kept for signature
            compatibility.
        mass_conc : array-like
            liquid-phase mass concentrations [kg/m**3].
        temp : float
            liquid temperature [K].
        temp_ht : float or None
            jacket temperature [K]. ``None`` when the unit carries no
            jacket state.
        vol : float
            liquid volume [m**3].
        mu_n : array-like
            crystal size distribution moments [m**n] (total basis).
        h_in : float
            volumetric enthalpy of the inlet stream [J/m**3].

        Returns
        -------
        float or tuple
            ``dtemp_dt`` [K/s], or the pair (``dtemp_dt``, ``dtht_dt``) [K/s]
            when a jacket state is present.
        """

        rho_susp, rho_in = rhos

        # Input properties
        input_flow = u_inputs['Inlet']['vol_flow']  # [m**3/s]
        input_flow = np.max([eps, input_flow])

        vol_solid = mu_n[3] * self.Solid_1.kv  # mu_3 is total, not by volume
        vol_total = vol + vol_solid  # [m**3], slurry volume

        phi = vol / vol_total  # [-], liquid volume fraction
        phis = [phi, 1 - phi]  # [-], [liq, sol]
        dens_slurry = np.dot(rho_susp, phis)  # [kg/m**3]

        # Suspension properties
        capacitance = self.Slurry.getCp(temp, phis, rho_susp,
                                        times_vliq=True)  # [J/m**3 liquid/K]
        h_sp = self.Slurry.getEnthalpy(temp, phis, rho_susp)  # [J/m**3]

        # Renaming
        dh_cryst = -1.46e4  # [J/kg]
        # dh_cryst = -self.Liquid_1.delta_fus[self.target_ind] / \
        #     self.Liquid_1.mw[self.target_ind] * 1000  # [J/kg]

        # Terms
        dens_in_liq = rho_in[0]  # [kg/m**3]
        dmass_dt = input_flow * dens_in_liq  # [kg/s]

        accum_term = dmass_dt * h_sp/dens_slurry  # [J/s]
        flow_term = input_flow * h_in  # [J/s]

        source_term = dh_cryst * cryst_rate  # [J/s]

        height_liq = vol / (np.pi/4 * self.diam_tank**2)  # [m]
        # [m**2], wetted lateral area plus tank base
        area_ht = np.pi * self.diam_tank * height_liq + self.area_base

        if self.adiabatic:
            ht_term = 0  # [J/s]
        else:
            ht_term = self.u_ht*area_ht*(temp - temp_ht)  # [J/s]

        # Balance inside the tank
        dtemp_dt = (flow_term - source_term - ht_term - accum_term) / \
            capacitance / vol  # [K/s]

        if temp_ht is not None:
            ht_media = self.Utility.get_inputs(time)
            tht_in = ht_media['temp_in']  # [K], Utility inlet temperature
            flow_ht = ht_media['vol_flow']  # [m**3/s]
            cp_ht = self.Utility.cp  # [J/kg/K]
            rho_ht = self.Utility.rho  # [kg/m**3]
            vol_ht = (self.vol_tank / self.vol_offset * DEFAULT_JACKET_VOLUME_RATIO
                      if self.vol_ht is None else self.vol_ht)  # [m**3]

            dtht_dt = flow_ht / vol_ht * (tht_in - temp_ht) - \
                self.u_ht*area_ht*(temp_ht - temp) / rho_ht/vol_ht/cp_ht

            return dtemp_dt, dtht_dt

        else:
            return dtemp_dt
