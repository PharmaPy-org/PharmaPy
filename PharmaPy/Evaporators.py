# -*- coding: utf-8 -*-


from typing import Optional, Sequence, Union

import numpy as np
from PharmaPy.ThermoModule import validate_activity_model
# from autograd import numpy as np
# from autograd import jacobian as jacauto
from PharmaPy.Commons import (mid_fn, trapezoidal_rule, eval_state_events,
                              handle_events, unpack_states, flatten_states,
                              TerminateSimulation)
from PharmaPy._assimulo import IDA, Implicit_Problem
from PharmaPy.Connections import get_inputs, get_inputs_new
from PharmaPy.Streams import LiquidStream, VaporStream
from PharmaPy.Phases import LiquidPhase, VaporPhase, classify_phases
from PharmaPy.Presence import SpeciesPresence

from PharmaPy.Results import DynamicResult
from PharmaPy.Plotting import plot_function

from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from pathlib import Path
import copy

gas_ct = 8.314
eps = np.finfo(float).eps
MAX_LIQUID_VOLUME_FRACTION = 0.95  # [-], legacy drum design reserves 5% vapor headspace


def merge_supercritical(flags, x_liq, y_vap, z_super):
    x = np.zeros(len(x_liq) + sum(flags))
    y = np.zeros_like(x)

    x[~flags] = x_liq

    y[flags] = (1 - y_vap.sum()) * z_super / z_super.sum()
    y[~flags] = y_vap

    return x, y


def _solve_flash(model, seed: np.ndarray, vapor_index: int,
                 residual_tolerance: float) -> np.ndarray:
    """Solve and validate a flash before publishing phase amounts.

    Parameters
    ----------
    model : callable
        Scaled flash residual (material fractions [-], energy divided by
        the configured energy scale for an adiabatic flash).
    seed : numpy.ndarray
        Initial state: fractions [-] and optional temperature [K].
    vapor_index : int
        Position of the vapor fraction; liquid fraction is at index zero.
    residual_tolerance : float
        Positive maximum absolute scaled residual and phase-bound allowance
        [-]. The public default sqrt(float64 epsilon) is a numerical design
        choice consistent with MINPACK's default relative step tolerance.

    Returns
    -------
    numpy.ndarray
        Converged state with only tolerance-sized phase-bound excursions clipped.

    Raises
    ------
    ValueError
        If the residual tolerance is not finite and positive.
    RuntimeError
        If convergence fails, residuals are too large, or phase fractions
        materially violate [0, 1].
    """
    if not np.isfinite(residual_tolerance) or residual_tolerance <= 0:
        raise ValueError("Flash residual_tolerance must be finite and positive")
    solution, info, status, message = fsolve(model, seed, full_output=True)
    return _validate_flash_solution(model, solution, status, message,
                                     vapor_index, residual_tolerance)


def _validate_flash_solution(model, solution: np.ndarray, status: int,
                             message: str, vapor_index: int,
                             residual_tolerance: float) -> np.ndarray:
    """Validate a nonlinear flash result before publishing phase amounts.

    Parameters
    ----------
    model : callable
        Scaled flash residual with dimensionless material/energy components.
    solution : numpy.ndarray
        Candidate fractions [-] and optional temperature [K]. Tolerance-sized
        phase-bound excursions are clipped in place before residual evaluation.
    status : int
        MINPACK convergence status; one indicates convergence.
    message : str
        Native solver diagnostic included in convergence errors.
    vapor_index : int
        Position of the vapor fraction; the liquid fraction is at index zero.
    residual_tolerance : float
        Finite positive scaled residual and phase-bound allowance [-].

    Returns
    -------
    numpy.ndarray
        The validated candidate with phase-bound roundoff clipped.

    Raises
    ------
    RuntimeError
        If convergence, finiteness, phase bounds, or scaled closure fails.
    """
    if status != 1 or not np.all(np.isfinite(solution)):
        raise RuntimeError(f"Flash failed to converge: {message}")
    fractions = solution[[0, vapor_index]]  # [-]
    if np.any(fractions < -residual_tolerance) or np.any(fractions > 1 + residual_tolerance):
        raise RuntimeError("Flash returned phase fractions outside [0, 1]")
    solution[[0, vapor_index]] = np.clip(fractions, 0, 1)  # [-], boundary roundoff only
    residual = model(solution)  # [-], after configured energy scaling
    if not np.all(np.isfinite(residual)) or np.max(np.abs(residual)) > residual_tolerance:
        raise RuntimeError("Flash scaled residual exceeds residual_tolerance")
    return solution


def _flash_states_di(name_species: Sequence[str],
                     adiabatic: bool = False) -> dict:
    """Describe the algebraic unknowns of a steady-state flash.

    Parameters
    ----------
    name_species : sequence of str
        Names of the N species in the solved composition blocks, in
        database order: every feed species for ``IsothermalFlash``, the
        species that are not supercritical for ``AdiabaticFlash``.
    adiabatic : bool, optional
        If True, append the drum temperature, the extra unknown of
        ``AdiabaticFlash``.

    Returns
    -------
    dict
        State metadata in the order of the solved unknown vector: the
        liquid fraction of the feed ``frac_liq`` [-], the liquid mole
        fractions ``x_liq`` [-] and vapor mole fractions ``y_vap`` [-]
        (N each, indexed by species), the vapor fraction of the feed
        ``frac_vap`` [-] and, if ``adiabatic``, the temperature ``temp``
        [K]. Every state is algebraic (``'type': 'alg'``), so a flash has
        2N + 2 unknowns, or 2N + 3 when adiabatic.

    Notes
    -----
    ``SimulationResult`` reads this metadata for its unit-operation
    summary (issue #457).
    """
    names = list(name_species)
    states_di = {
        'frac_liq': {'dim': 1, 'type': 'alg', 'units': ''},
        'x_liq': {'dim': len(names), 'type': 'alg', 'units': '',
                  'index': names},
        'y_vap': {'dim': len(names), 'type': 'alg', 'units': '',
                  'index': names},
        'frac_vap': {'dim': 1, 'type': 'alg', 'units': ''}}
    if adiabatic:
        states_di['temp'] = {'dim': 1, 'type': 'alg', 'units': 'K'}
    return states_di


def _model_liquid_volume(liquid) -> float:
    """Return the liquid volume the evaporator models assign to a charge.

    Parameters
    ----------
    liquid : LiquidPhase
        Evaporator liquid holdup with ``moles`` [mol], ``mole_frac`` [-]
        and ``temp`` [K].

    Returns
    -------
    float
        Liquid volume [m**3] on the models' own basis: the moles over the
        ideal-mixing molar density of their composition,
        ``moles / getDensity(mole_frac, temp, basis='mole') / 1000``
        (1000 L/m**3), as ``material_balances`` evaluates it.

    Notes
    -----
    The evaporator states are moles and mole fractions, closed by ideal
    mixing. A liquid on the ideal-mixing basis (fractions, or concentrations
    with ``name_solv``) already stores that volume, which is returned
    unchanged so its results stay bit-identical. A liquid whose
    concentrations are authoritative and do not close ideal mixing
    (``LiquidPhase.concentration_basis``, issue #439) keeps its moles but
    occupies its ideal-mixing volume in the evaporator model; seeding the
    stored volume instead would leave an inconsistent initial volume and
    energy balance.
    """
    if not getattr(liquid, 'concentration_basis', False):
        return liquid.vol  # [m**3]

    molar_density = liquid.getDensity(mole_frac=liquid.mole_frac,
                                      temp=liquid.temp,
                                      basis='mole')  # [mol/L]
    return liquid.moles / molar_density / 1000  # [m**3]


class _StructuralPresenceMixin:
    """Pass a structural species-presence mask to the property kernels.

    The evaporators and flash units carry every species as a solver state
    or unknown, so an absent species picks up round-off fractions. Their
    balances therefore weight the property kernels (``weights=``) with
    ``_presence``, a ``PharmaPy.Presence.SpeciesPresence`` record built from
    the charge (or flash feed) and from the feed rows that flow, never from
    states: an absent species needs no property data (issue #425). An
    absent species also gets a unit VLE ratio in place of its data-dependent
    one, so its closures reduce to exactly zero fractions whatever its
    data. Without a record (phases installed by hand, as some tests do) the
    kernels keep the fraction-based rule.
    """

    def _start_presence(self, num_species: int,
                        *amounts: np.ndarray) -> None:
        """Start a new presence record from the given exact inputs.

        Parameters
        ----------
        num_species : int
            Number of species of the unit's states [-].
        *amounts : array-like
            Charge or feed fractions [-] or amounts, shape
            ``(num_species,)`` or ``(num_rows, num_species)``; a species
            with a nonzero entry is present.
        """
        self._presence = SpeciesPresence(num_species)
        self._presence.record(*amounts)

    @staticmethod
    def _feed_weights(mole_flow: Union[float, np.ndarray],
                      mole_frac: np.ndarray) -> np.ndarray:
        """Flag the feed entries that actually flow.

        Parameters
        ----------
        mole_flow : float or array-like
            Effective feed flow [mol/s], scalar or shape ``(num_rows,)``,
            after any shut-off.
        mole_frac : array-like
            Feed mole fractions [-], shape ``(num_species,)`` or
            ``(num_rows, num_species)``.

        Returns
        -------
        numpy.ndarray
            1.0 where both the row's flow and the species' fraction are
            exactly nonzero, 0.0 otherwise [-]: shape ``(num_species,)``
            for a scalar flow, ``(num_rows, num_species)`` for a flow
            profile. A species in a row without flow is not fed.

        Notes
        -----
        For this presence bookkeeping, a flow with one entry (a 0-d or
        ``(1,)`` array) paired with one composition, shape
        ``(num_species,)``, is a scalar flow; only a flow with one entry per
        composition row is a profile. The evaporators do not support
        ``(1,)``-shaped feed flows yet: their initialization and energy
        balances fail on them (issue #520).
        """
        mole_frac = np.asarray(mole_frac, dtype=float)  # [-]
        flowing = np.asarray(mole_flow) != 0  # [-], per row
        if flowing.size == 1 and mole_frac.ndim == 1:
            flowing = flowing.reshape(())  # [-], a scalar flow
        elif flowing.ndim > 0:
            flowing = flowing[:, np.newaxis]  # [-], one flag per row

        return ((mole_frac != 0) & flowing).astype(float)

    def _record_feed(self, mole_flow: Union[float, np.ndarray],
                     mole_frac: np.ndarray,
                     time: Optional[Union[float, np.ndarray]] = None
                     ) -> np.ndarray:
        """Add the species of the flowing feed rows to the record.

        Parameters
        ----------
        mole_flow : float or array-like
            Effective feed flow [mol/s], scalar or shape ``(num_rows,)``,
            after any shut-off.
        mole_frac : array-like
            Feed mole fractions [-], shape ``(num_species,)`` or
            ``(num_rows, num_species)``.
        time : float or array-like, optional
            Absolute evaluation time [s] of the feed, scalar or one per row;
            it becomes the species' latch time if earlier than a recorded
            one. A single feed row evaluated for several times (controls
            that ignore the time) latches at the earliest of them. None
            records the species from the start.

        Returns
        -------
        numpy.ndarray
            The feed weights of ``_feed_weights`` [-]; a species becomes
            present only where they are nonzero, and stays present
            (latched). Without a record nothing is recorded.
        """
        fed = self._feed_weights(mole_flow, mole_frac)  # [-]
        presence = getattr(self, '_presence', None)
        if presence is not None:
            row_time = time
            if np.ndim(time) > 0 and np.ndim(fed) == 1:
                # One feed row for a time profile (controls that ignore the
                # time): it applies at every time, so it latches at the first.
                row_time = float(np.min(time))  # [s]
            presence.record(fed, time=row_time)

        return fed

    @staticmethod
    def _phase_state(phase) -> dict:
        """Copy the complete state of a phase, stream or unit object.

        Parameters
        ----------
        phase : LiquidPhase, VaporPhase, a stream, or an evaporator
            Object whose attributes a solve may change.

        Returns
        -------
        dict
            The object's attributes, with arrays [units as stored], lists
            and dicts copied so that later in-place changes do not reach
            them; other values (scalars, strings, references to other
            objects such as controllers) are kept as they are.

        Notes
        -----
        Phases hold their property data, composition, amounts and derived
        fields (mass fractions, mean molar mass, flows) as plain
        attributes, so restoring them restores a consistent phase. A unit's
        results are replaced or appended to (``profiles_runs``), never
        changed inside, so one level of copying suffices for it too.
        """
        return {name: (value.copy()
                       if isinstance(value, (np.ndarray, list, dict))
                       else value)
                for name, value in vars(phase).items()}

    def _solve_snapshot(self) -> dict:
        """Record what a failed evaporator solve must leave unchanged.

        Returns
        -------
        dict
            The unit's own attributes (``'unit'``, as ``_phase_state``
            copies them: elapsed time, terminal states, profiles, results,
            duties, outlet, inlet mode, phase references), a copy of the
            presence record (``'presence'``, None without one), and the
            complete state of ``Liquid_1`` and ``Vapor_1`` (``'states'``),
            which ``unit_model``, ``init_unit`` and ``retrieve_results``
            overwrite.
        """
        presence = getattr(self, '_presence', None)
        phases = (self.Liquid_1, self.Vapor_1)
        return {
            'unit': self._phase_state(self),
            'presence': None if presence is None else copy.deepcopy(presence),
            'states': [(phase, self._phase_state(phase)) for phase in phases]}

    def _restore_solve_snapshot(self, snapshot: dict) -> None:
        """Undo what a failed solve changed in the unit and its phases.

        Parameters
        ----------
        snapshot : dict
            Record returned by ``_solve_snapshot`` before the solve.

        Notes
        -----
        A failed solve keeps no results: whether the initialization, the
        integration or the publication of the results raised, the unit's
        attributes are restored in place, so a partly published segment
        (elapsed time, terminal states, profiles, results, duties, outlet)
        is withdrawn. The species that only its
        attempted integration latched (a feed evaluated past the failure,
        for example) must not stay present, while species latched by
        earlier, successful solves are in the snapshot and remain. The
        attached phases are restored too, attribute by attribute and in
        place: ``unit_model`` leaves the last iterate's temperature and
        fractions in them, including round-off of absent species, and
        ``init_unit`` updates their amounts and derived fields, so an
        initialization (the next attempt, or the presence rebuilt from the
        charge) would otherwise start from that iterate, and the phase
        would be inconsistent (fractions of one state, mass fractions of
        another).
        """
        vars(self).clear()
        vars(self).update(snapshot['unit'])
        self._presence = snapshot['presence']
        for phase, state in snapshot['states']:
            # In place, so that every reference to the phase sees it.
            vars(phase).clear()
            vars(phase).update(state)

    def _absent_unit_ratios(self, k_vals: np.ndarray,
                            time: Optional[float] = None) -> np.ndarray:
        """Replace the VLE ratios of absent species with one.

        Parameters
        ----------
        k_vals : numpy.ndarray
            VLE ratios ``y_i/x_i`` [-], shape ``(num_species,)``; those of
            absent species may be NaN (no Henry data) or data-dependent.
        time : float, optional
            Absolute evaluation time [s]: a species recorded only later
            counts as absent. None counts every recorded species.

        Returns
        -------
        numpy.ndarray
            ``k_vals`` with absent species at exactly 1 [-]; unchanged
            without a presence record.

        Notes
        -----
        An absent species has zero feed and zero holdup, so its component
        balance and its equilibrium ``y_i - K_i x_i = 0`` must give
        ``x_i = y_i = 0``. Any finite positive ratio does: with
        ``L + V = 1`` the flash pair ``-L x_i - V y_i = 0``,
        ``y_i - K_i x_i = 0`` has determinant ``-(L + V K_i)``, never
        zero. The unit ratio, the identity ``y_i = x_i``, makes that
        determinant -1 for every phase split and the result independent of
        the species' data, which an absent species may lack.
        """
        presence = getattr(self, '_presence', None)
        if presence is None:
            return k_vals

        present = (presence.mask if time is None
                   else presence.mask_at(time))  # [-]
        return np.where(present, k_vals, 1.0)  # [-]

    def _presence_weights(self, fractions: np.ndarray,
                          used: Optional[np.ndarray] = None,
                          time: Optional[Union[float, np.ndarray]] = None
                          ) -> Optional[np.ndarray]:
        """Return the kernel weights for some fractions.

        Parameters
        ----------
        fractions : array-like
            Mole fractions [-], shape ``(num_species,)`` or
            ``(num_rows, num_species)``.
        used : array-like of bool, optional
            Species whose results the caller uses, shape
            ``(num_species,)``.
        time : float or array-like, optional
            Absolute time [s] of the fractions, scalar or one per row: only
            species recorded at or before it count, so that a replay of
            earlier rows uses the presence of their time. None counts every
            recorded species.

        Returns
        -------
        numpy.ndarray or None
            0/1 presence weights [-] with the shape of ``fractions``. Without
            a presence record: ``None``, or ``fractions * used`` when
            ``used`` is given, the fraction-based rule.
        """
        presence = getattr(self, '_presence', None)
        if presence is None:
            return None if used is None else fractions * used

        return presence.weights(fractions, used, time=time)

    def _project_absent(self, fractions: np.ndarray,
                        time: Optional[Union[float, np.ndarray]] = None
                        ) -> np.ndarray:
        """Zero the round-off fractions of absent species.

        Parameters
        ----------
        fractions : array-like
            Mole fractions [-] from solver states, shape ``(num_species,)``
            or ``(num_rows, num_species)``.
        time : float or array-like, optional
            Absolute time [s] of the fractions, scalar or one per row (see
            ``_presence_weights``).

        Returns
        -------
        numpy.ndarray
            ``fractions`` with absent species at exactly zero, used to build
            the published phases, which would otherwise need data for those
            species; unchanged without a presence record.
        """
        presence = getattr(self, '_presence', None)
        if presence is None:
            return np.asarray(fractions, dtype=float)

        return presence.project(fractions, time=time)


class IsothermalFlash(_StructuralPresenceMixin):
        
    """
    Create an isothermal flash object. The model run by the instances of
    this class is steady state isothermal flash given T and P. This
    model is often used to determine the thermodynamic state of a given
    stream or phase (only liquid, vapor-liquid, only vapor)

    Parameters
    ----------
    temp_drum : float, optional
        equipment temperature [K]. If None, temperature will be taken from
        the current Inlet object aggregated to the flash (from
        PharmaPy.Phases or PharmaPy.Streams modules), every time an inlet
        is assigned and every time the flash is solved. The default is
        None.
    pres_drum : float, optional
        equipment pressure [Pa]. If None, pressure will be taken from
        the current Inlet object, as for ``temp_drum``. The default is
        None.
    gamma_method : str, optional
        one of 'ideal', 'UNIFAC' or 'UNIQUAC'. If 'UNIFAC' or 'UNIQUAC' is
        passed, the pure-component .json file must have the interaction
        parameters of the model. The default is 'ideal'.

    Returns
    -------
    None.

    """

    def __init__(self, temp_drum=None, pres_drum=None,
                 gamma_method: str = 'ideal') -> None:
        """Configure an isothermal flash with an explicit activity model.

        Parameters
        ----------
        temp_drum : float, optional
            Drum temperature [K]; None uses the current inlet temperature.
            Kept as ``temp_drum``; the effective drum temperature is
            ``temp``.
        pres_drum : float, optional
            Drum pressure [Pa]; None uses the current inlet pressure. Kept
            as ``pres_drum``; the effective drum pressure is ``pres``.
        gamma_method : {'ideal', 'UNIFAC', 'UNIQUAC'}, optional
            Liquid activity model, with parameters supplied by the inlet.

        Raises
        ------
        ValueError
            If ``gamma_method`` is not a supported selector.

        Notes
        -----
        The flash is a steady-state stream unit, so it declares the
        flowsheet destination contract of a continuous unit (issue #445):
        ``oper_mode = 'Continuous'``, ``is_continuous = True`` and
        ``names_states_in``, the stream fields it reads from its ``Inlet``:
        mole fractions [-], molar flow [mol/s], and the temperature [K] and
        pressure [Pa] used when ``temp_drum`` or ``pres_drum`` is None. It
        reads them from the stored state of the ``Inlet`` object, so a
        connected upstream time profile is not resolved: the flash
        separates the upstream outlet's final state. It publishes
        ``LiquidOut`` and ``VaporOut`` but no ``Outlet`` or time profile,
        so it cannot be a flowsheet source.

        The user's drum settings stay in ``temp_drum`` [K] and
        ``pres_drum`` [Pa]; the effective drum conditions ``temp`` [K] and
        ``pres`` [Pa] are derived from them and the current ``Inlet`` on
        every ``Inlet`` assignment and at the start of every
        ``solve_unit`` (issue #452).
        """
        validate_activity_model(gamma_method, param_name='gamma_method')

        self.temp_drum = temp_drum  # [K], or None to follow the inlet
        self.pres_drum = pres_drum  # [Pa], or None to follow the inlet
        self._Inlet = None
        self._update_drum_conditions()
        self.gamma_method = gamma_method

        self.oper_mode = 'Continuous'
        self.is_continuous = True
        self.names_states_in = ['mole_frac', 'mole_flow', 'temp', 'pres']

    def flatten_states(self) -> None:
        """Do nothing: the steady-state flash keeps no state profiles.

        Notes
        -----
        ``SimulationExec.SolveFlowsheet`` calls ``flatten_states`` on every
        solved unit outside ``PharmaPy.Containers``.
        """

    def get_reported_outlets(self) -> dict:
        """Return the flash products that the stream table reports.

        Returns
        -------
        dict of str to LiquidStream/VaporStream or LiquidPhase/VaporPhase
            ``{'liquid': LiquidOut, 'vapor': VaporOut}``: the liquid and
            vapor products, as flows [mol/s] for a stream ``Inlet`` or as
            amounts [mol] for a phase ``Inlet``, at the drum temperature
            [K] and pressure [Pa].

        Raises
        ------
        AttributeError
            If the flash has not been solved, so it has no ``LiquidOut``.

        Notes
        -----
        ``SimulationResult.GetStreamTable`` reads this hook before
        ``Outlet`` and reports the products as the rows
        ``'Outlet_liquid'`` and ``'Outlet_vapor'`` (issue #451). The
        IsothermalFlash deliberately has no ``Outlet``: a ``Connection`` hands a
        source's ``Outlet`` to its successor, and a flash cannot be a
        flowsheet source (issue #445).
        """
        return {'liquid': self.LiquidOut, 'vapor': self.VaporOut}

    def _update_drum_conditions(self) -> None:
        """Derive the effective drum temperature and pressure.

        Sets ``temp`` [K] to ``temp_drum`` when the user set it, otherwise
        to the current ``Inlet`` temperature, and ``pres`` [Pa] likewise
        from ``pres_drum`` or the ``Inlet`` pressure. Without an ``Inlet``,
        an unset value stays None.

        Notes
        -----
        The user's settings are never overwritten, so a re-assigned inlet,
        or an inlet changed in place before a new solve, is flashed at its
        own conditions. Earlier releases stored the first inlet's values
        as the drum settings and flashed every later inlet at them
        (issue #452).
        """
        inlet = self._Inlet
        temp_inlet = None if inlet is None else inlet.temp  # [K]
        pres_inlet = None if inlet is None else inlet.pres  # [Pa]

        self.temp = (self.temp_drum if self.temp_drum is not None
                     else temp_inlet)  # [K]
        self.pres = (self.pres_drum if self.pres_drum is not None
                     else pres_inlet)  # [Pa]

    @property
    def Inlet(self):
        """Feed of the flash (``LiquidPhase`` or ``LiquidStream``).

        Returns
        -------
        LiquidPhase or LiquidStream or None
            The assigned feed, or None before one is assigned.
        """
        return self._Inlet

    @Inlet.setter
    def Inlet(self, instance):
        """Attach the feed and derive the effective drum conditions.

        Parameters
        ----------
        instance : LiquidPhase or LiquidStream
            Feed with mole fractions [-], temperature [K], pressure [Pa],
            and inventory [mol] or flow [mol/s].

        Notes
        -----
        ``temp`` [K] and ``pres`` [Pa] are derived again from ``temp_drum``
        and ``pres_drum`` and this feed (see ``_update_drum_conditions``).
        ``states_di`` describes the flash unknowns for this feed's species
        (see ``_flash_states_di``).
        """
        self._Inlet = instance

        self._update_drum_conditions()

        self.num_comp = self._Inlet.num_species
        self.states_di = _flash_states_di(self._Inlet.name_species)

        self.k_func = self.Inlet.getKeqVLE

        if self.Inlet.__module__ == 'PharmaPy.Streams':
            self.in_flow = self.Inlet.mole_flow
        else:
            self.in_flow = self.Inlet.moles

    def material_balance(self, liq: float, vap: float, x_i: np.ndarray,
                         y_i: np.ndarray, z_i: np.ndarray) -> np.ndarray:
        """Evaluate the material residuals of the isothermal flash.

        Parameters
        ----------
        liq, vap : float
            Liquid and vapor fractions of the feed [-].
        x_i, y_i : numpy.ndarray
            Liquid and vapor mole fractions [-], shape (num_species,).
        z_i : numpy.ndarray
            Feed mole fractions [-], shape (num_species,).

        Returns
        -------
        numpy.ndarray
            Global balance, component balances, equilibria and the
            vapor-flow complementarity residual [-], shape
            (2 * num_species + 2,).

        Raises
        ------
        MissingPropertyError
            If a species of the feed lacks the VLE data its ratio needs at
            the drum temperature; species absent from the feed need none.

        Notes
        -----
        The ratios are evaluated at the drum conditions with the activity
        coefficients of the feed composition, as before. Species absent
        from the feed (structural presence of ``solve_unit``) get a unit
        ratio (see ``_absent_unit_ratios``), so their rows give exactly
        ``x_i = y_i = 0`` whatever their data.
        """
        k_i = self.k_func(
            self.temp, self.pres, gamma_model=self.gamma_method,
            weights=self._presence_weights(self.Inlet.mole_frac))  # [-]
        k_i = self._absent_unit_ratios(k_i)  # [-]

        global_bce = 1 - liq - vap
        comp_bces = z_i - liq*x_i - vap*y_i
        equilibria = y_i - k_i * x_i

        diff_frac = np.sum(x_i - y_i)
        args_mid = np.array([vap, diff_frac, vap - 1])
        vap_flow = mid_fn(args_mid)

        balance = np.concatenate(
            (np.array([global_bce]),
             comp_bces, equilibria,
             np.array([vap_flow]))
        )

        return balance

    def energy_balance(self, liq: float, vap: float, x_i: np.ndarray,
                       y_i: np.ndarray, z_i: np.ndarray) -> float:
        """Return the heat that the isothermal flash requires.

        Parameters
        ----------
        liq, vap : float
            Liquid and vapor fractions of the feed [-].
        x_i, y_i : numpy.ndarray
            Liquid and vapor mole fractions [-], shape (num_species,).
        z_i : numpy.ndarray
            Feed mole fractions [-]; unused, the feed enthalpy uses the
            ``Inlet`` composition.

        Returns
        -------
        float
            Heat duty [W] for a stream ([J] for a batch inventory), with the
            liquid product as the enthalpy reference at the drum
            temperature.

        Raises
        ------
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``. If a
            species of the feed lacks the heat-capacity or latent-heat data
            of the vapor or feed enthalpy; species absent from the feed need
            none.

        Notes
        -----
        The vapor enthalpy is weighted with the structural presence of
        ``solve_unit``: species absent from the feed need no data.
        """
        liq_flow = liq * self.in_flow  # mol/s
        vap_flow = vap * self.in_flow  # mol/s

        tref = self.temp

        h_liq = 0
        h_vap = self.VaporOut.getEnthalpy(
            self.temp, temp_ref=tref, mole_frac=y_i, basis='mole',
            weights=self._presence_weights(y_i))  # [J/mol]

        h_in = self.Inlet.getEnthalpy(temp_ref=tref, basis='mole')

        heat_duty = liq_flow * h_liq + vap_flow * h_vap - self.in_flow * h_in

        return heat_duty  # W

    def unit_model(self, states, material=True):
        liq_pr = states[0]
        vap_pr = states[-1]

        fracs = states[1:-1]
        x_i = fracs[:self.num_comp]
        y_i = fracs[self.num_comp:]
        z_i = self.Inlet.mole_frac

        if material:
            balance = self.material_balance(liq_pr, vap_pr, x_i, y_i, z_i)
        else:
            balance = self.energy_balance(liq_pr, vap_pr, x_i, y_i, z_i)

        return balance

    def solve_unit(self, residual_tolerance: float = np.sqrt(eps)) -> tuple:
        """Solve the isothermal flash and publish validated phase amounts.

        Parameters
        ----------
        residual_tolerance : float, optional
            Absolute scaled residual and phase-bound allowance [-]. Default
            sqrt(float64 epsilon) matches MINPACK's relative step scale.

        Returns
        -------
        solution : numpy.ndarray
            Liquid fraction, species liquid/vapor fractions, vapor fraction [-].
        heat_bce : float
            Required heat [W] for a stream, or energy [J] for a batch inventory.

        Raises
        ------
        RuntimeError
            If the solve does not converge or fails residual/phase-bound checks.
        ValueError
            If residual_tolerance is not finite and positive.
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``. If a
            species of the feed (structural presence) lacks the VLE data
            (Antoine ``p_vap``, or ``henry_constant`` above its critical
            temperature), enthalpy data or density data that the flash and
            its published products need; species absent from the feed need
            none.

        Notes
        -----
        The effective drum temperature ``temp`` [K] and pressure ``pres``
        [Pa] are derived from ``temp_drum``, ``pres_drum`` and the current
        ``Inlet`` before solving, so an inlet changed in place since its
        assignment is flashed at its current conditions.

        A species with zero feed fraction is absent (structural presence,
        issue #425): it needs no property data, its VLE ratio is replaced by
        one, and the published ``LiquidOut`` and ``VaporOut`` hold it at
        exactly zero instead of the solver's round-off.
        """
        self._update_drum_conditions()
        self._start_presence(self.num_comp, self.Inlet.mole_frac)

        # Set seeds
        l_seed = 0.5
        v_seed = 0.5

        x_seed = np.ones(self.num_comp) * 1 / self.num_comp
        y_seed = x_seed

        seed = np.concatenate(([l_seed], x_seed, y_seed, [v_seed]))

        # Solve nonlinear system
        solution = _solve_flash(self.unit_model, seed, -1, residual_tolerance)

        # Retrieve solution
        liq_flash = solution[0] * self.in_flow
        vap_flash = solution[-1] * self.in_flow

        fracs = solution[1:2*self.num_comp + 1]
        x_flash = self._project_absent(fracs[:self.num_comp])  # [-]
        y_flash = self._project_absent(fracs[self.num_comp:])  # [-]

        path = self.Inlet.path_data

        self.result = {'phi': solution[[0, -1]],
                       'x_out': x_flash, 'y_out': y_flash}

        # Store in objects
        if self.Inlet.__module__ == 'PharmaPy.Streams':
            self.LiquidOut = LiquidStream(path, mole_frac=x_flash,
                                          mole_flow=liq_flash,
                                          temp=self.temp, pres=self.pres)
            self.VaporOut = VaporStream(path, mole_frac=y_flash,
                                        mole_flow=vap_flash,
                                        temp=self.temp, pres=self.pres)
        else:
            self.LiquidOut = LiquidPhase(path, mole_frac=x_flash,
                                         moles=liq_flash,
                                         temp=self.temp, pres=self.pres)
            self.VaporOut = VaporPhase(path, mole_frac=y_flash,
                                       moles=vap_flash,
                                       temp=self.temp, pres=self.pres)

        heat_bce = self.unit_model(solution, material=False)

        return solution, heat_bce


class AdiabaticFlash(_StructuralPresenceMixin):
    """
    Create an adiabatic flash object.

    Parameters
    ----------
    pres_drum : float
        tank pressure [Pa].
    div_energybce : float, optional
        Constant by which the energy balance equation is divided.
        It helps improving the conditioning of the numerical solver.
        The default is 1e3.
    gamma_method : str, optional
        one of 'ideal', 'UNIFAC' or 'UNIQUAC'. If 'UNIFAC' or 'UNIQUAC' is
        passed, the pure-component .json property file must have required
        parameters for the activity coefficient model. 
    mult_midfun : float, optional
        Multiplier [-] of the phase-fraction closure residual; unity leaves
        the residual unscaled.
    seed_basedon_input : bool, optional
        Seed compositions from the inlet when True; defaults to False.

    Returns
    -------
    None.

    """
    def __init__(self, pres_drum, div_energybce=1e3,
                 gamma_method: str = 'ideal', mult_midfun=1,
                 seed_basedon_input=False) -> None:
        """Configure an adiabatic flash and its residual scaling.

        Parameters
        ----------
        pres_drum : float
            Drum pressure [Pa].
        div_energybce : float, optional
            Numerical divisor [-] applied to the molar energy residual
            [J/mol]; the existing default 1e3 reduces its numerical magnitude.
        gamma_method : {'ideal', 'UNIFAC', 'UNIQUAC'}, optional
            Liquid activity model, with parameters supplied by the inlet.
        mult_midfun : float, optional
            Dimensionless multiplier [-] of the phase-fraction closure
            residual; the default unity leaves it unscaled.
        seed_basedon_input : bool, optional
            If True, seed the solve from the inlet vapor-liquid split;
            otherwise use the default phase-fraction seed.

        Raises
        ------
        ValueError
            If ``gamma_method`` is not a supported selector.

        Notes
        -----
        The flash is a steady-state stream unit, so it declares the
        flowsheet destination contract of a continuous unit (issue #445):
        ``oper_mode = 'Continuous'``, ``is_continuous = True`` and
        ``names_states_in``, the stream fields it reads from its ``Inlet``:
        mole fractions [-], molar flow [mol/s] and temperature [K]; the
        drum pressure is ``pres_drum``. It reads them from the stored state
        of the ``Inlet`` object, so a connected upstream time profile is
        not resolved: the flash separates the upstream outlet's final
        state. It publishes ``LiquidOut`` and ``VaporOut`` but no
        ``Outlet`` or time profile, so it cannot be a flowsheet source.
        """
        validate_activity_model(gamma_method, param_name='gamma_method')

        self.pres = pres_drum  # [Pa]

        self.div_energybce = div_energybce  # [-]
        self.mult_midfun = mult_midfun  # [-]
        self.seed_basedon_input = seed_basedon_input

        self._Inlet = None

        self.gamma_method = gamma_method

        self.oper_mode = 'Continuous'
        self.is_continuous = True
        self.names_states_in = ['mole_frac', 'mole_flow', 'temp']

    def flatten_states(self) -> None:
        """Do nothing: the steady-state flash keeps no state profiles.

        Notes
        -----
        ``SimulationExec.SolveFlowsheet`` calls ``flatten_states`` on every
        solved unit outside ``PharmaPy.Containers``.
        """

    def get_reported_outlets(self) -> dict:
        """Return the flash products that the stream table reports.

        Returns
        -------
        dict of str to LiquidStream/VaporStream or LiquidPhase/VaporPhase
            ``{'liquid': LiquidOut, 'vapor': VaporOut}``: the liquid and
            vapor products, as flows [mol/s] for a stream ``Inlet`` or as
            amounts [mol] for a phase ``Inlet``, at the drum temperature
            [K] and pressure [Pa].

        Raises
        ------
        AttributeError
            If the flash has not been solved, so it has no ``LiquidOut``.

        Notes
        -----
        ``SimulationResult.GetStreamTable`` reads this hook before
        ``Outlet`` and reports the products as the rows
        ``'Outlet_liquid'`` and ``'Outlet_vapor'`` (issue #451). The
        AdiabaticFlash deliberately has no ``Outlet``: a ``Connection`` hands a
        source's ``Outlet`` to its successor, and a flash cannot be a
        flowsheet source (issue #445).
        """
        return {'liquid': self.LiquidOut, 'vapor': self.VaporOut}

    @property
    def Inlet(self):
        """Feed of the flash (``LiquidPhase`` or ``LiquidStream``), or None."""
        return self._Inlet

    @Inlet.setter
    def Inlet(self, instance: LiquidPhase) -> None:
        """Attach the feed and prepare a zero-flow vapor property object.

        Parameters
        ----------
        instance : LiquidPhase or LiquidStream
            Feed with mole fractions [-], temperature [K], pressure [Pa],
            and inventory [mol] or flow [mol/s]. The vapor placeholder uses
            its species order and composition until the flash is solved.

        Notes
        -----
        ``states_di`` describes the flash unknowns for this feed: the
        species that are not supercritical at the feed temperature, which
        the solve keeps in its composition blocks, and the drum
        temperature (see ``_active_states_di``). ``num_comp`` is the number
        of feed species; a solve never changes it (issue #319).

        The zero-flow vapor object is both the ``VaporOut`` placeholder
        and ``_vapor_props``, the property object whose composition the
        energy balance updates while solving. A solve publishes a new
        ``VaporOut``, so a later solve leaves the published products of
        earlier solves unchanged.
        """
        self._Inlet = instance
        self.num_comp = self._Inlet.num_species  # [-], feed species
        self.states_di = self._active_states_di(self._supercritical_mask())

        path = self.Inlet.path_data

        self.VaporOut = VaporStream(path, pres=self.pres,
                                    mole_frac=instance.mole_frac,
                                    check_input=False)
        self._vapor_props = self.VaporOut

        self.k_func = self.Inlet.getKeqVLE

        if self.Inlet.__module__ == 'PharmaPy.Streams':
            self.in_flow = self.Inlet.mole_flow
        else:
            self.in_flow = self.Inlet.moles

    def _supercritical_mask(self) -> np.ndarray:
        """Flag the feed species the solve removes as supercritical.

        Returns
        -------
        numpy.ndarray of bool, shape (num_species,)
            True for each species of the feed (nonzero fraction) whose
            critical temperature [K] is below the current ``Inlet``
            temperature [K]; ``solve_unit`` removes these species from the
            liquid and vapor composition unknowns.

        Notes
        -----
        A species absent from the feed is never removed: its vapor share
        would be ``0 / 0`` in ``merge_supercritical`` when no fed species is
        supercritical. It stays among the unknowns with a unit VLE ratio and
        zero weights, so it needs no data (structural presence, issue #425).
        """
        present = SpeciesPresence(self.num_comp).record(
            self.Inlet.mole_frac)  # [-], species of the current feed
        return (self.Inlet.temp > self.Inlet.t_crit) & present

    def _active_states_di(self, is_supercritic: np.ndarray) -> dict:
        """Describe the unknowns of the flash for its active species.

        Parameters
        ----------
        is_supercritic : numpy.ndarray of bool, shape (num_species,)
            Supercritical flags of the feed species, as returned by
            ``_supercritical_mask``.

        Returns
        -------
        dict
            ``_flash_states_di`` for the species that are not
            supercritical, with the drum temperature [K]: 2N + 3
            algebraic unknowns for N such species.
        """
        active = [name for name, flag in zip(self.Inlet.name_species,
                                             is_supercritic) if not flag]
        return _flash_states_di(active, adiabatic=True)

    def material_balances(self, liq: float, vap: float, x_i: np.ndarray,
                          y_i: np.ndarray, z_i: np.ndarray,
                          temp: float) -> np.ndarray:
        """Evaluate the material residuals of the adiabatic flash.

        Parameters
        ----------
        liq, vap : float
            Liquid and vapor fractions of the feed [-].
        x_i, y_i : numpy.ndarray
            Liquid and vapor mole fractions of every feed species [-],
            shape (num_species,).
        z_i : numpy.ndarray
            Feed mole fractions [-], shape (num_species,).
        temp : float
            Drum temperature [K].

        Returns
        -------
        numpy.ndarray
            Global balance, component balances and equilibria of the species
            that are not supercritical, and the vapor-flow complementarity
            residual [-].

        Raises
        ------
        MissingPropertyError
            If a species of the feed (structural presence) that is not
            supercritical lacks the VLE data its ratio needs. Species absent
            from the feed need none, nor do supercritical ones, whose
            equilibria are removed (as in the batch ``Evaporator``).

        Notes
        -----
        Species absent from the feed get a unit ratio (see
        ``_absent_unit_ratios``), so their rows give exactly
        ``x_i = y_i = 0`` whatever their data.
        """
        k_i = self.Inlet.getKeqVLE(
            temp, self.pres, x_i, y_i, gamma_model=self.gamma_method,
            weights=self._presence_weights(
                x_i, ~self.is_supercritic))  # [-]
        k_i = self._absent_unit_ratios(k_i)  # [-]

        # Material balances
        global_bce = 1 - liq - vap
        comp_bces = z_i - liq*x_i - vap*y_i
        equilibria = y_i - k_i * x_i

        diff_frac = np.sum(x_i - y_i)
        args_mid = np.array([vap, diff_frac, vap - 1])
        vap_flow = mid_fn(args_mid) * self.mult_midfun

        if any(self.is_supercritic):
            comp_bces = comp_bces[~self.is_supercritic]
            equilibria = equilibria[~self.is_supercritic]

        material = np.concatenate((
            np.array([global_bce]),
            comp_bces, equilibria,
            np.array([vap_flow]))
        )

        return material

    def energy_balance(self, liq, vap, x_i, y_i, z_i, temp):
        """Evaluate the scaled adiabatic energy residual.

        Parameters
        ----------
        liq, vap : float
            Liquid and vapor fractions of the feed [-].
        x_i, y_i : numpy.ndarray
            Liquid and vapor mole fractions of all feed species [-].
        z_i : numpy.ndarray
            Feed mole fractions [-].
        temp : float
            Drum temperature [K], also the enthalpy reference.

        Returns
        -------
        numpy.ndarray, shape (1,)
            Molar energy residual [J/mol] divided by ``div_energybce`` [-].

        Raises
        ------
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``. If a
            species of the feed lacks the heat-capacity or latent-heat data
            of the vapor or feed enthalpy; species absent from the feed need
            none.

        Notes
        -----
        The vapor enthalpy is evaluated with ``_vapor_props``, whose
        composition is set to ``y_i``; the published ``VaporOut`` of an
        earlier solve is not modified (issue #319).
        """
        self._vapor_props.mole_frac = y_i

        # Energy balance
        h_liq = 0  # reference state
        h_vap = self._vapor_props.getEnthalpy(
            temp, temp_ref=temp, basis='mole',
            weights=self._presence_weights(y_i))  # [J/mol]
        h_feed = self.Inlet.getEnthalpy(temp_ref=temp, basis='mole')

        energy_bce = liq * h_liq + vap * h_vap - h_feed
        energy_bce = np.atleast_1d(energy_bce / self.div_energybce)

        return energy_bce

    def unit_model(self, states):
        """Evaluate the flash residuals for the active species.

        Parameters
        ----------
        states : numpy.ndarray
            Unknowns in the order of ``states_di``: liquid fraction [-],
            liquid and vapor mole fractions of the N species that are not
            supercritical [-], vapor fraction [-] and temperature [K].

        Returns
        -------
        numpy.ndarray
            Material residuals [-] followed by the scaled energy residual.

        Notes
        -----
        N is counted from ``is_supercritic``, the mask of the current
        solve, so no count accumulates across solves (issue #319).
        """
        liq = states[0]
        vap = states[-2]

        num_active = int(np.sum(~self.is_supercritic))  # [-]
        fracs = states[1:-2]
        x_i = fracs[:num_active]
        y_i = fracs[num_active:]
        z_i = self.Inlet.mole_frac

        if any(self.is_supercritic):
            x_i, y_i = merge_supercritical(self.is_supercritic, x_i, y_i,
                                           self.z_super)

        temp = states[-1]

        material_bces = self.material_balances(liq, vap, x_i, y_i, z_i, temp)
        energy_bce = self.energy_balance(liq, vap, x_i, y_i, z_i, temp)

        balances = np.concatenate((material_bces, energy_bce))

        return balances

    def solve_unit(self, v_seed: float = 0.5,
                   residual_tolerance: float = np.sqrt(eps)) -> np.ndarray:
        """ Solve AdiabaticFlash unit


        Parameters
        ----------
        v_seed : float, optional
            A seed for output fraction of vapor with respect to feed material
            to the flash [-]. It must be in the range 0-1. The default is 0.5.
        residual_tolerance : float, optional
            Scaled residual and phase-bound allowance [-]. Default
            sqrt(float64 epsilon) matches MINPACK's relative step scale.

        Returns
        -------
        solution : numpy.ndarray
            Phase/species fractions [-], followed by equilibrium temperature [K].

        Raises
        ------
        RuntimeError
            If convergence, scaled residual, or phase-bound validation fails.
        ValueError
            If residual_tolerance is not finite and positive.
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``. If a
            species of the feed (structural presence) lacks the VLE data its
            ratio needs (supercritical fed species, whose ratios are unused,
            need no ``henry_constant`` unless ``seed_basedon_input`` seeds
            from them), or enthalpy or density data that the flash and its
            published products need; species absent from the feed need none.

        Notes
        -----
        The supercritical species and the active species count are derived
        from the current ``Inlet`` on every solve, without changing
        ``num_comp``, so repeated solves of the same feed give the same
        products. Earlier releases reduced ``num_comp`` on every solve of a
        feed with a supercritical species, and the second solve failed
        with ``IndexError`` (issue #319).

        A species with zero feed fraction is absent (structural presence,
        issue #425): it needs no property data, it is never classified as
        supercritical, its VLE ratio is replaced by one, and the published
        ``LiquidOut`` and ``VaporOut`` hold it at exactly zero instead of the
        solver's round-off.
        """
        self._start_presence(self.num_comp, self.Inlet.mole_frac)

        if self.seed_basedon_input:
            x_seed = self.Inlet.mole_frac
            k_seed = self._absent_unit_ratios(self.Inlet.getKeqVLE(
                pres=self.pres,
                weights=self._presence_weights(x_seed)))  # [-]
            y_seed = x_seed * k_seed  # [-], unnormalized
            y_seed = y_seed / y_seed.sum()
        else:
            x_seed = np.ones(self.num_comp) * 1 / self.num_comp
            y_seed = x_seed

        is_supercritic = self._supercritical_mask()
        num_active = int(np.sum(~is_supercritic))  # [-], solved species

        if any(is_supercritic):
            x_seed = x_seed[~is_supercritic]
            y_seed = y_seed[~is_supercritic]

            self.z_super = self.Inlet.mole_frac[is_supercritic]  # [-]

        self.is_supercritic = is_supercritic
        # The feed may have changed in place since it was assigned.
        self.states_di = self._active_states_di(is_supercritic)

        l_seed = 1 - v_seed

        temp_seed = self.Inlet.temp

        seed = np.concatenate(([l_seed], x_seed, y_seed, [v_seed, temp_seed]))

        solution = _solve_flash(self.unit_model, seed, -2, residual_tolerance)

        # Retrieve solution
        liq_flash = solution[0] * self.in_flow
        vap_flash = solution[-2] * self.in_flow
        temp = solution[-1]

        fracs = solution[1:2*num_active + 1]
        x_flash = fracs[:num_active]
        y_flash = fracs[num_active:]

        if any(is_supercritic):
            x_flash, y_flash = merge_supercritical(self.is_supercritic,
                                                   x_flash, y_flash,
                                                   self.z_super)

        x_flash = self._project_absent(x_flash)  # [-]
        y_flash = self._project_absent(y_flash)  # [-]

        path = self.Inlet.path_data

        # Store in objects
        if self.Inlet.__module__ == 'PharmaPy.Streams':
            self.LiquidOut = LiquidStream(path, mole_frac=x_flash,
                                          mole_flow=liq_flash,
                                          temp=temp, pres=self.pres)
            self.VaporOut = VaporStream(path, mole_frac=y_flash,
                                        mole_flow=vap_flash,
                                        temp=temp, pres=self.pres)
        else:
            self.LiquidOut = LiquidPhase(path, mole_frac=x_flash,
                                         moles=liq_flash,
                                         temp=temp, pres=self.pres)
            self.VaporOut = VaporPhase(path, mole_frac=y_flash,
                                       moles=vap_flash,
                                       temp=temp, pres=self.pres)

        return solution


class Evaporator(_StructuralPresenceMixin):
    """ 
    Create a batch/semibatch evaporator object
    
    Parameters
    ----------
    vol_drum : float
        total drum volume [m**3].
    pressure : float, optional
        pressure setpoint [Pa] (actual pressure is computed by the
        evaporator model). The default is 101325.
    diam_out : float, optional
        diameter of the vapor outlet pipe [m]. The default is 2.54e-2.
    k_vap : float, optional
        constant for vapor flow proportional control. Vapor flow is
        calculated as F_v = k_vap * (P_model - P) + F_in. The default is 1.
    h_conv : float, optional
        convective heat transfer coefficient for the liquid phase
        [W/m**2/K]. The default is 1000.
    activity_model : str, optional
        model to be used for calculation activity coefficient in
        vapor-liquid equilibria. Choose one of 'ideal', 'UNIFAC' and
        'UNIQUAC' The default is 'ideal'.
    state_events : list of dicts, optional
        list of dictionaries containing the specification of a state event.
        To learn about the structure of a state event, see the
        PharmaPy.Commons.eval_state_events documentation.
        The default is None.
    stop_at_maxvol : bool, optional
        Action when liquid reaches 95% of the drum volume. True terminates
        integration; False stops the inlet and continues evaporation.
        The default is True.
    flash_kwargs : dict, optional
        Keyword arguments passed to the ``AdiabaticFlash`` constructor
        used by ``init_unit``.
        The default is None.

    Returns
    -------
    A vaporizer object (VO). If a PharmaPy.Stream object is aggregated to the
    resulting instance (instance.Inlet = PharmaPy.Stream(...)),
    the VO will be interpreted as a Semi-batch evaporator object.
    Otherwise, a Batch evaporator will be run.

    """
    def __init__(self, vol_drum,
                 pressure=101325, diam_out=2.54e-2,
                 k_vap=1, cv_gas=0.8,
                 h_conv=1000,
                 activity_model: str = 'ideal', state_events=None,
                 stop_at_maxvol=True, flash_kwargs=None,
                 include_nitrogen=False) -> None:
        """Configure a batch or semibatch evaporator.

        Parameters
        ----------
        vol_drum : float
            Total drum volume [m**3].
        pressure : float, optional
            Downstream pressure used by the vapor outlet law [Pa].
        diam_out : float, optional
            Vapor outlet diameter [m].
        k_vap, cv_gas : float, optional
            Dimensionless multipliers [-] in the vapor outlet law
            ``F_v = rho_mol*area_out*velocity*cv_gas*k_vap`` [mol/s].
        h_conv : float, optional
            Liquid-side heat-transfer coefficient [W/m**2/K].
        activity_model : {'ideal', 'UNIFAC', 'UNIQUAC'}, optional
            Liquid activity model for both VLE and pressure closure.
        state_events : list of dict or dict, optional
            Event specifications passed to ``eval_state_events``.
        stop_at_maxvol : bool, optional
            Action when liquid reaches 95% of the drum volume. True
            terminates integration; False stops the inlet and continues
            evaporation. Defaults to True.
        flash_kwargs : dict, optional
            Keyword arguments passed to the ``AdiabaticFlash`` constructor
            used by ``init_unit``.
        include_nitrogen : bool, optional
            Include nitrogen in initialization and state bookkeeping.

        Raises
        ------
        ValueError
            If ``activity_model`` is not a supported selector.
        """
        validate_activity_model(activity_model, param_name='activity_model')

        self._Inlet = None
        self._Phases = None
        self.include_nitrogen = include_nitrogen

        self._Utility = None
        self.material_from_upstream = False

        self.k_vap = k_vap
        self.cv_gas = cv_gas

        if isinstance(state_events, list):
            self.state_events = state_events
        elif isinstance(state_events, dict):
            self.state_events = [state_events]
        elif state_events is None:
            self.state_events = []

        self.stop_at_maxvol = stop_at_maxvol

        if flash_kwargs is None:
            self.flash_kwargs = {}
        else:
            self.flash_kwargs = flash_kwargs

        # Geometry
        self.vol_tot = vol_drum
        self.diam_tank = (4/np.pi * vol_drum)**(1/3)
        self.area_base = np.pi / 4 * self.diam_tank**2
        self.area_out = np.pi / 4 * diam_out**2

        self.pres = pressure

        # # Jacobians
        # self.jac_states = jacauto(self.unit_model, 1)
        # self.jac_sdot = jacauto(self.unit_model, 2)

        # Heat transfer
        self.h_conv = h_conv

        self.oper_mode = 'Batch'  # If inlet setter, then Semibatch

        self.is_continuous = False

        self.profiles_runs = []

        self.activity_model = activity_model

        self.nomenclature()

        self.vol_offset = 1

        self.elapsed_time = 0
        self.allow_flow = True
        self.inlet_stop_time = None  # [s], absolute; set when the volume event shuts the inlet

        self.outputs = None

    @property
    def Phases(self):
        return self._Phases

    @Phases.setter
    def Phases(self, phase: LiquidPhase) -> None:
        """Attach liquid holdup and describe the complete packed species set.

        Parameters
        ----------
        phase : LiquidPhase
            Condensable liquid with inventory [mol], volume [m**3],
            temperature [K], pressure [Pa], and mole fractions [-]. When
            nitrogen is enabled, state metadata appends nitrogen; the flash
            constructs the augmented phases during initialization. Public
            assignment starts a new charge: cached continuation state,
            elapsed time, profiles, and duties are cleared. This charge
            becomes the reference restored by ``reset()``.

        Notes
        -----
        The charge also starts the structural presence record (see
        ``_start_charge_presence``), which the first initialization rebuilds
        from the liquid as it is then.
        """
        self.__original_phase__ = copy.deepcopy(phase)
        self._set_phases(phase)
        self._start_charge_presence()
        self.elapsed_time = 0  # [s], a replacement charge starts a new run
        self._terminal_states = None
        self.heat_duty = np.zeros(2)  # [J], no energy spent on the new charge
        self.profiles_runs = []
        self.outputs = None
        self.allow_flow = True
        self.inlet_stop_time = None  # [s], the new charge is fed until an event
        for name in ('result', 'Outlet', 'heat_profile', 'is_supercritic'):
            self.__dict__.pop(name, None)

    def _start_charge_presence(self) -> None:
        """Start the structural presence record from the current charge.

        Notes
        -----
        The species with a nonzero fraction in ``Liquid_1``, and nitrogen
        when enabled (it fills the initial headspace), are present (see
        ``_StructuralPresenceMixin``). ``unit_model`` and ``init_unit`` add
        every species fed with a nonzero effective flow, that is, after the
        inlet shut-off of the liquid-volume event. The record persists
        across continuations until a new ``Phases`` assignment,
        ``reset()`` or first initialization, which rebuilds it so that a
        charge edited in place (``updatePhase``) is honored.
        """
        charge = np.zeros(len(self.name_species))  # [-], nitrogen last
        charge[:self.num_species] = np.asarray(self.Liquid_1.mole_frac)  # [-]
        if self.include_nitrogen:
            charge[-1] = 1  # [-], flag: nitrogen fills the initial headspace
        self._start_presence(len(self.name_species), charge)

    def _set_phases(self, phase: LiquidPhase) -> None:
        """Install condensable phases and metadata without changing run history.

        Parameters
        ----------
        phase : LiquidPhase
            Liquid inventory [mol] with temperature [K], pressure [Pa], and
            mole fractions [-]. Internal nitrogen post-processing uses this
            path to preserve the terminal state and original charge.
        """
        path_comp = phase.path_data
        path_inert = '/data/evaporator/props_nitrogen.json'
        path_inert = str(Path(__file__).parents[1]) + path_inert
        paths = [path_comp, path_inert]
        self.paths = paths

        liquid = phase
        vapor = VaporStream(path_comp, pres=self.pres,
                            mole_frac=liquid.mole_frac,
                            mole_flow=eps)

        self._Phases = [liquid, vapor]
        classify_phases(self)

        self.num_species = len(self.Liquid_1.mole_frac)
        self.name_species = list(phase.name_species)
        if self.include_nitrogen:
            self.name_species.append('nitrogen')

        self.nomenclature()

        self.states_di = {
            'mol_i': {'index': self.name_species, 'units': 'mol',
                      'dim': len(self.name_species), 'type': 'diff'},
            'x_liq': {'index': self.name_species, 'units': '',
                      'dim': len(self.name_species), 'type': 'alg'},
            'y_vap': {'index': self.name_species, 'units': '',
                      'dim': len(self.name_species), 'type': 'alg'},
            'mol_liq': {'units': 'mol', 'dim': 1, 'type': 'alg'},
            'mol_vap': {'units': 'mol', 'dim': 1, 'type': 'alg'},
            'pres': {'units': 'Pa', 'dim': 1, 'type': 'alg'},
            'u_int': {'units': 'J', 'dim': 1, 'type': 'diff'},
            'temp': {'units': 'K', 'dim': 1, 'type': 'alg'},
            }

        self.fstates_di = {
            'vol_liq': {'units': 'm^3', 'dim': 1},
            'vol_vap': {'units': 'm^3', 'dim': 1}
                }

        self.name_states = list(self.states_di.keys())

        self.dim_states = [di['dim'] for di in self.states_di.values()]

    def reset(self) -> None:
        """Restore the original charge and clear simulation history.

        Raises
        ------
        RuntimeError
            If no charge has been attached through ``Phases``.

        Notes
        -----
        Restores a copy of the most recent public ``Phases`` assignment.
        Clears elapsed time [s], cached terminal states, accumulated duties
        [J], and profiles. Inlet and utility settings remain configured.
        Call after attaching a charge; the next solve uses first-run
        initialization rather than continuation.
        """
        if self.Phases is None:
            raise RuntimeError("Attach a charge through Phases before calling reset().")
        self.Phases = copy.deepcopy(self.__original_phase__)

    @property
    def Inlet(self):
        return self._Inlet

    @Inlet.setter
    def Inlet(self, inlet: LiquidStream) -> None:
        """Attach the original condensable feed for semibatch operation.

        Parameters
        ----------
        inlet : LiquidStream
            Feed with temperature [K], flow [mol/s], and mole fractions [-]
            in the original liquid species order. The inlet, its controller,
            and its upstream profiles remain attached throughout each solve.
            Nitrogen is appended only to evaluated model inputs.
        """
        self._Inlet = inlet
        self.states_in_dict = {'Inlet': dict(zip(
            self.names_states_in, [inlet.num_species, 1, 1]))}
        self.oper_mode = 'Semibatch'

    @property
    def Utility(self):
        return self._Utility

    @Utility.setter
    def Utility(self, utility):
        self.u_ht = 1 / (1 / self.h_conv + 1 / utility.h_conv)
        self._Utility = utility

    def nomenclature(self):
        self.names_states_in = ['mole_frac', 'mole_flow', 'temp']
        self.names_states_out = ['mole_frac', 'moles', 'temp']

        self.names_upstream = None
        self.bipartite = None

    def get_inputs(self, time: Union[float, np.ndarray]) -> dict:
        """Evaluate the original feed and append nitrogen in model species order.

        Parameters
        ----------
        time : float or ndarray
            Absolute evaluation times [s], scalar or shape (num_times,).

        Returns
        -------
        dict
            ``Inlet`` contains flow [mol/s], temperature [K], and mole
            fractions [-], shape (num_species + include_nitrogen,) or
            (num_times, num_species + include_nitrogen). Nitrogen feed is
            zero. Static values, controllers, and connected upstream profiles
            are evaluated on the original inlet without changing ownership
            or the controller's composition shape. With no inlet, return
            static zero-feed values for any evaluation time. The flow is the
            inlet's, before the shut-off of the liquid-volume event, so the
            callers record the presence of the effective feed.
        """
        if self.Inlet is None:
            inputs = {'mole_flow': 0,  # [mol/s], no feed
                      'mole_frac': np.zeros(
                          self.num_species + self.include_nitrogen),  # [-]
                      'temp': 298.15}  # [K], unused reference for absent feed

            inputs['Inlet'] = inputs
        else:
            inputs = get_inputs_new(time, self.Inlet, self.states_in_dict)
            if self.include_nitrogen:
                fractions = np.asarray(inputs['Inlet']['mole_frac'])  # [-]
                nitrogen = np.zeros(fractions.shape[:-1] + (1,))  # [-], no inlet nitrogen
                inputs['Inlet']['mole_frac'] = np.concatenate(
                    (fractions, nitrogen), axis=-1)  # [-], nitrogen is last

        return inputs

    def material_balances(self, time: float, mol_i: np.ndarray,
                          x_liq: np.ndarray, y_vap: np.ndarray,
                          mol_liq: float, mol_vap: float, pres: float,
                          u_int: float, temp: float, u_inputs: dict,
                          dmoli_dt: Optional[np.ndarray] = None) -> tuple:
        """Evaluate component balances and thermodynamic closure.

        Parameters
        ----------
        time : float
            Current time [s]; retained for the unit-model interface.
        mol_i : ndarray
            Component holdups [mol], shape (num_species,).
        x_liq, y_vap : ndarray
            Liquid and vapor mole fractions [-], shape (num_species,).
        mol_liq, mol_vap : float
            Total liquid and vapor holdups [mol].
        pres : float
            Drum pressure [Pa].
        u_int : float
            Internal energy [J]; used by the separate energy balance.
        temp : float
            Drum temperature [K].
        u_inputs : dict
            Inlet ``mole_flow`` [mol/s] and ``mole_frac`` [-] in species order.
        dmoli_dt : ndarray, optional
            Component holdup derivatives [mol/s], shape (num_species,).
            None requests only the vapor flow and liquid volume.

        Returns
        -------
        tuple
            If ``dmoli_dt`` is None: vapor flow [mol/s], liquid volume [m**3].
            Otherwise: component differential residuals [mol/s], algebraic
            residuals, vapor flow [mol/s], liquid volume [m**3]. Algebraic
            order is component holdups [mol], species closures [-], total
            holdup [mol], volume [m**3], pressure [Pa]. Supercritical species
            retain their explicit vapor partial pressures in pressure closure.

        Raises
        ------
        ValueError
            If the configured activity model is unknown during VLE evaluation.
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``, from
            ``getKeqVLE``: if a present species (structural presence: in
            the charge, fed with a nonzero effective flow, or nitrogen) that
            is not excluded as supercritical lacks the Antoine or Henry data
            its ratio needs, or from ``getDensity`` if it lacks
            ``rho_liq``. Absent species need no data, whatever their
            round-off fractions.

        Notes
        -----
        Pressure closure uses the same Henry-based K-values as equilibrium
        for included species above their critical temperatures. The previous
        pressure sum extrapolated Antoine for every included species.
        Species excluded by ``is_supercritic`` retain explicit vapor partial
        pressures and are indexed out before multiplying K by liquid fraction;
        missing Henry data for these excluded species cannot poison the sum,
        and ``getKeqVLE`` receives them with zero weight, so it does not
        require their data.
        Every excluded supercritical species (not only nitrogen) replaces
        its VLE equation with x_i = 0, in species order. This agrees with
        its exclusion from liquid component holdup and the flash seed.
        The mask is frozen at initialization. An included species crossing its
        critical temperature mid-run therefore adds a Henry/Antoine step to
        pressure closure; the equilibrium residual already had that step.
        """

        input_flow = u_inputs['mole_flow']
        input_fracs = u_inputs['mole_frac']

        mw_vap = np.dot(self.Vapor_1.mw, y_vap)

        rho_mol = pres / gas_ct / temp  # mol/m**3
        rho_gas = rho_mol * (mw_vap / 1000)  # kg/m**3
        vel_vap = np.sqrt(np.maximum(eps, 2 * (pres - self.pres)/rho_gas))

        flow_vap = rho_mol * self.area_out * vel_vap * self.cv_gas * self.k_vap

        rho_liq = self.Liquid_1.getDensity(
            mole_frac=x_liq, temp=temp, basis='mole',
            weights=self._presence_weights(x_liq, time=time))  # [mol/L]

        vol_liq = mol_liq / rho_liq / 1000  # m**3
        vol_vap = mol_vap * gas_ct * temp / pres

        if dmoli_dt is None:
            return flow_vap, vol_liq
        else:
            # Differential eqns
            diff_i = input_flow * input_fracs - flow_vap * y_vap - dmoli_dt  # bce - dMi_dt

            # Algebraic eqns
            not_super = 1 - self.is_supercritic
            component_bce = mol_liq * x_liq * not_super + mol_vap * y_vap - mol_i
            global_bce = mol_liq + mol_vap - sum(mol_i)

            vol_eqn = vol_liq + vol_vap - self.vol_tot

            # One closure per species, retaining species order. Excluded
            # supercritical species cannot occupy the liquid phase, so their
            # ratios are never used and need no property data; nor do those
            # of absent species (structural presence, issue #425).
            condensable = ~self.is_supercritic
            k_i = self.Liquid_1.getKeqVLE(
                temp, pres, x_liq, gamma_model=self.activity_model,
                weights=self._presence_weights(
                    x_liq, condensable, time=time))  # [-]
            # Absent species: unit ratio, so x_i = y_i = 0 whatever their
            # data, also once they become supercritical mid-run.
            k_i = self._absent_unit_ratios(k_i, time=time)  # [-]
            equilibria = x_liq.copy()  # [-], x_i = 0 for excluded species
            equilibria[condensable] = (y_vap[condensable]
                                      - k_i[condensable] * x_liq[condensable])

            p_super = y_vap[self.is_supercritic] * pres  # [Pa]

            # sum(y) = sum_condensable(K*x) + sum_supercritical(y) = 1.
            # With gamma_i = 1 and subcritical condensables, P*K_i = p_sat_i, giving
            # exactly the previous sum(x*p_sat) + sum(p_super) - P.
            # Index before multiplication: excluded K values may be NaN when
            # their Henry data are absent, and zero times NaN is still NaN.
            pres_eqn = (pres * np.dot(k_i[condensable], x_liq[condensable])
                        + sum(p_super) - pres)  # [Pa]

            alg_balances = np.concatenate((component_bce,  # x_liq
                                           equilibria,  # y_vap
                                           np.array([global_bce]),  # M_L
                                           np.array([vol_eqn]),  # M_V
                                           np.array([pres_eqn])  # P
                                           )
                                          )

            return diff_i, alg_balances, flow_vap, vol_liq

    def energy_balances(self, time: float, vol_liq: float, flow_vap: float,
                        u_int: float, temp: float, x_liq: np.ndarray,
                        y_vap: np.ndarray, mol_liq: float, mol_vap: float,
                        pres: float, mol_i: np.ndarray, u_inputs: dict,
                        du_dt: Optional[float] = None) -> Union[float, np.ndarray]:
        """Evaluate heat input and the drum energy residuals.

        Parameters
        ----------
        time : float
            Utility and feed evaluation time [s].
        vol_liq : float
            Liquid volume [m**3].
        flow_vap : float
            Vapor outlet flow [mol/s].
        u_int, temp : float
            Drum internal energy [J] and temperature [K].
        x_liq, y_vap : ndarray
            Liquid and vapor mole fractions [-], shape (num_species,).
        mol_liq, mol_vap : float
            Liquid and vapor holdups [mol].
        pres : float
            Drum pressure [Pa].
        mol_i : ndarray
            Component holdups [mol], retained for the packed-state interface.
        u_inputs : dict
            Feed flow [mol/s], composition [-], and temperature [K]. The
            composition includes the appended zero nitrogen fraction when
            enabled; inlet enthalpy uses the original condensable species.
        du_dt : float, optional
            Internal-energy derivative [J/s]; None requests heat input only.

        Returns
        -------
        float or ndarray
            With None: heat_transfer [J/s], positive into the drum.
            Otherwise: energy-rate residual [J/s] and internal-energy
            closure [J], shape (2,).

        Raises
        ------
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``. If a
            present species (structural presence) lacks the heat-capacity or
            latent-heat data of the liquid, vapor or feed enthalpy; absent
            species need none, and a feed without flow needs none.
        """

        input_flow = u_inputs['mole_flow']  # [mol/s]
        # The original inlet has no nitrogen property entry; its appended
        # zero fraction contributes neither feed material nor enthalpy.
        input_fracs = np.asarray(u_inputs['mole_frac'])[..., :self.num_species]  # [-]
        input_temp = u_inputs['temp']  # [K]

        # Enthalpies. A 0-d array flow (for example a controller's np.where
        # at a scalar time) is a scalar flow; in a flow profile, a row
        # without flow needs no feed data.
        if np.ndim(input_flow) > 0:
            fed = self._feed_weights(input_flow, input_fracs)  # [-]
            h_in = self.Inlet.getEnthalpy(
                temp=input_temp,
                mole_frac=np.broadcast_to(input_fracs, fed.shape),
                basis='mole', weights=fed)  # [J/mol]
        else:
            if input_flow > 0:
                h_in = self.Inlet.getEnthalpy(temp=input_temp,
                                              mole_frac=input_fracs,
                                              basis='mole')
            else:
                h_in = 0

        h_liq = self.Liquid_1.getEnthalpy(
            temp, mole_frac=x_liq, basis='mole',
            weights=self._presence_weights(x_liq, time=time))  # [J/mol]
        h_vap = self.Vapor_1.getEnthalpy(
            temp, mole_frac=y_vap, basis='mole',
            weights=self._presence_weights(y_vap, time=time))  # [J/mol]

        # Heat transfer
        area_ht = 4 / self.diam_tank * vol_liq + self.area_base  # m**2
        temp_ht = self.Utility.get_inputs(time)['temp_in']

        heat_transfer = -self.u_ht * area_ht * (temp - temp_ht)

        if du_dt is None:
            return heat_transfer
        else:
            # Compute balances
            diff_bce = heat_transfer + input_flow * h_in - flow_vap * h_vap - du_dt
            pv_term = pres * self.vol_tot
            internal_energy = mol_liq * h_liq + mol_vap * h_vap - pv_term - u_int

            out_energy = np.array([diff_bce, internal_energy])

        return out_energy

    def unit_model(self, time: float, states: np.ndarray,
                   states_dot: Optional[np.ndarray], sw: Optional[Union[bool, list]],
                   params=None) -> Union[np.ndarray, tuple]:
        """Evaluate the packed material and energy model.

        Parameters
        ----------
        time : float
            Inlet and utility evaluation time [s].
        states : ndarray
            Packed state with species order and units in ``states_di``:
            holdups [mol], mole fractions [-], pressure [Pa], energy [J],
            and temperature [K].
        states_dot : ndarray or None
            Same layout, with component derivatives [mol/s] and energy
            derivative [J/s]; algebraic entries are ignored.
        sw : bool, list of bool, or None
            Event switches, unused by the residual.
        params : object, optional
            Retained unused parameter argument.

        Returns
        -------
        ndarray or tuple
            With derivatives: component residuals [mol/s], thermodynamic
            closure (see ``material_balances``), energy rate [J/s], and
            internal-energy closure [J]. With None: a nested tuple
            ``((vapor_flow, liquid_volume), heat_transfer)`` with units
            (([mol/s], [m**3]), [J/s]); heat is positive into the drum.

        Notes
        -----
        Side effect: the evaluated feed, after the shut-off of the
        liquid-volume event, adds every species it supplies with a nonzero
        flow to the structural presence record, where it stays (latched)
        until the record is rebuilt or a failed solve restores it.
        """

        # Decompose states
        di_states = unpack_states(states, self.dim_states, self.name_states)

        # Decompose derivatives
        if states_dot is None:
            dmoli_dt = None
            du_dt = None
        else:
            di_dot = unpack_states(states_dot, self.dim_states,
                                   self.name_states)
            dmoli_dt = di_dot['mol_i']
            du_dt = di_dot['u_int']

        # Inputs
        u_inputs = self.get_inputs(time)['Inlet']

        u_inputs['mole_flow'] *= self.allow_flow
        # Species fed with a nonzero effective flow are present from now on.
        self._record_feed(u_inputs['mole_flow'], u_inputs['mole_frac'],
                          time=time)

        if states_dot is None:
            material_bces = self.material_balances(time, **di_states,
                                                   u_inputs=u_inputs,
                                                   dmoli_dt=dmoli_dt)

            flow_vap, vol_liq = material_bces

            # Energy balance
            energy_bce = self.energy_balances(time, vol_liq, flow_vap,
                                              u_inputs=u_inputs,
                                              du_dt=du_dt,
                                              **di_states)

            return material_bces, energy_bce
        else:
            material_bces = self.material_balances(time, **di_states,
                                                   u_inputs=u_inputs,
                                                   dmoli_dt=dmoli_dt)

            material_bce = np.concatenate(material_bces[:-2])
            flow_vap, vol_liq = material_bces[-2:]

            # Energy balance
            energy_bce = self.energy_balances(time, vol_liq, flow_vap,
                                              u_inputs=u_inputs,
                                              du_dt=du_dt,
                                              **di_states)

            # Concatenate balances
            balances = np.concatenate((material_bce, energy_bce))

            # Refresh temperature and mole fractions only; derived fields are unchanged.
            self.Liquid_1.temp = di_states['temp']
            self.Vapor_1.temp = di_states['temp']

            self.Liquid_1.mole_frac = di_states['x_liq']  # [-]
            self.Vapor_1.mole_frac = di_states['y_vap']  # [-]

            return balances

    def unit_jacobian(self, c, time, states, sdot, params=None):
        jac_states = self.jac_states(time, states, sdot, params)
        jac_sdot = self.jac_sdot(time, states, sdot, params)

        jac_system = jac_states + c * jac_sdot

        return jac_system

    def init_unit(self) -> tuple:
        """Seed phase equilibrium, holdups, and energy for the DAE solver.

        Returns
        -------
        states_init : ndarray
            Packed state, shape ``(3*n + 5,)``, where n includes nitrogen
            when enabled. Order: component holdups [mol], liquid and vapor
            mole fractions [-], liquid and vapor holdups [mol], pressure
            [Pa], internal energy [J], temperature [K]. Continuations copy
            the retrieved terminal state, including differential energy,
            without re-flashing. On the first solve, bubble-point seeds
            use the configured activity model; nitrogen uses an adiabatic
            flash. Heat transfer uses the wetted wall and base area with
            the overall coefficient [W/m**2/K].
        sdot_init : ndarray
            Same shape; component derivatives [mol/s] and energy derivative
            [J/s] are seeded, with zero placeholders for algebraic entries.
            Initial outflow through the vapor valve is neglected on the
            first solve. Feed rates use the current original inlet evaluated
            at time zero, including controls and upstream profiles, with
            zero nitrogen appended to component inputs. Continuations evaluate
            component and energy rates at the terminal state and current input
            time.

        Raises
        ------
        ValueError
            If liquid volume reaches the drum volume, or a terminating
            95% fill cap is already reached within sqrt(float64 epsilon)
            times drum volume [m**3]. This root-surface allowance covers
            roundoff on both sides of the event. Reset or drain before
            continuing; no state is advanced on rejection.
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``. If a
            present species (charged, fed at the start with a nonzero flow,
            or nitrogen; on a continuation, also species latched earlier or
            fed at the current time) lacks data the initialization needs:
            ``p_vap`` or ``henry_constant`` [Pa] for the bubble-point seed,
            ``cp_liq`` and Watson data (``delta_hvap``, ``tref_hvap``) for
            the enthalpies, ``rho_liq`` for the volume, or the activity
            parameters of a non-ideal model. Absent species need none.

        Notes
        -----
        The initial liquid volume [m**3], which sets the headspace and the
        fill checks, is taken on the model's ideal-mixing basis by
        :func:`_model_liquid_volume`: the stored volume for fraction and
        ``name_solv`` charges, the ideal-mixing volume of the moles for a
        charge whose concentrations are authoritative (issue #439).

        The first initialization rebuilds the structural presence record
        from the charge as it is now (see ``_start_charge_presence``) and
        adds the species of the time-zero feed if it flows; a continuation
        keeps the record.
        """
        # [m**3], model basis (see _model_liquid_volume), checked on every
        # initialization path
        liquid_volume = _model_liquid_volume(self.Liquid_1)
        volume_tolerance = np.sqrt(eps) * self.vol_tot  # [m**3], root-surface roundoff allowance
        if not np.isfinite(liquid_volume) or liquid_volume >= self.vol_tot:
            raise ValueError("Drum volume must exceed the finite liquid volume [m**3]")
        volume_cap = MAX_LIQUID_VOLUME_FRACTION * self.vol_tot  # [m**3]
        if self.stop_at_maxvol and liquid_volume >= volume_cap - volume_tolerance:
            raise ValueError("Evaporator liquid-volume cap reached; reset or drain the "
                             "vessel before another filling solve")
        if self.elapsed_time > 0:
            states_init = self._terminal_states.copy()  # units in states_di
            values = unpack_states(states_init, self.dim_states, self.name_states)
            if self.include_nitrogen:
                # Post-processing exposes condensables; reconstruct the full
                # stored inventory without another adiabatic flash.
                self.Liquid_1 = LiquidPhase(
                    self.paths, temp=values['temp'], pres=values['pres'],
                    moles=values['mol_liq'],
                    mole_frac=self._project_absent(values['x_liq']))
                self.Vapor_1 = VaporPhase(
                    self.paths, temp=values['temp'], pres=values['pres'],
                    moles=values['mol_vap'],
                    mole_frac=self._project_absent(values['y_vap']))
                self._Phases = [self.Liquid_1, self.Vapor_1]
                classify_phases(self)
            else:
                # This object is a stream-shaped property holder, but the
                # stored amount is holdup [mol], not a flow alias [mol/s].
                VaporPhase.updatePhase(
                    self.Vapor_1, temp=values['temp'], pres=values['pres'],
                    moles=values['mol_vap'],
                    mole_frac=self._project_absent(values['y_vap']))
            # A synthetic retrieved segment may not have entered solve_unit.
            if not hasattr(self, 'is_supercritic'):
                self.is_supercritic = values['temp'] > self.Liquid_1.t_crit
            sdot_init = np.zeros_like(states_init)  # derivative units in Returns
            residual = self.unit_model(self.elapsed_time, states_init,
                                       sdot_init, None)  # units in unit_model
            sdot_init[:self.dim_states[0]] = residual[:self.dim_states[0]]  # [mol/s]
            sdot_init[-2] = residual[-2]  # [J/s]
            return states_init, sdot_init

        # A charge edited in place since its assignment (updatePhase) must
        # not keep the presence of the assigned one.
        self._start_charge_presence()

        temp_init = self.Liquid_1.temp  # [K]
        pres_init = self.pres  # [Pa]

        x_init = self.Liquid_1.mole_frac  # [-]
        y_init = np.zeros_like(x_init)  # [-]

        # Moles of phases
        mol_liq = self.Liquid_1.moles  # [mol]
        vol_liq = liquid_volume  # [m**3], model basis

        vol_vap = self.vol_tot - vol_liq  # [m**3]
        if vol_vap < 0:
            raise ValueError(r"Drum volume ({:.2e} m3) lower than the liquid "
                             r"volume ({:.2e} m3)".format(self.vol_tot, vol_liq))

        mol_vap = self.pres * vol_vap / gas_ct / temp_init  # [mol]
        mol_tot = mol_liq + mol_vap  # [mol]

        if self.include_nitrogen:
            x_init = np.append(x_init, 0)  # [-]

            y_init = np.append(y_init, 1)  # [-]

            # Moles of i
            mol_i = mol_liq * x_init + mol_vap * y_init  # [mol]

            z_flash = mol_i / mol_i.sum()  # [-]
            pres_in = self.Liquid_1.pres  # [Pa]

            LiqEvap = LiquidPhase(self.paths, temp_init, pres=pres_in,
                                  moles=mol_tot, mole_frac=z_flash)

            FlashInit = AdiabaticFlash(pres_drum=pres_init,
                                       gamma_method=self.activity_model,
                                       **self.flash_kwargs)

            FlashInit.Inlet = LiqEvap
            FlashInit.solve_unit()

            # Update phases and initial states with flash results
            self.Liquid_1 = FlashInit.LiquidOut
            self.Vapor_1 = FlashInit.VaporOut
            self._Phases = [self.Liquid_1, self.Vapor_1]

            x_init = FlashInit.LiquidOut.mole_frac  # [-]
            y_init = FlashInit.VaporOut.mole_frac  # [-]

            temp_init = FlashInit.LiquidOut.temp  # [K]

            mol_liq = FlashInit.LiquidOut.moles  # [mol]
            # The flash liquid's own moles to volume (issue #439); it is
            # built from fractions, so this is its ideal-mixing density.
            dens_liq = self.Liquid_1.getInventoryDensity(basis='mole')  # [mol/L]
            vol_liq = mol_liq / dens_liq / 1000  # [m**3]
            vol_vap = self.vol_tot - vol_liq  # [m**3]

            mol_vap = pres_init * vol_vap / gas_ct / temp_init  # [mol]

            mol_i = mol_liq * x_init + mol_vap * y_init  # [mol]
            mol_tot = mol_liq + mol_vap  # [mol]


        else:
            temp_init, y_init = self.Liquid_1.getBubblePoint(
                pres=self.pres, y_vap=True,
                thermo_method=self.activity_model,
                weights=self._presence_weights(
                    self.Liquid_1.mole_frac))  # [K], [-]
            # The headspace shares the bubble-point temperature of the seed.
            mol_vap = pres_init * vol_vap / gas_ct / temp_init  # [mol]

        # Moles of i
        mol_i = mol_liq * x_init + mol_vap * y_init  # [mol]

        if self.Inlet is None:
            dm_init = np.zeros_like(x_init)  # [mol/s]
            inlet_flow = 0  # [mol/s]
            hin_init = 0  # [J/mol]
        else:
            inlet_inputs = self.get_inputs(0)['Inlet']
            inlet_flow = inlet_inputs['mole_flow']  # [mol/s]
            fed = self._record_feed(inlet_flow,
                                    inlet_inputs['mole_frac'],
                                    time=self.elapsed_time)  # [-]
            dm_init = inlet_flow * inlet_inputs['mole_frac']  # [mol/s]
            # A species in a feed without flow needs no data.
            hin_init = self.Inlet.getEnthalpy(
                temp=inlet_inputs['temp'],
                mole_frac=inlet_inputs['mole_frac'][..., :self.num_species],
                basis='mole',
                weights=fed[..., :self.num_species])  # [J/mol], condensables

        # ---------- Energy balance states
        # Enthalpies

        hliq_init = self.Liquid_1.getEnthalpy(
            temp_init, basis='mole',
            weights=self._presence_weights(x_init))  # [J/mol]
        hvap_init = self.Vapor_1.getEnthalpy(
            temp_init, mole_frac=y_init, basis='mole',
            weights=self._presence_weights(y_init))  # [J/mol]

        # Wetted cylindrical wall plus base, as in energy_balances.
        area_ht = 4 / self.diam_tank * vol_liq + self.area_base  # [m**2]

        temp_ht = self.Utility.get_inputs(0)['temp_in']  # [K]

        ht_init = -self.u_ht * area_ht * (temp_init - temp_ht)  # [J/s]

        du_init = ht_init + inlet_flow * hin_init  # [J/s]

        # Internal energy
        u_init = (mol_liq * hliq_init + mol_vap * hvap_init
                  - pres_init * self.vol_tot)  # [J]

        # ---------- Retrieve results
        states_init = np.concatenate(
            (mol_i, x_init, y_init, [mol_liq, mol_vap],
             [pres_init, u_init, temp_init])
        )

        sdot_init = np.zeros_like(states_init)
        sdot_init[:self.num_species + self.include_nitrogen] = dm_init
        sdot_init[-2] = du_init

        return states_init, sdot_init

    def __state_event(self, time: float, states: np.ndarray,
                      sdot: np.ndarray, switch: list) -> np.ndarray:
        """Evaluate the user events and the liquid-volume event.

        Parameters
        ----------
        time : float
            Absolute solver time [s].
        states, sdot : numpy.ndarray
            Packed states and derivatives, layout and units of
            ``states_di``.
        switch : list of bool
            Event switches of IDA.

        Returns
        -------
        numpy.ndarray
            Event functions: user events, then the free liquid volume
            [m**3] below ``MAX_LIQUID_VOLUME_FRACTION`` of the drum; a sign
            change triggers the event. Empty when every switch is off.

        Notes
        -----
        The liquid density is weighted with the structural presence, so
        absent species need no ``rho_liq``.
        """
        events = []

        states_trim = np.split(states, self.trim_idx)
        sdot_trim = np.split(sdot, self.trim_idx)

        dict_states = dict(zip(self.name_states, states_trim))
        dict_sdot = dict(zip(self.name_states, sdot_trim))

        if any(switch):

            for di in self.state_events:
                if 'state_fn' in di.keys():
                    event_flag = di['state_fn'](time, self.Phases,
                                                dict_states, dict_sdot)
                else:
                    state_name = di['state_name']
                    state_idx = di['state_idx']
                    ref_value = di['value']

                    checked_value = dict_states[state_name][state_idx]

                    event_flag = ref_value - checked_value

                events.append(event_flag)

            # Maximum liquid constraint
            rho_liq = self.Liquid_1.getDensity(
                mole_frac=dict_states['x_liq'],
                temp=dict_states['temp'][0],
                basis='mole',
                weights=self._presence_weights(dict_states['x_liq'],
                                               time=time)
            )  # [mol/L]

            vol_liq = dict_states['mol_liq'][0] / rho_liq / 1000  # m**3

            events.append(MAX_LIQUID_VOLUME_FRACTION * self.vol_tot - vol_liq)

        return np.array(events)

    def __handle_event(self, solver, event_info) -> None:
        """Terminate on user events or apply the liquid-volume event action.

        Parameters
        ----------
        solver : assimulo.solvers.IDA
            Running solver; ``solver.t`` is the absolute event time [s].
        event_info : tuple
            Assimulo event information; ``event_info[0]`` holds one sign
            change indicator [-] per user event, then the volume event.

        Raises
        ------
        TerminateSimulation
            If a user event fires in its configured direction, or the
            volume event fires with ``stop_at_maxvol=True``.

        Notes
        -----
        With ``stop_at_maxvol=False`` the volume event shuts the inlet for
        the rest of the charge. The first such event records its absolute
        time in ``inlet_stop_time`` [s]; raw-material accounting stops the
        feed there. Later crossings of the volume limit leave it unchanged.
        """
        state_event = event_info[0]

        for ind, val in enumerate(state_event[:-1]):
            direction = self.state_events[ind].get('direction')
            terminate = False
            id_event = self.state_events[ind].get('event_name')
            if val:
                if direction is None:
                    terminate = True
                elif direction == val:
                    terminate = True

                if terminate:
                    if id_event is None:
                        print('State event %i was reached' % (ind + 1))
                    else:
                        print("State event '%s' was reached" % id_event)

                    raise TerminateSimulation

        if state_event[-1]:
            print('95%% of the available tank volume reached by the liquid '
                  'phase')

            if self.stop_at_maxvol:
                raise TerminateSimulation
            else:
                if self.allow_flow:
                    self.inlet_stop_time = float(solver.t)  # [s], absolute
                self.allow_flow = False

    def solve_unit(self, runtime: float, verbose: bool = True,
                   sundials_opts: Optional[dict] = None,
                   time_grid: Optional[Sequence[float]] = None) -> tuple:
        """ Solve Evaporator model


        Parameters
        ----------
        runtime : float
            Duration of this simulation segment [s]. Continuations retain
            the previous terminal state and advance absolute time by runtime.
            With ``stop_at_maxvol=False``, an inlet stopped by the liquid
            volume event remains stopped on continuation, and the absolute
            stop time [s] is kept in ``inlet_stop_time``. If the call
            raises, both are restored to their values before it (see
            Notes). ``reset()`` or a
            new public ``Phases`` assignment re-enables the inlet and clears
            that time.
        verbose : bool, optional
            if True, integrator statistics will be displayed after the model
            is solved. The default is True.
        sundials_opts : dict, optional
            options to be passed to SUNDIALS. For a list of available options,
            visit https://jmodelica.org/assimulo/DAE_IDA.html
            The default is None.
        time_grid : sequence of float, optional
            Absolute output times [s], finite and strictly increasing within
            this segment's start/end bounds. IDA also reports the segment start.
            Omit to retain adaptive output times. This does not change runtime.

        Returns
        -------
        time : list
            Absolute solver times [s], including the segment start.
        states : numpy array
            Packed state profiles; order and units are in ``states_di``.

        Raises
        ------
        ValueError
            If time_grid is empty, not one-dimensional, nonfinite, unordered,
            or outside the absolute simulation interval.
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``. Raised by
            the initialization if a present species (structural presence:
            charged, fed with a nonzero effective flow at the start, or
            nitrogen) lacks data that it needs, and by the publication of
            the results (the heat-duty replay with its bubble points and
            enthalpies, and the published phases) if a species present in a
            row lacks data that they need; absent species need none.
        IDAError
            From Assimulo, if the balances fail during the integration, for
            example because a species fed only after the start lacks data.
            The missing-data error is not chained to it and the species is
            not named (issue #435).

        Notes
        -----
        A call that raises keeps no results, whatever the exception and
        whether the initialization, the integration or the publication of
        the results raised: the unit's attributes (elapsed time, terminal
        states, inlet mode, profiles, results, duties, outlet), the
        structural presence record and the complete state of the attached
        phases (temperature [K], fractions [-], amounts and derived
        fields) are restored to their values before the call (issue #425).
        """

        final_time = self.elapsed_time + runtime  # [s], absolute segment endpoint
        if time_grid is not None:
            time_grid = np.asarray(time_grid, dtype=float)  # [s]
            if (time_grid.ndim != 1 or time_grid.size == 0
                    or not np.all(np.isfinite(time_grid))
                    or np.any(np.diff(time_grid) <= 0)
                    or time_grid[0] < self.elapsed_time
                    or time_grid[-1] > final_time):
                raise ValueError(
                    "time_grid must contain finite, strictly increasing absolute "
                    "times [s] within the simulation segment")

        self.args_inputs = (self, self.num_species)

        # A solve that raises keeps no results: everything from the
        # initialization (which evaluates the feed on a continuation)
        # through the integration (events may shut the inlet, and feed
        # evaluations latch species) and the publication of the results is
        # undone, including the inlet mode.
        before_solve = self._solve_snapshot()
        try:
            time, states = self._integrate_segment(final_time, verbose,
                                                   sundials_opts, time_grid)
            self.retrieve_results(time, states)
        except BaseException:
            self._restore_solve_snapshot(before_solve)
            raise

        return time, states

    def _integrate_segment(self, final_time: float, verbose: bool,
                           sundials_opts: Optional[dict],
                           time_grid: Optional[np.ndarray]) -> tuple:
        """Initialize and integrate one segment without keeping results.

        Parameters
        ----------
        final_time : float
            Absolute end of the segment [s].
        verbose : bool
            Show the integrator statistics.
        sundials_opts : dict or None
            Attributes set on the IDA solver.
        time_grid : numpy.ndarray or None
            Validated absolute output times [s].

        Returns
        -------
        time : list
            Absolute solver times [s], including the segment start.
        states : numpy.ndarray
            Packed state profiles; order and units are in ``states_di``.

        Raises
        ------
        MissingPropertyError
            From the initialization, if a present species (structural
            presence) lacks data that it needs.
        IDAError
            From Assimulo, if the balances fail during the integration; a
            missing-data error there is not chained to it (issue #435).

        Notes
        -----
        ``solve_unit`` undoes every change made here if it raises.
        """
        states_init, sdot_init = self.init_unit()

        # ---------- Solve
        n_comp = self.num_species + self.include_nitrogen
        alg_map = np.concatenate((np.ones(n_comp),
                                  np.zeros(2*n_comp + 3),
                                  [1, 0])
                                 )

        self.alg_map = (1 - alg_map).astype(bool)

        # ---------- Count states
        len_states = [n_comp] * 3 + [1] * 5
        self.trim_idx = np.cumsum(len_states)[:-1]

        # ---------- Check supercritical components
        temp_init = states_init[-1]

        if self.elapsed_time == 0:
            self.is_supercritic = temp_init > self.Liquid_1.t_crit

        # ---------- Solve problem
        # Create problem
        switches = [True] * len(self.state_events) + [True]
        problem = Implicit_Problem(self.unit_model,
                                   states_init, sdot_init,
                                   t0=self.elapsed_time, sw0=switches)

        problem.state_events = self.__state_event
        problem.handle_event = self.__handle_event

        problem.algvar = alg_map
        # problem.jac = self.unit_jacobian

        # Set solver
        solver = IDA(problem)
        solver.make_consistent('IDA_YA_YDP_INIT')
        solver.suppress_alg = True

        if not verbose:
            solver.verbosity = 50

        if sundials_opts is not None:
            for name, val in sundials_opts.items():
                setattr(solver, name, val)

                if name == 'time_limit':
                    solver.report_continuously = True

        time, states, sdot = solver.simulate(final_time, ncp_list=time_grid)

        return time, states

    def retrieve_results(self, time: np.ndarray, states: np.ndarray) -> None:
        """Store an absolute-time segment and update the liquid outlet.

        Parameters
        ----------
        time : array-like
            Absolute solver times [s], shape (num_times,). The final time
            replaces ``elapsed_time`` for the next duration-based solve.
        states : ndarray
            Packed profiles, shape (num_times, sum(dim_states)), with units
            and species order in ``states_di``. Nitrogen remains in results
            and duty calculations; condensable phases are restored afterwards
            for the next initialization. The original inlet remains attached.

        Notes
        -----
        The published liquid, and the vapor holdup ``Vapor_1``, take the
        terminal state with absent species (structural presence) at
        exactly zero instead of the solver's round-off, so their property
        methods need no data for them.
        """
        self.elapsed_time = time[-1]  # [s], solver time is already absolute
        self._terminal_states = np.asarray(states[-1]).copy()  # units in states_di

        # ---------- Create result object
        dp = unpack_states(states, self.dim_states, self.name_states)

        dp['time'] = np.asarray(time)

        self.outputs = dp

        self.profiles_runs.append(dp)
        dp = self.flatten_states()

        self.result = DynamicResult(self.states_di, self.fstates_di, **dp)

        # Evaluate augmented profiles before restoring condensable phases.
        self.get_heat_duty(time, states)

        # ---------- Update phases
        self.Liquid_1.temp = dp['temp'][-1]
        self.Liquid_1.pres = dp['pres'][-1]

        xliq_update = dp['x_liq']
        if self.include_nitrogen:
            Liquid_1 = LiquidPhase(
                self.paths[0], temp=dp['temp'][-1], pres=dp['pres'][-1],
                moles=dp['mol_liq'][-1],
                mole_frac=self._project_absent(dp['x_liq'][-1])[:-1])

            self._set_phases(Liquid_1)
        else:
            self.Liquid_1.updatePhase(
                mole_frac=self._project_absent(xliq_update[-1]),
                moles=dp['mol_liq'][-1], temp=dp['temp'][-1],
                pres=dp['pres'][-1])
            # unit_model left the last iterate's fractions in the vapor,
            # with round-off of absent species; publish the terminal vapor
            # holdup [mol], as the continuation does.
            VaporPhase.updatePhase(
                self.Vapor_1, temp=dp['temp'][-1], pres=dp['pres'][-1],
                moles=dp['mol_vap'][-1],
                mole_frac=self._project_absent(dp['y_vap'][-1]))

        # Output info
        self.Outlet = self.Liquid_1

    def flatten_states(self):
        out = flatten_states(self.profiles_runs)

        return out

    def get_heat_duty(self, time: np.ndarray, states: np.ndarray) -> None:
        """Calculate the supplied segment's powers and accumulate its duties.

        Parameters
        ----------
        time : array-like
            Segment times [s], shape (num_times,).
        states : ndarray
            Segment states, shape (num_times, sum(dim_states)); units and
            species order are given in ``states_di``.

        Notes
        -----
        ``heat_profile`` [J/s] describes this segment only. Each call adds
        its trapezoidal energy integral to cumulative ``heat_duty`` [J].
        Batch duties are signed: drum heat input is positive, and cooling
        in the condenser is negative.

        """
        dp = unpack_states(states, self.dim_states, self.name_states)
        # ---------- Heat balance
        # Heating duty
        heat_bce = np.zeros(len(time))  # [J/s]
        flow_vap = np.zeros_like(heat_bce)  # [mol/s]
        h_liq = np.zeros_like(heat_bce)  # [J/mol]
        h_vap = np.zeros_like(heat_bce)  # [J/mol]

        for ind, row in enumerate(states):
            mass_bce, q_ht = self.unit_model(time[ind], row, None, False)

            heat_bce[ind] = q_ht
            flow_vap[ind] = mass_bce[0]

            x_liq = dp['x_liq'][ind]  # [-]
            y_vap = dp['y_vap'][ind]  # [-]

            weights = self._presence_weights(x_liq, time=time[ind])  # [-]
            temp_bubble = self.Liquid_1.getBubblePoint(
                pres=dp['pres'][ind], mole_frac=x_liq,
                thermo_method=self.activity_model, weights=weights)  # [K]

            h_liq[ind] = self.Liquid_1.getEnthalpy(
                temp=temp_bubble, mole_frac=x_liq,
                basis='mole', weights=weights)  # [J/mol]

            h_vap[ind] = self.Vapor_1.getEnthalpy(
                temp=dp['temp'][ind], mole_frac=y_vap,
                basis='mole', weights=weights)  # [J/mol]

        # Condensation duty
        heat_cond_prof = flow_vap * (h_liq - h_vap)  # [J/s]

        self.heat_profile = np.column_stack((heat_bce, heat_cond_prof))  # [J/s]
        segment_duty = trapezoidal_rule(time, self.heat_profile)  # [J]
        self.heat_duty = getattr(self, 'heat_duty', np.zeros(2)) + segment_duty  # [J]

        # TODO: select duty types from operating temperature.
        self.duty_type = [0, 0]

    def plot_profiles(self, pick_comp=None, **fig_kwargs):
        """
        Convenience function to plot model solution. Dynamic profiles displayed
        by this funcion are x_liq vs t, y_vap vs t, T/P vs t
        and mol_liq/mol_vap vs t.

        Parameters
        ----------
        pick_comp : list of int, optional
            indexes of states to be plot. If None, all the states are plotted.
            The default is None.
        fig_kwargs : keyword arguments
            keyword arguments to be passed to the construction of fig and
            axes object of matplotlib (``plt.subplots(**fig_kwargs)``).
            Do not use nrows or ncols arguments, since the plot grid is already
            defined by PharmaPy

        Returns
        -------
        fig : TYPE
            figure object.
        ax : numpy array
            grid of axis objects.

        """

        # Fractions
        if pick_comp is None:
            states_plot = ('x_liq', 'y_vap', 'temp', 'pres', 'mol_liq',
                           'mol_vap')
        else:
            states_plot = (['x_liq', pick_comp], ['y_vap', pick_comp], 'temp',
                           'pres', 'mol_liq', 'mol_vap')

        ylabels = ('x_liq', 'y_vap', 'T', 'P', 'N_L', 'N_V')
        fig, ax = plot_function(self, states_plot, fig_map=(0, 1, 2, 2, 3, 3),
                                ylabels=ylabels,
                                nrows=2, ncols=2, **fig_kwargs)

        for axis in ax.flatten():
            axis.xaxis.set_minor_locator(AutoMinorLocator(2))
            axis.yaxis.set_minor_locator(AutoMinorLocator(2))

        fig.text(0.5, 0, 'time (s)', ha='center')

        fig.tight_layout()

        return fig, ax


class ContinuousEvaporator(_StructuralPresenceMixin):
    """
    Create a continuous evaporator object

    Parameters
    ----------
    vol_drum : float
        total drum volume [m**3].
    adiabatic : bool, optional
        If True, omit drum-wall heat transfer; reflux condenser duty remains.
        The default is False.
    pressure : TYPE, optional
        pressure setpoint [Pa] (actual pressure is computed by the
        evaporator model). The default is 101325.
    diam_out : float, optional
        diameter of the vapor outlet pipe [m]. The default is 2.54e-2.
    frac_liq : float, optional
        setpoint for the fraction of the total tank volume occupied by the
        liquid phase. The default is 0.5.
    k_liq : float, optional
        Liquid volume controller gain [mol/s/m**3]. The outlet flow is
        ``max(0, k_liq*(V_L - vol_drum*frac_liq) + input_flow
        - (1 - reflux_ratio)*flow_vap)``
        [mol/s], including inlet feedforward. The default is 100.
    k_vap : float, optional
        proportional control constant for pressure, which
        actual pressure (P) by changing output vapor molar flow (F_V), with
        F_V = k_vap * f(pressure - P). The default is 1.
    h_conv : float, optional
        convective heat transfer coefficient for the liquid phase
        [W/m**2/K]. The default is 1000.
    activity_model : str, optional
        model to be used for calculation activity coefficient in
        vapor-liquid equilibria. Choose one of 'ideal', 'UNIFAC' and
        'UNIQUAC' The default is 'ideal'.
    num_interp_points : int, optional
        Number of interpolation points to be used by the Interpolation
        module of PharmaPy to calculate inputs if the evaporator receives
        material from a dynamic upstream unit operation. The default is 3.
    state_events : list of dicts, optional
        list of dictionaries containing the specification of a state event.
        To learn about the structure of a state event, see the
        PharmaPy.Commons.eval_state_events documentation.
        The default is None.
    reflux_ratio : float, optional
        reflux ratio ranging from (0 - 1), which dictates the fraction
        of the vapor flow which is sent back to the unit, assuming total
        condensation. The default is 0.

    Returns
    -------
    A continuous vaporizer object.

    """
    def __init__(self, vol_drum, adiabatic=False,
                 pressure=101325, diam_out=2.54e-2, frac_liq=0.5,
                 k_liq=100, k_vap=1,
                 cv_gas=0.8,
                 h_conv=1000,
                 activity_model: str = 'ideal', num_interp_points=3,
                 mult_flash=1, state_events=None, reflux_ratio=0) -> None:
        """Configure a continuous evaporator and its outlet controls.

        Parameters
        ----------
        vol_drum : float
            Total drum volume [m**3].
        adiabatic : bool, optional
            Whether to omit drum-wall heat transfer; condenser duty remains.
        pressure : float, optional
            Downstream pressure used by the vapor outlet law [Pa].
        diam_out : float, optional
            Vapor outlet diameter [m].
        frac_liq : float, optional
            Liquid volume setpoint as a fraction of drum volume [-].
        k_liq : float, optional
            Liquid volume controller gain [mol/s/m**3], applied to
            ``vol_liq - vol_drum*frac_liq`` with feedforward ``+ input_flow``
            [mol/s] before vapor-flow subtraction and flow limiting.
        k_vap, cv_gas : float, optional
            Dimensionless multipliers [-] in the vapor outlet law
            ``F_v = rho_mol*area_out*velocity*cv_gas*k_vap`` [mol/s].
        h_conv : float, optional
            Liquid-side heat-transfer coefficient [W/m**2/K].
        activity_model : {'ideal', 'UNIFAC', 'UNIQUAC'}, optional
            Liquid activity model for both VLE and pressure closure.
        num_interp_points : int, optional
            Number of interpolation points for dynamic inlet data [-].
        mult_flash : float, optional
            Retained unused flash multiplier [-] for API compatibility.
        state_events : list of dict, optional
            State event specifications.
        reflux_ratio : float, optional
            Fraction of vapor outlet returned as total condensate [-],
            in the closed interval [0, 1].

        Raises
        ------
        ValueError
            If ``activity_model`` is not a supported selector or
            ``reflux_ratio`` is not finite and in [0, 1].
        """
        validate_activity_model(activity_model, param_name='activity_model')
        if not np.isfinite(reflux_ratio) or not 0 <= reflux_ratio <= 1:
            raise ValueError("reflux_ratio must be finite and in [0, 1]")

        self._Inlet = None
        self._Phases = None
        self._Utility = None

        self.vol_tot = vol_drum
        self.vol_liq_set = vol_drum * frac_liq

        # Geometry
        self.diam_tank = (4/np.pi * vol_drum)**(1/3)
        self.area_base = np.pi / 4 * self.diam_tank**2
        self.area_out = np.pi / 4 * diam_out**2

        # Control
        self.k_liq = k_liq
        self.k_vap = k_vap
        self.cv_gas = cv_gas

        self.vol_offset = 1

        self.adiabatic = adiabatic

        self.pres = pressure

        # Jacobians
        # self.jac_states = jacauto(self.unit_model, 1)
        # self.jac_sdot = jacauto(self.unit_model, 2)

        # Heat transfer
        self.h_conv = h_conv

        self.oper_mode = 'Continuous'

        self.is_continuous = True

        self.tau = None

        self.profiles_runs = []

        self.activity_model = activity_model
        self.num_interp_points = num_interp_points

        self.nomenclature()

        self.state_event_list = state_events

        self.reflux_ratio = reflux_ratio

        self.outputs = None

    @property
    def Phases(self):
        return self._Phases

    @Phases.setter
    def Phases(self, phase: LiquidPhase) -> None:
        """Attach a new charge and clear the previous simulation history.

        Parameters
        ----------
        phase : LiquidPhase
            Liquid charge, including composition [-], amount [mol],
            temperature [K], and pressure [Pa].

        Notes
        -----
        Saves a copy for ``reset()``. Inlet and utility settings are retained.
        Internal result retrieval preserves the original charge and history.
        The charge starts the structural presence record (issue #425; see
        ``_StructuralPresenceMixin``); every ``init_unit`` rebuilds it from
        the liquid as it is then, and ``get_inputs`` adds the species of
        every feed row that flows.
        """
        self.__original_phase__ = copy.deepcopy(phase)
        self._set_phases(phase)
        self._start_presence(len(phase.name_species), phase.mole_frac)
        self._signed_heat_duty = np.zeros(2)  # [J]
        self.heat_duty = np.zeros(2)  # [J]
        self.profiles_runs = []
        self.outputs = None
        self.tau = None  # [s], residence time is recomputed for the new charge
        for name in ('result', 'Outlet', 'heat_profile', 'liqFlowProf', 'vapFlowProf'):
            self.__dict__.pop(name, None)

    def _set_phases(self, phase: LiquidPhase) -> None:
        """Install phases and state metadata without clearing run history.

        Parameters
        ----------
        phase : LiquidPhase
            Liquid phase with composition [-], amount [mol], temperature [K],
            and pressure [Pa].
        """
        path_comp = phase.path_data

        vapor = VaporStream(path_comp, pres=self.pres,
                            mole_frac=np.zeros(len(phase.name_species)),
                            check_input=False, verbose=False)

        self._Phases = [phase, vapor]
        classify_phases(self)

        self.Liquid_1 = phase

        self.num_species = len(self.Liquid_1.mole_frac)
        self.name_species = self.Liquid_1.name_species

        self.states_di = {
            'mol_i': {'index': self.name_species, 'units': 'mol',
                      'dim': len(self.name_species), 'type': 'diff'},
            'x_liq': {'index': self.name_species, 'units': '',
                      'dim': len(self.name_species), 'type': 'alg'},
            'y_vap': {'index': self.name_species, 'units': '',
                      'dim': len(self.name_species), 'type': 'alg'},
            'mol_liq': {'units': 'mol', 'dim': 1, 'type': 'alg'},
            'mol_vap': {'units': 'mol', 'dim': 1, 'type': 'alg'},
            'pres': {'units': 'Pa', 'dim': 1, 'type': 'alg'},
            'u_int': {'units': 'J', 'dim': 1, 'type': 'diff'},
            'temp': {'units': 'K', 'dim': 1, 'type': 'alg'},
            }

        self.fstates_di = {
            'flow_liq': {'units': 'mol/s', 'dim': 1},
            'flow_vap': {'units': 'mol/s', 'dim': 1},
            'vol_liq': {'units': 'm**3', 'dim': 1},
            'vol_vap': {'units': 'm**3', 'dim': 1},
            }

        self.name_states = list(self.states_di.keys())

        self.dim_states = [di['dim'] for di in self.states_di.values()]

    def reset(self) -> None:
        """Restore the original charge and clear simulation history.

        Raises
        ------
        RuntimeError
            If no charge has been attached through ``Phases``.

        Notes
        -----
        Restores a copy of the most recent public ``Phases`` assignment and
        clears accumulated duties [J], profiles, outputs, and cached residence
        time [s]. Inlet and utility settings remain configured.
        """
        if self.Phases is None:
            raise RuntimeError("Attach a charge through Phases before calling reset().")
        self.Phases = copy.deepcopy(self.__original_phase__)

    @property
    def Inlet(self):
        return self._Inlet

    @Inlet.setter
    def Inlet(self, inlet):  # Create an inlet with additional species (N2)

        self._Inlet = inlet
        self._Inlet.num_interpolation_points = self.num_interp_points

        num_comp = len(self.Inlet.name_species)
        len_in = [num_comp, 1, 1]
        states_in_dict = dict(zip(self.names_states_in, len_in))

        self.states_in_dict = {'Inlet': states_in_dict}

    @property
    def Utility(self):
        return self._Utility

    @Utility.setter
    def Utility(self, utility):
        self.u_ht = 1 / (1 / self.h_conv + 1 / utility.h_conv)
        self._Utility = utility

    def nomenclature(self):
        self.names_states_in = ['mole_frac', 'mole_flow', 'temp']
        self.names_states_out = ['mole_frac', 'mole_flow', 'temp']
        # self.name_states = ['moles_i', 'x_liq', 'y_vap', 'mol_liq', 'mol_vap',
        #                     'pres', 'U_internal', 'temp']

        self.name_differential = ['moles_i', 'U_internal']
        self.name_algebraic = ['x_liq', 'y_vap', 'mol_liq', 'mol_vap',
                               'pres', 'temp']

    def get_inputs(self, time: Union[float, np.ndarray]) -> dict:
        """Evaluate the feed and record its species as present.

        Parameters
        ----------
        time : float or ndarray
            Absolute evaluation times [s], scalar or shape (num_times,).

        Returns
        -------
        dict
            ``Inlet`` with flow [mol/s], temperature [K] and mole fractions
            [-], shape (num_species,) or (num_times, num_species), from the
            static inlet, its controller or its upstream profile.

        Notes
        -----
        Every species with a nonzero evaluated feed fraction in a row with
        a nonzero flow joins the structural presence record (see
        ``_StructuralPresenceMixin``); a feed without flow adds nothing.
        """
        inputs = get_inputs_new(time, self.Inlet, self.states_in_dict)
        self._record_feed(inputs['Inlet']['mole_flow'],
                          inputs['Inlet']['mole_frac'], time=time)

        return inputs

    def get_volumes(self, temp: Union[float, np.ndarray],
                    pres: Union[float, np.ndarray], x_i: np.ndarray,
                    mol_liq: Union[float, np.ndarray],
                    mol_vap: Union[float, np.ndarray]) -> tuple:
        """Return the liquid and vapor volumes of the drum holdups.

        Parameters
        ----------
        temp, pres : float or ndarray
            Drum temperature [K] and pressure [Pa].
        x_i : ndarray
            Liquid mole fractions [-], shape (num_species,) or
            (num_times, num_species).
        mol_liq, mol_vap : float or ndarray
            Liquid and vapor holdups [mol].

        Returns
        -------
        vol_liq, vol_vap : float or ndarray
            Ideal-mixing liquid volume and ideal-gas vapor volume [m**3].

        Raises
        ------
        MissingPropertyError
            If a present species (structural presence) has no ``rho_liq``.
        """
        rho_liq = self.Liquid_1.getDensity(
            mole_frac=x_i, temp=temp, basis='mole',
            weights=self._presence_weights(x_i))  # [mol/L]

        vol_liq = mol_liq / rho_liq / 1000  # m**3
        vol_vap = mol_vap * gas_ct * temp / pres

        return vol_liq, vol_vap

    def get_mole_flows(self, temp: Union[float, np.ndarray],
                       pres: Union[float, np.ndarray], x_i: np.ndarray,
                       y_i: np.ndarray, mol_liq: Union[float, np.ndarray],
                       mol_vap: Union[float, np.ndarray],
                       input_flow: Union[float, np.ndarray],
                       time: Optional[Union[float, np.ndarray]] = None
                       ) -> tuple:
        """Compute outlet flows with net vapor removal in the level control.

        Parameters
        ----------
        temp, pres : float or ndarray
            Drum temperature [K] and pressure [Pa], scalar or time profile.
        x_i, y_i : ndarray
            Liquid and vapor mole fractions [-], shape (num_species,) or
            (num_times, num_species).
        mol_liq, mol_vap : float or ndarray
            Liquid and vapor holdups [mol], scalar or time profile.
        input_flow : float or ndarray
            Feed molar flow [mol/s], scalar or time profile.
        time : float or ndarray, optional
            Absolute time [s] of the states, scalar or one per row; the
            liquid density then needs data only for the species present at
            that time (see ``_presence_weights``).

        Returns
        -------
        tuple
            Liquid volume [m**3], vapor volume [m**3], liquid outlet [mol/s],
            and gross vapor outlet [mol/s], preserving the time shape.

        Notes
        -----
        ``reflux_ratio`` is the fraction of gross vapor returned as total
        condensate [-], so net removal is (1 - reflux_ratio)*flow_vap [mol/s].
        At the level setpoint, liquid outlet is feed minus this net removal.
        The outlet clips at zero; if feed is below net vapor removal, holdup
        decreases because the controller cannot supply a negative outlet.
        """
        # Volumes
        rho_liq = self.Liquid_1.getDensity(
            mole_frac=x_i, temp=temp, basis='mole',
            weights=self._presence_weights(x_i, time=time))  # [mol/L]

        vol_liq = mol_liq / rho_liq / 1000  # m**3
        vol_vap = mol_vap * gas_ct * temp / pres

        mw_vap = np.dot(self.Vapor_1.mw, y_i.T)

        # Gas density
        rho_mol = pres / gas_ct / temp  # mol/m**3
        rho_gas = rho_mol * (mw_vap / 1000)  # kg/m**3

        delta_p = (pres - self.pres)
        vel_vap = np.sqrt(np.maximum(eps, 2 * delta_p/rho_gas))

        flow_vap = rho_mol * self.area_out * vel_vap * self.cv_gas * self.k_vap

        flow_liq = self.k_liq * (vol_liq - self.vol_liq_set) + input_flow  # [mol/s]
        net_vapor_flow = (1 - self.reflux_ratio) * flow_vap  # [mol/s]
        flow_liq = np.maximum(0, flow_liq - net_vapor_flow)  # [mol/s]

        return vol_liq, vol_vap, flow_liq, flow_vap

    def material_balances(self, time: float, mol_i: np.ndarray,
                          x_liq: np.ndarray, y_vap: np.ndarray,
                          mol_liq: float, mol_vap: float, pres: float,
                          u_int: float, temp: float, u_inputs: dict,
                          flows_out: bool = False) -> tuple:
        """Evaluate component balances and thermodynamic closure.

        Parameters
        ----------
        time : float
            Current time [s]; retained for the unit-model interface.
        mol_i : ndarray
            Component holdups [mol], shape (num_species,).
        x_liq, y_vap : ndarray
            Liquid and vapor mole fractions [-], shape (num_species,).
        mol_liq, mol_vap : float
            Total liquid and vapor holdups [mol].
        pres : float
            Drum pressure [Pa].
        u_int : float
            Internal energy [J]; used by the separate energy balance.
        temp : float
            Drum temperature [K].
        u_inputs : dict
            Inlet ``mole_flow`` [mol/s] and ``mole_frac`` [-] in species order.
        flows_out : bool, optional
            If True, return only outlet flows and liquid volume.

        Returns
        -------
        tuple
            If ``flows_out``: liquid flow [mol/s], vapor flow [mol/s], liquid
            volume [m**3]. Otherwise: component rates [mol/s], algebraic
            residuals, liquid flow [mol/s], vapor flow [mol/s], liquid volume
            [m**3]. Algebraic order is component holdups [mol], VLE [-],
            total holdup [mol], volume [m**3], pressure [Pa].

        Raises
        ------
        ValueError
            If the configured activity model is unknown or the K-value of a
            present species is non-finite. Supercritical present species
            require Henry data [Pa]; absent species (structural presence)
            get a unit ratio and need none.
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``, from
            ``getKeqVLE``: if a present species (structural presence: in
            the liquid at initialization, or fed with a nonzero flow) lacks
            the Antoine or Henry data its ratio needs, or from
            ``getDensity`` if it lacks ``rho_liq``. Absent species need no
            data, whatever their round-off fractions.

        Notes
        -----
        Pressure closure uses the same Henry-based K-values as equilibrium
        for included species above their critical temperatures. The previous
        pressure sum extrapolated Antoine for every included species.
        No species are excluded in this unit. Finite Henry data retain the
        supercritical closure; non-finite K-values raise before entering the
        equilibrium and pressure residuals.
        """

        input_flow = u_inputs['mole_flow']
        input_fracs = u_inputs['mole_frac']

        vol_liq, vol_vap, flow_liq, flow_vap = self.get_mole_flows(
            temp, pres, x_liq, y_vap, mol_liq, mol_vap, input_flow, time=time)

        vol_eqn = vol_liq + vol_vap - self.vol_tot

        if flows_out:
            return flow_liq, flow_vap, vol_liq
        else:
            # Differential eqns
            dmoli_dt = input_flow * input_fracs - flow_liq * x_liq - \
                (1 - self.reflux_ratio) * flow_vap * y_vap

            # Algebraic eqns
            component_bce = mol_liq * x_liq + mol_vap * y_vap - mol_i
            global_bce = mol_liq + mol_vap - sum(mol_i)

            k_i = self.Liquid_1.getKeqVLE(
                temp, pres, x_liq, gamma_model=self.activity_model,
                weights=self._presence_weights(x_liq, time=time))  # [-]
            # Absent species: unit ratio, so x_i = y_i = 0 whatever their data.
            k_i = self._absent_unit_ratios(k_i, time=time)  # [-]
            invalid_k = ~np.isfinite(k_i)
            if np.any(invalid_k):
                species = [name for name, invalid in
                           zip(self.Liquid_1.name_species, invalid_k) if invalid]
                raise ValueError(
                    f"Non-finite VLE K-values for species {species}. Check "
                    "missing or non-finite 'henry_constant' [Pa] data for "
                    "supercritical species before evaluating material balances.")
            equilibria = y_vap - k_i * x_liq

            # sum(y) = sum(K*x) = 1. For gamma_i = 1 and subcritical
            # species, P*K_i = p_sat_i: exactly the old dot(x, p_sat) - P.
            pres_eqn = pres * (np.dot(x_liq, k_i) - 1)  # [Pa]

            alg_balances = np.concatenate((component_bce,  # x_liq
                                           equilibria,  # y_vap
                                           np.array([global_bce]),  # M_L
                                           np.array([vol_eqn]),  # M_V
                                           np.array([pres_eqn])  # P
                                           )
                                          )

            return dmoli_dt, alg_balances, flow_liq, flow_vap, vol_liq

    def energy_balances(self, time: float, flow_liq: float, flow_vap: float,
                        vol_liq: float, u_int: float, temp: float,
                        x_liq: np.ndarray, y_vap: np.ndarray, mol_i: np.ndarray,
                        mol_liq: float, mol_vap: float, pres: float,
                        u_inputs: dict, heat_prof: bool = False
                        ) -> Union[np.ndarray, tuple]:
        """Compute the DAE energy residuals.

        Parameters
        ----------
        time : float
            Current time [s].
        flow_liq : float
            Liquid outlet molar flow rate [mol/s].
        flow_vap : float
            Vapor outlet molar flow rate [mol/s].
        vol_liq : float
            Liquid volume [m**3].
        u_int : float
            Internal energy state [J].
        temp : float
            Unit temperature [K].
        x_liq : ndarray
            Liquid mole fractions [-].
        y_vap : ndarray
            Vapor mole fractions [-].
        mol_i : ndarray
            Component mole holdups [mol].
        mol_liq : float
            Liquid molar holdup [mol].
        mol_vap : float
            Vapor molar holdup [mol].
        pres : float
            Unit pressure [Pa].
        u_inputs : dict
            Inlet controls with ``mole_flow`` [mol/s], ``mole_frac`` [-],
            and ``temp`` [K].
        heat_prof : bool, optional
            Return heat profile terms instead of residuals [-].

        Returns
        -------
        ndarray or tuple
            If ``heat_prof`` is False, residuals for energy rate [J/s] and
            internal energy [J]. If True, net enthalpy flow [J/s] and
            heat-transfer rate [J/s], positive for heat removed. This
            includes condenser removal when reflux is enabled, even when
            ``adiabatic=True`` omits drum-wall heat transfer.

        Raises
        ------
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``. If a
            present species (structural presence) lacks the heat-capacity,
            latent-heat or VLE data of the liquid, vapor, condensate or feed
            enthalpy; absent species need none, and a feed without flow
            needs none.
        """

        input_flow = u_inputs['mole_flow']  # [mol/s]
        input_fracs = u_inputs['mole_frac']  # [-]
        input_temp = u_inputs['temp']  # [K]

        # Molar enthalpies [J/mol]; a species in a feed without flow needs
        # no data.
        h_in = self.Inlet.getEnthalpy(
            temp=input_temp, mole_frac=input_fracs, basis='mole',
            weights=self._feed_weights(input_flow, input_fracs))  # [J/mol]

        h_liq = self.Liquid_1.getEnthalpy(
            temp, mole_frac=x_liq, basis='mole',
            weights=self._presence_weights(x_liq, time=time))  # [J/mol]

        if self.reflux_ratio == 0:
            h_top = self.Vapor_1.getEnthalpy(
                temp, mole_frac=y_vap, basis='mole',
                weights=self._presence_weights(y_vap, time=time))  # [J/mol]
            h_vap = h_top  # [J/mol]
        else:
            weights = self._presence_weights(y_vap, time=time)  # [-]
            temp_bubble = self.Liquid_1.getBubblePoint(
                pres, mole_frac=y_vap, thermo_method=self.activity_model,
                weights=weights)  # [K]
            h_top = self.Liquid_1.getEnthalpy(
                temp=temp_bubble, mole_frac=y_vap, basis='mole',
                weights=weights)  # [J/mol]
            h_vap = self.Vapor_1.getEnthalpy(
                temp, mole_frac=y_vap, basis='mole',
                weights=weights)  # [J/mol]

        # Heat-transfer rate [J/s]
        if self.adiabatic:
            heat_transfer = 0  # [J/s]
        else:
            height_liq = vol_liq / (np.pi/4 * self.diam_tank**2)  # [m]
            area_ht = np.pi * self.diam_tank * height_liq + self.area_base  # [m^2]

            ht_controls = self.Utility.get_inputs(time)  # contains temp_in [K]
            temp_ht = ht_controls['temp_in']  # [K]

            heat_transfer = self.u_ht * area_ht * (temp - temp_ht)  # [J/s]

        # The drum can be adiabatic while its external condenser removes heat.
        q_cond = flow_vap * (h_vap - h_top)  # [J/s], zero without reflux
        heat_transfer += q_cond

        # Net enthalpy flow [J/s]
        flow_term = input_flow * h_in - flow_liq * h_liq - \
            (1 - self.reflux_ratio) * flow_vap * h_top  # [J/s]
        if heat_prof:
            return flow_term, heat_transfer
        else:
            # Compute balances
            duint_dt = flow_term - heat_transfer  # [J/s]

            pressure_volume_energy = pres * self.vol_tot  # [J]
            internal_energy = mol_liq * h_liq + mol_vap * h_vap - \
                pressure_volume_energy - u_int  # [J]

            out_energy = np.array([duint_dt, internal_energy])  # [J/s, J]

            return out_energy

    def unit_model(self, time: float, states: np.ndarray,
                   states_dot: Optional[np.ndarray], sw: Optional[Union[bool, list]],
                   params=None, enrgy_bce: bool = False) -> Union[np.ndarray, tuple]:
        """Evaluate the packed material and energy model.

        Parameters
        ----------
        time : float
            Inlet and utility evaluation time [s].
        states : ndarray
            Packed state with species order and units in ``states_di``:
            holdups [mol], mole fractions [-], pressure [Pa], energy [J],
            and temperature [K].
        states_dot : ndarray or None
            Same layout, with component derivatives [mol/s] and energy
            derivative [J/s]; algebraic entries are ignored.
        sw : bool, list of bool, or None
            Event switches, unused by the residual.
        params : object, optional
            Retained unused parameter argument.
        enrgy_bce : bool, optional
            Return flow and heat terms for duty calculation; default False.

        Returns
        -------
        ndarray or tuple
            ``3*num_species + 5`` residuals, ordered as component rates
            [mol/s], component holdup [mol], VLE [-], total holdup [mol],
            volume [m**3], pressure [Pa], energy rate [J/s], internal-energy
            closure [J]. None derivatives mean zero accumulation (steady
            state). With ``enrgy_bce=True``, return the nested tuple
            ``((liquid_flow, vapor_flow, liquid_volume),
            (net_enthalpy_flow, heat_transfer))`` with units
            (([mol/s], [mol/s], [m**3]), ([J/s], [J/s])). Heat transfer is
            positive for heat removed from the drum and condenser.
        """

        # Decompose states
        di_states = unpack_states(states, self.dim_states, self.name_states)

        # Inputs
        u_inputs = self.get_inputs(time)['Inlet']

        # Material balance
        material_bces = self.material_balances(time, **di_states,
                                               u_inputs=u_inputs)

        material_bce = np.concatenate(material_bces[:-3])
        flow_liq, flow_vap, vol_liq = material_bces[-3:]

        if enrgy_bce:
            energy_terms = self.energy_balances(time, flow_liq, flow_vap, vol_liq,
                                                **di_states,
                                                u_inputs=u_inputs,
                                                heat_prof=True)

            return (flow_liq, flow_vap, vol_liq), energy_terms
        else:
            # Energy balance
            energy_bce = self.energy_balances(time, flow_liq, flow_vap, vol_liq,
                                              **di_states, u_inputs=u_inputs)

            # Concatenate balances
            balances = np.hstack((material_bce, energy_bce))

            if states_dot is not None:
                # Decompose derivatives
                di_dot = unpack_states(states_dot, self.dim_states,
                                       self.name_states)

                dmolesi_dt = di_dot['mol_i']
                duint_dt = di_dot['u_int']

                balances[:self.num_species] = balances[:self.num_species] - \
                    dmolesi_dt
                balances[-2] = balances[-2] - duint_dt

            # Refresh temperature and mole fractions only; derived fields are unchanged.
            self.Liquid_1.temp = di_states['temp']
            self.Vapor_1.temp = di_states['temp']

            self.Liquid_1.mole_frac = di_states['x_liq']  # [-]
            self.Vapor_1.mole_frac = di_states['y_vap']  # [-]

            return balances

    def unit_jacobian(self, c, time, states, sdot, params=None):
        jac_states = self.jac_states(time, states, sdot, params)
        jac_sdot = self.jac_sdot(time, states, sdot, params)

        jac_system = jac_states + c * jac_sdot

        return jac_system

    def init_unit(self) -> tuple:
        """Seed phase equilibrium, holdups, and energy for the DAE solver.

        Returns
        -------
        states_init : ndarray
            Packed state, shape ``(3*num_species + 5,)``. Order: component
            holdups [mol], liquid and vapor mole fractions [-], liquid and
            vapor holdups [mol], pressure
            [Pa], internal energy [J], temperature [K]. Bubble-point seeds
            use the configured activity model. Heat transfer uses the
            wetted wall and base area with the overall coefficient
            [W/m**2/K].
        sdot_init : ndarray
            Same shape; component derivatives [mol/s] and energy derivative
            [J/s] are seeded, with zero placeholders for algebraic entries.
            Initial outflow through the vapor valve is neglected.

        Raises
        ------
        ValueError
            If the liquid volume exceeds the drum volume [m**3].
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``. If a
            present species (in the liquid, or fed at time zero with a
            nonzero flow) lacks data the initialization needs: ``p_vap`` or
            ``henry_constant`` [Pa] for the bubble-point seed, ``cp_liq``
            and Watson data (``delta_hvap``, ``tref_hvap``) for the
            enthalpies, ``rho_liq`` for the volume, or the activity
            parameters of a non-ideal model. Absent species need none.

        Notes
        -----
        The initial liquid volume [m**3] is taken on the model's
        ideal-mixing basis by :func:`_model_liquid_volume`, as in
        :meth:`Evaporator.init_unit` (issue #439).

        The structural presence record is rebuilt from the liquid as it is
        now (a charge edited in place, or the previous solve's outlet), and
        ``get_inputs`` adds the species of the flowing feed.
        """
        self._start_presence(self.num_species, self.Liquid_1.mole_frac)
        temp_bubble_init, y_init = self.Liquid_1.getBubblePoint(
            pres=self.pres, y_vap=True, thermo_method=self.activity_model,
            weights=self._presence_weights(
                self.Liquid_1.mole_frac))  # [K], [-]
        pres_init = self.pres  # [Pa]

        # Bubble-point liquid composition
        x_init = self.Liquid_1.mole_frac  # [-]

        # Moles of phases
        mol_liq = self.Liquid_1.moles  # [mol]
        # [m**3], model basis (see _model_liquid_volume)
        vol_liq = _model_liquid_volume(self.Liquid_1)

        vol_vap = self.vol_tot - vol_liq  # [m**3]
        if vol_vap < 0:
            raise ValueError(r"Drum volume ({:.2f} m3) lower than the liquid "
                             r"volume ({:.2f} m3)".format(self.vol_tot,
                                                          vol_liq))

        mol_vap = self.pres * vol_vap / gas_ct / temp_bubble_init  # [mol]

        self.Vapor_1.updatePhase(moles=mol_vap, mole_frac=y_init,
                                 temp=temp_bubble_init, pres=pres_init)

        # Moles of i
        mol_i = mol_liq * x_init + mol_vap * y_init  # [mol]

        u_inlet = self.get_inputs(0)['Inlet']
        inlet_flow = u_inlet['mole_flow']  # [mol/s]

        # Liquid flow
        flow_liq = self.k_liq * (vol_liq - self.vol_liq_set) + inlet_flow  # [mol/s]
        flow_liq = np.maximum(eps, flow_liq)  # [mol/s]

        dm_init = inlet_flow * u_inlet['mole_frac'] - flow_liq * x_init  # [mol/s]

        # ---------- Energy balance states
        # Enthalpies of the evaluated feed, as in energy_balances (a
        # controller or upstream profile may differ from the stored inlet);
        # a species in a feed without flow needs no data.
        hin_init = self.Inlet.getEnthalpy(
            basis='mole', temp=u_inlet['temp'],
            mole_frac=u_inlet['mole_frac'],
            weights=self._feed_weights(inlet_flow,
                                       u_inlet['mole_frac']))  # [J/mol]
        hliq_init = self.Liquid_1.getEnthalpy(
            temp_bubble_init, basis='mole',
            weights=self._presence_weights(x_init))  # [J/mol]
        hvap_init = self.Vapor_1.getEnthalpy(
            temp_bubble_init, basis='mole', mole_frac=y_init,
            weights=self._presence_weights(y_init))  # [J/mol]

        # Heat transfer
        if self.adiabatic:
            heat_transfer = 0  # [J/s]
        else:
            height_liq = vol_liq / (np.pi/4 * self.diam_tank**2)  # [m]
            area_ht = np.pi * self.diam_tank * height_liq + self.area_base  # [m**2]

            ht_controls = self.Utility.get_inputs(0)
            temp_ht = ht_controls['temp_in']  # [K]

            heat_transfer = self.u_ht * area_ht * (temp_bubble_init -
                                                  temp_ht)  # [J/s]

        # Disregard vapor flow at the beginning (vapor phase is at P_0)
        du_init = inlet_flow * hin_init - flow_liq * hliq_init - heat_transfer  # [J/s]

        # Internal energy
        u_init = (mol_liq * hliq_init + mol_vap * hvap_init
                  - pres_init * self.vol_tot)  # [J]

        # ---------- Retrieve results
        states_init = np.concatenate(
            (mol_i, x_init, y_init, [mol_liq, mol_vap],
             [pres_init, u_init, temp_bubble_init])
        )

        sdot_init = np.zeros_like(states_init)
        sdot_init[:self.num_species] = dm_init
        sdot_init[-2] = du_init

        return states_init, sdot_init

    def _get_tau(self):
        time_upstream = getattr(self.Inlet, 'time_upstream')
        if time_upstream is None:
            time_upstream = [0]

        inputs = self.get_inputs(time_upstream[-1])['Inlet']

        dens_inlet = self.Liquid_1.getDensity(mole_frac=inputs['mole_frac'],
                                              temp=inputs['temp'],
                                              basis='mole') * 1000  # mol/m**3

        volflow_in = inputs['mole_flow'] / dens_inlet
        tau = self.vol_liq_set / volflow_in

        self.tau = tau
        return tau

    def _eval_state_events(self, time, states, sdot, sw):

        events = eval_state_events(
            time, states, sw, self.len_states,
            self.name_states, self.state_event_list, sdot=sdot,
            discretized_model=False, state_map=self.state_map)

        print(events)

        return events

    def solve_unit(self, runtime: float, steady_state: bool = False,
                   verbose: bool = True, sundials_opts: Optional[dict] = None,
                   any_event: bool = True,
                   fsolve_opts: Optional[dict] = None) -> Union[np.ndarray, tuple]:
        """
        Solve ContinuousEvaporator model

        Parameters
        ----------
        runtime : float
            Final time of the dynamic simulation [s]; ignored in steady mode.
        steady_state : bool, optional
            If True, solve all ``3*num_species + 5`` residuals with zero
            component and energy accumulation, plus component holdup, VLE,
            total holdup, volume, pressure, and internal-energy closure.
            Inputs are evaluated at time zero [s]. Default is False.
        verbose : bool, optional
            if True, integrator statistics will be displayed after the model
            is solved. The default is True.
        sundials_opts : dict, optional
            options to be passed to SUNDIALS. For a list of available options,
            visit https://jmodelica.org/assimulo/ODE_CVode.html.
            The default is None.
        any_event : bool, optional
            Stop when any configured terminating event fires; default True.
        fsolve_opts : dict, optional
            Options for SciPy ``fsolve`` in steady mode, such as ``xtol``
            [-], ``maxfev`` (evaluation count), and ``diag`` (positive
            inverse state scales in packed-state order). Defaults to SciPy
            settings. ``full_output`` is always True so convergence is
            checked. Dynamic mode ignores these options.
            Convergence is fixture-dependent; ``diag`` is not a general recipe.

        Returns
        -------
        ndarray or tuple
            Steady mode returns the packed state vector from ``fsolve``.
            Dynamic mode returns times [s] and packed state profiles; state
            units and order are given in ``states_di``.

        Raises
        ------
        RuntimeError
            If the steady solver does not converge or a trial leaves the
            thermophysical property's supported domain. Convergence failures
            include SciPy's ``ier`` and ``msg``; property failures, including
            a missing-data error in the steady balances, preserve the
            original exception as their cause and identify solver scaling
            and initial trust-region settings as diagnostic inputs.
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``. Raised by
            the initialization if a species in the liquid at the start or
            fed with a nonzero flow at the start lacks data that it needs,
            and, in dynamic mode, by the publication of the results (the
            heat-duty replay with its bubble points of the vapor
            composition, and the published phases) if a species present in
            a row lacks data that they need; absent species need none.
        IDAError
            From Assimulo, in dynamic mode, if the balances fail during the
            integration, for example because a species fed only after the
            start lacks data. The missing-data error is not chained to it
            and the species is not named (issue #435).

        Notes
        -----
        A call that raises keeps no results, whatever the exception and
        whether the initialization, the solver or the publication of the
        results raised: the unit's attributes (profiles, results, duties,
        outlet), the structural presence record and the complete state of
        the attached phases (temperature [K], fractions [-], amounts and
        derived fields) are restored to their values before the call
        (issue #425).
        """

        self.args_inputs = (self, self.num_species)

        # A solve that raises keeps no results: everything from the
        # initialization through the solver and the publication of the
        # results is undone.
        before_solve = self._solve_snapshot()
        try:
            result = self._solve_attempt(runtime, steady_state, verbose,
                                         sundials_opts, any_event,
                                         fsolve_opts)
            if not steady_state:
                self.retrieve_results(*result)
        except BaseException:
            self._restore_solve_snapshot(before_solve)
            raise

        return result

    def _solve_attempt(self, runtime: float, steady_state: bool,
                       verbose: bool, sundials_opts: Optional[dict],
                       any_event: bool,
                       fsolve_opts: Optional[dict]
                       ) -> Union[np.ndarray, tuple]:
        """Initialize and solve without keeping results.

        Parameters
        ----------
        runtime, steady_state, verbose, sundials_opts, any_event, fsolve_opts
            As in ``solve_unit``: final time [s] of the dynamic solve and
            the solver options.

        Returns
        -------
        numpy.ndarray or tuple
            The steady packed state, or times [s] and packed state profiles
            of the dynamic solve (units in ``states_di``).

        Raises
        ------
        RuntimeError
            If the steady solver does not converge or a trial leaves the
            property domain, including a missing-data error in the steady
            balances, which it keeps as its cause (see ``solve_unit``).
        MissingPropertyError
            From the initialization, if a present species (structural
            presence) lacks data that it needs.
        IDAError
            From Assimulo, if the dynamic balances fail during the
            integration; a missing-data error there is not chained to it
            (issue #435).

        Notes
        -----
        ``solve_unit`` undoes every change made here if it raises.
        """
        states_initial, sdev_initial = self.init_unit()

        # ---------- Solve
        n_comp = self.num_species
        alg_map = np.concatenate((np.ones(n_comp),
                                  np.zeros(2*n_comp + 3),
                                  [1, 0])
                                 )
        # dM_i/dt, (x_i, y_i), (P, M_L, M_V), (dU/dt, T)
        state_map = [1] + [0] * 2 + [0] * 3 + [1, 0]
        self.state_map = state_map

        # ---------- Solve problem
        if steady_state:
            def obj_fn(states: np.ndarray) -> np.ndarray:
                """Evaluate the full steady residual at the initial inlet time.

                Parameters
                ----------
                states : ndarray
                    Packed states with units and order given in ``states_di``.

                Returns
                -------
                ndarray
                    ``3*num_species + 5`` residuals: zero component rates
                    [mol/s], holdup closure [mol], VLE [-], total holdup
                    [mol], volume [m**3], pressure [Pa], zero energy rate
                    [J/s], and internal-energy closure [J].
                """
                try:
                    return self.unit_model(0, states, states_dot=None, sw=None)
                except (AttributeError, ValueError, RuntimeError) as error:
                    raise RuntimeError(
                        "Steady evaporator property evaluation failed; check "
                        "thermophysical data and fsolve_opts (state scaling "
                        "and initial trust-region factor)."
                    ) from error
            options = {} if fsolve_opts is None else dict(fsolve_opts)
            options['full_output'] = True
            steady_solution, info, ier, msg = fsolve(obj_fn, states_initial,
                                                    **options)
            if ier != 1:
                raise RuntimeError(f"Steady evaporator solve failed: ier={ier}, msg={msg}")

            return steady_solution
        else:
            if self.state_event_list is None:
                def model(t, y, ydot, params=None, energy=False):
                    return self.unit_model(t, y, ydot, None, params, energy)
                problem = Implicit_Problem(model,
                                           states_initial, sdev_initial,
                                           t0=0)
            else:
                switches = [True] * len(self.state_event_list)

                def model(t, y, ydot, sw):
                    return self.unit_model(t, y, ydot, sw, None, False)

                problem = Implicit_Problem(model,
                                           states_initial, sdev_initial,
                                           t0=0, sw0=switches)

                def new_handle(solver, info):
                    return handle_events(
                        solver, info, self.state_event_list,
                        any_event=any_event)

                problem.state_events = self._eval_state_events
                problem.handle_event = new_handle

            problem.algvar = alg_map
            self.alg_map = alg_map
            # problem.jac = self.unit_jacobian

            # See self.name_states
            self.len_states = [self.num_species] * 3 + [1] * 5

            # Set solver
            solver = IDA(problem)
            solver.make_consistent('IDA_YA_YDP_INIT')
            solver.suppress_alg = True
            solver.report_continuously = True

            if sundials_opts is not None:
                for name, val in sundials_opts.items():
                    setattr(solver, name, val)

                    if name == 'time_limit':
                        solver.report_continuously = True

            if not verbose:
                solver.verbosity = 50

            if sundials_opts is not None:
                for name, val in sundials_opts.items():
                    setattr(solver, name, val)

            # Solve
            time, states, sdot = solver.simulate(runtime)

            return time, states

    def retrieve_results(self, time: np.ndarray, states: np.ndarray) -> None:
        """Append a segment's results, update phases, and accumulate duties.

        Parameters
        ----------
        time : array-like
            Segment times [s], shape (num_times,).
        states : ndarray
            Segment states, shape (num_times, sum(dim_states)); units and
            species order are given in ``states_di``.

        Notes
        -----
        Installing the terminal liquid phase preserves simulation history and
        the original charge used by ``reset()``.
        """

        # ---------- Create result object
        time = np.asarray(time)

        dp = unpack_states(states, self.dim_states, self.name_states)
        dp['time'] = time

        inputs_all = self.get_inputs(time)['Inlet']

        vol_liq, vol_vap, flow_liq, flow_vap = self.get_mole_flows(
            dp['temp'], dp['pres'], dp['x_liq'], dp['y_vap'], dp['mol_liq'],
            dp['mol_vap'], inputs_all['mole_flow'], time=time)

        dp['vol_liq'] = vol_liq
        dp['vol_vap'] = vol_vap

        dp['flow_liq'] = flow_liq
        dp['flow_vap'] = flow_vap

        # The liquid outlet is the stream that connections read; absent
        # species are published at exactly zero, as in the Outlet.
        dp['mole_frac'] = self._project_absent(dp['x_liq'], time=time)  # [-]
        dp['mole_flow'] = dp['flow_liq']

        self.profiles_runs.append(dp)

        dp = self.flatten_states()

        self.outputs = dp

        self.result = DynamicResult(self.states_di, self.fstates_di, **dp)

        # ---------- Update phases
        self.Liquid_1.temp = dp['temp'][-1]
        self.Liquid_1.pres = dp['pres'][-1]
        # Absent species leave only round-off in the states; the published
        # phases hold them at exactly zero (structural presence).
        x_final = self._project_absent(dp['x_liq'][-1])  # [-]
        self.Liquid_1.updatePhase(mole_frac=x_final,
                                  moles=dp['mol_liq'][-1])

        self._set_phases(self.Liquid_1)

        # ---------- Output info
        self.Outlet = LiquidStream(self.Liquid_1.path_data,
                                   temp=dp['temp'][-1],
                                   pres=dp['pres'][-1],
                                   mole_frac=x_final,
                                   mole_flow=flow_liq[-1])

        # ---------- Heat duties
        self.get_heat_duty(time, states)

    def get_heat_duty(self, time: np.ndarray, states: np.ndarray) -> None:
        """Calculate the supplied segment's powers and accumulate its duties.

        Parameters
        ----------
        time : array-like
            Segment times [s], shape (num_times,).
        states : ndarray
            Segment states, shape (num_times, sum(dim_states)); units and
            species order are given in ``states_di``.

        Notes
        -----
        ``heat_profile`` [J/s] describes this segment only. Column 0 is
        jacket/utility heat, positive for heat removed from the drum;
        column 1 is heat delivered to the condensing vapor, negative for
        cooling. The condenser uses the gross vapor composition and flow,
        including when reflux is zero. Condenser heat is reported only in
        column 1, though the reflux energy residual includes it.
        Each call adds signed segment integrals to ``_signed_heat_duty``
        [J]. ``heat_duty`` [J] is the magnitude of that cumulative integral,
        so partitioning a fixed trajectory does not change the reported duty.
        ``solve_unit`` restarts continuous simulations at time zero; split
        solves need not reproduce the same trajectory.

        """
        dp = unpack_states(states, self.dim_states, self.name_states)
        heat_bce = np.zeros(len(time))  # [J/s]
        h_liq = np.zeros_like(heat_bce)  # [J/mol]
        h_vap = np.zeros_like(heat_bce)  # [J/mol]

        flow_liq = np.zeros_like(heat_bce)  # [mol/s]
        flow_vap = np.zeros_like(heat_bce)  # [mol/s]

        for ind, row in enumerate(states):
            mat, energy = self.unit_model(time[ind], row,
                                          states_dot=None, sw=None,
                                          enrgy_bce=True)
            heat_bce[ind] = energy[1]

            weights = self._presence_weights(dp['y_vap'][ind],
                                             time=time[ind])  # [-]
            temp_bubble = self.Liquid_1.getBubblePoint(
                pres=dp['pres'][ind],
                mole_frac=dp['y_vap'][ind],
                thermo_method=self.activity_model, weights=weights)  # [K]

            h_liq[ind] = self.Liquid_1.getEnthalpy(
                temp=temp_bubble,
                mole_frac=dp['y_vap'][ind], basis='mole',
                weights=weights)  # [J/mol]

            h_vap[ind] = self.Vapor_1.getEnthalpy(
                temp=dp['temp'][ind],
                mole_frac=dp['y_vap'][ind], basis='mole',
                weights=weights)  # [J/mol]

            flow_liq[ind] = mat[0]
            flow_vap[ind] = mat[1]

        # Condensation duty
        heat_cond_prof = flow_vap * (h_liq - h_vap)  # [J/s]

        if self.reflux_ratio > 0:
            # energy_balances includes condenser removal; subtract it from
            # the jacket column because column 1 already reports that duty.
            heat_bce += heat_cond_prof  # [J/s], jacket/utility only

        self.heat_profile = np.column_stack((heat_bce, heat_cond_prof))  # [J/s]
        segment_duty = trapezoidal_rule(time, self.heat_profile)  # [J]
        self._signed_heat_duty = getattr(self, '_signed_heat_duty', np.zeros(2)) + segment_duty  # [J]
        self.heat_duty = np.abs(self._signed_heat_duty)  # [J]
        # TODO: select duty types from operating temperature.
        self.duty_type = [0, 0]

        self.liqFlowProf = flow_liq
        self.vapFlowProf = flow_vap

    def flatten_states(self):
        out = flatten_states(self.profiles_runs)
        return out

    def plot_profiles(self, pick_comp=None, vol_plot=False, **fig_kwargs):
        """
        Convenience function to plot model solution. Dynamic profiles displayed
        by this funcion are x_liq vs t, y_vap vs t, T/P vs t and
        either mol_liq-mol_vap vs t or vol_liq-vol_vap vs t.

        Parameters
        ----------
        pick_comp : list of int, optional
            indexes of states to be plot. If None, all the states are plotted.
            The default is None.
        vol_plot : bool, optional
            If True, vol_liq-vol_vap vs t is plotted. Otherwise,
            mol_liq-mol_vap vs t is plotted.
        fig_kwargs : keyword arguments
            keyword arguments to be passed to the construction of fig and
            axes objects of matplotlib (``plt.subplots(**fig_kwargs)``).
            Do not use nrows or ncols arguments, since the plot grid is already
            defined by PharmaPy

        Returns
        -------
        fig : TYPE
            figure object.
        ax : numpy array
            grid of axis objects.

        """

        self.flatten_states()

        if pick_comp is None:
            pick_comp = np.arange(self.num_species)
        else:
            pick_comp = pick_comp

        # Fractions
        if pick_comp is None:
            states_plot = ['x_liq', 'y_vap', 'temp', 'pres']
        else:
            states_plot = [['x_liq', pick_comp], ['y_vap', pick_comp], 'temp',
                           'pres']

        ylabels = ['x_liq', 'y_vap', 'T', 'P']
        if vol_plot:
            states_plot += ['vol_liq', 'vol_vap']
            ylabels += ['V_L', 'V_V']
        else:
            states_plot += ['mol_liq', 'mol_vap']
            ylabels += ['N_L', 'N_V']

        fig, ax = plot_function(self, states_plot, fig_map=(0, 1, 2, 2, 3, 3),
                                ylabels=ylabels,
                                nrows=2, ncols=2, **fig_kwargs)

        for axis in ax.flatten():
            axis.xaxis.set_minor_locator(AutoMinorLocator(2))
            axis.yaxis.set_minor_locator(AutoMinorLocator(2))

        fig.text(0.5, 0, 'time (s)', ha='center')

        fig.tight_layout()

        return fig, ax
