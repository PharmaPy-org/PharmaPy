from typing import Optional

import numpy as np
from scipy import linalg
from scipy.optimize import root
from PharmaPy._assimulo import IDA, Implicit_Problem

from PharmaPy.Phases import classify_phases
from PharmaPy.Connections import get_inputs_new
from PharmaPy.Commons import unpack_discretized, retrieve_pde_result, flatten_states
from PharmaPy.Results import DynamicResult
from PharmaPy.Streams import LiquidStream

from PharmaPy.Extractors import BatchExtractor, validate_gamma_method
from PharmaPy.Plotting import plot_distrib
from PharmaPy.Presence import StructuralPresenceMixin

import copy
import functools

from matplotlib.ticker import MaxNLocator, AutoMinorLocator


def get_alg_map(states_di: dict, nstages: int = 1) -> np.ndarray:
    """Build Assimulo algebraic/differential state flags.

    Parameters
    ----------
    states_di : dict
        Insertion-ordered state metadata containing ``dim`` [-] and ``type``
        (``'diff'`` or ``'alg'``). Each type applies to its entire field.
    nstages : int, optional
        Number of dynamic extractor stages [-].

    Returns
    -------
    ndarray
        One flag per state entry [-]: 1 for differential, 0 for algebraic.
        Fields follow dictionary order within each stage; the complete stage
        block repeats ``nstages`` times, matching stage-major state flattening.
    """
    maps = []  # [-], one flag block per field in a single stage
    for val in states_di.values():
        if val['type'] == 'diff':
            maps.append(np.ones(val['dim']))
        elif val['type'] == 'alg':
            maps.append(np.zeros(val['dim']))

    return np.tile(np.hstack(maps), nstages)


def complete_molefrac(mole_frac, mapping):
    """Insert the dependent mole fraction into a reduced composition vector.

    Parameters
    ----------
    mole_frac : ndarray
        Independent mole fractions [-].
    mapping : ndarray of bool
        Boolean mask marking independent components [-].

    Returns
    -------
    ndarray
        Complete mole-fraction vector or matrix [-].
    """
    if mole_frac.ndim == 1:
        out = np.zeros(len(mole_frac) + 1)
        out[mapping] = mole_frac
        out[~mapping] = 1 - sum(mole_frac)
    else:
        out = np.zeros((mole_frac.shape[0], mole_frac.shape[1] + 1))
        out[:, mapping] = mole_frac
        out[:, ~mapping] = 1 - mole_frac.sum(axis=1)

    return out


class DynamicExtractor(StructuralPresenceMixin):
    """Model a staged dynamic liquid-liquid extractor.

    Notes
    -----
    The current formulation keeps interstage light- and heavy-phase outlet
    flows fixed at the inlet values. With unequal component distribution
    coefficients, that constant-flow assumption cannot satisfy every component
    material balance, every component equilibrium relation, and both phase
    normalization equations simultaneously. The implemented closure therefore
    preserves all component balances and phase normalization by replacing the
    last-indexed component's phase-equilibrium relation. Results may depend on
    the order of ``name_species`` until the variable-outlet-flow MESH
    formulation tracked in #195 replaces this approximation.

    The stage mole fractions of both liquid phases are IDA states, which
    difference-quotient Jacobians, the initial ``root`` iterates and
    round-off perturb, so the extractor decides species presence
    structurally (issue #425,
    ``PharmaPy.Presence.StructuralPresenceMixin``): a species is present if
    it has a nonzero fraction in the holdup charge (``Liquid_1``) as it is
    at the start of the solve, if it is the last-indexed species, whose
    fractions close both phases' normalization and so are part of every
    stage by construction, or, from the time of that evaluation on, if the
    feed or the solvent supplies it with a nonzero molar flow (with a stage
    efficiency other than 1, also if the light-phase inlet holds it, flow
    or not, because the Murphree term carries that inlet's composition into
    the first stage). One record
    serves both phases: stage equilibrium distributes a present species
    into both liquids with a finite distribution ratio. The default
    activity-based ``k_fun``, the stage enthalpies and the initial
    densities are weighted by that record, and the published compositions
    hold absent species at exactly zero, so an absent species needs no
    property data other than ``mw``. A user ``k_fun`` receives the state
    fractions unchanged.
    """

    def __init__(self, num_stages, k_fun=None, eff=1, gamma_model='UNIQUAC'):
        """Create a staged dynamic extractor.

        Parameters
        ----------
        num_stages : int
            Number of extractor stages [-].
        k_fun : callable, optional
            Equilibrium function. It must accept light-phase mole fractions
            ``x_light`` [-], heavy-phase mole fractions ``x_heavy`` [-], and
            temperature ``temp`` [K], and return distribution coefficients
            ``K_i = gamma_light/gamma_heavy`` [-].
        eff : float, optional
            Stage efficiency [-].
        gamma_model : {'ideal', 'UNIFAC', 'UNIQUAC'}, optional
            Activity-coefficient method used by the default ``k_fun``.
        """

        self.num_stages = num_stages

        validate_gamma_method(gamma_model, param_name='gamma_model')
        self.gamma_model = gamma_model
        if callable(k_fun):
            self.k_fun = k_fun
        else:
            self.k_fun = self._default_k_fun

        self._Phases = None
        self._Inlet = None
        self.inlets = {}

        self.names_states_in = ('mole_flow', 'temp', 'mole_frac')

        self.heavy_phase = None
        self.eff = eff

        self.fixed_vals = {}
        # self.stream_out = stream_out

        self.profiles_runs = []
        self.oper_mode = 'Continuous'
        self.is_continuous = True
        self.default_output = 'feed'

    def _default_k_fun(self, x_light: np.ndarray, x_heavy: np.ndarray,
                       temp: np.ndarray, time: Optional[float] = None
                       ) -> np.ndarray:
        """Return liquid-liquid distribution coefficients.

        Parameters
        ----------
        x_light : ndarray
            Light-phase liquid mole fractions [-], shape ``(num_comp,)`` or
            ``(num_stages, num_comp)``.
        x_heavy : ndarray
            Heavy-phase liquid mole fractions [-], with the shape of
            ``x_light``.
        temp : ndarray
            Liquid temperature [K].
        time : float, optional
            Absolute evaluation time [s] of the structural presence (issue
            #425): only species recorded at or before it count. None counts
            every recorded species.

        Returns
        -------
        ndarray
            Distribution coefficients ``K_i = gamma_light/gamma_heavy`` [-].
            With a presence record, exactly 1 for an absent species.

        Raises
        ------
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``. With
            UNIQUAC and a presence record, if a present species lacks
            ``ri``, ``qi`` or (when the database has the key) ``qip``.

        Notes
        -----
        Activity coefficients are dimensionless, so their ratio is the
        dimensionless extraction equilibrium constant. The activity model
        is evaluated on the species present at ``time`` only
        (``getActivityCoeff`` with presence weights), as if the others were
        absent from both liquids; they get the neutral coefficient 1 in
        both phases and need no activity parameters. Without a presence
        record every species enters the model, as before.
        """
        gamma_light = self.Liquid_1.getActivityCoeff(
            method=self.gamma_model, mole_frac=x_light, temp=temp,
            weights=self._presence_weights(x_light, time=time))  # [-]

        gamma_heavy = self.Liquid_1.getActivityCoeff(
            method=self.gamma_model, mole_frac=x_heavy, temp=temp,
            weights=self._presence_weights(x_heavy, time=time))  # [-]

        return gamma_light / gamma_heavy  # K_i [-]

    def _evaluate_distribution_coefficients(
            self, x_light: np.ndarray, x_heavy: np.ndarray,
            temp: np.ndarray, time: Optional[float] = None) -> np.ndarray:
        """Evaluate and broadcast distribution coefficients.

        Parameters
        ----------
        x_light : ndarray
            Stage light-phase liquid mole fractions [-].
        x_heavy : ndarray
            Stage heavy-phase liquid mole fractions [-].
        temp : ndarray
            Stage liquid temperatures [K].
        time : float, optional
            Absolute evaluation time [s], passed to the default ``k_fun``
            for its structural presence; a user ``k_fun`` keeps its
            three-argument signature and does not receive it.

        Returns
        -------
        ndarray
            Distribution coefficients broadcast to the stage/component layout
            of ``x_light`` [-].

        Raises
        ------
        ValueError
            If ``k_fun`` returns coefficients that cannot broadcast to the
            stage/component layout of ``x_light``.

        Notes
        -----
        Component-wise callbacks returning ``(num_comp,)`` remain compatible by
        broadcasting across stages. A stricter stage-wise callback contract is
        tracked in #123.
        """
        if self.k_fun == self._default_k_fun:
            k_i = np.asarray(self._default_k_fun(
                x_light, x_heavy, temp, time=time))  # K_i [-]
        else:
            k_i = np.asarray(self.k_fun(x_light, x_heavy, temp))  # K_i [-]
        expected_shape = np.shape(x_light)

        try:
            return np.broadcast_to(k_i, expected_shape)  # K_i [-]
        except ValueError as err:
            raise ValueError(
                "k_fun must return distribution coefficients broadcastable "
                f"to the stage/component shape {expected_shape}; got "
                f"shape {k_i.shape}."
            ) from err

    def nomenclature(self) -> None:
        """Create dynamic extractor state metadata.

        Returns
        -------
        None
            The method populates ``states_di``, ``name_states``,
            ``dim_states``, ``states_in_dict``, ``alg_map``, ``fstates_di``,
            and ``names_states_out`` on the instance.

        Notes
        -----
        The current dynamic-extractor state vector contains light and heavy
        liquid mole fractions [-], internal energy [J], and temperature [K].
        The mole-fraction vectors retain all components for compatibility. The
        final light- and heavy-phase entries are closed by phase
        normalization; in this constant-flow approximation the final
        heavy-phase entry is no longer constrained by its own component
        equilibrium relation, so results may depend on component ordering.
        The light-phase field contains differential variables, except for
        its final dependent component. ``alg_map`` records this exception and
        repeats the stage-major flags used by IDA: independent light-phase
        mole fractions and energy are differential; all other states are
        algebraic. The dictionary metadata retains its existing field layout.

        ``SimulationResult`` counts whole fields by their metadata type, so
        its coarse differential-equation summary includes the dependent final
        light-phase fraction. ``alg_map`` provides the exact per-component
        classification used by IDA.
        """
        num_comp = self.num_comp  # [-]
        name_species = self.name_species
        self.states_di = {
            'x_i': {'dim': num_comp, 'type': 'diff', 'index': name_species,
                    'units': '[-]'},
            'y_i': {'dim': num_comp, 'type': 'alg', 'index': name_species,
                    'units': '[-]'},
            'u_int': {'dim': 1, 'type': 'diff', 'units': '[J]'},
            'temp': {'dim': 1, 'type': 'alg', 'units': '[K]'}
            }

        self.name_states = list(self.states_di.keys())
        self.dim_states = [di['dim'] for di in self.states_di.values()]  # [-]

        states_in_dict = {'mole_flow': 1, 'temp': 1, 'mole_frac': num_comp}  # [-]
        self.states_in_dict = {'Inlet': states_in_dict}

        self.alg_map = get_alg_map(self.states_di, self.num_stages)  # [-]
        # The last x_i row is sum(x_i) - 1, with no time derivative. All y_i
        # rows and temperature are already algebraic in the field metadata.
        light_field_position = self.name_states.index('x_i')
        light_field_offset = sum(self.dim_states[:light_field_position])
        dependent_component = light_field_offset + num_comp - 1
        self.alg_map.reshape(self.num_stages, -1)[:, dependent_component] = 0  # [-]

        self.fstates_di = {}

        self.names_states_out = ('mole_frac', 'mole_flow', 'temp')

    def flatten_states(self):
        pass

    @property
    def Phases(self):
        return self._Phases

    @Phases.setter
    def Phases(self, phases):
        """Attach the liquid holdup charge of the extractor.

        Parameters
        ----------
        phases : LiquidPhase or list or tuple of LiquidPhase
            Holdup charge with mole fractions [-], amount [mol] and
            temperature [K]; the batch extraction of ``initialize_model``
            splits it into the two liquid phases.

        Notes
        -----
        The setter classifies the phases (``Liquid_1``), declares the state
        names and starts the structural presence record from the charge and
        the closure species (issue #425, ``_start_extractor_presence``);
        ``solve_unit`` rebuilds that record at every solve.
        """
        if isinstance(phases, (list, tuple)):
            self._Phases = phases
        else:
            self._Phases = [phases]

        classify_phases(self)

        self.name_species = self.Liquid_1.name_species
        self.num_comp = len(self.name_species)

        self.nomenclature()
        self._start_extractor_presence()

    def _start_extractor_presence(self) -> None:
        """Start the structural presence record from the holdup charge.

        Notes
        -----
        A species is present from the start if its mole fraction in the
        holdup charge (``Liquid_1``) is nonzero, or if it is the
        last-indexed species: its stage fractions in both phases are the
        normalization closures ``1 - sum`` of the others, which the
        constant-flow approximation (#195) leaves nonzero during transients
        whatever the species' feed, so it is part of every stage by
        construction (issue #425). The feed and the solvent add their
        species while they are evaluated (``unit_model``).
        """
        closure = np.zeros(self.num_comp)  # [-], flag of the closure species
        closure[-1] = 1.0
        self._start_presence(self.num_comp, self.Liquid_1.mole_frac, closure)

    def _record_inlets(self, inputs: dict, time: float) -> None:
        """Add the species that the inlets bring into the stages.

        Parameters
        ----------
        inputs : dict
            Inlet values of ``get_inputs``, keyed by ``'feed'`` and/or
            ``'solvent'``, each with ``Inlet`` molar flow [mol/s] and mole
            fractions [-].
        time : float
            Absolute evaluation time [s] of the inputs; a species becomes
            present from it on (latched). The phase roles
            (``target_states``) must be assigned (``initialize_model``).

        Notes
        -----
        A species of an inlet with a nonzero molar flow is present. With a
        stage efficiency ``eff`` other than 1, the first stage's Murphree
        relation ``y_1 = m (x_1 - (1 - eff) x_in)`` uses the light-phase
        inlet composition ``x_in`` whatever that inlet's flow, so the
        species of the light inlet enter the stages by construction of the
        model even when it does not flow, and they are recorded as present
        too (issue #425). Making that term flow-aware would change the
        model's equations, which this presence record must not do.
        """
        for inlet in inputs.values():
            self._record_feed(inlet['Inlet']['mole_flow'],
                              inlet['Inlet']['mole_frac'], time=time)

        presence = getattr(self, '_presence', None)
        if presence is not None and np.any(np.asarray(self.eff) != 1):
            light = self.target_states['light_phase']
            if light in inputs:
                presence.record(inputs[light]['Inlet']['mole_frac'],
                                time=time)

    @property
    def Inlet(self):
        return self._Inlet

    @Inlet.setter
    def Inlet(self, inlet: dict) -> None:
        """Assign the feed and/or solvent inlet streams.

        Parameters
        ----------
        inlet : dict
            Mapping with the key ``'feed'``, ``'solvent'`` or both, each to
            a liquid stream with molar flow [mol/s], temperature [K] and
            mole fractions [-].

        Raises
        ------
        KeyError
            If ``inlet`` has neither ``'feed'`` nor ``'solvent'``.
        TypeError
            If ``inlet`` is not a dictionary.

        Notes
        -----
        The assignment is merged into the held inlets: each key of
        ``inlet`` replaces the stream held under it, and keys it does not
        name are kept, so ``{'solvent': s}`` after ``{'feed': f}`` keeps
        ``f``. A flowsheet ``Connection`` assigns ``{'feed': stream}``, so
        a repeated transfer replaces the earlier feed. Earlier releases
        kept the held stream when a key was assigned again (issue #449).
        """
        if isinstance(inlet, dict):
            if 'feed' not in inlet.keys() and 'solvent' not in inlet.keys():
                raise KeyError(
                    "The passed dictionary must have the key 'feed' "
                    "or the key 'solvent' identifying the passed streams ")

        else:
            raise TypeError(
                "'inlet' object must be a dictionary containing one or "
                "both 'feed' or 'solvent' as keys and "
                "LiquidStreams as values")

        self.inlets = self.inlets | inlet  # the new streams win
        self._Inlet = self.inlets

    def get_stage_dimensions(self, di_states, rhos):
        """Set the dynamic-stage liquid volume.

        Parameters
        ----------
        di_states : dict
            Stage holdups. ``holdup_heavy`` and ``holdup_light`` are [mol].
        rhos : sequence of float
            Heavy- and light-phase molar densities [mol/L].

        Returns
        -------
        None
            The method stores ``self.vol`` [m**3].
        """
        vol = di_states['holdup_heavy'][0] / rhos[0] + \
            di_states['holdup_light'][0] / rhos[1]  # [L]

        self.vol = vol / 1000  # [m**3]

    def get_inputs(self, time):
        inputs = {key: get_inputs_new(time, inlet, self.states_in_dict) for
                  key, inlet in self.Inlet.items()}

        return inputs

    def get_augmented_arrays(
            self, di_states: dict, inputs: dict) -> tuple[np.ndarray, ...]:
        """Build inlet-augmented state arrays for stage balances.

        Parameters
        ----------
        di_states : dict
            Stage mole fractions ``x_i`` and ``y_i`` [-], each with shape
            ``(num_stages, num_comp)``, and ``temp`` [K] with shape
            ``(num_stages,)``.
        inputs : dict
            Feed and solvent dictionaries containing ``Inlet`` values:
            scalar ``mole_flow`` [mol/s], component ``mole_frac`` [-] with
            shape ``(num_comp,)``, and scalar ``temp`` [K].

        Returns
        -------
        x_augm, y_augm : ndarray
            Light/heavy mole fractions [-], shape
            ``(num_stages + 1, num_comp)``. The light inlet precedes the
            light stages; the heavy inlet follows the heavy stages.
        temp_augm : ndarray
            Light-inlet, stage, then heavy-inlet temperatures [K], shape
            ``(num_stages + 2,)``. Inlet temperatures follow the same phase
            roles as inlet compositions, including when the feed is heavy.
        light_flows, heavy_flows : ndarray
            Fixed phase molar flows [mol/s], each with shape
            ``(num_stages + 1,)``.
        """
        light = self.target_states['light_phase']
        heavy = self.target_states['heavy_phase']

        x_in = inputs[light]['Inlet']['mole_frac']  # [-]
        y_in = inputs[heavy]['Inlet']['mole_frac']  # [-]

        temp_in = {key: val['Inlet']['temp'] for key, val in inputs.items()}  # [K]

        x_augm = np.vstack((x_in, di_states['x_i']))  # [-]
        y_augm = np.vstack((di_states['y_i'], y_in))  # [-]
        temp_augm = np.hstack((temp_in[light], di_states['temp'],
                               temp_in[heavy]))  # [K]

        light_flows = np.zeros(self.num_stages + 1)  # [mol/s]
        heavy_flows = np.zeros_like(light_flows)  # [mol/s]

        heavy_in = inputs[heavy]['Inlet']['mole_flow']  # [mol/s]
        light_in = inputs[light]['Inlet']['mole_flow']  # [mol/s]

        light_flows[0] = light_in  # [mol/s]
        light_flows[1:] = light_in  # [mol/s]

        heavy_flows[-1] = heavy_in  # [mol/s]
        heavy_flows[:-1] = heavy_in  # [mol/s]

        augm_arrays = (x_augm, y_augm, temp_augm, light_flows, heavy_flows)

        return augm_arrays

    def unit_model(self, time, states, sdot=None):
        """Evaluate dynamic extractor DAE residuals.

        Parameters
        ----------
        time : float
            Current integration time [s].
        states : ndarray
            Flattened staged states. Mole fractions are [-], internal energy is
            [J], and temperature is [K].
        sdot : ndarray, optional
            Flattened staged state derivatives. Mole-fraction derivatives are
            [1/s] and internal-energy derivatives are [J/s].

        Returns
        -------
        ndarray
            Flattened material and energy residuals. Entries combine
            mole-fraction rates [1/s], equilibrium residuals [-],
            internal-energy rates [J/s], and energy residuals [J].

        Raises
        ------
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``. If a species
            present at ``time`` lacks the ``cp_liq`` of the stage enthalpies
            or, with the default UNIQUAC ``k_fun``, its activity parameters;
            absent species need none (issue #425).

        Notes
        -----
        Side effect: the evaluated feed and solvent add every species they
        supply with a nonzero molar flow (and, with ``eff`` other than 1,
        every species of the light-phase inlet, see ``_record_inlets``) to
        the structural presence record at ``time``, where it stays
        (latched) until the record is rebuilt.
        """

        # ---------- Unpack variables
        di_states = unpack_discretized(states,
                                       self.dim_states, self.name_states)

        if sdot is None:
            di_sdot = sdot
        else:
            di_sdot = unpack_discretized(sdot, self.dim_states,
                                         self.name_states)

        inputs = self.get_inputs(time)
        # Species fed with a nonzero molar flow are present from now on.
        self._record_inlets(inputs, time)

        augm_arrays = self.get_augmented_arrays(di_states, inputs)

        holdup_light = np.ones_like(di_states['temp']) * self.fixed_vals['H_R']  # [mol]
        holdup_heavy = np.ones_like(di_states['temp']) * self.fixed_vals['H_E']  # [mol]

        # ---------- Balances
        material = self.material_balances(time,
                                          augm_arrays=augm_arrays,
                                          di_sdot=di_sdot,
                                          holdup_heavy=holdup_heavy,
                                          holdup_light=holdup_light,
                                          **di_states)

        energy = self.energy_balances(time,
                                      augm_arrays=augm_arrays,
                                      holdup_heavy=holdup_heavy,
                                      holdup_light=holdup_light,
                                      di_sdot=di_sdot,
                                      **di_states)

        balances = np.column_stack(material + energy).ravel()

        return balances

    def material_balances(self, time, x_i, y_i,
                          holdup_light, holdup_heavy,
                          u_int, temp,
                          di_sdot, augm_arrays):
        """Evaluate dynamic extractor material residuals.

        Parameters
        ----------
        time : float
            Current integration time [s].
        x_i, y_i : ndarray
            Stage mole fractions in light and heavy phases [-].
        holdup_light, holdup_heavy : ndarray
            Stage liquid holdups [mol].
        u_int : ndarray
            Stage internal energies [J].
        temp : ndarray
            Stage temperatures [K].
        di_sdot : dict or None
            Assimulo state derivatives, with mole-fraction rates [1/s].
        augm_arrays : tuple of ndarray
            Inlet-augmented mole fractions [-], temperatures [K], and flows
            [mol/s].

        Returns
        -------
        list of ndarray
            Mole-fraction residuals and equilibrium residuals. The first
            ``num_comp - 1`` light-phase entries are mole-fraction rates
            [1/s], the final light-phase entry is ``sum(x_i) - 1`` [-],
            the first ``num_comp - 1`` heavy-phase entries are equilibrium
            residuals [-], and the final heavy-phase entry is
            ``sum(y_i) - 1`` [-].

        Warnings
        --------
        Because light and heavy outlet flows are fixed at their inlet values,
        the final-indexed component's equilibrium relation is omitted. This
        closure keeps both phases normalized, but it is not permutation
        invariant with respect to component order.

        Notes
        -----
        The default ``k_fun`` evaluates the activity model on the species
        present at ``time`` (structural presence, issue #425); an absent
        species gets the unit distribution ratio.
        """

        x_augm, y_augm, temp_augm, light_flows, heavy_flows = augm_arrays
        # Augmented mole fractions [-], temperatures [K], and flows [mol/s].

        # ---------- Equilibrium
        k_ij = self._evaluate_distribution_coefficients(
            x_i, y_i, temp, time=time)  # K_i [-]
        m_ij = k_ij / self.eff  # [-]

        # ---------- Differential block
        # Component flow imbalance [mol/s].
        dxij_dt = y_augm[1:] * heavy_flows[1:, np.newaxis] \
            + x_augm[:-1] * light_flows[:-1, np.newaxis] \
            - y_augm[:-1] * heavy_flows[:-1, np.newaxis] \
            - x_augm[1:] * light_flows[1:, np.newaxis]

        # Effective liquid holdup [mol].
        div = holdup_light[:, np.newaxis] + holdup_heavy[:, np.newaxis] * m_ij

        dxij_dt *= 1 / div  # mole-fraction rate [1/s]

        # ---------- Algebraic block
        equilibrium_alg = m_ij * (x_augm[1:] - x_augm[:-1] * (1 - self.eff)) \
            - y_i  # [-]

        # ---------- Modify outputs for stage two on
        if self.num_stages > 1:
            carry_coeff = holdup_heavy[1:, np.newaxis] * m_ij[1:] * \
                (1 - self.eff) / div[1:]  # [-]

            # The stage closure couples dx_j/dt to dx_(j-1)/dt, so the staged
            # system is lower bidiagonal and resolves by forward substitution.
            for stage in range(1, self.num_stages):
                dxij_dt[stage] += carry_coeff[stage - 1] * dxij_dt[stage - 1]

        if di_sdot is not None:
            dxij_dt = dxij_dt - di_sdot['x_i']  # [1/s]

        # The constant-flow approximation cannot satisfy all component
        # equilibrium equations and both phase closures simultaneously. The
        # final-indexed species is the dependent component until the
        # variable-flow MESH formulation in #195 replaces this closure.
        material_residuals = dxij_dt.copy()  # first num_comp - 1: [1/s]
        equilibrium_residuals = equilibrium_alg.copy()  # [-]
        material_residuals[:, -1] = x_i.sum(axis=1) - 1  # [-]
        equilibrium_residuals[:, -1] = y_i.sum(axis=1) - 1  # [-]

        out = [material_residuals, equilibrium_residuals]  # [mixed, -]

        return out

    def energy_balances(self, time, x_i, y_i,
                        holdup_light, holdup_heavy,
                        u_int, temp,
                        di_sdot, augm_arrays):
        """Evaluate dynamic extractor energy residuals.

        Parameters
        ----------
        time : float
            Current integration time [s].
        x_i, y_i : ndarray
            Stage mole fractions in light and heavy phases [-].
        holdup_light, holdup_heavy : ndarray
            Stage liquid holdups [mol].
        u_int : ndarray
            Stage internal energies [J].
        temp : ndarray
            Stage temperatures [K].
        di_sdot : dict or None
            Assimulo state derivatives. ``u_int`` derivatives are [J/s].
        augm_arrays : tuple of ndarray
            Inlet-augmented mole fractions [-], temperatures [K], and flows
            [mol/s].

        Returns
        -------
        list of ndarray
            Internal-energy rates [J/s] and algebraic energy residuals [J].

        Raises
        ------
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``. If a species
            present at ``time`` lacks ``cp_liq``; absent species need none.

        Notes
        -----
        The stage and inlet enthalpies are weighted with the structural
        presence at ``time`` (issue #425) rather than with the state
        fractions, whose absent entries the solver perturbs; an absent
        species without ``cp_liq`` then adds exactly zero. An inlet species
        that flows was recorded as present before this evaluation
        (``unit_model``), and one that does not flow adds nothing to the
        balances, since its inlet enthalpy is multiplied by a zero flow.
        """

        x_augm, y_augm, temp_augm, light_flows, heavy_flows = augm_arrays

        h_light = self.Liquid_1.getEnthalpy(
            mole_frac=x_augm, temp=temp_augm[:-1], basis='mole',
            weights=self._presence_weights(x_augm, time=time))  # [J/mol]

        h_heavy = self.Liquid_1.getEnthalpy(
            mole_frac=y_augm, temp=temp_augm[1:], basis='mole',
            weights=self._presence_weights(y_augm, time=time))  # [J/mol]

        duint_dt = heavy_flows[1:] * h_heavy[1:] \
            + light_flows[:-1] * h_light[:-1] \
            - heavy_flows[:-1] * h_heavy[:-1] - light_flows[1:] * h_light[1:]  # [J/s]

        if di_sdot is not None:
            duint_dt = duint_dt - di_sdot['u_int']  # [J/s]

        temp_eqns = holdup_heavy * h_heavy[:-1] + holdup_light * h_light[1:] \
            - u_int  # [J]

        out = [duint_dt, temp_eqns]  # [J/s, J]

        return out

    def initialize_model(self):
        """Create thermodynamically consistent initial dynamic states.

        Returns
        -------
        dict
            Initial light/heavy mole fractions [-], stage internal energy [J],
            and temperature [K] for each dynamic extraction stage.

        Raises
        ------
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``. If a species
            present at the start lacks ``rho_liq``, ``cp_liq`` or, with the
            default UNIQUAC ``k_fun``, its activity parameters; absent
            species need none.

        Notes
        -----
        The nested ``BatchExtractor`` receives both the selected ``k_fun`` and
        ``gamma_model``. The full staged solve still depends on broader
        stage-wise ``K_i`` support tracked in #123; the handoff itself is
        covered here without replacing the real thermodynamic callbacks.

        The initial correction uses the same constant-flow approximation as
        ``material_balances``: the final light- and heavy-phase components are
        phase-normalization equations, and the final-indexed heavy component's
        own equilibrium relation is omitted. Initial compositions can therefore
        depend on component ordering until #195 introduces variable outlet
        flows.

        The initial densities and enthalpies, and the default ``k_fun`` of the
        batch split and of the equilibrium correction (whose ``root``
        iterates perturb every fraction), are weighted with the structural
        presence at the start, time 0 [s] (issue #425): absent species need
        no data, and a present species without the density or heat capacity
        the stages need is reported by name here, before IDA starts. This
        holds also when the record has species latched by later
        evaluations, as in a re-initialization after a run; a custom
        ``k_fun`` is passed to the batch split unchanged. That
        presence includes the species that the inlets evaluated at time 0
        bring into the stages (``_record_inlets``, once the phase roles are
        known): those of the inlets that flow and, with ``eff`` other than
        1, those of the light-phase inlet.
        """
        # ---------- Equilibrium calculations
        if self.k_fun == self._default_k_fun:
            # The batch split is part of the initialization at time 0 [s],
            # so the default ratio uses the presence at that time (#425).
            batch_k_fun = functools.partial(self._default_k_fun, time=0.0)
        else:
            batch_k_fun = self.k_fun
        extr = BatchExtractor(k_fun=batch_k_fun,
                              gamma_method=self.gamma_model)
        extr.Phases = copy.deepcopy(self.Liquid_1)

        extr.solve_unit()
        res = extr.result  # extraction result state

        # ---------- Discriminate heavy and light phases
        rhos_streams = {key: obj.getDensity(basis='mole')
                        for key, obj in self.Inlet.items()}  # [mol/L]

        rhos_holdups = [res.rho_heavy, res.rho_light]  # [mol/L]

        idx_raff = np.argmin(abs(rhos_holdups - rhos_streams['feed']))  # [-]

        target_states = {'x_light': 'x_i', 'x_heavy': 'y_i'}

        if idx_raff == 0:
            target_states['light_phase'] = 'solvent'
            target_states['heavy_phase'] = 'feed'

        elif idx_raff == 1:
            target_states['light_phase'] = 'feed'
            target_states['heavy_phase'] = 'solvent'

        self.target_states = target_states
        # Species that the inlets bring in at the start are present, so the
        # initial densities and enthalpies below check their data (#425).
        self._record_inlets(self.get_inputs(0.0), 0.0)

        # Store fixed values
        self.fixed_vals['H_R'] = res.mol_light  # [mol]
        self.fixed_vals['H_E'] = res.mol_heavy  # [mol]

        # ---------- Create dictionary of initial values
        di_init = {}

        for equiv, name in target_states.items():
            if 'phase' not in equiv:
                attr = getattr(res, equiv)
                if isinstance(attr, np.ndarray):
                    if attr.ndim > 0:
                        attr = np.tile(attr, (self.num_stages, 1))
                    else:
                        attr = np.repeat(attr[0], self.num_stages)
                else:
                    attr = np.ones(self.num_stages) * attr

                di_init[name] = attr  # [-]

        di_init['temp'] = self.Liquid_1.temp * np.ones(self.num_stages)  # [K]

        keys_frac = ('y_i', 'x_i')

        # Present species at the start (time 0 [s]) need the stage data,
        # whatever their initial fraction (issue #425).
        rhos_holdups = [
            self.Liquid_1.getDensity(
                mole_frac=di_init[key], basis='mole', temp=di_init['temp'],
                weights=self._presence_weights(di_init[key], time=0.0))
            for key in keys_frac]  # [mol/L]

        # Energy balance calculations
        h_light = self.Liquid_1.getEnthalpy(
            mole_frac=di_init['x_i'], temp=di_init['temp'], basis='mole',
            weights=self._presence_weights(di_init['x_i'],
                                           time=0.0))  # [J/mol]

        h_heavy = self.Liquid_1.getEnthalpy(
            mole_frac=di_init['y_i'], temp=di_init['temp'], basis='mole',
            weights=self._presence_weights(di_init['y_i'],
                                           time=0.0))  # [J/mol]

        u_int = res.mol_light * h_light + res.mol_heavy * h_heavy  # [J]

        di_init['u_int'] = u_int  # [J]

        # ---------- Equilibrium correction for x_i and y_i
        def equilibrium_eqns(fracs: np.ndarray, temp: np.ndarray,
                             mol_i: np.ndarray, holdups: list) -> np.ndarray:
            """Evaluate the stage equilibrium-correction residuals.

            Parameters
            ----------
            fracs : numpy.ndarray
                Stage light- then heavy-phase mole fractions [-], flattened
                stage by stage, shape ``(num_stages * 2 * num_comp,)``.
            temp : numpy.ndarray
                Stage temperatures [K], shape ``(num_stages,)``.
            mol_i : numpy.ndarray
                Species amounts of one stage [mol], shape ``(num_comp,)``.
            holdups : list of float
                Light- and heavy-phase holdups [mol].

            Returns
            -------
            numpy.ndarray
                Component-balance residuals divided by the light holdup and
                equilibrium residuals [-], with the last component of each
                phase replaced by its normalization, flattened like
                ``fracs``.
            """
            var = fracs.reshape(self.num_stages, -1)  # mole fractions [-]

            xi = var[:, :self.num_comp]  # light-phase mole fractions [-]
            yi = var[:, self.num_comp:2 * self.num_comp]  # heavy-phase [-]

            HR, HE = holdups  # liquid holdups [mol]

            x_eqns = (HR * xi + HE * yi - mol_i) / HR  # [-]

            k_ij = self._evaluate_distribution_coefficients(
                xi, yi, temp, time=0.0)  # K_i [-], presence at the start
            m_ij = k_ij / self.eff  # [-]

            y_eqns = np.zeros_like(yi)  # equilibrium residuals [-]
            y_eqns[0] = m_ij[0] * xi[0] - yi[0]

            if self.num_stages > 1:
                y_eqns[1:] = m_ij[1:] * (
                    xi[1:] - xi[:-1] * (1 - self.eff)
                ) - yi[1:]

            x_eqns[:, -1] = xi.sum(axis=1) - 1  # [-]
            y_eqns[:, -1] = yi.sum(axis=1) - 1  # [-]

            eqns = np.column_stack((x_eqns, y_eqns)).ravel()  # [-]

            return eqns

        holdups = [res.mol_light, res.mol_heavy]  # [mol]
        mol_i = res.mol_light * res.x_light + res.mol_heavy * res.x_heavy  # [mol]

        args_eq = (di_init['temp'], mol_i, holdups)  # [K], [mol], [mol]
        x0 = np.column_stack((di_init['x_i'], di_init['y_i'])).ravel()  # [-]

        frac_result = root(equilibrium_eqns, x0, args=args_eq)

        frac_noneq = frac_result.x  # mole fractions [-]

        frac_noneq = frac_noneq.reshape(self.num_stages, -1)  # [-]

        x_noneq, y_noneq = np.split(frac_noneq, 2, axis=1)  # [-]

        di_init['x_i'] = x_noneq
        di_init['y_i'] = y_noneq

        return di_init

    def solve_unit(self, runtime, sundials_opts=None, verbose=True):
        """Integrate the dynamic extractor DAE model.

        Parameters
        ----------
        runtime : float
            Simulation duration [s].
        sundials_opts : dict, optional
            Solver option names and values forwarded to IDA.
        verbose : bool, optional
            If False, suppress solver output [-].

        Returns
        -------
        time : ndarray
            Simulated time points [s].
        states : ndarray
            Flattened state history. Mole fractions are [-], internal energy is
            [J], and temperature is [K].

        Raises
        ------
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``, before IDA
            starts: if a species present at the start (in the holdup charge,
            the closure species, or fed with a nonzero flow) lacks the
            ``rho_liq`` or ``cp_liq`` of the stages or, with the default
            UNIQUAC ``k_fun``, its activity parameters. Absent species need
            no property data other than ``mw`` (issue #425). A species first
            fed later without data is reported by IDA instead, whose error
            does not name it (issue #435).

        Notes
        -----
        The structural presence record is rebuilt from the holdup charge as
        it is at the start of the call. A solve that raises, in the
        initialization, the integration or the publication of the results,
        keeps no results: the extractor's attributes (``fixed_vals``,
        ``target_states``, ``result``, ``outputs``, ``Outlet`` and
        ``profiles_runs``) and its presence record are restored to their
        values before the call.
        """
        # A failed solve keeps no results: the extractor and its presence
        # record are restored if anything below raises (issue #425).
        before_solve = self._solve_snapshot()
        try:
            self._start_extractor_presence()

            di_init = self.initialize_model()
            di_init = {key: di_init[key] for key in self.name_states}

            init_states = np.column_stack(tuple(di_init.values()))
            init_states = init_states.ravel()

            # Evaluated before IDA, so that missing data of the species fed
            # at the start are reported by name (issue #425).
            init_deriv = self.unit_model(0, init_states)

            problem = Implicit_Problem(self.unit_model,
                                       y0=init_states, yd0=init_deriv)

            problem.algvar = self.alg_map

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

            time, states, sdot = solver.simulate(runtime)

            self.retrieve_results(time, states)
        except BaseException:
            self._restore_solve_snapshot(before_solve)
            raise

        return time, states

    def retrieve_results(self, time, states):
        """Store dynamic extractor profiles and outlet streams.

        Parameters
        ----------
        time : array_like
            Simulated time points [s].
        states : ndarray
            Flattened state history. Mole fractions are [-], internal energy is
            [J], and temperature is [K].

        Returns
        -------
        None
            The method updates ``result``, ``outputs``, ``Outlet``, and
            ``profiles_runs`` in place.

        Raises
        ------
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``. If a species
            present at the final time lacks the ``rho_liq`` that the outlet
            streams need; absent species need none.

        Notes
        -----
        The published compositions ``x_i`` and ``y_i`` (``profiles_runs``,
        ``result``, ``outputs`` and ``Outlet``) hold the species absent at
        each row's time (structural presence, issue #425) at exactly zero
        instead of the solver's round-off, so the outlets need no data for
        them; the ``states`` returned by ``solve_unit`` keep the solver
        values.
        """
        time = np.asarray(time)

        indexes = {key: self.states_di[key].get('index', None)
                   for key in self.name_states}

        di = unpack_discretized(states, self.dim_states, self.name_states,
                                indexes=indexes)
        # Species absent at a row's time are published at exactly zero.
        for key in ('x_i', 'y_i'):
            # [-], shape (num_times, num_stages, num_comp)
            fractions = np.stack(list(di[key].values()), axis=-1)
            num_times, num_stages = fractions.shape[:2]
            projected = self._project_absent(
                fractions.reshape(num_times * num_stages, -1),
                time=np.repeat(time, num_stages)).reshape(
                    fractions.shape)  # [-]
            di[key] = {name: projected[..., idx]
                       for idx, name in enumerate(di[key])}  # [-]
        di['stage'] = np.arange(1, self.num_stages + 1)
        di['time'] = time

        self.profiles_runs.append(di)

        self.result = DynamicResult(self.states_di, self.fstates_di, **di)

        # Create time-dependent results for the last stage (outlet)
        di_last = retrieve_pde_result(di, x_name='stage', x=self.num_stages)

        frac_keys = ('x_i', 'y_i')
        frac_last = {key: np.column_stack(
            list(di_last[key].values())) for key in frac_keys}

        inputs = self.get_inputs(time[-1])

        flow_light = inputs[self.target_states['light_phase']]['Inlet']['mole_flow']
        flow_heavy = inputs[self.target_states['heavy_phase']]['Inlet']['mole_flow']

        kws = {
            self.target_states['light_phase']: {
                'mole_flow': flow_light,
                'mole_frac': frac_last['x_i'][-1]},
            self.target_states['heavy_phase']: {
                'mole_flow': flow_heavy,
                'mole_frac': frac_last['y_i'][-1]}
                }

        self.Outlet = {key: LiquidStream(
            self.Liquid_1.path_data, temp=di_last['temp'][-1], **kws[key])
            for key in ('feed', 'solvent')}

        # Complete di_last with appropriate keys
        if self.target_states['light_phase'] == self.default_output:
            di_last['mole_frac'] = frac_last['x_i']
        else:
            di_last['mole_frac'] = frac_last['y_i']

        di_last['mole_flow'] = np.repeat(kws[self.default_output]['mole_flow'],
                                         len(time))

        self.outputs = di_last

    def plot_profiles(self, times=None, stages=None, pick_comp=None,
                      **fig_kwargs):
        """Plot dynamic extractor state profiles.

        Parameters
        ----------
        times : array_like, optional
            Time points to plot [s].
        stages : array_like, optional
            Stage positions to plot [-].
        pick_comp : sequence, optional
            Component indices or names to include [-].
        **fig_kwargs
            Matplotlib figure keyword arguments.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Created figure.
        axis : ndarray
            Axes containing mole-fraction [-] and temperature [K] profiles.
        """

        if pick_comp is None:
            states_plot = ('x_i', 'y_i', 'temp')
            num_species_plot = len(self.name_species)
        else:
            states_plot = (('x_i', pick_comp), ('y_i', pick_comp), 'temp')
            num_species_plot = len(pick_comp)

        ylabels = ('x_i', 'y_i', 'T')

        fig, axis = plot_distrib(self, states_plot, 'stage', times=times,
                                 x_vals=stages, nrows=1, ncols=3,
                                 ylabels=ylabels, **fig_kwargs)

        if times is not None:
            marks = ('s', 'o', '^', 'd', '+')

            for ax in axis:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            for ct, ax in enumerate(axis):
                lines = ax.lines

                for ind, line in enumerate(lines):
                    mark = marks[ind % num_species_plot]

                    line.set_marker(mark)
                    line.set_markerfacecolor('None')

                ax.yaxis.set_minor_locator(AutoMinorLocator(2))

                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        elif stages is not None:
            fig.text(0.5, 0, '$t$ (s)', ha='center')

            for ax in axis:
                ax.xaxis.set_minor_locator(AutoMinorLocator(2))

        for ax in axis:
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        fig.tight_layout()

        return fig, axis
