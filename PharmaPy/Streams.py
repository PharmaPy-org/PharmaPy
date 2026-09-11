# -*- coding: utf-8 -*-
"""
Created on Wed May 27 10:12:13 2020

@author: dcasasor
"""

from typing import Optional
from numpy.typing import ArrayLike

from PharmaPy.Phases import LiquidPhase, SolidPhase, VaporPhase, classify_phases
from PharmaPy.Interpolation import NewtonInterpolation
from PharmaPy.Results import DynamicResult

from scipy.interpolate import CubicSpline
import numpy as np


def Interpolation(t_data, y_data, time, newton=True, num_points=3):
    idx_time = np.argmin(abs(time - t_data))

    idx_lower = max(0, idx_time - 1)
    idx_upper = min(len(t_data) - 1, idx_lower + num_points)

    t_interp = t_data[idx_lower:idx_upper]
    y_interp = y_data[idx_lower:idx_upper]

    # Newton interpolation (quadratic, three points)
    interp = NewtonInterpolation(t_interp, y_interp)
    y_target = interp.evalPolynomial(time)

    return y_target


class BatchToFlowConnector:
    def __init__(self, cycle_time, flow_mult=1):
        self.flow_mult = flow_mult
        self.cycle_time = cycle_time

        self._Phases = None
        self._Inlet = None
        self.oper_mode = 'Batch'

        self.is_continuous = True

    @property
    def Phases(self):
        return self._Phases

    @Phases.setter
    def Phases(self, phases):
        self.has_solids = False

        if isinstance(phases, (list, tuple)):
            self._Phases = phases
        elif 'LiquidPhase' in phases.__class__.__name__:
            self._Phases = [phases]
        elif 'Slurry' in phases.__class__.__name__:
            self._Phases = phases.Phases
            self.has_solids = True

        classify_phases(self)
        self.nomenclature()

    def nomenclature(self):
        comp = self.Liquid_1.name_species
        self.states_di = {
            'mass_frac': {'dim': len(comp), 'index': comp},
            'mass_flow': {'dim': 1, 'units': 'kg/s'},
            'temp': {'dim': 1, 'units': 'K'}}

        self.names_states_out = ('temp', 'pres', 'mass_frac', 'mass_flow')

    def flatten_states(self):
        pass

    def solve_unit(self):
        self.retrieve_results()

    def retrieve_results(self):

        fields = ('temp', 'pres', 'mass_frac', 'path_data')

        # if self.cycle_time is None:
        #     self.cycle_time = self.Phases[0].time_upstream

        if self.has_solids:
            pass  # TODO: add this if continuous downstream solid processing is made available
        else:
            kw_phase = {key: getattr(self.Liquid_1, key) for key in fields}

            kw_phase['path_thermo'] = kw_phase.pop('path_data')
            mass_flow = self.Liquid_1.mass / self.cycle_time * self.flow_mult
            outlet = LiquidStream(**kw_phase, mass_flow=mass_flow)

            kw_phase.pop('path_thermo')
            kw_phase['mass_flow'] = mass_flow

        # kw_phase['time'] = [self.cycle_time]
        kw_phase['time'] = None

        self.result = DynamicResult(self.states_di, **kw_phase)

        self.Outlet = outlet
        self.outputs = kw_phase


class LiquidStream(LiquidPhase):
    def __init__(self, path_thermo=None, temp=298.15, pres=101325,
                 mass_flow=0, vol_flow=0, mole_flow=0,
                 controls=None, args_control=None,
                 mass_frac=None, mole_frac=None, mass_conc=None, mole_conc=None,
                 name_solv=None, num_interpolation_points=3,
                 verbose=True, check_input=True):

        super().__init__(path_thermo, temp, pres,
                         mass=mass_flow, vol=vol_flow, moles=mole_flow,
                         mass_frac=mass_frac, mole_frac=mole_frac,
                         mass_conc=mass_conc, mole_conc=mole_conc,
                         name_solv=name_solv,
                         verbose=verbose, check_input=check_input)

        self.mass_flow = self.mass
        self.vol_flow = self.vol
        self.mole_flow = self.moles

        self._DynamicInlet = None
        self.controllable = ('mass_flow', 'mole_flow', 'vol_flow', 'temp')

        # del self.mass
        # del self.vol
        # del self.moles

        # Outputs from upstream UO
        self.y_upstream = None
        self.time_upstream = None
        self.bipartite = None

        # Controls
        if controls is None:
            controls = {}
        else:
            if args_control is None:
                args_control = {key: () for key in controls.keys()}

            update_dict = {}
            for key, fun in controls.items():
                update_dict[key] = fun(0, *args_control[key])

            self.updatePhase(**update_dict)

        self.controls = controls
        self.args_control = args_control

        self.num_interpolation_points = num_interpolation_points

    @property
    def DynamicInlet(self):
        return self._DynamicInlet

    @DynamicInlet.setter
    def DynamicInlet(self, dynamic_object: Optional[object]) -> None:
        """Attach an inlet controller or restore static stream inputs.

        Parameters
        ----------
        dynamic_object : object or None
            Object exposing ``evaluate_inputs(time)`` with time [s].
            None clears the controller; static flow [kg/s, mol/s, m**3/s]
            and temperature [K] attributes then supply evaluated inputs.
        """
        self._DynamicInlet = dynamic_object
        if dynamic_object is None:
            return

        dynamic_object.controllable = self.controllable
        dynamic_object.parent_instance = self

    def InterpolateInputs(self, time):
        if isinstance(time, (float, int)):
            # Assume steady state for extrapolation
            time = min(time, self.time_upstream[-1])

            y_interpol = Interpolation(self.time_upstream, self.y_inlet,
                                       time,
                                       num_points=self.num_interpolation_points)
        else:
            interpol = CubicSpline(self.time_upstream, self.y_inlet)
            flags_interpol = time > self.time_upstream[-1]

            if any(flags_interpol):
                time_interpol = time[~flags_interpol]
                y_interp = interpol(time_interpol)

                y_extrapol = np.tile(y_interp[-1],
                                     (sum(flags_interpol), 1))
                y_interpol = np.vstack((y_interp, y_extrapol))
            else:
                y_interpol = interpol(time)

        return y_interpol

    def updatePhase(self, concentr: Optional[ArrayLike] = None,
                    mass_conc: Optional[ArrayLike] = None,
                    mass_frac: Optional[ArrayLike] = None,
                    mole_frac: Optional[ArrayLike] = None,
                    vol_flow: Optional[float] = None,
                    mass_flow: Optional[float] = None,
                    mole_flow: Optional[float] = None) -> None:
        """Update liquid composition and reconcile explicit flow rates.

        Parameters
        ----------
        concentr, mass_conc : array-like, optional
            Species molar [mol/L] and mass [kg/m**3] concentrations, shape
            ``(num_species,)``. ``concentr`` is the liquid stream's name for
            the phase's ``mole_conc`` argument.
        mass_frac, mole_frac : array-like, optional
            Species mass and mole fractions [-], shape ``(num_species,)``.
        vol_flow, mass_flow, mole_flow : float, optional
            Volume [m**3/s], mass [kg/s], and molar [mol/s] flow rates.
            None and zero mean no explicit flow was supplied.

        Returns
        -------
        None
            The stream state and flow aliases are updated in place.

        Notes
        -----
        Only explicit positive flows participate in mass, volume, then mole
        precedence. For scalar inventory and one-dimensional composition,
        composition-only updates conserve mass flow [kg/s] and recompute the
        dependent flows. Composition precedence, solvent completion, and
        unchanged profile inventories follow :meth:`LiquidPhase.updatePhase`.
        A no-argument update retains the state and flow values. As in previous
        versions, updates delete the inherited ``mass``, ``vol``, and ``moles``
        attributes; use the flow aliases to read the reconciled amounts.
        Zero means "not supplied", so this method cannot set flow to zero.
        """
        # The parent reconciles amounts on the stream's per-second basis.
        self.mass = self.mass_flow  # [kg/s]
        self.vol = self.vol_flow  # [m**3/s]
        self.moles = self.mole_flow  # [mol/s]

        super().updatePhase(
            mole_conc=concentr, mass_conc=mass_conc,
            mass_frac=mass_frac, mole_frac=mole_frac,
            vol=0 if vol_flow is None else vol_flow,
            mass=0 if mass_flow is None else mass_flow,
            moles=0 if mole_flow is None else mole_flow,
        )

        self.mass_flow = self.mass  # [kg/s]
        self.vol_flow = self.vol  # [m**3/s]
        self.mole_flow = self.moles  # [mol/s]

        del self.mass
        del self.vol
        del self.moles

    def evaluate_inputs(self, time):
        if self.DynamicInlet is None:
            inputs = {}
            for attr in self.controllable:
                inputs[attr] = getattr(self, attr)

        else:
            inputs = self.DynamicInlet.evaluate_inputs(time)

        return inputs


class SolidStream(SolidPhase):
    def __init__(self, path_thermo=None, temp=298.15, pres=101325,
                 mass_flow=0, mass_frac=None,
                 distrib=None, x_distrib=None, kv=1):

        super().__init__(path_thermo, temp, pres=pres,
                         mass=mass_flow, mass_frac=mass_frac,
                         # moments=moments,
                         distrib=distrib, x_distrib=x_distrib, kv=kv)

        self.mass_flow = self.mass
        # self.vol_flow = self.vol
        self.mole_flow = self.moles

        # del self.mass
        # # del self.vol
        # del self.moles

    def updatePhase(self, x_distrib: Optional[np.ndarray] = None,
                    distrib: Optional[np.ndarray] = None,
                    mass: Optional[float] = None,
                    moments: Optional[np.ndarray] = None,
                    mass_flow: Optional[float] = None) -> None:
        """Update solid-stream state and synchronize flow-basis aliases.

        Parameters
        ----------
        x_distrib : numpy.ndarray, optional
            Crystal-size grid with shape ``(num_sizes,)`` [um].
        distrib : numpy.ndarray, optional
            Total-population number distribution with shape ``(num_sizes,)``
            [#/um].
        mass : float, optional
            Solid mass flow represented by the inherited phase amount
            attribute [kg/s].
        moments : numpy.ndarray, optional
            Total-population moments with shape ``(num_moments,)``. Entry
            ``n`` has the inherited solid-phase unit [m**n], with order zero
            a crystal count [-].
        mass_flow : float, optional
            Additive flow-oriented alias for ``mass`` [kg/s]. Specify at most
            one of ``mass`` and ``mass_flow``.

        Raises
        ------
        ValueError
            If both ``mass`` and ``mass_flow`` are supplied.

        Notes
        -----
        ``SolidStream`` preserves the historical ``SolidPhase`` amount
        attributes as flow-rate storage. After the phase update, ``mass`` and
        ``moles`` therefore map to ``mass_flow`` [kg/s] and ``mole_flow``
        [mol/s], respectively. When both ``vol`` and an existing ``vol_flow``
        alias are present, ``vol_flow`` is refreshed from ``vol`` [m**3/s].
        """
        if mass is not None and mass_flow is not None:
            raise ValueError("Specify either 'mass' or 'mass_flow', not both")

        resolved_mass_flow = mass if mass_flow is None else mass_flow  # [kg/s]
        super().updatePhase(
            x_distrib=x_distrib,
            distrib=distrib,
            mass=resolved_mass_flow,
            moments=moments,
        )

        self.mass_flow = self.mass  # [kg/s]
        self.mole_flow = self.moles  # [mol/s]
        if hasattr(self, 'vol_flow') and hasattr(self, 'vol'):
            self.vol_flow = self.vol  # [m**3/s]


class VaporStream(VaporPhase):
    """Homogeneous vapor flow with ideal-gas amounts and density.

    Phase amount attributes are retained as per-second flow quantities and
    synchronized with ``mass_flow``, ``vol_flow``, and ``mole_flow`` on updates.
    """
    def __init__(self, path_thermo: Optional[str] = None,
                 temp: float = 298.15, pres: float = 101325,
                 mass_flow: float = 0, vol_flow: float = 0, mole_flow: float = 0,
                 mass_frac: Optional[np.ndarray] = None,
                 mole_frac: Optional[np.ndarray] = None,
                 mole_conc: Optional[np.ndarray] = None,
                 check_input: bool = True, verbose: bool = True) -> None:
        """Initialize a vapor stream and its consistent flow aliases.

        Parameters
        ----------
        path_thermo : str, optional
            Path to the species thermophysical-property JSON file.
        temp, pres : float, optional
            Temperature [K] and pressure [Pa], defaulting to standard ambient
            temperature and one standard atmosphere.
        mass_flow, vol_flow, mole_flow : float, optional
            Mass [kg/s], volume [m**3/s], and molar [mol/s] flow. The first
            positive value in that order controls; all zeros give zero flow.
        mass_frac, mole_frac : array-like, optional
            Species mass and mole fractions [-], shape ``(num_species,)`` or
            ``(num_points, num_species)``.
        mole_conc : array-like, optional
            Species molar concentrations [mol/L], shape ``(num_species,)`` or
            ``(num_points, num_species)``.
        check_input, verbose : bool, optional
            Enable zero-flow warnings and composition diagnostics, respectively.

        Raises
        ------
        ValueError
            If no composition measure is supplied, a flow is negative, or
            positive mass or volume flow is requested with zero molar mass.
        RuntimeWarning
            If more than one composition measure is supplied.

        Warns
        -----
        RuntimeWarning
            If all flows are zero and input checking is enabled.

        Notes
        -----
        Exactly one composition measure is required. Supplied concentrations
        are retained; concentrations derived from fractions use the converters'
        liquid basis, not gas-EOS values. The first positive flow applies to
        every composition row. Composition-dependent flows have one value per
        row; shared flows independent of composition remain scalar.
        """

        super().__init__(path_thermo, temp, pres,
                         mass=mass_flow, vol=vol_flow, moles=mole_flow,
                         mass_frac=mass_frac, mole_frac=mole_frac,
                         mole_conc=mole_conc, check_input=check_input,
                         verbose=verbose)

        self.mass_flow = self.mass  # [kg/s]
        self.vol_flow = self.vol  # [m**3/s]
        self.mole_flow = self.moles  # [mol/s]

        self.controllable = ('mass_flow', 'mole_flow', 'vol_flow', 'temp')

        self._DynamicInlet = None

    def updatePhase(self, mole_conc: Optional[np.ndarray] = None,
                    mass_conc: Optional[np.ndarray] = None,
                    mass_frac: Optional[np.ndarray] = None,
                    mole_frac: Optional[np.ndarray] = None,
                    vol: float = 0, mass: float = 0, moles: float = 0,
                    vol_flow: Optional[float] = None,
                    mass_flow: Optional[float] = None,
                    mole_flow: Optional[float] = None,
                    temp: Optional[float] = None,
                    pres: Optional[float] = None) -> None:
        """Update vapor flow using phase-style amounts or flow aliases.

        Parameters
        ----------
        mole_conc, mass_conc : ndarray, optional
            Species molar [mol/L] and mass [kg/m**3] concentrations, shape
            ``(num_species,)`` or ``(num_points, num_species)``. Supplied
            concentrations retain their basis.
        mass_frac, mole_frac : ndarray, optional
            Species mass and mole fractions [-], shape ``(num_species,)`` or
            ``(num_points, num_species)``.
        vol, mass, moles : float, optional
            Phase-style names for volume [m**3/s], mass [kg/s], and molar
            [mol/s] flow. Zero means no amount was supplied.
        vol_flow, mass_flow, mole_flow : float, optional
            Flow aliases [m**3/s], [kg/s], and [mol/s], respectively. Each
            non-None alias replaces its corresponding phase-style argument.
        temp, pres : float, optional
            Temperature [K] and pressure [Pa]; None retains stored state.

        Raises
        ------
        ValueError
            If a positive phase-style amount and a non-None alias are both
            supplied for the same quantity, any explicit amount or flow is
            negative, or positive mass or volume flow has zero molar mass.

        Notes
        -----
        Only explicit amounts participate in mass, volume, then moles
        precedence. With no positive explicit amount, state changes conserve
        molar flow. All flow aliases are refreshed after every update; the
        phase-style attributes remain available on the same per-second basis.
        Composition precedence and the retained liquid concentration basis
        follow :meth:`VaporPhase.updatePhase`.
        A zero alias or amount means "not supplied", so flow cannot be set to
        zero through ``updatePhase``. For composition profiles, conserved flows
        retain the shared-scalar or per-row form described by the constructor.
        """
        for name, amount in (('mass', mass), ('vol', vol), ('moles', moles),
                             ('mass_flow', mass_flow), ('vol_flow', vol_flow),
                             ('mole_flow', mole_flow)):
            # amount uses [kg/s], [m**3/s], or [mol/s], according to its name.
            if amount is not None and amount < 0:
                raise ValueError(f"{name} must be nonnegative")

        if mass > 0 and mass_flow is not None:
            raise ValueError("Specify either 'mass' or 'mass_flow', not both")
        if vol > 0 and vol_flow is not None:
            raise ValueError("Specify either 'vol' or 'vol_flow', not both")
        if moles > 0 and mole_flow is not None:
            raise ValueError("Specify either 'moles' or 'mole_flow', not both")

        super().updatePhase(
            mole_conc=mole_conc, mass_conc=mass_conc,
            mass_frac=mass_frac, mole_frac=mole_frac,
            vol=vol if vol_flow is None else vol_flow,
            mass=mass if mass_flow is None else mass_flow,
            moles=moles if mole_flow is None else mole_flow,
            temp=temp, pres=pres,
        )
        self.mass_flow = self.mass  # [kg/s]
        self.vol_flow = self.vol  # [m**3/s]
        self.mole_flow = self.moles  # [mol/s]

    @property
    def DynamicInlet(self):
        return self._DynamicInlet

    @DynamicInlet.setter
    def DynamicInlet(self, dynamic_object: Optional[object]) -> None:
        """Attach an inlet controller or restore static stream inputs.

        Parameters
        ----------
        dynamic_object : object or None
            Object exposing ``evaluate_inputs(time)`` with time [s].
            None clears the controller; static flow [kg/s, mol/s, m**3/s]
            and temperature [K] attributes then supply evaluated inputs.
        """
        self._DynamicInlet = dynamic_object
        if dynamic_object is None:
            return

        dynamic_object.controllable = self.controllable
        dynamic_object.parent_instance = self

    def evaluate_inputs(self, time):
        if self.DynamicInlet is None:
            inputs = {}
            for attr in self.controllable:
                inputs[attr] = getattr(self, attr)

        else:
            inputs = self.DynamicInlet.evaluate_inputs(time)

        return inputs
