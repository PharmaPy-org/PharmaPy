# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 12:44:44 2020

@author: dcasasor
"""

import numpy as np
import pandas as pd
from PharmaPy.ThermoModule import ThermoPhysicalManager
from PharmaPy.ParamEstim import ParameterEstimation, MultipleCurveResolution
from PharmaPy.StatsModule import StatisticsClass

from PharmaPy.Connections import (Connection, convert_str_flowsheet,
                                  get_inputs_new, topological_bfs)
from PharmaPy.Errors import PharmaPyNonImplementedError
from PharmaPy.Results import SimulationResult, flatten_dict_fields, get_name_object

from PharmaPy.Commons import trapezoidal_rule, check_steady_state
from PharmaPy.CheckModule import check_modeling_objects

import time
from typing import Optional, Sequence, Union

# Consumed inlet fields used to account dynamic raw feeds, in the order a
# layout's flow and composition are selected.
RAW_FLOW_FIELDS = ('mass_flow', 'mole_flow', 'vol_flow')  # [kg/s], [mol/s], [m**3/s]
# [-], [-], [kg/m**3], [mol/L]
RAW_COMPOSITION_FIELDS = ('mass_frac', 'mole_frac', 'mass_conc', 'mole_conc')
RAW_POPULATION_FIELDS = ('mu_n', 'distrib')  # [m**n/m**3], [#/m**3/um]
GRAMS_PER_KILOGRAM = 1000.0  # [g/kg], exact; relates mw [g/mol] to [kg/mol]
LITERS_PER_CUBIC_METER = 1000.0  # [L/m**3], exact; mole_conc [mol/L] to [mol/m**3]
THIRD_MOMENT_ORDER = 3  # [-], index of mu_3 [m**3/m**3] in moments ordered from n = 0


def _broadcast_samples(value, shape: tuple) -> np.ndarray:
    """Return a writable float copy of ``value`` broadcast to sample shape.

    Parameters
    ----------
    value : float or array_like
        Scalar or per-sample value in its own physical units, for example a
        constant control result or a static stream attribute.
    shape : tuple of int
        Target shape, (num_times,) or (num_times, num_species).

    Returns
    -------
    numpy.ndarray
        Float array of ``shape`` in the units of ``value``.

    Raises
    ------
    ValueError
        If ``value`` cannot be broadcast to ``shape``.
    """
    return np.array(np.broadcast_to(np.asarray(value, dtype=float), shape))


class SimulationExec:
    def __init__(self, pure_path, flowsheet):
        """Set up a flowsheet and its deterministic execution order.

        Parameters
        ----------
        pure_path : str
            Path to the pure-component thermophysical database (JSON).
        flowsheet : dict of str to sequence of str, or str
            Directed adjacency mapping each unit name to its successors, or
            a linear ``'A --> B --> C'`` string.

        Raises
        ------
        PharmaPyNonImplementedError
            If the graph contains a recycle, that is, if
            ``PharmaPy.Connections.topological_bfs`` cannot schedule every
            unit, including units that appear only as successors. The
            message names the unscheduled units.
        TypeError
            If a successor collection is a set or frozenset.

        Notes
        -----
        Units run in ``execution_names`` order, described in
        ``PharmaPy.Connections.topological_bfs``.
        """

        # Interfaces
        thermo_instance = ThermoPhysicalManager(pure_path)
        self.NamesSpecies = thermo_instance.name_species

        # Outputs
        self.StreamTable = None

        self.uos_instances = {}
        self.oper_mode = []

        if isinstance(flowsheet, dict):
            graph = flowsheet
        elif isinstance(flowsheet, str):
            graph = convert_str_flowsheet(flowsheet)

        self.graph = graph
        self.in_degree, self.execution_names = topological_bfs(graph)

        # Residual counters cover every node, including successor-only units.
        if len(self.execution_names) < len(self.in_degree):
            unscheduled = [name for name, count in self.in_degree.items()
                           if count > 0]
            raise PharmaPyNonImplementedError(
                "Provided flowsheet contains recycle stream(s); units on or "
                "downstream of a recycle cannot be scheduled: "
                + ', '.join(unscheduled))

    def _transfer_to_neighbors(self, name: str, connections: dict, count: int,
                               pick_units: Optional[Sequence[str]] = None) -> int:
        """
        Transfer output data from a unit operation to graph successors.

        Parameters
        ----------
        name : str
            Name of the source unit operation in the flowsheet graph. A
            successor-only unit has no adjacency entry and no successors.
        connections : dict
            Mapping used to store generated Connection objects. The mapping is
            updated in place.
        count : int
            Counter used to build connection names.
        pick_units : sequence of str, optional
            Unit operation names selected for execution. Successors outside
            this sequence are skipped when provided.

        Returns
        -------
        count : int
            Next available connection counter after transfers are created.

        """
        # Successor-only units are graph nodes without an adjacency entry.
        for uo_next in self.graph.get(name, ()):
            if pick_units is not None and uo_next not in pick_units:
                continue

            connection = Connection(
                source_uo=getattr(self, name),
                destination_uo=getattr(self, uo_next))

            conn_name = 'CONN%i' % count
            connections[conn_name] = connection

            connection.transfer_data()

            count += 1

        return count

    def SolveFlowsheet(self, kwargs_run=None, pick_units=None, verbose=True,
                       steady_state_di=None, tolerances_ss=None, ss_time=0):
        """
        Solve unit operations and transfer stream data through the flowsheet.

        The flowsheet is executed in topological order. Unit operations listed
        in ``pick_units`` are solved before their output data are transferred
        to selected graph successors. Units not selected in ``pick_units`` can
        still transfer existing output data when already solved.

        Parameters
        ----------
        kwargs_run : dict, optional
            Keyword arguments passed to each unit operation ``solve_unit`` call,
            keyed by unit operation name.
        pick_units : sequence of str, optional
            Unit operation names to solve. If omitted, all unit operations in
            the execution order are solved.
        verbose : bool, optional
            If true, print progress messages for each solved unit operation.
        steady_state_di : dict, optional
            Steady-state event configuration keyed by unit operation name.
        tolerances_ss : dict, optional
            Reserved steady-state tolerance mapping.
        ss_time : float, optional
            Initial steady-state time horizon accumulator.

        Returns
        -------
        None
            Results are stored on ``time_processing``, ``result``, and
            ``connections`` attributes.

        """

        if kwargs_run is None:
            kwargs_run = {}

        if steady_state_di is None:
            steady_state_di = {}

        if pick_units is None:
            pick_units = self.execution_names

        if tolerances_ss is None:
            tolerances_ss = {}

        time_processing = {}

        # Run loop
        connections = {}
        count = 1

        # ss_time = 0
        for ind, name in enumerate(self.execution_names):
            instance = getattr(self, name)

            if name in pick_units:
                self.uos_instances[name] = instance
                check_modeling_objects(instance, name)

                if verbose:
                    print()
                    print('{}'.format('-'*30))
                    print('Running {}'.format(name))
                    print('{}'.format('-'*30))
                    print()

                kwargs_uo = kwargs_run.get(name, {})

                if name in steady_state_di:
                    kw_ss = steady_state_di[name]

                    tau = 0
                    if hasattr(instance, '_get_tau'):
                        tau = instance._get_tau()

                    ss_time += tau

                    if instance.__class__.__name__ == 'Mixer':
                        pass
                    else:
                        defaults = {'time_stop': ss_time,
                                    # 'threshold': 1e-6,
                                    'tau': tau}

                        for key, val in defaults.items():
                            kw_ss.setdefault(key, val)

                        ss_event = {'callable': check_steady_state,
                                    'num_conditions': 1,
                                    'event_name': 'steady_state',
                                    'kwargs': kw_ss
                                    }

                        # instance.state_event_list = [ss_event]
                        instance.state_event_list.append(ss_event)
                        kwargs_uo['any_event'] = False

                # check_modeling_objects(instance, name)
                instance.solve_unit(**kwargs_uo)

                uo_type = instance.__module__
                if uo_type != 'PharmaPy.Containers':
                    instance.flatten_states()

                if verbose:
                    print()
                    print('Done!')
                    print()

                # Create connection object if needed
                count = self._transfer_to_neighbors(
                    name, connections, count, pick_units)

                # Processing times
                if hasattr(instance.result, 'time'):
                    time_prof = instance.result.time
                    time_processing[name] = time_prof[-1] - time_prof[0]

            # instance is already solved, pass data to connection
            elif isinstance(instance.outputs, dict):
                count = self._transfer_to_neighbors(
                    name, connections, count, pick_units)

        self.time_processing = time_processing

        self.result = SimulationResult(self)
        self.connections = connections

    def SetParamEstimation(self, x_data, y_data=None, y_spectra=None,
                           fit_spectra=False,
                           wrapper_kwargs=None,
                           phase_modifiers=None, control_modifiers=None,
                           pick_unit=None, **inputs_paramest):
        """
        Set parameter estimation using the aggregated unit operation to a
        simulation object

        Parameters
        ----------
        x_data : TYPE
            DESCRIPTION.
        y_data : TYPE, optional
            DESCRIPTION. The default is None.
        spectra : TYPE, optional
            DESCRIPTION. The default is None.
        fit_spectra : TYPE, optional
            DESCRIPTION. The default is False.
        wrapper_kwargs : TYPE, optional
            DESCRIPTION. The default is None.
        phase_modifiers : dict, optional
            Dictionary containing values to be set to the initial state
            of a phase for each experiment. Keys of 'phase_modifiers'
            must be experiment names, and fields must be dictionaries
            with keys matching the fields used to create a PharmaPy phase.
            An example for a reactor would be:

                my_modifier = {
                    'exp_1': {'temp': 300, 'mole_frac': [...]},
                    'exp_2': {'temp': 320, 'mole_frac': [...]}}

            For multi-phase systems such as crystallizer, an additional layer
            is needed to indicate which phase is being modified, e.g.

                my_modifier = {
                    'exp_1': {'Liquid_1': {'temp': 300, 'mole_frac': [...]}},
                    'exp_2': {'Liquid_1': {'temp': 320, 'mole_frac': [...]}, 'Solid_1': {'distrib':}}
                    }

            The default is None.
        control_modifiers : dict, optional
            Dictionary containing arguments to be passed to a control function.
            For instance, for a crystallizer with a temperature control with
            signature my_control(time, temp_init, ramp):

                my_modifier = {'temp': {'args': (320, -0.2)}}

            The default is None.
        pick_unit : TYPE, optional
            DESCRIPTION. The default is None.
        **inputs_paramest : TYPE
            DESCRIPTION.

        Raises
        ------
        RuntimeError
            DESCRIPTION.
        ValueError
            If ``x_data`` holds more than one unnamed experiment, because
            modifiers and ``wrapper_kwargs`` are routed to experiments by
            name. Pass ``x_data`` as a dictionary keyed by experiment name.

        Returns
        -------
        None.

        """

        # self.LoadUOs()

        if len(self.graph) == 1:
            target_unit = getattr(self, list(self.graph.keys())[0])
            # target_unit.reset_states = True
        else:
            if pick_unit is None:
                raise RuntimeError("Two or more unit operations detected. "
                                   "Select one using the 'pick_unit' argument")
            else:
                pass  # remember setting reset_states to True!!

        if not isinstance(x_data, dict):
            num_experiments = (1 if isinstance(x_data, np.ndarray)
                               else len(x_data))
            if num_experiments > 1:
                raise ValueError(
                    f"x_data holds {num_experiments} unnamed experiments; "
                    "SetParamEstimation routes phase_modifiers, "
                    "control_modifiers and wrapper_kwargs to each experiment "
                    "by name, so pass x_data as a dictionary keyed by "
                    "experiment name")

        if phase_modifiers is None:
            if isinstance(x_data, dict):
                phase_modifiers = {key: {} for key in x_data}
            else:
                phase_modifiers = {}

        if control_modifiers is None:
            if isinstance(x_data, dict):
                control_modifiers = {key: {} for key in x_data}
            else:
                control_modifiers = {}

        if wrapper_kwargs is None:
            wrapper_kwargs = {}

        if isinstance(x_data, dict):
            kwargs_wrapper = {
                key: {'modify_phase': phase_modifiers[key],
                      'modify_controls': control_modifiers[key]}
                for key in x_data}

            for di in kwargs_wrapper.values():
                di.update({'run_args': wrapper_kwargs})
        else:
            kwargs_wrapper = {'modify_phase': phase_modifiers,
                              'modify_controls': control_modifiers}

            kwargs_wrapper['run_args'] = wrapper_kwargs

        # Get 1D array of parameters from the UO class
        param_seed = inputs_paramest.pop('param_seed', None)
        if param_seed is not None:
            target_unit.Kinetics.set_params(param_seed)

        if hasattr(target_unit, 'Kinetics'):
            param_seed = target_unit.Kinetics.concat_params()
        else:
            param_seed = getattr(target_unit, 'param_seed', target_unit.params)

        name_params = inputs_paramest.get('name_params')

        if name_params is None:
            name_params = []
            for ind, logic in enumerate(target_unit.mask_params):
                if logic:
                    if hasattr(target_unit, 'Kinetics'):
                        name_params.append(
                            target_unit.Kinetics.name_params[ind])
                    else:
                        name_params.append(target_unit.name_params[ind])

        name_states = target_unit.states_uo

        inputs_paramest['name_states'] = name_states
        inputs_paramest['name_params'] = name_params

        # Instantiate parameter estimation
        if fit_spectra:
            self.ParamInst = MultipleCurveResolution(
                target_unit.paramest_wrapper,
                param_seed=param_seed, time_data=x_data, y_spectra=y_spectra,
                kwargs_fun=kwargs_wrapper,
                **inputs_paramest)
        else:
            self.ParamInst = ParameterEstimation(
                target_unit.paramest_wrapper,
                param_seed=param_seed, x_data=x_data, y_data=y_data,
                kwargs_fun=kwargs_wrapper,
                **inputs_paramest)

    def EstimateParams(self, optim_options=None, method='LM', bounds=None,
                       verbose=True):
        tic = time.time()
        results = self.ParamInst.optimize_fn(optim_options=optim_options,
                                             method=method,
                                             bounds=bounds, verbose=verbose)
        toc = time.time()

        elapsed = toc - tic

        print('Optimization time: {:.2e} s.'.format(elapsed))

        return results

    def get_equipment_size(self):
        size_equipment = {}

        for key, instance in self.uos_instances.items():
            if hasattr(instance, 'vol_tot'):
                size_equipment[key] = instance.vol_tot
            elif hasattr(instance, 'vol_phase'):
                off_vol = instance.vol_offset
                size_equipment[key] = instance.vol_phase / off_vol

            elif hasattr(instance, 'area_filt'):
                size_equipment[key] = instance.area_filt

        return size_equipment

    def GetCAPEX(self, size_equipment=None, k_vals=None, b_vals=None,
                 cepci_vals=None, f_pres=None, f_mat=None, min_capacity=None):

        if size_equipment is None:
            size_equipment = self.get_equipment_size()

        num_equip = len(size_equipment)
        name_equip = size_equipment.keys()
        if cepci_vals is None:
            cepci_vals = np.ones(2)

        if f_pres is None:
            f_pres = np.ones(num_equip)

        if f_mat is None:
            f_mat = np.ones(num_equip)

        if k_vals is None:
            return size_equipment
        else:
            capacities = np.array(list(size_equipment.values()))

            if min_capacity is None:
                a_corr = capacities
            else:
                a_corr = np.maximum(min_capacity, capacities)

            k1, k2, k3 = k_vals.T
            cost_zero = 10**(k1 + k2*np.log10(a_corr) + k3*np.log10(a_corr)**2)

            b1, b2 = b_vals.T

            f_bare = b1 + b2 * f_mat * f_pres
            cost_equip = cost_zero * f_bare

            scale_corr = np.ones_like(capacities)
            if min_capacity is not None:
                for ind, capac in enumerate(capacities):
                    if capac < min_capacity[ind]:
                        scale_corr[ind] = (capac / min_capacity[ind])**0.6

            cost_equip *= scale_corr

            cost_equip = dict(zip(name_equip, cost_equip))

            return cost_equip

    def GetLabor(self, wage=35, num_weeks=48):
        # TODO: clarify whether labor cost is hourly [1/h] or per shift [1/shift].
        has_solids = []
        is_batch = []
        uo_names = []

        for key, uo in self.uos_instances.items():
            if uo.__class__.__name__ != 'Mixer':

                if hasattr(uo, 'Phases'):
                    if isinstance(uo.Phases, (list, tuple)):
                        is_solid = [phase.__class__.__name__ == 'SolidPhase'
                                    for phase in uo.Phases]
                    else:
                        is_solid = [
                            uo.Phases.__class__.__name__ == 'SolidPhase']
                else:
                    is_solid = [False]  # Mixers

                has_solids.append(any(is_solid))

                oper = uo.oper_mode == 'Batch' or uo.oper_mode == 'Semibatch'
                is_batch.append(oper)
                uo_names.append(key)

        has_solids = np.array(has_solids, dtype=bool)
        is_batch = np.array(is_batch, dtype=bool)

        # Number of operators per shift
        num_workers = has_solids * (2 + is_batch) + ~has_solids * (1 + is_batch)

        hr_week = 40
        labor_cost = 1.20 * num_workers * 5 * (hr_week * num_weeks) * wage  # [USD/yr]

        labor_array = np.column_stack(
            (has_solids, is_batch, num_workers, labor_cost))

        labor_df = pd.DataFrame(labor_array, index=uo_names,
                                columns=('has_solids', 'is_batch',
                                         'num_workers', 'labor_cost'))
        return labor_df

    def get_from_phases(self, phases, fields):
        """Collect named attributes from one phase or a mixed phase.

        Parameters
        ----------
        phases : object
            PharmaPy phase or mixed-phase object.
        fields : sequence of str
            Attribute names to collect. Typical values include fractions [-],
            temperature [K], pressure [Pa], amounts [kg] or [mol], and volumes
            [m**3].

        Returns
        -------
        dict
            Attribute records keyed by phase object name.
        """
        if phases.__module__ == 'PharmaPy.MixedPhases':
            phases = phases.Phases
        else:
            phases = [phases]

        out = {}
        for phase in phases:
            phase_data = {}
            for field in fields:
                phase_data[field] = getattr(phase, field)

            name_phase = get_name_object(phase)
            out[name_phase] = phase_data

        return out

    @staticmethod
    def _declared_inlet_layout(uo) -> dict:
        """Return the inlet layout a receiving unit declares.

        Parameters
        ----------
        uo : object
            Receiving unit operation.

        Returns
        -------
        dict
            Inlet groups mapped to declared field names and dimensions, from
            ``states_in_dict``, ``dict_states_in`` or ``states_dict``.

        Raises
        ------
        ValueError
            If the unit declares no inlet layout.
        """
        for attribute in ('states_in_dict', 'dict_states_in', 'states_dict'):
            layout = getattr(uo, attribute, None)
            if isinstance(layout, dict):
                return layout

        raise ValueError(
            f"Unit {type(uo).__name__} declares no inlet layout "
            "(states_in_dict, dict_states_in or states_dict), so the feed "
            "it consumes from a DynamicInlet cannot be accounted; remove "
            "the DynamicInlet from its raw inlet.")

    def _consumed_inlet_fields(self, uo, inlet, time: np.ndarray,
                               layout: dict) -> dict:
        """Evaluate the inlet fields a receiving unit consumes.

        Parameters
        ----------
        uo : object
            Receiving unit operation.
        inlet : object
            Raw inlet stream or mixed (slurry) inlet of ``uo``.
        time : numpy.ndarray
            Result times of ``uo`` [s], shape (num_times,).
        layout : dict
            The unit's declared inlet layout; see
            :meth:`_declared_inlet_layout`. Callers validate the controlled
            fields against it first.

        Returns
        -------
        dict
            Consumed field values from every inlet group, merged by name, in
            the units and basis the unit reads: flows [kg/s], [mol/s] or
            [m**3/s], fractions [-], ``mass_conc`` [kg/m**3], ``mole_conc``
            [mol/L], ``temp`` [K], ``mu_n`` [m**n/m**3] or ``distrib``
            [#/m**3/um]. Time is on the first axis: scalar fields have shape
            (num_times,), and composition and population fields
            (num_times, num_entries), including a single species.

        Notes
        -----
        ``DynamicCollector`` and the continuous crystallizers are evaluated
        through their own input methods (``get_inputs_new`` and
        ``get_inputs``), which add the slurry moment and liquid
        concentration fallbacks they consume. Other units are evaluated
        with :func:`PharmaPy.Connections.get_inputs_new` on their declared
        layout (``states_in_dict``, ``dict_states_in`` or ``states_dict``),
        exactly as in their own input evaluation. Integrating units evaluate
        each result time separately as a scalar, as during integration, so
        their control callables need not accept arrays; every sampled value
        is copied before the next evaluation and the samples are stacked. A
        ``Mixer`` evaluates its inlets once on its whole time grid, so its
        feed is evaluated the same way (see :meth:`_evaluates_full_grid`).
        """
        from PharmaPy.Containers import DynamicCollector
        from PharmaPy.Crystallizers import _BaseCryst

        num_times = len(time)
        vector_fields = RAW_COMPOSITION_FIELDS + RAW_POPULATION_FIELDS

        if self._evaluates_full_grid(uo):
            grouped = get_inputs_new(time, inlet, layout)
            dims = {name: dim for group in layout.values()
                    for name, dim in group.items()}
            fields = {}
            for group in grouped.values():
                for name, value in group.items():
                    value = np.asarray(value, dtype=float)
                    if name in vector_fields:
                        fields[name] = np.broadcast_to(
                            value.reshape(-1, dims[name]),
                            (num_times, dims[name])).copy()
                    else:
                        fields[name] = np.broadcast_to(
                            value.reshape(-1), (num_times,)).copy()

            return fields

        samples = []
        for sample_time in time:  # [s], one scalar evaluation per result time
            if isinstance(uo, DynamicCollector):
                grouped = uo.get_inputs_new(float(sample_time))
            elif isinstance(uo, _BaseCryst):
                grouped = uo.get_inputs(float(sample_time))
            else:
                grouped = get_inputs_new(float(sample_time), inlet, layout)

            # Copy every value: a control may reuse and mutate one buffer.
            samples.append({name: np.array(value, dtype=float)
                            for group in grouped.values()
                            for name, value in group.items()})

        fields = {}
        for name in samples[0]:
            stacked = np.stack([np.asarray(sample[name], dtype=float)
                                for sample in samples])  # time on axis 0
            if name in vector_fields:
                fields[name] = stacked.reshape(num_times, -1)
            elif stacked.size == num_times:
                fields[name] = stacked.reshape(num_times)
            else:
                fields[name] = stacked

        return fields

    @staticmethod
    def _evaluates_full_grid(uo) -> bool:
        """Report whether a unit evaluates its inlets on its whole time grid.

        Parameters
        ----------
        uo : object
            Receiving unit operation.

        Returns
        -------
        bool
            True for ``Mixer``, whose ``solve_unit`` evaluates every inlet
            once on the connected grid [s] (array time); False for units that
            evaluate scalar times during integration.
        """
        from PharmaPy.Containers import Mixer

        return isinstance(uo, Mixer)

    @staticmethod
    def _dynamic_record(stream, species_mass, species_moles, vol_flow, temp,
                        time, basis, composition_controlled,
                        static_temp) -> dict:
        """Integrate per-sample species flows into one raw-material record.

        Parameters
        ----------
        stream : object
            Stream or phase supplying static ``pres`` [Pa] and fractions [-]
            for uncontrolled fields.
        species_mass, species_moles : numpy.ndarray
            Consumed species flows [kg/s] and [mol/s], shape
            (num_times, num_species).
        vol_flow : numpy.ndarray
            Consumed volume flow [m**3/s], shape (num_times,).
        temp : numpy.ndarray
            Consumed temperature [K], shape (num_times,).
        time : numpy.ndarray
            Result times [s], shape (num_times,).
        basis : {'mass', 'mole'}
            Accounting basis.
        composition_controlled : bool
            Whether a DynamicInput controls the composition.
        static_temp : float
            Static temperature of the consumed inlet [K], reported when the
            consumed temperature varies but no material is fed.

        Returns
        -------
        dict
            ``mass`` [kg] or ``moles`` [mol], ``vol`` [m**3], ``temp`` [K],
            ``pres`` [Pa], then ``mass_frac`` or ``mole_frac`` [-], shape
            (num_species,).

        Notes
        -----
        Amounts are trapezoidal integrals over ``time``. A controlled
        composition is reported as integrated species amounts over the total
        on the accounting basis, else as the static fractions. The consumed
        temperature is reported as is when constant, otherwise as its
        mass-flow-weighted average on both bases, or ``static_temp`` when
        the fed mass is zero.
        """
        if basis == 'mass':
            amount_name, frac_name = 'mass', 'mass_frac'
            basis_flow = species_mass  # [kg/s], per species
        else:
            amount_name, frac_name = 'moles', 'mole_frac'
            basis_flow = species_moles  # [mol/s], per species

        species_total = trapezoidal_rule(time, basis_flow)  # [kg] or [mol]
        total = np.sum(species_total)  # [kg] or [mol]
        mass_flow = species_mass.sum(axis=1)  # [kg/s]
        total_mass = trapezoidal_rule(time, mass_flow)  # [kg]

        record = {amount_name: total,  # [kg] or [mol]
                  'vol': trapezoidal_rule(time, vol_flow)}  # [m**3]
        if np.all(temp == temp[0]):
            record['temp'] = temp[0]  # [K], constant consumed temperature
        elif total_mass != 0:
            record['temp'] = trapezoidal_rule(
                time, mass_flow * temp) / total_mass  # [K]
        else:
            record['temp'] = static_temp  # [K]
        record['pres'] = stream.pres  # [Pa], no unit consumes inlet pressure
        if composition_controlled and total != 0:
            record[frac_name] = species_total / total  # [-]
        else:
            record[frac_name] = getattr(stream, frac_name)  # [-]

        return record

    def _account_dynamic_stream(self, stream, fields: dict, controlled: list,
                                time: np.ndarray, basis: str) -> dict:
        """Account a single-phase raw stream from its consumed fields.

        Parameters
        ----------
        stream : LiquidStream, VaporStream, SolidStream or similar
            Raw stream; supplies ``mw`` [g/mol], ``frac_to_frac``,
            ``getDensity`` and static values.
        fields : dict
            Consumed fields; see :meth:`_consumed_inlet_fields`.
        controlled : list of str
            Fields controlled by the stream's DynamicInput.
        time : numpy.ndarray
            Result times [s], shape (num_times,).
        basis : {'mass', 'mole'}
            Accounting basis.

        Returns
        -------
        dict
            Raw-material record; see :meth:`_dynamic_record`.

        Raises
        ------
        ValueError
            If the unit consumes no flow, or a population field is controlled.

        Notes
        -----
        The consumed flow and composition define the species flows: a
        concentration with a volume flow gives ``Q c`` directly; otherwise
        the composition is expressed as fractions (concentrations
        normalized) and multiplied by the mass or molar flow, or by the
        volume flow times the density of each sample's composition and
        temperature. Volume is the consumed volume flow, or mass flow over
        that density. Without a consumed composition, the static mass
        fractions are used. Samples whose composition is exactly zero, such
        as a stopped feed, contribute no material; they add no volume when
        volume is derived from the mass or molar flow, whereas a consumed
        ``vol_flow`` is reported as supplied. Non-finite values propagate to
        the totals rather than being dropped.
        """
        population = [name for name in controlled if name in RAW_POPULATION_FIELDS]
        if population:
            raise ValueError(
                f"DynamicInput controls {population} on the single-phase raw "
                f"stream '{get_name_object(stream)}', which carries no crystal "
                "population; remove those controls.")

        flow_name = next((name for name in RAW_FLOW_FIELDS if name in fields), None)
        if flow_name is None:
            # Defensive guard: every current unit with a declared continuous
            # inlet layout consumes one of RAW_FLOW_FIELDS.
            raise ValueError(
                f"The unit fed by '{get_name_object(stream)}' consumes no inlet "
                f"flow ({list(RAW_FLOW_FIELDS)}), so its dynamic feed cannot be "
                "accounted; remove the DynamicInlet from this raw inlet.")
        comp_name = next((name for name in RAW_COMPOSITION_FIELDS
                          if name in fields), 'mass_frac')

        num_times = len(time)
        molar_mass = np.asarray(stream.mw, dtype=float)  # [g/mol]
        sample_shape = (num_times,)
        composition_shape = (num_times, len(molar_mass))
        flow = _broadcast_samples(fields[flow_name], sample_shape)
        # [kg/s], [mol/s] or [m**3/s], per flow_name
        composition = _broadcast_samples(
            fields.get(comp_name, getattr(stream, comp_name)), composition_shape)
        # [-], [kg/m**3] or [mol/L], per comp_name
        temp = _broadcast_samples(fields.get('temp', stream.temp),
                                  sample_shape)  # [K]

        if flow_name == 'vol_flow' and comp_name == 'mass_conc':
            species_mass = flow[:, np.newaxis] * composition  # [kg/s]
            species_moles = species_mass * GRAMS_PER_KILOGRAM / molar_mass  # [mol/s]
        elif flow_name == 'vol_flow' and comp_name == 'mole_conc':
            species_moles = (flow[:, np.newaxis] * composition
                             * LITERS_PER_CUBIC_METER)  # [mol/s]
            species_mass = species_moles * molar_mass / GRAMS_PER_KILOGRAM  # [kg/s]
        else:
            # Rows whose composition is exactly zero (a stopped feed) keep
            # zero fractions and contribute nothing; any other row, including
            # a non-finite one, is normalized so invalid values stay visible.
            row_total = composition.sum(axis=1)  # [-], [kg/m**3] or [mol/L]
            fed = ~np.all(composition == 0, axis=1)
            fractions = np.zeros_like(composition)  # [-]
            fractions[fed] = composition[fed] / row_total[fed, np.newaxis]
            converted = np.zeros_like(composition)  # [-], other basis
            if comp_name in ('mole_frac', 'mole_conc'):
                mole_frac = fractions  # [-]
                if fed.any():
                    converted[fed] = stream.frac_to_frac(mole_frac=mole_frac[fed])
                mass_frac = converted  # [-]
            else:
                mass_frac = fractions  # [-]
                if fed.any():
                    converted[fed] = stream.frac_to_frac(mass_frac=mass_frac[fed])
                mole_frac = converted  # [-]

            density = np.zeros(num_times)  # [kg/m**3], zero for unfed rows
            if fed.any():
                density[fed] = _broadcast_samples(stream.getDensity(
                    mass_frac=mass_frac[fed], temp=temp[fed]),
                    (np.count_nonzero(fed),))  # [kg/m**3]

            if flow_name == 'mole_flow':
                species_moles = flow[:, np.newaxis] * mole_frac  # [mol/s]
                species_mass = species_moles * molar_mass / GRAMS_PER_KILOGRAM
                # [kg/s]
            else:
                if flow_name == 'vol_flow':
                    mass_flow = flow * density  # [kg/s]
                else:
                    mass_flow = flow  # [kg/s]
                species_mass = mass_flow[:, np.newaxis] * mass_frac  # [kg/s]
                species_moles = species_mass * GRAMS_PER_KILOGRAM / molar_mass
                # [mol/s]

        if flow_name == 'vol_flow':
            vol_flow = flow  # [m**3/s]
        else:
            vol_flow = np.divide(species_mass.sum(axis=1), density,
                                 out=np.zeros(num_times),
                                 where=density != 0)  # [m**3/s], NaN propagates

        return self._dynamic_record(
            stream, species_mass, species_moles, vol_flow, temp, time, basis,
            composition_controlled=any(name in controlled
                                       for name in RAW_COMPOSITION_FIELDS),
            static_temp=stream.temp)

    def _account_dynamic_slurry(self, slurry, fields: dict, consumed: list,
                                controlled: list, time: np.ndarray,
                                basis: str) -> dict:
        """Account the liquid and solid phases of a dynamic slurry feed.

        Parameters
        ----------
        slurry : SlurryStream
            Raw slurry inlet with ``Liquid_1`` and ``Solid_1`` phases.
        fields : dict
            Consumed fields; see :meth:`_consumed_inlet_fields`.
        consumed : list of str
            Field names the receiving unit declares.
        controlled : list of str
            Fields controlled by the slurry's DynamicInput.
        time : numpy.ndarray
            Result times [s], shape (num_times,).
        basis : {'mass', 'mole'}
            Accounting basis.

        Returns
        -------
        dict
            Records keyed by liquid and solid phase name; see
            :meth:`_dynamic_record`.

        Raises
        ------
        ValueError
            If the unit does not consume the slurry volume flow, liquid
            ``mass_conc`` and a population (``mu_n`` or ``distrib``).

        Notes
        -----
        The solid volume fraction is ``kv * mu_3`` [-], with ``mu_3``
        [m**3/m**3] taken from the controlled population field, else from
        consumed moments, else from the consumed distribution on the solid
        phase's size grid. Liquid species flows are
        ``Q (1 - kv mu_3) mass_conc`` [kg/s], as in the crystallizer feed
        balance; solid flow is ``Q kv mu_3`` times the solid density
        [kg/m**3] of the static solid composition, split by the static solid
        mass fractions. Both phases report the consumed slurry temperature,
        and records follow the order of ``slurry.Phases``.
        """
        populations = [name for name in RAW_POPULATION_FIELDS if name in fields]
        if ('vol_flow' not in fields or 'mass_conc' not in fields
                or not populations):
            raise ValueError(
                f"The unit consumes {consumed} from slurry feed "
                f"'{get_name_object(slurry)}'; accounting a dynamic slurry "
                "needs its vol_flow, liquid mass_conc and a population (mu_n "
                "or distrib). Remove the DynamicInlet from this raw inlet.")

        liquid, solid = slurry.Liquid_1, slurry.Solid_1
        num_times = len(time)
        sample_shape = (num_times,)
        flow = _broadcast_samples(fields['vol_flow'], sample_shape)  # [m**3/s]
        temp = _broadcast_samples(fields.get('temp', slurry.temp),
                                  sample_shape)  # [K]
        mass_conc = _broadcast_samples(
            fields['mass_conc'], (num_times, len(liquid.mw)))  # [kg/m**3]

        population = next((name for name in populations if name in controlled),
                          populations[0])
        population_values = np.asarray(fields[population], dtype=float)
        # [m**n/m**3] for mu_n or [#/m**3/um] for distrib, time first
        if population == 'mu_n':
            third_moment = population_values[..., THIRD_MOMENT_ORDER]  # [m**3/m**3]
        else:
            distrib = _broadcast_samples(
                population_values,
                (num_times, population_values.shape[-1]))  # [#/m**3/um]
            third_moment = np.ravel(solid.getMoments(
                distrib=distrib, mom_num=THIRD_MOMENT_ORDER))  # [m**3/m**3]
        solid_fraction = solid.kv * _broadcast_samples(
            third_moment, sample_shape)  # [-]

        liquid_vol_flow = flow * (1 - solid_fraction)  # [m**3/s]
        liquid_mass = liquid_vol_flow[:, np.newaxis] * mass_conc  # [kg/s]
        solid_vol_flow = flow * solid_fraction  # [m**3/s]
        solid_mass = (solid_vol_flow * solid.getDensity())[:, np.newaxis] * \
            np.asarray(solid.mass_frac, dtype=float)  # [kg/s]

        # Per phase: species mass flows [kg/s] (num_times, num_species),
        # phase volume flow [m**3/s] (num_times,), composition-controlled flag.
        phase_flows = {id(liquid): (liquid_mass, liquid_vol_flow,
                                    'mass_conc' in controlled),
                       id(solid): (solid_mass, solid_vol_flow, False)}
        records = {}
        for phase in slurry.Phases:  # same row order as static records
            species_mass, vol_flow, composition_controlled = phase_flows[id(phase)]
            species_moles = (species_mass * GRAMS_PER_KILOGRAM
                             / np.asarray(phase.mw, dtype=float))  # [mol/s]
            records[get_name_object(phase)] = self._dynamic_record(
                phase, species_mass, species_moles, vol_flow, temp, time,
                basis, composition_controlled, static_temp=slurry.temp)

        return records

    def _account_dynamic_inlet(self, uo, inlet, basis: str) -> dict:
        """Account a raw inlet whose DynamicInlet the receiving unit consumes.

        Parameters
        ----------
        uo : object
            Receiving unit operation with ``result.time`` [s].
        inlet : object
            Raw inlet stream or slurry with a ``DynamicInlet``.
        basis : {'mass', 'mole'}
            Accounting basis.

        Returns
        -------
        dict
            Records keyed by stream or phase name; see
            :meth:`_dynamic_record`.

        Raises
        ------
        ValueError
            If ``basis`` is invalid, the DynamicInput controls a field the
            unit does not consume, or the consumed feed cannot be accounted.

        Notes
        -----
        At every result time the accounted feed is the one the unit consumes:
        controlled fields from the DynamicInput and the static stream or
        phase values of the other fields the unit declares. A unit that
        records an ``inlet_stop_time`` [s] (an ``Evaporator`` whose volume
        event shut its inlet) consumes no feed afterwards, so the accounted
        times end there: result times before it, then the stop time itself.
        """
        if basis not in ('mass', 'mole'):
            raise ValueError("basis must be either 'mass' or 'mole'")

        time = np.atleast_1d(np.asarray(uo.result.time, dtype=float))  # [s]
        stop_time = getattr(uo, 'inlet_stop_time', None)  # [s], absolute
        if stop_time is not None:
            time = np.append(time[time < stop_time], stop_time)  # [s]
        layout = self._declared_inlet_layout(uo)
        consumed = [name for group in layout.values() for name in group]
        probe_time = time if self._evaluates_full_grid(uo) else float(time[0])
        controlled = list(inlet.DynamicInlet.evaluate_inputs(probe_time))
        unconsumed = [name for name in controlled if name not in consumed]
        if unconsumed:
            raise ValueError(
                f"DynamicInput on raw inlet '{get_name_object(inlet)}' controls "
                f"{unconsumed}, which {type(uo).__name__} does not consume; it "
                f"consumes {consumed}. Control only consumed fields.")

        fields = self._consumed_inlet_fields(uo, inlet, time, layout)

        if inlet.__module__ == 'PharmaPy.MixedPhases':
            return self._account_dynamic_slurry(inlet, fields, consumed,
                                                controlled, time, basis)

        return {get_name_object(inlet): self._account_dynamic_stream(
            inlet, fields, controlled, time, basis)}

    def get_raw_inlets(self, uo, basis: str = 'mass') -> dict:
        """Collect raw inlet data for a unit operation.

        Parameters
        ----------
        uo : object
            Unit operation whose upstream-free inlet streams are treated as raw
            materials.
        basis : {'mass', 'mole'}, optional
            Accounting basis. Mass totals are reported in [kg] and mass-flow
            rates in [kg/s]; molar totals are reported in [mol] and molar-flow
            rates in [mol/s].

        Returns
        -------
        dict
            Raw inlet records keyed first by inlet name and then by stream or
            phase name. Records include totals, composition fractions [-],
            temperature [K], pressure [Pa], and volume [m**3]. Static
            continuous records also include their flow rates.

        Raises
        ------
        ValueError
            If ``basis`` is not ``'mass'`` or ``'mole'``, a phase of a mixed
            raw inlet has its own ``DynamicInlet``, or a dynamic inlet cannot
            be accounted (see :meth:`_account_dynamic_inlet`).

        Notes
        -----
        Continuous raw inlets with a ``DynamicInlet`` are accounted from the
        feed the unit consumes at each of its result times. Batch and static
        continuous records are read from the stream's stored state; static
        continuous totals are the flow times the fed duration. Both stop at
        a unit's recorded ``inlet_stop_time`` [s], set by an ``Evaporator``
        with ``stop_at_maxvol=False`` when its volume event shuts the inlet.
        """
        if basis not in ('mass', 'mole'):
            raise ValueError("basis must be either 'mass' or 'mole'")

        if hasattr(uo, 'Inlet'):
            if isinstance(uo.Inlet, dict):
                inlets = uo.Inlet
            else:
                inlets = [uo.Inlet]
        elif uo.__class__.__name__ == 'Mixer':
            inlets = uo.Inlets
        else:
            inlets = [None]

        if not isinstance(inlets, dict):
            inlets = {'Inlet_%i' % num: obj for num, obj in enumerate(inlets)}

        raws = {key: val for key, val in inlets.items()
                if val is not None and val.y_upstream is None}  # raw inlets

        out = {}

        for name, inlet in raws.items():
            if inlet.__module__ == 'PharmaPy.MixedPhases':
                streams = inlet.Phases
                phase_controlled = [get_name_object(phase) for phase in streams
                                    if getattr(phase, 'DynamicInlet', None)
                                    is not None]
                if phase_controlled:
                    raise ValueError(
                        f"Phases {phase_controlled} of raw inlet "
                        f"'{get_name_object(inlet)}' have their own "
                        "DynamicInlet, which no unit operation reads; attach "
                        "the DynamicInput to the mixed inlet itself.")
            else:
                streams = [inlet]

            if (uo.oper_mode != 'Batch'
                    and getattr(inlet, 'DynamicInlet', None) is not None):
                out[name] = self._account_dynamic_inlet(uo, inlet, basis)
                continue

            stream_data = {}
            for stream in streams:
                fields = ['temp', 'pres']  # [K], [Pa]

                name_stream = get_name_object(stream)

                stream_data[name_stream] = {}

                dens = stream.getDensity(basis=basis)  # [kg/m**3] or [mol/L]

                if uo.oper_mode == 'Batch':
                    if basis == 'mass':
                        total = stream.mass  # [kg]
                        stream_data[name_stream] = {'mass': total}
                        fields += ['mass_frac']
                    elif basis == 'mole':
                        total = stream.moles  # [mol]
                        stream_data[name_stream] = {'moles': total}
                        fields += ['mole_frac']
                else:
                    start_time = uo.result.time[0]  # [s]
                    end_time = uo.result.time[-1]  # [s]
                    stop_time = getattr(uo, 'inlet_stop_time', None)  # [s]
                    if stop_time is not None:
                        end_time = min(end_time, stop_time)  # [s], inlet shut
                    time = end_time - start_time  # [s], fed duration
                    if basis == 'mass':
                        flow = stream.mass_flow  # [kg/s]
                        total = flow*time  # [kg]

                        stream_data[name_stream] = {'mass': total}
                        fields += ['mass_frac', 'mass_flow', 'vol_flow']

                    else:
                        flow = stream.mole_flow  # [mol/s]
                        total = flow*time  # [mol]

                        stream_data[name_stream] = {'moles': total}
                        fields += ['mole_frac', 'mole_flow', 'vol_flow']

                vol = total / dens  # [m**3] or [L]
                if basis == 'mole':
                    vol *= 1/1000  # [m**3]

                stream_data[name_stream]['vol'] = vol  # [m**3]

            from_inlet = self.get_from_phases(inlet, fields)

            for key in from_inlet:
                stream_data[key].update(from_inlet[key])

            out[name] = stream_data

        return out

    def get_holdup(self, uo, basis='mass'):
        """Collect initial holdup raw-material records.

        Parameters
        ----------
        uo : object
            Unit operation that may retain an original phase or mixed phase.
        basis : {'mass', 'mole'}, optional
            Accounting basis. Mass holdups are [kg]; molar holdups are [mol].

        Returns
        -------
        dict
            Initial holdup records including composition fractions [-],
            temperature [K], pressure [Pa], and volume [m**3].
        """
        out = {}

        if hasattr(uo, '__original_phase__'):
            phases = uo.__original_phase__

            if basis == 'mass':
                fields = ['mass', 'mass_frac']
            elif basis == 'mole':
                fields = ['moles', 'mole_frac']

            fields += ['temp', 'pres', 'vol']

            if not phases.transferred_from_uo:
                out = self.get_from_phases(phases, fields)
                out = {'Initial_holdup': out}

        return out

    def GetRawMaterials(self, basis='mass', totals=True, steady_state=False,
                        include_holdups=True):
        """Get raw material use for all solved unit operations.

        Parameters
        ----------
        basis : {'mass', 'mole'}, optional
            Accounting basis. Mass totals are reported in [kg] and molar
            totals are reported in [mol].
        totals : bool, optional
            If true, aggregate each raw stream into total and per-species
            columns on the selected basis.
        steady_state : bool, optional
            Reserved for future steady-state raw material accounting. It is
            accepted for API compatibility but currently does not change the
            returned table.
        include_holdups : bool, optional
            If true, include initial holdups that were not transferred from an
            upstream unit operation.

        Returns
        -------
        raw_df : pandas.DataFrame
            Raw material table indexed by unit operation, raw source, and
            stream or phase name. Total columns are [kg] or [mol], and
            per-species columns use the same selected basis.

        Raises
        ------
        ValueError
            If ``basis`` is not ``'mass'`` or ``'mole'``, or a raw inlet's
            dynamic controls cannot be accounted.

        Notes
        -----
        Continuous raw inlets with a ``DynamicInlet`` are accounted from the
        feed the receiving unit consumes at each of its result times and
        integrated with the trapezoidal rule; see :meth:`get_raw_inlets`.
        Continuous raw feed stops at a unit's recorded ``inlet_stop_time``
        [s], such as an ``Evaporator`` inlet shut by its volume event.
        """
        if basis not in ('mass', 'mole'):
            raise ValueError("basis must be either 'mass' or 'mole'")

        out = {}
        for name, uo in self.uos_instances.items():
            out[name] = {}

            raw_inlets = self.get_raw_inlets(uo, basis=basis)
            if include_holdups:
                raw_holdup = self.get_holdup(uo, basis=basis)
            else:
                raw_holdup = {}

            for second in raw_inlets:  # flatten multidimensional states
                for third in raw_inlets[second]:
                    di_raw = flatten_dict_fields(raw_inlets[second][third],
                                                 index=self.NamesSpecies)
                    raw_inlets[second][third] = di_raw

            for second in raw_holdup:
                for third in raw_holdup[second]:
                    di_hold = flatten_dict_fields(raw_holdup[second][third],
                                                  index=self.NamesSpecies)
                    raw_holdup[second][third] = di_hold

            out[name].update(raw_inlets)
            out[name].update(raw_holdup)

        di_multiindex = {(i, j, k): out[i][j][k]
                         for i in out
                         for j in out[i]
                         for k in out[i][j]}

        if len(di_multiindex) == 0:
            raw_df = pd.DataFrame()
        else:
            multi_index = pd.MultiIndex.from_tuples(di_multiindex)
            raw_df = pd.DataFrame(list(di_multiindex.values()),
                                  index=multi_index)

            if totals:
                if basis == 'mass':
                    mass_frac = raw_df.filter(regex='mass_frac').values  # [-]

                    mass = raw_df['mass'].values[:, np.newaxis]  # [kg]
                    mass_comp = mass_frac * mass  # [kg]

                    cols = ['mass_%s' % comp for comp in self.NamesSpecies]
                    cols = ['mass'] + cols

                    raw_df = pd.DataFrame(np.column_stack((mass, mass_comp)),
                                          columns=cols, index=raw_df.index)

                elif basis == 'mole':
                    mole_frac = raw_df.filter(regex='mole_frac').values  # [-]
                    moles = raw_df['moles'].values[:, np.newaxis]  # [mol]
                    moles_comp = mole_frac * moles  # [mol]

                    cols = ['moles_%s' % comp for comp in self.NamesSpecies]
                    cols = ['moles'] + cols

                    raw_df = pd.DataFrame(np.column_stack((moles, moles_comp)),
                                          columns=cols, index=raw_df.index)

        return raw_df

    def GetDuties(self, full_output: bool = False) -> Union[pd.DataFrame, tuple]:
        """Collect unit-reported energies without changing their time/sign basis.

        Parameters
        ----------
        full_output : bool, optional
            Return utility-type identifiers alongside the energy table.

        Returns
        -------
        heat_duties : pandas.DataFrame
            Unit-reported energies [J], rows in equipment order and columns
            ``heating``, ``cooling``. These labels do not normalize signs or
            guarantee a common accumulation interval; see Notes.
        duties_ids : numpy.ndarray, optional
            Utility identifiers [-], shape (num_equipment, 2), returned only
            with full_output=True. Refrigeration uses -3, -2, -1; cooling water
            uses 0; heating uses 1, 2, 3 (1 is low-pressure steam).

        Notes
        -----
        BatchReactor, CSTR, and SemibatchReactor accumulate since reset in
        column 0, positive for heat added to liquid. PFR uses the same sign
        and column but reports only its latest run. Batch/MSMPR crystallizers
        report the latest segment in column 1, positive for heat removed;
        SemibatchCryst does not publish these energy diagnostics.

        Batch evaporators accumulate since reset or new Phases: column 0 is
        signed drum heat (positive inward), column 1 signed condenser heat
        (negative for cooling). Continuous evaporators accumulate signed
        utility/condenser energies and then report magnitudes in their two
        columns; utility heat is positive outward before taking the magnitude.

        This collector neither integrates profiles nor sums earlier PFR or
        crystallizer segments. Before duration-based comparisons or GetOPEX,
        accumulate those per-run energies externally over the intended common
        horizon. Mixing cumulative and last-segment rows otherwise undercounts
        repeated PFR/crystallizer operation.
        """
        heat_duties = []
        equipment_ids = []
        duty_ids = []

        for key, instance in self.uos_instances.items():
            if hasattr(instance, 'heat_duty'):
                duty_ids.append(instance.duty_type)

                heat_duties.append(instance.heat_duty)
                equipment_ids.append(key)

        heat_duties = np.array(heat_duties)
        heat_duties = pd.DataFrame(heat_duties, index=equipment_ids,
                                   columns=['heating', 'cooling'])

        duties_ids = np.array(duty_ids)

        if full_output:
            return heat_duties, duties_ids
        else:
            return heat_duties

    def GetOPEX(self, cost_raw, include_holdups=True, steady_raw=False,
                lumped=False, kwargs_items=None):
        """
        Get operating costs from duties, raw materials, and labor.

        Parameters
        ----------
        cost_raw : array_like
            Raw material unit costs compatible with the raw material table.
            On a mass basis, values are [USD/kg]; on a mole basis, values are
            [USD/mol]. A scalar applies to every raw-material column. A vector
            must have one entry per raw-material column: the first entry prices
            the total column and the remaining entries price per-species columns.
        include_holdups : bool, optional
            If true, raw material accounting includes initial holdups.
        steady_raw : bool, optional
            Forwarded to ``GetRawMaterials``. It is reserved for future
            steady-state raw accounting and currently does not change the raw
            material table.
        lumped : bool, optional
            Reserved for lumped OPEX reporting.
        kwargs_items : dict, optional
            Per-item keyword arguments for ``duties``, ``raw_materials``, and
            ``labor`` calculations. Top-level ``steady_raw`` and
            ``include_holdups`` take precedence over same-named entries in
            ``kwargs_items['raw_materials']``.

        Returns
        -------
        duty_cost, raw_cost, labor_cost : pandas.DataFrame
            ``duty_cost`` [USD] and ``raw_cost`` [USD] are per simulated run.
            ``labor_cost`` is [USD/yr]. Returned when ``lumped`` is false.
            When ``lumped`` is true, this method currently returns ``None``.

        Raises
        ------
        ValueError
            If ``cost_raw`` has more than one dimension or has a one-dimensional
            width other than one or the raw-material table width, or if
            ``GetRawMaterials`` cannot account a raw inlet's dynamic controls.

        """

        opex_items = ('duties', 'raw_materials', 'labor')
        if kwargs_items is None:
            kwargs_items = {key: {} for key in opex_items}

        cost_raw = np.asarray(cost_raw)

        # ---------- Heat duties
        # Energy cost [USD/GJ], keyed by GetDuties duty type [-].
        duty_cost_by_type = {
            -3: 14.12,  # refrigeration
            -2: 8.49,  # refrigeration
            -1: 4.77,  # refrigeration
            0: 0.378,  # cooling water
            1: 4.54,  # steam
            2: 4.77,  # steam
            3: 5.66,  # steam
        }

        duties, duty_types = self.GetDuties(full_output=True,
                                            **kwargs_items.get('duties', {}))
        unknown_duty_types = sorted(
            int(duty_type) for duty_type in np.unique(duty_types)
            if duty_type not in duty_cost_by_type)
        if unknown_duty_types:
            raise ValueError("Unknown duty type(s): %s" % unknown_duty_types)

        duty_unit_cost = np.zeros_like(duty_types, dtype=np.float64)  # [USD/GJ]
        for duty_type, unit_cost in duty_cost_by_type.items():
            duty_unit_cost[duty_types == duty_type] = unit_cost

        duty_cost = np.abs(duties)*1e-9 * duty_unit_cost  # [USD]

        # ---------- Raw materials
        raw_kwargs = kwargs_items.get('raw_materials', {}).copy()
        raw_kwargs['steady_state'] = steady_raw
        raw_kwargs['include_holdups'] = include_holdups

        raw_materials = self.GetRawMaterials(**raw_kwargs)
        if cost_raw.ndim > 1:
            raise ValueError("cost_raw must be a scalar or a one-dimensional array")
        if cost_raw.ndim == 1 and cost_raw.size not in (1, raw_materials.shape[1]):
            raise ValueError(
                "cost_raw must be scalar or have one entry per raw-material "
                "column")
        raw_cost = cost_raw * raw_materials  # [USD]

        # ---------- Labor
        labor_cost = self.GetLabor(**kwargs_items.get('labor', {}))

        if lumped:
            pass
        else:
            return duty_cost, raw_cost, labor_cost

    def CreateStatsObject(self, alpha=0.95):
        statInst = StatisticsClass(self.ParamInst, alpha=alpha)
        return statInst
