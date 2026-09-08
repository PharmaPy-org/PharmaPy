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

from PharmaPy.Connections import Connection, convert_str_flowsheet, topological_bfs
from PharmaPy.Errors import PharmaPyNonImplementedError
from PharmaPy.Results import SimulationResult, flatten_dict_fields, get_name_object

from PharmaPy.Commons import trapezoidal_rule, check_steady_state
from PharmaPy.CheckModule import check_modeling_objects

import time
from typing import Optional, Sequence


class SimulationExec:
    def __init__(self, pure_path, flowsheet):

        # Interfaces
        thermo_instance = ThermoPhysicalManager(pure_path)
        self.NamesSpecies = thermo_instance.name_species

        # Outputs
        self.StreamTable = None

        self.uos_instances = {}  # TODO: check this under the new graph implem
        self.oper_mode = []

        if isinstance(flowsheet, dict):
            graph = flowsheet
        elif isinstance(flowsheet, str):
            graph = convert_str_flowsheet(flowsheet)

        self.graph = graph
        self.in_degree, self.execution_names = topological_bfs(graph)

        if len(self.execution_names) < len(self.graph):
            raise PharmaPyNonImplementedError(
                "Provided flowsheet contains recycle stream(s)")

    def _transfer_to_neighbors(self, name: str, connections: dict, count: int,
                               pick_units: Optional[Sequence[str]] = None) -> int:
        """
        Transfer output data from a unit operation to graph successors.

        Parameters
        ----------
        name : str
            Name of the source unit operation in the flowsheet graph.
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
        for uo_next in self.graph[name]:
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

    def get_dynamic_raw_inputs(self, inlet, stream, time):
        """Evaluate dynamic raw-material inputs for one stream or phase.

        Parameters
        ----------
        inlet : object
            Raw inlet object. It may be a single stream or a mixed phase.
        stream : object
            Stream or phase currently being accounted.
        time : ndarray
            Simulation time grid [s].

        Returns
        -------
        dict
            Dynamic input profiles. Flow entries are [kg/s], [mol/s], or
            [m**3/s] according to their key; temperature is [K] and pressure is
            [Pa].

        Notes
        -----
        If a mixed phase owns a single dynamic inlet profile, the total dynamic
        flow is split by each phase's steady flow fraction [-]. This preserves
        the total mixed feed instead of integrating the full mixed profile once
        for every phase.
        """
        if getattr(stream, 'DynamicInlet', None) is not None:
            return stream.DynamicInlet.evaluate_inputs(time)

        inputs = inlet.DynamicInlet.evaluate_inputs(time)
        if stream is inlet:
            return inputs

        scaled = {}
        flow_fields = ('mass_flow', 'mole_flow', 'vol_flow')
        for field in flow_fields:
            if field not in inputs:
                continue

            inlet_flow = getattr(inlet, field, 0)
            stream_flow = getattr(stream, field, 0)
            if inlet_flow == 0:
                flow_fraction = 0
            else:
                flow_fraction = stream_flow / inlet_flow  # [-]

            scaled[field] = inputs[field] * flow_fraction

        for field in ('temp', 'pres'):
            if field in inputs:
                scaled[field] = inputs[field]

        return scaled

    def get_raw_inlets(self, uo, basis='mass'):
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
            temperature [K], pressure [Pa], and volume [m**3].

        Raises
        ------
        ValueError
            If ``basis`` is not ``'mass'`` or ``'mole'``.
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

        # inlets = [inlet for inlet in inlets
        #           if inlet is not None and inlet.y_upstream is None]

        out = {}

        for name, inlet in raws.items():
            if inlet.__module__ == 'PharmaPy.MixedPhases':
                streams = inlet.Phases
            else:
                streams = [inlet]

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
                elif (getattr(stream, 'DynamicInlet', None) is None and
                      getattr(inlet, 'DynamicInlet', None) is None):
                    time = uo.result.time[-1] - uo.result.time[0]  # [s]
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

                else:
                    time = uo.result.time  # [s]
                    inputs = self.get_dynamic_raw_inputs(inlet, stream, time)

                    if basis == 'mass':
                        if 'mass_flow' in inputs:
                            flow = inputs['mass_flow']  # [kg/s]
                        else:
                            flow = inputs['mole_flow'] * stream.mw_av / 1000  # [kg/s]

                        total = trapezoidal_rule(time, flow)  # [kg]

                        stream_data[name_stream] = {'mass': total}

                        fields += ['mass_frac']

                    elif basis == 'mole':
                        if 'mole_flow' in inputs:
                            flow = inputs['mole_flow']  # [mol/s]
                        else:
                            flow = inputs['mass_flow'] / stream.mw_av * 1000  # [mol/s]

                        total = trapezoidal_rule(time, flow)  # [mol]

                        stream_data[name_stream] = {'moles': total}
                        fields += ['mole_frac']

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
            If ``basis`` is not ``'mass'`` or ``'mole'``.
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

    def GetDuties(self, full_output=False):
        """
        Get heat duties for all equipment that calculates an energy balance.

        Parameters
        ----------
        full_output : bool, optional
            if True, duties and duty types are returened. The default is False.

        Returns
        -------
        heat_duties : pandas dataframe
            heat duties [J].

        duties_ids : numpy array
            2D array with first column containing heating type and
            second column containing refrigeration type, according to the
            following convention:

            refrigeration: -3, -2, -1
            cooling water: 0
            heating: 1, 2, 3 (1 corresponding to low pressure steam)

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
            width other than one or the raw-material table width.

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
