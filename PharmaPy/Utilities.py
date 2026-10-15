#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 13:52:55 2020

@author: dcasasor
"""

from typing import Optional, Union

import numpy as np

from PharmaPy.Connections import (_map_dynamic_inputs, get_inputs_new,
                                 get_remaining_states)


class CoolingWater:
    """Liquid-water heat-transfer utility for jackets, baths and evaporators.

    The utility supplies an inlet temperature [K] and a flow that may be
    expressed on a volume basis [m**3/s] or a mass basis [kg/s]. Both bases
    are always reported, linked by the constant utility density ``rho``.

    Notes
    -----
    Water is modeled as an incompressible liquid with constant density and
    heat capacity, independent of temperature. The values are nominal round
    figures for liquid water near ambient conditions; they are the model's
    established constant-property assumption rather than fitted data.
    """

    def __init__(self, vol_flow: Optional[float] = None,
                 mass_flow: Optional[float] = None,
                 temp_in: float = 298.15, h_conv: float = 1000) -> None:
        """Create a cooling-water utility with one specified flow basis.

        Parameters
        ----------
        vol_flow : float, optional
            Utility volume flow [m**3/s]. Takes precedence over
            ``mass_flow`` when both are given.
        mass_flow : float, optional
            Utility mass flow [kg/s], used when ``vol_flow`` is None.
        temp_in : float, optional
            Utility inlet temperature [K].
        h_conv : float, optional
            Utility-side convective heat-transfer coefficient [W/m**2/K].

        Raises
        ------
        RuntimeError
            If both ``vol_flow`` and ``mass_flow`` are None.
        """

        # Constant-property liquid water near ambient conditions; see the
        # class Notes. These values link the volume and mass flow bases.
        self.rho = 1000  # [kg/m**3]
        self.cp = 4180  # [J/kg/K]
        self.h_conv = h_conv

        if vol_flow is None and mass_flow is None:
            raise RuntimeError("Both 'vol_flow' and 'mass_flow' are None. "
                               "Specify one of them.")

        self.updateObject(vol_flow, mass_flow, temp_in)

        self.controllable = ('temp_in', 'vol_flow', 'mass_flow')

        # Outputs
        self.temp_out = None
        self.y_upstream = None

        self._DynamicInlet = None

    @property
    def DynamicInlet(self):
        return self._DynamicInlet

    @DynamicInlet.setter
    def DynamicInlet(self, dynamic_object: Optional[object]) -> None:
        """Attach a controller or restore static utility inputs.

        Parameters
        ----------
        dynamic_object : object or None
            Object exposing ``evaluate_inputs(time)`` with time [s].
            None restores stored inlet
            temperature [K] and utility flow [m**3/s or kg/s].
        """
        self._DynamicInlet = dynamic_object
        if dynamic_object is None:
            return
        dynamic_object.controllable = self.controllable
        dynamic_object.parent_instance = self

    def updateObject(self, vol_flow=None, mass_flow=None, temp_in=None):
        if vol_flow is not None:
            self.vol_flow = vol_flow
            self.mass_flow = vol_flow * self.rho
        elif mass_flow is not None:
            self.mass_flow = mass_flow
            self.vol_flow = mass_flow / self.rho

        if temp_in is not None:
            self.temp_in = temp_in

    def evaluate_inputs(self, time):
        if self.DynamicInlet is None:
            inputs = {}
            for attr in self.controllable:
                inputs[attr] = getattr(self, attr)

        else:
            inputs = self.DynamicInlet.evaluate_inputs(time)

        return inputs

    def get_inputs(self, time: Union[float, np.ndarray]) -> dict:
        """Return the utility inlet temperature and both flow bases.

        Parameters
        ----------
        time : float or numpy.ndarray
            Evaluation time [s], scalar or shape (num_times,).

        Returns
        -------
        inputs : dict
            ``'vol_flow'`` [m**3/s], ``'temp_in'`` [K] and ``'mass_flow'``
            [kg/s]. Uncontrolled fields hold the stored static values,
            repeated per time for an array ``time``. Controlled fields have
            the shape returned by the controller, and a flow derived from a
            controlled flow has the same shape.

        Raises
        ------
        ValueError
            If the ``DynamicInlet`` controls both ``mass_flow`` and
            ``vol_flow``.

        Notes
        -----
        The ``DynamicInlet`` only needs ``evaluate_inputs(time)``; the keys
        it returns are the controlled fields. It is evaluated once per call,
        and its fields are merged with the static ones exactly as
        ``Connections.get_inputs_new`` does. A controlled flow basis takes
        precedence over the stored static value of the other basis, which is
        recomputed elementwise with the utility density ``rho`` [kg/m**3]:
        ``vol_flow = mass_flow / rho`` or ``mass_flow = vol_flow * rho``.
        Controlling both bases is rejected because the two controls could
        disagree; control one flow basis only. A ``temp_in``-only control
        keeps the stored, mutually consistent static flows.
        """
        layout = {'Inlet': {'vol_flow': 1, 'temp_in': 1, 'mass_flow': 1}}
        if self.DynamicInlet is None:
            return get_inputs_new(time, self, layout)['Inlet']

        # Same merge as get_inputs_new's DynamicInlet branch, keeping the
        # evaluated controls so their keys identify the controlled basis.
        controlled = self.DynamicInlet.evaluate_inputs(time)
        mapped = _map_dynamic_inputs(controlled, layout)
        static = get_remaining_states(layout, self, mapped, time)
        inputs = {**mapped['Inlet'], **static['Inlet']}

        return self._reconcile_flow_bases(inputs, controlled)

    def _reconcile_flow_bases(self, inputs: dict, controlled: dict) -> dict:
        """Derive the uncontrolled flow basis from the controlled one.

        Parameters
        ----------
        inputs : dict
            Merged utility inputs with ``'vol_flow'`` [m**3/s],
            ``'temp_in'`` [K] and ``'mass_flow'`` [kg/s], scalar or one value
            per evaluation time, from the controls and stored static values.
        controlled : dict
            Fields returned by ``DynamicInlet.evaluate_inputs``; only its
            keys are used, to identify the controlled flow basis.

        Returns
        -------
        dict
            ``inputs`` with the uncontrolled flow basis replaced by the
            conversion of the controlled one using ``rho`` [kg/m**3]. It is
            returned unchanged when no flow basis is controlled.

        Raises
        ------
        ValueError
            If both ``mass_flow`` and ``vol_flow`` are controlled.
        """
        mass_controlled = 'mass_flow' in controlled
        vol_controlled = 'vol_flow' in controlled

        if mass_controlled and vol_controlled:
            raise ValueError(
                "CoolingWater.DynamicInlet controls both 'mass_flow' and "
                "'vol_flow'; control only one flow basis. The other basis is "
                "derived from it with the utility density "
                f"rho = {self.rho} kg/m**3.")
        elif mass_controlled:
            inputs['vol_flow'] = np.divide(
                inputs['mass_flow'], self.rho)  # [m**3/s]
        elif vol_controlled:
            inputs['mass_flow'] = np.multiply(
                inputs['vol_flow'], self.rho)  # [kg/s]

        return inputs
