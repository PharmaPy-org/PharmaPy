"""Test the Mixer adiabatic solids energy balance.

The regressions cover batch and continuous slurry inlets, independently solve
the expected outlet temperature from phase enthalpies, and validate dispatch
for unsupported solids-bearing collaborators.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import brentq

from PharmaPy.Containers import Mixer
from PharmaPy.MixedPhases import Slurry, SlurryStream
from PharmaPy.Phases import LiquidPhase, SolidPhase
from PharmaPy.Streams import LiquidStream, SolidStream


pytestmark = pytest.mark.unit

# Liquid composition [-] over the database species (A, B, C, D, solvent).
# Identical in both inlets so that the mixed liquid composition, and hence the
# liquid density, is unchanged by mixing and volumes are exactly additive.
LIQUID_MASS_FRAC = [0.1, 0.1, 0.1, 0.1, 0.6]

# Solid composition [-]: pure component A.
SOLID_MASS_FRAC = [1.0, 0.0, 0.0, 0.0, 0.0]

# Volumetric shape factor of the crystals [-].
KV = 1.0

# Crystal size grid [um] and a volume-specific number density [#/m**3/um]
# whose third moment is 0.2 m**3/m**3 by the trapezoidal rule on the 100 um
# grid: 100 * (1e9 * 100**3 + 1.25e8 * 200**3) * 1e-18 = 0.2.
X_DISTRIB = np.array([0.0, 100.0, 200.0, 300.0])  # [um]
DISTRIB = np.array([0.0, 1.0e9, 1.25e8, 0.0])  # [#/m**3/um]

# Mixer geometry-free operating point.
MASS_LIQUID_INLET = 1.0  # [kg], solids-free inlet
VOL_SLURRY_INLET = 1.0e-3  # [m**3], slurry inlet

# Continuous mixer operating point. Two slurry streams contribute 1 L/s and
# 2 L/s to the 3 L/s outlet.
VOL_FLOW_SLURRY_INLET_ONE = 1.0e-3  # [m**3/s]
VOL_FLOW_SLURRY_INLET_TWO = 2.0e-3  # [m**3/s]

# Root [K] of the independent phase-level enthalpy-rate balance for the fixed
# 350 K / 300 K slurry-stream fixture defined below.
EXPECTED_STREAM_MIX_TEMP = 316.7476545148889  # [K]

# Bracket for the independent outlet-temperature solve [K]. Wide enough to
# contain any physically admissible adiabatic mixing temperature for the
# inlet temperatures used below.
TEMP_BRACKET = (250.0, 450.0)  # [K]


@pytest.fixture(scope="module")
def thermo_path(data_path):
    """Path of the shared pure-component thermodynamic database."""
    return str(data_path["flowsheet"] / "compound_database.json")


def _make_slurry_inlet(thermo_path, temp):
    """Build a slurry inlet at a uniform temperature.

    Parameters
    ----------
    thermo_path : str
        Path to the pure-component thermodynamic database.
    temp : float
        Temperature of both slurry phases [K].

    Returns
    -------
    PharmaPy.MixedPhases.Slurry
        Slurry of volume ``VOL_SLURRY_INLET`` [m**3] whose liquid and solid
        masses [kg] follow from the volume-specific distribution.
    """
    liquid = LiquidPhase(thermo_path, mass_frac=LIQUID_MASS_FRAC, temp=temp)
    solid = SolidPhase(thermo_path, mass_frac=SOLID_MASS_FRAC, kv=KV,
                       temp=temp)

    slurry = Slurry(vol=VOL_SLURRY_INLET, x_distrib=X_DISTRIB,
                    distrib=DISTRIB)
    slurry.Phases = (liquid, solid)

    return slurry


def _make_slurry_stream_inlet(thermo_path, temp, vol_flow):
    """Build a slurry-stream inlet at a uniform temperature.

    Parameters
    ----------
    thermo_path : str
        Path to the pure-component thermodynamic database.
    temp : float
        Temperature of both slurry-stream phases [K].
    vol_flow : float
        Total slurry volumetric flow rate [m**3/s].

    Returns
    -------
    PharmaPy.MixedPhases.SlurryStream
        Slurry stream whose liquid and solid mass flows [kg/s] follow from the
        volume-specific crystal distribution.
    """
    liquid = LiquidStream(
        thermo_path, mass_frac=LIQUID_MASS_FRAC, temp=temp
    )
    solid = SolidStream(
        thermo_path, mass_frac=SOLID_MASS_FRAC, kv=KV, temp=temp
    )

    slurry = SlurryStream(
        vol_flow=vol_flow, x_distrib=X_DISTRIB, distrib=DISTRIB
    )
    # The public phase setter expects the liquid collaborator first and
    # derives each phase flow rate from the slurry volumetric flow [m**3/s].
    slurry.Phases = (liquid, solid)

    return slurry


def _build_mixer(thermo_path, temp_liquid, temp_slurry):
    """Assemble a Mixer fed one solids-free inlet and one slurry inlet.

    Parameters
    ----------
    thermo_path : str
        Path to the pure-component thermodynamic database.
    temp_liquid : float
        Temperature of the solids-free liquid inlet [K].
    temp_slurry : float
        Temperature of the slurry inlet [K].

    Returns
    -------
    mixer : PharmaPy.Containers.Mixer
        Mixer with both inlets attached and ``Outlet`` set to the mixed
        slurry, as ``Mixer.balances_solids`` does before solving the energy
        balance.
    u_inputs : dict
        Inlet mapping returned by the real ``Mixer.get_inputs_solids``.
    masses : dict
        Total liquid and solid mass of the mixture [kg], keyed ``'liquid'``
        and ``'solid'``.

    Notes
    -----
    Issues #186 and #187 currently prevent the public ``Mixer.solve_unit``
    path from reaching the energy balance. Until they are fixed, this helper
    constructs the same outlet state directly; afterward these regressions
    should exercise ``solve_unit`` and assert its reported outlet temperature.
    """
    liquid_inlet = LiquidPhase(thermo_path, mass=MASS_LIQUID_INLET,
                               mass_frac=LIQUID_MASS_FRAC, temp=temp_liquid)
    slurry_inlet = _make_slurry_inlet(thermo_path, temp_slurry)

    mixer = Mixer()
    # The solids-free inlet is listed first because ``Mixer.Inlets`` reads
    # ``name_species`` off ``Inlets[0]``, which ``Slurry`` does not define.
    mixer.Inlets = [liquid_inlet, slurry_inlet]

    u_inputs, _ = mixer.get_inputs_solids()

    total_liquid = float(u_inputs['mass_liq'].sum())  # [kg]
    total_solid = float(u_inputs['mass_solid'].sum())  # [kg]

    # Additive volumes: both inlets share the liquid composition, so the
    # mixed liquid density equals each inlet's liquid density.
    dens_liquid = slurry_inlet.Liquid_1.getDensity(
        mass_frac=np.array(LIQUID_MASS_FRAC))  # [kg/m**3]
    vol_liquid_inlet = MASS_LIQUID_INLET / dens_liquid  # [m**3]
    vol_total = vol_liquid_inlet + VOL_SLURRY_INLET  # [m**3]

    # Volume-specific outlet distribution [#/m**3/um]: the solids-free inlet
    # contributes no crystals, so the slurry population is simply diluted.
    distrib_out = DISTRIB * VOL_SLURRY_INLET / vol_total

    outlet = Slurry(vol=vol_total, x_distrib=X_DISTRIB, distrib=distrib_out)
    outlet.Phases = (
        LiquidPhase(thermo_path, mass_frac=LIQUID_MASS_FRAC),
        SolidPhase(thermo_path, mass_frac=SOLID_MASS_FRAC, kv=KV),
    )

    # Fixture self-check: the constructed outlet must hold the same inventory
    # the material balance produces, otherwise the energy comparison below is
    # meaningless.
    assert outlet.Liquid_1.mass == pytest.approx(total_liquid, rel=1e-10)
    assert outlet.Solid_1.mass == pytest.approx(total_solid, rel=1e-10)

    mixer.Outlet = outlet

    return mixer, u_inputs, {'liquid': total_liquid, 'solid': total_solid}


def _build_stream_mixer(thermo_path, temp_first, temp_second):
    """Assemble a stream-fed Mixer and its continuous input mapping.

    Parameters
    ----------
    thermo_path : str
        Path to the pure-component thermodynamic database.
    temp_first : float
        Temperature of both phases in the first slurry stream [K].
    temp_second : float
        Temperature of both phases in the second slurry stream [K].

    Returns
    -------
    mixer : PharmaPy.Containers.Mixer
        Mixer with real stream inlets and a mixed slurry-stream outlet.
    u_inputs : dict
        Continuous inlet mapping whose phase amounts are flow rates [kg/s].
    mass_flows : dict
        Total liquid and solid mass flow [kg/s], keyed ``'liquid'`` and
        ``'solid'``.

    Notes
    -----
    Issue #188 tracks the current ``Mixer.get_inputs_solids`` failure on the
    deleted ``LiquidStream.mass`` attribute. Until that is fixed, this helper
    constructs the same mapping from the stream collaborators' authoritative
    ``mass_flow`` values [kg/s].
    """
    first_inlet = _make_slurry_stream_inlet(
        thermo_path, temp_first, VOL_FLOW_SLURRY_INLET_ONE
    )
    second_inlet = _make_slurry_stream_inlet(
        thermo_path, temp_second, VOL_FLOW_SLURRY_INLET_TWO
    )

    mixer = Mixer()
    # ``Mixer.Inlets`` reads ``name_species`` from its first collaborator,
    # which ``SlurryStream`` does not expose. Use the real first slurry's
    # liquid phase while the public setter establishes metadata, then restore
    # the real mixed-phase collaborator for the energy-balance dispatch.
    mixer.Inlets = [first_inlet.Liquid_1, second_inlet]
    mixer.Inlets[0] = first_inlet

    mass_liquid = np.array([
        first_inlet.Liquid_1.mass_flow, second_inlet.Liquid_1.mass_flow
    ])  # [kg/s]
    mass_solid = np.array([
        first_inlet.Solid_1.mass_flow, second_inlet.Solid_1.mass_flow
    ])  # [kg/s]
    massfrac_liquid = np.array([
        first_inlet.Liquid_1.mass_frac, second_inlet.Liquid_1.mass_frac
    ])  # [-]
    temps = np.array([
        first_inlet.Liquid_1.temp, second_inlet.Liquid_1.temp
    ])  # [K]
    # Slurry enthalpy dispatch does not consume the cake-only distribution
    # field. Preserve the mapping shape on the continuous number-rate basis.
    num_distrib = np.vstack((
        first_inlet.Solid_1.distrib, second_inlet.Solid_1.distrib
    ))  # [#/um/s]

    u_inputs = {
        'temp': temps,
        'mass_frac': massfrac_liquid,
        'mass_liq': mass_liquid,
        'mass_solid': mass_solid,
        'num_distrib': num_distrib,
    }

    total_liquid = float(mass_liquid.sum())  # [kg/s]
    total_solid = float(mass_solid.sum())  # [kg/s]
    total_vol_flow = (
        VOL_FLOW_SLURRY_INLET_ONE + VOL_FLOW_SLURRY_INLET_TWO
    )  # [m**3/s]

    outlet = SlurryStream(
        vol_flow=total_vol_flow, x_distrib=X_DISTRIB, distrib=DISTRIB
    )
    outlet.Phases = (
        LiquidStream(thermo_path, mass_frac=LIQUID_MASS_FRAC),
        SolidStream(thermo_path, mass_frac=SOLID_MASS_FRAC, kv=KV),
    )

    assert outlet.Liquid_1.mass_flow == pytest.approx(
        total_liquid, rel=1e-10
    )
    assert outlet.Solid_1.mass_flow == pytest.approx(
        total_solid, rel=1e-10
    )

    mixer.Outlet = outlet

    mass_flows = {'liquid': total_liquid, 'solid': total_solid}  # [kg/s]
    return mixer, u_inputs, mass_flows


def _expected_outlet_temp(mixer, u_inputs, masses):
    """Solve the adiabatic outlet temperature from the phase enthalpies.

    The balance is closed directly over the individual liquid and solid phase
    enthalpies [J/kg], independently of ``Mixer.energy_balance`` and of
    ``Slurry.getEnthalpy``::

        sum_i (m_liq_i h_liq(T_i) + m_sol_i h_sol(T_i))
            = M_liq h_liq(T_out) + M_sol h_sol(T_out)

    Parameters
    ----------
    mixer : PharmaPy.Containers.Mixer
        Mixer whose inlets supply the phase collaborators.
    u_inputs : dict
        Inlet mapping from ``Mixer.get_inputs_solids``.
    masses : dict
        Total liquid and solid mass [kg], or mass flow [kg/s], of the mixture.

    Returns
    -------
    float
        Adiabatic outlet temperature [K].
    """
    liquid_probe = mixer.Outlet.Liquid_1
    solid_probe = mixer.Outlet.Solid_1

    enthalpy_in = 0.0  # [J] for batch inputs or [J/s] for continuous inputs
    for ind, temp in enumerate(u_inputs['temp']):
        enthalpy_in += u_inputs['mass_liq'][ind] * liquid_probe.getEnthalpy(
            temp=temp, mass_frac=u_inputs['mass_frac'][ind], basis='mass')
        enthalpy_in += u_inputs['mass_solid'][ind] * solid_probe.getEnthalpy(
            temp=temp, basis='mass')

    def residual(temp):
        """Return the phase-level outlet enthalpy residual.

        Parameters
        ----------
        temp : float
            Trial outlet temperature [K].

        Returns
        -------
        float or numpy.floating
            Inlet-minus-outlet enthalpy [J] for batch inputs or enthalpy rate
            [J/s] for continuous inputs.
        """
        enthalpy_out = (
            masses['liquid'] * liquid_probe.getEnthalpy(
                temp=temp, mass_frac=np.array(LIQUID_MASS_FRAC), basis='mass')
            + masses['solid'] * solid_probe.getEnthalpy(temp=temp,
                                                        basis='mass')
        )  # [J] for batch inputs or [J/s] for continuous inputs

        return enthalpy_in - enthalpy_out

    return brentq(residual, *TEMP_BRACKET, xtol=1e-12, rtol=1e-14)


def test_mixer_energy_balance_preserves_isothermal_inlet_temperature(
        thermo_path):
    """Mixing streams already at a common temperature must not shift it.

    An adiabatic mixer fed a solids-free liquid and a slurry, both at
    ``temp_common``, must return ``temp_common``: no enthalpy is exchanged.
    Weighting the mixed-phase inlet enthalpy by liquid mass while the outlet
    carries the solid mass as well breaks this invariant.
    """
    temp_common = 300.0  # [K]

    mixer, u_inputs, _ = _build_mixer(thermo_path, temp_common, temp_common)

    temp_out = mixer.energy_balance(u_inputs)  # [K]

    assert temp_out == pytest.approx(temp_common, abs=1e-8)


def test_mixer_energy_balance_weights_inlets_by_total_slurry_mass(
        thermo_path):
    """The solved temperature must close a total-slurry-mass enthalpy balance.

    A hot solids-free inlet and a cold slurry inlet have different solid-to-
    liquid mass ratios, so a liquid-only weighting cannot be absorbed into a
    common factor and the solved temperature differs from the correct one.
    """
    temp_liquid = 350.0  # [K]
    temp_slurry = 300.0  # [K]

    mixer, u_inputs, masses = _build_mixer(thermo_path, temp_liquid,
                                           temp_slurry)

    expected_temp = _expected_outlet_temp(mixer, u_inputs, masses)  # [K]

    # Sanity: an adiabatic mix must land strictly between the inlet
    # temperatures, so the reference value cannot be trivially satisfied.
    assert temp_slurry < expected_temp < temp_liquid

    temp_out = mixer.energy_balance(u_inputs)  # [K]

    assert temp_out == pytest.approx(expected_temp, rel=1e-9)


def test_mixer_energy_balance_preserves_isothermal_stream_temperature(
        thermo_path):
    """A common inlet temperature must be invariant on the [J/s] basis.

    Parameters
    ----------
    thermo_path : str
        Path to the pure-component thermodynamic database.
    """
    temp_common = 300.0  # [K]

    mixer, u_inputs, _ = _build_stream_mixer(
        thermo_path, temp_common, temp_common
    )

    temp_out = mixer.energy_balance(u_inputs)  # [K]

    assert temp_out == pytest.approx(temp_common, abs=1e-8)


def test_mixer_energy_balance_closes_stream_enthalpy_rate(thermo_path):
    """Continuous mixing must close the phase-level enthalpy-rate balance.

    Parameters
    ----------
    thermo_path : str
        Path to the pure-component thermodynamic database.
    """
    temp_first = 350.0  # [K]
    temp_second = 300.0  # [K]

    mixer, u_inputs, mass_flows = _build_stream_mixer(
        thermo_path, temp_first, temp_second
    )
    expected_temp = _expected_outlet_temp(
        mixer, u_inputs, mass_flows
    )  # [K]

    # Pin the fixed fixture to its independently derived phase-level root so a
    # coupled change cannot move both the helper and production result together.
    assert expected_temp == pytest.approx(
        EXPECTED_STREAM_MIX_TEMP, abs=1e-9
    )

    temp_out = mixer.energy_balance(u_inputs)  # [K]

    assert temp_out == pytest.approx(expected_temp, rel=1e-9)


def test_mixer_energy_balance_rejects_unknown_solids_inlet(thermo_path):
    """Unknown solids-bearing collaborators must fail at dispatch.

    Parameters
    ----------
    thermo_path : str
        Path to the pure-component thermodynamic database.
    """
    temp_common = 300.0  # [K]
    mixer, u_inputs, _ = _build_mixer(
        thermo_path, temp_common, temp_common
    )
    mixer.Inlets[1] = SimpleNamespace(Solid_1=object())

    message = (
        r'^Mixer\.energy_balance supports Slurry and Cake solids-bearing '
        r'inlets; got SimpleNamespace\.$'
    )
    with pytest.raises(TypeError, match=message):
        mixer.energy_balance(u_inputs)
