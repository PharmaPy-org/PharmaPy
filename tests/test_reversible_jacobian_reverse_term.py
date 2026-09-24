# -*- coding: utf-8 -*-
"""Regression test for issue #23.

For a reversible reaction the analytical state Jacobian returned by
``RxnKinetics.get_rxn_rates(..., jac=True)`` (``derivatives``) must equal the
finite-difference Jacobian of the net rate ``get_rxn_rates(..., jac=False)``.

``derivatives`` differentiates only the forward term (``self.df_dstates`` ->
``elem_df_dstates``), so for any reversible (``keq_params``) reaction the
reverse-term contribution ``-d/dC[prod(C_prod**beta)/Keq]`` is missing from the
Jacobian. This is pure NumPy (no solver backend), so no assimulo marker.
"""

import numpy as np

from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Reactors import BatchReactor


# A <-> B (reversible); C and solv are inert padding species in the database.
STOICH = [[-1, 1, 0, 0]]
PARTIC = ["A", "B", "C", "solv"]


def _db(data_path):
    return str(data_path["integration"] / "pfr_test_pure_comp.json")


def _reversible_kinetics(data_path):
    return RxnKinetics(
        path=_db(data_path),
        k_params=[1.0],        # kf = 1 [1/time] at temp_ref = inf
        ea_params=[0.0],       # [J/mol]
        stoich_matrix=STOICH,
        partic_species=PARTIC,
        keq_params=[2.0],
        delta_hrxn=[0.0],      # [J/mol_rxn]; Keq is temperature-independent
        tref_hrxn=298.15,      # [K]
    )


def _temperature_sensitive_reversible_kinetics(data_path):
    return RxnKinetics(
        path=_db(data_path),
        k_params=[1.0],        # kf = 1 [1/time] at temp_ref = inf
        ea_params=[0.0],       # [J/mol]
        stoich_matrix=STOICH,
        partic_species=PARTIC,
        keq_params=[2.0],
        delta_hrxn=[-5.0e4],   # [J/mol_rxn]
        tref_hrxn=298.15,      # [K]
    )


def _irreversible_kinetics(data_path):
    return RxnKinetics(
        path=_db(data_path),
        k_params=[1.0],        # kf = 1 [1/time] at temp_ref = inf
        ea_params=[0.0],       # [J/mol]
        stoich_matrix=STOICH,
        partic_species=PARTIC,
    )


def test_reversible_jacobian_includes_reverse_term(data_path):
    kin = _reversible_kinetics(data_path)

    temp = 298.15  # [K]
    conc = np.array([1.0, 1.0, 1.0, 1.0])   # [mol/L]; C_B makes reverse nonzero
    deltah = np.array([0.0])  # [J/mol_rxn]

    analytical = np.asarray(
        kin.get_rxn_rates(conc, temp, overall_rates=True, jac=True)
    )

    # Central finite-difference Jacobian of the net species rates.
    def net_rates(c):
        return np.asarray(
            kin.get_rxn_rates(c, temp, overall_rates=True, delta_hrxn=deltah)
        )

    n = conc.size
    fd = np.zeros((n, n))
    h = 1e-6  # [mol/L]
    for j in range(n):
        cp = conc.copy()
        cp[j] += h
        cm = conc.copy()
        cm[j] -= h
        fd[:, j] = (net_rates(cp) - net_rates(cm)) / (2 * h)

    # Hand-computed truth for the participating (A, B) block: Keq = 2, kf = 1.
    # net = [-(C_A - C_B/2), +(C_A - C_B/2), 0, 0]
    #   d/dC_A = [-1, 1];  d/dC_B = [+0.5, -0.5]
    np.testing.assert_allclose(fd[0, 1], 0.5, atol=1e-5)
    np.testing.assert_allclose(fd[1, 1], -0.5, atol=1e-5)

    # The analytical Jacobian must reproduce the reverse-term B column.
    np.testing.assert_allclose(analytical, fd, atol=1e-5)


def test_irreversible_forward_jacobian_is_unchanged(data_path):
    kin = _irreversible_kinetics(data_path)

    analytical = np.asarray(
        kin.get_rxn_rates(
            np.array([1.0, 1.0, 1.0, 1.0]),  # [mol/L]
            298.15,  # [K]
            overall_rates=True,
            jac=True,
        )
    )

    expected = np.array([
        [-1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])

    np.testing.assert_allclose(analytical, expected, atol=1e-12)


def test_reversible_jacobian_supports_batched_concentrations(data_path):
    kin = _reversible_kinetics(data_path)

    temp = np.array([298.15, 310.0])  # [K]
    conc = np.array([
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 0.5, 1.0, 1.0],
    ])  # [mol/L]
    deltah = np.array([0.0])  # [J/mol_rxn]

    analytical = np.asarray(
        kin.get_rxn_rates(conc, temp, overall_rates=True, jac=True)
    )

    def net_rates(c):
        return np.asarray(
            kin.get_rxn_rates(c, temp, overall_rates=True, delta_hrxn=deltah)
        )

    n_times, n_species = conc.shape
    fd = np.zeros((n_times, n_species, n_species))
    h = 1e-6  # [mol/L]
    for i in range(n_times):
        for j in range(n_species):
            cp = conc.copy()
            cp[i, j] += h
            cm = conc.copy()
            cm[i, j] -= h
            fd[i, :, j] = (net_rates(cp) - net_rates(cm))[i] / (2 * h)

    assert analytical.shape == fd.shape
    np.testing.assert_allclose(analytical, fd, atol=1e-5)


def test_reversible_jacobian_uses_runtime_delta_hrxn(data_path):
    kin = _temperature_sensitive_reversible_kinetics(data_path)

    temp = 350.0  # [K]
    conc = np.array([1.0, 1.0, 1.0, 1.0])  # [mol/L]
    runtime_deltah = np.array([-4.0e4])  # [J/mol_rxn]

    analytical = np.asarray(
        kin.get_rxn_rates(
            conc,
            temp,
            overall_rates=True,
            jac=True,
            delta_hrxn=runtime_deltah,
        )
    )

    def net_rates(c):
        return np.asarray(
            kin.get_rxn_rates(
                c,
                temp,
                overall_rates=True,
                delta_hrxn=runtime_deltah,
            )
        )

    n = conc.size
    fd = np.zeros((n, n))
    h = 1e-6  # [mol/L]
    for j in range(n):
        cp = conc.copy()
        cp[j] += h
        cm = conc.copy()
        cm[j] -= h
        fd[:, j] = (net_rates(cp) - net_rates(cm)) / (2 * h)

    np.testing.assert_allclose(analytical, fd, rtol=1e-5, atol=1e-5)


def test_reactor_jacobian_passes_runtime_delta_hrxn(data_path):
    """The real reactor and phase pass temperature-corrected heat to kinetics."""
    temperature = 350.0  # [K]
    states = np.array([1.0, 1.0, 1.0, 1.0])  # [mol/L]
    liquid = LiquidPhase(
        _db(data_path),
        temp=temperature,
        vol=1.0,  # [m**3]
        mole_conc=states,
        verbose=False,
    )
    kinetics = _temperature_sensitive_reversible_kinetics(data_path)
    reactor = BatchReactor(isothermal=True)
    reactor.Phases = liquid
    reactor.Kinetics = kinetics
    reactor.set_names()

    runtime_deltah = liquid.getHeatOfRxn(
        kinetics.stoich_matrix,
        temperature,
        reactor.mask_species,
        kinetics.delta_hrxn,
        kinetics.tref_hrxn,
    )  # [J/mol_rxn]
    expected = kinetics.derivatives(
        states, temperature, delta_hrxn=runtime_deltah
    )
    reference_only = kinetics.derivatives(
        states, temperature, delta_hrxn=kinetics.delta_hrxn
    )

    jac = reactor.get_jacobians(
        time=0.0, states=states, sw=None, sens=None, params=None
    )

    assert not np.allclose(runtime_deltah, kinetics.delta_hrxn)
    assert not np.allclose(expected, reference_only)
    np.testing.assert_allclose(jac, expected)
