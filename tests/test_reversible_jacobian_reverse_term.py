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

from types import SimpleNamespace

import numpy as np

from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.Reactors import _BaseReactor


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
        cp = conc.copy(); cp[j] += h
        cm = conc.copy(); cm[j] -= h
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
            cp = conc.copy(); cp[i, j] += h
            cm = conc.copy(); cm[i, j] -= h
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
        cp = conc.copy(); cp[j] += h
        cm = conc.copy(); cm[j] -= h
        fd[:, j] = (net_rates(cp) - net_rates(cm)) / (2 * h)

    np.testing.assert_allclose(analytical, fd, rtol=1e-5, atol=1e-5)


def test_reactor_jacobian_passes_runtime_delta_hrxn():
    runtime_deltah = np.array([-4.0e4])  # [J/mol_rxn]

    class CaptureKinetics:
        num_species = 2
        keq_params = np.array([2.0])
        stoich_matrix = np.array([[-1.0, 1.0]])
        delta_hrxn = np.array([-5.0e4])  # [J/mol_rxn]
        tref_hrxn = 298.15  # [K]

        def __init__(self):
            self.calls = []

        def derivatives(self, conc, temp, dstates=True, delta_hrxn=None):
            self.calls.append((conc.copy(), temp, dstates, delta_hrxn.copy()))
            return np.eye(2)

    class CaptureLiquid:
        temp = 350.0  # [K]

        def __init__(self):
            self.calls = []

        def getHeatOfRxn(self, stoich_matrix, temp, mask, heat_rxn_ref,
                         tref_hrxn):
            self.calls.append(
                (stoich_matrix.copy(), temp, mask.copy(),
                 heat_rxn_ref.copy(), tref_hrxn))
            return runtime_deltah

    kinetics = CaptureKinetics()
    liquid = CaptureLiquid()
    reactor = SimpleNamespace(
        Kinetics=kinetics,
        Liquid_1=liquid,
        isothermal=True,
        mask_species=np.array([True, True]),
    )

    states = np.array([1.0, 1.0])  # [mol/L]
    jac = _BaseReactor.get_jacobians(
        reactor, time=0.0, states=states, sw=None, sens=None, params=None)

    np.testing.assert_allclose(jac, np.eye(2))
    assert len(liquid.calls) == 1
    assert len(kinetics.calls) == 1
    np.testing.assert_allclose(kinetics.calls[0][3], runtime_deltah)
