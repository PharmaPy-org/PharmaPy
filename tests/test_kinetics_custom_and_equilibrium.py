# -*- coding: utf-8 -*-
"""Regression tests for issue #71.

Two independent RxnKinetics defects, grouped as one kinetics-correctness fix:

1. Constructing ``RxnKinetics`` with a user-defined ``kinetic_model``
   (``elem_flag=False``) raises ``UnboundLocalError`` in ``set_params``.
2. The vectorized (array-``temp``) equilibrium reverse term divides by ``Keq``
   indexed on the time axis instead of the reaction axis, so every time point
   reuses the first row's ``Keq``.

Both live in ``PharmaPy.Kinetics``, which is pure NumPy (no solver backend),
so these tests need no assimulo marker.
"""

import numpy as np
import pytest

from PharmaPy.Kinetics import RxnKinetics


STOICH = [[-1, -1, 1, 0]]          # A + B -> C
PARTIC = ["A", "B", "C", "solv"]


def _db(data_path):
    return str(data_path["integration"] / "pfr_test_pure_comp.json")


def _custom_rate(conc, params_f):
    conc = np.maximum(1e-15, conc)
    return np.exp(np.dot(np.log(conc), np.atleast_2d(params_f).T))


def test_custom_kinetic_model_constructs(data_path):
    """Defect 1: the documented custom-rate-law workflow must construct."""

    # Must not raise UnboundLocalError in set_params.
    kin = RxnKinetics(
        path=_db(data_path),
        k_params=[1.0],
        ea_params=[0.0],
        stoich_matrix=STOICH,
        partic_species=PARTIC,
        kinetic_model=_custom_rate,
        params_f=np.array([[1.0, 1.0]]),
    )

    assert kin.elem_flag is False
    assert kin.kinetic_model is _custom_rate
    # The user-supplied f-parameters must be stored, not dropped.
    assert kin.params_f is not None
    np.testing.assert_allclose(np.asarray(kin.params_f).ravel(), [1.0, 1.0])


def test_custom_kinetic_model_requires_params_f(data_path):
    """Custom rate laws still require explicit f-parameters."""
    with pytest.raises(RuntimeError, match="params_f"):
        RxnKinetics(
            path=_db(data_path),
            k_params=[1.0],
            ea_params=[0.0],
            stoich_matrix=STOICH,
            partic_species=PARTIC,
            kinetic_model=_custom_rate,
            params_f=None,
        )


def test_custom_kinetic_model_flat_set_params_updates_params_f(data_path):
    """Parameter-estimation vector updates custom model f-parameters."""
    kin = RxnKinetics(
        path=_db(data_path),
        k_params=[1.0],
        ea_params=[0.0],
        stoich_matrix=STOICH,
        partic_species=PARTIC,
        kinetic_model=_custom_rate,
        params_f=np.array([[1.0, 1.0]]),
    )

    params = kin.concat_params()
    params[-2:] = [2.0, 3.0]

    kin.set_params(params)

    np.testing.assert_allclose(np.asarray(kin.params_f).ravel(), [2.0, 3.0])


def test_vectorized_equilibrium_matches_scalar_path(data_path):
    """Defect 2: array-temp reverse term must use per-reaction Keq.

    Oracle is the trusted scalar-temp / 1-D branch: evaluating each time point
    separately and stacking must equal the single vectorized 2-D evaluation.
    """
    kin = RxnKinetics(
        path=_db(data_path),
        k_params=[1.0],
        ea_params=[0.0],
        stoich_matrix=STOICH,
        partic_species=PARTIC,
        keq_params=[2.0],
        delta_hrxn=[-5e3],
        tref_hrxn=298.15,
    )

    temp = np.array([290.0, 310.0, 330.0])
    conc = np.array([[1.0, 1.0, 1.0, 1.0],
                     [2.0, 2.0, 2.0, 2.0],
                     [0.5, 0.5, 0.5, 0.5]])
    deltah = np.atleast_1d(-5e3)

    vectorized = np.asarray(kin.equilibrium_model(conc, temp, deltah))
    per_time = np.array([
        np.asarray(
            kin.equilibrium_model(conc[i], float(temp[i]), deltah)
        ).ravel()
        for i in range(temp.size)
    ])

    np.testing.assert_allclose(vectorized.reshape(per_time.shape), per_time)


def test_vectorized_equilibrium_accepts_2d_deltah(data_path):
    """Array-temp equilibrium must accept per-time heat-of-reaction values."""
    kin = RxnKinetics(
        path=_db(data_path),
        k_params=[1.0],
        ea_params=[0.0],
        stoich_matrix=STOICH,
        partic_species=PARTIC,
        keq_params=[2.0],
        delta_hrxn=[-5e3],
        tref_hrxn=298.15,
    )

    temp = np.array([290.0, 310.0, 330.0])
    conc = np.array([[1.0, 1.0, 1.0, 1.0],
                     [2.0, 2.0, 2.0, 2.0],
                     [0.5, 0.5, 0.5, 0.5]])
    deltah = np.array([[-5.0e3], [-4.5e3], [-4.0e3]])

    vectorized = np.asarray(kin.equilibrium_model(conc, temp, deltah))
    per_time = np.array([
        np.asarray(
            kin.equilibrium_model(conc[i], float(temp[i]), deltah[i])
        ).ravel()
        for i in range(temp.size)
    ])

    np.testing.assert_allclose(vectorized.reshape(per_time.shape), per_time)
