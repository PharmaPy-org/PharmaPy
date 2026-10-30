#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
import json
import re
import warnings
from collections.abc import Mapping, Sequence
from typing import Optional, Union

from PharmaPy.Commons import get_permutation_indexes
from PharmaPy.Errors import PharmaPyTypeError, PharmaPyValueError

# from autograd import numpy as np

gas_ct = 8.314  # J/mol/K
eps = np.finfo(float).eps  # machine-epsilon floor for singularities

STOICH_COEFFICIENT_PATTERN = r'\d+(\.\d+)?(/\d+)?'
STOICH_COEFFICIENT_PREFIX = r'^' + STOICH_COEFFICIENT_PATTERN + r'\s?'
# Supported reaction-arrow tokens; '<-->' precedes '-->' so the longer token matches first.
REACTION_ARROWS = ('<-->', '-->', '<=>', '\u2192', '\u21c4')
REACTION_ARROW_PATTERN = re.compile(
    '|'.join(re.escape(arrow) for arrow in REACTION_ARROWS)
)
# Legacy zero-Keq guard avoids division by zero and makes the rate effectively irreversible.
ZERO_KEQ_REPLACEMENT = 1e20  # [Keq units], concentration basis of each raw reaction


def cryst_mechanism(sup_sat, moms, temp, temp_ref, params, reformulate, kv,
                    order):
    sec = False
    if len(params) == 3:
        phi_1, phi_2, exp = params
    else:
        phi_1, phi_2, exp, s_2 = params
        sec = True

    # absup = np.maximum(eps, sup_sat)
    absup_ = np.abs(sup_sat)

    absup = np.maximum(eps, absup_)
    if reformulate:
        pre_exp = np.exp(phi_1 + np.exp(phi_2)*(1/temp_ref - 1/temp))
    else:
        pre_exp = phi_1 * np.exp(-phi_2/gas_ct/temp)

    kinetic_term = pre_exp * sup_sat * absup**(exp - 1)

    if sec:
        if moms.ndim == 1:
            mom = np.maximum(0, moms[order]) # For vector moms

        elif moms.ndim == 2:
            mom= np.maximum(0, moms[:, order]) # for matrix
        kinetic_term *= (mom * kv)**s_2

    return kinetic_term

def _split_reaction_sides(rxn: str, sep: str = '-->') -> tuple[str, str]:
    """Split one reaction equation into reactant and product side strings.

    Parameters
    ----------
    rxn : str
        Reaction equation containing reactant and product groups separated by a
        reaction arrow.
    sep : str, optional
        Separator between reactants and products. When equal to the default
        ``'-->'``, any arrow in ``REACTION_ARROWS`` (``'-->'``, ``'<-->'``,
        ``'<=>'``, ``'→'``, ``'⇄'``) is accepted.

    Returns
    -------
    left : str
        Reactant-side substring.
    right : str
        Product-side substring.

    Raises
    ------
    ValueError
        If ``rxn`` is not a string, does not contain exactly one reaction
        arrow, or has an empty reactant or product side.
    """
    if not isinstance(rxn, str):
        raise ValueError(
            f"Reaction specification must be a string, got {type(rxn).__name__}."
        )
    if sep == '-->':
        matches = list(REACTION_ARROW_PATTERN.finditer(rxn))
        if len(matches) != 1:
            raise ValueError(
                f"Reaction {rxn!r} must contain exactly one reaction arrow "
                f"from {REACTION_ARROWS}."
            )
        match = matches[0]
        left = rxn[:match.start()]
        right = rxn[match.end():]
    else:
        if rxn.count(sep) != 1:
            raise ValueError(
                f"Reaction {rxn!r} must contain exactly one separator {sep!r}."
            )
        left, right = rxn.split(sep)

    if not left.strip() or not right.strip():
        raise ValueError(
            f"Reaction {rxn!r} must have non-empty reactant and product sides."
        )
    return left, right


def disect_rxns(
    rxns: Union[Sequence[str], Mapping[str, str]],
    sep: str = '-->',
) -> tuple[dict, list[str]]:
    """Separate reaction sides and collect participating species in order.

    Parameters
    ----------
    rxns : sequence of str or mapping of str to str
        Reactions as written, with coefficients [-] and ``'+'`` between
        species. May be supplied as an ordered sequence of reaction strings or
        a reaction-ID-keyed mapping.
    sep : str, optional
        Separator between reactants and products; defaults to ``'-->'``, which
        also accepts ``'<-->'``, ``'<=>'``, ``'→'``, and ``'⇄'``.

    Returns
    -------
    out : dict
        Mapping from zero-based reaction index to reactant and product strings,
        retaining the raw stoichiometric coefficients [-].
    species : list of str
        Unique species names in first-appearance order, with coefficients
        removed using the same prefix pattern as ``get_stoich``.

    Raises
    ------
    ValueError
        If ``rxns`` is empty, a reaction does not contain exactly one
        separator, or any reactant/product token lacks a species name.
    """
    if isinstance(rxns, Mapping):
        rxn_iterable = list(rxns.values())
    elif isinstance(rxns, str) or not isinstance(rxns, Sequence):
        raise ValueError(
            "rxns must be a sequence of reaction strings or a reaction-ID mapping."
        )
    else:
        rxn_iterable = list(rxns)

    if not rxn_iterable:
        raise ValueError("At least one reaction must be provided.")

    out = {}
    species = []

    for ind, rxn in enumerate(rxn_iterable):
        out[ind] = {}
        left, right = _split_reaction_sides(rxn, sep=sep)

        reactants = [x.strip() for x in left.split('+')]
        products = [x.strip() for x in right.split('+')]
        if any(not token for token in reactants + products):
            raise ValueError(
                f"Reaction {rxn!r} contains an empty species token."
            )

        out[ind]['reactants'] = reactants
        out[ind]['products'] = products

        species += reactants
        species += products

    for ind, sp in enumerate(species):
        cleaned = re.sub(STOICH_COEFFICIENT_PREFIX, '', sp).strip()
        if not cleaned:
            raise ValueError(
                f"Species token {sp!r} has no species identifier after its "
                "stoichiometric coefficient."
            )
        species[ind] = cleaned

    species = list(dict.fromkeys(species))

    return out, species


def get_coeff(pattern: str, expr: str) -> float:
    """Extract the leading stoichiometric coefficient magnitude from a token.

    Parameters
    ----------
    pattern : str
        Regular expression matching a leading integer, decimal, or fraction.
    expr : str
        Reactant or product token string.

    Returns
    -------
    float
        Stoichiometric coefficient magnitude [-]; defaults to ``1.0`` when
        omitted.
    """
    text = re.match(pattern, expr)

    if text is None:
        coeff = 1.0  # [-]
    elif '/' in text.group():
        num, denom = text.group().split('/')
        coeff = int(num) / int(denom)  # [-]
    else:
        coeff = float(text.group())  # [-]

    return coeff


def _get_side_stoich(
    di_rxn: dict,
    partic_species: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Build raw reactant and product stoichiometric matrices from parsed sides.

    Parameters
    ----------
    di_rxn : dict
        Reaction-index mapping from ``disect_rxns`` containing reactant and
        product strings with integer, decimal, or fractional coefficients [-].
    partic_species : list of str
        Species names in the desired output column order.

    Returns
    -------
    stoich_react : numpy.ndarray
        Shape ``(n_rxns, n_species)``; non-negative reactant stoichiometric
        coefficients [mol species/mol_rxn] on the reaction-as-written basis.
    stoich_prod : numpy.ndarray
        Shape ``(n_rxns, n_species)``; non-negative product stoichiometric
        coefficients [mol species/mol_rxn] on the reaction-as-written basis.

    Raises
    ------
    ValueError
        If a parsed species is absent from ``partic_species``.
    """
    num_rxns = len(di_rxn)
    num_species = len(partic_species)

    stoich_react = np.zeros((num_rxns, num_species), dtype=np.float64)  # [-], raw reactant basis
    stoich_prod = np.zeros((num_rxns, num_species), dtype=np.float64)  # [-], raw product basis

    for num, di in di_rxn.items():
        for r in di['reactants']:
            coeff = get_coeff(STOICH_COEFFICIENT_PATTERN, r)  # [-]
            r_name = re.sub(STOICH_COEFFICIENT_PREFIX, '', r).strip()
            col = partic_species.index(r_name)
            stoich_react[num, col] += coeff

        for p in di['products']:
            coeff = get_coeff(STOICH_COEFFICIENT_PATTERN, p)  # [-]
            p_name = re.sub(STOICH_COEFFICIENT_PREFIX, '', p).strip()
            col = partic_species.index(p_name)
            stoich_prod[num, col] += coeff

    return stoich_react, stoich_prod


def get_stoich(di_rxn: dict, partic_species: list) -> np.ndarray:
    """Build net stoichiometric rows from parsed reactions.

    Parameters
    ----------
    di_rxn : dict
        Reaction-index mapping from ``disect_rxns`` containing reactant and
        product strings with integer, decimal, or fractional coefficients [-].
    partic_species : list of str
        Species names in the desired output column order.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_rxns, n_species)``; net stoichiometric coefficients
        ``nu_prod - nu_react`` [mol species/mol_rxn] on the reaction-as-written
        basis (negative for net reactants, positive for net products, and zero
        for species appearing with equal coefficients on both sides).

    Raises
    ------
    ValueError
        If a parsed species is absent from ``partic_species``.
    """
    stoich_react, stoich_prod = _get_side_stoich(di_rxn, partic_species)  # [-]
    stoich = stoich_prod - stoich_react  # [-], net reaction basis
    return stoich


class RxnKinetics:
    """
    Create a reaction kinetics object. Reaction rate ``r_i`` is assumed to
    have the functional form ``r_i = f_1(T) * f_2(C_1, ..., C_n_comp)``, with
    the temperature-dependent term ``f_1 = k_i * exp(-Ea_i / R / T)``.

    The composition-dependent term ``f_2`` can be passed as a user-defined
    function. If not given, ``f_2`` is calculated from the concentrations and
    reaction orders of the reactants.

    Reaction orders are determined automatically by PharmaPy from the
    stoichiometric matrix. Custom orders can be passed through ``params_f``.

    Parameters
    ----------
    path : str
        path to the pure-component json file database
    k_params : list or tuple
        Pre-exponential factors for the temperature term. For total forward
        order ``m``, units are ``[mol/L]**(1-m)/time``; first-order factors
        have units [1/s] when the time basis is seconds.
    ea_params : list or tuple
        activation energy [J/mol] value(s) for the temperature-dependent
        term f\ :sub:`1`.
    rxn_list : list of str, optional
        list containing reactions represented by strings, where the
        pattern '+' separates reactants or products from one another, and
        the pattern --> separates groups of reactants from groups of
        products. Examples include ``A + B --> C``, ``2A --> B``, and
        ``2 H2O --> 2 H2 + O2``.

        Note that integer, float and fractional stoichiometric coefficients
        are supported.

        The names used for the reactions have to match those on the
        pure-component json file. If ``rxn_list`` is None, then both
        ``stoich_matrix`` and ``partic_species`` have to be passed
        (see below). The default is None.
    stoich_matrix : numpy array, optional
        stoichiometric matrix for the set of reactions. It must have
        n_rxn rows and n_comp columns, so the element (i, j) represents
        the coefficient of species j in reaction i [-] on the raw reaction
        basis. Every row must contain at least one negative reactant
        coefficient; product-only and all-zero rows are invalid. Stored as
        float64 after reordering columns to the component database order.
    partic_species : list or tuple of str, optional
        names of participating species. It will be assumed that the
        order of the names in ``partic_species`` is that of the columns of
        ``stoich_matrix``. The passed names must match those
        in the pure-component json file
    keq_params : array-like, optional
        Equilibrium constant for each reaction at ``tref_hrxn``. Units are
        ``[mol/L]**(sum of product orders - sum of reactant orders)``, so the
        constant is dimensionless only when those sums are equal. If provided,
        reversible rates use the elementary mass-action form documented below.
        Forward orders must equal the raw reactant stoichiometric coefficients
        for thermodynamic consistency. Orders are fixed and excluded from
        fitted parameters even when supplied explicitly. Custom
        ``kinetic_model`` callbacks are not supported with ``keq_params``.
        The default is None.
    params_f : numpy array, optional
        parameters for the concentration-dependent term f\ :sub:`2`.
        If no custom model is provided through the 'kinetic_model'
        argument, then 'params_f' values are interpreted as the reaction
        orders [-] of the built-in elementary reaction kinetic model.
        The params_f argument is optional only if no custom model is provided.
        If not given, the reaction orders are set to the stoichiometric
        coefficients for the involved reactants. With ``keq_params``, explicit
        orders are accepted within an absolute tolerance of 1e-12 [-], then
        replaced by the exact stoichiometric values. They are excluded from
        fitted parameters and parameter Jacobians. The default is None.
    temp_ref : float, optional
        reference temperature [K]. If not passed, it will be set to np.inf.
        The default is None.
    reformulate_kin : bool, optional
        if True, f\ :sub:`1` (T) will be reformulated as:

            f\ :sub:`1` (T) = exp[phi\ :sub:`1` + exp(phi\ :sub:`2`) * (1/T_ref - 1/T)]

        where phi\ :sub:`1` = ln(ki\ :sub:`i`) - Ea/R/T_ref and phi\ :sub:`2` = ln(Ea/R)
        We recommend to use this reparametrization when performing
        parameter estimation with datasets at different temperatures.
        The default is False.
    delta_hrxn : float or array-like, optional
        Heat of reaction at ``tref_hrxn`` for each reaction
        [J/mol of reaction as written]. Values are defined on the basis of
        the raw ``stoich_matrix`` rows rather than ``normalized_stoich``. A
        positive value is endothermic. The default is 0; None also selects 0.
    tref_hrxn : float, optional
        Reference temperature for ``delta_hrxn`` [K]. If None, it is set to
        ``temp_ref``. The default is 298.15.

    kinetic_model : callable, optional  
        kinetic model to be used to compute f\ :sub:`2`. It must have
        the signature:

            >>> kin_model(conc, params, *args). The default is None.

        Custom models are supported only for irreversible kinetics. Supplying
        both ``kinetic_model`` and ``keq_params`` raises ValueError because
        reversible kinetics use the built-in elementary form.

    df_dstates : callable, optional
        Derivative of a user-defined concentration term with respect to
        concentrations. The expected units are those of ``kinetic_model``
        divided by [mol/L]. The default is None.
    df_dtheta : callable, optional
        Derivative of a user-defined concentration term with respect to its
        parameters. The default is None.

    Attributes
    ----------
    stoich_normalization : numpy.ndarray, shape (n_rxns,)
        Magnitude of the first reactant's coefficient in each raw reaction
        [-]. Per-reaction rates from
        ``get_rxn_rates(overall_rates=False)`` use the reaction extent formed
        by dividing each raw reaction by this factor.
    normalized_stoich : numpy.ndarray, shape (n_comp, n_rxns)
        Stoichiometric matrix divided by ``stoich_normalization`` and
        transposed for mapping reaction rates to species rates [-].

    Returns
    -------
    RxnKinetics object.

    Raises
    ------
    PharmaPyValueError
        If any stoichiometric row lacks a negative reactant coefficient.
    ValueError
        If reversible forward orders differ from reactant stoichiometry beyond
        roundoff tolerance, or a custom model is combined with ``keq_params``.

    Notes
    -----
    For reversible elementary reactions, the rate is
    ``r_i = k_i(T) * (prod(C_reactant**alpha) - prod(C_product**beta)/Keq_i(T))``.
    Concentrations are [mol/L]; alpha and beta are magnitudes of the raw
    reactant and product stoichiometric coefficients [-]. ``Keq`` is the
    concentration-based equilibrium constant for that reaction as written,
    using the ideal concentration mass-action convention (no activity model).
    This concentration quotient equals ``Keq`` at equilibrium only when the
    forward orders equal alpha, so non-stoichiometric reversible orders are
    rejected at construction and by ``set_params``. Rates [mol/L/time] use
    the normalized extent defined by ``stoich_normalization``; species rates
    are ``normalized_stoich @ r``. Parameter Jacobians preserve this basis.

    Reactor energy balances convert raw-basis reaction enthalpies to the
    normalized rate basis before multiplying them by per-reaction rates.
    Supplying ``delta_hrxn`` values that were already divided by
    ``stoich_normalization`` would therefore apply the normalization twice.

    """
    def __init__(self, path, k_params=None, ea_params=None, rxn_list=None,
                 stoich_matrix=None, partic_species=None,
                 temp_ref=None, reformulate_kin=False,
                 keq_params=None, params_f=None, delta_hrxn=0,
                 tref_hrxn=298.15, kinetic_model=None, df_dstates=None,
                 df_dtheta=None, rxns=None, dh_rxn=None,
                 fit_paramsf: Optional[bool] = None,
                 params_r=None) -> None:
        """Initialize species ordering, reaction normalization, and parameters.

        Parameters
        ----------
        path : str
            Pure-component JSON database path; its order defines state columns.
        k_params : array-like or mapping of str to float
            Factors ``[mol/L]**(1-m)/time`` for total forward order m [-].
            May be a reaction-ID-keyed mapping, a 1-D sequence aligned with
            ``rxn_names``, or a scalar/length-1 value.
        ea_params : array-like or mapping of str to float, optional
            Activation energies [J/mol]. May be a reaction-ID-keyed mapping, a
            1-D sequence aligned with ``rxn_names``, or a scalar/length-1 value.
            Defaults to 0.0 [J/mol] when omitted or None.
        rxn_list : sequence of str or mapping of str to str, optional
            Reactions as written, used instead of ``stoich_matrix`` if given.
            Accepts ASCII (``'-->'``, ``'<-->'``, ``'<=>'``) and Unicode
            (``'→'``, ``'⇄'``) reaction arrows. When supplied as a mapping, its
            keys define ``self.rxn_names`` in insertion order.
        stoich_matrix : array-like, optional
            Raw coefficients [-], shape (n_rxns, n_species); each row requires
            a negative reactant coefficient.
        partic_species : list of str, optional
            Matrix column names; required with ``stoich_matrix``.
        temp_ref : float, optional
            Arrhenius reference temperature [K]; None uses infinity.
        reformulate_kin : bool, optional
            Store logarithmic Arrhenius parameters when True.
        keq_params : array-like or mapping of str to float, optional
            Raw-basis equilibrium constants with units
            ``[mol/L]**(sum(beta)-sum(alpha))``; None selects irreversible rates.
        params_f : array-like or mapping, optional
            Forward orders [-] grouped by reaction/reactant (including
            reaction-ID mappings of species-keyed order mappings), or custom
            model parameters in that model's units. Defaults to raw reactant
            orders.
        delta_hrxn : float, array-like, or mapping of str to float, optional
            Reference heat [J/mol_rxn] on the raw reaction basis; defaults to 0.
            None also selects 0.
        tref_hrxn : float, optional
            Heat reference temperature [K]; None uses ``temp_ref``.
        kinetic_model : callable, optional
            Concentration term accepting concentrations [mol/L], parameters,
            and extra arguments; defaults to the elementary power law.
            Custom models cannot be combined with ``keq_params``.
        df_dstates : callable, optional
            Custom concentration derivative, in concentration-term units
            divided by [mol/L].
        df_dtheta : callable, optional
            Custom parameter derivative, in concentration-term units divided
            by each custom parameter's units.
        rxns : sequence of str or mapping of str to str, optional
            Alias for ``rxn_list``. Cannot be passed simultaneously with
            ``rxn_list``.
        dh_rxn : float, array-like, or mapping of str to float, optional
            Alias for ``delta_hrxn`` [J/mol_rxn]. Cannot be passed
            simultaneously with a non-default ``delta_hrxn``.
        fit_paramsf : bool, optional
            Whether explicit elementary reaction orders in ``params_f`` are
            included in ``concat_params`` and parameter Jacobians. Defaults to
            True when ``params_f`` is provided for irreversible kinetics and
            False otherwise.
        params_r : array-like or mapping, optional
            Reverse reaction orders [-] grouped by reaction/product (including
            reaction-ID mappings of species-keyed order mappings). For
            reversible elementary reactions (``keq_params`` not ``None``),
            reverse orders must equal raw product stoichiometry ``stoich_prod``
            for thermodynamic consistency with ``Keq``; non-zero reverse
            orders without ``keq_params`` raise ``ValueError``.

        Raises
        ------
        PharmaPyTypeError
            If matrix input has no participating species list.
        PharmaPyValueError
            If a reaction lacks a negative reactant coefficient.
        ValueError
            If reaction or parameter mappings have invalid or mismatched
            reaction identifiers or species keys, if reversible elementary
            orders differ from raw reactant stoichiometry beyond roundoff
            tolerance, or if a custom kinetic model is combined with
            ``keq_params``.
        RuntimeError
            If a custom kinetic model has no ``params_f``.
        """
        if keq_params is not None and kinetic_model is not None:
            raise ValueError(
                "Reversible (equilibrium) kinetics use the built-in elementary "
                "form and are not supported with a custom kinetic model.")

        if rxn_list is not None and rxns is not None:
            raise ValueError(
                "Pass reaction specifications through either rxn_list or rxns, not both."
            )
        if rxns is not None:
            rxn_list = rxns

        if dh_rxn is not None:
            is_default_deltah = (
                isinstance(delta_hrxn, (int, float, np.number))
                and not isinstance(delta_hrxn, bool)
                and delta_hrxn == 0
            )
            if not is_default_deltah:
                raise ValueError(
                    "Pass heat of reaction through either delta_hrxn or dh_rxn, not both."
                )
            delta_hrxn = dh_rxn

        with open(path) as f:
            db = json.load(f)

        name_species = list(db.keys())

        # Stoichiometry
        if rxn_list is not None:
            if isinstance(rxn_list, Mapping):
                if not rxn_list:
                    raise ValueError("rxn_list mapping must not be empty.")
                for rxn_id in rxn_list.keys():
                    if not isinstance(rxn_id, str) or not rxn_id.strip():
                        raise ValueError(
                            f"Reaction identifiers must be non-empty strings, got {rxn_id!r}."
                        )
                rxn_names = tuple(rxn_list.keys())
                self._has_explicit_rxn_names = True
            else:
                rxn_names = None
                self._has_explicit_rxn_names = False

            di, partic_species = disect_rxns(rxn_list)
            stoich_react, stoich_prod = _get_side_stoich(di, partic_species)  # [-]
            stoich_matrix = stoich_prod - stoich_react  # [-]
            if rxn_names is None:
                rxn_names = tuple(
                    f"rxn_{ind}" for ind in range(1, stoich_matrix.shape[0] + 1)
                )
        else:
            stoich_matrix = np.atleast_2d(
                np.asarray(stoich_matrix, dtype=np.float64))  # [-]
            if partic_species is None:
                raise PharmaPyTypeError('Please provide a participating species list when using a stoichiometric matrix.')
            stoich_react = np.maximum(-stoich_matrix, 0.0)  # [-]
            stoich_prod = np.maximum(stoich_matrix, 0.0)  # [-]
            rxn_names = tuple(
                f"rxn_{ind}" for ind in range(1, stoich_matrix.shape[0] + 1)
            )
            self._has_explicit_rxn_names = False

        unknown_partic = [sp for sp in partic_species if sp not in name_species]
        if unknown_partic:
            raise ValueError(
                f"Unknown participating species {unknown_partic}; "
                f"available database species are {name_species}."
            )

        perm_idx = get_permutation_indexes(name_species, partic_species)
        stoich_matrix = stoich_matrix[:, perm_idx]  # [-]
        stoich_react = stoich_react[:, perm_idx]  # [-]
        stoich_prod = stoich_prod[:, perm_idx]  # [-]

        partic_species = [partic_species[ind] for ind in perm_idx]
        self.partic_species = partic_species
        self.rxn_names = rxn_names
        self.stoich_react = stoich_react  # [-]
        self.stoich_prod = stoich_prod  # [-]

        self.num_rxns, self.num_species = stoich_matrix.shape
        self._fit_paramsf_override = fit_paramsf

        if temp_ref is None:
            temp_ref = np.inf  # [K]

        self.temp_ref = temp_ref  # [K]
        self.reformulate_kin = reformulate_kin

        self.args_kin = ()

        # ---------- Kinetic model
        self.elem_flag = False
        if kinetic_model is None:
            self.kinetic_model = self.elem_f_model
            self.df_dstates = self.elem_df_dstates
            self.df_dthetaf = self.elem_df_dtheta

            self.elem_flag = True
        else:
            self.kinetic_model = kinetic_model
            self.df_dstates = df_dstates
            self.df_dthetaf = df_dtheta

        # Reject undefined reaction extents before choosing a normalization.
        invalid_rows = np.flatnonzero(~(stoich_matrix < 0).any(axis=1))
        if invalid_rows.size:
            raise PharmaPyValueError(
                "Each reaction requires a negative reactant coefficient; "
                f"invalid zero-based rows: {invalid_rows.tolist()}")

        # Normalize stoichiometric coefficients
        first_negative = (stoich_matrix < 0).argmax(axis=1)
        ref_stoich = np.zeros(self.num_rxns)  # [-]

        for ind in range(self.num_rxns):
            ref_stoich[ind] = stoich_matrix[ind, first_negative[ind]]

        # Magnitude of the first reactant coefficient in each raw reaction.
        self.stoich_normalization = abs(ref_stoich)  # [-]
        self.normalized_stoich = (
            stoich_matrix.T / self.stoich_normalization)  # [-]
        self.stoich_matrix = stoich_matrix  # [-]

        # Equilibrium kinetics
        if keq_params is None:
            self.keq_params = keq_params
        else:
            self.keq_params = self._align_rxn_param(
                keq_params, 'keq_params', allow_scalar=False
            )  # [(mol/L)**sum(stoich)]

        # ---------- Parameters
        params_dict = {'k_params': k_params, 'ea_params': ea_params,
                       'keq_params': self.keq_params, 'params_f': params_f,
                       'params_r': params_r}
        if fit_paramsf is not None:
            params_dict['fit_paramsf'] = fit_paramsf

        self.set_params(params_dict)
        self.nomenclature(stoich_matrix, self.phi_1)

        # Heat of reaction
        dh_param_name = 'dh_rxn' if dh_rxn is not None else 'delta_hrxn'
        if delta_hrxn is None:
            delta_hrxn = 0  # [J/mol_rxn], documented default on the raw basis
        if isinstance(delta_hrxn, Mapping):
            self.delta_hrxn = self._align_rxn_param(
                delta_hrxn, dh_param_name, allow_scalar=True
            )  # [J/mol_rxn]
        else:
            self.delta_hrxn = np.atleast_1d(delta_hrxn)  # [J/mol_rxn]
        if tref_hrxn is None:
            self.tref_hrxn = temp_ref  # [K]
        else:
            self.tref_hrxn = tref_hrxn  # [K]

        # Outputs
        self.rxn_rates = None
        self.time_profile = None
        self.conc_profile = None
        self.sensitivities = None

    def _align_rxn_param(
        self,
        value: Union[float, Sequence[float], np.ndarray, Mapping[str, float]],
        name: str,
        allow_scalar: bool = False,
        allow_shared_single: bool = False,
    ) -> np.ndarray:
        """Align a per-reaction parameter array or reaction-ID mapping.

        Parameters
        ----------
        value : float, sequence of float, numpy.ndarray, or mapping of str to float
            Parameter values to align with ``self.rxn_names`` in their stored
            physical units.
        name : str
            Parameter argument name used in validation messages.
        allow_scalar : bool, optional
            Whether a scalar (0-D) or length-1 value is permitted when
            ``self.num_rxns > 1``.
        allow_shared_single : bool, optional
            Whether a 1-D sequence of length 1 is permitted as a shared
            parameter across reactions.

        Returns
        -------
        numpy.ndarray
            One-dimensional float64 array aligned with ``self.rxn_names`` (or
            length 1 when a scalar/shared parameter is preserved).

        Raises
        ------
        ValueError
            If ``value`` is a mapping with missing or unexpected reaction IDs,
            or a sequence whose length does not match ``self.num_rxns``.
        """
        if isinstance(value, Mapping):
            non_str = [k for k in value.keys() if not isinstance(k, str)]
            if non_str:
                raise ValueError(
                    f"{name} mapping keys must be strings, got {non_str!r}."
                )
            missing = [k for k in self.rxn_names if k not in value]
            unexpected = [k for k in value.keys() if k not in self.rxn_names]
            if missing or unexpected:
                raise ValueError(
                    f"{name} keys must match reaction identifiers "
                    f"{list(self.rxn_names)}; missing={missing}, "
                    f"unexpected={unexpected}."
                )
            aligned = np.asarray(
                [value[k] for k in self.rxn_names], dtype=np.float64
            )  # [parameter units]
            if aligned.ndim != 1 or len(aligned) != self.num_rxns:
                raise ValueError(
                    f"{name} mapping values must be scalar numbers for each "
                    f"reaction in {list(self.rxn_names)}."
                )
            return aligned

        arr = np.asarray(value, dtype=np.float64)  # [parameter units]
        if arr.ndim == 0:
            if self.num_rxns > 1 and not (allow_scalar or allow_shared_single):
                raise ValueError(
                    f"{name} must provide one value per reaction "
                    f"({self.num_rxns} expected, got scalar)."
                )
            return np.atleast_1d(arr)

        if arr.ndim != 1:
            raise ValueError(
                f"{name} must be a 1-D sequence or reaction-ID mapping, "
                f"got shape {arr.shape}."
            )

        if len(arr) != self.num_rxns and not (
            (allow_scalar or allow_shared_single) and len(arr) == 1
        ):
            raise ValueError(
                f"{name} length ({len(arr)}) must match the number of "
                f"reactions ({self.num_rxns})."
            )
        return arr

    def _coerce_elementary_orders(
        self,
        params_f: Union[Sequence, np.ndarray, Mapping],
        *,
        stoich_side: Optional[np.ndarray] = None,
        param_name: str = 'params_f',
    ) -> tuple[np.ndarray, np.ndarray]:
        """Normalize explicit elementary reaction orders to database species order.

        Parameters
        ----------
        params_f : sequence, numpy.ndarray, or mapping
            Reaction orders [-]. Accepts:
            - A reaction-ID-keyed mapping whose values are either species-keyed
              order mappings ``{species: order}`` or per-side order sequences.
            - A sequence of species-keyed order mappings, one per reaction.
            - A 2-D array of shape ``(num_rxns, num_species)`` in
              ``self.partic_species`` order.
            - A sequence of per-reaction side-order sequences in
              ``self.partic_species`` order.
        stoich_side : numpy.ndarray, optional
            Default stoichiometric exponents [-], shape
            ``(num_rxns, num_species)``. Defaults to ``self.stoich_react``
            for forward orders and ``self.stoich_prod`` for reverse orders.
        param_name : str, optional
            Parameter argument name used in error messages (``'params_f'`` or
            ``'params_r'``).

        Returns
        -------
        orders : numpy.ndarray
            Reaction orders [-], shape ``(num_rxns, num_species)``.
        order_map : numpy.ndarray
            Boolean mask of active order positions, shape
            ``(num_rxns, num_species)``.

        Raises
        ------
        ValueError
            If reaction IDs, species names, or row lengths are invalid.
        """
        if stoich_side is None:
            stoich_side = self.stoich_react  # [-]
        default_side_map = stoich_side > 0

        if isinstance(params_f, Mapping):
            non_str = [k for k in params_f.keys() if not isinstance(k, str)]
            if non_str:
                raise ValueError(
                    f"{param_name} mapping keys must be strings, got {non_str!r}."
                )
            missing = [k for k in self.rxn_names if k not in params_f]
            unexpected = [k for k in params_f.keys() if k not in self.rxn_names]
            all_species_mappings = bool(params_f) and all(
                isinstance(val, Mapping) for val in params_f.values()
            )
            if not params_f or unexpected or (missing and not all_species_mappings):
                raise ValueError(
                    f"{param_name} keys must match reaction identifiers "
                    f"{list(self.rxn_names)}; missing={missing}, "
                    f"unexpected={unexpected}."
                )
            row_specs = [params_f.get(k, {}) for k in self.rxn_names]
        elif isinstance(params_f, np.ndarray):
            if params_f.ndim == 2 and params_f.shape == (self.num_rxns, self.num_species):
                orders = np.asarray(params_f, dtype=np.float64)  # [-]
                order_map = default_side_map | (orders != 0)
                return orders, order_map
            if params_f.ndim == 1 and self.num_rxns == 1:
                row_specs = [params_f]
            elif params_f.ndim == 2 and len(params_f) == self.num_rxns:
                row_specs = list(params_f)
            else:
                raise ValueError(
                    f"{param_name} array shape {params_f.shape} is incompatible with "
                    f"{self.num_rxns} reaction(s) and {self.num_species} species."
                )
        elif isinstance(params_f, Sequence) and not isinstance(params_f, str):
            if len(params_f) == 0:
                raise ValueError(f"{param_name} cannot be empty.")
            if not isinstance(params_f[0], (list, tuple, np.ndarray, Mapping)):
                row_specs = [params_f]
            else:
                row_specs = list(params_f)
        else:
            raise ValueError(
                f"{param_name} must be a sequence, 2-D array, or reaction-ID mapping."
            )

        if len(row_specs) != self.num_rxns:
            raise ValueError(
                f"{param_name} length ({len(row_specs)}) must match the number of "
                f"reactions ({self.num_rxns})."
            )

        orders = np.zeros_like(self.stoich_matrix, dtype=np.float64)  # [-]
        for ind, row_spec in enumerate(row_specs):
            side_mask = default_side_map[ind]
            num_side = int(side_mask.sum())
            if isinstance(row_spec, Mapping):
                unknown_sp = [
                    sp for sp in row_spec.keys() if sp not in self.partic_species
                ]
                if unknown_sp:
                    raise ValueError(
                        f"{param_name} for reaction {self.rxn_names[ind]!r} contains "
                        f"unknown participating species {unknown_sp}; valid "
                        f"participating species are {self.partic_species}."
                    )
                orders[ind] = stoich_side[ind].copy()  # [-]
                for sp_name, order_val in row_spec.items():
                    sp_idx = self.partic_species.index(sp_name)
                    orders[ind, sp_idx] = float(order_val)  # [-]
            else:
                row_arr = np.atleast_1d(np.asarray(row_spec, dtype=np.float64))  # [-]
                if row_arr.ndim != 1:
                    raise ValueError(
                        f"{param_name} row {ind} must be 1-D, got shape {row_arr.shape}."
                    )
                if len(row_arr) == num_side:
                    orders[ind, side_mask] = row_arr
                elif len(row_arr) == self.num_species:
                    orders[ind] = row_arr
                else:
                    raise ValueError(
                        f"{param_name} row {ind} for reaction {self.rxn_names[ind]!r} "
                        f"has {len(row_arr)} entries, expected {num_side} "
                        f"reactant order(s) or {self.num_species} species order(s)."
                    )

        order_map = default_side_map | (orders != 0)
        return orders, order_map

    def transform_params(self, kvals, evals):
        """Transform Arrhenius parameters into native or logarithmic coordinates.

        Parameters
        ----------
        kvals : numpy.ndarray
            Pre-exponential or reference-temperature rate factors
            ``[mol/L]**(1-m)/time`` for total forward order ``m`` [-].
        evals : numpy.ndarray
            Activation energies [J/mol].

        Returns
        -------
        phi_1 : numpy.ndarray
            Native ``kvals`` when ``reformulate_kin=False``, or logarithmic
            intercept ``ln(kvals) - evals / (R * temp_ref)`` [-] when
            ``reformulate_kin=True``.
        phi_2 : numpy.ndarray
            Native ``evals`` [J/mol] when ``reformulate_kin=False``, or
            logarithmic slope ``ln(evals / R)`` [-] when
            ``reformulate_kin=True``.
        """
        if self.reformulate_kin:
            ea_term = evals / gas_ct / self.temp_ref  # [-]

            phi_1 = np.log(kvals) - ea_term  # [-]
            phi_2 = np.log(evals / gas_ct)  # [-]

        else:
            phi_1 = kvals  # [k units]
            phi_2 = evals  # [J/mol]

        return phi_1, phi_2

    def set_params(self, params: Union[dict, np.ndarray]) -> None:
        """Set kinetic parameters while preserving reversible mass action.

        Parameters
        ----------
        params : dict or numpy.ndarray
            Dictionary with ``k_params`` in ``[mol/L]**(1-m)/time`` for total
            order m [-], ``ea_params`` [J/mol], optional ``params_f``
            (forward orders [-] as sequence, 2-D array, or reaction-ID/species
            mapping, or custom model parameters), and optional ``params_r``
            (reverse orders [-] for elementary reversible kinetics, defaulting
            to ``stoich_prod`` when ``keq_params`` is not ``None`` and zeros
            otherwise). Alternatively, a flat vector in ``concat_params``
            order: all phi_1, all phi_2, then any fitted concentration
            parameters. Without reformulation phi_1 and phi_2 are k and Ea;
            reformulated logarithmic values are numerical parameters [-] in
            the class's stated unit convention.

        Raises
        ------
        RuntimeError
            If a dictionary omits parameters required by a custom model.
        ValueError
            If reaction or species keys are mismatched, if reversible forward
            or reverse orders differ from raw reactant/product stoichiometry,
            if non-zero ``params_r`` is supplied without ``keq_params``, or if
            ``params_r`` is supplied for a custom kinetic model. Such an
            update leaves the existing parameters intact.

        Notes
        -----
        Equilibrium constants are fixed at construction, outside the fitted
        parameter vector. The reversible rate law always uses elementary
        concentration powers. Orders within absolute tolerance 1e-12 [-]
        (zero relative tolerance) are canonicalized to raw reactant
        stoichiometry for exact mass action. This tolerance is well above
        double-precision roundoff for order magnitudes O(1). Reversible orders
        are never fitted: even explicit orders are excluded from
        ``concat_params``, parameter names, flat updates, and Jacobians.
        """
        if isinstance(params, dict):
            raw_k = params.get('k_params', None)
            if raw_k is None:
                raise ValueError("k_params is required to initialize RxnKinetics.")
            raw_ea = params.get('ea_params', None)
            if raw_ea is None:
                raw_ea = 0.0  # [J/mol], isothermal default
            k_aligned = self._align_rxn_param(
                raw_k, 'k_params', allow_scalar=True, allow_shared_single=True
            )  # [k units]
            ea_aligned = self._align_rxn_param(
                raw_ea, 'ea_params', allow_scalar=True, allow_shared_single=True
            )  # [J/mol]

            # Broadcast a scalar (0-D) Arrhenius parameter when paired with per-reaction values.
            if (
                len(k_aligned) == self.num_rxns
                and len(ea_aligned) == 1
                and not isinstance(raw_ea, Mapping)
                and np.ndim(raw_ea) == 0
            ):
                ea_aligned = np.full(
                    self.num_rxns, float(ea_aligned[0]), dtype=np.float64
                )  # [J/mol]
            elif (
                len(ea_aligned) == self.num_rxns
                and len(k_aligned) == 1
                and not isinstance(raw_k, Mapping)
                and np.ndim(raw_k) == 0
            ):
                k_aligned = np.full(
                    self.num_rxns, float(k_aligned[0]), dtype=np.float64
                )  # [k units]

            if len(k_aligned) != len(ea_aligned):
                raise ValueError(
                    f"k_params length ({len(k_aligned)}) and ea_params length "
                    f"({len(ea_aligned)}) must match."
                )

            k_params = k_aligned + eps  # [k units]
            ea_params = ea_aligned + eps  # [J/mol]
            phi_1, phi_2 = self.transform_params(
                k_params, ea_params)  # [k units, J/mol] or [-]

            fit_override = params.get('fit_paramsf', self._fit_paramsf_override)
            fit_paramsf = True if fit_override is None else bool(fit_override)
            if self.elem_flag:
                params_f = params.get('params_f', None)  # [-], forward orders
                if params_f is None:
                    orders = self.stoich_react.copy()  # [-]
                    order_map = self.stoich_react > 0
                    fit_paramsf = False
                else:
                    orders, order_map = self._coerce_elementary_orders(params_f)  # [-]

                if orders.ndim == 1:
                    orders = orders[np.newaxis, ...]
                params_f = orders
                raw_params_r = params.get('params_r', None)  # [-], reverse orders
                if raw_params_r is None:
                    params_r = (
                        self.stoich_prod.copy()
                        if self.keq_params is not None
                        else np.zeros_like(self.stoich_prod, dtype=np.float64)
                    )  # [-]
                else:
                    params_r, _ = self._coerce_elementary_orders(
                        raw_params_r,
                        stoich_side=self.stoich_prod,
                        param_name='params_r',
                    )  # [-]
            else:
                raw_params_r = params.get('params_r', None)
                if raw_params_r is not None:
                    raise ValueError(
                        "params_r is only supported for elementary reversible "
                        "kinetics (with keq_params), not custom kinetic models."
                    )
                params_f = params.get('params_f', None)  # [custom model units]
                if params_f is None:
                    raise RuntimeError("For user-defined kinetic function, "
                                       "argument 'params_f' is mandatory.")
                params_f = np.asarray(params_f)
                order_map = self.stoich_react > 0
                params_r = np.zeros_like(self.stoich_prod, dtype=np.float64)  # [-]
        else:
            phi_1, phi_2 = np.split(params[:self.num_paramsk], 2)  # [k units, J/mol] or [-]
            fit_paramsf = self.fit_paramsf
            params_f = self.params_f  # [-] or [custom model units]
            params_r = getattr(
                self,
                'params_r',
                self.stoich_prod.copy()
                if self.keq_params is not None
                else np.zeros_like(self.stoich_prod, dtype=np.float64),
            )  # [-]
            order_map = self.order_map
            if self.elem_flag:
                if fit_paramsf:
                    params_f = np.zeros_like(self.stoich_matrix,
                                             dtype=np.float64)  # [-]
                    params_f[self.order_map] = params[self.num_paramsk:]
            else:
                params_f = np.asarray(params[self.num_paramsk:])
                params_f = params_f.reshape(self.params_f_shape)

        # Accept arithmetic roundoff well below meaningful O(1) order changes.
        order_atol = 1e-12  # [-], well above double roundoff for O(1) orders
        if self.elem_flag and self.keq_params is not None:
            reactant_orders = self.stoich_react.copy()  # [-]
            if not np.allclose(params_f, reactant_orders, rtol=0, atol=order_atol):
                raise ValueError(
                    "For thermodynamic consistency with keq_params, params_f "
                    "forward orders must equal the raw reactant stoichiometric "
                    "coefficients (elementary mass action).")
            product_orders = self.stoich_prod.copy()  # [-]
            if not np.allclose(params_r, product_orders, rtol=0, atol=order_atol):
                raise ValueError(
                    "For thermodynamic consistency with keq_params, params_r "
                    "reverse orders must equal the raw product stoichiometric "
                    "coefficients (elementary mass action).")
            params_f = reactant_orders  # [-], exact mass-action exponents
            params_r = product_orders  # [-], exact mass-action exponents
            fit_paramsf = False
        elif self.elem_flag and self.keq_params is None:
            if not np.allclose(params_r, 0.0, rtol=0, atol=order_atol):
                raise ValueError(
                    "params_r requires keq_params (reversible kinetics); "
                    "non-zero reverse reaction orders are not supported for "
                    "irreversible reactions.")

        self.phi_1, self.phi_2 = phi_1, phi_2  # [k units, J/mol] or [-]
        if not self.reformulate_kin:
            self.k_params = self.phi_1  # [k units]
            self.ea_params = self.phi_2  # [J/mol]
        else:
            self.ea_params = gas_ct * np.exp(self.phi_2)  # [J/mol]
            self.k_params = np.exp(
                self.phi_1 + self.ea_params / (gas_ct * self.temp_ref)
            )  # [k units]
        self.params_f = params_f  # [-] for orders; custom model units otherwise
        self.params_r = params_r  # [-], reverse reaction orders
        if isinstance(params, dict):
            self.num_paramsk = len(self.phi_1) + len(self.phi_2)
            self.fit_paramsf = fit_paramsf
            self.order_map = order_map
            if not self.elem_flag:
                self.params_f_shape = params_f.shape

    def nomenclature(self, stoich_matrix, kvals):

        # Names
        num_kpar = len(self.phi_1)

        if self.reformulate_kin:
            name_k = ['\\varphi_{1, %i}' % ind for ind in range(1, num_kpar + 1)]
            name_e = ['\\varphi_{2, %i}' % ind for ind in range(1, num_kpar + 1)]
        else:
            name_k = ['k_%i' % ind for ind in range(1, num_kpar + 1)]
            name_e = ['E_{a, %i}' % ind for ind in range(1, num_kpar + 1)]

        if self.fit_paramsf:
            if self.elem_flag:
                num_orders = int(self.order_map.sum())
                name_orders = [r'\alpha_{}'.format(ind)
                               for ind in range(1, num_orders + 1)]
            else:
                num_orders = np.asarray(self.params_f).size
                name_orders = [r'\theta_{f,%i}' % ind
                               for ind in range(1, num_orders + 1)]
        else:
            name_orders = []

        self.name_params = name_k + name_e + name_orders
        self.num_params = len(self.name_params)
        self.params = dict(zip(self.name_params, (self.phi_1, self.phi_2)))

    def concat_params(self):

        params_k_conc = np.concatenate((self.phi_1, self.phi_2))
        if self.elem_flag:
            if self.fit_paramsf:  # rxn orders not fixed
                orders = self.params_f[self.order_map]
                params_concat = np.concatenate((params_k_conc, orders))

            else:  # rxn orders are fixed
                params_concat = params_k_conc

        else:
            params_concat = np.concatenate(
                (params_k_conc, np.asarray(self.params_f).ravel()))

        return params_concat

    def temp_term(self, temp):

        temp = np.asarray(temp)
        inv_temp = (1/self.temp_ref - 1/temp)

        if self.reformulate_kin:

            if temp.ndim == 0:
                k_temp = np.exp(self.phi_1 + np.exp(self.phi_2) * inv_temp)
            else:
                k_temp = np.exp(self.phi_1 +
                                np.outer(inv_temp, np.exp(self.phi_2)))

        else:
            if temp.ndim == 0:
                k_temp = self.phi_1 * np.exp(self.phi_2/gas_ct * inv_temp)
            else:
                k_temp = self.phi_1 * \
                    np.exp(np.outer(inv_temp, self.phi_2/gas_ct))

        return k_temp

    def equil_term(self, temp, deltah_temp):
        temp = np.asarray(temp)
        deltah_temp = np.asarray(deltah_temp)
        inv_temp = (1/temp - 1/self.tref_hrxn)

        if temp.ndim == 0:
            k_eq = self.keq_params * np.exp(-deltah_temp/gas_ct * inv_temp)
        else:
            if deltah_temp.ndim <= 1:
                exponent = np.outer(inv_temp, deltah_temp/gas_ct)
            else:
                exponent = inv_temp[:, np.newaxis] * deltah_temp/gas_ct

            k_eq = self.keq_params * np.exp(-exponent)

        return k_eq

    def dk_dkparams(self, temp: Union[float, np.ndarray]) -> np.ndarray:
        """Differentiate temperature-dependent rate constants.

        Parameters
        ----------
        temp : float or numpy.ndarray
            Temperature [K], scalar or shape ``(n_times,)``.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_pairs, 2*n_pairs)`` for scalar temperature, or
            ``(n_times, n_pairs, 2*n_pairs)`` for a vector, where n_pairs is
            the number of stored Arrhenius parameter pairs. It is one for a
            shared pair or n_rxns for independent reaction parameters. Columns
            contain all phi_1 derivatives followed by all phi_2 derivatives,
            in ``concat_params`` order. Units are rate-constant units divided by
            parameter units: k units and [J/mol] without reformulation, or
            numerical logarithmic parameters [-] with reformulation.
        """
        if np.ndim(temp) == 1:
            temp = np.asarray(temp)  # [K]
        temp_term = self.temp_term(temp)  # [k units]
        inv_temp = (1/self.temp_ref - 1/temp)  # [1/K]

        if np.ndim(temp) == 1:
            inv_temp = inv_temp[:, np.newaxis]  # [1/K], time then reaction
            if self.reformulate_kin:
                first = temp_term  # [k units]
                second = temp_term * inv_temp * np.exp(self.phi_2)  # [k units]
            else:
                first = np.exp(self.phi_2/gas_ct * inv_temp)  # [-]
                second = temp_term/gas_ct * inv_temp  # [k units mol/J]
            identity = np.eye(len(self.phi_1))  # [-], independent stored constants
            return np.concatenate((first[..., :, None] * identity,
                                   second[..., :, None] * identity), axis=-1)

        if self.reformulate_kin:
            drate_dphi1 = np.diag(temp_term)
            dphi_2 = temp_term * inv_temp * np.exp(self.phi_2)
        else:
            drate_dphi1 = np.diag(np.exp(self.phi_2/gas_ct * inv_temp))
            dphi_2 = temp_term/gas_ct * inv_temp

        drate_dphi2 = np.diag(dphi_2)
        drate_dk = np.hstack((drate_dphi1, drate_dphi2))

        return np.atleast_2d(drate_dk)

    def elem_f_model(self, conc, rxn_orders):
        """ Compute elementary reaction rates for each participating reaction

        Parameters
        ----------
        conc : array-like
            molar concentrations for each participating species (size n_comp)

        Returns
        -------
        rxn_rates : array
            rate for each reaction taking place in the system (size n_rxns)
        rates_species : array
            rate for each species in each reaction (size n_rxns x n_comp)
        total_rates : array
            net reaction rate for each component among all the reactions it
            participates in (size n_comp)
        """

        conc = np.maximum(eps, conc)
        f_term = np.exp(np.dot(np.log(conc), rxn_orders.T))
        return f_term

    def equilibrium_model(self, conc, temp, deltah_rxn) -> np.ndarray:
        """Compute reversible concentration terms for each reaction.

        Parameters
        ----------
        conc : array-like
            Participating species molar concentrations [mol/L]. Accepts shape
            ``(n_species,)`` or ``(n_times, n_species)``.
        temp : float or array-like
            Temperature [K]. Array inputs are interpreted along the same time
            axis as 2-D ``conc``.
        deltah_rxn : array-like or None
            Heat of reaction at ``temp`` [J/mol_rxn] on the raw reaction basis.
            None uses ``self.delta_hrxn``; any explicit value, including zero,
            takes precedence. Shape (n_rxns,) or (n_times, n_rxns).

        Returns
        -------
        overall_rate : ndarray
            Reversible concentration term, ``forward - reverse``. Multiplying
            by ``temp_term(temp)`` gives per-reaction rates whose time basis is
            set by ``k_params``. Units for reaction i are ``[mol/L]**m_i``,
            where m_i is the total forward order. Shape (n_rxns,) for a scalar
            state or (n_times, n_rxns) for a concentration batch; vector
            temperatures must be paired with 2-D concentrations.
        """
        is_product = self.stoich_prod > 0
        orders = abs(is_product * self.stoich_prod)  # [-]
        conc = np.asarray(conc)
        n_conc = len(conc)

        if deltah_rxn is None:
            deltah_rxn = self.delta_hrxn  # [J/mol_rxn], raw reaction basis
        keq_temp = self.equil_term(temp, deltah_rxn)

        keq_temp[keq_temp == 0] = ZERO_KEQ_REPLACEMENT

        # Forward term
        f_term = self.elem_f_model(conc, self.params_f)

        # Backward term
        if conc.ndim == 1:
            r_term = np.zeros(self.num_rxns)
            for ind in range(self.num_rxns):
                r_term[ind] = np.prod(conc**(orders[ind])) / keq_temp[ind]
        else:
            r_term = np.zeros((n_conc, self.num_rxns))
            for ind in range(self.num_rxns):
                if keq_temp.ndim == 1:
                    keq_rxn = keq_temp[ind]
                else:
                    keq_rxn = keq_temp[:, ind]

                r_term[:, ind] = np.prod(
                    conc**(orders[ind]), axis=1) / keq_rxn

        overall_rate = f_term - r_term

        return overall_rate

    def elem_df_dstates(self, conc):
        """Differentiate elementary concentration terms with respect to states.

        Parameters
        ----------
        conc : array-like
            Participating species molar concentrations [mol/L]. Accepts shape
            ``(n_species,)`` or ``(n_times, n_species)``.

        Returns
        -------
        df_dconc : ndarray
            Derivative of the elementary concentration term for each reaction
            with respect to each concentration. Units are concentration-term
            units divided by [mol/L].

        Notes
        -----
        Concentrations in derivative denominators are floored at
        ``eps = np.finfo(float).eps``. This matches the concentration floor
        used by ``elem_f_model`` before evaluating logarithms/powers and avoids
        division by zero at depleted species concentrations.
        """

        conc = np.asarray(conc)
        conc_safe = np.maximum(eps, conc)
        f_term = self.elem_f_model(conc, self.params_f)

        if conc.ndim == 1:
            conc_term = conc_safe
        else:
            conc_term = conc_safe[:, np.newaxis, :]

        df_dconc = f_term[..., np.newaxis] * self.params_f / conc_term

        return df_dconc

    def _reverse_df_dstates(self, conc, temp, deltah_rxn=None):
        """Differentiate reversible reverse concentration terms.

        Parameters
        ----------
        conc : array-like
            Participating species molar concentrations [mol/L]. Accepts shape
            ``(n_species,)`` or ``(n_times, n_species)``.
        temp : float or array-like
            Temperature [K].
        deltah_rxn : array-like, optional
            Heat of reaction at ``temp`` [J/mol_rxn]. If omitted, the reference
            heat of reaction stored on the kinetics object is used.

        Returns
        -------
        dr_dconc : ndarray
            Derivative of the reverse concentration term with respect to each
            concentration. Units are concentration-term units divided by
            [mol/L].

        Notes
        -----
        The same ``eps`` concentration floor used by ``elem_df_dstates`` is
        applied to the reverse product term and derivative denominator. This
        keeps depleted or slightly negative numerical states from producing
        singular concentration derivatives.
        """
        is_product = self.stoich_prod > 0
        orders = abs(is_product * self.stoich_prod)  # [-]
        conc = np.asarray(conc)
        conc_safe = np.maximum(eps, conc)

        if deltah_rxn is None:
            deltah_rxn = self.delta_hrxn

        keq_temp = self.equil_term(temp, deltah_rxn)
        keq_temp[keq_temp == 0] = ZERO_KEQ_REPLACEMENT

        if conc.ndim == 1:
            r_term = np.zeros(self.num_rxns)
            for ind in range(self.num_rxns):
                r_term[ind] = np.prod(
                    conc_safe**orders[ind]) / keq_temp[ind]

            dr_dconc = r_term[:, np.newaxis] * orders / conc_safe
        else:
            n_conc = len(conc)
            r_term = np.zeros((n_conc, self.num_rxns))
            for ind in range(self.num_rxns):
                if keq_temp.ndim == 1:
                    keq_rxn = keq_temp[ind]
                else:
                    keq_rxn = keq_temp[:, ind]

                r_term[:, ind] = np.prod(
                    conc_safe**orders[ind], axis=1) / keq_rxn

            dr_dconc = (
                r_term[..., np.newaxis] * orders /
                conc_safe[:, np.newaxis, :])

        return dr_dconc

    def elem_df_dtheta(self, conc):

        f_term = self.elem_f_model(conc, self.params_f)

        conc_correc = np.maximum(np.ones_like(conc) * eps, conc)

        num_orders = self.order_map.sum()
        drate_dorder = np.zeros((self.num_rxns, num_orders))

        count = 0

        for ind, row in enumerate(self.order_map):
            conc_m = conc_correc[row]  # see Section 3.2.2

            norder_i = sum(row)
            drate_dorder[ind,
                         count:count + norder_i] = np.log(conc_m) * f_term[ind]

            count += norder_i
        drate_dorder = drate_dorder

        return drate_dorder

    def derivatives(self, conc, temp, dstates: bool = True,
                    delta_hrxn=None) -> np.ndarray:
        """Calculate reaction-rate Jacobians.

        Parameters
        ----------
        conc : array-like
            Participating species molar concentrations [mol/L]. Accepts shape
            ``(n_species,)`` or ``(n_times, n_species)``.
        temp : float or array-like
            Temperature [K].
        dstates : bool, optional
            If True, return derivatives with respect to concentrations. If
            False, return derivatives with respect to kinetic parameters.
            The default is True.
        delta_hrxn : array-like, optional
            Runtime heat of reaction [J/mol of reaction as written] used to
            evaluate reversible equilibrium constants. Values use the raw
            ``stoich_matrix`` row basis. If omitted, ``self.delta_hrxn`` is
            used.

        Returns
        -------
        jac_states : ndarray
            Species-rate Jacobian with respect to concentrations. Units are
            species-rate units divided by [mol/L].
        jac_params : ndarray
            Species-rate Jacobian with respect to kinetic parameters. Returned
            when ``dstates`` is False. Shape ``(n_species, n_params)`` for
            scalar temperature and 1-D concentrations, or
            ``(n_times, n_species, n_params)`` for a concentration batch.
            A temperature vector (n_times,) pairs with concentration rows;
            a scalar temperature applies to every row. Columns follow
            ``concat_params``: all phi_1, all phi_2, then any fitted forward
            parameters. Units are normalized species rates [mol/L/time]
            divided by parameter units; time is set by ``k_params``.
            Equilibrium constants and reaction heats are held fixed.
        """
        temp_terms = self.temp_term(temp)  # [k units]

        if dstates:  # --------------- wrt states
            df_dstates = self.df_dstates(conc, *self.args_kin)
            if self.keq_params is not None:
                df_dstates = df_dstates - self._reverse_df_dstates(
                    conc, temp, delta_hrxn)

            dr_dstates = df_dstates * temp_terms[..., np.newaxis]

            if dr_dstates.ndim == 2:
                jac_states = np.dot(self.normalized_stoich, dr_dstates)
            else:
                jac_states = np.einsum(
                    'sr,trc->tsc', self.normalized_stoich, dr_dstates)

            return jac_states
        else:  # --------------- wrt parameters
            if self.keq_params is None:
                f_terms = self.kinetic_model(
                    conc, self.params_f, *self.args_kin)  # [concentration**order]
            else:
                f_terms = self.equilibrium_model(
                    conc, temp, delta_hrxn)  # [concentration**order]
            dk_dphi = self.dk_dkparams(temp)  # [k units/parameter units]
            if dk_dphi.ndim == 2 and np.ndim(f_terms) == 1:
                dr_dthetak = (dk_dphi.T * f_terms).T  # [mol/L/time/parameter units]
            else:
                dr_dthetak = dk_dphi * np.asarray(f_terms)[..., np.newaxis]

            if self.fit_paramsf:
                if np.ndim(conc) == 1:
                    dr_dthetaf = self.df_dthetaf(
                        conc, *self.args_kin)  # [concentration-term units/parameter units]
                else:
                    # Keep the scalar callback contract for custom models.
                    dr_dthetaf = np.stack([
                        self.df_dthetaf(row, *self.args_kin) for row in conc
                    ])  # [concentration-term units/parameter units]
                if dr_dthetak.ndim == 2:
                    dr_dthetaf = (dr_dthetaf.T * temp_terms).T
                    dr_dparams = np.hstack(
                        (dr_dthetak, dr_dthetaf))  # [mol/L/time/parameter units]
                else:
                    dr_dthetaf = dr_dthetaf * temp_terms[..., np.newaxis]
                    dr_dparams = np.concatenate((dr_dthetak, dr_dthetaf), axis=-1)
            else:
                dr_dparams = dr_dthetak

            if dr_dparams.ndim == 2:
                jac_params = np.dot(
                    self.normalized_stoich, dr_dparams)  # [mol/L/time/parameter units]
            else:
                jac_params = np.matmul(self.normalized_stoich, dr_dparams)

            if jac_params.ndim == 1:
                jac_params = jac_params[..., np.newaxis]

            return jac_params

    def get_rxn_rates(self, conc, temp=298.15, overall_rates: bool = True,
                      jac: bool = False, delta_hrxn=None) -> np.ndarray:
        """Evaluate reaction rates or their concentration Jacobian.

        Parameters
        ----------
        conc : array-like
            Participating species molar concentrations [mol/L]. Accepts shape
            ``(n_species,)`` or ``(n_times, n_species)``.
        temp : float or array-like, optional
            Temperature [K]. The default is 298.15.
        overall_rates : bool, optional
            If True, return species rates. If False, return per-reaction rates.
            The default is True.
        jac : bool, optional
            If True, return the concentration Jacobian instead of rates. The
            default is False.
        delta_hrxn : array-like, optional
            Runtime heat of reaction [J/mol of reaction as written] for
            reversible rate or Jacobian evaluations. Values use the raw
            ``stoich_matrix`` row basis. None uses ``self.delta_hrxn``;
            explicit values, including zero, take precedence.

        Returns
        -------
        total_rates : ndarray
            Species rates [mol/L/time] when ``overall_rates`` is True and
            ``jac`` is False. The time unit is set by ``k_params``.
        rxn_rates : ndarray
            Per-reaction rates [mol/L/time] when ``overall_rates`` is False and
            ``jac`` is False. The time unit is set by ``k_params``.
        jac_states : ndarray
            Species-rate Jacobian with respect to concentrations when ``jac``
            is True. Units are species-rate units divided by [mol/L].
        """

        if jac:
            jac_states = self.derivatives(conc, temp, delta_hrxn=delta_hrxn)
            return jac_states

        else:
            temp_terms = self.temp_term(temp)

            if self.keq_params is None:
                f_terms = self.kinetic_model(conc, self.params_f,
                                             *self.args_kin)
            else:
                f_terms = self.equilibrium_model(conc, temp, delta_hrxn)

            rxn_rates = temp_terms * f_terms
            if overall_rates:  # per species
                total_rates = np.dot(rxn_rates, self.normalized_stoich.T)
                return total_rates
            else:  # per rxn
                return rxn_rates


class CrystKinetics:
    """Model signed crystallization rates with power-law driving forces.

    ``relative`` and ``ratio`` both use ``(c - c_sat) / c_sat`` [-].
    Thus ``ratio`` means the excess ratio ``S - 1``, where ``S = c/c_sat``,
    rather than ``S`` itself. ``absolute`` uses ``c - c_sat`` [kg/m**3].
    Positive driving force enables nucleation and growth; negative driving
    force enables dissolution, whose rate is negative. At saturation the
    built-in rates are zero. For relative/ratio kinetics, results are undefined
    for non-positive solubility (not validated in the rate path). The prefactors
    must match the selected concentration basis.

    Parameters
    ----------
    coeff_solub : array-like, optional
        Polynomial coefficients in ascending powers of temperature,
        coefficient j in [kg/m**3/K**j]; for Apelblat, coefficients
        A [-], B [K], C [-] give exp(A + B/T + C*log(T)) [kg/m**3]
        using numerical temperature in kelvin.
    solub_fn : callable, optional
        ``solub_fn(temp, conc)`` returns solubility [kg/m**3], with
        temperature [K] and species concentrations [kg/m**3]. Overrides
        the built-in solubility correlation.
    nucl_prim, growth, dissolution : array-like of length 3, optional
        Physical parameters [k, E, n], with E [J/mol] and n [-].
        k has rate units divided by driving-force units to power n.
        Nucleation rate units are [#/m**3/s]; growth and dissolution
        rate units are [um/s].
    nucl_sec : array-like of length 4, optional
        Physical parameters [k, E, n, s_2], with E [J/mol], n and s_2
        [-]. k additionally divides by the selected moment basis to
        power s_2. Omitted secondary nucleation is inactive.
    solubility_type : {'polynomial', 'apelblat'}, optional
        Built-in solubility correlation; default 'polynomial'.
    sup_sat_type : {'relative', 'ratio', 'absolute'}, optional
        'relative' and 'ratio' use (c - c_sat)/c_sat [-]; 'absolute'
        uses c - c_sat [kg/m**3]. Default 'relative'.
    reformulate_kin : bool, optional
        Transform supplied physical k and E to phi_1 = log(k) - E/R/Tref
        and phi_2 = log(E/R), using numerical values in the specified
        units. Default False.
    alpha_fn : callable, optional
        Composition-dependent growth inhibition factor [-], default unity.
    temp_ref : float, optional
        Reference temperature [K], default 298.15 K (25 degrees Celsius).
    custom_mechanisms : dict of callables, optional
        Mechanism overrides receiving signed driving force, solubility
        [kg/m**3], moments, temperature [K], reference temperature [K],
        and parameters. See ``get_kinetics`` for the moment basis.
    mu_sec_nucl : {'area', 'volume'}, optional
        Select moment order 2 or 3 for (kv*moment)**s_2; default 'volume'.

    Raises
    ------
    ValueError
        If sup_sat_type is not 'relative', 'ratio', or 'absolute'.
    PharmaPyTypeError
        If custom_mechanisms is not a dictionary.

    Warns
    -----
    FutureWarning
        If sup_sat_type='ratio', which is deprecated and now means S - 1 [-],
        identical to 'relative'. Prefactors fitted to the old S law must be
        refitted.
    """

    def __init__(self, coeff_solub=None, solub_fn=None,
                 nucl_prim=None, nucl_sec=None, growth=None, dissolution=None,
                 solubility_type='polynomial', sup_sat_type: str = 'relative',
                 reformulate_kin=False, alpha_fn=None,
                 temp_ref=298.15, custom_mechanisms=None,
                 mu_sec_nucl='volume') -> None:
        """Initialize kinetics; see the class docstring for parameters and errors."""
        if sup_sat_type not in ('relative', 'ratio', 'absolute'):
            raise ValueError("sup_sat_type must be 'relative', 'ratio', or "
                             f"'absolute'; got {sup_sat_type!r}.")
        if sup_sat_type == 'ratio':
            warnings.warn(
                "sup_sat_type='ratio' is deprecated: it now means S - 1 "
                "(identical to 'relative'). Prefactors fitted to the old S "
                "law must be refitted.",
                FutureWarning, stacklevel=2)

        self.target_idx = None

        self.temp_ref = temp_ref  # [K]
        self.sup_sat_type = sup_sat_type
        self.reformulate_kin = reformulate_kin

        if solub_fn is None:
            self.get_solubility = self.solubility_temp
        else:
            self.get_solubility = solub_fn

        mu_sec = {'area': 2, 'volume': 3}
        self.mu_sec_nucl = mu_sec[mu_sec_nucl]

        if custom_mechanisms is None:
            custom_mechanisms = {}
        elif not isinstance(custom_mechanisms, dict):
            raise PharmaPyTypeError('Provide a dictionary of callables for kinetics.')

        self.custom_mechanisms = custom_mechanisms

        # ---------- Parameters
        self.names_mechanisms = ('nucl_prim', 'nucl_sec', 'growth',
                                 'dissolution')

        params_all = [nucl_prim, nucl_sec, growth, dissolution]

        params_keys = [name for param, name
                       in zip(params_all, self.names_mechanisms)
                       if param is not None]

        params_list = [item for item in params_all if item is not None]

        self.num_par_tuple = [len(item) for item in params_list]

        param_dict = {key: value
                      for key, value in zip(params_keys, params_list)}

        self.set_params(param_dict)  # create self.params

        self.coeff_solub = np.asarray(coeff_solub)
        self.solub_type = solubility_type

        if reformulate_kin:
            self.name_params = ('\log(k_{bp})', '\log(E_{bp}/R)', 'b',
                                '\log(k_{bs})', '\log(E_{bs}/R)', 's_1', 's_2',
                                '\log(k_{g})', '\log(E_{g}/R)', 'g',
                                '\log(k_{d})', '\log(E_{d}/R)', 'd')
        else:
            self.name_params = ('k_{bp}', 'E_{bp}', 'b',
                                'k_{bs}', 'E_{bs}', 's_1', 's_2',
                                'k_{g}', 'E_{g}', 'g',
                                'k_{d}', 'E_{d}', 'd')

        self.num_params = len(self.name_params)

        # ---------- Growth decreasing fn
        if alpha_fn is None:
            self.alpha_fn = lambda conc: 1
        else:
            self.alpha_fn = alpha_fn

    def set_params(self, params_in):
        if isinstance(params_in, dict):
            params_complete = self.transform_params(params_in,
                                                    self.reformulate_kin)
        else:
            param_lens = np.array([3, 4, 3, 3])
            split_idx = param_lens.cumsum()[:-1]

            params_in = np.split(params_in, split_idx)

            params_complete = dict(zip(self.names_mechanisms, params_in))

        self.params = params_complete

    def transform_params(self, param_dict, reparam):
        self.params_sec = param_dict.get('nucl_sec')
        if reparam:
            zero_log = np.log(eps)

            nucl_prim = [zero_log, 0, 0]
            nucl_sec = [zero_log, 0, 0, 0]
            growth = [zero_log, 0, 0]
            dissol = [zero_log, 0, 0]

            params_parsed = {'nucl_prim': nucl_prim, 'nucl_sec': nucl_sec,
                             'growth': growth, 'dissolution': dissol}

            for name in self.names_mechanisms:
                if name in param_dict.keys():
                    vals = param_dict[name]

                    tref = self.temp_ref
                    phi_1 = np.log(vals[0] + eps) - vals[1]/gas_ct/tref
                    phi_2 = np.log((vals[1] + eps)/gas_ct)

                    params_parsed[name] = list(vals)

                    params_parsed[name][0] = phi_1
                    params_parsed[name][1] = phi_2
        else:
            nucl_prim = [0, 0, 0]
            nucl_sec = [0, 0, 0, 0]
            growth = [0, 0, 0]
            dissol = [0, 0, 0]

            params_parsed = {'nucl_prim': nucl_prim, 'nucl_sec': nucl_sec,
                             'growth': growth, 'dissolution': dissol}

            for name in self.names_mechanisms:
                if name in param_dict.keys():
                    vals = param_dict[name]
                    tref = self.temp_ref
                    phi_1 = vals[0]
                    phi_2 = vals[1]

                    params_parsed[name] = list(vals)

                    params_parsed[name][0] = phi_1
                    params_parsed[name][1] = phi_2

        return params_parsed

    def concat_params(self):
        params = [np.array(vals) for vals in self.params.values()]
        params_conc = np.concatenate(params)
        return params_conc

    def solubility_temp(self, temp, conc=None):
        if self.solub_type == 'polynomial':
            int_coeff = np.arange(len(self.coeff_solub))

            temp = np.asarray(temp)
            if temp.ndim == 0:
                c_satur = (temp**int_coeff * self.coeff_solub).sum()
            else:
                temp = temp[..., np.newaxis]
                c_satur = (temp**int_coeff * self.coeff_solub).sum(axis=1)
                
        elif self.solub_type == 'apelblat':
            a1, a2, a3 = self.coeff_solub
            c_satur = np.exp(a1 + a2/temp + a3*np.log(temp))
        else:
            raise NameError("Bad 'solub_type' name. It must be either "
                            "'polynomial' or 'apelblat")

        return c_satur

    def _driving_force(self, conc_target: "float | np.ndarray",
                       conc_sat: "float | np.ndarray") -> "float | np.ndarray":
        """Return the signed driving force shared by rates and sensitivities.

        Parameters
        ----------
        conc_target, conc_sat : float or ndarray
            Target and saturation mass concentrations [kg/m**3], with
            broadcast-compatible shapes. For relative and ratio kinetics,
            results are undefined for non-positive solubility (not validated
            in the rate path).

        Returns
        -------
        float or ndarray
            c - c_sat [kg/m**3] for absolute kinetics, otherwise
            (c - c_sat)/c_sat [-], preserving the broadcast shape.
        """
        concentration_difference = conc_target - conc_sat  # [kg/m**3]
        if self.sup_sat_type == 'absolute':
            return concentration_difference
        return concentration_difference / conc_sat

    def get_kinetics(self, conc: "float | np.ndarray",
                     temp: "float | np.ndarray", kv_cry: float,
                     moments: "np.ndarray | None" = None,
                     nucl_sec_out: bool = False) -> tuple:
        """Evaluate crystallization kinetics for target concentration states.

        Parameters
        ----------
        conc : float or ndarray
            Mass concentrations [kg/m**3]. A scalar is the selected target;
            an array has species on the last axis, selected by target_idx.
        temp : float or ndarray
            Temperature [K], scalar for one state or shape (N,) for N states.
        kv_cry : float
            Crystal volume shape factor [-].
        moments : ndarray, optional
            Moments with order on the last axis, shape (M,) or (N, M).
            No unit conversion is performed here: the secondary prefactor
            must match the supplied moment basis. Crystallizer callers pass
            SI length moments (order j in [m**j] for total moments or
            [m**j/m**3] for volume-normalized moments). Required for active
            built-in secondary nucleation and for vector state evaluation.
        nucl_sec_out : bool, optional
            Return primary and secondary nucleation separately, default False.

        Returns
        -------
        tuple
            (total nucleation, growth, dissolution), or (primary nucleation,
            secondary nucleation, growth, dissolution) when nucl_sec_out is
            True. Nucleation is [#/m**3/s]; growth and dissolution are [um/s].
            Each rate is scalar for one state or shape (N,) for N states.
            Dissolution is negative. Driving-force definitions are given in
            the class docstring and also apply to custom mechanisms.

        Notes
        -----
        Scalar evaluations cache rates and moment inputs for deriv_cryst.
        """
        conc_array = np.asarray(conc)  # [kg/m**3]
        if conc_array.ndim == 0:
            conc_target = conc_array.item()  # [kg/m**3]
        else:
            conc_target = conc_array.T[self.target_idx]  # [kg/m**3]

        conc_sat = self.get_solubility(temp, conc)  # [kg/m**3]
        sup_sat = self._driving_force(conc_target, conc_sat)  # [-] or [kg/m**3]

        def is_default_secondary(name):
            """Return whether secondary nucleation is the inactive default."""
            if name != 'nucl_sec' or name in self.custom_mechanisms:
                return False

            default = np.zeros_like(self.params['nucl_sec'], dtype=float)
            if self.reformulate_kin:
                default[0] = np.log(eps)

            return np.allclose(self.params['nucl_sec'], default)

        par_p, par_s, par_g, par_d = self.params.values()

        if 'nucl_sec' in self.custom_mechanisms:
            args_sec = ()
            par_sec = self.params_sec
        else:
            args_sec = (kv_cry, self.reformulate_kin)
            par_sec = par_s

        if np.ndim(sup_sat) == 0:
            sup_sat = np.asarray(sup_sat).item()
            conc_sat = np.asarray(conc_sat).item()
            if np.ndim(temp) == 0:
                temp = np.asarray(temp).item()

            args = [sup_sat, conc_sat, moments, temp, self.temp_ref]
            if sup_sat >= 0:

                subset_mech = ('nucl_prim', 'nucl_sec', 'growth')
            else:

                subset_mech = ('dissolution', )

            mechs = {}

            for name in subset_mech:
                if is_default_secondary(name):
                    mechs[name] = 0
                elif name in self.custom_mechanisms:
                    args_concat = args + [self.params[name]]
                    mechs[name] = self.custom_mechanisms[name](*args_concat)
                else:
                    mechs[name] = cryst_mechanism(sup_sat, moments, temp,
                                                  self.temp_ref,
                                                  self.params[name],
                                                  self.reformulate_kin,
                                                  kv_cry, self.mu_sec_nucl)

            for ky in self.names_mechanisms:
                if ky not in mechs:
                    mechs[ky] = 0

            # Retain the moment basis for a zero-prefactor partial derivative.
            self._moment_inputs = (moments, kv_cry)  # [input moment units], [-]

            # Returns
            self.prim_nucl = mechs['nucl_prim']
            self.sec_nucl = mechs['nucl_sec']
            self.growth = mechs['growth']  # um/s
            self.dissol = mechs['dissolution']  # um/s

        else:
            # Divide positive and negative supersaturation periods
            positive_map = sup_sat > 0

            sup_positive = sup_sat[positive_map]
            sup_negative = sup_sat[~positive_map]

            temp_positive = temp[positive_map]
            temp_negative = temp[~positive_map]

            conc_sat_positive = conc_sat[positive_map]
            conc_sat_negative = conc_sat[~positive_map]

            moments_positive = moments[positive_map]
            moments_negative = moments[~positive_map]

            subset_mech = ('nucl_prim', 'nucl_sec', 'growth')

            mechs = {}

            for name in self.names_mechanisms:
                mechs[name] = np.zeros_like(sup_sat)

            args = [sup_positive, conc_sat_positive, moments_positive,
                    temp_positive, self.temp_ref]

            for name in subset_mech:
                if is_default_secondary(name):
                    continue
                elif name in self.custom_mechanisms:
                    args_concat = args + [self.params[name]]
                    mechs[name][positive_map] = self.custom_mechanisms[name](*args_concat)
                else:
                    mechs[name][positive_map] = cryst_mechanism(sup_positive,
                                                                moments_positive,
                                                                temp_positive,
                                                                self.temp_ref,
                                                                self.params[name],
                                                                self.reformulate_kin,
                                                                kv_cry, self.mu_sec_nucl)

            subset_mech = ('dissolution', )

            args = [sup_negative, conc_sat_negative, moments_negative,
                    temp_negative, self.temp_ref]

            for name in subset_mech:
                if name in self.custom_mechanisms:
                    args_concat = args + [self.params[name]]
                    mechs[name][~positive_map] = self.custom_mechanisms[name](*args_concat)
                else:
                    mechs[name][~positive_map] = cryst_mechanism(sup_negative,
                                                                 moments_negative,
                                                                temp_negative,
                                                                self.temp_ref,
                                                                self.params[name],
                                                                self.reformulate_kin,
                                                                kv_cry, self.mu_sec_nucl)

        if nucl_sec_out:
            return mechs['nucl_prim'], mechs['nucl_sec'], mechs['growth'], mechs['dissolution']
        else:
            nucl = mechs['nucl_prim'] + mechs['nucl_sec']
            return nucl, mechs['growth'], mechs['dissolution']

    def deriv_cryst(self, conc_tg: float, conc: np.ndarray,
                    temp: float) -> tuple:
        """Return built-in mechanism parameter partials at a scalar state.

        Parameters
        ----------
        conc_tg : float
            Target mass concentration [kg/m**3].
        conc : ndarray
            Species mass concentrations [kg/m**3], shape (num_species,).
        temp : float
            Temperature [K].

        Returns
        -------
        dbp_dpar, dbs_dpar, dgr_dpar, ddiss_dpar : ndarray
            Shape (3,) each, ordered as primary nucleation, secondary
            nucleation, growth, and dissolution. For physical parameters,
            columns are partials with respect to [k, E, n]: units are
            [rate/k], [rate/(J/mol)], and [rate], respectively. For
            reformulated parameters, columns are [phi_1, phi_2, n] with
            units [rate], where phi_1 = log(k) - E/R/Tref and
            phi_2 = log(E/R) use numerical values in the configured units.
            Rate units are [#/m**3/s] for nucleation and [um/s] for growth
            and dissolution. The signed force and its epsilon-floored
            magnitude follow the built-in cryst_mechanism expression.
        conc_sat : float
            Saturation mass concentration [kg/m**3].

        Notes
        -----
        Call get_kinetics for the same scalar state and parameters first;
        these partials use its cached rates and unchanged moment inputs.
        The zero-prefactor rebuild uses the current driving force and
        temperature with moments cached by the last scalar get_kinetics call.
        The cached moments are immaterial for omitted mechanisms, whose s_2
        is zero (or absent).
        Custom mechanism derivatives are not supported. Inactive branches
        return zero partials. The omitted default secondary mechanism with
        no moment input also returns zero partials.

        s_2 is deliberately not returned: Crystallizers.jac_params appends
        that fourth secondary-nucleation column itself, using the configured
        secondary-nucleation moment basis. These partials hold concentration,
        temperature, and moments fixed.
        """
        conc_sat = self.get_solubility(temp, conc)  # [kg/m**3]
        ssat = self._driving_force(conc_tg, conc_sat)  # [-] or [kg/m**3]
        absup = max(eps, abs(ssat))  # [-] or [kg/m**3], as in cryst_mechanism

        def dmech_dparam(mech, params, active):
            """Differentiate one active built-in mechanism.

            Parameters
            ----------
            mech : float
                Cached signed rate [#/m**3/s] or [um/s].
            params : sequence
                [k, E, n, optional s_2] or [phi_1, phi_2, n, optional s_2],
                with units and basis described in deriv_cryst.
            active : bool
                Whether the current driving force enables this mechanism.

            Returns
            -------
            ndarray
                Shape (3,) parameter partials in deriv_cryst column order
                and units; the secondary moment exponent is excluded.
            """
            if not active:
                return np.zeros(3)
            if self.reformulate_kin:
                phi_2 = params[1]  # [-], log of numerical E/R in kelvin
                dmech = np.array(  # [rate] for all three columns
                    [mech,
                     mech * (1 / self.temp_ref - 1 / temp) * np.exp(phi_2),
                     mech * np.log(absup)])
            else:
                prefactor = params[0]  # [rate / force**n / moment**s_2]
                if prefactor != 0:
                    d_prefactor = mech / prefactor  # [rate / prefactor]
                else:
                    moments, kv_cry = self._moment_inputs  # [moment units], [-]
                    if len(params) == 4 and moments is None:
                        return np.zeros(3)
                    unit_params = list(params)  # same units as params
                    unit_params[0] = 1  # [prefactor units], exact linear factor
                    d_prefactor = cryst_mechanism(  # [rate / prefactor]
                        ssat, moments, temp, self.temp_ref, unit_params,
                        False, kv_cry, self.mu_sec_nucl)
                dmech = np.array(  # [rate/k], [rate/(J/mol)], [rate]
                    [d_prefactor,
                     -mech / (gas_ct * temp),
                     mech * np.log(absup)])

            return dmech

        b_par, s_par, g_par, d_par = self.params.values()
        growing = ssat >= 0
        # All four arrays: [rate/parameter], with column units in Returns.
        dbp_dpar = dmech_dparam(self.prim_nucl, b_par, growing)
        dbs_dpar = dmech_dparam(self.sec_nucl, s_par, growing)
        dgr_dpar = dmech_dparam(self.growth, g_par, growing)
        ddiss_dpar = dmech_dparam(self.dissol, d_par, not growing)

        return dbp_dpar, dbs_dpar, dgr_dpar, ddiss_dpar, conc_sat
