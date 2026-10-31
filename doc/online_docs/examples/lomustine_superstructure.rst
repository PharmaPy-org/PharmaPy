======================================================
Lomustine Hybrid Superstructure Flowsheet Simulation
======================================================

Introduction
============

This guide walks through the end-to-end dynamic flowsheet simulation of the
anticancer active pharmaceutical ingredient (API) lomustine across alternative
batch and hybrid batch-continuous equipment topologies using ``PharmaPy``.

The reaction network, thermophysical property set, evaporative solvent switch,
and hybrid superstructure alternatives are drawn from the published lomustine
kinetic, techno-economic, superstructure-synthesis, and discrete optimization
studies (see also :doc:`../publications`):

- **Reaction kinetics and uncertainty analysis**:
  Casas-Orozco et al. (2023a), *Chemical Engineering Science*, 275, 118591
  (`doi:10.1016/j.ces.2023.118591 <https://doi.org/10.1016/j.ces.2023.118591>`_).
- **Dynamic end-to-end campaign manufacturing and techno-economic analysis**:
  Casas-Orozco et al. (2023b), *AIChE Journal*, 69(9), e18142
  (`doi:10.1002/aic.18142 <https://doi.org/10.1002/aic.18142>`_).
- **Hybrid rule-based and optimization-driven superstructure synthesis**:
  Barhate et al. (2025), *AIChE Journal*, 71(9), e18888
  (`doi:10.1002/aic.18888 <https://doi.org/10.1002/aic.18888>`_).
- **Discrete superstructure landscape exploration and Ising/MIP optimization**:
  Park & Bernal Neira (2026), *AIChE Journal*, e70656
  (`doi:10.1002/aic.70656 <https://doi.org/10.1002/aic.70656>`_; simulation
  benchmark files in ``ds-mfg/simulation/`` of ``SECQUOIA/pd_ising``).

Process Chemistry and Superstructure Overview
---------------------------------------------

Lomustine (``lom``) is synthesized in tetrahydrofuran (``THF``) via a two-step
reaction sequence:

1. **Carbamoylation** (``R01``): 1-chloro-2-isocyanatoethane (``ISO``,
   :math:`\text{C}_3\text{H}_4\text{ClNO}`) reacts with cyclohexylamine
   (``CHA``, :math:`\text{C}_6\text{H}_{13}\text{N}`) to form the urea
   intermediate 1-(2-chloroethyl)-3-cyclohexylurea (``interm``,
   :math:`\text{C}_9\text{H}_{17}\text{ClN}_2\text{O}`):

   .. math::

      \text{ISO} + \text{CHA} \xrightarrow{k_1} \text{interm}

2. **Nitrosation and competing side reaction** (``R02``): *tert*-butyl nitrite
   (``TBN``, :math:`\text{C}_4\text{H}_9\text{NO}_2`) nitrosates ``interm`` to
   form lomustine (``lom``, :math:`\text{C}_9\text{H}_{16}\text{ClN}_3\text{O}_2`)
   and *tert*-butanol (``TBA``, :math:`\text{C}_4\text{H}_{10}\text{O}`), while
   unreacted ``ISO`` can react with ``TBA`` to produce the urethane byproduct
   *tert*-butyl (2-chloroethyl)carbamate (``SUB1``,
   :math:`\text{C}_7\text{H}_{14}\text{ClNO}_2`):

   .. math::

      \text{interm} + \text{TBN} \xrightarrow{k_2} \text{lom} + \text{TBA}

   .. math::

      \text{ISO} + \text{TBA} \xrightarrow{k_3} \text{SUB1}

3. **Evaporative solvent switch** (``VAP01``): Because lomustine is highly
   soluble in ``THF``, *n*-heptane (``C7``, :math:`\text{C}_7\text{H}_{16}`) is
   added as an antisolvent in ``MIX02`` while volatile ``THF`` is vaporized in a
   batch ``Evaporator`` (Casas-Orozco et al., 2023b).
4. **Crystallization and filtration** (``CR01`` and ``F01``): The concentrated
   solution is crystallized either in an unseeded cooling ``BatchCryst``
   (``batchU``) or, via a ``BatchToFlowConnector``, in a continuous ``MSMPR``
   cascade (``cont1``--``cont3``) followed by a ``DynamicCollector`` and dead-end
   ``Filter`` (Barhate et al., 2025; Park & Bernal Neira, 2026).

Imports and Property Database
=============================

The 9-species property database (``ISO``, ``CHA``, ``interm``, ``TBN``, ``lom``,
``TBA``, ``SUB1``, ``THF``, ``C7``) is bundled in
``workshop_data/Case_study_3/compounds_lomustine.json`` (matching
``ds-mfg/simulation/data/compounds_lomustine.json`` in ``SECQUOIA/pd_ising``).
In this database, ``SUB1`` (:math:`\text{C}_7\text{H}_{14}\text{ClNO}_2`) carries
its exact stoichiometric molecular weight (:math:`179.643\text{ g/mol}`) while
adopting ``TBA`` liquid/vapor correlation coefficients as a liquid-phase
thermophysical surrogate because ``SUB1`` remains at trace concentrations in solution.

.. testcode::

   from pathlib import Path
   from typing import Optional, Union

   import numpy as np
   from PharmaPy.Reactors import BatchReactor
   from PharmaPy.Streams import BatchToFlowConnector
   from PharmaPy.Phases import LiquidPhase, SolidPhase
   from PharmaPy.Kinetics import CrystKinetics, RxnKinetics
   from PharmaPy.Crystallizers import BatchCryst, MSMPR
   from PharmaPy.Evaporators import Evaporator
   from PharmaPy.SolidLiquidSep import Filter
   from PharmaPy.Containers import DynamicCollector, Mixer
   from PharmaPy.Interpolation import PiecewiseLagrange
   from PharmaPy.Utilities import CoolingWater
   from PharmaPy.SimExec import SimulationExec

   for repository_root in (Path.cwd(), *Path.cwd().parents):
       data_dir = repository_root / "workshop_data" / "Case_study_3"
       if (data_dir / "compounds_lomustine.json").is_file():
           break
   else:
       raise FileNotFoundError("Run this guide inside the PharmaPy checkout.")
   path_phys = str(data_dir / "compounds_lomustine.json")

Kinetics and Solvent-Dependent Solubility
=========================================

Reaction rates in ``R01`` and ``R02`` follow elementary bimolecular mass-action
kinetics :math:`r_i = k_i \exp(-E_{a,i} / (R T)) C_A C_B` with negligible heat
of reaction (:math:`\Delta H_{\text{rxn}, i} = 0\text{ J/mol}`) under jacketed
temperature control, as formulated in the superstructure simulation module
(``ds-mfg/simulation/code/helper.py`` in ``SECQUOIA/pd_ising``; Casas-Orozco et
al., 2023a, 2023b; Barhate et al., 2025; Park & Bernal Neira, 2026).

The equilibrium solubility of lomustine (:math:`C_{\text{sat}}` in
:math:`\text{kg/m}^3`) is modeled as the product of a pure-*n*-heptane
quadratic temperature polynomial and a linear ``THF`` cosolvent enhancement
factor:

.. math::

   C_{\text{sat}}(T, x_{\text{THF}})
   = \left(a_0 + a_1 T + a_2 T^2\right)\left(1 + m_{\text{THF}} x_{\text{THF}}\right)

The quadratic polynomial is calibrated over :math:`255\text{--}305\text{ K}`
(Casas-Orozco et al., 2023b). During the initial segment of batch cooling from
the upstream reactor temperature (:math:`323.15\text{ K}` down to
:math:`305\text{ K}`), the polynomial is smoothly extrapolated; in that high-temperature
region the post-evaporation solution is undersaturated (:math:`C < C_{\text{sat}}`),
so nucleation and growth rates are identically zero until cooling brings the
slurry into the calibrated supersaturated regime.

Crystallization rates in ``CrystKinetics`` use relative supersaturation
:math:`S = (C - C_{\text{sat}}) / C_{\text{sat}}` with parameter tuples
``(A, E_a, order)`` for primary nucleation (:math:`B_p = A_p \exp(-E_{a,p}/(RT)) S^{b_p}`
in :math:`\text{\#/m}^3\text{/s}`), crystal growth (:math:`G = A_g \exp(-E_{a,g}/(RT)) S^{g}`
in :math:`\mu\text{m/s}`), and dissolution, and ``(A, E_a, order, s_crit)`` for
secondary nucleation (:math:`\text{\#/m}^3\text{/s}`).

.. testcode::

   NUM_SPECIES = 9
   SPECIES_MW = np.array([
       105.52, 99.1741, 204.6941, 103.1198, 233.7, 74.123, 179.643,
       72.107, 100.205,
   ])  # [g/mol], ordered as ISO, CHA, interm, TBN, lom, TBA, SUB1, THF, C7
   THF_INDEX = 7
   LOMUSTINE_INDEX = 4
   MICROMETERS_PER_METER = 1.0e6  # [um/m], exact length conversion


   def lomustine_solubility(
       temp: Union[float, np.ndarray],
       conc: Optional[np.ndarray] = None,
       mole_frac: Optional[np.ndarray] = None,
   ) -> Union[float, np.ndarray]:
       """Evaluate lomustine solubility as a function of temperature and THF content.

       Parameters
       ----------
       temp : float or numpy.ndarray
           Liquid temperature [K]. Calibrated over 255.0 to 305.0 K and
           extrapolated up to 323.15 K during initial cooling while undersaturated.
       conc : numpy.ndarray, optional
           Species mass concentrations [kg/m**3], shape ``(9,)`` or
           ``(num_times, 9)``. Takes precedence over ``mole_frac`` when both
           are supplied.
       mole_frac : numpy.ndarray, optional
           Species liquid mole fractions [-], shape ``(9,)`` or
           ``(num_times, 9)``. Used when ``conc`` is ``None``; if both ``conc``
           and ``mole_frac`` are ``None``, ``x_thf = 0.0`` is assumed.

       Returns
       -------
       float or numpy.ndarray
           Equilibrium lomustine concentration [kg/m**3] (g/L), matching the
           broadcast shape of ``temp`` and the time axis of ``conc`` or
           ``mole_frac``.

       Raises
       ------
       ValueError
           If ``temp``, ``conc``, or ``mole_frac`` contains non-finite values,
           if ``conc`` or ``mole_frac`` does not have 1 or 2 dimensions with
           9 species along the last axis, if any entry of ``conc`` or
           ``mole_frac`` is negative, or if any row of ``conc`` has a
           non-positive total mass concentration.

       Notes
       -----
       Correlation and coefficients from ``ds-mfg/simulation/code/helper.py``
       (``solub_fn``) in ``SECQUOIA/pd_ising`` (Casas-Orozco et al., 2023b;
       Barhate et al., 2025; Park & Bernal Neira, 2026).
       """
       temp_arr = np.asarray(temp, dtype=float)  # [K]
       if not np.all(np.isfinite(temp_arr)):
           raise ValueError("Temperature input 'temp' must be finite.")

       thf_solubility_multiplier = 6.5  # [-], empirical THF cosolvent enhancement (helper.py)
       poly_coeffs = np.array([
           3.24643349e2, -2.53376972e0, 4.97645771e-3,
       ])  # [kg/m**3], [kg/m**3/K], [kg/m**3/K**2] (helper.py solub_fn)

       if conc is not None:
           conc_arr = np.asarray(conc, dtype=float)  # [kg/m**3]
           if conc_arr.ndim not in (1, 2) or conc_arr.shape[-1] != NUM_SPECIES:
               raise ValueError(
                   f"'conc' must have shape (9,) or (num_times, 9); got {conc_arr.shape}."
               )
           if not np.all(np.isfinite(conc_arr)) or np.any(conc_arr < 0.0):
               raise ValueError("'conc' must contain finite, non-negative mass concentrations.")
           if conc_arr.ndim == 1:
               total_conc = float(conc_arr.sum())  # [kg/m**3]
               if total_conc <= 0.0:
                   raise ValueError("'conc' must have a strictly positive total concentration.")
               mass_frac = conc_arr / total_conc  # [-]
               x_frac = (mass_frac / SPECIES_MW) / np.dot(mass_frac, 1.0 / SPECIES_MW)  # [-]
               x_thf = float(x_frac[THF_INDEX])  # [-]
           else:
               total_conc = conc_arr.sum(axis=1, keepdims=True)  # [kg/m**3]
               if np.any(total_conc <= 0.0):
                   raise ValueError("Every row of 'conc' must have a positive total concentration.")
               mass_frac = conc_arr / total_conc  # [-]
               x_frac = (
                   (mass_frac / SPECIES_MW)
                   / np.dot(mass_frac, 1.0 / SPECIES_MW)[..., np.newaxis]
               )  # [-]
               x_thf = x_frac[:, THF_INDEX]  # [-]
       elif mole_frac is not None:
           mole_frac_arr = np.asarray(mole_frac, dtype=float)  # [-]
           if mole_frac_arr.ndim not in (1, 2) or mole_frac_arr.shape[-1] != NUM_SPECIES:
               raise ValueError(
                   "'mole_frac' must have shape (9,) or (num_times, 9); "
                   f"got {mole_frac_arr.shape}."
               )
           if not np.all(np.isfinite(mole_frac_arr)) or np.any(mole_frac_arr < 0.0):
               raise ValueError("'mole_frac' must contain finite, non-negative mole fractions.")
           x_thf = (
               float(mole_frac_arr[THF_INDEX])
               if mole_frac_arr.ndim == 1
               else mole_frac_arr[:, THF_INDEX]
           )  # [-]
       else:
           x_thf = 0.0  # [-]

       base_solub = (
           poly_coeffs[0] + poly_coeffs[1] * temp + poly_coeffs[2] * temp**2
       )  # [kg/m**3]
       return base_solub * (1.0 + thf_solubility_multiplier * x_thf)  # [kg/m**3]


   def build_reaction_kinetics(path: str) -> RxnKinetics:
       """Construct the three-reaction lomustine synthesis kinetics object.

       Parameters
       ----------
       path : str
           Filesystem path to ``compounds_lomustine.json``.

       Returns
       -------
       RxnKinetics
           Carbamoylation, nitrosation, and side-reaction kinetics.

       Notes
       -----
       Rate constants and activation energies from ``get_rxn_kinetics`` in
       ``ds-mfg/simulation/code/helper.py`` (``SECQUOIA/pd_ising``;
       Casas-Orozco et al., 2023a, 2023b; Barhate et al., 2025;
       Park & Bernal Neira, 2026). Reaction enthalpies are set to zero under
       the isothermal-jacket modeling assumption of the superstructure study.
       """
       rxns = [
           "ISO + CHA --> interm",
           "interm + TBN --> lom + TBA",
           "ISO + TBA --> SUB1",
       ]
       k_vals = np.array([2.10, 0.07, 0.04])  # [L/mol/s], helper.py get_rxn_kinetics
       ea_vals = np.array([2.0e3, 2.0e3, 2.0e3])  # [J/mol], helper.py get_rxn_kinetics
       delta_hrxn = [0.0, 0.0, 0.0]  # [J/mol], zero-enthalpy assumption in helper.py
       return RxnKinetics(
           path, rxn_list=rxns, k_params=k_vals,
           ea_params=ea_vals, delta_hrxn=delta_hrxn,
       )


   def build_crystallization_kinetics() -> CrystKinetics:
       """Construct lomustine crystallization kinetics.

       Returns
       -------
       CrystKinetics
           Configured crystallization kinetics instance.

       Notes
       -----
       Parameter tuples ``(A, E_a, order)`` and ``(A, E_a, order, s_crit)`` from
       ``get_cryst_kinetics`` in ``ds-mfg/simulation/code/helper.py``
       (``SECQUOIA/pd_ising``; Casas-Orozco et al., 2023b; Barhate et al., 2025;
       Park & Bernal Neira, 2026).
       """
       prim_params = (3.0e8, 0.0, 3.0)  # [#/m**3/s], [J/mol], [-]
       sec_params = (4.46e10, 0.0, 2.0, 1.0e-5)  # [#/m**3/s], [J/mol], [-], [-]
       growth_params = (5.0, 0.0, 1.32)  # [um/s], [J/mol], [-]
       dissol_params = (1.0, 0.0, 1.0)  # [um/s], [J/mol], [-]
       return CrystKinetics(
           solub_fn=lomustine_solubility,
           nucl_prim=prim_params,
           nucl_sec=sec_params,
           growth=growth_params,
           dissolution=dissol_params,
       )

All-Batch Superstructure Configuration
======================================

We first assemble the all-batch superstructure configuration
``('batch', 'batch', 'batch', 'batchU')`` with flowsheet graph
``R01 --> MIX01 --> R02 --> MIX02 --> VAP01 --> CR01 --> F01`` at the NOMAD
optimized operating point recorded for combination ``"0"`` in
``ds-mfg/simulation/data/optimal_points_2.json`` (decoded via ``bounds.json`` in
``SECQUOIA/pd_ising``; Barhate et al., 2025; Park & Bernal Neira, 2026).

.. testcode::

   # Unscaled NOMAD optimum for combo "0" in optimal_points_2.json + bounds.json
   c_in_batch = 0.35  # [mol/L], equimolar ISO and CHA feed concentration
   time_r01_batch = 2825.8434  # [s], R01 batch reaction time
   vol_batch = 0.1  # [m**3], R01 initial batch charge volume (100 L)
   c_tbn_batch = 0.35  # [mol/L], TBN charge concentration basis in MIX01
   time_r02_batch = 3600.0  # [s], R02 batch reaction time
   ratio_hept_batch = 0.50001715  # [-], C7-to-batch volume ratio in MIX02
   time_vap_batch = 3600.0  # [s], VAP01 batch evaporation duration
   pressure_vap_batch = 25002.3  # [Pa], VAP01 operating pressure
   time_cr_batch = 3600.0  # [s], CR01 cooling crystallization duration
   temp_cr_1 = 291.08157  # [K], first cooling segment endpoint
   temp_cr_2 = 287.9188  # [K], second cooling segment endpoint
   temp_cr_f = 260.0012114  # [K], final cooling temperature
   dp_filt_batch = 100000.0  # [Pa], F01 filtration pressure driving force
   diam_filt_batch = 0.5  # [m], F01 filter station diameter

   # Fixed equipment/utility design parameters from ds-mfg/simulation/code/make_units.py
   rxn_temp = 323.15  # [K], reactor and jacket temperature (50 degC)
   reactor_cw_flow = 0.1  # [kg/s], reactor cooling water mass flow
   evap_utility_temp = 307.15  # [K], evaporator heating utility temperature
   evap_utility_flow = 1.0  # [kg/s], evaporator utility mass flow
   evap_headspace_beta = 0.75  # [-], liquid-to-drum volume ratio (25% vapor space)
   evap_h_conv = 1000.0  # [W/m**2/K], evaporator convective heat-transfer coefficient
   cryst_scale = 1e-11  # [-], population-balance moment scaling factor
   max_cryst_step = 100.0  # [s], maximum solver step in batch crystallizer
   liters_per_m3 = 1000.0  # [L/m**3], exact volume conversion

   graph_batch = "R01 --> MIX01 --> R02 --> MIX02 --> VAP01 --> CR01 --> F01"
   sim_batch = SimulationExec(path_phys, flowsheet=graph_batch)

   # Step 1: R01 (BatchReactor) and MIX01 (TBN reagent addition)
   conc_init_batch = [c_in_batch, c_in_batch] + [0.0] * 7  # [mol/L]
   sim_batch.R01 = BatchReactor(isothermal=False)
   sim_batch.R01.Phases = LiquidPhase(
       path_phys, vol=vol_batch, mole_conc=conc_init_batch,
       temp=rxn_temp, name_solv="THF",
   )
   sim_batch.R01.Kinetics = build_reaction_kinetics(path_phys)
   sim_batch.R01.Utility = CoolingWater(temp_in=rxn_temp, mass_flow=reactor_cw_flow)

   mole_frac_tbn = [0.0] * 3 + [1.0] + [0.0] * 5  # [-], pure TBN
   moles_tbn_batch = c_tbn_batch * vol_batch * liters_per_m3  # [mol]
   sim_batch.MIX01 = Mixer()
   sim_batch.MIX01.Inlets = LiquidPhase(
       path_phys, moles=moles_tbn_batch, mole_frac=mole_frac_tbn,
   )

   # Step 2: R02 (BatchReactor) and MIX02 (C7 antisolvent addition)
   sim_batch.R02 = BatchReactor(isothermal=False)
   sim_batch.R02.Kinetics = build_reaction_kinetics(path_phys)
   sim_batch.R02.Utility = CoolingWater(temp_in=rxn_temp, mass_flow=reactor_cw_flow)

   mass_frac_hept = [0.0] * 8 + [1.0]  # [-], pure C7
   vol_hept_batch = vol_batch * ratio_hept_batch  # [m**3]
   sim_batch.MIX02 = Mixer()
   sim_batch.MIX02.Inlets = LiquidPhase(
       path_phys, vol=vol_hept_batch, mass_frac=mass_frac_hept,
   )

   # Step 3: VAP01 (Batch Evaporator)
   vol_drum_batch = vol_batch * (1.0 + ratio_hept_batch) / evap_headspace_beta  # [m**3]
   sim_batch.VAP01 = Evaporator(
       pressure=pressure_vap_batch, vol_drum=vol_drum_batch,
       flash_kwargs={"seed_basedon_input": True},
       include_nitrogen=False, h_conv=evap_h_conv,
   )
   sim_batch.VAP01.Utility = CoolingWater(
       temp_in=evap_utility_temp, mass_flow=evap_utility_flow,
   )

   # Step 4: CR01 (Unseeded BatchCryst) and F01 (Filter)
   x_grid = np.geomspace(1.0, 500.0, 25)  # [um], 25-point geometric size grid
   solid_frac = [0] * 4 + [1, 0, 0, 0, 0]  # [-], pure crystalline lomustine
   temp_program = np.array([
       [rxn_temp, temp_cr_1],
       [temp_cr_1, temp_cr_2],
       [temp_cr_2, temp_cr_f],
   ], dtype=float)  # [K], three-segment cooling profile
   interp_cr01 = PiecewiseLagrange(time_final=time_cr_batch, y_vals=temp_program)

   sim_batch.CR01 = BatchCryst(
       target_comp="lom", method="1D-FVM", scale=cryst_scale,
       controls={"temp": interp_cr01.evaluate_poly},
   )
   sim_batch.CR01.Phases = SolidPhase(
       path_phys, distrib=np.zeros_like(x_grid), x_distrib=x_grid,
       mass_frac=solid_frac,
   )
   sim_batch.CR01.Kinetics = build_crystallization_kinetics()
   sim_batch.F01 = Filter(station_diam=diam_filt_batch)

   runargs_batch = {
       "R01": {"runtime": time_r01_batch, "verbose": False},
       "R02": {"runtime": time_r02_batch, "verbose": False},
       "VAP01": {"runtime": time_vap_batch, "verbose": False},
       "CR01": {
           "runtime": time_cr_batch,
           "sundials_opts": {"maxh": max_cryst_step},
           "verbose": False,
       },
       "F01": {"deltaP": dp_filt_batch, "verbose": False},
   }
   sim_batch.SolveFlowsheet(kwargs_run=runargs_batch, verbose=False)

   moments_batch = sim_batch.CR01.Solid_1.getMoments()  # [#/m**3, m/m**3, ...]
   if moments_batch[0] <= 0.0:
       raise RuntimeError("No crystal population formed in CR01 (all-batch).")
   d10_batch = (moments_batch[1] / moments_batch[0]) * MICROMETERS_PER_METER  # [um]
   cycle_batch = max(
       time_r01_batch, time_r02_batch, time_vap_batch, time_cr_batch,
   )  # [s], limiting batch unit duration in repeating campaign (generating_structures.py)
   productivity_batch = (
       sim_batch.F01.result.mass_cake_dry[-1] / cycle_batch
   )  # [kg/s]

Hybrid Batch-Reaction / Continuous-MSMPR Superstructure Configuration
=====================================================================

Next, we construct the hybrid batch-reaction / continuous-crystallization
configuration ``('batch', 'batch', 'batch', 'cont1')`` with flowsheet graph
``R01 --> MIX01 --> R02 --> MIX02 --> VAP01 --> BatchToFlow01 --> CR01 --> HOLD01 --> F01``
at the NOMAD optimized operating point recorded for combination ``"1"`` in
``ds-mfg/simulation/data/optimal_points_2.json`` (Barhate et al., 2025; Park &
Bernal Neira, 2026).

In this hybrid campaign model:

1. **Repeated upstream batch supply**: Parallel/staggered upstream batch trains
   complete one batch every ``b2f_cycle_time = max(time_R01, time_R02, time_VAP01)``
   (:math:`3600\text{ s} = 1\text{ h}`). ``BatchToFlowConnector`` converts the
   evaporator liquid batch into a continuous stream of volumetric flow rate
   :math:`Q_{\text{in}} = V_{\text{VAP01, out}} / t_{\text{cycle}}` that feeds the
   continuous ``MSMPR`` across the 10-hour ``campaign_horizon`` (:math:`36000\text{ s}`,
   corresponding to 10 consecutive upstream batches).
2. **Pre-solve vs. actual MSMPR residence time**: Because ``SimulationExec``
   requires ``CR01.Phases`` to be initialized before ``VAP01`` is solved, the
   superstructure formulation in ``make_units.py`` (``add_crystallizer``) sizes
   the initial ``CR01`` liquid holdup volume using the nominal pre-solve flow
   proxy ``effective_flow = vol_drum_hybrid / campaign_horizon``:
   ``vol_cryst_hybrid = effective_flow * tau_cr01_hybrid``. Once ``VAP01`` and
   ``BatchToFlow01`` execute, the actual hydraulic residence time in ``CR01`` is
   :math:`\tau_{\text{actual}} = V_{\text{CR01}} / Q_{\text{in}}`, which we compute
   explicitly below.

.. testcode::

   # Unscaled NOMAD optimum for combo "1" in optimal_points_2.json + bounds.json
   c_in_hybrid = 0.35  # [mol/L], equimolar ISO and CHA feed concentration
   time_r01_hybrid = 3420.324  # [s], R01 batch reaction time
   vol_hybrid = 0.1  # [m**3], R01 initial batch charge volume
   c_tbn_hybrid = 0.35  # [mol/L], TBN charge concentration basis in MIX01
   time_r02_hybrid = 3600.0  # [s], R02 batch reaction time
   ratio_hept_hybrid = 0.5  # [-], C7-to-batch volume ratio in MIX02
   time_vap_hybrid = 3600.0  # [s], VAP01 batch evaporation duration
   pressure_vap_hybrid = 25000.0  # [Pa], VAP01 operating pressure
   tau_cr01_hybrid = 5399.64  # [s], nominal CR01 MSMPR residence-time parameter
   temp_cr01_hybrid = 260.00096  # [K], CR01 MSMPR operating temperature
   dp_filt_hybrid = 100000.0  # [Pa], F01 filtration pressure driving force
   diam_filt_hybrid = 0.5  # [m], F01 filter station diameter
   campaign_horizon = 10.0 * 3600.0  # [s], 10-hour continuous operating horizon

   graph_hybrid = (
       "R01 --> MIX01 --> R02 --> MIX02 --> VAP01 "
       "--> BatchToFlow01 --> CR01 --> HOLD01 --> F01"
   )
   sim_hybrid = SimulationExec(path_phys, flowsheet=graph_hybrid)

   conc_init_hybrid = [c_in_hybrid, c_in_hybrid] + [0.0] * 7  # [mol/L]
   sim_hybrid.R01 = BatchReactor(isothermal=False)
   sim_hybrid.R01.Phases = LiquidPhase(
       path_phys, vol=vol_hybrid, mole_conc=conc_init_hybrid,
       temp=rxn_temp, name_solv="THF",
   )
   sim_hybrid.R01.Kinetics = build_reaction_kinetics(path_phys)
   sim_hybrid.R01.Utility = CoolingWater(temp_in=rxn_temp, mass_flow=reactor_cw_flow)

   moles_tbn_hybrid = c_tbn_hybrid * vol_hybrid * liters_per_m3  # [mol]
   sim_hybrid.MIX01 = Mixer()
   sim_hybrid.MIX01.Inlets = LiquidPhase(
       path_phys, moles=moles_tbn_hybrid, mole_frac=mole_frac_tbn,
   )

   sim_hybrid.R02 = BatchReactor(isothermal=False)
   sim_hybrid.R02.Kinetics = build_reaction_kinetics(path_phys)
   sim_hybrid.R02.Utility = CoolingWater(temp_in=rxn_temp, mass_flow=reactor_cw_flow)

   vol_hept_hybrid = vol_hybrid * ratio_hept_hybrid  # [m**3]
   sim_hybrid.MIX02 = Mixer()
   sim_hybrid.MIX02.Inlets = LiquidPhase(
       path_phys, vol=vol_hept_hybrid, mass_frac=mass_frac_hept,
   )

   vol_drum_hybrid = (
       vol_hybrid * (1.0 + ratio_hept_hybrid) / evap_headspace_beta
   )  # [m**3]
   sim_hybrid.VAP01 = Evaporator(
       pressure=pressure_vap_hybrid, vol_drum=vol_drum_hybrid,
       flash_kwargs={"seed_basedon_input": True},
       include_nitrogen=False, h_conv=evap_h_conv,
   )
   sim_hybrid.VAP01.Utility = CoolingWater(
       temp_in=evap_utility_temp, mass_flow=evap_utility_flow,
   )

   b2f_cycle_time = max(
       time_r01_hybrid, time_r02_hybrid, time_vap_hybrid,
   )  # [s], upstream batch train cycle time (make_units.py add_b2f)
   sim_hybrid.BatchToFlow01 = BatchToFlowConnector(cycle_time=b2f_cycle_time)

   # Pre-solve superstructure holdup parameterization (make_units.py add_crystallizer)
   effective_flow = vol_drum_hybrid / campaign_horizon  # [m**3/s]
   vol_cryst_hybrid = effective_flow * tau_cr01_hybrid  # [m**3]
   sim_hybrid.CR01 = MSMPR(
       target_comp="lom", method="1D-FVM", scale=cryst_scale,
   )
   sim_hybrid.CR01.Phases = (
       LiquidPhase(
           path_phys, mass_frac=mass_frac_hept, vol=vol_cryst_hybrid,
           temp=temp_cr01_hybrid,
       ),
       SolidPhase(
           path_phys, distrib=np.zeros_like(x_grid), x_distrib=x_grid,
           mass_frac=solid_frac,
       ),
   )
   sim_hybrid.CR01.Kinetics = build_crystallization_kinetics()
   sim_hybrid.CR01.Utility = CoolingWater(
       temp_in=temp_cr01_hybrid, mass_flow=evap_utility_flow,
   )

   sim_hybrid.HOLD01 = DynamicCollector()
   sim_hybrid.F01 = Filter(station_diam=diam_filt_hybrid)

   runargs_hybrid = {
       "R01": {"runtime": time_r01_hybrid, "verbose": False},
       "R02": {"runtime": time_r02_hybrid, "verbose": False},
       "VAP01": {"runtime": time_vap_hybrid, "verbose": False},
       "CR01": {"runtime": campaign_horizon, "verbose": False},
       "HOLD01": {"runtime": campaign_horizon, "verbose": False},
       "F01": {"deltaP": dp_filt_hybrid, "verbose": False},
   }
   sim_hybrid.SolveFlowsheet(kwargs_run=runargs_hybrid, verbose=False)

   q_in_cr01 = float(sim_hybrid.BatchToFlow01.Outlet.vol_flow)  # [m**3/s], actual MSMPR inlet flow
   tau_actual_cr01 = vol_cryst_hybrid / q_in_cr01  # [s], actual hydraulic residence time

   moments_hybrid = sim_hybrid.CR01.Solid_1.getMoments()  # [#/m**3, m/m**3, ...]
   if moments_hybrid[0] <= 0.0:
       raise RuntimeError("No crystal population formed in CR01 (hybrid MSMPR).")
   d10_hybrid = (moments_hybrid[1] / moments_hybrid[0]) * MICROMETERS_PER_METER  # [um]
   productivity_hybrid = (
       sim_hybrid.F01.result.mass_cake_dry[-1] / campaign_horizon
   )  # [kg/s]
