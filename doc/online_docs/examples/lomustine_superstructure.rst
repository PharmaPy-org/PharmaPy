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
kinetic, techno-economic, and process-synthesis studies (see also
:doc:`../publications`):

- **Reaction kinetics and uncertainty analysis**:
  Casas-Orozco et al. (2023a), *Chemical Engineering Science*, 280, 118987
  (`doi:10.1016/j.ces.2023.118987 <https://doi.org/10.1016/j.ces.2023.118987>`_).
- **Dynamic end-to-end campaign manufacturing and techno-economic analysis**:
  Casas-Orozco et al. (2023b), *AIChE Journal*, 69(9), e18142
  (`doi:10.1002/aic.18142 <https://doi.org/10.1002/aic.18142>`_).
- **Hybrid rule-based and optimization-driven superstructure synthesis**:
  Barhate et al. (2025), *AIChE Journal*, e18888
  (`doi:10.1002/aic.18888 <https://doi.org/10.1002/aic.18888>`_).

Process Chemistry and Superstructure Overview
---------------------------------------------

Lomustine (``lom``) is synthesized in tetrahydrofuran (``THF``) via a two-step
reaction sequence:

1. **Carbamoylation** (``R01``): 1-chloro-2-isocyanatoethane (``ISO``) reacts
   with cyclohexylamine (``CHA``) to form the urea intermediate
   1-(2-chloroethyl)-3-cyclohexylurea (``interm``):

   .. math::

      \text{ISO} + \text{CHA} \xrightarrow{k_1} \text{interm}

2. **Nitrosation and competing side reaction** (``R02``): *tert*-butyl nitrite
   (``TBN``) nitrosates ``interm`` to form lomustine (``lom``) and *tert*-butanol
   (``TBA``), while unreacted ``ISO`` can react with ``TBA`` to produce the
   urethane byproduct (``SUB1``):

   .. math::

      \text{interm} + \text{TBN} \xrightarrow{k_2} \text{lom} + \text{TBA}

   .. math::

      \text{ISO} + \text{TBA} \xrightarrow{k_3} \text{SUB1}

3. **Evaporative solvent switch** (``VAP01``): Because lomustine is highly
   soluble in ``THF``, *n*-heptane (``C7``) is added as an antisolvent in
   ``MIX02`` while volatile ``THF`` is vaporized in a batch ``Evaporator``
   (Casas-Orozco et al., 2023b).
4. **Crystallization and filtration** (``CR01`` and ``F01``): The concentrated
   solution is crystallized either in an unseeded cooling ``BatchCryst``
   (``batchU``) or, via a ``BatchToFlowConnector``, in a continuous ``MSMPR``
   cascade (``cont1``--``cont3``) followed by a ``DynamicCollector`` and dead-end
   ``Filter`` (Barhate et al., 2025).

Imports and Property Database
=============================

The 9-species property database (``ISO``, ``CHA``, ``interm``, ``TBN``, ``lom``,
``TBA``, ``SUB1``, ``THF``, ``C7``) is bundled in
``workshop_data/Case_study_3/compounds_lomustine.json``.

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

Reaction rates in ``R01`` and ``R02`` follow elementary bimolecular kinetics
(Casas-Orozco et al., 2023a), while the equilibrium solubility of lomustine
depends on both temperature and the residual ``THF`` mole fraction after the
``THF``/``C7`` solvent switch (Casas-Orozco et al., 2023b; Barhate et al., 2025).

.. testcode::

   SPECIES_MW = np.array([
       105.52, 99.1741, 204.6941, 103.1198, 233.7, 74.123, 179.643,
       72.107, 100.205,
   ])  # [g/mol], ordered as ISO, CHA, interm, TBN, lom, TBA, SUB1, THF, C7
   THF_INDEX = 7
   LOMUSTINE_INDEX = 4


   def lomustine_solubility(
       temp: Union[float, np.ndarray],
       conc: Optional[np.ndarray] = None,
       mole_frac: Optional[np.ndarray] = None,
   ) -> Union[float, np.ndarray]:
       """Evaluate lomustine solubility as a function of temperature and THF content.

       Parameters
       ----------
       temp : float or numpy.ndarray
           Liquid temperature [K], calibrated over 255 to 305 K.
       conc : numpy.ndarray, optional
           Species mass concentrations [kg/m**3], shape ``(9,)`` or
           ``(num_times, 9)``.
       mole_frac : numpy.ndarray, optional
           Species liquid mole fractions [-], shape ``(9,)`` or
           ``(num_times, 9)``.

       Returns
       -------
       float or numpy.ndarray
           Equilibrium lomustine concentration [kg/m**3] (g/L).

       Notes
       -----
       Correlation from Casas-Orozco et al. (2023b) and Barhate et al. (2025).
       """
       thf_solubility_multiplier = 6.5  # [-], empirical THF cosolvent enhancement
       poly_coeffs = np.array([
           3.24643349e2, -2.53376972e0, 4.97645771e-3,
       ])  # [kg/m**3], [kg/m**3/K], [kg/m**3/K**2]

       if conc is not None:
           if conc.ndim == 1:
               mass_frac = conc / conc.sum()  # [-]
               x_frac = (mass_frac / SPECIES_MW) / np.dot(mass_frac, 1.0 / SPECIES_MW)  # [-]
               x_thf = x_frac[THF_INDEX]  # [-]
           else:
               mass_frac = conc / conc.sum(axis=1)[..., np.newaxis]  # [-]
               x_frac = (
                   (mass_frac / SPECIES_MW)
                   / np.dot(mass_frac, 1.0 / SPECIES_MW)[..., np.newaxis]
               )  # [-]
               x_thf = x_frac[:, THF_INDEX]  # [-]
       elif mole_frac is not None:
           x_thf = mole_frac[THF_INDEX] if mole_frac.ndim == 1 else mole_frac[:, THF_INDEX]  # [-]
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
       """
       rxns = [
           "ISO + CHA --> interm",
           "interm + TBN --> lom + TBA",
           "ISO + TBA --> SUB1",
       ]
       k_vals = np.array([210.0, 7.0, 4.0]) * 1e-2  # [L/mol/s]
       ea_vals = np.array([2.0e3, 2.0e3, 2.0e3])  # [J/mol]
       delta_hrxn = [0.0, 0.0, 0.0]  # [J/mol], negligible heat of reaction
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
``R01 --> MIX01 --> R02 --> MIX02 --> VAP01 --> CR01 --> F01`` at its optimized
operating point from Barhate et al. (2025).

.. testcode::

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
   d10_batch = moments_batch[1] / (moments_batch[0] + 1e-6) * 1e6  # [um]
   cycle_batch = max(
       time_r01_batch, time_r02_batch, time_vap_batch, time_cr_batch,
   )  # [s], limiting batch unit duration
   productivity_batch = (
       sim_batch.F01.result.mass_cake_dry[-1] / (cycle_batch + 1e-6)
   )  # [kg/s]

Hybrid Batch-Reaction / Continuous-MSMPR Superstructure Configuration
=====================================================================

Next, we construct the hybrid batch-reaction / continuous-crystallization
configuration ``('batch', 'batch', 'batch', 'cont1')`` with flowsheet graph
``R01 --> MIX01 --> R02 --> MIX02 --> VAP01 --> BatchToFlow01 --> CR01 --> HOLD01 --> F01``
(Barhate et al., 2025). Here ``BatchToFlowConnector`` converts the batch
evaporator holdup into a continuous feed discharged over the upstream batch
cycle time ``b2f_cycle_time = max(time_R01, time_R02, time_VAP01)`` [s] into a
continuous ``MSMPR`` crystallizer, followed by a ``DynamicCollector``
(``HOLD01``) and dead-end ``Filter`` (``F01``).

.. testcode::

   c_in_hybrid = 0.35  # [mol/L], equimolar ISO and CHA feed concentration
   time_r01_hybrid = 3420.324  # [s], R01 batch reaction time
   vol_hybrid = 0.1  # [m**3], R01 initial batch charge volume
   c_tbn_hybrid = 0.35  # [mol/L], TBN charge concentration basis in MIX01
   time_r02_hybrid = 3600.0  # [s], R02 batch reaction time
   ratio_hept_hybrid = 0.5  # [-], C7-to-batch volume ratio in MIX02
   time_vap_hybrid = 3600.0  # [s], VAP01 batch evaporation duration
   pressure_vap_hybrid = 25000.0  # [Pa], VAP01 operating pressure
   tau_cr01_hybrid = 5399.64  # [s], CR01 MSMPR residence time
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
   )  # [s], upstream batch train cycle time
   sim_hybrid.BatchToFlow01 = BatchToFlowConnector(cycle_time=b2f_cycle_time)

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

   moments_hybrid = sim_hybrid.CR01.Solid_1.getMoments()  # [#/m**3, m/m**3, ...]
   d10_hybrid = moments_hybrid[1] / (moments_hybrid[0] + 1e-6) * 1e6  # [um]
   productivity_hybrid = (
       sim_hybrid.F01.result.mass_cake_dry[-1] / (campaign_horizon + 1e-6)
   )  # [kg/s]
