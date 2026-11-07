Crystallization with QMOM
=========================

The refactored crystallizers can carry their crystals with the quadrature
method of moments (QMOM). QMOM tracks a few moments of the crystal size
distribution and, at every right-hand side, recovers from them a small set of
sizes (the nodes) and counts (the weights). Terms that plain moment equations
cannot close, namely size-dependent growth, agglomeration and breakage, are then
evaluated on those nodes (McGraw, 1997; Marchisio et al., 2003).

.. important::

   The constants on this page are **illustrative**, not fitted to any real
   system. QMOM in PharmaPy has been verified against closed-form solutions and
   conservation laws (see `Verification`_). That is **not evidence** that a
   given kernel describes a given crystallizer, which takes experimental data.

Choosing a population balance
-----------------------------

The three forms attach to the solid phase in the same way, and work in
``BatchCrystallizer``, ``SemiBatchCrystallizer`` and
``ContinuousCrystallizer``.

.. list-table::
   :header-rows: 1
   :widths: 28 24 24 24

   * -
     - ``MomentsPopulationBalance``
     - ``QMOMPopulationBalance``
     - ``OneDFVMMechanism``
   * - State carried
     - Moments
     - An even number of moments, 2N
     - Number density on a size grid
   * - Size-dependent growth
     - No
     - Yes
     - Yes
   * - Agglomeration and breakage
     - No
     - Yes
     - No
   * - Size distribution available
     - No
     - No, only the N nodes
     - Yes

Use QMOM when you need agglomeration, breakage or size-dependent growth and the
moments and mean sizes are enough. Use the FVM form when you need the
distribution itself.

Setting up a crystallizer
-------------------------

The state is the moments :math:`\mu_0 \ldots \mu_{2N-1}` [um**k/m**3 of slurry],
so ``moments_init`` needs an even number of entries, at least four. Six moments
give three nodes, the setting used throughout this page and the one Marchisio et al.
(2003) call a good trade-off. Here they come from a population of three seed sizes,
:math:`\mu_k = \sum_i n_i L_i^k`.

The helper below builds one isothermal crystallizer, and the later examples
reuse it. It runs on SciPy, so it does not need Assimulo. The crystal density
equals the liquid density of the solute in this database, which keeps the slurry
volume constant and makes the results easy to check.

.. testcode::

   from pathlib import Path

   import numpy as np

   from PharmaPy.CrystallizationKernels import (BinaryFragmentation, ConstantKernel,
                                                ConstantSelection)
   from PharmaPy.Crystallizers_Refactored import (BatchCrystallizer, ContinuousCrystallizer,
                                                  SemiBatchCrystallizer)
   from PharmaPy.IntegratorBackends import ScipyBackend
   from PharmaPy.Kinetics import CrystKinetics
   from PharmaPy.Mechanisms import QMOMPopulationBalance
   from PharmaPy.Phases_Refactored import LiquidPhase, SolidPhase
   from PharmaPy.SimExec import SimulationExec
   from PharmaPy.Streams_Refactored import LiquidStream

   for repository_root in (Path.cwd(), *Path.cwd().parents):
       path_data = repository_root / "tests" / "Flowsheet" / "data" / "compound_database.json"
       if path_data.is_file():
           break
   else:
       raise FileNotFoundError("Run this guide inside the PharmaPy checkout.")
   path_data = str(path_data)

   LIQUID = [0.0, 0.0, 0.2, 0.0, 0.8]  # [-], mass fractions of A, B, C, D, solvent
   CRYSTAL = [0.0, 0.0, 1.0, 0.0, 0.0]  # [-], pure C
   TEMP = 298.15  # [K]
   SOLUBILITY = 50.0  # [kg/m**3], constant


   def crystallizer(vessel_class, moments, growth=(0.0, 0.0, 1.0), feed=None, **kernels):
       """Build an isothermal crystallizer holding 1 kg of liquid and QMOM crystals."""
       solid = SolidPhase(path_data, mass=0.0, mass_frac=CRYSTAL, temp=TEMP)
       solid.mechanisms = QMOMPopulationBalance(
           owning_phase=solid, target_components="C", solvent_name="solvent",
           moments_init=moments, density=1200.0, kv=0.5, **kernels)
       vessel = vessel_class(
           integrator=ScipyBackend(method="LSODA", options={"rtol": 1e-10, "atol": 1e-10}),
           isothermal=True)
       vessel.Phases = [LiquidPhase(path_data, mass=1.0, mass_frac=LIQUID, temp=TEMP,
                                    name_solv="solvent"), solid]
       vessel.CrystKinetics = CrystKinetics(coeff_solub=[SOLUBILITY], growth=growth)
       if feed is not None:
           vessel.Inlet = LiquidStream(path_data, mass_flow=feed, mass_frac=LIQUID, temp=TEMP,
                                       name_solv="solvent")
       return vessel


   seed_sizes = np.array([40.0, 60.0, 90.0])  # [um]
   seed_counts = np.array([5e9, 3e9, 1e9])  # [#/m**3 of slurry]
   seed_moments = np.sum(seed_counts[:, None] * seed_sizes[:, None] ** np.arange(6),
                         axis=0)  # [um**k/m**3 of slurry]
   print(f"{seed_moments[0]:.3e} crystals per m**3, "
         f"L43 = {seed_moments[4] / seed_moments[3]:.1f} um")

.. testoutput::

   9.000e+09 crystals per m**3, L43 = 69.1 um

Size-dependent growth
---------------------

``CrystKinetics`` takes the growth parameters
``(k_g, E_g, g, alpha, beta)``, with ``k_g`` [um/s], ``E_g`` [J/mol], ``g`` [-],
``alpha`` [-] and ``beta`` [1/um]. The last two give the size factor

.. math::

   G(L) = G_0 \, (1 + \beta L)^{\alpha}.

The quadrature integrates polynomials of degree below 2N exactly, so
``alpha = 0`` and ``alpha = 1`` are exact, and other exponents are approximated,
the more closely the more nodes. Leave out ``alpha`` and ``beta``, or set
``alpha = 0``, and QMOM reduces exactly to the moment form.

Agglomeration
-------------

Pass a kernel as ``agglomeration``. Each one returns the rate coefficient
:math:`\beta(L_i, L_j)` [m**3/s] of a crystal pair per m**3 of slurry. Table 1 of
Marchisio et al. (2003) gives only each kernel's shape, in sizes scaled so that the
primary particle is 1. The classes put physical constants in front of that shape, so
the rate comes out in SI units. In the table, :math:`v = k_v L^3` is the crystal
volume, sizes are in metres, and the degree is how the shape scales with size, the
paper's homogeneity index.

.. list-table::
   :header-rows: 1
   :widths: 26 26 18 30

   * - Class
     - :math:`\beta`
     - Shape in Table 1, and degree
     - Parameters
   * - ``ConstantKernel``
     - :math:`\beta_0`
     - 1, degree 0
     - ``rate_constant`` :math:`\beta_0` [m**3/s]
   * - ``BrownianKernel``
     - :math:`\frac{2 k_B T}{3 \mu} \frac{(L_i + L_j)^2}{L_i L_j}`
     - :math:`\frac{(L + \lambda)^2}{L \lambda}`, degree 0
     - ``viscosity`` :math:`\mu` [Pa s]. Needs a nucleus size ``rad`` above zero
   * - ``SumKernel``
     - :math:`\beta_0 (v_i + v_j)`
     - :math:`L^3 + \lambda^3`, degree 3
     - ``rate_constant`` :math:`\beta_0` [1/s]
   * - ``ProductKernel``
     - :math:`\beta_0 v_i v_j`
     - Not in Table 1, degree 6
     - ``rate_constant`` :math:`\beta_0` [1/(m**3 s)]
   * - ``ShearKernel`` (hydrodynamic)
     - :math:`\frac{e \gamma}{6} (L_i + L_j)^3`
     - :math:`(L + \lambda)^3`, degree 3
     - ``shear_rate`` :math:`\gamma` [1/s], ``efficiency`` :math:`e` [-]
   * - ``DifferentialSedimentationKernel`` (differential force)
     - :math:`e \frac{\pi g \Delta\rho}{72 \mu} (L_i + L_j)^2 |L_i^2 - L_j^2|`
     - :math:`(L + \lambda)^2 |L^2 - \lambda^2|`, degree 4
     - ``density_difference`` :math:`\Delta\rho` [kg/m**3], ``viscosity``
       :math:`\mu` [Pa s], ``efficiency`` :math:`e` [-]. Valid under Stokes drag

The constants are PharmaPy's, derived from the collision physics rather than taken
from the paper:

- **Hydrodynamic.** Smoluchowski's (1917) rate in laminar shear,
  :math:`\frac{4}{3} \gamma (a_i + a_j)^3` for radii :math:`a = L/2`, gives
  :math:`\gamma / 6`. The efficiency :math:`e` is the fraction of collisions that
  stick.
- **Differential force.** The collision cross-section
  :math:`\frac{\pi}{4} (L_i + L_j)^2` times the difference of the Stokes settling
  velocities :math:`g \Delta\rho L^2 / (18 \mu)` gives
  :math:`\pi g \Delta\rho / (72 \mu)`. The paper uses this shape for turbulent
  inertia as well, which has other constants and no class here.
- **Brownian.** Smoluchowski's (1917) rate :math:`4 \pi (D_i + D_j)(a_i + a_j)`, with
  Stokes-Einstein diffusivities, gives :math:`2 k_B T / (3 \mu)`.

To get the paper's scaled forms, choose the constants so the factor in front is 1.
``ShearKernel(6e18)`` gives :math:`(L_i + L_j)^3` with sizes in um, as in
``tests/test_qmom_marchisio_benchmark.py``.

.. warning::

   Marchisio et al. give a degree of at least 3 as the condition for gelation, where
   agglomeration runs away into ever larger crystals. That covers every kernel above
   except the constant and Brownian ones, and the way a run ends differs. These are
   runs without breakage, from two sizes in scaled units, with the constants set to
   give the paper's shapes:

   - ``ProductKernel``: the solver fails at :math:`t = 0.40`, the closed-form gel time
     :math:`1 / \mu_6(0)`.
   - ``DifferentialSedimentationKernel``: the nodes merge into one by :math:`t = 1.9`.
     Crystals of equal size never meet, so the count freezes and **the run finishes
     without an error**, on a mean size that means nothing. See issue
     `#564 <https://github.com/PharmaPy-org/PharmaPy/issues/564>`__.
   - ``ShearKernel``: L43 reaches 358 by :math:`t = 2`, and the solve does not reach
     :math:`t = 5` in a minute.
   - ``SumKernel``: stays finite, as its closed form does (Scott, 1968), but L43 reaches
     16 000 by :math:`t = 10`.

   Breakage held the sizes back in the cases under `Comparison with the published
   results`_, but the paper notes that with both processes the transition may or may
   not occur. Check that the mean size stays believable.

Breakage
--------

Breakage takes a selection function, how often a crystal of size :math:`L` breaks,
and a fragment distribution, what it breaks into. Pass both or neither. They
follow Tables 2 and 3 of Marchisio et al. (2003).

.. list-table::
   :header-rows: 1
   :widths: 32 30 38

   * - ``breakage=``
     - :math:`a(L)` [1/s]
     - Parameters
   * - ``ConstantSelection``
     - :math:`S_0`
     - ``rate_constant`` :math:`S_0` [1/s]
   * - ``PowerLawSelection``
     - :math:`S_0 (L / L_{ref})^{\alpha}`
     - ``rate_constant`` :math:`S_0` [1/s], ``exponent`` :math:`\alpha \ge 0` [-],
       ``reference_size`` :math:`L_{ref}` [um]. A negative exponent is refused:
       small crystals would shatter into sizes the moments cannot carry
   * - ``ExponentialSelection``
     - :math:`S_0 \exp(\delta L^3)`
     - ``rate_constant`` :math:`S_0` [1/s], ``volume_coefficient``
       :math:`\delta \ge 0` [1/um**3]

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - ``fragments=``
     - What one parent becomes
   * - ``BinaryFragmentation``
     - Two fragments with the fractions ``volume_fraction`` and its complement of
       the parent's volume. 0.5, the default, is symmetric; 0.2 is the mass ratio
       1:4
   * - ``ErosionFragmentation``
     - One fragment of ``fragment_size`` [um] and the eroded parent. A parent no
       larger than the fragment does not erode
   * - ``ParabolicFragmentation``
     - Two fragments on average, spread by ``shape_parameter`` from 0 to 3; 2 is
       uniform
   * - ``UniformFragmentation``
     - Two fragments on average, spread evenly in volume below the parent's size

Both processes conserve crystal volume. The fragments' volume is checked as the
solve runs, so a user-written distribution that does not conserve it is refused.

Crystals that do not break
~~~~~~~~~~~~~~~~~~~~~~~~~~

Each selection function takes ``minimum_size`` [um]: a crystal of that size or
smaller does not break, like the primary particle in Table 4 of Marchisio et al.
(2003). Leave it out and every crystal breaks. Here, seeds of 100 um break once,
because their fragments of 79.4 um are below the minimum.

.. testcode::

   one_seed_size = 1e9 * 100.0 ** np.arange(6)  # [um**k/m**3 of slurry]
   once = crystallizer(BatchCrystallizer, one_seed_size,
                       breakage=ConstantSelection(0.01, minimum_size=80.0),
                       fragments=BinaryFragmentation())
   once.solve_unit(runtime=1500.0, verbose=False)
   once_count = np.asarray(once.result.mu_n_solid0)[:, 0]  # [#/m**3 of slurry]
   print(f"Crystal count: {once_count[0]:.3e} to {once_count[-1]:.3e} #/m**3")
   print(f"L43: {np.ravel(once.result.L43)[0]:.1f} to {np.ravel(once.result.L43)[-1]:.1f} um")

.. testoutput::

   Crystal count: 1.000e+09 to 2.000e+09 #/m**3
   L43: 100.0 to 79.4 um

Cases 4, 5 and 8 of the paper's table let a crystal exactly at the threshold break.
Here it does not, as in the paper's other seven cases, where the primary particle
itself stays whole. That one size is the only difference.

.. warning::

   The frequency jumps to zero at ``minimum_size``. If a node settles there, breakage
   switches on and off from step to step and adaptive solvers can stall or go wrong.
   Of the 30 runs under `Comparison with the published results`_, LSODA did not finish
   10 in a minute, and 7 of those finished once the threshold was removed.
   **Loosening the tolerance is not a safe way out.** On case 1, BDF at
   ``rtol=1e-4`` finished in under a second with L43 at 2271 instead of 4.2, and raised
   no error; at ``rtol=1e-6`` it gave 4.39. See issue
   `#564 <https://github.com/PharmaPy-org/PharmaPy/issues/564>`__.

   When you use a minimum size, compare two tolerances, and look for a node on the
   threshold in ``result.quadrature_nodes``.

A batch with everything on
--------------------------

This batch has linear size-dependent growth, constant agglomeration and symmetric
breakage. The kernels act within the solid whatever the liquid does, so they keep
running after the supersaturation is gone.

.. testcode::

   KERNEL = 1e-12  # [m**3/s]
   SELECTION = 1e-3  # [1/s]
   RUNTIME = 600.0  # [s]

   vessel = crystallizer(
       BatchCrystallizer, seed_moments,
       growth=(0.02, 0.0, 1.0, 1.0, 0.02),  # k_g, E_g, g, alpha, beta
       agglomeration=ConstantKernel(KERNEL),
       breakage=ConstantSelection(SELECTION),
       fragments=BinaryFragmentation())
   vessel.solve_unit(runtime=RUNTIME, verbose=False)

   result = vessel.result
   moments = np.asarray(result.mu_n_solid0)  # [um**k/m**3 of slurry], one row per time
   nodes = np.asarray(result.quadrature_nodes)  # [um], one row per time
   weights = np.asarray(result.quadrature_weights)  # [#/m**3 of slurry], one row per time
   print(f"Crystal count: {moments[0, 0]:.3e} to {moments[-1, 0]:.3e} #/m**3")
   print(f"L43: {np.ravel(result.L43)[0]:.1f} to {np.ravel(result.L43)[-1]:.1f} um")
   print("Final nodes [um]:", np.round(nodes[-1], 1))
   print("Final weights [#/m**3]:", np.array2string(weights[-1], precision=3))

.. testoutput::

   Crystal count: 9.000e+09 to 3.490e+09 #/m**3
   L43: 69.1 to 235.1 um
   Final nodes [um]: [102.6 192.7 304. ]
   Final weights [#/m**3]: [1.143e+09 1.956e+09 3.904e+08]

Reading the results
-------------------

A solve adds these to ``vessel.result``, with one row or value per time.

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Attribute
     - Units
     - Meaning
   * - ``mu_n_solid0``
     - [um**k/m**3 of slurry]
     - The moments, shape ``(times, 2N)``
   * - ``quadrature_nodes``
     - [um]
     - The node sizes in ascending order, shape ``(times, N)``
   * - ``quadrature_weights``
     - [#/m**3 of slurry]
     - The node weights, shape ``(times, N)``
   * - ``L32``
     - [um]
     - :math:`\mu_3 / \mu_2`, the surface-weighted mean size
   * - ``L43``
     - [um]
     - :math:`\mu_4 / \mu_3`, the volume-weighted mean size. Only from three
       nodes up, since two nodes do not carry :math:`\mu_4`

Nodes and weights read NaN where the moments hold fewer than N distinct sizes,
such as a monodisperse seed. Mean sizes read NaN when there are no crystals.

The nodes are a quadrature rule, not a size distribution. They reproduce the
moments, but they do not say how many crystals are at a given size, and a plot of
them is not a CSD.

Writing your own kernel
-----------------------

Any callable of the right shape works. A kernel takes the two size arrays [um] and
a ``KernelConditions``, whose fields are ``temp`` [K], ``supersaturation``,
``growth_rate`` :math:`G_0` [um/s] and ``shape_factor`` :math:`k_v` [-], and it
must broadcast over the two arrays. A selection function is
``selection(size, conditions)``, and a fragment distribution is any object with a
``moments(parent_sizes, orders)`` method.

A kernel or selection function that is infinite at zero size should set
``needs_finite_nuclei = True``, and the population balance then asks for ``rad``
above zero. The law below is an illustration, not a fitted model.

.. testcode::

   def supersaturation_kernel(size_i, size_j, conditions):
       """Rate coefficient [m**3/s], proportional to the supersaturation."""
       return KERNEL * conditions.supersaturation * np.ones_like(size_i * size_j)


   custom = crystallizer(BatchCrystallizer, seed_moments,
                         agglomeration=supersaturation_kernel)
   custom.solve_unit(runtime=60.0, verbose=False)
   custom_count = np.asarray(custom.result.mu_n_solid0)[:, 0]  # [#/m**3 of slurry]
   print(f"Crystal count: {custom_count[0]:.3e} to {custom_count[-1]:.3e} #/m**3")

.. testoutput::

   Crystal count: 9.000e+09 to 5.169e+09 #/m**3

Semibatch, continuous and flowsheets
------------------------------------

Set ``Inlet`` on a semibatch or continuous unit as usual. In a flowsheet, a unit's
crystals enter the next one through its ``Inlet``, and the receiving unit's kernels
keep acting on them.

.. testcode::

   sheet = SimulationExec(path_data, flowsheet="U01 --> U02")
   sheet.U01 = crystallizer(ContinuousCrystallizer, seed_moments, feed=1e-3)  # [kg/s]
   sheet.U02 = crystallizer(SemiBatchCrystallizer, np.zeros(6), feed=1e-3)  # [kg/s]
   run = {"runtime": RUNTIME, "verbose": False}
   sheet.SolveFlowsheet(kwargs_run={"U01": run, "U02": run}, verbose=False)


   def held(unit):
       """Crystals in a unit over the solve [#]."""
       return (np.asarray(unit.result.mu_n_solid0)[:, 0]
               * np.ravel(unit.result.vessel_vol))


   print(f"Left U01: {held(sheet.U01)[0] - held(sheet.U01)[-1]:.5e} crystals")
   print(f"In U02: {held(sheet.U02)[-1]:.5e} crystals")

.. testoutput::

   Left U01: 4.33771e+06 crystals
   In U02: 4.33771e+06 crystals

These handoffs are checked before anything is lost, and the refused ones stop
with a message that says what to do.

.. list-table::
   :header-rows: 1
   :widths: 50 16 34

   * - Into a QMOM unit
     - Result
     - Notes
   * - Feed from a QMOM or moment-form unit, same number of moments
     - Accepted
     - Also from QMOM into a moment-form unit
   * - Feed with another number of moments
     - Refused
     - The missing moments cannot be recovered
   * - Feed carrying a size distribution (FVM)
     - Refused
     - In other units such a feed is lost silently: issue
       `#560 <https://github.com/PharmaPy-org/PharmaPy/issues/560>`__
   * - Charge of crystals
     - Refused
     - Use the ``Inlet``. See issue
       `#559 <https://github.com/PharmaPy-org/PharmaPy/issues/559>`__
   * - Charge of liquid, from either stack
     - Accepted
     - The vessel keeps its crystals and kernels
   * - Crystals from a legacy unit
     - Refused
     - They would arrive without the vessel's model. Hand over the liquid only

Going the other way, ``to_legacy_state`` expresses the moments as totals for a
legacy solid. That conversion is tested, but a QMOM unit feeding a legacy unit has
not been run through a flowsheet.

Assumptions and limits
----------------------

- **Refactored crystallizers only.** The legacy ``BatchCryst``, ``MSMPR`` and
  ``SemibatchCryst`` do not have QMOM.
- **Isothermal so far.** The tests run isothermal vessels, and agglomeration and
  breakage add no heat term.
- **Dissolution** is size independent, and crystals do not disappear at
  :math:`L = 0`, as in the moment form.
- **Nuclei** are born at size ``rad``, which defaults to 0. Kernels and selection
  functions see each node at no less than ``rad``.
- **Nodes below zero.** A node that integration noise pushes below zero is clipped
  to zero and keeps its weight.
- **Nucleation onset.** The crystal count right after onset can follow the
  integrator's steps with some secondary-nucleation laws: issue
  `#529 <https://github.com/PharmaPy-org/PharmaPy/issues/529>`__. The constructor
  warns when a kinetics law is exposed to this.
- **Fitting.** Kernel constants are not ``CrystKinetics`` parameters, and the
  refactored crystallizers have no ``paramest_wrapper``, so fitting them would need
  your own model callback. That has not been tried.

Verification
------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Behaviour
     - Checked against
   * - Moment inversion
     - Known populations of one to three sizes, including rounding near
       degenerate sets (``tests/test_moment_inversion.py``)
   * - Kernel, selection and fragment formulas
     - The published forms, volume conservation and scaling
       (``tests/test_crystallization_kernels.py``)
   * - Nucleation and growth
     - Method of characteristics, and the steady moments of a continuous
       crystallizer (``tests/test_qmom_population_balance.py``)
   * - Agglomeration
     - Exact zeroth moments of the constant, sum and product kernels (Scott, 1968),
       and collision counting
   * - Breakage
     - Exact moments under constant selection for every self-similar distribution,
       and the linear zeroth moment of volume-linear selection (Ziff and McGrady,
       1985)
   * - Agglomeration with breakage
     - The logistic balance of the two in a batch, and the steady crystal count of
       a continuous crystallizer
   * - Minimum breakage size
     - The exact moments of seeds that break once and whose fragments stay whole
   * - Published numbers
     - Cases 2, 4, 7 and 8 of Marchisio et al. (2003) with three nodes, through
       ``BatchCrystallizer`` (``tests/test_qmom_marchisio_benchmark.py``)
   * - Mass and crystal balances
     - Total mass in a semibatch with every mechanism on, and crystals leaving one
       unit against those arriving in the next
   * - This page
     - ``tests/test_qmom_guide.py`` runs every block above and checks the printed
       numbers against closed forms

Comparison with the published results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Eq. 16 and Tables 1 to 3 of Marchisio et al. (2003) agree with the code. The ten
cases of their Table 4 were also rerun with two, three and four nodes, and the
asymptotes compared with their Tables 5 and 6. A run counts as matching when
:math:`m_0` and :math:`d_{43}` are both within one unit of the last printed digit.
**17 of the 30 runs match and 13 do not.**

.. list-table::
   :header-rows: 1
   :widths: 8 24 20 18 10 10 10

   * - Case
     - Kernel
     - Selection
     - Fragments
     - 2 nodes
     - 3 nodes
     - 4 nodes
   * - 1
     - Constant
     - Constant
     - Symmetric
     - Yes
     - Yes
     - Yes
   * - 2
     - Sum
     - Power law
     - Symmetric
     - **No**
     - Yes
     - Yes
   * - 3
     - Hydrodynamic
     - Exponential
     - Symmetric
     - Yes
     - Yes
     - **No**
   * - 4
     - Hydrodynamic
     - Exponential
     - 1:4
     - Yes
     - Yes
     - Yes
   * - 5
     - Hydrodynamic
     - Exponential
     - Erosion
     - **No**
     - **No**
     - **No**
   * - 6
     - Hydrodynamic
     - Exponential
     - Uniform
     - Yes
     - Yes
     - Yes
   * - 7
     - Hydrodynamic
     - Exponential
     - Uniform
     - Yes
     - Yes
     - Yes
   * - 8
     - Hydrodynamic
     - Power law
     - Erosion
     - **No**
     - Yes
     - **No**
   * - 9
     - Hydrodynamic
     - Power law
     - Parabolic
     - **No**
     - **No**
     - **No**
   * - 10
     - Differential force
     - Power law
     - Uniform
     - **No**
     - **No**
     - **No**

**The cause of the 13 mismatches is not known.** They range from 0.4 % to more than a
factor of five, the largest on case 10. These checks did not explain them:

- Halving the time step changed none of the 30 results.
- A separate evaluation of Eq. 16, solved for its steady states, gave the code's values
  for cases 2, 5 and 8 with two and three nodes, case 4 with three, and case 10 with
  three and four. It found no physical root for the other runs, which may be a limit
  of the search.
- Seven of the mismatching runs, started on the paper's numbers, left them and went
  toward the code's values. Case 10 with two nodes was still moving at the end of the
  run, so its value is not final.

For case 7 with three nodes only :math:`d_{43}` was compared. Table 5 prints
:math:`m_0 = 0.0178`, out of line with the 0.0058 and 0.0053 it gives for two and four
nodes; the run gives 0.0054.

These kernels have **no matching published number**, so they rest on the formula and
conservation checks above:

- ``DifferentialSedimentationKernel``: only in case 10, which does not match.
- ``ParabolicFragmentation``: only in case 9, which does not match.
- ``BrownianKernel``: in no case.
- ``ProductKernel``: not in the paper.

``ErosionFragmentation`` matches only in case 8 with three nodes.

The table comes from one run on 6 Nov 2026, using fixed-step Runge-Kutta on the
production rate functions, and it is not in the test suite. The test suite holds only
the four cases listed under `Verification`_.

References
----------

- Golub, G. H.; Welsch, J. H. (1969). Calculation of Gauss quadrature rules.
  *Mathematics of Computation*, 23, 221-230.
  https://doi.org/10.1090/S0025-5718-69-99647-1
- Marchisio, D. L.; Vigil, R. D.; Fox, R. O. (2003). Quadrature method of moments
  for aggregation-breakage processes. *Journal of Colloid and Interface Science*,
  258, 322-334. https://doi.org/10.1016/S0021-9797(02)00054-1
- McGraw, R. (1997). Description of aerosol dynamics by the quadrature method of
  moments. *Aerosol Science and Technology*, 27, 255-265.
  https://doi.org/10.1080/02786829708965471
- Scott, W. T. (1968). Analytic studies of cloud droplet coalescence I. *Journal of
  the Atmospheric Sciences*, 25, 54-65.
- Smoluchowski, M. (1917). Versuch einer mathematischen Theorie der
  Koagulationskinetik kolloider Lösungen. *Zeitschrift für Physikalische Chemie*, 92,
  129-168. https://doi.org/10.1515/zpch-1918-9209
- Wheeler, J. C. (1974). Modified moments and Gaussian quadratures. *Rocky Mountain
  Journal of Mathematics*, 4, 287-296. https://doi.org/10.1216/RMJ-1974-4-2-287
- Ziff, R. M.; McGrady, E. D. (1985). The kinetics of cluster fragmentation and
  depolymerisation. *Journal of Physics A*, 18, 3027-3037.
  https://doi.org/10.1088/0305-4470/18/15/026
