Scheduling a simulated reference batch
=======================================

A State-Task Network (STN) represents material inventories as states and
processing operations as tasks that occupy equipment. This example solves two
isothermal batch reactors through ``SimulationExec`` and exports the solved
batch sequence to an indexed STN JSON instance. The reaction and timing are
synthetic demonstration inputs. The property database ships with the checkout.

From the checkout root, use the locked Assimulo environment to run both
reactors and write the instance.

.. code-block:: bash

   pixi run -e assimulo python -m doc.online_docs.examples.stn_batch_example batch-stn.json

The first reactor processes a batch for 25 s and the second for 43 s. The
20 s grid therefore reserves two and three periods, respectively. The feed,
intermediate, and product are complete liquid mixtures, each approximately
0.782 kg. The storage limits and feed inventory each allow one batch. A
withdrawal of one product batch at period five requires the first reactor to
start at period zero and the second to start at period two. The buffer holds
material which the downstream task can take at the same grid point at which
it is released. Transfer, startup, and turnaround have zero length in this
example.

Data available to the scheduler
-------------------------------

``PharmaPy.Scheduling.export_batch_stn`` takes a solved ``SimulationExec``
and a mapping from selected unit names to ``BatchTask`` declarations. A
``BatchTask`` names one resource and the amounts consumed and produced per
reference batch [kg], by material state. Use the same state on either side
of a direct edge between selected units to represent a handoff. If you omit
an intermediate unit, specify how its transfers and timing enter the scheduling
model. Export immediately after a successful solve. The physical durations
span the full retained result. For a reactor continued across solves, declare
the masses for that entire span or start from a fresh batch.

Give tasks the same equipment identifier if they compete for the same unit.
Supply finite physical storage limits, initial inventory, and product demand
for the scenario. Document how you obtained these quantities and which material
streams the scheduling boundary covers.

The output uses ``I`` for tasks, ``J`` for equipment, and ``K`` for material
states. It records ``delta`` and ``tau_p`` in seconds, ``lastT`` as the final
period, and ``tau`` as the positive number of periods reserved for a task.
The grid includes period zero. The indexed tables use ``index`` and ``value``
records. Task-material, task-equipment, and material-period indices are pairs.
Storage and initial inventory have a single string state index.

``rho_minus`` and ``rho_plus`` are kilograms per kilogram of reference input
batch. ``beta_min`` and ``beta_max`` are both equal to the reference batch
mass [kg]. Include every material input. If you omit a discharge or loss from
the scheduling inventory, the sum of the tracked output coefficients is less
than one. If numerical or measurement uncertainty exceeds arithmetic roundoff,
pass a justified absolute allowance [kg] as ``mass_balance_tolerance_kg``.
The export keeps the declared material yields. ``gamma``, ``S0``, and ``demand``
are storage limits, initial
inventory, and scheduled withdrawals [kg]. Holding the batch size fixed keeps
the scheduling instance within the simulated operating point. The scheduler
accounts for consumption at start and production at completion. Use this
approximation only when the batch boundary is an appropriate representation
of the transfers. A model with continuous releases needs a different task
representation.

Interface decisions for the next prototype
------------------------------------------

* `Declarative flowsheets <https://github.com/PharmaPy-org/PharmaPy/issues/348>`_
  can supply the executed graph and use the same unit names in the scheduling
  declarations. Authoring remains upstream of the export. Scheduling-specific
  material identities and equipment assignments are explicit case inputs.
* `Cost reporting <https://github.com/PharmaPy-org/PharmaPy/issues/151>`_
  should provide a sourced cost per task start before populating the STN
  ``cost`` table. The example leaves the table empty for a feasibility
  demonstration. Keep material and utility amounts in their reported bases
  until a cost model maps them to a per-start amount.
* For the `full scheduling export <https://github.com/PharmaPy-org/PharmaPy/issues/367>`_,
  retain the declared material amounts and physical durations for RTN and add
  explicit equipment acquisition and release as resource changes. For the
  `bioconjugation sequence <https://github.com/PharmaPy-org/PharmaPy/pull/530>`_,
  connect reduction, conjugation, desalting, and purification through distinct
  intermediate states. Use the case's total solution masses [kg] to derive
  coefficients, and record each occupancy and storage limit before choosing
  the grid.
  Different equipment-dependent recipes, distributed transfers, and changeovers
  require their own case data and verification.
