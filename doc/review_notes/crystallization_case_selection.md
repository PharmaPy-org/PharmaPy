# Crystallization case selection, data provenance and PAT observables (#358)

This note records the case selected for the December crystallization
demonstrator ([#358](https://github.com/PharmaPy-org/PharmaPy/issues/358),
parent [#153](https://github.com/PharmaPy-org/PharmaPy/issues/153)): the
process, the data, the property sources, the model path and its assumptions, the
parameters to estimate, and one measurement equation per PAT channel. It was
written against `master` at `7fe3045`. It fixes the equations before they are
coded. **It does not claim a calibration: no model has been fitted to these
data yet.**

The selection stands for now. It may be replaced or joined by data from other
compounds if those prove more useful; the readers and the measurement
equations are written to carry over.

## 1. Ownership

| Role | Owner |
| --- | --- |
| Implementation | Luisdomingo Guzman Julio (`lguzmaj`) |
| WP2 workstream owners (from #153) | Zachary Hillman (`ZacharyHillman18`), Luisdomingo Guzman Julio (`lguzmaj`) |
| Domain review | Not yet named. To be recorded here before #358 closes. |

## 2. Case and data

Seeded batch cooling crystallization of caffeine from water, four runs in a
2 x 2 design of seed load and final cooling rate, measured in the Nagy group
(Purdue) in October and November 2026. The group has agreed that the data can
be shared in this repository.

| Quantity | Value | Basis |
| --- | --- | --- |
| Water | 0.350 kg | Lab recipe sheet (target, not a weighed mass) |
| Caffeine load | 16.5434e-3 kg, 0.04727 kg per kg water | Lab recipe sheet (target) |
| Seed, high and low | 0.5713e-3 kg and 0.1143e-3 kg | Lab recipe sheet: 5 % and 1 % of its theoretical yield |
| Temperature program | 20 C, heat to 45 C, hold, cool to 37 C, hold and seed, cool to 20 C at 0.1 or 0.5 C/min, hold | Setpoint program files |
| Seeding time | About 8,620 to 8,760 s on the run clock | FBRM spike of each run; no process note records it |

Instruments: ReactIR 700 with a DiComp probe (about one spectrum per minute),
ParticleTrack G400 FBRM (one sample every 10 s), ReactRaman 802L, and a
thermostat log every 2 s. The raw files stay outside the repository. The
reduced tables, their layouts, the run description and the reduction script
are in [`examples/pat/caffeine`](../../examples/pat/caffeine/README.md).

| Run | Seed | Cooling | IR concentration trace | FBRM | Use |
| --- | --- | --- | --- | --- | --- |
| `high_seed_slow_cool` | high | 0.1 C/min | Complete | Complete | Fit |
| `low_seed_fast_cool` | low | 0.5 C/min | Complete | Complete | Fit |
| `low_seed_slow_cool` | low | 0.1 C/min | About 20 spectra after seeding | Complete | Validation |
| `high_seed_fast_cool` | high | 0.5 C/min | Not usable: the probe was not in proper contact with the liquid | Complete | Excluded until the lab explains the run |

The excluded run also shows poor temperature control, so its logged
temperature may not be the liquid temperature.

## 3. Species and properties

| Property | Value | Source |
| --- | --- | --- |
| Caffeine molar mass | 194.19 g/mol | NIST Chemistry WebBook |
| Solid that crystallizes | Caffeine hydrate, needle shaped | Nagy et al. (2007): anhydrous caffeine added to water converts to hydrate needles 5 to 12 minutes after the addition, depending on agitation. `high_seed_slow_cool` records the same event at 20 C: within a quarter of an hour of the addition the fine FBRM counts rise eightfold and the IR band falls from 0.15 to 0.097. To be confirmed with the Raman spectra. |
| Hydrate unit cell | a = 14.8(1), b = 16.7(1), c = 3.97(3) angstrom, beta = 97(5) degrees, Z = 4, volume 973.9 cubic angstrom | Sutor (1958), from Crystallography Open Database entry 5000108 |
| Caffeine per crystal volume, `rho_c` | 1324 kg caffeine per m3 of hydrate crystal, about +/- 21 | Derived: Z caffeine molecules per cell volume. The cell uncertainties give 1.6 %. |
| Volume shape factor, `k_v` | pi/6 | Definition, see section 5 |

The model solid is counted as caffeine, not as hydrate. `rho_c` is the mass of
caffeine in a unit volume of hydrate crystal, so `rho_c * k_v * mu_3` is the
crystallized caffeine mass whatever the water content of the hydrate.

The property database of the case is
[`compounds.json`](../../examples/pat/caffeine/compounds.json); each entry
states its sources. It holds only what the two crystallizers use here:

| Property | Value | Source and limit |
| --- | --- | --- |
| Water density | 996.6 kg/m3 | NIST fluid properties at 300 K and 1 atm; 998.8 to 989.4 between 290 and 320 K. The models take it as constant. |
| Water heat capacity | 75.3 J/mol/K | Same NIST table |
| Dissolved caffeine, as a liquid density | 1344 kg/m3 | Molar mass over the limiting apparent molar volume in water, 144.45 cm3/mol at 303.15 K (Vraneš et al. 2022); 1359 to 1331 between 293 and 313 K |
| Caffeine solid heat capacity | 232 J/mol/K | Bothe and Cammenga (1979) through NIST, anhydrous caffeine at 298 K. Cesàro and Starec (1980) give 173. |
| Dissolved caffeine heat capacity | 232 J/mol/K | Assumption: equal to the solid. No measurement was found. |
| Water vapor pressure | Antoine constants | NIST, from Stull (1947), 256 to 373 K |

The heat capacities enter only the heat duty that the legacy unit reports. The
vapor pressure is never evaluated; the refactored liquid phase requires it to
exist when it is built.

## 4. Solubility

Basis: `X` [kg caffeine per kg water]. The kinetics receive the mass fraction
`w = X / (1 + X)` [kg caffeine per kg solution], which is exact for a
two-component liquid.

The curve is built from the lab's own data and checked against literature.

1. The load dissolves completely on heating between 39.9 and 41.4 C (last FBRM
   sample above 100 counts, three runs, heating at about 0.5 C/min).
2. The saturated slurry at 20 C at the start and the end of each run, through
   the IR calibration of section 7.
3. The heating ramp of each run: every spectrum taken while solid is present
   gives the dissolved caffeine at its temperature. Dissolution lags the
   heating, so these 177 points between 19.6 and 41 C are a lower bound.

The lab curve is the fit `ln(X) = intercept + slope / T` to the points of
item 3 over the three runs, under each calibration treatment of section 7. The
constants are in
[`solubility.json`](../../examples/pat/caffeine/solubility.json).

| Source | 20 C | 25 C | 37 C | 40 C | 45 C | Saturation temperature of the load |
| --- | --- | --- | --- | --- | --- | --- |
| Lab curve, runs share the calibration slope | 1.85 | 2.34 | 3.98 | 4.51 | outside the fit | 41.1 C |
| Lab curve, runs share the water baseline | 1.74 | 2.23 | 3.94 | 4.52 | outside the fit | 41.0 C |
| Nagy et al. (2007), van't Hoff fit | 1.45 (extrapolated) | 1.92 (extrapolated) | 3.63 | 4.23 | 5.41 | 42.2 C |
| Vraneš et al. (2022), Table 1 | 1.63 | 2.07 | 3.82 (interpolated) | 4.39 | not measured | about 41.6 C (extrapolated) |
| Shalmashi and Golmohammad (2010), Table 1 | not measured | 2.27 | | 4.73 | 8.6 | 40 C |
| Zhong et al. (2017), Apelblat fit | 1.53 | 1.78 | 2.55 | 2.78 | 3.22 | about 58 C |

Values are in g caffeine per 100 g water. The Nagy et al. fit is
`ln(X) = -dH/(R T) + dS/R` with `dH` = 40.8 kJ/mol, `dS` = 104.0 J/mol/K and
`R` = 8.31 J/mol/K, measured from clear points between about 33 and 52 C
(range read from their Figure 13).

- Nagy et al. and Shalmashi and Golmohammad agree with anchor 1 within about
  2 K. Either can serve as the cross-check between 33 and 45 C.
- Zhong et al. is not used: with their curve the load could not dissolve at
  45 C, and the runs show that it does.
- Vraneš et al. measured 20 to 40 C by weighing the residue of filtered
  saturated solutions, five repeats per temperature, with a stated uncertainty
  of 0.001 in mass fraction (about 0.1 g per 100 g water). They did not
  characterize the solid.
- No source measured 20 C with the hydrate characterized. From the lab data,
  item 2 gives 1.6 to 2.0 g per 100 g water (section 7).
- Near 40 C the lab curve agrees with Vraneš et al. within 3 % and lies
  between Nagy et al. and Shalmashi and Golmohammad.
- Between 20 and 30 C the lab curve is 13 to 16 % above Vraneš et al. The lab
  points should be a lower bound, so the likely cause is the IR calibration,
  whose temperature term is measured at 37 and 45 C only. The fit of #359
  must be repeated with the Vraneš et al. values to show what this changes.
- The three runs scatter 4.5 % about the pooled curve when they share the
  calibration slope and 7.8 % when they share the water baseline. The
  solubility is one physical curve, so this favors a shifted baseline as the
  cause of the difference between runs. It is an indication, not a proof.

## 5. Model path and assumptions

The December model is the legacy `BatchCryst` with the method of moments. It
is the only crystallizer path whose parameter sensitivities are validated on
`master` (`tests/test_calibration_validation_crystallizer.py`).

States: total moments `mu_0` to `mu_3` of the crystal population (order n in
[um**n] in the solver, [m**n] in results), liquid mass concentrations of
caffeine and water [kg/m**3], and liquid volume [m**3].

| Assumption | Reason and limit |
| --- | --- |
| The measured temperature drives the model as a prescribed control. No energy balance is solved. | The reactor temperature is logged every 2 s. Whether the "External T" sensor sits in the liquid is not yet confirmed. |
| One characteristic length `L`, defined as the diameter of the sphere with the crystal's volume, so `k_v` = pi/6 exactly. | The needles are not resolved. Growth rate and sizes are volume-equivalent values, not needle lengths. This holds while the crystals keep their shape as they grow. |
| The solid is counted as caffeine. Water taken up by the hydrate is neglected. | At most 1.06 g of water for the full theoretical yield, 0.3 % of the solvent (one water per caffeine, the cell formula). |
| The model starts at the seeding spike: all caffeine of the load dissolved, solid equal to the seed mass. | In the runs used, FBRM counts stay at the background level from the clear point to the seeding spike. The seed was slurried in liquid taken from the vessel, which is supersaturated and cools, so the returned seed may exceed the weighed mass. |
| The seed mass is counted as caffeine. | The lab believes the seed was as-received caffeine, which is not recorded. Nagy et al. (2007) indicate that such a seed turns into hydrate within minutes of meeting the liquid. If it was weighed as hydrate instead, its caffeine content is up to 8.5 % lower. |
| The seed is one size, `L_seed`. | No seed size was measured. The nominal `L_seed` is 31.6 um, the geometric mean of 10 and 100 um, the chord band that holds about two thirds of the FBRM counts at the seeding spike. It is scanned between 10 and 150 um, not estimated. |
| Well mixed, no agglomeration, no breakage, size-independent growth. | Counts fall and chords grow for hours at 20 C after the crystallization. The fit window ends before that drift. |

## 6. Kinetics and parameters to estimate

Driving force: `sigma = (w - w_sat(T)) / w_sat(T)` [-], the `relative` option
of `CrystKinetics`.

- Growth: `G = k_g * sigma**g` [um/s].
- Secondary nucleation: `B = k_s * sigma**s_1 * (k_v * mu_3 / V)**s_2`
  [#/m**3/s], where `k_v * mu_3 / V` is the solid volume fraction of the
  slurry [-].
- Primary nucleation and dissolution are inactive. The solution stays clear
  for 18 to 20 minutes at 37 C before seeding, and Nagy et al. (2007) report a
  wide metastable zone for caffeine in water. The fit window starts
  supersaturated and only cools.

| Parameter | Unit | Treatment |
| --- | --- | --- |
| `k_g`, `g` | um/s, - | Estimated |
| `k_s`, `s_1` | #/m**3/s, - | Estimated |
| `s_2` | - | Fixed at 1: nucleation proportional to the solid volume present. Assumption. |
| Activation energies | J/mol | Fixed at 0. The window spans 20 to 37 C, and temperature and supersaturation change together along a cooling ramp. |
| `k_F` (section 7) | counts * m**(3 - j) | Estimated |
| `L_seed` | um | Scanned, see section 5 |

Which of these the data support is decided by the derivative check and the
identifiability report of #359, not here.

## 7. Measurement equations

Every prediction is evaluated at the measurement times of its own channel.

**Dissolved concentration (IR).** Band height `h` [-]: absorbance at
1700 cm-1 above the line through 1760 and 1720 cm-1.

    h = h_0 + (a_0 + a_1 * (T - T_ref)) * X,    T_ref = 310.15 K

- `h_0` is the band height of water alone: 0.0666, the median of the eleven
  spectra of `high_seed_slow_cool` taken before the caffeine was added. They
  drift from 0.0651 to 0.0681, which bounds `h_0`. The other runs have no
  spectrum of water alone.
- `a_0` and `a_1` come from the two dissolved plateaus of each run, at 45 C
  and 37 C, where `X` is the known load.
- The model predicts `X = c_caffeine / c_water` from its two liquid
  concentrations, so no density enters.
- Assumptions: the band height is linear in `X`, and the temperature term
  found between 37 and 45 C holds down to 20 C. The 20 C slurry of each run
  tests both: its start and end values must agree.

Median band height in each window of the reduced tables (scatter 0.0002 to
0.0004):

| Run | Dissolved, 45 C | Dissolved, 37 C | Slurry at 20 C, start | Slurry at 20 C, end |
| --- | --- | --- | --- | --- |
| `high_seed_slow_cool` | 0.1351 | 0.1371 | 0.0967 | 0.0987 |
| `low_seed_slow_cool` | 0.1308 | 0.1325 | 0.0897 | 0.0916 |
| `low_seed_fast_cool` | 0.1308 | 0.1329 | 0.0910 | 0.0933 |

Two things are unresolved and enter the calibration uncertainty:

- The dissolved plateaus differ by 0.004 between runs at the same nominal
  load, 6 % of the signal above water. The data cannot tell a shifted
  baseline from a different slope. Both treatments are carried.
- The slurry reads 0.0019 to 0.0023 higher at the end of every run than at
  its start. Either one of the two is not at equilibrium, or solid on the
  probe window adds to the band.

Over the three runs, both treatments and both ends, the 20 C slurry gives 1.6
to 2.0 g caffeine per 100 g water. That is above the extrapolated literature
values of section 4 (1.45 and 1.53). The constants are in
[`band_calibration.json`](../../examples/pat/caffeine/band_calibration.json).

**Solid concentration.** `m_solid = rho_c * k_v * mu_3` [kg caffeine], with
`mu_3` in [m**3]. Conservation check for every output row: dissolved caffeine
plus `m_solid` equals load plus seed.

**Particle count (FBRM).** Total counts between 1 and 1000 um:

    N_F = k_F * mu_j / V_slurry        [counts]

- `mu_j / V_slurry` is the j-th moment per slurry volume [m**(j - 3)].
- `k_F` is an instrument constant. It absorbs the optics, the crystal shape
  and the unknown aspect ratio. If the crystals keep their shape, every chord
  scales with `L`, so the moment order does not depend on the shape.
- `j` is not known beforehand. `j = 1` (encounters proportional to number
  times a length) and `j = 2` (proportional to projected area) are both
  evaluated. The choice is reported with the fit and tested on the validation
  run. This is a modeling assumption.
- FBRM is not converted to a size distribution. For needles the chord depends
  on the aspect ratio, which was not measured.

**Published without comparison to data.** Sauter mean size `mu_3 / mu_2` [m],
on the volume-equivalent length. The measured mean chord is reported, not
fitted.

**Raman.** One qualitative job: compare the initial solid, the final solid
and the dissolved state to state which solid form is present.

## 8. PharmaPy path and linked defects

Estimation runs through `BatchCryst.paramest_wrapper` and
`ParameterEstimation` with `Experiment` and `Measurement` objects, one
experiment per run, and is recorded with `CalibrationRecord`. The measurement
equations wrap `paramest_wrapper` from outside.

| Issue | Effect on this case | Handling |
| --- | --- | --- |
| [#222](https://github.com/PharmaPy-org/PharmaPy/issues/222) | An unmasked sensitivity solve stalls | Fit the subset through `mask_params` |
| [#314](https://github.com/PharmaPy-org/PharmaPy/issues/314) | Heat duty is NaN at repeated sample times | No replicate times within a channel |
| [#535](https://github.com/PharmaPy-org/PharmaPy/issues/535) | A masked fit cannot be recorded with a partial name list | The record names all 13 kinetic parameters |
| [#494](https://github.com/PharmaPy-org/PharmaPy/issues/494) | `param_wrapper` fails with the finite-volume method | Moments only, and no `param_wrapper` |
| [#537](https://github.com/PharmaPy-org/PharmaPy/issues/537) | A grid at the charge time only is refused | The first sample is after the seeding time |
| No issue | In CVode's default output mode the legacy unit leaves out requested times inside its last internal step: 3 of 329 spectrum times on this case | Solve with `report_continuously`, and check that every requested time came back |

The legacy volume equation carries a note that it is not consistent with
`mu_3`. On a synthetic seeded cooling case
(`tests/test_slurry_measurements_models.py`) the legacy unit conserves the solute
to 2e-7 of the charge, but its liquid volume ends 0.6 % below the volume that
its own ideal-mixing density gives for the same contents. The predictions then
differ from the refactored `BatchCrystallizer` by up to 1 %. This is a known
limitation, not a new defect: the legacy crystallizers treat the liquid
density as constant, and the refactored ones do not.

The caffeine case runs on both units through
[`batch_cooling.py`](../../examples/pat/batch_cooling.py), with the starting
kinetics of the dataset description, which are not fitted values. Checked on
`high_seed_slow_cool` at 52 spectrum times
(`tests/test_pat_batch_cooling_models.py`):

- One row per requested time, also for all 3422 FBRM sample times.
- The solute in the vessel drifts by 4e-8 of the charge on the legacy unit
  and 3e-7 on the refactored one.
- The predictions of the two units agree within 0.9 %. The refactored unit
  ends at saturation; the legacy unit ends 0.7 % below it, the effect of its
  constant liquid density on this case.
- Doubling the growth constant lowers the dissolved caffeine during the
  desupersaturation by 9 %, a hundredfold nucleation constant gives 8.5
  times the crystals, and repeating the first parameters gives the first
  result bit for bit.
- One simulation of the run takes a few seconds on either unit.

## 9. Open inputs

1. Weighed masses of water, caffeine and seed in each run.
2. Confirmation that the seed was as-received caffeine, and the caffeine
   grade.
3. Reactor working volume, impeller and speed, and where "External T" is
   measured.
4. The cause of the probe and temperature problems in `high_seed_fast_cool`.
5. Any offline result: final crystal size or images, powder diffraction,
   yield.
6. The temperature above which the hydrate is no longer the stable form in
   water. It is reported by Bothe and Cammenga (1980), which has not been read
   yet. The runs stay at or below 45 C.

## 10. References

- Nagy, Z. K., Gillon, A. L., Steele, G., Makwana, N. and Rielly, C. D. (2007).
  Using process analytical technology for in situ monitoring of the
  polymorphic transformation of organic compounds. 8th International IFAC
  Symposium on Dynamics and Control of Process Systems, Preprints Vol. 3,
  133-138. Equation 2 and Figures 7, 8, 13 and 14.
- Shalmashi, A. and Golmohammad, F. (2010). Solubility of caffeine in water,
  ethyl acetate, ethanol, carbon tetrachloride, methanol, chloroform,
  dichloromethane, and acetone between 298 and 323 K. Latin American Applied
  Research 40, 283-285. Table 1.
- Sutor, D. J. (1958). The structures of the pyrimidines and purines. VII. The
  crystal structure of caffeine. Acta Crystallographica 11, 453-458.
  doi:10.1107/S0365110X58001286. Cell read from Crystallography Open Database
  entry 5000108.
- NIST Chemistry WebBook, SRD 69. Caffeine (CAS 58-08-2): molar mass and
  solid heat capacities of Bothe and Cammenga (1979, J. Therm. Anal. 16, 267)
  and Cesàro and Starec (1980, J. Phys. Chem. 84, 1345). Water (CAS
  7732-18-5): fluid properties at 0.101325 MPa, and Antoine constants from
  the data of Stull (1947, Ind. Eng. Chem. 39, 517).
- Vraneš, M., Borović, T. T., Drid, P., Trivić, T., Tomaš, R. and Janković, N.
  (2022). Influence of sodium salicylate on self-aggregation and caffeine
  solubility in water: a new hypothesis from experimental and computational
  data. Pharmaceutics 14, 2304. doi:10.3390/pharmaceutics14112304. Table 1
  (solubility in water) and Tables 3 and 5 (volume of dissolved caffeine).
- Zhong, J. et al. (2017). Journal of Chemical and Engineering Data,
  supporting information, Table S6. Read and not used.
- Bothe, H. and Cammenga, H. K. (1980). Composition, properties, stability and
  thermal dehydration of crystalline caffeine hydrate. Thermochimica Acta 40,
  29-39. Not read yet; see section 9.
