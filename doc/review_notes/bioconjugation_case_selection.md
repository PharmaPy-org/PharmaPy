# Bioconjugation case selection, kinetic data provenance, and PAT observables (#357)

This review note records the Month 6 (M6) checkpoint selection for Work Package
1.5 (`[WP1.5][M6] Select the bioconjugation case, data and PAT observables`,
[#357](https://github.com/PharmaPy-org/PharmaPy/issues/357)), verified against
`master` at commit `3194efd0` and accompanied by executable contract tests in
[`tests/test_bioconjugation_case_selection.py`](../../tests/test_bioconjugation_case_selection.py).

Full deliverable D3b acceptance (calibrated bioconjugation flowsheet,
identifiability/experiment-design study, and user-facing notebook) belongs to
the WP1.5 case-study tracker
([#356](https://github.com/PharmaPy-org/PharmaPy/issues/356)). This M6
checkpoint selects the representative reaction network, species, stoichiometry,
kinetic parameter provenance, mixing regime analysis, spectroscopic PAT
measurement/prediction contract, existing PharmaPy execution paths and
demonstrated blockers, Month 7–12 execution sequence, and the four-step
process-sequence inputs needed by Work Package 4
([#366](https://github.com/PharmaPy-org/PharmaPy/issues/366),
[#367](https://github.com/PharmaPy-org/PharmaPy/issues/367)). **This checkpoint
does not claim a completed empirical conjugation calibration.**

---

## 1. Governance and ownership

| Role | Owner | Affiliation / Scope |
| --- | --- | --- |
| WP1 / WP1.5 Implementation & Coordination Lead | David E. Bernal Neira (`@bernalde`) | Purdue University (SECQUOIA); reaction/PAT integration (`RxnKinetics`, `BatchReactor`, `SemibatchReactor`, `PlugFlowReactor`, `ParameterEstimation`, `CalibrationRecord`) and WP4 scheduling handoff |
| Bioconjugation & Reaction-Modeling Domain Reviewer | Daniel Casas-Orozco (`@dcasasor`) | Eli Lilly and Company; industrial bioconjugation chemistry, reaction-network specification ([#337](https://github.com/PharmaPy-org/PharmaPy/issues/337), [#339](https://github.com/PharmaPy-org/PharmaPy/issues/339)), and PAT observability review |
| WP1 / Project Co-PIs | Zoltan K. Nagy, Gintaras V. Reklaitis, Can Li (`@CanLi1`) | Purdue University (CryPTSys / SECQUOIA); PAT calibration ([#352](https://github.com/PharmaPy-org/PharmaPy/issues/352)) and WP4 STN/RTN scheduling export ([#366](https://github.com/PharmaPy-org/PharmaPy/issues/366), [#367](https://github.com/PharmaPy-org/PharmaPy/issues/367)) |

---

## 2. Literature review of bioconjugation chemistries, kinetics, and PAT monitoring

### 2.1 Comparative evaluation of industrial bioconjugation modalities

Bioconjugation manufacturing couples a targeting biomacromolecule—most commonly
an IgG1 monoclonal antibody ($\text{mAb}$, molar mass
$M_{\text{mAb}} \approx 145\text{–}150\text{ kg/mol}$)—with a small-molecule
cytotoxic linker-payload (antibody–drug conjugates, ADCs) or a functionalized
oligonucleotide/siRNA payload (antibody–oligonucleotide/RNA conjugates,
AOCs/ARCs). Across the peer-reviewed process-engineering literature and the
Purdue–Lilly 360 project scope, four reaction modalities dominate industrial
practice:

1. **Site-specific engineered-cysteine thiol–maleimide conjugation (target
   $\text{DAR} = 2$)**:
   - *Primary sources*: Andris, Seidel, and Hubbuch (2019), *J. Biotechnol.*
     306, 71–80 (`doi:10.1016/j.jbiotec.2019.09.013`); Weggen, Seidel, Bean,
     Wendeler, and Hubbuch (2023), *Front. Bioeng. Biotechnol.* 11, 1123842
     (`doi:10.3389/fbioe.2023.1123842`, `PMC10111256`); Weggen, Bean, Hui,
     Wendeler, and Hubbuch (2024), *Front. Bioeng. Biotechnol.* 12, 1403644
     (`doi:10.3389/fbioe.2024.1403644`, `PMC11274341`).
   - *Chemistry*: An IgG1 with two engineered heavy-chain cysteines (one per
     heavy chain, e.g., S239C) is uncapped by full reduction with
     tris(2-carboxyethyl)phosphine hydrochloride (TCEP), buffer-exchanged, and
     selectively re-oxidized with dehydroascorbic acid (DHAA) to reform the
     four native interchain disulfide bridges while leaving two free thiol
     groups ($-\text{SH}$) per intact $\text{mAb}$. Subsequent Michael addition
     of a maleimide-functionalized payload ($D$) in aqueous buffer ($20\text{ mM}$
     sodium phosphate, $1\text{ mM}$ EDTA, $\text{pH } 7.0$, $10\%\text{ v/v}$
     DMSO cosolvent) proceeds consecutively from unconjugated antibody
     ($\text{mAb}_0$) to mono-conjugate ($\text{mAb}_1$) and bi-conjugate
     ($\text{mAb}_2$), in competition with first-order payload
     inactivation/depletion ($D \to D_{\text{depl}}$ via maleimide hydrolysis or
     nonspecific sink).
   - *Kinetic & identifiability findings*: Andris et al. (2019) evaluated six
     candidate ODE structures on an intact-antibody basis ($\text{mAb}_0$,
     $\text{mAb}_1$, $\text{mAb}_2$, $D$, analyzed by non-reducing RP-UHPLC) and
     identified two consecutive bimolecular steps
     ($k_1 = 0.797\text{ L}\cdot\text{mmol}^{-1}\cdot\text{s}^{-1}$,
     $k_2 = 1.476\text{ L}\cdot\text{mmol}^{-1}\cdot\text{s}^{-1}$) plus a
     first-order payload sink ($k_3 = 1.55 \times 10^{-3}\text{ s}^{-1}$); Weggen
     et al. (2023, Sec. 2.2.5 & 3.2.2, Eqs. 3–5) validated this parameter set in
     a $300\text{ mL}$ lab-scale stirred tank (`GST-1`, $60\text{ s}$ syringe-pump
     feed) with $R^2 = 0.979$ for the ideal-mixing 0D model and $R^2 = 0.985$ for
     the 3D CFD model. Weggen et al. (2024) subsequently modeled site-specific
     $\text{DAR } 2$ conjugation (`ADC1` with cytotoxic payload `Drug1` in Dataset 1
     and surrogate `NPM` in Dataset 2, measured in $1\text{ mL}$ batch mode, Table 1
     & Sec. 2.1.2) on a reduced heavy-chain analytical basis ($H_0, H_1, H_2$
     measured by reducing RP-UHPLC after DTT disulfide cleavage) alongside general
     batch/fed-batch material balances (Eqs. 1–10, applied experimentally to $4\text{ mL}$
     syringe-pump fed-batch runs in Dataset 3). They demonstrated that when $H_2$
     (two payloads on a single heavy chain, an over-conjugation species from
     residual unreoxidized interchain cysteines) remains low ($5\%$ of total heavy
     chain at saturation, Sec. 3.3.1), the second heavy-chain rate constant $k_2$
     in the `Detailed (2k)` model has a large parameter standard deviation
     ($\hat{\theta}_2 = 1.712\text{ L}\cdot\text{mmol}^{-1}\cdot\text{s}^{-1}$,
     $\sigma_2 = 6.585\text{ L}\cdot\text{mmol}^{-1}\cdot\text{s}^{-1}$, $95\%$ CI
     $[-11.515, 14.939]\text{ L}\cdot\text{mmol}^{-1}\cdot\text{s}^{-1}$, Table 2),
     so a single shared rate constant
     ($\hat{\theta}_1 = 0.251\text{ L}\cdot\text{mmol}^{-1}\cdot\text{s}^{-1}$,
     standard deviation $\sigma_1 = 0.023\text{ L}\cdot\text{mmol}^{-1}\cdot\text{s}^{-1}$,
     $95\%$ CI $[0.204, 0.298]\text{ L}\cdot\text{mmol}^{-1}\cdot\text{s}^{-1}$,
     with fixed UV/Vis-derived $k_{\text{drug,Drug1}} = 0.001\text{ s}^{-1}$,
     `Simple 1k` model) is selected by leave-one-run-out cross-validation
     ($\text{RMSE}_{\text{CV}} = 1.863\ \mu\text{mol/L}$,
     $\text{RMSE}_{\text{P}} = 3.188\ \mu\text{mol/L}$).

2. **Partial interchain-disulfide TCEP reduction followed by cysteine
   thiol–maleimide conjugation (target $\text{DAR} = 4\text{–}8$)**:
   - *Primary sources*: Sun et al. (2005), *Bioconjugate Chem.* 16(5),
     1282–1290 (`doi:10.1021/bc050201y`); Liu et al. (2010), *Anal. Chem.*
     82(12), 5219–5226 (`doi:10.1021/ac100575n`); Kantner, Alkhawaja, and Watts
     (2017), *ACS Omega* 2(9), 5785–5791 (`doi:10.1021/acsomega.7b01094`);
     Matsuda et al. (2020), *ChemistrySelect* 5(28), 8435–8439
     (`doi:10.1002/slct.202001822`); Song et al. (2022), *Appl. Microbiol.
     Biotechnol.* 106(3), 1057–1066 (`doi:10.1007/s00253-022-11778-5`); Nayak and
     Richter (2023), *Org. Process Res. Dev.* 27(11), 2091–2099
     (`doi:10.1021/acs.oprd.3c00264`); Weggen et al. (2024), *Front. Bioeng.
     Biotechnol.* 12, 1403644 (`doi:10.3389/fbioe.2024.1403644`); Weggen,
     González, Hui, Bean, Wendeler, and Hubbuch (2025), *Biotechnol. Bioeng.*
     122(2), 342–356 (`doi:10.1002/bit.28899`, `PMC11808428`).
   - *Chemistry & two-step mechanism*: Mild reduction of the four interchain
     disulfide bonds of human IgG1 (two hinge heavy–heavy $\text{HH}$ bonds and
     two heavy–light $\text{HL}$ bonds) by TCEP generates even-numbered free
     cysteine thiols ($2, 4, 6, 8$ thiols per antibody, corresponding to bridged
     fragments $\text{Intact}$, $\text{HHL}$, $\text{HH}$, $\text{HL}$,
     $\text{H}$, and $\text{L}$ in non-reducing CGE-SDS), followed by rapid
     Michael addition of maleimide linker-payload to each liberated thiol (with
     optional in-situ PEG-azide quenching of residual TCEP when TCEP excess would
     otherwise compete for maleimide, Kantner et al., 2017).
   - *Kinetic findings*:
     - **Step 1 (Disulfide reduction by TCEP)**: Nayak and Richter (2023) and
       Weggen et al. (2025) established that TCEP disulfide cleavage exhibits
       strong structural cooperativity. In Weggen et al. (2025, Sec. 2.3.1,
       2.3.3, 3.1.1, 3.1.2 & Table 2, `WithInt + ReOx` `8k-model` calibrated across
       $T \in \{4, 12, 20, 37\}\ ^\circ\text{C}$, $c_{\text{mAb},0} \in [3.75, 20]\text{ g/L}$,
       and TCEP molar excess $[2, 10]\times$ at reference temperature
       $T_{\text{ref}} = 293.15\text{ K}$, selected by $\text{AIC}_{\text{c}} = 24.96$
       vs. $35.84$ for the `9k-model`), the chemical reaction network comprises
       10 species (including the three single-hinge-cleaved intermediates
       $\text{IntactR}, \text{HHLR}, \text{HHR}$ in the two-step hinge inter-HH
       disulfide cleavage), 11 reactions, and 8 second-order rate constants at
       $T_{\text{ref}} = 293.15\text{ K}$ (paired with 8 Arrhenius activation
       energies $E_a$ initialized from $E_{a,0} = 20\text{ kJ/mol}$ and
       $k_{T_{\text{ref}},0} = 1.0 \times 10^{-4}\text{ L}\cdot\mu\text{mol}^{-1}\cdot\text{s}^{-1} = 100.0\text{ L}\cdot\text{mol}^{-1}\cdot\text{s}^{-1}$,
       Sec. 2.4):
       1. **3 fragment-specific inter-HL disulfide reduction rates** (for
          $\text{Intact}/\text{IntactR}$, $\text{HHL}/\text{HHLR}$, and $\text{HL}$),
          which decrease sequentially from $6.98 \times 10^{-4}$ to
          $6.81 \times 10^{-5}\text{ L}\cdot\mu\text{mol}^{-1}\cdot\text{s}^{-1}$
          (i.e., from **$698.0$ to $68.1\text{ L}\cdot\text{mol}^{-1}\cdot\text{s}^{-1}$**
          under the exact $10^6\ \mu\text{mol/mol}$ prefix conversion, Sec. 3.1.2);
       2. **2 stepwise hinge inter-HH disulfide reduction rates** ($k_4, k_5$),
          where the second hinge inter-HH cleavage rate $k_5$ is $18\times$
          faster than the first hinge inter-HH cleavage rate $k_4$ ($k_5 / k_4 = 18$,
          Sec. 3.1.2) due to conformational opening of the Fc hinge after the
          first HH bond breaks; and
       3. **3 fragment-specific inter-HL disulfide re-oxidation rates** (capturing
          re-formation of reduced HL disulfide bonds over long hold times at low
          TCEP excess, Sec. 2.3.1 & 3.1.1; individual bar-chart estimates and
          $95\%$ CIs are reported in Figure 3 and Supporting Information S1: Table S4).
     - **Step 2 (Cysteine thiol–maleimide conjugation)**: For $\text{DAR } 8$
       interchain conjugation (`ADC2 + NPM` across 10 fed-batch and 3 batch runs),
       Weggen et al. (2024, Table 2 & Sec. 3.2) showed that a four-parameter
       chain-resolved kinetic model (`Detailed 4k`: light-chain conjugation
       $L_0^1 + D \xrightarrow{k_1} L_1^1$ with $\hat{\theta}_1 = 1.220\text{ L}\cdot\text{mmol}^{-1}\cdot\text{s}^{-1}$,
       $\sigma_1 = 0.003$, $95\%$ CI $[1.214, 1.227]$; and heavy-chain steps
       $H_0^c \xrightarrow{k_2} H_1^c \xrightarrow{k_3} H_2^c \xrightarrow{k_{4/5}} H_3^c \xrightarrow{k_{4/5}} H_4^c$
       with $\hat{\theta}_2 = 1.853$, $\sigma_2 = 0.006$, $95\%$ CI $[1.841, 1.865]$;
       $\hat{\theta}_3 = 5.117$, $\sigma_3 = 1.015$, $95\%$ CI $[3.101, 7.133]$; and
       $\hat{\theta}_{4/5} = 2.312$, $\sigma_{4/5} = 0.356$, $95\%$ CI
       $[1.605, 3.020]\text{ L}\cdot\text{mmol}^{-1}\cdot\text{s}^{-1}$)
       reduces $\text{RMSE}_{\text{CV}}$ by $49\%$ ($0.620$ vs. $1.213\ \mu\text{mol/L}$)
       and $\text{RMSE}_{\text{P}}$ by $28\%$ ($0.964$ vs. $1.339\ \mu\text{mol/L}$)
       compared with the single-rate `Simple (1k)` model
       ($\hat{\theta}_1 = 1.733$, $\sigma_1 = 0.018$, $95\%$ CI
       $[1.697, 1.768]\text{ L}\cdot\text{mmol}^{-1}\cdot\text{s}^{-1}$) because
       attachment of the first hydrophobic payload to the heavy chain accelerates
       conjugation of the second cysteine ($k_3 > k_2$).

3. **Stochastic lysine amide coupling / NHS-ester acylation and PEGylation**:
   - *Primary sources*: Pfister, Bourgeaux, and Morbidelli (2015), *Chem. Eng.
     Sci.* 137, 816–827 (`doi:10.1016/j.ces.2015.07.031`); Pfister et al.
     (2016), *Chem. Ing. Tech.* 88(11), 1598–1608
     (`doi:10.1002/cite.201600046`); Mao, Russell, and Carmali (2022),
     *Bioconjugate Chem.* 33(9), 1643–1653
     (`doi:10.1021/acs.bioconjchem.2c00262`).
   - *Chemistry & modeling limits*: Activated $N$-hydroxysuccinimide (NHS)
     esters react with surface-accessible $\varepsilon$-amines across $80+$
     lysine residues in parallel with aqueous NHS-ester hydrolysis (comparative
     engineering scale $k_{\text{hyd}} \sim 10^{-3}\text{–}10^{-2}\text{ s}^{-1}$
     at $\text{pH } 7.5\text{–}8.5$, `origin="synthetic"`). Because
     $2^{N_{\text{sites}}}$ positional isomers are lumping-dependent and
     site-specific rate constants are unidentifiable without peptide-mapping time
     courses, stochastic lysine conjugation is less suited than cysteine
     conjugation for benchmarking mechanistic parameter estimation and structural
     identifiability.

4. **Heterobifunctional linker-payload and AOC/ARC coupling with competing
   isomerization or hydrolysis**:
   - *Primary sources*: Shen et al. (2012), *Nat. Biotechnol.* 30(2), 184–189
     (`doi:10.1038/nbt.2108`); Tumey et al. (2014), *Bioconjugate Chem.* 25(10),
     1871–1880 (`doi:10.1021/bc500357n`); repository design issues
     [#154](https://github.com/PharmaPy-org/PharmaPy/issues/154),
     [#337](https://github.com/PharmaPy-org/PharmaPy/issues/337),
     [#339](https://github.com/PharmaPy-org/PharmaPy/issues/339), and
     [#356](https://github.com/PharmaPy-org/PharmaPy/issues/356).
   - *Chemistry & timescale separation*: Either (a) a biomacromolecule $A$ is
     coupled with a heterobifunctional linker $B$ and oligonucleotide/drug
     payload $C$ to form conjugate $D$ via sequential bimolecular steps
     ($A + B \to I_{\text{AB}}$, $I_{\text{AB}} + C \to D$) alongside linker
     hydrolysis, or (b) reactive species $A$ and $B$ form conjugate $C$
     (`coupling: A + B --> C`) while $C$ undergoes a unimolecular ring-opening,
     thiosuccinimide hydrolysis, or retro-Michael deconjugation reaction
     (`iso: C --> I`, tracked in `#339`). Under the plasma and buffer
     stabilization conditions studied by Shen et al. (2012) and Tumey et al.
     (2014), thiosuccinimide ring-opening hydrolysis and retro-Michael
     deconjugation exhibit half-lives of $t_{1/2} \sim 24\text{–}120\text{ h}$
     ($k_{\text{iso}} \sim 1.6 \times 10^{-6}\text{–}8.0 \times 10^{-6}\text{ s}^{-1}$),
     which is one to two orders of magnitude ($24\text{–}240\times$) longer than
     the $0.5\text{–}1.0\text{ h}$ conjugation reactor horizon (corresponding to
     at most $1 - 2^{-1/24} \approx 2.85\%$ first-order conversion over $1.0\text{ h}$).
     Weggen et al. (2024, Sec. 2.2.1.2) therefore treat thiol–maleimide addition
     as irreversible during the $\le 1\text{ h}$ reaction window ($< 3\%$
     kinetic-truncation assumption, flagged for domain confirmation with
     `@dcasasor`) and explicitly model free-payload depletion
     $D \xrightarrow{k_{\text{drug}}} D_{\text{depl}}$ instead. When
     high-$\text{pH}$ ($\text{pH } 8.0\text{–}9.0$) post-conjugation succinimide
     ring stabilization is modeled, `iso: mAb_2Drug --> mAb_2Drug_stabilized` can
     be appended directly as a fourth elementary reaction.

| Modality | Representative network | Bimolecular rate $k$ $[\text{L}\cdot\text{mol}^{-1}\cdot\text{s}^{-1}]$ | Competing side reaction | Nominal $\tau_R = 1/(k\,c_{\text{tot}})$ $[\text{s}]$ at $c_{\text{tot}} = 10^{-4}\text{ mol/L}$ | Public kinetic parameter & error tables |
| --- | --- | --- | --- | --- | --- |
| Site-specific Cys–maleimide ($\text{DAR } 2$) | $\text{mAb}_0 + D \xrightarrow{k_1} \text{mAb}_1$, $\text{mAb}_1 + D \xrightarrow{k_2} \text{mAb}_2$, $D \xrightarrow{k_3} D_{\text{depl}}$ | $2.51 \times 10^2\text{–}4.84 \times 10^3$ (literature) | First-order maleimide depletion ($k_3 = 1.0 \times 10^{-3}\text{–}4.1 \times 10^{-2}\text{ s}^{-1}$) | $2.1\text{–}39.8$ | Yes: Andris et al. (2019); Weggen et al. (2023, 2024) |
| Interchain TCEP reduction + Cys–maleimide ($\text{DAR } 4\text{–}8$) | Step 1: 8-rate `WithInt + ReOx` TCEP disulfide cleavage + HL reoxidation; Step 2: $L_0 \to L_1$ and $H_0 \to H_1 \to H_2 \to H_3 \to H_4$ + $D \to D_{\text{depl}}$ | Reduction: $6.81 \times 10^1\text{–}6.98 \times 10^2$ (inter-HL, converted via $\times 10^6\ \mu\text{mol/mol}$); Conjugation: $1.22 \times 10^3\text{–}5.12 \times 10^3$ | Inter-HL disulfide re-oxidation (3 fragment rates); payload depletion | Reduction: $14.3\text{–}147$ (bimolecular scale; $300\text{–}14400\text{ s}$ CGE assay window); Conjugation: $2.0\text{–}8.2$ | Yes: Weggen et al. (2024, 2025); Nayak & Richter (2023) |
| Stochastic Lys–NHS amide coupling | $\text{mAb}_{k} + L \xrightarrow{k_k} \text{mAb}_{k+1}$, $L \xrightarrow{k_{\text{hyd}}} L_{\text{hyd}}$ ($k = 0,\dots,N$) | $10^1\text{–}10^2$ (`origin="synthetic"` comparative engineering scale; model structure in Pfister et al., 2015) | Aqueous NHS-ester hydrolysis ($10^{-3}\text{–}10^{-2}\text{ s}^{-1}$, `origin="synthetic"` comparative scale) | $100\text{–}1000$ | Partial (PEGylation structure in Pfister et al., 2015) |
| Lilly coupling + isomerization prototype (`#339`) | `coupling`: $A + B \xrightarrow{k_c} C$; `iso`: $C \xrightarrow{k_{\text{iso}}} I$ | $10^2\text{–}10^3$ (`origin="synthetic"` design range, `#339`) | Unimolecular isomerization/ring-opening ($10^{-6}\text{–}10^{-2}\text{ s}^{-1}$, `origin="synthetic"`) | $10\text{–}100$ | Synthetic design specification (`#339`) |

### 2.2 Selected representative case architecture for WP1.5

To satisfy both the scientific requirements of the Purdue–Lilly 360 proposal
(two-step bioconjugation reaction system with competing reactions, mixing
sensitivity, PAT spectroscopic observability, parameter estimation with
identifiability/experiment design, and four-step WP4 scheduling export) and
PharmaPy's public reproducibility policy (`AGENTS.md`), WP1.5 adopts a
**tiered cysteine-bioconjugation benchmark architecture** anchored in the
open-access KIT/AstraZeneca kinetic series (Andris et al., 2019; Weggen et al.,
2023, 2024, 2025):

1. **Core benchmark (`Tier 1`, primary kinetic, mixing, and PAT calibration
   target)**:
   - **Site-specific $\text{DAR } 2$ cysteine–maleimide conjugation with
     competing payload depletion** in two complementary representations derived
     from distinct analytical assays and experimental campaigns:
     - **Model 1A (Intact-antibody consecutive `2k + sink` model, Andris et al.,
       2019; Weggen et al., 2023, Eqs. 3–5)**: tracks intact unconjugated
       antibody ($\text{mAb}_0$), mono-conjugate ($\text{mAb}_1$),
       bi-conjugate ($\text{mAb}_2$), free payload ($D$), and inactivated
       payload ($D_{\text{depl}}$) measured by non-reducing intact-ADC RP-UHPLC
       across batch (`BatchReactor`), fed-batch (`SemibatchReactor`, e.g., $60\text{ s}$
       syringe-pump addition in $300\text{ mL}$ `GST-1`, Weggen et al., 2023,
       Sec. 2.2.6), and continuous (`Mixer` + `PlugFlowReactor`) operating modes.
     - **Model 1B (Reduced heavy-chain `Simple 1k + sink` and `Detailed 2k + sink`
       models, Weggen et al., 2024, Eqs. 1–10 & Table 2)**: tracks heavy-chain
       species $H_0^0, H_0^1, H_0^2, H_1^1, H_1^2, H_2^2$ (measured as summed
       load states $H_0, H_1, H_2$ by reducing RP-UHPLC after DTT cleavage),
       free payload $D$, and depleted payload $D_{\text{depl}}$. In Weggen et al.
       (2024, Table 1 & Sec. 2.1.2), Dataset 1 (`ADC1 + Drug1`) and Dataset 2
       (`ADC1 + NPM`) were experimentally measured in $1\text{ mL}$ batch mode,
       while the general $4\text{ mL}$ syringe-pump fed-batch material balance
       (Eqs. 1–2, with $10, 20, 30\text{ min}$ payload addition) was experimentally
       applied to Dataset 3 (`ADC2 + NPM`). Combining the `Simple 1k + sink` and
       `Detailed 2k + sink` DAR 2 rate equations (Eqs. 4–10) with the fed-batch
       balance (Eqs. 1–2) provides a benchmark for **practical identifiability
       analysis** ($k_1$ vs. $k_2$) and **model-based experiment design (MBDoE)**
       via feed-rate manipulation.

---

## 3. Species, stoichiometry, rate laws, and kinetic parameters

### 3.1 Species definitions and physical bases

All liquid-phase species concentrations in PharmaPy's `RxnKinetics` and
`LiquidPhase` are tracked on a **molar concentration basis** $[\text{mol/L}]$ in
an aqueous buffer solvent (`buffer`, $20\text{ mM}$ $\text{NaH}_2\text{PO}_4$,
$1\text{ mM}$ EDTA, $\text{pH } 7.0$, with $10\%\text{ v/v}$ DMSO cosolvent;
mixture density $\rho_{\text{liq}} = 1010.5\text{ kg/m}^3$ and dynamic viscosity
$\mu_{\text{liq}} = 1.06 \times 10^{-3}\text{ Pa}\cdot\text{s}$ at
$T = 295.15\text{ K}$, Weggen et al., 2023, Sec. 2.2.2).

| Symbol | PharmaPy species ID | Molar mass $M_i$ $[\text{g/mol}]$ | Physical meaning, analytical basis, and provenance | Typical initial concentration $c_{i,0}$ $[\text{mol/L}]$ |
| --- | --- | --- | --- | --- |
| $\text{mAb}_0$ | `mAb` | $1.500000 \times 10^5$ | Intact unconjugated IgG1 antibody (2 reactive Cys thiols, $\text{DAR}=0$) | $1.00 \times 10^{-5}\text{–}6.67 \times 10^{-5}$ ($1.5\text{–}10\text{ g/L}$) |
| $\text{mAb}_1$ | `mAb_1Drug` | $1.5029731 \times 10^5$ | Intact mono-conjugated antibody ($M_{\text{mAb}} + M_{\text{Drug}}$, $\text{DAR}=1$) | $0.0$ |
| $\text{mAb}_2$ | `mAb_2Drug` | $1.5059462 \times 10^5$ | Intact bi-conjugated antibody ($M_{\text{mAb}} + 2\,M_{\text{Drug}}$, $\text{DAR}=2$) | $0.0$ |
| $H_0^0$ | `H0_0` | $5.000000 \times 10^4$ | Heavy chain with 0 payloads bound and 0 reactive cysteines (pre-inactivated; $11\%$ of total heavy chain in Weggen et al., 2024, Sec. 3.3.1) | $0.11 \, c_{H,0}$ ($2.2 \times 10^{-6}\text{–}1.47 \times 10^{-5}$) |
| $H_0^1$ | `H0_1` | $5.000000 \times 10^4$ | Heavy chain with 0 payloads bound and 1 reactive engineered cysteine ($84\%$ of total heavy chain in Weggen et al., 2024, Sec. 3.3.1) | $0.84 \, c_{H,0}$ ($1.68 \times 10^{-5}\text{–}1.12 \times 10^{-4}$) |
| $H_0^2$ | `H0_2` | $5.000000 \times 10^4$ | Heavy chain with 0 payloads bound and 2 reactive cysteines ($5\%$ of total heavy chain in Weggen et al., 2024, Sec. 3.3.1) | $0.05 \, c_{H,0}$ ($1.0 \times 10^{-6}\text{–}6.67 \times 10^{-6}$) |
| $H_1^1$ | `H1_1` | $5.029731 \times 10^4$ | Heavy chain with 1 payload bound and 0 remaining reactive cysteines ($M_H + M_{\text{Drug}}$) | $0.0$ |
| $H_1^2$ | `H1_2` | $5.029731 \times 10^4$ | Heavy chain with 1 payload bound and 1 remaining reactive cysteine ($M_H + M_{\text{Drug}}$) | $0.0$ |
| $H_2^2$ | `H2_2` | $5.059462 \times 10^4$ | Heavy chain with 2 payloads bound and 0 remaining reactive cysteines ($M_H + 2\,M_{\text{Drug}}$) | $0.0$ |
| $D$ | `Drug` | $2.9731 \times 10^2$ | Reactive maleimide-functionalized payload ($N$-(1-pyrenyl)maleimide, $\text{C}_{20}\text{H}_{11}\text{NO}_2$, $M_{\text{NPM}} = 297.31\text{ g/mol}$ in Model 1A; for vc-MMAE linker-payload, $M_D \approx 1316.6\text{ g/mol}$) | Batch: $3.0 \times 10^{-5}\text{–}3.3 \times 10^{-4}$; Fed-batch feed: $1.0 \times 10^{-3}\text{–}1.0 \times 10^{-2}$ |
| $D_{\text{depl}}$ | `Drug_depl` | $2.9731 \times 10^2$ | Inactivated/depleted payload sink ($M_{\text{Drug\_depl}} = M_{\text{Drug}} = 297.31\text{ g/mol}$ so first-order payload inactivation $D \to D_{\text{depl}}$ in Andris et al., 2019 / Weggen et al., 2023 Eq. 5 & Weggen et al., 2024 Sec. 3.1 is strictly mass-conserving without solvent consumption) | $0.0$ |
| $\text{Solv}$ | `buffer` | $1.967 \times 10^1$ | Aqueous phosphate/EDTA buffer with $10\%\text{ v/v}$ DMSO cosolvent (mole-fraction-weighted average molar mass of $90\%\text{ v/v}$ $\text{H}_2\text{O}$ at $18.015\text{ g/mol}$, $0.90 \times 55.5 = 49.95\text{ mol/L}$ and $10\%\text{ v/v}$ DMSO at $78.13\text{ g/mol}$, $0.10 \times 14.1 = 1.41\text{ mol/L}$: $M_{\text{solv}} = 1010.5\text{ g/L} / 51.36\text{ mol/L} = 19.67\text{ g/mol}$) | Completed by `LiquidPhase` volume closure ($\approx 51.3\text{ mol/L}$) |

*Note on schema-required physical properties in `BIOCONJUGATION_DATABASE`*:
PharmaPy's `LiquidPhase` JSON schema requires liquid density `rho_liq`
$[\text{kg/m}^3]$, molar heat capacity polynomial `cp_liq`
$[\text{J}\cdot\text{mol}^{-1}\cdot\text{K}^{-1}]$, and critical temperature
`t_crit` $[\text{K}]$ for every species. For isothermal ($T = 295.15\text{ K}$)
dilute aqueous bioconjugation simulations ($\rho_{\text{mAb}} \le 10\text{ g/L}$,
volume fraction $< 1\%$), all species are assigned the measured mixture density
$\rho_{\text{liq}} = 1010.5\text{ kg/m}^3$ (Weggen et al., 2023, Sec. 2.2.2) so
that constant-density dilution balances hold exactly, while `cp_liq` and
`t_crit` are synthetic schema-required placeholders (`origin='synthetic'`) that
do not enter the isothermal material balance.

*Note on heavy-chain versus intact-mAb concentration and rate-constant bases*:
Every intact IgG1 molecule comprises two heavy chains and two light chains, so
the total heavy-chain molar concentration is twice the intact antibody molar
concentration:

$$c_{H,\text{total}} = 2 \, c_{\text{mAb,total}} = 2 \, \frac{\rho_{\text{mAb}}}{M_{\text{mAb}}} \quad [\text{mol/L}]$$

For example, at a protein mass concentration of $\rho_{\text{mAb}} = 5.0\text{ g/L}$
and $M_{\text{mAb}} = 1.50 \times 10^5\text{ g/mol}$, the intact antibody molar
concentration is $c_{\text{mAb,total}} = 3.333 \times 10^{-5}\text{ mol/L}$
($33.33\ \mu\text{mol/L}$) and the heavy-chain molar concentration is
$c_{H,\text{total}} = 6.667 \times 10^{-5}\text{ mol/L}$ ($66.67\ \mu\text{mol/L}$).

Crucially, **rate constants from Model 1A (intact-mAb basis) and Model 1B
(reduced heavy-chain basis) are not algebraically transferable by a simple
factor of $2$**:
- If an intact antibody carried two identical, independent thiols with
  microscopic per-site rate constant $k_{\text{site}}$, statistical factors would
  give $k_{1,\text{intact}} = 2\,k_{\text{site}}$ (two available sites on
  $\text{mAb}_0$) and $k_{2,\text{intact}} = k_{\text{site}}$ (one remaining site
  on $\text{mAb}_1$), so $k_{2,\text{intact}} / k_{1,\text{intact}} = 0.5$.
- In Model 1A (Andris et al., 2019; Weggen et al., 2023, Eqs. 3–5, non-reducing
  RP-UHPLC), the calibrated ratio is $k_2 / k_1 = 1.476 / 0.797 = 1.852 > 0.5$,
  reflecting **positive hydrophobic cooperativity** in the hinge region after the
  first payload attaches.
- In Model 1B (Weggen et al., 2024, Sec. 2.1.3 & 4.2.1), DTT disulfide reduction
  prior to RP-UHPLC cleaves the antibody into heavy and light chains ($k_1 = 0.251\text{ L}\cdot\text{mmol}^{-1}\cdot\text{s}^{-1}$
  for `ADC1 + Drug1`; $k_1 = 4.840\text{ L}\cdot\text{mmol}^{-1}\cdot\text{s}^{-1}$
  for `ADC1 + NPM` in Dataset 2). Note that in Model 1A the instantaneous
  per-thiol consumption rate $(k_1 c_{\text{mAb}_0} + k_2 c_{\text{mAb}_1}) / (2 c_{\text{mAb}_0} + c_{\text{mAb}_1})$
  is bounded strictly within $[k_1/2, k_2] = [0.399, 1.476]\text{ L}\cdot\text{mmol}^{-1}\cdot\text{s}^{-1}$,
  so the $3.3\text{–}12\times$ higher heavy-chain estimate
  $k_1 = 4.840\text{ L}\cdot\text{mmol}^{-1}\cdot\text{s}^{-1}$ in Dataset 2 of
  Weggen et al. (2024, Table 2) **cannot** be explained by intact-versus-heavy-chain
  basis lumping alone; it reflects inter-campaign material, tube-mixing, or
  analytical differences between the 2019 glass-vessel campaign and the 2024
  $1.5\text{ mL}$ polypropylene microcentrifuge-tube campaign (alongside the
  much faster NPM loss rate $k_{\text{drug,NPM}} = 0.041\text{ s}^{-1}$ vs.
  $k_3 = 1.55 \times 10^{-3}\text{ s}^{-1}$, Weggen et al., 2024, Sec. 4.1.1).
  Each model must therefore be used with its own calibrated parameter set and
  analytical observable, and the inter-study `ADC1 + NPM` rate discrepancy is
  flagged for domain review with `@dcasasor` during `#356`.

### 3.2 Stoichiometry and elementary rate laws

#### Model 1A — Intact-antibody consecutive conjugation with competing payload depletion (Andris et al., 2019; Weggen et al., 2023, Eqs. 3–5)

In PharmaPy `RxnKinetics` syntax (`rxn_list`), the three elementary irreversible
reactions are:

$$\begin{aligned}
\text{R1 (first conjugation):} &\quad \text{mAb} + \text{Drug} \xrightarrow{k_1} \text{mAb\_1Drug}, & r_1 &= k_1 \, c_{\text{mAb}} \, c_{\text{Drug}} \quad [\text{mol}\cdot\text{L}^{-1}\cdot\text{s}^{-1}] \\
\text{R2 (second conjugation):} &\quad \text{mAb\_1Drug} + \text{Drug} \xrightarrow{k_2} \text{mAb\_2Drug}, & r_2 &= k_2 \, c_{\text{mAb\_1Drug}} \, c_{\text{Drug}} \quad [\text{mol}\cdot\text{L}^{-1}\cdot\text{s}^{-1}] \\
\text{R3 (payload depletion):} &\quad \text{Drug} \xrightarrow{k_3} \text{Drug\_depl}, & r_3 &= k_3 \, c_{\text{Drug}} \quad [\text{mol}\cdot\text{L}^{-1}\cdot\text{s}^{-1}]
\end{aligned}$$

*( Note: In the validation run of Weggen et al., 2023, Sec. 2.2.5, an initial
intact-mAb cysteine distribution of $88.59\%$ with 2 reactive cysteines, $8.60\%$
with 1 reactive cysteine, and $2.81\%$ with 0 reactive cysteines—giving a
maximum achievable $\text{DAR}_{\max} = 2(0.8859) + 1(0.0860) = 1.858$—was used
for the experimental comparison, while the 3D CFD vessel-comparison study set
$100\%$ active $\text{mAb}_0$ as in Eqs. 3–5 above.)*

For species ordered as
$\mathbf{c} = [c_{\text{mAb}}, c_{\text{mAb\_1Drug}}, c_{\text{mAb\_2Drug}}, c_{\text{Drug}}, c_{\text{Drug\_depl}}]^\top$
$[\text{mol/L}]$, the stoichiometric matrix $\boldsymbol{\nu} \in \mathbb{R}^{3 \times 5}$
(stored as `RxnKinetics.stoich_matrix`, rows = reactions, columns =
participating species) and net species production rates
$\mathbf{r}_{\text{species}} = \boldsymbol{\nu}^\top \mathbf{r}$
$[\text{mol}\cdot\text{L}^{-1}\cdot\text{s}^{-1}]$ are:

$$\boldsymbol{\nu} = \begin{bmatrix}
-1 & +1 &  0 & -1 &  0 \\
 0 & -1 & +1 & -1 &  0 \\
 0 &  0 &  0 & -1 & +1
\end{bmatrix}, \qquad
\mathbf{r}_{\text{species}} = \begin{bmatrix}
-r_1 \\
r_1 - r_2 \\
r_2 \\
-r_1 - r_2 - r_3 \\
r_3
\end{bmatrix} \quad [\text{mol}\cdot\text{L}^{-1}\cdot\text{s}^{-1}]$$

With $M_{\text{mAb\_1Drug}} = M_{\text{mAb}} + M_{\text{Drug}}$,
$M_{\text{mAb\_2Drug}} = M_{\text{mAb}} + 2\,M_{\text{Drug}}$, and
$M_{\text{Drug\_depl}} = M_{\text{Drug}}$, the stoichiometric mass residual
vector $\boldsymbol{\nu}\,\mathbf{M} = \mathbf{0}\ [\text{g/mol}]$ vanishes
identically across all three reactions (`R1`, `R2`, `R3`), and two exact linear
molar invariants hold at all times in a closed batch reactor:

1. **Antibody core conservation**:
   $r_{\text{mAb}} + r_{\text{mAb\_1Drug}} + r_{\text{mAb\_2Drug}} = 0$, so
   $c_{\text{mAb}}(t) + c_{\text{mAb\_1Drug}}(t) + c_{\text{mAb\_2Drug}}(t) = c_{\text{mAb},0}$
   $[\text{mol/L}]$.
2. **Total drug-moiety conservation**:
   $r_{\text{Drug}} + r_{\text{Drug\_depl}} + r_{\text{mAb\_1Drug}} + 2\,r_{\text{mAb\_2Drug}} = 0$,
   so $c_{\text{Drug}}(t) + c_{\text{Drug\_depl}}(t) + c_{\text{mAb\_1Drug}}(t) + 2\,c_{\text{mAb\_2Drug}}(t) = c_{\text{Drug},0}$
   $[\text{mol/L}]$.

#### Model 1B — Reduced heavy-chain conjugation with initial cysteine heterogeneity and competing payload depletion (Weggen et al., 2024, Eqs. 1–10)

When analytical time courses are measured by reduced RP-UHPLC (Weggen et al.,
2024, Sec. 2.2.1.1 & 2.2.1.2, Eqs. 4–10), heavy chains carry $c \in \{0, 1, 2\}$
initial reactive thiols ($H_0^0, H_0^1, H_0^2$, with $\mathrm{d}c_{H_0^0}/\mathrm{d}t = 0$
in batch mode, Eq. 6, and zero light-chain conjugation $L_1 = 0$ observed for
DAR 2 `ADC1`) and react according to:

$$\begin{aligned}
\text{R1a (Eqs. 5, 8):} &\quad H_0^1 + \text{Drug} \xrightarrow{k_1} H_1^1, & r_{1a} &= k_1 \, c_{H_0^1} \, c_{\text{Drug}} \quad [\text{mol}\cdot\text{L}^{-1}\cdot\text{s}^{-1}] \\
\text{R1b (Eqs. 4, 7):} &\quad H_0^2 + \text{Drug} \xrightarrow{k_1} H_1^2, & r_{1b} &= k_1 \, c_{H_0^2} \, c_{\text{Drug}} \quad [\text{mol}\cdot\text{L}^{-1}\cdot\text{s}^{-1}] \\
\text{R2 (Eqs. 7, 9):} &\quad H_1^2 + \text{Drug} \xrightarrow{k_2} H_2^2, & r_2 &= k_2 \, c_{H_1^2} \, c_{\text{Drug}} \quad [\text{mol}\cdot\text{L}^{-1}\cdot\text{s}^{-1}] \\
\text{R3 (Eq. 10):} &\quad \text{Drug} \xrightarrow{k_{\text{drug}}} \text{Drug\_depl}, & r_3 &= k_{\text{drug}} \, c_{\text{Drug}} \quad [\text{mol}\cdot\text{L}^{-1}\cdot\text{s}^{-1}]
\end{aligned}$$

*( Note: Weggen et al., 2024, Eq. 4 write $\mathrm{d}c_{H_0^2}/\mathrm{d}t = -k_1 c_{H_0^2} c_{\text{Drug}}$
without a statistical factor of $2$, treating $k_1$ and $k_2$ as empirical step
rate constants on the heavy-chain populations.)*

Reducing RP-UHPLC resolves heavy chains by the number of bound payloads
($0, 1, 2$) regardless of how many unreacted thiols remain, so the measured
chromatographic species are the linear projections:

$$c_{H_0}(t) = c_{H_0^0}(t) + c_{H_0^1}(t) + c_{H_0^2}(t), \qquad
c_{H_1}(t) = c_{H_1^1}(t) + c_{H_1^2}(t), \qquad
c_{H_2}(t) = c_{H_2^2}(t) \quad [\text{mol/L}]$$

With the published `ADC1 + Drug1` initial heavy-chain distribution of $11\%\ H_0^0$,
$84\%\ H_0^1$, and $5\%\ H_0^2$ (Weggen et al., 2024, Sec. 3.3.1), complete
saturation of all reactive thiols ($\ge 3\times$ payload excess) yields a
plateau drug load of $0.84 + 2(0.05) = 0.94$ drugs per heavy chain, or
$\text{DAR}_{\max} = 2 \times 0.94 = 1.88$.

In a fed-batch reactor (`SemibatchReactor`) with initial liquid volume $V_0$
$[\text{L}]$, constant volumetric payload feed rate $F_{\text{in}}$
$[\text{L/s}]$ over $t \in [0, t_{\text{add}}]$, and feed payload concentration
$c_{\text{Drug,in}}$ $[\text{mol/L}]$, the liquid volume and species dilution
balances are (Weggen et al., 2024, Eqs. 1–3):

$$\frac{\mathrm{d}V}{\mathrm{d}t} = F(t) \quad [\text{L/s}], \qquad
\frac{\mathrm{d}c_i}{\mathrm{d}t} = r_{\text{species},i}(\mathbf{c}) + \frac{F(t)}{V(t)} \left(c_{i,\text{in}} - c_i(t)\right) \quad [\text{mol}\cdot\text{L}^{-1}\cdot\text{s}^{-1}]$$

where $F(t) = F_{\text{in}}$ for $0 \le t \le t_{\text{add}}$ and $F(t) = 0$ for
$t > t_{\text{add}}$, $c_{\text{Drug,in}} > 0$, and $c_{i,\text{in}} = 0$ for
all protein and depleted-payload species. In the `Simple 1k` model, setting
$k_2 = k_1$ ties the two conjugation rate constants while retaining all three
heavy-chain sub-populations ($H_0^0, H_0^1, H_0^2$). Only in the **idealized
mono-thiol heavy-chain limiting case** ($c_{H_0^0,0} = c_{H_0^2,0} = 0$,
$c_{H_0^1,0} = c_{H,0}$) do `R1b` and `R2` vanish identically so the system
reduces to the two reactions $H_0^1 + \text{Drug} \xrightarrow{k_1} H_1^1$ and
$\text{Drug} \xrightarrow{k_{\text{drug}}} \text{Drug\_depl}$.

### 3.3 Unit conversion to PharmaPy's SI-derived molar basis

Literature bioconjugation papers report concentrations in $[\text{mmol/L}]$ or
$[\mu\text{mol/L}]$ and second-order rate constants in
$[\text{L}\cdot\text{mmol}^{-1}\cdot\text{s}^{-1}]$ (Andris et al., 2019;
Weggen et al., 2023, 2024) or
$[\text{L}\cdot\mu\text{mol}^{-1}\cdot\text{s}^{-1}]$ (Weggen et al., 2025).
PharmaPy's `RxnKinetics` evaluates elementary rate laws on a $[\text{mol/L}]$ and
$[\text{s}]$ basis:

$$r_i\ [\text{mol}\cdot\text{L}^{-1}\cdot\text{s}^{-1}] = k_{i,\text{PharmaPy}}\ [\text{L}\cdot\text{mol}^{-1}\cdot\text{s}^{-1}] \cdot c_A\ [\text{mol/L}] \cdot c_B\ [\text{mol/L}]$$

Because $1\text{ mol} = 10^3\text{ mmol} = 10^6\ \mu\text{mol}$, dimensional
consistency requires:

$$k_{i,\text{PharmaPy}}\ [\text{L}\cdot\text{mol}^{-1}\cdot\text{s}^{-1}]
= 10^3 \, k_{i,\text{lit}}\ [\text{L}\cdot\text{mmol}^{-1}\cdot\text{s}^{-1}]
= 10^6 \, k_{i,\text{lit}}\ [\text{L}\cdot\mu\text{mol}^{-1}\cdot\text{s}^{-1}]$$

First-order rate constants ($k_3, k_{\text{drug}}$) have units of
$[\text{s}^{-1}]$, which are invariant to the molar concentration scale.

### 3.4 Published kinetic parameter tables and provenance

| Model & System | Parameter | Literature Estimate $\hat{\theta}_j$, Standard Deviation $\sigma_j$, & $95\%$ CI $[\text{lb}, \text{ub}]$ | PharmaPy Value (`[mol/L, s]` basis) | Conditions & Validated Range | Primary Source |
| --- | --- | --- | --- | --- | --- |
| **Model 1A**: Intact $\text{DAR } 2$ (`ADC1 + NPM`, `2k + sink`, non-reducing RP-UHPLC) | $k_1$ ($\text{mAb}_0 + D$) | $\hat{\theta}_1 = 0.797\text{ L}\cdot\text{mmol}^{-1}\cdot\text{s}^{-1}$ | $797.0\text{ L}\cdot\text{mol}^{-1}\cdot\text{s}^{-1}$ | $22\ ^\circ\text{C}$ ($295.15\text{ K}$), $\text{pH } 7.0$, $20\text{ mM}$ $\text{NaH}_2\text{PO}_4$, $1\text{ mM}$ EDTA, $10\%\text{ v/v}$ DMSO; $c_{\text{mAb},0} \in [1.5, 10]\text{ g/L}$ ($10\text{–}66.7\ \mu\text{mol/L}$), $5\times$ payload excess | Andris et al. (2019), `10.1016/j.jbiotec.2019.09.013`; Weggen et al. (2023), `10.3389/fbioe.2023.1123842`, Sec. 2.2.5, Eq. (3) |
| **Model 1A**: Intact $\text{DAR } 2$ (`ADC1 + NPM`, `2k + sink`, non-reducing RP-UHPLC) | $k_2$ ($\text{mAb}_1 + D$) | $\hat{\theta}_2 = 1.476\text{ L}\cdot\text{mmol}^{-1}\cdot\text{s}^{-1}$ | $1476.0\text{ L}\cdot\text{mol}^{-1}\cdot\text{s}^{-1}$ | Same as above (positive hinge cooperativity $k_2 / k_1 = 1.852$) | Andris et al. (2019); Weggen et al. (2023), `10.3389/fbioe.2023.1123842`, Sec. 2.2.5, Eq. (4) |
| **Model 1A**: Intact $\text{DAR } 2$ (`ADC1 + NPM`, `2k + sink`, non-reducing RP-UHPLC) | $k_3$ ($D \to D_{\text{depl}}$) | $\hat{\theta}_3 = 1.55 \times 10^{-3}\text{ s}^{-1}$ | $1.55 \times 10^{-3}\text{ s}^{-1}$ | Same as above ($R^2 = 0.979$ for 0D model and $0.985$ for 3D CFD in $300\text{ mL}$ `GST-1`, Sec. 3.2.2) | Andris et al. (2019); Weggen et al. (2023), `10.3389/fbioe.2023.1123842`, Sec. 2.2.5, Eq. (5) |
| **Model 1B**: Heavy-chain $\text{DAR } 2$ `Simple 1k` (`ADC1 + Drug1`, reducing RP-UHPLC) | $k_1$ ($H_0^1, H_0^2, H_1^2 + D$) | $\hat{\theta}_1 = 0.251$, $\sigma_1 = 0.023$, $95\%$ CI $[0.204, 0.298]\text{ L}\cdot\text{mmol}^{-1}\cdot\text{s}^{-1}$ | $\hat{\theta}_1 = 251.0$, $\sigma_1 = 23.0$, $95\%$ CI $[204.0, 298.0]\text{ L}\cdot\text{mol}^{-1}\cdot\text{s}^{-1}$ | $22\ ^\circ\text{C}$ ($295.15\text{ K}$), $\text{pH } 7.0$, $10\%\text{ v/v}$ DMSO; Dataset 1 ($1\text{ mL}$ batch, $1.5\text{–}10\text{ g/L}$ mAb, $1\text{–}8\times$ Drug1, 8 conditions), $\text{RMSE}_{\text{CV}} = 1.863\ \mu\text{mol/L}$, $\text{RMSE}_{\text{P}} = 3.188\ \mu\text{mol/L}$ | Weggen et al. (2024), `10.3389/fbioe.2024.1403644`, Table 1 & Table 2 |
| **Model 1B**: Heavy-chain $\text{DAR } 2$ `Simple 1k` (`ADC1 + Drug1`, reducing RP-UHPLC) | $k_{\text{drug,Drug1}}$ ($D \to D_{\text{depl}}$) | $1.00 \times 10^{-3}\text{ s}^{-1}$ (fixed from UV/Vis stability study) | $1.00 \times 10^{-3}\text{ s}^{-1}$ | $0.1\text{ mM}$ Drug1 in conjugation buffer ($10\%\text{ v/v}$ DMSO) over $1\text{ h}$ | Weggen et al. (2024), `10.3389/fbioe.2024.1403644`, Sec. 2.1.4, 2.2.1.3 & 3.1 |
| **Model 1B**: Heavy-chain $\text{DAR } 2$ `Detailed 2k` (`ADC1 + Drug1`, reducing RP-UHPLC) | $k_1, k_2$ | $k_1 = 0.297$ ($\sigma_1 = 0.027$, CI $[0.242, 0.351]$); $k_2 = 1.712$ ($\sigma_2 = 6.585$, CI $[-11.515, 14.939]$) $\text{L}\cdot\text{mmol}^{-1}\cdot\text{s}^{-1}$ | $k_1 = 297.0$ ($\sigma_1 = 27.0$); $k_2 = 1712.0$ ($\sigma_2 = 6585.0$) $\text{L}\cdot\text{mol}^{-1}\cdot\text{s}^{-1}$ | Demonstrates practical unidentifiability of $k_2$ when $H_0^2$ is $5\%$ of total heavy chain ($\text{RMSE}_{\text{CV}} = 1.808\ \mu\text{mol/L}$, $\text{RMSE}_{\text{P}} = 3.207\ \mu\text{mol/L}$) | Weggen et al. (2024), `10.3389/fbioe.2024.1403644`, Table 2 & Sec. 3.3.1 |
| **Model 1B**: Heavy-chain $\text{DAR } 2$ `Simple 1k` (`ADC1 + NPM`, reducing RP-UHPLC) | $k_1, k_{\text{drug,NPM}}$ | $k_1 = 4.840\text{ L}\cdot\text{mmol}^{-1}\cdot\text{s}^{-1}$, $k_{\text{drug,NPM}} = 0.041\text{ s}^{-1}$ | $k_1 = 4840.0\text{ L}\cdot\text{mol}^{-1}\cdot\text{s}^{-1}$, $k_{\text{drug,NPM}} = 4.10 \times 10^{-2}\text{ s}^{-1}$ | Dataset 2 ($1\text{ mL}$ batch, $1.5\text{–}3\text{ g/L}$ mAb, $3\text{–}5\times$ NPM, 4 conditions; $R^2 = 93.1\%, 94.4\%, 86.1\%$ for $H_0, H_1, H_2$) | Weggen et al. (2024), `10.3389/fbioe.2024.1403644`, Table 1, Sec. 3.1 & 3.3.2 |
| **Tier 2**: Chain-resolved $\text{DAR } 8$ `Detailed 4k` (`ADC2 + NPM`, reducing RP-UHPLC) | $k_1 (\text{LC}), k_2 (\text{HC}_1), k_3 (\text{HC}_2), k_{4/5} (\text{HC}_{3,4})$ | $k_1 = 1.220$ ($\sigma = 0.003$, CI $[1.214, 1.227]$); $k_2 = 1.853$ ($\sigma = 0.006$, CI $[1.841, 1.865]$); $k_3 = 5.117$ ($\sigma = 1.015$, CI $[3.101, 7.133]$); $k_{4/5} = 2.312$ ($\sigma = 0.356$, CI $[1.605, 3.020]$) $\text{L}\cdot\text{mmol}^{-1}\cdot\text{s}^{-1}$ | $1220.0$, $1853.0$, $5117.0$, $2312.0\text{ L}\cdot\text{mol}^{-1}\cdot\text{s}^{-1}$ | Dataset 3 ($4\text{ mL}$ fed-batch with $10, 20, 30\text{ min}$ syringe-pump feed & $1\text{ mL}$ batch, $1.5\text{–}3\text{ g/L}$ mAb, $6\text{–}13\times$ NPM; $\text{RMSE}_{\text{CV}} = 0.620\ \mu\text{mol/L}$, $\text{RMSE}_{\text{P}} = 0.964\ \mu\text{mol/L}$) | Weggen et al. (2024), `10.3389/fbioe.2024.1403644`, Table 1 & Table 2 |
| **Tier 2**: IgG1 TCEP Disulfide Reduction (`WithInt + ReOx` `8k-model`, non-reducing CGE-SDS) | 3 inter-HL reduction rates ($k_{\text{HL},1..3}$), 2 hinge inter-HH reduction rates ($k_4, k_5$), 3 inter-HL re-oxidation rates at $T_{\text{ref}} = 293.15\text{ K}$ | $k_{\text{HL},1..3}$ decrease from $6.98 \times 10^{-4}$ to $6.81 \times 10^{-5}\text{ L}\cdot\mu\text{mol}^{-1}\cdot\text{s}^{-1}$; $k_5 / k_4 = 18$; initial seeds $k_{T_{\text{ref}},0} = 1.0 \times 10^{-4}\text{ L}\cdot\mu\text{mol}^{-1}\cdot\text{s}^{-1}$, $E_{a,0} = 20\text{ kJ/mol}$ (individual bar values & CIs in Fig. 3 & SI Table S4) | $k_{\text{HL},1..3}$ decrease from $698.0$ to $68.1\text{ L}\cdot\text{mol}^{-1}\cdot\text{s}^{-1}$; $k_5 / k_4 = 18$; seed $k_{T_{\text{ref}},0} = 100.0\text{ L}\cdot\text{mol}^{-1}\cdot\text{s}^{-1}$ ($\times 10^6\ \mu\text{mol/mol}$) | $T \in [277.15, 310.15]\text{ K}$ ($4\text{–}37\ ^\circ\text{C}$), $c_{\text{mAb},0} \in [3.75, 20]\text{ g/L}$ ($25\text{–}133\ \mu\text{mol/L}$), TCEP excess $2\text{–}10\times$ ($\text{AIC}_{\text{c}} = 24.96$ vs. $35.84$ for `9k-model`) | Weggen et al. (2025), `10.1002/bit.28899`, Table 2, Sec. 2.3.1, 2.3.3, 3.1.1 & 3.1.2 |

---

## 4. Public dataset availability, data-access gap, and fallback policy

1. **Industrial and literature data-access gap**:
   - Eli Lilly and Company internal bioconjugation reaction trajectories and
     inline spectroscopic datasets are proprietary and cannot be committed to the
     public repository without formal data-release clearance.
   - In the open-access KIT/AstraZeneca publications, the complete ODE equations,
     initial concentrations, reactor volumes, syringe-pump dosing durations,
     calibrated rate constants, standard deviations, $95\%$ confidence intervals,
     and cross-validation/prediction errors are published in the main text, while
     the raw time-series measurements have the following availability:
     - **Weggen et al. (2023)** (*Front. Bioeng. Biotechnol.* 11, 1123842,
       `PMC10111256`, CC BY 4.0) and **Weggen et al. (2024)** (*Front. Bioeng.
       Biotechnol.* 12, 1403644, `PMC11274341`, CC BY 4.0): experimental time
       courses are plotted in Figures 1, 2, 4, 5, 6 and Supplementary Figures
       S3–S9, with the Data Availability Statement *"The raw data supporting the
       conclusion of this article will be made available by the authors, without
       undue reservation."*
     - **Weggen et al. (2025)** (*Biotechnol. Bioeng.* 122(2), 342–356,
       `PMC11808428`, CC BY 4.0): Section 2.1.3.2 notes that CGE and RP-UHPLC
       data are referenced as Supporting Information S2 (`bit28899-suppl-0002`),
       and the Data Availability Statement states *"The data that support the
       findings of this study are available from the corresponding author upon
       reasonable request."*

2. **Public fallback and provenance-tagging contract**:
   - In accordance with `AGENTS.md` and `#351` (`CalibrationRecord`), repository
     verification fixtures and tutorials use **two distinct, explicitly tagged
     `DataSource` records**:
     - `origin='literature'`: records the peer-reviewed rate constants, initial
       reactant concentrations, reactor volumes, feed rates, and $248/280\text{ nm}$
       UV-Vis extinction coefficients from Weggen et al. (2023, 2024, 2025) and
       Hamblett et al. (2004), citing the exact DOI, page/equation/table numbers,
       and CC BY 4.0 license terms where applicable.
     - `origin='synthetic'`: records forward-simulated concentration and spectral
       time-course observations generated from those literature models (pinned by
       the SHA-256 digest of the serialized synthetic observation arrays and
       generator configuration), explicitly stating in `sharing` and `description`
       that the time series are synthetic verification data and **not** an
       empirical calibration dataset.
   - During M7–M8 (`#356`), the CC BY 4.0 time-series figures in Weggen et al.
     (2023, Fig. 4) and Weggen et al. (2024, Figs. 4–6) can also be digitized
     (recording the digitizer tool, axis-calibration uncertainty, and SHA-256
     digest as `origin='literature'` with a digitization `Transformation`) or
     supplemented with author-shared Supporting Information S2 tables from
     KIT/AstraZeneca or a cleared Lilly benchmark dataset.

---

## 5. Mixing assumptions, operating modes, and Damköhler scaling analysis

Following Vicum et al. (2004, Eqs. 5–6) and Weggen et al. (2023, Sec. 2.2.8,
Eqs. 9–10), the characteristic chemical reaction timescale $\tau_{R,i}$ $[\text{s}]$
of a bimolecular conjugation step with rate constant $k_i$
$[\text{L}\cdot\text{mol}^{-1}\cdot\text{s}^{-1}]$ and local reactant
concentrations $\tilde{c}_1, \tilde{c}_2$ $[\text{mol/L}]$ is:

$$\tau_{R,i} = \frac{1}{k_i \left(\tilde{c}_1 + \tilde{c}_2\right)} \quad [\text{s}]$$

Two complementary Damköhler metrics govern whether an ideal well-mixed 0D
reactor model (`BatchReactor` or `SemibatchReactor`) is sufficient:

1. **Micromixing engulfment Damköhler number** $\text{Da}_{E,i}$ $[-]$
   (Bałdyga & Bourne, 1999; Vicum et al., 2004, Eq. 5; Weggen et al., 2023,
   Sec. 2.2.8, Eqs. 9–10), which compares the turbulent engulfment time
   $\tau_E = 17.24 \left(\nu / \varepsilon\right)^{1/2}$ $[\text{s}]$ (evaluated
   with the measured conjugation buffer kinematic viscosity
   $\nu = \mu_{\text{liq}} / \rho_{\text{liq}} = (1.06 \times 10^{-3}\text{ Pa}\cdot\text{s}) / (1010.5\text{ kg/m}^3) = 1.049 \times 10^{-6}\text{ m}^2/\text{s}$,
   Weggen et al., 2023, Sec. 2.2.2, and vessel-mean specific power dissipation
   $\varepsilon = (P/V)/\rho_{\text{liq}}$ $[\text{m}^2/\text{s}^3]$) against
   $\tau_{R,i}$:
   $$\text{Da}_{E,i} = \frac{\tau_E}{\tau_{R,i}} = 17.24 \left(\frac{\nu}{\varepsilon}\right)^{1/2} k_i \left(\tilde{c}_1 + \tilde{c}_2\right) \quad [\text{-}]$$
   Note that in surface-dosed stirred tanks, local turbulent energy dissipation
   $\varepsilon_{\text{local}}$ at the liquid surface feed point is substantially
   lower than the vessel-mean $(P/V)/\rho_{\text{liq}}$ (which is concentrated near
   the impeller), so vessel-mean $\tau_E$ is a lower bound on the local feed-zone
   micromixing time.
2. **Macromixing vessel-blend screening index** $\text{Da}_{\text{blend},i}$
   $[-]$ (conservative vessel-scale upper bound using the global $95\%$
   homogenization time $\tau_{\text{mix,95}}$ $[\text{s}]$ reported in Weggen
   et al., 2023, Table 1):
   $$\text{Da}_{\text{blend},i} = \frac{\tau_{\text{mix,95}}}{\tau_{R,i}} = k_i \left(\tilde{c}_1 + \tilde{c}_2\right) \tau_{\text{mix,95}} \quad [\text{-}]$$

Using the first conjugation step of Model 1A
($k_1 = 797.0\text{ L}\cdot\text{mol}^{-1}\cdot\text{s}^{-1}$) at the standard
manufacturing condition of Weggen et al. (2023, Table 2:
$\rho_{\text{mAb},0} = 5.0\text{ g/L} \implies c_{\text{mAb},0} = 3.333 \times 10^{-5}\text{ mol/L}$,
$5\times$ molar payload excess $\implies c_{\text{Drug},0}^{\text{bulk}} = 1.667 \times 10^{-4}\text{ mol/L}$,
and concentrated feed stock $c_{\text{Drug,feed}} = 10.0\text{ mmol/L} = 1.00 \times 10^{-2}\text{ mol/L}$):

- **Bulk well-mixed reaction timescale**:
  $$\tau_{R,1}^{\text{bulk}} = \frac{1}{797.0 \times \left(3.333 \times 10^{-5} + 1.667 \times 10^{-4}\right)} = 6.274\text{ s}$$
- **Unmixed bolus feed-plume reaction timescale** (upper-bound screening scale combining undiluted feed-nozzle payload $\tilde{c}_{\text{Drug}} = c_{\text{Drug,feed}} = 1.00 \times 10^{-2}\text{ mol/L}$ with bulk $\tilde{c}_{\text{mAb}} = 3.333 \times 10^{-5}\text{ mol/L}$ at the initial contact interface):
  $$\tau_{R,1}^{\text{plume}} = \frac{1}{797.0 \times \left(3.333 \times 10^{-5} + 1.00 \times 10^{-2}\right)} = 0.1251\text{ s}$$

| Vessel / Reactor Configuration | Liquid Volume $V$ $[\text{L}]$ | Blend Time $\tau_{\text{mix,95}}$ $[\text{s}]$ | Vessel-Mean Engulfment Time $\tau_E$ $[\text{s}]$ | Bulk $\text{Da}_{\text{blend},1}^{\text{bulk}}$ $[-]$ | Plume $\text{Da}_{\text{blend},1}^{\text{plume}}$ $[-]$ (bolus) | Plume $\text{Da}_{E,1}^{\text{plume}}$ $[-]$ (vessel-mean $\varepsilon$) | Max $\|\Delta\text{DAR}(t)\|$ (0D vs. 3D CFD) | PharmaPy Unit-Operation Mapping & Assumption |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Internally stirred $1\text{–}4\text{ mL}$ tube / cuvette (`origin="synthetic"` engineering bound) | $1.0 \times 10^{-3}\text{–}4.0 \times 10^{-3}$ | $\approx 0.5$ (bound) | $\approx 0.02$ (bound) | $0.080$ | $4.00$ | $0.16$ | $< 0.002$ (bound) | `BatchReactor` or `SemibatchReactor` (ideal 0D well-mixed assumption valid; $\text{Da}_{\text{blend},1}^{\text{bulk}} \ll 1$) |
| Lab-scale glass stirred tank (`GST-1`, $60\text{ rpm}$, $P/V = 3.81\text{ W/m}^3$, Table 1) | $0.300$ | $9.4$ | $0.288$ | $1.50$ | $75.2$ | $2.30$ | $0.0035$ ($60\text{ s}$ feed, Sec. 3.2.3) | `SemibatchReactor` with $t_{\text{add}} \ge 60\text{ s}$ (ideal 0D model agrees with experimental DAR time course to $R^2 = 0.979$, Sec. 3.2.2) |
| Large-scale single-use mixer (`SUM`, $400\text{ rpm}$, $P/V = 23.71\text{ W/m}^3$, Table 1) | $25.0$ | $17.6$ | $0.115$ | $2.81$ | $141$ | $0.922$ | $0.035$ ($60\text{ s}$ feed, Sec. 3.2.3; surface dosing experiences lower local $\varepsilon$ than vessel-mean $P/V$) | `SemibatchReactor` with subsurface payload dosing ($t_{\text{add}} = 60\text{–}300\text{ s}$) |
| Large-scale glass stirred tank (`GST-2`, $60\text{ rpm}$, $P/V = 2.44\text{ W/m}^3$, Table 1) | $22.0$ | $32.2$ | $0.359$ | $5.13$ | $257$ | $2.87$ | $0.20$ (bolus, $t_{\text{add}}=0\text{ s}$); $0.08$ ($60\text{ s}$); $< 0.01$ ($300\text{ s}$) | **Mixing-sensitive under bolus addition** ($\text{Da}_{\text{blend},1}^{\text{plume}} \gg 1$, $\text{Da}_{E,1}^{\text{plume}} > 1$); requires `SemibatchReactor` controlled addition ($t_{\text{add}} \ge 300\text{ s}$) to restore 0D validity |
| Continuous flow microreactor / static mixer (Nakahara et al., 2022, `10.1021/acs.oprd.2c00217`; `origin="synthetic"` engineering bound) | $1.0 \times 10^{-3}\text{–}1.0 \times 10^{-2}$ | $< 0.05$ (bound) | $< 0.005$ (bound) | $< 0.008$ | $< 0.40$ | $< 0.04$ | Negligible (bound) | `Mixer` + `PlugFlowReactor` (plug-flow axial transport with rapid radial micromixing) |

**Key modeling conclusion for WP1.5**:
1. In small-scale kinetic tubes ($1\text{–}4\text{ mL}$) and in pilot/commercial
   tanks operated in **controlled semi-batch mode** (`SemibatchReactor` with
   payload addition time $t_{\text{add}} \gg \tau_{\text{mix,95}}$, e.g.,
   $t_{\text{add}} = 300\text{–}1800\text{ s}$), the ideal well-mixed 0D
   concentration balance accurately predicts the species trajectories and final
   $\text{DAR}$ without 3D CFD.
2. Furthermore, in parameter estimation (`Weggen et al., 2024`), operating in
   `SemibatchReactor` mode ($t_{\text{add}} = 600\text{–}1800\text{ s}$)
   deliberately stretches the reaction timescale from $< 60\text{ s}$ to
   $10\text{–}30\text{ min}$, making intermediate species trajectories ($H_1^2$,
   $\text{mAb}_1$) observable by offline/online PAT and improving parameter
   identifiability.

---

## 6. Spectroscopic and analytical PAT measurement/prediction contract

### 6.1 Channel 1 — Quenched chromatographic species and DAR measurements (`Experiment` / `Measurement`)

Offline or at-line samples quenched with $N$-acetylcysteine (NAC) and analyzed by
non-reducing RP-UHPLC / HIC (Model 1A) or reducing RP-UHPLC (Model 1B) provide
direct molar concentration measurements $y_i(t_k)$ $[\text{mol/L}]$ of the
protein conjugate species and free payload. The derived quality attributes are:

- **Intact-antibody basis (Model 1A)**:
  $$c_{\text{mAb,total}}(t) = c_{\text{mAb}}(t) + c_{\text{mAb\_1Drug}}(t) + c_{\text{mAb\_2Drug}}(t) \quad [\text{mol/L}]$$
  $$\text{DAR}(t) = \frac{c_{\text{mAb\_1Drug}}(t) + 2 \, c_{\text{mAb\_2Drug}}(t)}{c_{\text{mAb,total}}(t)} \quad [\text{-}]$$
- **Reduced heavy-chain basis (Model 1B, 2 heavy chains per mAb)**:
  $$c_{H,\text{total}}(t) = c_{H_0^0}(t) + c_{H_0^1}(t) + c_{H_0^2}(t) + c_{H_1^1}(t) + c_{H_1^2}(t) + c_{H_2^2}(t) \quad [\text{mol/L}]$$
  $$\text{DAR}(t) = 2 \, \frac{c_{H_1^1}(t) + c_{H_1^2}(t) + 2 \, c_{H_2^2}(t)}{c_{H,\text{total}}(t)} \quad [\text{-}]$$

In PharmaPy, these observations are wrapped as `Measurement(field, x, values,
uncertainty, units='mol/L', basis='molar concentration', x_units='s')` inside an
`Experiment` (`PharmaPy.ParamEstim`).

### 6.2 Channel 2 — Dual- and multi-wavelength UV-Vis Beer–Lambert spectroscopic contract

#### Forward multi-wavelength observation operator and optical pathlength regime

For a spectroscopic flow cell or micro-volume cuvette of optical pathlength
$\ell$ $[\text{cm}]$ measuring absorbance across $N_\lambda$ wavelengths
$\boldsymbol{\lambda} = [\lambda_1, \dots, \lambda_{N_\lambda}]$ $[\text{nm}]$ at
$N_t$ time points, the Beer–Lambert linear superposition law gives the absorbance
matrix $\mathbf{S} \in \mathbb{R}^{N_t \times N_\lambda}$ $[\text{-}]$
($\text{AU}$):

$$\mathbf{S}(t_k, \lambda_m) = \ell \sum_{j=1}^{N_{\text{species}}} c_j(t_k) \, \varepsilon_j(\lambda_m) \quad [\text{-}], \qquad \mathbf{S} = \ell \, \mathbf{C} \, \mathbf{E}$$

where $\mathbf{C} \in \mathbb{R}^{N_t \times N_{\text{species}}}$ has units of
$[\text{mol/L}]$ and the molar extinction coefficient matrix
$\mathbf{E} \in \mathbb{R}^{N_{\text{species}} \times N_\lambda}$ has units of
$[\text{L}\cdot\text{mol}^{-1}\cdot\text{cm}^{-1}]$
($\text{M}^{-1}\cdot\text{cm}^{-1}$). Calling $\mathbf{A}_{\text{eff}} = \ell \, \mathbf{E}$
$[\text{L/mol}]$, $\mathbf{S} = \mathbf{C} \, \mathbf{A}_{\text{eff}}$ matches
the bilinear measurement equation `S = C @ A` of `MultipleCurveResolution` in
`PharmaPy.ParamEstim`.

- **Detector linearity and pathlength selection**: At a typical conjugation
  protein concentration $c_{\text{mAb,total}} \approx 3.33 \times 10^{-5}\text{ mol/L}$
  ($5.0\text{ g/L}$) with $\varepsilon_{\text{mAb}}^{280} = 2.3415 \times 10^5\text{ L}\cdot\text{mol}^{-1}\cdot\text{cm}^{-1}$,
  a standard $\ell = 1.0\text{ cm}$ cuvette yields an undiluted protein
  absorbance $A_{280} \approx 7.8\text{ AU}$, which saturates standard UV-Vis
  photodiodes (linear dynamic range $\le 1.5\text{–}2.5\text{ AU}$). In practice,
  UV-Vis PAT uses either a **short-pathlength flow cell / micro-volume plate**
  ($\ell = 0.05\text{–}0.10\text{ cm}$, such as the Unchained Labs Stunner used
  in Weggen et al., 2024, Sec. 2.1.4, giving $A_{280} \approx 0.78\text{ AU}$ at
  $\ell = 0.10\text{ cm}$) or $5\text{–}10\times$ inline dilution prior to a
  $\ell = 1.0\text{ cm}$ cell.

#### Dual-wavelength UV-Vis inversion (Hamblett et al., 2004; Wakankar et al., 2011; Chen, 2013)

At two diagnostic wavelengths—$\lambda_1 = \lambda_{\text{drug,max}}$ (e.g.,
$248\text{ nm}$ for vc-MMAE in Hamblett et al., 2004, p. 7064,
`doi:10.1158/1078-0432.CCR-04-0789`; $252\text{ nm}$ for DM1 in Wakankar et al.,
2011, `doi:10.4161/mabs.3.2.14960`; $260\text{ nm}$ for oligonucleotide payloads
in AOCs/ARCs) and $\lambda_2 = 280\text{ nm}$ (aromatic Trp/Tyr protein
absorbance maximum)—the total antibody chromophore concentration
$c_{\text{mAb,total}}$ $[\text{mol/L}]$ and total drug chromophore concentration
$c_{D,\text{chrom}}$ $[\text{mol/L}]$ satisfy:

$$\begin{bmatrix} A(\lambda_1) \\ A(280) \end{bmatrix}
= \ell \begin{bmatrix}
\varepsilon_{\text{mAb}}^{\lambda_1} & \varepsilon_{D}^{\lambda_1} \\
\varepsilon_{\text{mAb}}^{280} & \varepsilon_{D}^{280}
\end{bmatrix}
\begin{bmatrix} c_{\text{mAb,total}} \\ c_{D,\text{chrom}} \end{bmatrix} \quad [\text{-}]$$

With determinant
$\Delta_\varepsilon = \varepsilon_{\text{mAb}}^{\lambda_1}\varepsilon_{D}^{280} - \varepsilon_{\text{mAb}}^{280}\varepsilon_{D}^{\lambda_1} \ne 0$
$[\text{L}^2\cdot\text{mol}^{-2}\cdot\text{cm}^{-2}]$, the exact Cramér's-rule
concentration inversion and absorbance-ratio $\text{DAR}$ equation (for a
purified/desalted conjugate where $c_{D,\text{chrom}} = c_{D,\text{conj}}$) are:

$$c_{\text{mAb,total}} = \frac{\varepsilon_{D}^{280} A(\lambda_1) - \varepsilon_{D}^{\lambda_1} A(280)}{\ell \, \Delta_\varepsilon} \quad [\text{mol/L}], \qquad
c_{D,\text{chrom}} = \frac{\varepsilon_{\text{mAb}}^{\lambda_1} A(280) - \varepsilon_{\text{mAb}}^{280} A(\lambda_1)}{\ell \, \Delta_\varepsilon} \quad [\text{mol/L}]$$

$$\text{DAR} = \frac{c_{D,\text{conj}}}{c_{\text{mAb,total}}}
= \frac{\varepsilon_{\text{mAb}}^{\lambda_1} - R \, \varepsilon_{\text{mAb}}^{280}}{R \, \varepsilon_{D}^{280} - \varepsilon_{D}^{\lambda_1}}, \qquad
R = \frac{A(\lambda_1)}{A(280)} \quad [\text{-}]$$

#### Provenance separation of literature vs. synthetic test extinction coefficients

| Chromophore / Wavelength Channel | $\varepsilon(\lambda)$ $[\text{L}\cdot\text{mol}^{-1}\cdot\text{cm}^{-1}]$ | Provenance Classification (`origin`) | Physical Basis, Source & Primary-Text Reconciliation |
| --- | --- | --- | --- |
| Monoclonal antibody (`cAC10` IgG1, $\text{mAb}_0$) at $248\text{ nm}$ / $280\text{ nm}$ | $\varepsilon_{\text{mAb}}^{248} = 9.413 \times 10^4$, $\varepsilon_{\text{mAb}}^{280} = 2.3415 \times 10^5$ | `origin="literature"` | Intact antibody molar basis; Hamblett et al. (2004), *Clin. Cancer Res.* 10(20), 7063–7070, pp. 7064–7065 (`doi:10.1158/1078-0432.CCR-04-0789`) |
| Maleimide linker-payload (`vc-MMAE`, `SGD-1006`) at $248\text{ nm}$ / $280\text{ nm}$ | Physical assignment: $\varepsilon_{D}^{248} = 1.59 \times 10^4$, $\varepsilon_{D}^{280} = 1.50 \times 10^3$ (printed on p. 7065 with wavelength labels transposed as $\varepsilon_{\text{drug}}^{248} = 1.50 \times 10^3$, $\varepsilon_{\text{drug}}^{280} = 1.59 \times 10^4$) | `origin="literature"` (with documented primary-text label transposition) | Per mole of drug-linker moiety; Hamblett et al. (2004, p. 7064) explicitly state that $248\text{ nm}$ and $280\text{ nm}$ *"are the absorbance maxima of the drug and the antibody, respectively"*, which requires $\varepsilon_{D}^{248} > \varepsilon_{D}^{280}$ ($1.59 \times 10^4 > 1.50 \times 10^3\text{ M}^{-1}\text{cm}^{-1}$; see also Chen, 2013). Cramér's-rule inversion holds algebraically for either non-singular pair ($\Delta_\varepsilon \ne 0$); payload-specific extinction spectra for `#356` will be confirmed with `@dcasasor` |
| 4-wavelength verification matrix ($\lambda \in \{248, 280, 310, 342\}\text{ nm}$) for $\text{mAb}$, conjugated site ($\boldsymbol{\varepsilon}_{\text{conj\_site}}$), free maleimide $\text{Drug}$, and depleted $\text{Drug\_depl}$ | $\boldsymbol{\varepsilon}_{\text{mAb}} = [94130, 234150, 1200, 150]$; $\boldsymbol{\varepsilon}_{\text{conj\_site}} = [15900, 1500, 8200, 4100]$; $\boldsymbol{\varepsilon}_{\text{Drug}} = [15900, 1500, 14500, 9800]$; $\boldsymbol{\varepsilon}_{\text{Drug\_depl}} = [14200, 1200, 2100, 600]$ | `origin="synthetic"` (all `Drug_depl` values and all $310/342\text{ nm}$ columns are synthetic conditioning-driven test fixtures) | **Synthetic verification fixture** (`tests/test_bioconjugation_case_selection.py`) constructed with full row rank 4 to verify `MultipleCurveResolution` and `PCR_calibration` numerical recovery when pairing the Weggen et al. (2024) kinetic trajectory with a multi-wavelength UV-Vis observation operator. Not claimed as an empirical literature spectrum |

#### In-situ optical rank, stoichiometry invariants, and hybrid soft-sensor observability (Schiemer et al., 2023, `10.1002/bit.28428`)

During the conjugation reaction (prior to Stage 3 UF/DF desalting), the reaction
mixture contains intact protein species ($\text{mAb}_0$, $\text{mAb}_1$,
$\text{mAb}_2$) alongside free reactive payload ($\text{Drug}$) and quenched or
hydrolyzed payload ($\text{Drug\_depl}$). Writing the bound conjugate spectra as
an additive combination of the antibody core spectrum
$\boldsymbol{\varepsilon}_{\text{mAb}}(\lambda)$ and $k$ conjugated drug-linker
sites $\boldsymbol{\varepsilon}_{\text{conj\_site}}(\lambda)$,

$$\boldsymbol{\varepsilon}_{\text{mAb\_1Drug}}(\lambda) = \boldsymbol{\varepsilon}_{\text{mAb}}(\lambda) + \boldsymbol{\varepsilon}_{\text{conj\_site}}(\lambda), \qquad
\boldsymbol{\varepsilon}_{\text{mAb\_2Drug}}(\lambda) = \boldsymbol{\varepsilon}_{\text{mAb}}(\lambda) + 2\,\boldsymbol{\varepsilon}_{\text{conj\_site}}(\lambda)$$

creates two exact linear algebraic constraints on in-situ UV-Vis spectroscopy:

1. **Linear-independence condition for reaction progress to be optically observable in a closed batch**:
   Using the antibody conservation invariant
   $c_{\text{mAb,total}} = c_{\text{mAb}}(t) + c_{\text{mAb\_1Drug}}(t) + c_{\text{mAb\_2Drug}}(t)$
   and drug-moiety conservation invariant
   $c_{\text{Drug,total}} = c_{\text{Drug}}(t) + c_{\text{Drug\_depl}}(t) + c_{\text{conj\_sites}}(t)$
   where $c_{\text{conj\_sites}}(t) = c_{\text{mAb\_1Drug}}(t) + 2\,c_{\text{mAb\_2Drug}}(t)$
   $[\text{mol/L}]$, the mixture spectrum in a closed batch reactor reduces to:
   $$\frac{\mathbf{S}(t, \lambda)}{\ell} = c_{\text{mAb,total}}\,\boldsymbol{\varepsilon}_{\text{mAb}}(\lambda) + c_{\text{Drug,total}}\,\boldsymbol{\varepsilon}_{\text{Drug}}(\lambda) + c_{\text{conj\_sites}}(t)\underbrace{\left[\boldsymbol{\varepsilon}_{\text{conj\_site}}(\lambda) - \boldsymbol{\varepsilon}_{\text{Drug}}(\lambda)\right]}_{\Delta\boldsymbol{\varepsilon}_{\text{conj}}(\lambda)} + c_{\text{Drug\_depl}}(t)\underbrace{\left[\boldsymbol{\varepsilon}_{\text{Drug\_depl}}(\lambda) - \boldsymbol{\varepsilon}_{\text{Drug}}(\lambda)\right]}_{\Delta\boldsymbol{\varepsilon}_{\text{depl}}(\lambda)}$$
   Therefore:
   - If $\Delta\boldsymbol{\varepsilon}_{\text{conj}}(\lambda) = \mathbf{0}$
     and $\Delta\boldsymbol{\varepsilon}_{\text{depl}}(\lambda) = \mathbf{0}$ at the
     monitored wavelengths (e.g., if conjugated and free drug have identical
     extinction coefficients at $248\text{ nm}$ and $280\text{ nm}$), the closed-batch
     spectrum is **strictly time-invariant** ($\partial \mathbf{S}/\partial t = \mathbf{0}$),
     and unassisted post-desalting dual-wavelength UV-Vis cannot track the closed-batch
     reaction in situ.
   - To resolve the two time-varying reaction extents $c_{\text{conj\_sites}}(t)$ and
     $c_{\text{Drug\_depl}}(t)$ alongside the background coordinates
     $(c_{\text{mAb,total}}, c_{\text{Drug,total}})$ from spectra alone, it is not
     sufficient that $\Delta\boldsymbol{\varepsilon}_{\text{conj}}(\lambda) \ne \mathbf{0}$
     and $\Delta\boldsymbol{\varepsilon}_{\text{depl}}(\lambda) \ne \mathbf{0}$: the four
     pure-component row vectors of $\mathbf{E}_{4 \times N_\lambda}$
     ($[\boldsymbol{\varepsilon}_{\text{mAb}},\, \boldsymbol{\varepsilon}_{\text{conj\_site}},\, \boldsymbol{\varepsilon}_{\text{Drug}},\, \boldsymbol{\varepsilon}_{\text{Drug\_depl}}]^\top$,
     or equivalently $[\boldsymbol{\varepsilon}_{\text{mAb}},\, \boldsymbol{\varepsilon}_{\text{Drug}},\, \Delta\boldsymbol{\varepsilon}_{\text{conj}},\, \Delta\boldsymbol{\varepsilon}_{\text{depl}}]^\top$)
     must be **linearly independent** ($\operatorname{rank}(\mathbf{E}_{4 \times N_\lambda}) = 4$
     with well-conditioned singular values across $N_\lambda \ge 4$ wavelengths). If
     $\Delta\boldsymbol{\varepsilon}_{\text{conj}}(\lambda)$ and
     $\Delta\boldsymbol{\varepsilon}_{\text{depl}}(\lambda)$ are nonzero but collinear
     (e.g., if both conjugation and hydrolysis only eliminate the same maleimide
     ring transition), pure optical deconvolution can only observe the combined
     consumed maleimide $c_{\text{conj\_sites}}(t) + c_{\text{Drug\_depl}}(t)$ and
     requires the kinetic rate ratio $k_1 c_{\text{SH}} / k_3$ to separate bound
     drug from hydrolyzed drug.
2. **Optical rank-4 limitation for individual intact species ($\text{mAb}_0$, $\text{mAb}_1$, $\text{mAb}_2$) and hybrid kinetic-model resolution**:
   Because $\boldsymbol{\varepsilon}_{\text{mAb\_1Drug}} = \frac{1}{2}\left(\boldsymbol{\varepsilon}_{\text{mAb}} + \boldsymbol{\varepsilon}_{\text{mAb\_2Drug}}\right)$,
   two moles of $\text{mAb}_1$ have the exact same UV-Vis spectrum as one mole of
   $\text{mAb}_0$ plus one mole of $\text{mAb}_2$. The 5-species stoichiometry matrix
   $\mathbf{E} \in \mathbb{R}^{5 \times N_\lambda}$ for
   $[\text{mAb}_0, \text{mAb}_1, \text{mAb}_2, \text{Drug}, \text{Drug\_depl}]$
   therefore has **maximum column rank 4** regardless of $N_\lambda$: pure optical
   deconvolution can recover $c_{\text{mAb,total}}(t)$ and $c_{\text{conj\_sites}}(t)$
   (when $\operatorname{rank}(\mathbf{E}_{4 \times N_\lambda}) = 4$),
   but cannot algebraically separate $c_{\text{mAb\_1Drug}}(t)$ from
   $c_{\text{mAb\_2Drug}}(t)$ without a separation step (RP-UHPLC/HIC) or a kinetic
   model. In the adaptive soft-sensor architecture of Schiemer et al. (2023,
   *Biotechnol. Bioeng.* 120(7), 1914–1928, `doi:10.1002/bit.28428`), UV-Vis
   absorbance spectra are calibrated via chemometric/data-driven regression to
   predict the optically observable bound-drug/reaction coordinate and fused with
   the pre-calibrated consecutive kinetic ODE model (`BatchReactor` /
   `SemibatchReactor`, $k_1, k_2, k_3$) to reconstruct the individual intact
   conjugate species $c_{\text{mAb}}(t)$, $c_{\text{mAb\_1Drug}}(t)$, and
   $c_{\text{mAb\_2Drug}}(t)$ in real time.

### 6.3 Channel 3 — In-line Raman spectroscopy for Stage 1 disulfide reduction (and prospective Stage 2 monitoring)

In-line Raman spectroscopy has been experimentally demonstrated for **Stage 1
monoclonal antibody disulfide reduction** by Pople et al. (2026, *Pharm. Res.*,
`doi:10.1007/s11095-026-04154-9`), tracking disulfide $\nu(\text{S–S})$
stretching ($500\text{–}550\text{ cm}^{-1}$) and conformational amide/aromatic
markers during TCEP reduction. Extending in-line Raman to **Stage 2
thiol–maleimide conjugation**—via thiol $\nu(\text{S–H})$ stretching
($2550\text{–}2600\text{ cm}^{-1}$), maleimide alkene $\nu(\text{C}=\text{C})$
stretching ($1580\text{–}1600\text{ cm}^{-1}$), and imide carbonyl
$\nu(\text{C}=\text{O})$ bands ($1700\text{–}1770\text{ cm}^{-1}$)—is a
**prospective PAT extension** for WP1.5 rather than an established primary claim
of Pople et al. (2026) (because micromolar protein thiol concentrations in
dilute aqueous solution are weak Raman scatterers and require surface-enhanced or
high-sensitivity chemometric calibration). Both Stage 1 Raman reduction spectra
and prospective Stage 2 Raman spectra map through PharmaPy's `PCR_calibration`
(`PharmaPy.Calibration`, with optional SNV preprocessing repaired in `#138` and
`#253`, and PLS expansion tracked in `#352`) to predict molar species
concentrations $[\text{mol/L}]$.

---

## 7. Existing PharmaPy reaction/estimation paths and demonstrated blockers

### 7.1 Verified working paths on `master` (`3194efd0`)

| Capability | PharmaPy Class / Module | Status on `master` (`3194efd0`) & Verification Evidence |
| --- | --- | --- |
| Elementary & custom reaction kinetics and Jacobians | `RxnKinetics` (`PharmaPy/Kinetics.py`) | Ready: parses `rxn_list`, computes $\mathbf{r}_{\text{species}}$ $[\text{mol}\cdot\text{L}^{-1}\cdot\text{s}^{-1}]$, analytical state Jacobian $\partial \mathbf{r}/\partial \mathbf{c}$, and parameter Jacobian $\partial \mathbf{r}/\partial \boldsymbol{\theta}$ (`tests/test_bioconjugation_case_selection.py`, `tests/test_kinetics_custom_and_equilibrium.py`, `tests/test_reaction_kinetics_validation_and_parameters.py`) |
| Batch, fed-batch, and plug-flow reactor simulation & sensitivities | `BatchReactor`, `SemibatchReactor`, `PlugFlowReactor` (`PharmaPy/Reactors.py`) | Ready for state integration across `BatchReactor`, `SemibatchReactor`, and `PlugFlowReactor` (`tests/test_reactor_modes.py`, `tests/test_reactor_correctness.py`, `tests/test_reactor_sensitivity_jacobian.py`, `tests/test_bioconjugation_case_selection.py`). Forward sensitivity integration (`eval_sens=True`, stored on `batch_reactor.sensit` and returned by `solve_unit`) is ready on `BatchReactor` (`#55` closed), whereas `SemibatchReactor.solve_unit(eval_sens=True)` currently raises `NotImplementedError` (`PharmaPy/Reactors.py:3064`, tracked in [#486](https://github.com/PharmaPy-org/PharmaPy/issues/486) and Section 7.2) |
| Mixing and hold-tank unit operations | `Mixer`, `DynamicCollector` (`PharmaPy/Containers.py`) | Ready for inline dilution/mixing (`Mixer`) and dynamic hold/pooling (`DynamicCollector`), verified in `tests/test_mixer_container_balances.py`, `tests/test_dynamic_collector_inventory_balance.py`, and `tests/test_dynamic_collector_state_ordering.py` |
| Structured experiment & measurement containers | `Experiment`, `Measurement`, `ParameterEstimation` (`PharmaPy/ParamEstim.py`) | Ready (`#484`, `#236`, `#238`, `#239`, `#252` closed): handles multi-experiment batch + fed-batch fits, missing observations (`NaN`), per-sample uncertainty weighting, `estimate_parameters`, and `get_confidence_intervals` (`tests/test_paramestim_experiment_objects.py`, `tests/test_simexec_experiment_alignment.py`, `tests/test_bioconjugation_case_selection.py`) |
| Calibration provenance & deterministic replay | `CalibrationRecord`, `DataSource`, `Transformation`, `ModelRevision` (`PharmaPy/Records.py`) | Ready (PR [#480](https://github.com/PharmaPy-org/PharmaPy/pull/480) merged under open WP1.1 issue [#351](https://github.com/PharmaPy-org/PharmaPy/issues/351)): JSON round-trip, SHA-256 source pinning, `origin='literature'` / `'synthetic'`, and `CalibrationRecord.rerun` (`tests/test_calibration_records_guide.py`, `tests/test_records_simexec_capture.py`, `tests/test_bioconjugation_case_selection.py`) |
| Chemometric & self-modeling spectral calibration | `MultipleCurveResolution` (`PharmaPy/ParamEstim.py`), `PCR_calibration` (`PharmaPy/Calibration.py`) | Ready (`#138`, `#240`, `#253`, `#254` closed): MCR bilinear $S = C A$ decomposition and PCR `get_regression`/`predict` with SNV preprocessing (`tests/test_calibration_snv_prediction.py`, `tests/test_calibration_prediction_diagnostics.py`, `tests/test_bioconjugation_case_selection.py`) |

### 7.2 Demonstrated open blockers and workarounds for `#356`

| Open Issue | Defect / Gap Summary | Impact on WP1.5 Bioconjugation Case (`#356`) | Disposition / Workaround on `master` |
| --- | --- | --- | --- |
| [#275](https://github.com/PharmaPy-org/PharmaPy/issues/275) | `RxnKinetics`: toggling `reformulate_kin=False` with a finite `temp_ref` switches to centered Arrhenius $k_i \exp(-(E_{a,i}/R)(1/T - 1/T_{\text{ref}}))$ instead of preserving standard Arrhenius | Affects multi-temperature Arrhenius fitting of Stage 1 TCEP disulfide reduction ($4\text{–}37\ ^\circ\text{C}$, $T_{\text{ref}} = 293.15\text{ K}$, Weggen et al., 2025) if comparing `reformulate_kin=True` vs. `False` at finite `temp_ref`; **does not affect isothermal Stage 2 conjugation** ($E_a = 0\text{ J/mol}$) | Keep `ea_params=[0.0, ...]` and `temp_ref=np.inf` for isothermal conjugation; resolve `#275` before multi-temperature Stage 1 TCEP Arrhenius comparison in `#356` |
| [#339](https://github.com/PharmaPy-org/PharmaPy/issues/339) | `RxnKinetics`: align reaction specifications and parameters by reaction identifier (`rxns={'coupling': ..., 'iso': ...}`) | Needed for the dictionary-keyed reaction syntax tracked in [#337](https://github.com/PharmaPy-org/PharmaPy/issues/337) and [#339](https://github.com/PharmaPy-org/PharmaPy/issues/339) | Use positional `rxn_list=[...]` and `k_params=[...]` on `master` until `#339` lands |
| [#338](https://github.com/PharmaPy-org/PharmaPy/issues/338), [#341](https://github.com/PharmaPy-org/PharmaPy/issues/341) | Named dictionary composition inputs on `LiquidPhase`/`LiquidStream` (`#338`) and dictionary indexing on `DynamicResult.retrieve_state` (`#341`) | Ergonomic API improvements tracked under the workflow epic [#337](https://github.com/PharmaPy-org/PharmaPy/issues/337) | Use database-ordered arrays and `output_names` tuple mapping on `master` until `#338`/`#341` land |
| [#283](https://github.com/PharmaPy-org/PharmaPy/issues/283), [#287](https://github.com/PharmaPy-org/PharmaPy/issues/287), [#290](https://github.com/PharmaPy-org/PharmaPy/issues/290) | Custom reaction-order species indexing (`#283`), reversible negative-concentration flooring (`#287`), and repeated species in one reaction string (`#290`) | Selected bioconjugation reactions are irreversible elementary reactions with distinct reactant/product tokens, so `#283`, `#287`, and `#290` are not triggered by Models 1A/1B | Avoid writing buffer/catalyst on both sides of a reaction string (`#290`) unless `#290` is merged |
| [#440](https://github.com/PharmaPy-org/PharmaPy/issues/440) | `Reactors`: result retrieval re-completes the solvent after a volume-changing reaction | In dilute aqueous bioconjugation ($c_{\text{mAb}} \le 10^{-4}\text{ mol/L}$, $c_{\text{Drug}} \le 10^{-3}\text{ mol/L}$), reaction volume change is $< 10^{-4}$, and `paramest_wrapper` reads integrated solute columns directly | Track `#440` for concentrated non-ideal liquid mixtures; negligible effect on dilute aqueous bioconjugation solute columns |
| [#352](https://github.com/PharmaPy-org/PharmaPy/issues/352), [#353](https://github.com/PharmaPy-org/PharmaPy/issues/353), [#354](https://github.com/PharmaPy-org/PharmaPy/issues/354), [#355](https://github.com/PharmaPy-org/PharmaPy/issues/355) | `[WP1.3/WP1.4][M6] Validate shared calibration, sensitivities and uncertainty` (`#352`), `[WP1.2] Property and thermodynamics interface for the PAT cases` (`#353`), `[WP1.3] Flowsheet-aware calibration and identifiability demonstration` (`#354`), and `[WP1.3][M6] Shared model-based experiment-design interface` (`#355`) | Shared calibration/uncertainty (`#352`), property interfaces (`#353`), flowsheet identifiability (`#354`), and MBDoE (`D`/`A`/`E`-optimality ranking across candidate feed durations $t_{\text{add}}$ and initial payload excesses, `#355`) will be consumed by `#356` | Weighted sensitivity Jacobians $\mathbf{J}$ and Fisher Information Matrix $\mathbf{F} = \mathbf{J}\mathbf{J}^\top$ are already computable from `ParameterEstimation.get_gradient(params, out_array=True)` on `master`; adopt shared `#352`–`#355` interfaces in M7–M12 once merged |
| [#486](https://github.com/PharmaPy-org/PharmaPy/issues/486) (`SemibatchReactor` forward sensitivities `eval_sens=True`) | `BatchReactor.solve_unit(eval_sens=True)` supports forward CVode sensitivity integration (`#55`), whereas `SemibatchReactor.solve_unit(eval_sens=True)` raises `NotImplementedError` (`PharmaPy/Reactors.py:3064`, reproduced in `tests/test_bioconjugation_case_selection.py` and `tests/test_reactor_modes.py`) | Fed-batch parameter estimation (`Weggen et al., 2024`, $10\text{–}30\text{ min}$ syringe-pump feed runs) cannot use `eval_sens=True` on `SemibatchReactor` | Run `SemibatchReactor` with `eval_sens=False` so `ParameterEstimation` uses finite-difference sensitivities (`dx_finitediff`), or land `#486` during M7–M8 (`#356`) |
| [#531](https://github.com/PharmaPy-org/PharmaPy/issues/531) (Micromolar CVode `atol` scaling on `BatchReactor` / `SemibatchReactor`) | Assimulo CVode's default absolute tolerance is `atol = 1e-6` $[\text{mol/L}]$, which is $3\%$ of a $3.33 \times 10^{-5}\text{ mol/L}$ ($5\text{ g/L}$) antibody concentration and produces $O(10^{-6})\text{ mol/L}$ negative-concentration undershoot on $\text{mAb}_0/\text{mAb}_1$ and unphysical $\text{DAR} > 2.0$ overshoot (reproduced in `tests/test_bioconjugation_case_selection.py` and tracked in [#531](https://github.com/PharmaPy-org/PharmaPy/issues/531)) | Directly affects all micromolar $[\text{mol/L}]$ bioconjugation reactor solves and parameter estimation wrappers | Always pass `sundials_opts={'rtol': 1e-8, 'atol': 1e-11}` (or `atol <= 1e-10` $[\text{mol/L}]$) to `solve_unit` and `run_args={'sundials_opts': {'rtol': 1e-8, 'atol': 1e-11}}` in `Experiment` / `SetParamEstimation` until `#531` is fixed |

---

## 8. Month 7–12 roadmap (`#356`) and WP4 process-sequence inputs (`#366`, `#367`)

### 8.1 Month 7–12 model, calibration, and notebook sequence toward D3b (`#356`)

1. **Months 7–8 (M7–M8 — Mechanistic Reactor + Mixing + PAT Forward/Inverse
   Implementation)**:
   - Implement the complete `BatchReactor` and `SemibatchReactor` flowsheet
     scripts for Models 1A and 1B using the pure-component bioconjugation
     database (`mAb`, `mAb_1Drug`, `mAb_2Drug`, `H0_0`–`H2_2`, `Drug`,
     `Drug_depl`, `buffer`).
   - Couple reactor state trajectories $\mathbf{C}(t)$ $[\text{mol/L}]$ to the
     Beer–Lambert UV-Vis observation operator ($\lambda \in [240, 360]\text{ nm}$,
     including the $248/280\text{ nm}$ dual-wavelength channel and short-pathlength
     $\ell = 0.10\text{ cm}$ flow cell) and to `MultipleCurveResolution` /
     `PCR_calibration` / `PLS_calibration` (`#352`) spectral deconvolution.
   - Quantify the semi-batch payload addition window ($t_{\text{add}} \in [0, 1800]\text{ s}$)
     across lab (`GST-1`, $0.3\text{ L}$) and manufacturing (`GST-2`, $22\text{ L}$;
     `SUM`, $25\text{ L}$) mixing timescales.
   - Digitize the CC BY 4.0 published time-series curves from Weggen et al.
     (2023, Fig. 4) and Weggen et al. (2024, Figs. 4–6) into versioned
     `DataSource(origin='literature')` JSON tables with documented digitizer
     transformations, and incorporate any raw Supporting Information S2 tables
     received from KIT/AstraZeneca or a cleared Lilly benchmark dataset.
2. **Months 9–10 (M9–M10 — Multi-Experiment Parameter Estimation, Identifiability,
   and Experiment Design)**:
   - Calibrate Model 1A (`2k + sink`) and Model 1B (`Simple 1k + sink` vs.
     `Detailed 2k + sink`) across combined batch and syringe-pump fed-batch
     `Experiment` / `Measurement` datasets through `SimulationExec` and
     `ParameterEstimation`.
   - Compute parameter covariance matrices, $95\%$ confidence intervals, and
     correlation matrices (`StatsModule`), reproducing the practical
     identifiability contrast between `Simple 1k` and `Detailed 2k` reported by
     Weggen et al. (2024, Table 2).
   - Execute a model-based experiment design (MBDoE, `#355`) ranking candidate
     fed-batch payload addition times ($t_{\text{add}} \in \{60, 600, 1200, 1800\}\text{ s}$)
     and molar payload ratios ($[2.5, 5.0, 7.5]\times$) by $D$-optimality
     ($\det \mathbf{F}$) and parameter collinearity to maximize identifiability of
     consecutive conjugation vs. payload depletion.
   - Export and archive the fitted `CalibrationRecord` JSON with verified
     `DataSource` provenance (`#351`, `#480`).
3. **Months 11–12 (M11–M12 — End-to-End 4-Step Flowsheet, WP4 Scheduling Export,
   and D3b Notebook Delivery)**:
   - Assemble the four-step bioconjugation manufacturing sequence
     (Reduction/Activation $\to$ Conjugation $\to$ Desalting/UF-DF $\to$
     Purification), evaluate continuous-flow `Mixer` + `PlugFlowReactor`
     alternatives alongside WP1.2/WP1.3 (`#353`, `#354`), and export the STN/RTN
     scheduling representation for WP4 (`#366`, `#367`).
   - Author the self-contained Sphinx/Jupyter tutorial notebook for deliverable
     D3b (`#356`) and complete domain review with `@dcasasor`.

### 8.2 Four-step bioconjugation process-sequence inputs for WP4 (`#366`, `#367`)

Work Package 4 (issues [#366](https://github.com/PharmaPy-org/PharmaPy/issues/366)
and [#367](https://github.com/PharmaPy-org/PharmaPy/issues/367)) maps a PharmaPy
flowsheet into State-Task Network (STN) and Resource-Task Network (RTN)
scheduling models. To ensure an unambiguous, mass-conservative contract between
WP1.5 and WP4:

- **Explicit physical basis of STN/RTN material proportions**: Every input and
  output state proportion $p_{s,t}$ $[-]$ in the canonical four-task table below
  is defined on an **exact liquid volume fraction basis $[\text{L/L}]$** so that
  $\sum_{s \in \text{In}(t)} p_{s,t} = 1.0000$ and
  $\sum_{s \in \text{Out}(t)} p_{s,t} = 1.0000$ hold identically for each task
  $t \in \{\text{T1}, \text{T2}, \text{T3}, \text{T4}\}$. Because neat DMSO in
  `T2_conjugation` has a higher liquid density ($\rho_{\text{DMSO}} = 1.1004\text{ kg/L}$)
  than dilute aqueous phosphate buffer ($\rho_{\text{buffer}} = 1.000\text{ kg/L}$,
  or $\rho_{\text{mix}} = 1.0105\text{ kg/L}$ at $10\%\text{ v/v}$ DMSO, Weggen et
  al., 2023, Sec. 2.2.2), both the exact liquid volumes $[\text{L}]$, total
  solution masses $[\text{kg}]$ (and mass fractions $[\text{kg/kg}]$), reagent
  stock concentrations $[\text{mol/L}]$, and protein active pharmaceutical
  ingredient (mAb/ADC solute) masses $[\text{kg}]$ are reported explicitly in the
  fifth column.
- **Canonical 4-task baseline (`Tier 2` direct TCEP partial reduction) vs.
  expanded `Tier 1` activation sub-tasks**:
  - **Canonical 4-task baseline (`T1`–`T4` in table below)**: `T1_reduction`
    models **single-step stoichiometric TCEP partial reduction** (`Tier 2`,
    Weggen et al., 2025, Sec. 2.2: $19.60\text{ L}$ native mAb at $10.204\text{ g/L}$
    $= 0.200\text{ kg}$ mAb, $1.333\text{ mmol}$ mAb + $0.40\text{ L}$ of
    $20.0\text{ mmol/L}$ aqueous TCEP stock $= 8.00\text{ mmol}$ TCEP, $6.0\text{ eq}$
    per mAb $\to 20.00\text{ L}$ reduced mAb at $10.00\text{ g/L}$, with zero
    liquid waste stream; any trace unreacted TCEP is quenched in situ with
    water-soluble PEG-azide per Kantner et al., 2017 prior to `T2`).
  - **Expanded `Tier 1` engineered-Cys (`ADC1`) activation sub-network
    (`T1a` $\to$ `T1b` $\to$ `T1c`)**: When scheduling the full `Tier 1`
    engineered-cysteine uncapping sequence (Weggen et al., 2023, Sec. 2.1.1;
    Weggen et al., 2024, Sec. 2.1.1), `T1_reduction` expands into three
    mass-balanced sub-tasks sharing `R_reduction_vessel` and `U_ufdf_skid`:
    1. `T1a_tcep_decap` on `R_reduction_vessel` ($\tau = 3600\text{ s}$):
       $19.60\text{ L}$ capped `ADC1` ($0.200\text{ kg}$) + $0.40\text{ L}$ of
       $50.0\text{ mmol/L}$ TCEP stock ($20.0\text{ mmol}$ TCEP, $15\text{ eq}$)
       $\to 20.00\text{ L}$ decapped mAb (`S1a_decapped_mAb`,proportion $1.0000$).
    2. `T1b_decap_ufdf` on `U_ufdf_skid` ($\tau = 3600\text{ s}$, $5.0$
       diavolumes): $20.00\text{ L}$ `S1a_decapped_mAb` ($0.1667\ [\text{L/L}]$)
       + $100.00\text{ L}$ diafiltration buffer ($0.8333\ [\text{L/L}]$) $\to$
       $19.60\text{ L}$ TCEP-free retentate (`S1b_exchanged_mAb`,
       $0.1633\ [\text{L/L}]$, $0.200\text{ kg}$ mAb) + $100.40\text{ L}$ TCEP/Cys
       permeate waste (`S_t1b_permeate_waste`, $0.8367\ [\text{L/L}]$).
    3. `T1c_dhaa_reox` on `R_reduction_vessel` ($\tau = 3600\text{ s}$):
       $19.60\text{ L}$ `S1b_exchanged_mAb` ($0.9800\ [\text{L/L}]$) + $0.40\text{ L}$
       of $15.0\text{ mmol/L}$ dehydroascorbic acid (DHAA) stock ($6.00\text{ mmol}$
       DHAA, $4.5\text{ eq}$, $0.0200\ [\text{L/L}]$) $\to 20.00\text{ L}$
       re-oxidized `S1_reduced_mAb` ($1.0000\ [\text{L/L}]$, $0.200\text{ kg}$
       activated `ADC1` with 2 free engineered hinge thiols).
- **Provenance of scheduling durations and hold-time limits**: Task durations
  $\tau_{\text{task}}$ $[\text{s}]$, hold-time limits $\tau_{\text{hold,max}}$
  $[\text{s}]$, diavolume counts, and step recoveries are **process-scheduling
  design assumptions (`origin="synthetic"`)** informed by the cited bench/pilot
  operating windows; they are parameterized for WP4 sensitivity studies (`#366`,
  `#367`) and flagged for domain review with `@dcasasor`.

| Task ID & Name | Input States (Liquid Volume Fraction $[\text{L/L}]$) | Output States (Liquid Volume Fraction $[\text{L/L}]$) | Equipment Resource Class & PharmaPy Unit | Liquid Volume $[\text{L}]$, Solution Mass $[\text{kg}]$ ($[\text{kg/kg}]$), Reagent Stock & Protein Solute Mass Balance | Nominal Duration $\tau_{\text{task}}$ $[\text{s}]$ (Range $[\text{s}]$) | Transfer / Storage Policy & Design-Assumption Rationale |
| --- | --- | --- | --- | --- | --- | --- |
| **`T1_reduction`** (Baseline Tier 2 stoichiometric TCEP reduction; see text above for Tier 1 `T1a`–`T1c` expansion) | `S0_native_mAb` ($0.9800$), `S_TCEP_reductant` ($0.0200$) | `S1_reduced_mAb` ($1.0000$) | `R_reduction_vessel` (`BatchReactor` or `SemibatchReactor`, jacketed at $T = 293.15\text{ K}$) | **In**: $19.60\text{ L}$ native mAb in aqueous buffer ($19.600\text{ kg}$, $0.9800\ [\text{kg/kg}]$; $0.200\text{ kg}$ mAb at $10.204\text{ g/L} = 6.803 \times 10^{-5}\text{ mol/L}$, $1.333\text{ mmol}$ mAb) + $0.40\text{ L}$ of $20.0\text{ mmol/L}$ aqueous TCEP stock ($0.400\text{ kg}$, $0.0200\ [\text{kg/kg}]$; $8.00\text{ mmol}$ TCEP, $6.0\text{ eq}$) $= 20.00\text{ L}$ ($20.000\text{ kg}$ solution) $\to$ **Out**: $20.00\text{ L}$ `S1_reduced_mAb` ($20.000\text{ kg}$, $1.0000\ [\text{kg/kg}]$; $0.200\text{ kg}$ reduced mAb at $10.00\text{ g/L} = 6.667 \times 10^{-5}\text{ mol/L}$, $100\%$ protein recovery) | $5400\text{ s}$ ($3600\text{–}7200\text{ s}$, $1.0\text{–}2.0\text{ h}$; `origin="synthetic"` scheduling assumption anchored to the $90\text{ min}$ TCEP reduction window in Weggen et al., 2025, Sec. 2.2) | **Finite-wait (`FW`)**: maximum hold $\tau_{\text{hold,S1,max}} = 1800\text{ s}$ ($0.5\text{ h}$, `origin="synthetic"` scheduling design assumption under $1\text{ mmol/L}$ EDTA and inert $\text{N}_2$ headspace to limit uncapped thiol re-oxidation prior to `T2`; allowable thiol-loss specification to be confirmed with `@dcasasor`) |
| **`T2_conjugation`** (Semi-batch maleimide payload addition + NAC quench) | `S1_reduced_mAb` ($0.8889$), `S_payload_DMSO` ($0.1000$), `S_NAC_quench` ($0.0111$) | `S2_crude_ADC` ($1.0000$) | `R_conjugation_vessel` (`SemibatchReactor` with metering pump, $T = 295.15\text{ K}$, $10.00\%\text{ v/v}$ DMSO in final $22.50\text{ L}$ batch, $10.11\%\text{ v/v}$ pre-quench) | **In**: $20.00\text{ L}$ `S1_reduced_mAb` ($20.000\text{ kg}$, $0.8801\ [\text{kg/kg}]$; $0.200\text{ kg}$ mAb) + $2.25\text{ L}$ neat DMSO payload stock at $2.963\text{ mmol/L}$ payload ($\rho_{\text{DMSO}} = 1.1004\text{ kg/L} \implies 2.476\text{ kg}$, $0.1090\ [\text{kg/kg}]$; $6.667\text{ mmol}$ payload, $5.0\text{ eq}$, giving $2.25 / 22.50 = 10.00\%\text{ v/v}$ DMSO in final batch) + $0.25\text{ L}$ aqueous $40.0\text{ mmol/L}$ NAC quench ($0.250\text{ kg}$, $0.0110\ [\text{kg/kg}]$; $10.0\text{ mmol}$ NAC) $= 22.50\text{ L}$ ($22.726\text{ kg}$ solution at $\rho_{\text{mix}} = 1.010\text{ kg/L}$) $\to$ **Out**: $22.50\text{ L}$ `S2_crude_ADC` ($22.726\text{ kg}$, $1.0000\ [\text{kg/kg}]$; $0.200\text{ kg}$ ADC protein solute, $100\%$ protein recovery) | $2700\text{ s}$ ($1800\text{–}3600\text{ s}$, $0.5\text{–}1.0\text{ h}$; `origin="synthetic"` scheduling assumption covering $600\text{–}1800\text{ s}$ syringe-pump feed + $600\text{ s}$ completion hold + $300\text{ s}$ NAC quench, Weggen et al., 2023, 2024) | **Finite-wait (`FW`)**: maximum hold $\tau_{\text{hold,S2,max}} = 3600\text{ s}$ ($1.0\text{ h}$, `origin="synthetic"` scheduling design assumption in $10.0\%\text{ v/v}$ DMSO to limit organic-cosolvent-induced protein aggregation prior to diafiltration `T3`, Fernandez-Cerezo et al., 2023) |
| **`T3_desalting_ufdf`** (Ultrafiltration / diafiltration small-molecule clearance & buffer exchange) | `S2_crude_ADC` ($0.1385$), `S_diafiltration_buffer` ($0.8615$) | `S3_desalted_ADC` ($0.1231$), `S_ufdf_permeate_waste` ($0.8769$) | `U_ufdf_skid` ($30\text{ kDa}$ MWCO tangential-flow filtration skid; Fernandez-Cerezo et al., 2023, `10.1021/acs.oprd.3c00137`) | **In**: $22.50\text{ L}$ `S2_crude_ADC` ($22.726\text{ kg}$, $0.1397\ [\text{kg/kg}]$) + $140.00\text{ L}$ aqueous diafiltration buffer ($140.000\text{ kg}$, $0.8603\ [\text{kg/kg}]$; $7.0$ diavolumes on $20.00\text{ L}$ retentate basis) $= 162.50\text{ L}$ ($162.726\text{ kg}$ total liquid) $\to$ **Out**: $20.00\text{ L}$ `S3_desalted_ADC` retentate ($20.000\text{ kg}$, $0.1229\ [\text{kg/kg}]$; $0.196\text{ kg}$ ADC protein solute at $98.0\%$ TFF step recovery, $>99\%$ free payload/DMSO/NAC cleared) + $142.50\text{ L}$ permeate waste ($142.726\text{ kg}$, $0.8771\ [\text{kg/kg}]$) | $4500\text{ s}$ ($3600\text{–}5400\text{ s}$, $1.0\text{–}1.5\text{ h}$; `origin="synthetic"` scheduling assumption for $7.0$ diavolumes at membrane flux $35\text{–}50\text{ L}\cdot\text{m}^{-2}\cdot\text{h}^{-1}$, Fernandez-Cerezo et al., 2023) | **Default `Unlimited` (`UIS`) within single-day campaign horizon** ($\le 24\text{ h}$ at $277.15\text{–}281.15\text{ K}$) in `DynamicCollector` hold tank; configurable parameter `t_hold_s3_max = 86400` $[\text{s}]$ for multi-day finite-wait (`FW`) scheduling sensitivity studies (`#366`, `#367`). Post-desalting dual-wavelength UV-Vis ($248/280\text{ nm}$) release check applies to `S3_desalted_ADC` |
| **`T4_purification`** (Polishing chromatography for DAR distribution & aggregate removal) | `S3_desalted_ADC` ($0.2500$), `S_chrom_mobile_phase` ($0.7500$) | `S4_purified_ADC_DS` ($0.2250$), `S_chrom_strip_waste` ($0.7750$) | `U_chrom_column` (Preparative HIC / AEX column skid) | **In**: $20.00\text{ L}$ `S3_desalted_ADC` ($20.000\text{ kg}$, $0.2500\ [\text{kg/kg}]$; $0.196\text{ kg}$ ADC protein solute) + $60.00\text{ L}$ equilibration/elution/strip mobile phase ($60.000\text{ kg}$, $0.7500\ [\text{kg/kg}]$; $3.0\times$ load volume) $= 80.00\text{ L}$ ($80.000\text{ kg}$ total liquid) $\to$ **Out**: $18.00\text{ L}$ `S4_purified_ADC_DS` eluate pool ($18.000\text{ kg}$, $0.2250\ [\text{kg/kg}]$; $0.180\text{ kg}$ purified ADC solute at $91.84\%$ polishing yield and $90.0\%$ overall four-step protein mass yield, $c_{\text{ADC}} = 10.0\text{ g/L}$) + $62.00\text{ L}$ flowthrough/strip waste ($62.000\text{ kg}$, $0.7750\ [\text{kg/kg}]$) | $3600\text{ s}$ ($2700\text{–}5400\text{ s}$, $0.75\text{–}1.5\text{ h}$; `origin="synthetic"` scheduling assumption for single-cycle preparative HIC/AEX load-wash-elute-regenerate sequence) | Final Drug Substance (`S4_purified_ADC_DS`, `Unlimited` storage in formulation hold tank) |

---

## 9. Primary literature references

1. **Andris, S., Seidel, J., & Hubbuch, J. (2019)**. Kinetic reaction modeling
   for antibody-drug conjugate process development. *Journal of Biotechnology*,
   306, 71–80. `doi:10.1016/j.jbiotec.2019.09.013`.
2. **Chen, Y. (2013)**. Drug-to-antibody ratio (DAR) by UV/Vis spectroscopy. In
   L. Ducry (Ed.), *Antibody-Drug Conjugates* (*Methods in Molecular Biology*,
   Vol. 1045, pp. 267–273). Humana Press. `doi:10.1007/978-1-62703-541-5_16`.
3. **Fernandez-Cerezo, L., Holstein, M., Thirumalairajan, S., Romanini, D.,
   Kos, P., & Raglione, T. (2023)**. Strategies for UF/DF-based impurity removal
   in the post-conjugation purification of antibody-drug conjugates. *Organic
   Process Research & Development*, 27(7), 1258–1268.
   `doi:10.1021/acs.oprd.3c00137`.
4. **Hamblett, K. J., Senter, P. D., Chace, D. F., Sun, M. M. C., Lenox, J.,
   Cerveny, C. G., Kissler, K. M., Bernhardt, S. X., Kopcha, A. K., Zabinski,
   R. F., Meyer, D. L., & Francisco, J. A. (2004)**. Effects of drug loading on
   the antitumor activity of a monoclonal antibody drug conjugate. *Clinical
   Cancer Research*, 10(20), 7063–7070. `doi:10.1158/1078-0432.CCR-04-0789`.
5. **Kantner, T., Alkhawaja, B., & Watts, A. G. (2017)**. In situ quenching of
   trialkylphosphine reducing agents using water-soluble PEG-azides improves
   maleimide conjugation to proteins. *ACS Omega*, 2(9), 5785–5791.
   `doi:10.1021/acsomega.7b01094`.
6. **Liu, H., Chumsae, C., Gaza-Bulseco, G., Hurkmans, K., & Radziejewski,
   C. H. (2010)**. Ranking the susceptibility of disulfide bonds in human IgG1
   antibodies by reduction, differential alkylation, and LC-MS analysis.
   *Analytical Chemistry*, 82(12), 5219–5226. `doi:10.1021/ac100575n`.
7. **Mao, L., Russell, A. J., & Carmali, S. (2022)**. Moving protein PEGylation
   from an art to a data science. *Bioconjugate Chemistry*, 33(9), 1643–1653.
   `doi:10.1021/acs.bioconjchem.2c00262`.
8. **Matsuda, Y., Tawfiq, Z., Leung, M., & Mendelsohn, B. A. (2020)**. Insight
   into temperature dependency and design of experiments towards process
   development for cysteine-based antibody-drug conjugates. *ChemistrySelect*,
   5(28), 8435–8439. `doi:10.1002/slct.202001822`.
9. **Nakahara, Y., Mendelsohn, B. A., & Matsuda, Y. (2022)**. Antibody–drug
   conjugate synthesis using continuous flow microreactor technology. *Organic
   Process Research & Development*, 26(9), 2766–2770.
   `doi:10.1021/acs.oprd.2c00217`.
10. **Nayak, S., & Richter, S. M. (2023)**. Kinetic studies of the partial
    reduction and conjugation reactions in an antibody-drug conjugate (ADC)
    synthesis. *Organic Process Research & Development*, 27(11), 2091–2099.
    `doi:10.1021/acs.oprd.3c00264`.
11. **Pfister, D., Bourgeaux, E., & Morbidelli, M. (2015)**. Kinetic modeling of
    protein PEGylation. *Chemical Engineering Science*, 137, 816–827.
    `doi:10.1016/j.ces.2015.07.031`.
12. **Pfister, D., Ulmer, N., Klaue, A., Ingold, O., & Morbidelli, M. (2016)**.
    Modeling the kinetics of protein conjugation reactions. *Chemie Ingenieur
    Technik*, 88(11), 1598–1608. `doi:10.1002/cite.201600046`.
13. **Pople, D., Wang, Z., Kozyryev, A., Patel, B., Appiah-Amponsah, E., &
    Feng, H. (2026)**. Raman spectroscopy for characterizing monoclonal antibody
    reduction: A process analytical technology approach for antibody–drug
    conjugation process development. *Pharmaceutical Research*.
    `doi:10.1007/s11095-026-04154-9`.
14. **Schiemer, R., Weggen, J. T., Schmitt, K. M., & Hubbuch, J. (2023)**. An
    adaptive soft-sensor for advanced real-time monitoring of an antibody-drug
    conjugation reaction. *Biotechnology and Bioengineering*, 120(7),
    1914–1928. `doi:10.1002/bit.28428`.
15. **Song, Y., Cai, H., Tan, Z., Mussa, N., & Li, Z. J. (2022)**. Mechanistic
    insights into inter-chain disulfide bond reduction of IgG1 and IgG4
    antibodies. *Applied Microbiology and Biotechnology*, 106(3), 1057–1066.
    `doi:10.1007/s00253-022-11778-5`.
16. **Sun, M. M. C., Beam, K. S., Cerveny, C. G., Hamblett, K. J., Blackmore,
    R. S., Torgov, M. Y., Handley, F. G. M., Ihle, N. C., Senter, P. D., &
    Alley, S. C. (2005)**. Reduction–alkylation strategies for the modification
    of specific monoclonal antibody disulfides. *Bioconjugate Chemistry*, 16(5),
    1282–1290. `doi:10.1021/bc050201y`.
17. **Vicum, L., Ottiger, S., Mazzotti, M., Makowski, Ł., & Bałdyga, J.
    (2004)**. Multi-scale modeling of a reactive mixing process in a semibatch
    stirred tank. *Chemical Engineering Science*, 59(8–9), 1767–1781.
    `doi:10.1016/j.ces.2004.01.032`.
18. **Wakankar, A., Chen, Y., Gokarn, Y., & Jacobson, F. S. (2011)**. Analytical
    methods for physicochemical characterization of antibody drug conjugates.
    *mAbs*, 3(2), 161–172. `doi:10.4161/mabs.3.2.14960` (`PMC3092617`).
19. **Weggen, J. T., Seidel, J., Bean, R., Wendeler, M., & Hubbuch, J. (2023)**.
    Kinetic studies and CFD-based reaction modeling for insights into the
    scalability of ADC conjugation reactions. *Frontiers in Bioengineering and
    Biotechnology*, 11, 1123842. `doi:10.3389/fbioe.2023.1123842`
    (`PMC10111256`).
20. **Weggen, J. T., Bean, R., Hui, K., Wendeler, M., & Hubbuch, J. (2024)**.
    Kinetic models towards an enhanced understanding of diverse ADC conjugation
    reactions. *Frontiers in Bioengineering and Biotechnology*, 12, 1403644.
    `doi:10.3389/fbioe.2024.1403644` (`PMC11274341`).
21. **Weggen, J. T., González, P., Hui, K., Bean, R., Wendeler, M., & Hubbuch,
    J. (2025)**. Kinetic modeling of the antibody disulfide bond reduction
    reaction with integrated prediction of the drug load profile for
    cysteine-conjugated ADCs. *Biotechnology and Bioengineering*, 122(2),
    342–356. `doi:10.1002/bit.28899` (`PMC11808428`).
