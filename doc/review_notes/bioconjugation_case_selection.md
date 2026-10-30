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
| Bioconjugation & Reaction-Modeling Domain Reviewer | Daniel Casas-Orozco (`@dcasasor` / `@dcasasor-purdue`) | Eli Lilly and Company; industrial bioconjugation chemistry, reaction-network specification ([#337](https://github.com/PharmaPy-org/PharmaPy/issues/337), [#339](https://github.com/PharmaPy-org/PharmaPy/issues/339)), and PAT observability review |
| WP1 / Project Co-PIs | Zoltan K. Nagy, Gintaras V. Reklaitis, Can Li (`@CanLi1`) | Purdue University (CryPTSys / SECQUOIA);PAT calibration (`#352`) and WP4 STN/RTN scheduling export (`#366`, `#367`) |

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
     $\text{mAb}_1$, $\text{mAb}_2$, $D$) and identified two consecutive
     bimolecular steps ($k_1 = 0.797\text{ L}\cdot\text{mmol}^{-1}\cdot\text{s}^{-1}$,
     $k_2 = 1.476\text{ L}\cdot\text{mmol}^{-1}\cdot\text{s}^{-1}$) plus a
     first-order payload sink ($k_3 = 1.55 \times 10^{-3}\text{ s}^{-1}$) with
     external validation $R^2 = 0.978$. Weggen et al. (2024) reformulated the
     same chemistry on a reduced heavy-chain analytical basis ($H_0, H_1, H_2$
     measured by reducing RP-UHPLC after $N$-acetylcysteine quenching) across
     both batch and syringe-pump fed-batch experiments. They demonstrated that
     when $H_2$ (two payloads on a single heavy chain, an impurity from residual
     unreoxidized interchain cysteines) remains near the limit of
     quantification, the second heavy-chain rate constant $k_2$ has a wide
     $95\%$ confidence interval ($k_2 = 1.712 \pm 6.585\text{ L}\cdot\text{mmol}^{-1}\cdot\text{s}^{-1}$
     in the 2k model), so a single shared rate constant
     ($k_1 = k_2 = 0.251 \pm 0.023\text{ L}\cdot\text{mmol}^{-1}\cdot\text{s}^{-1}$,
     $k_{\text{drug}} = 0.001\text{ s}^{-1}$, `Simple 1k` model) is selected by
     leave-one-experiment-out cross-validation
     ($\text{RMSE}_{\text{CV}} = 1.863\ \mu\text{mol/L}$,
     $\text{RMSE}_{\text{P}} = 3.188\ \mu\text{mol/L}$).

2. **Partial interchain-disulfide TCEP reduction followed by cysteine
   thiol–maleimide conjugation (target $\text{DAR} = 4\text{–}8$)**:
   - *Primary sources*: Sun et al. (2005), *Bioconjugate Chem.* 16(5),
     1282–1290 (`doi:10.1021/bc050201y`); Liu et al. (2010), *Anal. Chem.*
     82(12), 5219–5226 (`doi:10.1021/ac100575n`); Matsuda et al. (2020),
     *ChemistrySelect* 5(28), 8435–8439 (`doi:10.1002/slct.202001822`); Song et
     al. (2022), *Appl. Microbiol. Biotechnol.* 106(3), 1057–1066
     (`doi:10.1007/s00253-022-11778-5`); Nayak and Richter (2023), *Org.
     Process Res. Dev.* 27(11), 2091–2099 (`doi:10.1021/acs.oprd.3c00264`);
     Weggen et al. (2024), *Front. Bioeng. Biotechnol.* 12, 1403644
     (`doi:10.3389/fbioe.2024.1403644`); Weggen, González, Hui, Bean, Wendeler,
     and Hubbuch (2025), *Biotechnol. Bioeng.* 122(2), 342–356
     (`doi:10.1002/bit.28899`, `PMC11808428`).
   - *Chemistry & two-step mechanism*: Mild reduction of the four interchain
     disulfide bonds of human IgG1 (two hinge heavy–heavy $\text{HH}$ bonds and
     two heavy–light $\text{HL}$ bonds) by TCEP generates even-numbered free
     cysteine thiols ($2, 4, 6, 8$ thiols per antibody, corresponding to bridged
     fragments $\text{Intact}$, $\text{HHL}$, $\text{HH}$, $\text{HL}$,
     $\text{H}$, and $\text{L}$ in denaturing CE-SDS), followed by rapid
     Michael addition of maleimide linker-payload to each liberated thiol.
   - *Kinetic findings*:
     - **Step 1 (Disulfide reduction by TCEP)**: Nayak and Richter (2023) and
       Weggen et al. (2025) established that TCEP disulfide cleavage exhibits
       strong structural cooperativity: in Weggen et al. (2025, Table 2,
       `8k-model` calibrated across $T \in \{4, 12, 20, 37\}\ ^\circ\text{C}$,
       $c_{\text{mAb},0} \in [3.75, 20]\text{ g/L}$, and TCEP molar excess
       $[2, 10]\times$ at reference temperature $T_{\text{ref}} = 293.15\text{ K}$),
       the inter-HL disulfide reduction rates decrease sequentially as more
       bonds are cleaved ($k_1 = 0.698$, $k_2 = 0.254$, $k_3 = 0.146$,
       $k_6 = 0.0681\text{ L}\cdot\text{mol}^{-1}\cdot\text{s}^{-1}$ at
       $20\ ^\circ\text{C}$, converted from
       $6.98 \times 10^{-4}\text{ to }6.81 \times 10^{-5}\text{ L}\cdot\mu\text{mol}^{-1}\cdot\text{s}^{-1}$),
       whereas the second hinge inter-HH disulfide cleavage
       ($k_5 = 2.66\text{ L}\cdot\text{mol}^{-1}\cdot\text{s}^{-1}$) is $18\times$
       faster than the first hinge inter-HH cleavage
       ($k_4 = 0.148\text{ L}\cdot\text{mol}^{-1}\cdot\text{s}^{-1}$) due to
       conformational opening of the Fc hinge after the first HH bond breaks.
       Re-oxidation of reduced HL thiols ($k_{\text{reox}} = 0.0365\text{ L}\cdot\text{mol}^{-1}\cdot\text{s}^{-1}$)
       and reactive intermediate states ($\text{IntactR}, \text{HHLR},
       \text{HHR}$ with first-order relaxation $k_7 = 5.45 \times 10^{-3}\text{ s}^{-1}$)
       explain the partial loss of free thiols over long hold times.
     - **Step 2 (Cysteine thiol–maleimide conjugation)**: For $\text{DAR } 8$
       interchain conjugation, Weggen et al. (2024, Table 2) showed that a
       four-parameter chain-resolved kinetic model (`Detailed 4k`: light-chain
       conjugation $L_0^1 + D \xrightarrow{k_1} L_1^1$ with
       $k_1 = 1.220 \pm 0.003\text{ L}\cdot\text{mmol}^{-1}\cdot\text{s}^{-1}$,
       and heavy-chain steps
       $H_0^c \xrightarrow{k_2} H_1^c \xrightarrow{k_3} H_2^c \xrightarrow{k_{4/5}} H_3^c \xrightarrow{k_{4/5}} H_4^c$
       with $k_2 = 1.853 \pm 0.006$, $k_3 = 5.117 \pm 1.015$, and
       $k_{4/5} = 2.312 \pm 0.356\text{ L}\cdot\text{mmol}^{-1}\cdot\text{s}^{-1}$)
       reduces prediction error by more than $2.5\times$ compared with a
       single-rate-constant model ($\text{RMSE}_{\text{P}} = 0.964$ vs.
       $2.548\ \mu\text{mol/L}$ for ADC2 + NPM) because attachment of the first
       hydrophobic payload to the hinge accelerates conjugation of the
       geminal cysteine on the same disulfide pair ($k_3 > k_2$).

3. **Stochastic lysine amide coupling / NHS-ester acylation and PEGylation**:
   - *Primary sources*: Pfister, Bourgeaux, and Morbidelli (2015), *Chem. Eng.
     Sci.* 137, 816–827 (`doi:10.1016/j.ces.2015.07.031`); Pfister et al.
     (2016), *Chem. Ing. Tech.* 88(11), 1598–1608
     (`doi:10.1002/cite.201600046`); Mao, Russell, and Carmali (2022),
     *Bioconjugate Chem.* 33(9), 1643–1653
     (`doi:10.1021/acs.bioconjchem.2c00262`).
   - *Chemistry & modeling limits*: Activated $N$-hydroxysuccinimide (NHS)
     esters react with surface-accessible $\varepsilon$-amines across $80+$
     lysine residues in parallel with rapid aqueous NHS-ester hydrolysis
     ($k_{\text{hyd}} \sim 10^{-3}\text{–}10^{-2}\text{ s}^{-1}$ at
     $\text{pH } 7.5\text{–}8.5$). Because $2^{N_{\text{sites}}}$ positional
     isomers are lumping-dependent and site-specific rate constants are
     unidentifiable without peptide-mapping time courses, stochastic lysine
     conjugation is less suited than cysteine conjugation for benchmarking
     mechanistic parameter estimation and structural identifiability.

4. **Heterobifunctional linker-payload and AOC/ARC coupling with competing
   isomerization or hydrolysis**:
   - *Primary sources*: Kantner, Alkhawaja, and Watts (2017), *ACS Omega* 2(9),
     5785–5791 (`doi:10.1021/acsomega.7b01094`); Casas-Orozco (2026), *PharmaPy
     Software Development v2* (Eli Lilly and Company, Aug. 21, 2026, slide 4,
     tracked in [#337](https://github.com/PharmaPy-org/PharmaPy/issues/337) and
     [#339](https://github.com/PharmaPy-org/PharmaPy/issues/339)); Purdue–Lilly
     360 Proposal (2026, WP1.5, p. 3).
   - *Chemistry*: Either (a) a biomacromolecule $A$ is coupled with a
     heterobifunctional linker $B$ and oligonucleotide/drug payload $C$ to form
     conjugate $D$ via sequential bimolecular steps ($A + B \to I_{\text{AB}}$,
     $I_{\text{AB}} + C \to D$) alongside linker hydrolysis, or (b) reactive
     species $A$ and $B$ form desired conjugate $C$ (`coupling: A + B --> C`)
     while $C$ (or an intermediate) undergoes a unimolecular ring-opening,
     succinimide hydrolysis, or positional isomerization side reaction
     (`iso: C --> I`).

| Modality | Representative network | Typical bimolecular rate $k$ $[\text{L}\cdot\text{mol}^{-1}\cdot\text{s}^{-1}]$ | Competing side reaction | Characteristic reaction time $\tau_R$ $[\text{s}]$ at $0.1\text{ mmol/L}$ | Public kinetic parameter & RMSE tables |
| --- | --- | --- | --- | --- | --- |
| Site-specific Cys–maleimide ($\text{DAR } 2$) | $\text{mAb}_0 + D \xrightarrow{k_1} \text{mAb}_1$, $\text{mAb}_1 + D \xrightarrow{k_2} \text{mAb}_2$, $D \xrightarrow{k_3} D_{\text{depl}}$ | $2.51 \times 10^2\text{–}4.84 \times 10^3$ | First-order maleimide depletion/hydrolysis ($k_3 = 1.0 \times 10^{-3}\text{–}4.1 \times 10^{-2}\text{ s}^{-1}$) | $2\text{–}40$ | Yes: Andris et al. (2019); Weggen et al. (2023, 2024) |
| Interchain TCEP reduction + Cys–maleimide ($\text{DAR } 4\text{–}8$) | Step 1: 8-rate TCEP disulfide cleavage + HL reoxidation; Step 2: $L_0 \to L_1$ and $H_0 \to H_1 \to H_2 \to H_3 \to H_4$ + $D \to D_{\text{depl}}$ | Reduction: $6.81 \times 10^{-2}\text{–}2.66$; Conjugation: $1.22 \times 10^3\text{–}5.12 \times 10^3$ | Inter-HL disulfide re-oxidation ($k_{\text{reox}} = 3.65 \times 10^{-2}\text{ L}\cdot\text{mol}^{-1}\cdot\text{s}^{-1}$); payload depletion | Reduction: $600\text{–}3600$; Conjugation: $1\text{–}10$ | Yes: Weggen et al. (2024, 2025); Nayak & Richter (2023) |
| Stochastic Lys–NHS amide coupling | $\text{mAb}_{k} + L \xrightarrow{k_k} \text{mAb}_{k+1}$, $L \xrightarrow{k_{\text{hyd}}} L_{\text{hyd}}$ ($k = 0,\dots,N$) | $10^1\text{–}10^2$ (lumped) | Fast NHS-ester hydrolysis ($10^{-3}\text{–}10^{-2}\text{ s}^{-1}$) | $50\text{–}500$ | Partial (PEGylation in Pfister et al., 2015) |
| Lilly coupling + isomerization prototype (`#339`) | `coupling`: $A + B \xrightarrow{k_c} C$; `iso`: $C \xrightarrow{k_{\text{iso}}} I$ | $10^2\text{–}10^3$ | Unimolecular conjugate isomerization/ring-opening ($10^{-4}\text{–}10^{-2}\text{ s}^{-1}$) | $10\text{–}100$ | Synthetic / internal Lilly specification |

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
     competing payload depletion** in two directly linked representations:
     - **Model 1A (Intact-antibody consecutive `2k + sink` model, Andris et al.,
       2019; Weggen et al., 2023, Eqs. 3–5)**: tracks intact unconjugated
       antibody ($\text{mAb}_0$), mono-conjugate ($\text{mAb}_1$),
       bi-conjugate ($\text{mAb}_2$), free payload ($D$), and inactivated
       payload ($D_{\text{depl}}$) across batch (`BatchReactor`), fed-batch
       (`SemibatchReactor`), and continuous (`Mixer` + `PlugFlowReactor`)
       operating modes.
     - **Model 1B (Reduced heavy-chain `Simple 1k + sink` and `Detailed 2k + sink`
       models, Weggen et al., 2024, Eqs. 1–7 & Table 2)**: tracks heavy-chain
       species $H_0^0, H_0^1, H_0^2, H_1^1, H_1^2, H_2^2$, free payload $D$,
       and depleted payload $D_{\text{depl}}$ across batch ($1\text{ mL}$) and
       syringe-pump fed-batch ($4\text{ mL}$ initial volume with $10, 20, 30\text{ min}$
       payload addition) experiments, providing a published benchmark for
       **practical identifiability analysis** ($k_1$ vs. $k_2$) and
       **model-based experiment design (MBDoE)** via feed-rate manipulation.
2. **Upstream reduction & multi-step extension (`Tier 2`, multi-temperature
   Arrhenius reduction and chain-resolved $\text{DAR } 8$ conjugation)**:
   - **Stage 1 disulfide reduction (`Weggen et al., 2025`, `10.1002/bit.28899`)**:
     TCEP reduction of IgG1 interchain disulfide bonds across
     $T \in [277.15, 310.15]\text{ K}$ ($4\text{–}37\ ^\circ\text{C}$,
     $T_{\text{ref}} = 293.15\text{ K}$) generating the initial reactive-thiol
     distribution that feeds Stage 2 conjugation.
   - **Stage 2 chain-resolved $\text{DAR } 8$ conjugation (`Detailed 4k`,
     `Weggen et al., 2024`, Table 2)**: light-chain ($L_0^1 \to L_1^1$) and
     sequential heavy-chain ($H_0^c \to H_1^c \to H_2^c \to H_3^c \to H_4^c$)
     conjugation with competing payload depletion.

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

| Symbol | PharmaPy species ID | Representative molar mass $M_i$ $[\text{g/mol}]$ | Physical meaning and analytical basis | Typical initial concentration $c_{i,0}$ $[\text{mol/L}]$ |
| --- | --- | --- | --- | --- |
| $\text{mAb}_0$ | `mAb` | $1.500 \times 10^5$ | Intact unconjugated antibody (2 reactive Cys thiols, $\text{DAR}=0$) | $1.00 \times 10^{-5}\text{–}6.67 \times 10^{-5}$ ($1.5\text{–}10\text{ g/L}$) |
| $\text{mAb}_1$ | `mAb_1Drug` | $1.507 \times 10^5$ | Intact mono-conjugated antibody (1 payload bound, $\text{DAR}=1$) | $0.0$ |
| $\text{mAb}_2$ | `mAb_2Drug` | $1.514 \times 10^5$ | Intact bi-conjugated antibody (2 payloads bound, $\text{DAR}=2$) | $0.0$ |
| $H_0^0$ | `H0_0` | $5.000 \times 10^4$ | Heavy chain with 0 payloads bound and 0 reactive cysteines (pre-inactivated) | $0.088 \, c_{H,0}$ ($0.0\text{–}1.2 \times 10^{-5}$) |
| $H_0^1$ | `H0_1` | $5.000 \times 10^4$ | Heavy chain with 0 payloads bound and 1 reactive engineered cysteine | $0.880 \, c_{H,0}$ ($1.5 \times 10^{-5}\text{–}1.2 \times 10^{-4}$) |
| $H_0^2$ | `H0_2` | $5.000 \times 10^4$ | Heavy chain with 0 payloads bound and 2 reactive cysteines (over-reduced interchain) | $0.032 \, c_{H,0}$ ($0.0\text{–}4.3 \times 10^{-6}$) |
| $H_1^1$ | `H1_1` | $5.070 \times 10^4$ | Heavy chain with 1 payload bound and 0 remaining reactive cysteines | $0.0$ |
| $H_1^2$ | `H1_2` | $5.070 \times 10^4$ | Heavy chain with 1 payload bound and 1 remaining reactive cysteine | $0.0$ |
| $H_2^2$ | `H2_2` | $5.140 \times 10^4$ | Heavy chain with 2 payloads bound and 0 remaining reactive cysteines | $0.0$ |
| $D$ | `Drug` | $7.000 \times 10^2$ | Reactive maleimide-functionalized payload (e.g., Drug1, vc-MMAE, or NPM) | Batch: $3.0 \times 10^{-5}\text{–}3.3 \times 10^{-4}$; Fed-batch feed: $1.0 \times 10^{-3}\text{–}1.0 \times 10^{-2}$ |
| $D_{\text{depl}}$ | `Drug_depl` | $7.180 \times 10^2$ | Hydrolyzed/inactivated maleimide payload (non-reactive sink) | $0.0$ |
| $\text{Solv}$ | `buffer` | $2.050 \times 10^1$ | Aqueous phosphate/EDTA buffer with $10\%\text{ v/v}$ DMSO cosolvent | Completed by `LiquidPhase` volume closure ($\approx 49.2\text{ mol/L}$) |

*Note on heavy-chain versus intact-mAb molar concentration basis*: Every intact
IgG1 molecule comprises two heavy chains and two light chains, so the total
heavy-chain molar concentration is exactly twice the intact antibody molar
concentration:

$$c_{H,\text{total}} = 2 \, c_{\text{mAb,total}} = 2 \, \frac{\rho_{\text{mAb}}}{M_{\text{mAb}}} \quad [\text{mol/L}]$$

For example, at a protein mass concentration of $\rho_{\text{mAb}} = 5.0\text{ g/L}$
and $M_{\text{mAb}} = 1.50 \times 10^5\text{ g/mol}$, the intact antibody molar
concentration is $c_{\text{mAb,total}} = 3.333 \times 10^{-5}\text{ mol/L}$
($33.33\ \mu\text{mol/L}$) and the heavy-chain molar concentration is
$c_{H,\text{total}} = 6.667 \times 10^{-5}\text{ mol/L}$ ($66.67\ \mu\text{mol/L}$).

### 3.2 Stoichiometry and elementary rate laws

#### Model 1A — Intact-antibody consecutive conjugation with competing payload depletion (Andris et al., 2019; Weggen et al., 2023, Eqs. 3–5)

In PharmaPy `RxnKinetics` syntax (`rxn_list`), the three elementary irreversible
reactions are:

$$\begin{aligned}
\text{R1 (first conjugation):} &\quad \text{mAb} + \text{Drug} \xrightarrow{k_1} \text{mAb\_1Drug}, & r_1 &= k_1 \, c_{\text{mAb}} \, c_{\text{Drug}} \quad [\text{mol}\cdot\text{L}^{-1}\cdot\text{s}^{-1}] \\
\text{R2 (second conjugation):} &\quad \text{mAb\_1Drug} + \text{Drug} \xrightarrow{k_2} \text{mAb\_2Drug}, & r_2 &= k_2 \, c_{\text{mAb\_1Drug}} \, c_{\text{Drug}} \quad [\text{mol}\cdot\text{L}^{-1}\cdot\text{s}^{-1}] \\
\text{R3 (payload depletion):} &\quad \text{Drug} \xrightarrow{k_3} \text{Drug\_depl}, & r_3 &= k_3 \, c_{\text{Drug}} \quad [\text{mol}\cdot\text{L}^{-1}\cdot\text{s}^{-1}]
\end{aligned}$$

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

Two exact linear invariants hold at all times in a closed batch reactor:

1. **Antibody core conservation**:
   $r_{\text{mAb}} + r_{\text{mAb\_1Drug}} + r_{\text{mAb\_2Drug}} = 0$, so
   $c_{\text{mAb}}(t) + c_{\text{mAb\_1Drug}}(t) + c_{\text{mAb\_2Drug}}(t) = c_{\text{mAb},0}$
   $[\text{mol/L}]$.
2. **Total drug-moiety conservation**:
   $r_{\text{Drug}} + r_{\text{Drug\_depl}} + r_{\text{mAb\_1Drug}} + 2\,r_{\text{mAb\_2Drug}} = 0$,
   so $c_{\text{Drug}}(t) + c_{\text{Drug\_depl}}(t) + c_{\text{mAb\_1Drug}}(t) + 2\,c_{\text{mAb\_2Drug}}(t) = c_{\text{Drug},0}$
   $[\text{mol/L}]$.

#### Model 1B — Reduced heavy-chain conjugation with initial cysteine heterogeneity and competing payload depletion (Weggen et al., 2024, Eqs. 1–7)

When analytical time courses are measured by reduced RP-UHPLC, heavy chains carry
$c \in \{0, 1, 2\}$ initial reactive thiols ($H_0^0, H_0^1, H_0^2$, with
$H_0^0$ inert) and react according to:

$$\begin{aligned}
\text{R1a:} &\quad H_0^1 + \text{Drug} \xrightarrow{k_1} H_1^1, & r_{1a} &= k_1 \, c_{H_0^1} \, c_{\text{Drug}} \quad [\text{mol}\cdot\text{L}^{-1}\cdot\text{s}^{-1}] \\
\text{R1b:} &\quad H_0^2 + \text{Drug} \xrightarrow{k_1} H_1^2, & r_{1b} &= k_1 \, c_{H_0^2} \, c_{\text{Drug}} \quad [\text{mol}\cdot\text{L}^{-1}\cdot\text{s}^{-1}] \\
\text{R2:} &\quad H_1^2 + \text{Drug} \xrightarrow{k_2} H_2^2, & r_2 &= k_2 \, c_{H_1^2} \, c_{\text{Drug}} \quad [\text{mol}\cdot\text{L}^{-1}\cdot\text{s}^{-1}] \\
\text{R3:} &\quad \text{Drug} \xrightarrow{k_{\text{drug}}} \text{Drug\_depl}, & r_3 &= k_{\text{drug}} \, c_{\text{Drug}} \quad [\text{mol}\cdot\text{L}^{-1}\cdot\text{s}^{-1}]
\end{aligned}$$

In a fed-batch reactor (`SemibatchReactor`) with initial liquid volume $V_0$
$[\text{L}]$, constant volumetric payload feed rate $F_{\text{in}}$
$[\text{L/s}]$ over $t \in [0, t_{\text{add}}]$, and feed payload concentration
$c_{\text{Drug,in}}$ $[\text{mol/L}]$, the liquid volume and species dilution
balances are (Weggen et al., 2024, Eqs. 1–7):

$$\frac{\mathrm{d}V}{\mathrm{d}t} = F(t) \quad [\text{L/s}], \qquad
\frac{\mathrm{d}c_i}{\mathrm{d}t} = r_{\text{species},i}(\mathbf{c}) + \frac{F(t)}{V(t)} \left(c_{i,\text{in}} - c_i(t)\right) \quad [\text{mol}\cdot\text{L}^{-1}\cdot\text{s}^{-1}]$$

where $F(t) = F_{\text{in}}$ for $0 \le t \le t_{\text{add}}$ and $F(t) = 0$ for
$t > t_{\text{add}}$, $c_{\text{Drug,in}} > 0$, and $c_{i,\text{in}} = 0$ for
all protein and depleted-payload species. When the initial heavy-chain population
is purely mono-thiol ($c_{H_0^2,0} = 0$, or when $H_0 = H_0^1$ and $H_1 = H_1^1$
are lumped in the `Simple 1k` model with $k_2 = k_1$), the core reaction network
reduces to two reactions ($H_0^1 + \text{Drug} \xrightarrow{k_1} H_1^1$ and
$\text{Drug} \xrightarrow{k_{\text{drug}}} \text{Drug\_depl}$).

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

First-order rate constants ($k_3, k_{\text{drug}}, k_7$) have units of
$[\text{s}^{-1}]$, which are invariant to the molar concentration scale.

### 3.4 Published kinetic parameter tables and provenance

| Model & System | Parameter | Literature value & $95\%$ CI | PharmaPy value (`[mol/L, s]` basis) | Conditions & Validated Range | Primary Source |
| --- | --- | --- | --- | --- | --- |
| **Model 1A**: Intact $\text{DAR } 2$ (`ADC1 + NPM`, `2k + sink`) | $k_1$ ($\text{mAb}_0 + D$) | $0.797\text{ L}\cdot\text{mmol}^{-1}\cdot\text{s}^{-1}$ | $797.0\text{ L}\cdot\text{mol}^{-1}\cdot\text{s}^{-1}$ | $22\ ^\circ\text{C}$ ($295.15\text{ K}$), $\text{pH } 7.0$, $20\text{ mM}$ $\text{NaH}_2\text{PO}_4$, $1\text{ mM}$ EDTA, $10\%\text{ v/v}$ DMSO; $c_{\text{mAb},0} \in [1.5, 10]\text{ g/L}$ ($10\text{–}66.7\ \mu\text{mol/L}$), $5\times$ payload excess | Andris et al. (2019), `10.1016/j.jbiotec.2019.09.013`; Weggen et al. (2023), `10.3389/fbioe.2023.1123842`, Eq. (3) |
| **Model 1A**: Intact $\text{DAR } 2$ (`ADC1 + NPM`, `2k + sink`) | $k_2$ ($\text{mAb}_1 + D$) | $1.476\text{ L}\cdot\text{mmol}^{-1}\cdot\text{s}^{-1}$ | $1476.0\text{ L}\cdot\text{mol}^{-1}\cdot\text{s}^{-1}$ | Same as above | Weggen et al. (2023), `10.3389/fbioe.2023.1123842`, Eq. (4) |
| **Model 1A**: Intact $\text{DAR } 2$ (`ADC1 + NPM`, `2k + sink`) | $k_3$ ($D \to D_{\text{depl}}$) | $1.55 \times 10^{-3}\text{ s}^{-1}$ | $1.55 \times 10^{-3}\text{ s}^{-1}$ | Same as above ($R^2 = 0.979$ for 0D model in $300\text{ mL}$ `GST-1`) | Weggen et al. (2023), `10.3389/fbioe.2023.1123842`, Eq. (5) |
| **Model 1B**: Heavy-chain $\text{DAR } 2$ `Simple 1k` (`ADC1 + Drug1`) | $k_1$ ($H_0^1, H_0^2, H_1^2 + D$) | $0.251 \pm 0.023\text{ L}\cdot\text{mmol}^{-1}\cdot\text{s}^{-1}$ ($[0.204, 0.298]$) | $251.0 \pm 23.0\text{ L}\cdot\text{mol}^{-1}\cdot\text{s}^{-1}$ | $22\ ^\circ\text{C}$ ($295.15\text{ K}$), $\text{pH } 7.0$, $10\%\text{ v/v}$ DMSO; batch ($1\text{ mL}$) & fed-batch ($4\text{ mL}$, $10\text{–}30\text{ min}$ feed), $\text{RMSE}_{\text{CV}} = 1.863\ \mu\text{mol/L}$, $\text{RMSE}_{\text{P}} = 3.188\ \mu\text{mol/L}$ | Weggen et al. (2024), `10.3389/fbioe.2024.1403644`, Table 2 |
| **Model 1B**: Heavy-chain $\text{DAR } 2$ `Simple 1k` (`ADC1 + Drug1`) | $k_{\text{drug,Drug1}}$ ($D \to D_{\text{depl}}$) | $1.00 \times 10^{-3}\text{ s}^{-1}$ | $1.00 \times 10^{-3}\text{ s}^{-1}$ | Same as above | Weggen et al. (2024), `10.3389/fbioe.2024.1403644`, Sec. 3.1 |
| **Model 1B**: Heavy-chain $\text{DAR } 2$ `Detailed 2k` (`ADC1 + Drug1`) | $k_1, k_2$ | $k_1 = 0.297 \pm 0.027$, $k_2 = 1.712 \pm 6.585\text{ L}\cdot\text{mmol}^{-1}\cdot\text{s}^{-1}$ | $k_1 = 297.0 \pm 27.0$, $k_2 = 1712.0 \pm 6585.0\text{ L}\cdot\text{mol}^{-1}\cdot\text{s}^{-1}$ | Demonstrates practical unidentifiability of $k_2$ when $c_{H_0^2,0}$ is small ($\approx 3.2\%$) | Weggen et al. (2024), `10.3389/fbioe.2024.1403644`, Table 2 |
| **Model 1B**: Heavy-chain $\text{DAR } 2$ `Simple 1k` (`ADC1 + NPM`) | $k_1, k_{\text{drug,NPM}}$ | $k_1 = 4.840\text{ L}\cdot\text{mmol}^{-1}\cdot\text{s}^{-1}$, $k_{\text{drug}} = 0.041\text{ s}^{-1}$ | $k_1 = 4840.0\text{ L}\cdot\text{mol}^{-1}\cdot\text{s}^{-1}$, $k_{\text{drug}} = 4.10 \times 10^{-2}\text{ s}^{-1}$ | $22\ ^\circ\text{C}$, $\text{pH } 7.0$, $10\%\text{ v/v}$ DMSO ($\text{RMSE}_{\text{CV}} = 3.599\ \mu\text{mol/L}$) | Weggen et al. (2024), `10.3389/fbioe.2024.1403644`, Sec. 3.1 & Table 2 |
| **Tier 2**: Chain-resolved $\text{DAR } 8$ `Detailed 4k` (`ADC2 + NPM`) | $k_1 (\text{LC}), k_2 (\text{HC}_1), k_3 (\text{HC}_2), k_{4/5} (\text{HC}_{3,4})$ | $1.220 \pm 0.003$, $1.853 \pm 0.006$, $5.117 \pm 1.015$, $2.312 \pm 0.356\text{ L}\cdot\text{mmol}^{-1}\cdot\text{s}^{-1}$ | $1220.0$, $1853.0$, $5117.0$, $2312.0\text{ L}\cdot\text{mol}^{-1}\cdot\text{s}^{-1}$ | $22\ ^\circ\text{C}$, $\text{pH } 7.0$, $10\%\text{ v/v}$ DMSO ($\text{RMSE}_{\text{CV}} = 0.620\ \mu\text{mol/L}$, $\text{RMSE}_{\text{P}} = 0.964\ \mu\text{mol/L}$) | Weggen et al. (2024), `10.3389/fbioe.2024.1403644`, Table 2 |
| **Tier 2**: IgG1 TCEP Disulfide Reduction (`8k-model`) | $k_1\text{–}k_6, k_{\text{reox}}$ at $T_{\text{ref}} = 293.15\text{ K}$; $k_7$ | $k_1..k_6 = [6.98, 2.54, 1.46, 1.48, 26.6, 0.681] \times 10^{-4}\text{ L}\cdot\mu\text{mol}^{-1}\cdot\text{s}^{-1}$; $k_{\text{reox}} = 3.65 \times 10^{-5}\text{ L}\cdot\mu\text{mol}^{-1}\cdot\text{s}^{-1}$; $k_7 = 5.45 \times 10^{-3}\text{ s}^{-1}$ | $k_1..k_6 = [0.698, 0.254, 0.146, 0.148, 2.66, 0.0681]\text{ L}\cdot\text{mol}^{-1}\cdot\text{s}^{-1}$; $k_{\text{reox}} = 0.0365\text{ L}\cdot\text{mol}^{-1}\cdot\text{s}^{-1}$; $k_7 = 5.45 \times 10^{-3}\text{ s}^{-1}$ | $T \in [277.15, 310.15]\text{ K}$ ($4\text{–}37\ ^\circ\text{C}$), $c_{\text{mAb},0} \in [3.75, 20]\text{ g/L}$ ($25\text{–}133\ \mu\text{mol/L}$), TCEP excess $2\text{–}10\times$ ($\text{RMSE}_{\text{P}} = 2.09\ \mu\text{mol/L}$) | Weggen et al. (2025), `10.1002/bit.28899`, Table 2 |

---

## 4. Public dataset availability, data-access gap, and fallback policy

1. **Industrial data-access gap**:
   - Eli Lilly and Company internal bioconjugation reaction trajectories and
     inline spectroscopic datasets are proprietary and cannot be committed to the
     public repository without formal data-release clearance.
   - In the open-access KIT/AstraZeneca publications (`Weggen et al., 2023`,
     `PMC10111256`; `Weggen et al., 2024`, `PMC11274341`; `Weggen et al., 2025`,
     `PMC11808428`, all distributed under CC BY 4.0), the complete ODE equations,
     initial concentrations, reactor volumes, syringe-pump dosing durations,
     calibrated rate constants, $95\%$ confidence intervals, parameter
     correlation matrices, and cross-validation/prediction RMSEs are published
     in tables, whereas the underlying quenched RP-UHPLC / CE-SDS time-series
     points are presented as figures rather than machine-readable CSV supplements
     (stated in each article's Data Availability Statement: *"The original
     contributions presented in the study are included in the article/Supplementary
     Material, further inquiries can be directed to the corresponding author"*).

2. **Public fallback and provenance-tagging contract**:
   - In accordance with `AGENTS.md` and `#351` (`CalibrationRecord`), repository
     verification fixtures and tutorials use **two distinct, explicitly tagged
     `DataSource` records**:
     - `origin='literature'`: records the peer-reviewed rate constants, initial
       reactant concentrations, reactor volumes, feed rates, and UV-Vis
       extinction coefficients from Weggen et al. (2023, 2024, 2025) and
       Hamblett et al. (2004), citing the exact DOI, equation/table numbers, and
       CC BY 4.0 license terms where applicable.
     - `origin='synthetic'`: records forward-simulated concentration and spectral
       time-course observations generated from those literature models (with
       documented, seeded Gaussian measurement noise when testing statistical
       inference), explicitly stating in `sharing` and `description` that the
       time series are synthetic verification data and **not** an empirical
       calibration dataset.
   - If digitized or author-shared raw CSV tables from KIT/AstraZeneca or a
     cleared Lilly benchmark dataset become available during M7–M10, they will be
     added under `#356` with their SHA-256 digests and explicit license metadata.

---

## 5. Mixing assumptions, operating modes, and Damköhler scaling analysis

Following Vicum et al. (2004) and Weggen et al. (2023, Sec. 2.2.8, Eq. 9), the
characteristic chemical reaction timescale $\tau_{R,i}$ $[\text{s}]$ of a
bimolecular conjugation step with rate constant $k_i$
$[\text{L}\cdot\text{mol}^{-1}\cdot\text{s}^{-1}]$ and local reactant
concentrations $\tilde{c}_1, \tilde{c}_2$ $[\text{mol/L}]$ is:

$$\tau_{R,i} = \frac{1}{k_i \left(\tilde{c}_1 + \tilde{c}_2\right)} \quad [\text{s}]$$

Comparing $\tau_{R,i}$ with the vessel's $95\%$ homogenization mixing time
$\tau_{\text{mix,95}}$ $[\text{s}]$ defines the mixing Damköhler number:

$$\text{Da}_i = \frac{\tau_{\text{mix,95}}}{\tau_{R,i}} = k_i \left(\tilde{c}_1 + \tilde{c}_2\right) \tau_{\text{mix,95}} \quad [\text{-}]$$

Using the first conjugation step of Model 1A
($k_1 = 797.0\text{ L}\cdot\text{mol}^{-1}\cdot\text{s}^{-1}$) at the standard
manufacturing condition of Weggen et al. (2023, Table 2:
$\rho_{\text{mAb},0} = 5.0\text{ g/L} \implies c_{\text{mAb},0} = 3.333 \times 10^{-5}\text{ mol/L}$,
$5\times$ molar payload excess $\implies c_{\text{Drug},0}^{\text{bulk}} = 1.667 \times 10^{-4}\text{ mol/L}$,
and concentrated feed stock $c_{\text{Drug,feed}} = 10.0\text{ mmol/L} = 1.00 \times 10^{-2}\text{ mol/L}$):

- **Bulk well-mixed reaction timescale**:
  $$\tau_{R,1}^{\text{bulk}} = \frac{1}{797.0 \times \left(3.333 \times 10^{-5} + 1.667 \times 10^{-4}\right)} = 6.274\text{ s}$$
- **Unmixed bolus feed-plume reaction timescale** ($\tilde{c}_{\text{Drug}} = c_{\text{Drug,feed}} = 1.00 \times 10^{-2}\text{ mol/L}$):
  $$\tau_{R,1}^{\text{plume}} = \frac{1}{797.0 \times \left(3.333 \times 10^{-5} + 1.00 \times 10^{-2}\right)} = 0.1251\text{ s}$$

| Vessel / Reactor Configuration | Liquid Volume $V$ $[\text{L}]$ | Mixing Time $\tau_{\text{mix,95}}$ $[\text{s}]$ | Bulk $\text{Da}_1^{\text{bulk}}$ $[-]$ | Plume $\text{Da}_1^{\text{plume}}$ $[-]$ (bolus) | Max $\|\Delta\text{DAR}(t)\|$ (0D vs. 3D CFD) | PharmaPy Unit-Operation Mapping & Assumption |
| --- | --- | --- | --- | --- | --- | --- |
| Internally stirred $1.5\text{ mL}$ tube / cuvette | $1.0 \times 10^{-3}\text{–}4.0 \times 10^{-3}$ | $\approx 0.5$ | $0.080$ | $4.00$ | $< 0.002$ | `BatchReactor` or `SemibatchReactor` (ideal 0D well-mixed assumption valid; $\text{Da}_{\text{bulk}} \ll 1$) |
| Lab-scale glass stirred tank (`GST-1`, $60\text{ rpm}$, $P/V = 3.81\text{ W/m}^3$) | $0.300$ | $9.4$ | $1.50$ | $75.2$ | $0.0035$ ($60\text{ s}$ feed) | `SemibatchReactor` with $t_{\text{add}} \ge 60\text{ s}$ (ideal 0D model agrees with 3D CFD to $R^2 = 0.979$) |
| Large-scale single-use mixer (`SUM`, $400\text{ rpm}$, $P/V = 23.71\text{ W/m}^3$) | $25.0$ | $17.6$ | $2.81$ | $141$ | $0.035$ ($60\text{ s}$ feed) | `SemibatchReactor` with subsurface payload dosing ($t_{\text{add}} = 60\text{–}300\text{ s}$) |
| Large-scale glass stirred tank (`GST-2`, $60\text{ rpm}$, $P/V = 2.44\text{ W/m}^3$) | $22.0$ | $32.2$ | $5.13$ | $257$ | $0.20$ (bolus, $t_{\text{add}}=0\text{ s}$); $0.08$ ($60\text{ s}$); $< 0.01$ ($300\text{ s}$) | **Mixing-sensitive under bolus addition** ($\text{Da}_{\text{plume}} \gg 1$); requires `SemibatchReactor` controlled addition ($t_{\text{add}} \ge 300\text{ s}$) to restore 0D validity |
| Continuous flow microreactor / static mixer (Nakahara et al., 2022, `10.1021/acs.oprd.2c00217`) | $1.0 \times 10^{-3}\text{–}1.0 \times 10^{-2}$ | $< 0.05$ | $< 0.008$ | $< 0.40$ | Negligible | `Mixer` + `PlugFlowReactor` (plug-flow axial transport with rapid radial micromixing) |

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

#### Forward multi-wavelength observation operator

For a spectroscopic flow cell or probe of optical pathlength $\ell$ $[\text{cm}]$
measuring absorbance across $N_\lambda$ wavelengths
$\boldsymbol{\lambda} = [\lambda_1, \dots, \lambda_{N_\lambda}]$ $[\text{nm}]$ at
$N_t$ time points, the Beer–Lambert linear superposition law gives the absorbance
matrix $\mathbf{S} \in \mathbb{R}^{N_t \times N_\lambda}$ $[\text{-}]$
($\text{AU}$):

$$\mathbf{S}(t_k, \lambda_m) = \ell \sum_{j=1}^{N_{\text{species}}} c_j(t_k) \, \varepsilon_j(\lambda_m) \quad [\text{-}], \qquad \mathbf{S} = \ell \, \mathbf{C} \, \mathbf{E}$$

where $\mathbf{C} \in \mathbb{R}^{N_t \times N_{\text{species}}}$ has units of
$[\text{mol/L}]$ and the molar extinction coefficient matrix
$\mathbf{E} \in \mathbb{R}^{N_{\text{species}} \times N_\lambda}$ has units of
$[\text{L}\cdot\text{mol}^{-1}\cdot\text{cm}^{-1}]$
($\text{M}^{-1}\cdot\text{cm}^{-1}$). For unit pathlength ($\ell = 1.0\text{ cm}$),
$\mathbf{S} = \mathbf{C}\mathbf{E}$ matches the bilinear measurement equation
`S = C @ A` of `MultipleCurveResolution` in `PharmaPy.ParamEstim` (`A` in
$[\text{L/mol}]$ for fixed $\ell = 1.0\text{ cm}$).

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

| Chromophore / Species | $\varepsilon(\lambda_1 = 248\text{ nm})$ $[\text{L}\cdot\text{mol}^{-1}\cdot\text{cm}^{-1}]$ | $\varepsilon(\lambda_2 = 280\text{ nm})$ $[\text{L}\cdot\text{mol}^{-1}\cdot\text{cm}^{-1}]$ | Physical Basis & Source |
| --- | --- | --- | --- |
| Monoclonal antibody (`cAC10` IgG1, $\text{mAb}_0$) | $9.413 \times 10^4$ | $2.3415 \times 10^5$ | Intact antibody molar basis; Hamblett et al. (2004), *Clin. Cancer Res.* 10(20), 7063–7070, p. 7064 (`doi:10.1158/1078-0432.CCR-04-0789`) |
| Maleimide linker-payload (`vc-MMAE`, `SGD-1006`) | $1.59 \times 10^4$ | $1.50 \times 10^3$ | Per mole of drug-linker moiety; Hamblett et al. (2004), p. 7064 |
| Conjugated species $\text{mAb}_k$ ($k \in \{0,1,\dots,8\}$) | $\varepsilon_{\text{mAb}}^{248} + k\,\varepsilon_{D,\text{conj}}^{248}$ | $\varepsilon_{\text{mAb}}^{280} + k\,\varepsilon_{D,\text{conj}}^{280}$ | Additive chromophore model; in situ during reaction, subtle succinimide thioether vs. free maleimide spectral shifts ($\Delta\lambda \approx 2\text{–}15\text{ nm}$ across $240\text{–}360\text{ nm}$) require multivariate deconvolution (`MultipleCurveResolution` / `PCR_calibration`) or soft-sensor state estimation (Schiemer et al., 2023, `doi:10.1002/bit.28428`) |

#### In-situ observability nuance (Schiemer et al., 2023, `10.1002/bit.28428`)

During the conjugation reaction (prior to Stage 3 UF/DF desalting), the reaction
mixture contains both conjugated payload ($c_{\text{mAb\_1Drug}} + 2\,c_{\text{mAb\_2Drug}}$)
and unconjugated payload ($c_{\text{Drug}} + c_{\text{Drug\_depl}}$). If
$\varepsilon_{D,\text{conj}}(\lambda) = \varepsilon_{\text{Drug}}(\lambda) = \varepsilon_{\text{Drug\_depl}}(\lambda)$
at all wavelengths, then by the drug-moiety conservation invariant
$c_{\text{Drug}}(t) + c_{\text{Drug\_depl}}(t) + c_{\text{mAb\_1Drug}}(t) + 2\,c_{\text{mAb\_2Drug}}(t) = c_{\text{Drug},0}$,
the dual-wavelength absorbance vector $[A(248), A(280)]^\top$ would remain time-invariant
in a closed batch reactor! Real-time in-situ UV-Vis monitoring therefore relies on
**either**:
1. **Spectral differentiation of the maleimide ring**: Conjugation (thiol addition
   across the maleimide $\text{C}=\text{C}$ double bond to form a succinimidyl
   thioether) and maleimide hydrolysis (ring opening to maleamic acid) alter the
   maleimide $\pi \to \pi^*$ transition ($290\text{–}350\text{ nm}$ and
   $220\text{–}260\text{ nm}$), giving distinct pure-component spectra
   $\boldsymbol{\varepsilon}_{\text{mAb}}(\lambda)$,
   $\boldsymbol{\varepsilon}_{\text{conj\_site}}(\lambda)$,
   $\boldsymbol{\varepsilon}_{\text{Drug}}(\lambda)$, and
   $\boldsymbol{\varepsilon}_{\text{Drug\_depl}}(\lambda)$ that can be resolved
   by `MultipleCurveResolution` (`PharmaPy.ParamEstim`) or `PCR_calibration`
   (`PharmaPy.Calibration`, with PLS/chemometric expansion tracked in
   [#352](https://github.com/PharmaPy-org/PharmaPy/issues/352)); **and**
2. **Soft-sensor fusion with the kinetic ODE model** (Schiemer et al., 2023,
   *Biotechnol. Bioeng.* 120(7), 1914–1928, `doi:10.1002/bit.28428`), which
   combines the spectroscopic prediction of total conjugated drug with the
   mechanistic reaction model to reconstruct the individual conjugate states
   ($\text{mAb}_0, \text{mAb}_1, \text{mAb}_2, D, D_{\text{depl}}$).

### 6.3 Channel 3 — In-line Raman spectroscopy for disulfide reduction and conjugation

For Stage 1 TCEP disulfide reduction and Stage 2 thiol–maleimide conjugation
(Pople et al., 2026, *Pharm. Res.*, `doi:10.1007/s11095-026-04154-9`), in-line
Raman spectra track disulfide $\nu(\text{S–S})$ stretching
($500\text{–}550\text{ cm}^{-1}$), thiol $\nu(\text{S–H})$ stretching
($2550\text{–}2600\text{ cm}^{-1}$), and maleimide $\nu(\text{C}=\text{C})$ /
$\nu(\text{C}=\text{O})$ bands ($1700\text{–}1770\text{ cm}^{-1}$) through
PharmaPy's `PCR_calibration` (`PharmaPy.Calibration`, with optional SNV
preprocessing repaired in `#138` and `#253`, and PLS expansion tracked in `#352`)
to predict molar species concentrations $[\text{mol/L}]$.

---

## 7. Existing PharmaPy reaction/estimation paths and demonstrated blockers

### 7.1 Verified working paths on `master` (`3194efd0`)

| Capability | PharmaPy Class / Module | Status on `master` (`3194efd0`) |
| --- | --- | --- |
| Elementary & custom reaction kinetics and Jacobians | `RxnKinetics` (`PharmaPy/Kinetics.py`) | Ready: parses `rxn_list`, computes $\mathbf{r}_{\text{species}}$ $[\text{mol}\cdot\text{L}^{-1}\cdot\text{s}^{-1}]$, analytical state Jacobian $\partial \mathbf{r}/\partial \mathbf{c}$, and parameter Jacobian $\partial \mathbf{r}/\partial \boldsymbol{\theta}$ |
| Batch, fed-batch, and plug-flow reactor simulation & sensitivities | `BatchReactor`, `SemibatchReactor`, `PlugFlowReactor` (`PharmaPy/Reactors.py`) | Ready: isothermal and non-isothermal integration and forward sensitivity Jacobians (`#55` closed) |
| Mixing and hold-tank unit operations | `Mixer`, `DynamicCollector` (`PharmaPy/Containers.py`) | Ready for inline dilution/mixing (`Mixer`) and dynamic hold/pooling (`DynamicCollector`) |
| Structured experiment & measurement containers | `Experiment`, `Measurement`, `ParameterEstimation` (`PharmaPy/ParamEstim.py`) | Ready (`#484`, `#236`, `#238`, `#239`, `#252` closed): handles multi-experiment batch + fed-batch fits, missing observations (`NaN`), per-sample uncertainty weighting, `estimate_parameters`, and `get_confidence_intervals` |
| Calibration provenance & deterministic replay | `CalibrationRecord`, `DataSource`, `Transformation`, `ModelRevision` (`PharmaPy/Records.py`) | Ready (`#351` / `#480` closed): JSON round-trip, SHA-256 source pinning, `origin='literature'` / `'synthetic'`, and `CalibrationRecord.rerun` |
| Chemometric & self-modeling spectral calibration | `MultipleCurveResolution` (`PharmaPy/ParamEstim.py`), `PCR_calibration` (`PharmaPy/Calibration.py`) | Ready (`#138`, `#240`, `#253`, `#254` closed): MCR bilinear $S = C A$ decomposition and PCR `get_regression`/`predict` with SNV preprocessing |

### 7.2 Demonstrated open blockers and workarounds for `#356`

| Open Issue | Defect / Gap Summary | Impact on WP1.5 Bioconjugation Case (`#356`) | Disposition / Workaround on `master` |
| --- | --- | --- | --- |
| [#275](https://github.com/PharmaPy-org/PharmaPy/issues/275) | `RxnKinetics`: toggling `reformulate_kin=False` with a finite `temp_ref` switches to centered Arrhenius $k_i \exp(-(E_{a,i}/R)(1/T - 1/T_{\text{ref}}))$ instead of preserving standard Arrhenius | Affects multi-temperature Arrhenius fitting of Stage 1 TCEP disulfide reduction ($4\text{–}37\ ^\circ\text{C}$, $T_{\text{ref}} = 293.15\text{ K}$, Weggen et al., 2025) if comparing `reformulate_kin=True` vs. `False` at finite `temp_ref`; **does not affect isothermal Stage 2 conjugation** ($E_a = 0\text{ J/mol}$) | Keep `ea_params=[0.0, ...]` and `temp_ref=np.inf` for isothermal conjugation; resolve `#275` before multi-temperature Stage 1 TCEP Arrhenius comparison in `#356` |
| [#339](https://github.com/PharmaPy-org/PharmaPy/issues/339) | `RxnKinetics`: align reaction specifications and parameters by reaction identifier (`rxns={'coupling': ..., 'iso': ...}`) | Needed for the dictionary-keyed reaction syntax proposed in the Lilly deck (`Pharmapy software development - v2.pdf`, slide 4) | Use positional `rxn_list=[...]` and `k_params=[...]` on `master` until `#339` lands |
| [#338](https://github.com/PharmaPy-org/PharmaPy/issues/338), [#341](https://github.com/PharmaPy-org/PharmaPy/issues/341) | Named dictionary composition inputs on `LiquidPhase`/`LiquidStream` (`#338`) and dictionary indexing on `DynamicResult.retrieve_state` (`#341`) | Ergonomic API improvements from the Lilly deck (`#337`) | Use database-ordered arrays and `output_names` tuple mapping on `master` until `#338`/`#341` land |
| [#283](https://github.com/PharmaPy-org/PharmaPy/issues/283), [#287](https://github.com/PharmaPy-org/PharmaPy/issues/287), [#290](https://github.com/PharmaPy-org/PharmaPy/issues/290) | Custom reaction-order species indexing (`#283`), reversible negative-concentration flooring (`#287`), and repeated species in one reaction string (`#290`) | Selected bioconjugation reactions are irreversible elementary reactions with distinct reactant/product tokens, so `#283`, `#287`, and `#290` are not triggered by Models 1A/1B | Avoid writing buffer/catalyst on both sides of a reaction string (`#290`) unless `#290` is merged |
| [#440](https://github.com/PharmaPy-org/PharmaPy/issues/440) | `Reactors`: result retrieval re-completes the solvent after a volume-changing reaction | In dilute aqueous bioconjugation ($c_{\text{mAb}} \le 10^{-4}\text{ mol/L}$, $c_{\text{Drug}} \le 10^{-3}\text{ mol/L}$), reaction volume change is $< 10^{-4}$, and `paramest_wrapper` reads integrated solute columns directly | Track `#440` for concentrated non-ideal liquid mixtures; negligible effect on dilute aqueous bioconjugation solute columns |
| [#355](https://github.com/PharmaPy-org/PharmaPy/issues/355) | `[WP1.3][M6] Shared model-based experiment-design interface` | Shared MBDoE API (`D`/`A`/`E`-optimality ranking across candidate feed durations $t_{\text{add}}$ and initial payload excesses) will be consumed by `#356` | Weighted sensitivity Jacobians $\mathbf{J}$ and Fisher Information Matrix $\mathbf{F} = \mathbf{J}\mathbf{J}^\top$ are already computable from `ParameterEstimation.get_gradient(params, out_array=True)` on `master`; adopt shared `#355` interface in M9–M10 once merged |
| `SemibatchReactor` forward sensitivities (`eval_sens=True`) | `BatchReactor.solve_unit(eval_sens=True)` supports forward CVode sensitivity integration (`#55`), whereas `SemibatchReactor.solve_unit(eval_sens=True)` raises `NotImplementedError` (`tests/test_reactor_modes.py`) | Fed-batch parameter estimation (`Weggen et al., 2024`, $10\text{–}30\text{ min}$ syringe-pump feed runs) cannot use `eval_sens=True` on `SemibatchReactor` | Run `SemibatchReactor` with `eval_sens=False` so `ParameterEstimation` uses finite-difference sensitivities (`dx_finitediff`), or implement `SemibatchReactor` forward sensitivities during M7–M8 (`#356`) |
| Micromolar CVode `atol` scaling (`BatchReactor` / `SemibatchReactor`) | Assimulo CVode's default absolute tolerance is `atol = 1e-6` $[\text{mol/L}]$, which is $3\%$ of a $3.33 \times 10^{-5}\text{ mol/L}$ ($5\text{ g/L}$) antibody concentration and allows $O(10^{-6})\text{ mol/L}$ negative-concentration undershoot on $\text{mAb}_0/\text{mAb}_1$ and unphysical $\text{DAR} > 2$ ($\approx 2.14$) | Directly affects all micromolar $[\text{mol/L}]$ bioconjugation reactor solves and parameter estimation wrappers | Always pass `sundials_opts={'rtol': 1e-8, 'atol': 1e-11}` (or `atol <= 1e-10` $[\text{mol/L}]$) to `solve_unit` and `run_args={'sundials_opts': {'rtol': 1e-8, 'atol': 1e-11}}` in `Experiment` / `SetParamEstimation` |

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
     including the $248/280\text{ nm}$ dual-wavelength channel) and to
     `MultipleCurveResolution` / `PLS_calibration` spectral deconvolution.
   - Quantify the semi-batch payload addition window ($t_{\text{add}} \in [0, 1800]\text{ s}$)
     across lab (`GST-1`, $0.3\text{ L}$) and manufacturing (`GST-2`, $22\text{ L}$;
     `SUM`, $25\text{ L}$) mixing timescales.
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
     ($\det \mathbf{F}$) to maximize identifiability of consecutive conjugation
     vs. payload depletion.
   - Export and archive the fitted `CalibrationRecord` JSON with verified
     `DataSource` provenance (`#351`).
3. **Months 11–12 (M11–M12 — End-to-End 4-Step Flowsheet, WP4 Scheduling Export,
   and D3b Notebook Delivery)**:
   - Assemble the four-step bioconjugation manufacturing sequence
     (Reduction $\to$ Conjugation $\to$ Desalting/UF-DF $\to$ Purification) and
     export its STN/RTN scheduling representation for WP4 (`#366`, `#367`).
   - Author the self-contained Sphinx/Jupyter tutorial notebook for deliverable
     D3b (`#356`) and pass domain review with `@dcasasor`.

### 8.2 Four-step bioconjugation process-sequence inputs for WP4 (`#366`, `#367`)

Work Package 4 (`PAT Proposal Purdue-Lilly 360.pdf`, p. 5; `PharmaPy_PAT_progress_2026-10-02.pdf`,
slide 11; issues [#366](https://github.com/PharmaPy-org/PharmaPy/issues/366) and
[#367](https://github.com/PharmaPy-org/PharmaPy/issues/367)) maps a PharmaPy
flowsheet into State-Task Network (STN) and Resource-Task Network (RTN)
scheduling models. Below is the canonical four-task bioconjugation manufacturing
sequence, including states, tasks, equipment resources, batch capacities,
durations $[\text{s}]$, mass/molar yield ratios, and intermediate hold-time
constraints (storage policies):

| Task ID & Name | Input States (Material Proportion $[-]$) | Output States (Material Proportion $[-]$) | Equipment Resource Class & PharmaPy Unit | Nominal Batch Size / Volume Basis | Nominal Duration $\tau_{\text{task}}$ $[\text{s}]$ (Range $[\text{s}]$) | Transfer / Storage Policy Between Tasks |
| --- | --- | --- | --- | --- | --- | --- |
| **`T1_reduction`** (Disulfide decapping / partial TCEP reduction) | `S0_native_mAb` ($0.980$), `S_TCEP_reductant` ($0.020$) | `S1_reduced_mAb` ($1.000$) | `R_reduction_vessel` (`BatchReactor` or `SemibatchReactor`, jacketed at $T = 293.15\text{ K}$) | $V_{\text{batch}} = 20.0\text{ L}$ ($c_{\text{mAb},0} = 10.0\text{ g/L} = 6.67 \times 10^{-5}\text{ mol/L}$, $0.200\text{ kg}$ $\text{mAb}$) | $5400\text{ s}$ ($3600\text{–}7200\text{ s}$, $1.0\text{–}2.0\text{ h}$, Weggen et al., 2025) | **Finite-wait / Zero-wait sensitive**: maximum hold $\tau_{\text{hold,max}} = 1800\text{ s}$ ($0.5\text{ h}$) before inter-HL disulfide re-oxidation ($k_{\text{reox}} = 0.0365\text{ L}\cdot\text{mol}^{-1}\cdot\text{s}^{-1}$, Weggen et al., 2025) degrades reactive thiol count |
| **`T2_conjugation`** (Semi-batch maleimide payload addition + NAC quench) | `S1_reduced_mAb` ($0.900$), `S_payload_DMSO` ($0.090$), `S_NAC_quench` ($0.010$) | `S2_crude_ADC` ($1.000$) | `R_conjugation_vessel` (`SemibatchReactor` with syringe/metering pump, $T = 295.15\text{ K}$, $10\%\text{ v/v}$ DMSO) | $V_{\text{batch}} = 22.2\text{ L}$ ($60\text{–}1800\text{ s}$ payload feed + reaction hold + $300\text{ s}$ NAC quench) | $2700\text{ s}$ ($1800\text{–}3600\text{ s}$, $0.5\text{–}1.0\text{ h}$, Weggen et al., 2023, 2024) | **Finite-wait sensitive**: maximum hold $\tau_{\text{hold,max}} = 3600\text{ s}$ ($1.0\text{ h}$) in $10\%\text{ v/v}$ DMSO to prevent DMSO-induced protein aggregation prior to buffer exchange |
| **`T3_desalting_ufdf`** (Ultrafiltration / diafiltration small-molecule clearance & buffer exchange) | `S2_crude_ADC` ($0.125$), `S_diafiltration_buffer` ($0.875$, $7\text{–}10$ diavolumes) | `S3_desalted_ADC` ($0.120$, $>99\%$ free payload/DMSO/TCEP/NAC cleared), `S_ufdf_permeate_waste` ($0.880$) | `U_ufdf_skid` ($30\text{ kDa}$ MWCO tangential-flow filtration skid; Fernandez-Cerezo et al., 2023, `10.1021/acs.oprd.3c00137`) | $V_{\text{retentate}} = 20.0\text{ L}$ ($7\text{–}10$ diavolumes = $140\text{–}200\text{ L}$ buffer) | $4500\text{ s}$ ($3600\text{–}5400\text{ s}$, $1.0\text{–}1.5\text{ h}$) | **Stable intermediate (`Unlimited` / `Finite` hold $\le 24\text{ h}$ at $277.15\text{–}281.15\text{ K}$)** in `DynamicCollector` hold tank; dual-wavelength UV-Vis ($248/280\text{ nm}$) release check valid here |
| **`T4_purification`** (Polishing chromatography for DAR distribution & aggregate removal) | `S3_desalted_ADC` ($0.250$), `S_chrom_mobile_phase` ($0.750$) | `S4_purified_ADC_DS` ($0.220$, $\ge 90\%$ monomer/target-DAR yield), `S_chrom_strip_waste` ($0.780$) | `U_chrom_column` (Preparative HIC / AEX column skid) | $0.180\text{ kg}$ purified ADC drug substance per cycle | $3600\text{ s}$ ($2700\text{–}5400\text{ s}$, $0.75\text{–}1.5\text{ h}$) | Final Drug Substance (`S4_purified_ADC_DS`) to formulation/fill-finish storage |

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
