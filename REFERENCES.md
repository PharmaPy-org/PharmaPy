# References

This file gathers literature cited by PharmaPy source, tests, and documentation.

## Solid-Liquid Separation and Drying

- Destro, F.; Hur, I.; Wang, V.; Abdi, M.; Feng, X.; Wood, E.; Coleman, S.;
  Firth, P.; Barton, A.; Barolo, M.; Nagy, Z. K. (2021). Mathematical modeling
  and digital design of an intensified filtration-washing-drying unit for
  pharmaceutical continuous manufacturing. Chemical Engineering Science, 244,
  116803. DOI: https://doi.org/10.1016/j.ces.2021.116803.

## Crystallization Population Balances

- Golub, G. H.; Welsch, J. H. (1969). Calculation of Gauss quadrature rules.
  Mathematics of Computation, 23, 221-230.
  DOI: https://doi.org/10.1090/S0025-5718-69-99647-1.
- Marchisio, D. L.; Vigil, R. D.; Fox, R. O. (2003). Quadrature method of moments
  for aggregation-breakage processes. Journal of Colloid and Interface Science,
  258, 322-334. DOI: https://doi.org/10.1016/S0021-9797(02)00054-1.
- McGraw, R. (1997). Description of aerosol dynamics by the quadrature method of
  moments. Aerosol Science and Technology, 27, 255-265.
  DOI: https://doi.org/10.1080/02786829708965471.
- Scott, W. T. (1968). Analytic studies of cloud droplet coalescence I. Journal of
  the Atmospheric Sciences, 25, 54-65.
  DOI: https://doi.org/10.1175/1520-0469(1968)025<0054:ASOCDC>2.0.CO;2.
- Smoluchowski, M. (1917). Versuch einer mathematischen Theorie der
  Koagulationskinetik kolloider Lösungen. Zeitschrift für Physikalische Chemie,
  92, 129-168. DOI: https://doi.org/10.1515/zpch-1918-9209.
- Wheeler, J. C. (1974). Modified moments and Gaussian quadratures. Rocky Mountain
  Journal of Mathematics, 4, 287-296. DOI: https://doi.org/10.1216/RMJ-1974-4-2-287.
- Ziff, R. M.; McGrady, E. D. (1985). The kinetics of cluster fragmentation and
  depolymerisation. Journal of Physics A, 18, 3027-3037.
  DOI: https://doi.org/10.1088/0305-4470/18/15/026.

## Numerical Linear Algebra

- Golub, G. H.; Van Loan, C. F. (2013). Matrix Computations, 4th ed. Johns
  Hopkins University Press, Baltimore. ISBN: 978-1-4214-0794-4.
- van der Sluis, A. (1969). Condition numbers and equilibration of matrices.
  Numerische Mathematik, 14(1), 14-23. DOI:
  https://doi.org/10.1007/BF02165096.

## Numerical Optimization

- Nocedal, J.; Wright, S. J. (2006). Numerical Optimization, 2nd ed.
  Springer Series in Operations Research and Financial Engineering.
  Springer, New York. ISBN: 978-0-387-30303-1. DOI:
  https://doi.org/10.1007/978-0-387-40065-5. Section 8.1, eq. (8.5): error
  bound of forward-difference derivatives (default `jacobian_rtol` and
  sensitivity error model of `ParameterEstimation.get_identifiability`).
  Section 8.1, central differences: step proportional to the cube root of
  the output accuracy (`jac_module.CENTRAL_DIFFERENCE_REL_STEP` and the
  step choice of `ParameterEstimation.check_derivatives`).

## Parameter Identifiability

- Brun, R.; Reichert, P.; Künsch, H. R. (2001). Practical identifiability
  analysis of large environmental simulation models. Water Resources
  Research, 37(4), 1015-1030. DOI: https://doi.org/10.1029/2000WR900350.
  Collinearity index of `StatsModule.IdentifiabilityReport`.
