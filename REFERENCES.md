# References

This file gathers literature cited by PharmaPy source, tests, and documentation.

## Solid-Liquid Separation and Drying

- Destro, F.; Hur, I.; Wang, V.; Abdi, M.; Feng, X.; Wood, E.; Coleman, S.;
  Firth, P.; Barton, A.; Barolo, M.; Nagy, Z. K. (2021). Mathematical modeling
  and digital design of an intensified filtration-washing-drying unit for
  pharmaceutical continuous manufacturing. Chemical Engineering Science, 244,
  116803. DOI: https://doi.org/10.1016/j.ces.2021.116803.

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

## Parameter Identifiability

- Brun, R.; Reichert, P.; Künsch, H. R. (2001). Practical identifiability
  analysis of large environmental simulation models. Water Resources
  Research, 37(4), 1015-1030. DOI: https://doi.org/10.1029/2000WR900350.
  Collinearity index of `StatsModule.IdentifiabilityReport`.
