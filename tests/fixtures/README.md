# Native four-species fixture

`native_four_species.json` supplies a fixed-density four-component liquid to
analytical native transfer, integration, and conservation tests. Molecular
weights are in g/mol and liquid densities in kg/m3. It preserves the property
values previously borrowed from the E. coli example; no biological mechanism
or experimental trajectory is involved. Tests specify any needed constant
heat capacities explicitly. This fixture makes native regressions independent
of demonstration folders.
