# Experimental single-phase density with Clapeyron

This example accompanies the property-interface work in [#353](https://github.com/PharmaPy-org/PharmaPy/issues/353). The provisional helper evaluates one homogeneous state at a time and returns two Python floats. `mass_density` is in kg/m³ and `molar_density` is in mol/m³. Supply absolute pressure in Pa, temperature in K and a one-dimensional mole-fraction vector in the supplied component order.

The helper supports `IAPWS95` for pure water and the default Peng–Robinson `PR` model for methane/ethane, in either order. Supply `phase="liquid"` or `phase="vapor"` to request a Clapeyron volume root. The requested phase guides root selection. The caller supplies the homogeneous phase composition and checks that the state lies in the model's physical domain. A volume root alone cannot establish phase stability or determine an equilibrium split. Unsupported model/component combinations and invalid inputs raise `ValueError`. A non-finite or non-positive backend result raises `RuntimeError`, and underlying Julia errors surface to the caller. [Clapeyron bulk properties](https://clapeyronthermo.github.io/Clapeyron.jl/v0.6/properties/bulk/) · [Peng and Robinson (1976)](https://doi.org/10.1021/i160057a011)

## Isolated setup and smoke test

The smoke test was run on Linux x86-64 with Python 3.12.12 and Julia 1.10.11. Use an absolute path to the matching Julia binary and run these commands from the repository root. The copied project and Julia depot live alongside the temporary Python environment. This smoke uses NumPy 2. Keep it in the isolated environment because the current PharmaPy Assimulo lock uses NumPy 1.26.4.

```bash
work_dir=$(mktemp -d)
example=doc/online_docs/examples/clapeyron_density
export PYTHON_JULIACALL_EXE=/absolute/path/to/julia-1.10.11/bin/julia
export PYTHON_JULIACALL_PROJECT="$work_dir/julia-project"
export PYTHON_JULIACALL_STARTUP_FILE=no
export JULIA_DEPOT_PATH="$work_dir/julia-depot:"
export JULIA_LOAD_PATH="@:@stdlib"
export JULIA_CONDAPKG_BACKEND=Null
export JULIA_NUM_THREADS=1
export JULIA_NUM_PRECOMPILE_TASKS=1
export OPENBLAS_NUM_THREADS=1

python3.12 -m venv "$work_dir/python"
"$work_dir/python/bin/python" -m pip install -r "$example/requirements.txt"
mkdir -p "$PYTHON_JULIACALL_PROJECT"
cp "$example/Project.toml" "$example/Manifest.toml" "$PYTHON_JULIACALL_PROJECT/"
"$PYTHON_JULIACALL_EXE" --startup-file=no --project="$PYTHON_JULIACALL_PROJECT" \
    -e 'using Pkg; Pkg.instantiate()'
"$work_dir/python/bin/python" -m pytest -q "$example/test_backend.py"
```

The Python file pins the smoke test's full Python environment, including pyclapeyron 0.1.1 and JuliaCall 0.9.37. The Julia project pins Clapeyron 0.6.21 and the matching PythonCall 0.9.37. The accompanying Julia dependency manifest pins the transitive versions and source trees used in the smoke test. Julia package downloads and initial precompilation can take several minutes. JuliaCall uses the selected executable and project. [pyclapeyron 0.1.1 metadata](https://pypi.org/project/pyclapeyron/0.1.1/) · [JuliaCall environment configuration](https://juliapy.github.io/PythonCall.jl/stable/juliacall/)

## Query pure water

Run this Python code from the repository root with the isolated environment above.

```python
from PharmaPy._clapeyron_density import single_phase_density

# IAPWS R6-95(2018), Table 7, first row.
temperature = 300.0           # [K]
pressure = 99_241.8352         # [Pa], converted from 0.0992418352 MPa
water_mole_fractions = [1.0]  # [-], pure water
density = single_phase_density(
    "IAPWS95", ("water",), temperature, pressure, water_mole_fractions,
    phase="liquid",
)

# 1 kmol = 1,000 mol and 1 m³ = 1,000 L by definition.
moles_per_kilomole = 1_000  # [mol/kmol]
pharmapy_molar_density = density.molar_density / moles_per_kilomole  # [kmol/m³]
print(density.mass_density)   # 996.5560 kg/m³
print(pharmapy_molar_density)  # 55.31730086 kmol/m³, equivalently mol/L
```

The molar conversion matches the unit convention of PharmaPy's `getDensity(basis="mole")`. The mass-density unit is the same for both calls. Importing the prototype is backend-free. Evaluation imports pyclapeyron lazily and requires the optional setup above.

## What the smoke test verifies

The Python test compares the wrapper's mass and molar densities with direct Julia evaluations from `smoke.jl`. It checks water against IAPWS R6-95(2018) Table 7 at 300 K, then checks the distinct homogeneous liquid and vapor roots against the first column of Table 8 at 275 K. The water checks verify the implementation against the published formulation's reference calculations. [IAPWS R6-95(2018), Tables 7 and 8](https://iapws.org/public/documents/k3PSK/IAPWS95-2018.pdf#page=15) · [IAPWS VSMOW molar mass](https://iapws.org/faqs/faq1)

For the 300 K, 100 kPa Peng–Robinson vapor example, methane/ethane mole fractions of 0.2/0.8 give about 1.100459929 kg/m³ and 40.36311360 mol/m³, or 0.04036311360 mol/L. The test reverses components and fractions together to preserve the state and reverses fractions alone to change it. Each mass-to-molar density ratio is compared with the mole-fraction-weighted NIST molar mass, allowing for the rounding used in Clapeyron's model data. This checks species order, units and cross-language handoff for the selected vapor state. [NIST methane](https://webbook.nist.gov/cgi/cbook.cgi?ID=C74828) · [NIST ethane](https://webbook.nist.gov/cgi/cbook.cgi?ID=C74840)
