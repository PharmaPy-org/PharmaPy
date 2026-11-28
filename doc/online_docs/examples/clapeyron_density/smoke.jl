using Clapeyron
using Pkg

pythoncall_version = only(info.version for info in values(Pkg.dependencies()) if info.name == "PythonCall")
println("versions ", VERSION, " ", pkgversion(Clapeyron), " ", pythoncall_version)

function print_densities(label, model, pressure, temperature, fractions, phase)
    # Scalar pressure [Pa], temperature [K], ordered mole fractions [-].
    mass = mass_density(model, pressure, temperature, fractions; phase) # [kg/m³]
    molar = molar_density(model, pressure, temperature, fractions; phase) # [mol/m³]
    println(label, " ", mass, " ", molar)
end

water = IAPWS95()
# IAPWS R6-95(2018), Tables 7 and 8, with MPa converted to Pa.
water_pressure = 99_241.8352 # [Pa], Table 7 first row
water_temperature = 300.0 # [K]
saturation_pressure = 698.451167 # [Pa], Table 8 first column
saturation_temperature = 275.0 # [K]
pure_water = [1.0] # [-], mole basis
print_densities("water", water, water_pressure, water_temperature, pure_water, :liquid)
print_densities("liquid", water, saturation_pressure, saturation_temperature, pure_water, :liquid)
print_densities("vapor", water, saturation_pressure, saturation_temperature, pure_water, :vapor)

# Dilute vapor example with unequal mole fractions to expose order errors.
binary_pressure = 100_000.0 # [Pa], one bar
binary_temperature = 300.0 # [K], near ambient
binary_fractions = [0.2, 0.8] # [-], methane-lean orientation
binary = PR(["methane", "ethane"])
print_densities("binary", binary, binary_pressure, binary_temperature, binary_fractions, :vapor)
