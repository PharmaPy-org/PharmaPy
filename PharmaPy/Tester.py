from PharmaPy.Phases import LiquidPhase, SolidPhase
from PharmaPy.Streams import LiquidStream
from PharmaPy.Reactors_refactor import ContinuousReactor,BatchReactor
from PharmaPy.IntegratorBackends import AssimuloBackend
from PharmaPy.Kinetics import RxnKinetics,CrystKinetics
from PharmaPy.Crystallizers_Refactor import BatchCrystallizer, ContinuousCrystallizer
from PharmaPy.Utilities import CoolingWater
from PharmaPy.ProcessControl_Refactor import Controller,DefaultContinuousVesselVolume, SimpleTemperatureController
from PharmaPy.Mechanisms import OneDFVMMechanism


import numpy as np

def build_crysts(params):
    kb, b, kg, g, beta = params

    return {
        "nucl_prim": (kb, 0, b),
        "nucl_sec":  (0, 0, 0, 0),
        "growth":    (kg, 0, g, 1, beta),
    }


dpath = r"C:\Users\zhillma\OneDrivePZH\Documents\Documents\_Grad_School\mypharmadev\PharmaPy\tests\Flowsheet\data\compound_database.json"

def temp_profile(x):
    """
    Linear profile from (0, 303) to (101, 271).
    """
    return np.interp(x, [0, 101], [303, 271])

temp_control = SimpleTemperatureController(temp_func=temp_profile)
# -----------------------------
# Reactor Setup
# -----------------------------

integrator = AssimuloBackend(options={'maxh':0.1})

vessel = BatchCrystallizer(
    integrator=integrator,
    h_conv=10000,
    diam=.01,
    controller=temp_control

)


# -----------------------------
# Initial phases
# -----------------------------
m = 1
liquid1 = LiquidPhase(
    dpath,
    mass=m,
    mass_frac=[0.4,0.6,0,0,0],

)
print("starting vol:", liquid1.vol)
solid1 = SolidPhase(
    dpath,
    mass=0,
    mass_frac=[0,0,1,0,0],
)
fvm = OneDFVMMechanism(solid1,target_components='C',solvent_name='solvent',x_grid=np.arange(1,200),distrib_init=np.zeros(199))
solid1.mechanisms = fvm
vessel.Phases = [liquid1,solid1]


# -----------------------------
# Feed
# -----------------------------

inlet = LiquidStream(
    dpath,
    mass_flow=m/100,
    mass_frac=[0.4,0.6,0,0,0]
)
rxns = ['A + B --> C', 'C + A --> D']
kvals_rxns = np.array([1,1e-1])#, 1e2]) #psuedo instantaneous
ea_vals = np.array([1e3,1e4])#,1e4]) #psuedo no activation energy
Rkinetics = RxnKinetics(path=dpath,rxn_list=rxns, k_params=kvals_rxns,ea_params=ea_vals)
fitted_kinetics = np.array([7.46855218e+10, 6.30671806e+00, 2.15782420e+03, 3.82241037e+00, 3.93676056e+00]) # HP volfunc big bounds
cryst_kinetics =build_crysts(fitted_kinetics)
Ckinetics = CrystKinetics(np.array([-28.13909202,0.001,5.900800253]),**cryst_kinetics,solubility_type='apelblat')
Utility = CoolingWater(mass_flow=100, temp_in=273.55)
# vessel.Utility = Utility
# vessel.RxnKinetics = Rkinetics
vessel.CrystKinetics = Ckinetics
# vessel.controller.target_volume=1e-2
# vessel.Inlet = inlet


# -----------------------------
# Solve
# -----------------------------

vessel.solve_unit(runtime=100)


# -----------------------------
# Check outlet
# -----------------------------

print("done")

# print(
#     "Outlet flow:",
#     vessel.Outlet.mass_flow
# )

# print(
#     "Outlet composition:",
#     vessel.Outlet.mass_frac
# )
# print("outlet temp:",
#       vessel.Outlet.temp)

print(
    "Vessel Final composition:",
    vessel.Phases.mass_frac
)
print(
    "Vessel Final mass:",
    vessel.Phases.mass
)
print(
    "Vessel Final vol:",
    vessel.Phases.vol
)
print(
    "Vessel Final temp:",
    vessel.Phases.temp
)
import matplotlib.pyplot as plt

if True:
    for mj,spec in zip(vessel.result.mass_j_liquid0.T,['A','B','C','D','Solvent']):
        if spec=='Solvent':continue
        plt.plot(vessel.result.time,mj,label=spec)
    plt.plot(vessel.result.time,vessel.result.Total_m_in_vessel, label='Total m')

    plt.legend()
    plt.show()
    print(vessel.result.time,vessel.result.Total_m_in_vessel)