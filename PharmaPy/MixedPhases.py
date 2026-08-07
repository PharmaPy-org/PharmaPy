# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:43:14 2020

@author: dcasasor
"""

from collections.abc import Sequence

from PharmaPy.Phases import classify_phases
from PharmaPy.Interpolation import NewtonInterpolation
from PharmaPy.Commons import trapezoidal_rule

import numpy as np
from scipy.optimize import newton
from scipy.interpolate import CubicSpline

def Interpolation(t_data, y_data, time):
    idx_time = np.argmin(abs(time - t_data))

    idx_lower = max(0, idx_time - 1)
    idx_upper = idx_lower + 3

    t_interp = t_data[idx_lower:idx_upper]
    y_interp = y_data[idx_lower:idx_upper]

    interp = NewtonInterpolation(t_interp, y_interp)

    y_target = interp.evalPolynomial(time)

    return y_target


def energy_balance(inst, mass_str):
    masses = [getattr(phase, mass_str) for phase in inst.Phases]
    h_in = [phase.getEnthalpy() for phase in inst.Phases]
    mass_tot = sum(masses)

    def fun(temp_out):
        h_tot = inst.getEnthalpy(temp_out, volumetric=False)
        balance = np.dot(masses, h_in) - mass_tot * h_tot

        return balance

    temps = [phase.temp for phase in inst.Phases]
    temp_phase = newton(fun, np.mean(temps))

    return temp_phase


class Slurry:
    """ Create Slurry object.
    
    Parameters
    ----------
    vol : float, optional
        Volume of the Slurry [m**3]. The default is 0.
    moments : array, optional
        Array of size N, containing the distribution moments in um**n, 
        for n = 0,...,N - 1. The default is None.
    # mass_slurry : TYPE, optional
        DESCRIPTION. The default is 0.
    x_distrib : array, optional
        Array of size N, containing the internal grid
        size coordinate of the solids [um]. The default is None
    distrib : array, optional
        Array of size N, constaining the initial distribution of crystals
        [#/m**3/um]. The default is None.

    Returns
    -------
    None.

    """
    def __init__(self, vol=0, moments=None,
                 # mass_slurry=0,
                 x_distrib=None, distrib=None):
       
        self._Phases = None

        self.vol = vol
        # self.mass_slurry = mass_slurry
        self.mass_slurry = None

        self.y_upstream = None
        self.moments = moments
        self.distrib = distrib
        self.x_distrib = x_distrib

        self.temp = None

        self.transferred_from_uo = False

    @property
    def Phases(self):
        return self._Phases

    @Phases.setter
    def Phases(self, phases_list: Sequence[object]) -> None:
        """Attach component phases and synchronize the slurry state.

        Parameters
        ----------
        phases_list : sequence of object
            Liquid and solid phase collaborators. For moment-based input,
            ``moments[3]`` is the volume-normalized third moment
            [m**3/m**3], and the solid phase supplies its volumetric shape
            factor ``kv`` [-].

        Raises
        ------
        ValueError
            If volume-specific moments are supplied for a zero-volume slurry.

        Notes
        -----
        The physical solid volume fraction is ``kv * moments[3]`` [-].
        For moment-based input, the solid mass [kg] is derived from that
        physical volume [m**3] and the solid mixture density [kg/m**3], then
        reconciled with the solid mole amount [mol].
        """
        if isinstance(phases_list, (list, tuple)):
            phases_list = list(phases_list)

        self._Phases = phases_list

        for phase in self._Phases:
            if phase.transferred_from_uo:
                self.transferred_from_uo = True
                break

        classify_phases(self)

        # TODO: this is not general enough (Dan - Energetics)
        if self.moments is not None:
            if self.vol == 0:
                raise ValueError('If the moments are provided, Slurry volume needs to be larger than 0.')

            solid_vol_frac = self.Solid_1.kv * self.moments[3]  # [-]
            vol_sol = self.vol * solid_vol_frac  # [m**3]
            vol_liq = self.vol * (1 - solid_vol_frac)  # [m**3]
            dens_sol = self.Solid_1.getDensity()  # [kg/m**3]
            mass_sol = vol_sol * dens_sol  # [kg]
            self.Solid_1.updatePhase(
                moments=self.moments * self.vol,
                mass=mass_sol,
            )
            self.Liquid_1.updatePhase(vol=vol_liq)
        elif self.distrib is None:
            vol_sol = self.Solid_1.vol
            vol_liq = self.Liquid_1.vol

            mass_liq = self.Liquid_1.mass
            mass_sol = self.Solid_1.mass

            self.vol = vol_sol + vol_liq
            self.mass_slurry = mass_liq + mass_sol

            self.x_distrib = self.Solid_1.x_distrib
            if self.Solid_1.x_distrib is None:
                self.distrib = self.Solid_1.distrib
                self.dx = None
                self.moments = self.Solid_1.moments
            else:
                self.distrib = self.Solid_1.distrib / self.vol

                self.dx = self.Solid_1.dx
                self.moments = self.Solid_1.getMoments(distrib=self.distrib)
        else:
            delta_x = np.diff(self.x_distrib)
            equal = np.isclose(delta_x[1:], delta_x[:-1]).all()

            if equal:
                self.dx = delta_x[0]
            else:  # assume geometric series and make adjustments
                ratio = self.x_distrib[1] / self.x_distrib[0]
                x_grid = np.zeros(len(self.x_distrib) + 1)
                x_gr = np.sqrt(self.x_distrib[1:] * self.x_distrib[:-1])

                x_grid[0] = x_gr[0] / ratio
                x_grid[-1] = x_gr[-1] * ratio

                x_grid[1:-1] = x_gr

                self.dx = np.diff(x_grid)

            self.Solid_1.x_distrib = self.x_distrib
            self.Solid_1.dx = self.dx

            self.moments = self.Solid_1.getMoments(distrib=self.distrib)

            if self.vol > 0:
                dens_liq = self.Liquid_1.getDensity()
                dens_sol = self.Solid_1.getDensity()
                dens_phases = np.array([dens_liq, dens_sol])

                vol_share = self.getFractions()
                vol_phases = vol_share * self.vol

                mass_liq, mass_sol = vol_phases * dens_phases
                self.mass_slurry = np.dot(vol_phases, dens_phases)
            elif self.mass_slurry > 0:
                mass_share = self.getFractions(vol_basis=False)
                mass_phases = self.mass_slurry * mass_share

                mass_liq, mass_sol = mass_phases

                self.vol = np.dot(mass_phases, 1/dens_phases)

            f_distr = self.vol * self.distrib

            self.Liquid_1.updatePhase(mass=mass_liq)
            self.Solid_1.updatePhase(distrib=f_distr, mass=mass_sol)

        self.num_species = self.Liquid_1.num_species
        self.temp = energy_balance(self, 'mass')

    def __check_distrib(self):
        if self.distrib is None:
            vol_sol = self.Solid_1.vol
            vol_liq = self.Liquid_1.vol

            mass_liq = self.Liquid_1.mass
            mass_sol = self.Solid_1.mass

            self.vol = vol_sol + vol_liq
            self.mass_slurry = mass_liq + mass_sol

            self.x_distrib = self.Solid_1.x_distrib
            if self.Solid_1.x_distrib is None:
                self.distrib = self.Solid_1.distrib
                self.dx = None
                self.moments = self.Solid_1.moments
            else:
                self.distrib = self.Solid_1.distrib / self.vol

                self.dx = self.Solid_1.dx
                self.moments = self.Solid_1.getMoments(distrib=self.distrib)
        else:
            delta_x = np.diff(self.x_distrib)
            equal = np.isclose(delta_x[1:], delta_x[:-1]).all()

            if equal:
                self.dx = delta_x[0]
            else:  # assume geometric series and make adjustments
                ratio = self.x_distrib[1] / self.x_distrib[0]
                x_grid = np.zeros(len(self.x_distrib) + 1)
                x_gr = np.sqrt(self.x_distrib[1:] * self.x_distrib[:-1])

                x_grid[0] = x_gr[0] / ratio
                x_grid[-1] = x_gr[-1] * ratio

                x_grid[1:-1] = x_gr

                self.dx = np.diff(x_grid)

            self.Solid_1.x_distrib = self.x_distrib
            self.moments = self.Solid_1.getMoments(distrib=self.distrib)

    def getDensity(self, temp=None, basis='mass', total=False):

        dens_liq = self.Liquid_1.getDensity(temp=temp, basis=basis)
        dens_solid = self.Solid_1.getDensity(temp=temp, basis=basis)

        if total:
            vfrac = self.getFractions()
            dens = np.array([dens_liq, dens_solid])
            density = np.dot(dens, vfrac)
        else:
            density = np.array([dens_liq, dens_solid])

        return density

    def getSolidsConcentr(self, distrib=None, basis='vol'):
        if distrib is None:
            moments = self.moments
        else:
            moments = self.Solid_1.getMoments(distrib=distrib)

        if moments.ndim == 1:
            mom_three = moments[3]  # m**3/m**3
        else:
            mom_three = moments[:, 3]  # m**3/m**3

        vol_frac = mom_three * self.Solid_1.kv
        dens_solid = self.Solid_1.getDensity(basis='mass')

        concentr_solids = vol_frac * dens_solid  # kg_solids / m**3

        if basis == 'mass':
            dens_liq = self.Liquid_1.getDensity(basis='mass')
            concentr_solids *= 1/(1 - vol_frac)/dens_liq

        return concentr_solids

    def getFractions(self, distrib=None, mu_3=None, vol_basis=True):
        if distrib is None and mu_3 is None:
            mom_three = self.moments[3]
        elif distrib is None:
            mom_three = mu_3
        elif mu_3 is None:
            mom_three = self.Solid_1.getMoments(distrib=distrib, mom_num=3)

        vol_solid = mom_three * self.Solid_1.kv
        vol_fracs = np.array([1 - vol_solid, vol_solid])

        if vol_basis:
            return vol_fracs

        else:
            density = self.getDensity()  # TODO: what is this?
            mass_phases = vol_fracs * density
            mass_fracs = mass_phases / mass_phases.sum()

            return mass_fracs

    def getTotalVol(self):
        vol_liq = self.Liquid_1.vol
        epsilon = 1 - self.Solid_1.kv * self.Solid_1.moments[3]

        vol_total = vol_liq / epsilon

        return vol_total

    def getEnthalpy(self, temp, volfracs=None, densMass=None, volumetric=True):
        # Individual phases
        hLiq = self.Liquid_1.getEnthalpy(temp=temp, basis='mass')
        hSol = self.Solid_1.getEnthalpy(temp=temp, basis='mass')

        hMass = np.array([hLiq, hSol])

        # Mixture
        if volfracs is None:
            volfracs = self.getFractions()

        if volumetric:
            if densMass is None:
                densMass = self.getDensity()

            hSlurry = sum(volfracs * densMass * hMass)  # J/m**3 susp
        else:
            phase_massfrac = self.getFractions(vol_basis=False)
            hSlurry = np.dot(phase_massfrac, hMass)

        return hSlurry

    def getCp(self, temp, volfracs=None, density=None, times_vliq=False,
              basis='mass'):

        # Individual phases
        cpLiq = self.Liquid_1.getCp(temp=temp, basis=basis)
        cpSol = self.Solid_1.getCp(temp=temp, basis=basis)

        cpPhases = np.array([cpLiq, cpSol])

        # Mixture
        if volfracs is None:
            volfracs = self.getFractions()
        volfracs = np.array(volfracs, copy=True)

        if density is None:
            density = self.getDensity()
        density = np.asarray(density)

        self.epsilon = volfracs.copy()
        self.densities = density

        if times_vliq:
            volfracs[1] *= 1/volfracs[0]

        cpSlurry = sum(volfracs * density * cpPhases)  # J/m**3/K

        return cpSlurry


class SlurryStream(Slurry):
    """ Create a slurry stream object.
    
    Parameters
    ----------
    vol_flow : float, optional
        Volumetric flow rate in which the slurry is transfered [m**3/s]. 
        The default is 0.
    moments : array, optional
        Array of size N, containing the distribution moments in um**n, 
        for n = 0,...,N - 1. The default is None.
    x_distrib : array, optional
        Array of size N, containing the internal grid
        size coordinate of the solids [um]. The default is None
    distrib : array, optional
        Array of size N, constaining the initial distribution of crystals
        [#/m**3/um]. The default is None.

    Returns
    -------
    None.

    """
    def __init__(self, vol_flow=0, moments=None, x_distrib=None, distrib=None):
        super().__init__(vol_flow, moments, x_distrib, distrib)

        self.mass_flow = self.mass_slurry  # TODO (this doesn't seem fine)
        # self.mole_flow = self.moles
        self.vol_flow = self.vol

        self.DynamicInlet = None
        self.controllable = ['vol_flow', 'temp']
        self.input_states = ['vol_flow', 'temp', 'distrib']

        # del self.mass
        # del self.moles
        # del self.vol

    @property
    def Phases(self):
        return self._Phases

    @Phases.setter
    def Phases(self, phases_list: Sequence[object]) -> None:
        """Attach component phases and synchronize the slurry-stream state.

        Parameters
        ----------
        phases_list : sequence of object
            Liquid and solid phase collaborators. For moment-based input,
            ``moments[3]`` is the volume-normalized third moment [m**3/m**3],
            and the solid phase supplies its volumetric shape factor ``kv``
            [-].

        Raises
        ------
        ValueError
            If volume-specific moments are supplied for a zero-volume slurry
            stream.

        Notes
        -----
        This override obtains phase volume shares from
        :meth:`Slurry.getFractions`, which applies ``kv`` when converting the
        third moment to a solid volume fraction [-]. Solid-stream mass and
        mole flow are reconciled through :meth:`SolidStream.updatePhase`.
        """
        if isinstance(phases_list, tuple):
            phases_list = list(phases_list)

        self._Phases = phases_list

        classify_phases(self)

        if self.moments is not None:
            if self.vol == 0:
                raise ValueError('If the moments are provided, Slurry volume needs to be larger than 0.')

            dens_liq = self.Liquid_1.getDensity()
            dens_sol = self.Solid_1.getDensity()
            dens_phases = np.array([dens_liq, dens_sol])

            vol_share = self.getFractions()
            vol_phases = vol_share * self.vol

            mass_liq, mass_sol = vol_phases * dens_phases  # [kg/s] each
            self.mass_slurry = np.dot(vol_phases, dens_phases)
            self.mass_flow = self.mass_slurry

            self.Liquid_1.updatePhase(mass_flow=mass_liq)

            self.Solid_1.updatePhase(
                moments=self.moments,
                mass_flow=mass_sol,
            )
            self.Solid_1.vol_flow = vol_phases[1]

        elif self.distrib is None:
            vol_sol = self.Solid_1.vol
            vol_liq = self.Liquid_1.vol

            mass_liq = self.Liquid_1.mass
            mass_sol = self.Solid_1.mass

            self.vol = vol_sol + vol_liq
            self.mass_slurry = mass_liq + mass_sol

            self.x_distrib = self.Solid_1.x_distrib
            self.distrib = self.Solid_1.distrib / self.vol

            self.dx = self.Solid_1.dx
            self.moments = self.Solid_1.getMoments(distrib=self.distrib)
        else:
            delta_x = np.diff(self.x_distrib)
            equal = np.isclose(delta_x[1:], delta_x[:-1]).all()

            if equal:
                self.dx = delta_x[0]
            else:  # assume geometric series and make adjustments
                ratio = self.x_distrib[1] / self.x_distrib[0]
                x_grid = np.zeros(len(self.x_distrib) + 1)
                x_gr = np.sqrt(self.x_distrib[1:] * self.x_distrib[:-1])

                x_grid[0] = x_gr[0] / ratio
                x_grid[-1] = x_gr[-1] * ratio

                x_grid[1:-1] = x_gr

                self.dx = np.diff(x_grid)

            # self.Solid_1.x_distrib = self.x_distrib
            self.moments = self.Solid_1.getMoments(self.x_distrib,
                                                   self.distrib)

            if self.vol > 0:
                dens_liq = self.Liquid_1.getDensity()
                dens_sol = self.Solid_1.getDensity()
                dens_phases = np.array([dens_liq, dens_sol])

                vol_share = self.getFractions()
                vol_phases = vol_share * self.vol

                mass_liq = vol_phases[0] * dens_liq  # [kg/s]
                self.mass_slurry = np.dot(vol_phases, dens_phases)
                self.mass_flow = self.mass_slurry

            elif self.mass_slurry > 0:
                dens_liq = self.Liquid_1.getDensity()  # [kg/m**3]
                dens_sol = self.Solid_1.getDensity()  # [kg/m**3]
                dens_phases = np.array(
                    [dens_liq, dens_sol]
                )  # [kg/m**3], ordered liquid then solid
                mass_share = self.getFractions(vol_basis=False)  # [-]
                mass_phases = self.mass_slurry * mass_share  # [kg/s] each
                vol_phases = mass_phases / dens_phases  # [m**3/s] each

                mass_liq = mass_phases[0]  # [kg/s]

                self.vol = vol_phases.sum()  # [m**3/s]
                self.mass_flow = self.mass_slurry  # [kg/s]

            f_distr = self.vol * self.distrib

            self.Liquid_1.updatePhase(mass_flow=mass_liq)

            # The total-population distribution [#/um] is the authoritative
            # solid inventory; its third moment determines mass flow [kg/s].
            self.Solid_1.updatePhase(
                x_distrib=self.x_distrib,
                distrib=f_distr,
            )
            self.Solid_1.vol_flow = vol_phases[1]

        self.num_species = self.Liquid_1.num_species
        self.temp = energy_balance(self, 'mass_flow')
        self.solid_conc = self.getSolidsConcentr(basis='mass')

    def InterpolateInputs(self, time):
        if isinstance(time, (float, int)):
            time = min(time, self.time_upstream[-1])

            y_interpol = Interpolation(self.time_upstream, self.y_inlet,
                                       time)
        else:
            interpol = CubicSpline(self.time_upstream, self.y_inlet)
            flags_interpol = time > self.time_upstream[-1]

            if any(flags_interpol):
                time_interpol = time[~flags_interpol]
                y_interp = interpol(time_interpol)

                y_extrapol = np.tile(y_interp[-1],
                                     (sum(flags_interpol), 1))
                y_interpol = np.vstack((y_interp, y_extrapol))
            else:
                y_interpol = interpol(time)

        return y_interpol

    def evaluate_inputs(self, time):
        if self.DynamicInlet is None:
            inputs = {}
            for attr in self.input_states:
                inputs[attr] = getattr(self, attr)

        else:
            inputs = self.DynamicInlet.evaluate_inputs(time)

        return inputs


class Cake:
    def __init__(self, z_external=None, num_discr=50, saturation=None):
        """ Create a cake object.

        Parameters
        ----------
        z_external : array, optional
            Array of size N, containing the internal spatial grid
            of length coordinate of the cakes [m]. The default is None
        num_discr : integer, optional
            The number which the cake length coordiante is discretized in. 
            The default is 50.
        saturation : array, optional
            Array of size N (N = num_discr). Volumetric liquid uptake 
            in each spatial node of the cake. The default is None.

        Returns
        -------
        None.

        """
    

        if z_external is None:
            self.z_external = np.linspace(0, 1, num_discr)
        else:
            self.z_external = z_external

        if saturation is None:
            self.saturation = np.ones(len(self.z_external))
        else:
            self.saturation = saturation

        self.mass_concentr = None

    @property
    def Phases(self):
        return self._Phases

    @Phases.setter
    def Phases(self, phases_list: Sequence[object]) -> None:
        """Attach component phases and compute packed-cake properties.

        Parameters
        ----------
        phases_list : sequence of object
            Liquid and solid phase collaborators. The solid third moment is
            the unscaled particle-volume moment [m**3], while ``kv`` is its
            volumetric shape factor [-].

        Notes
        -----
        The physical solid volume is ``kv * moments[3]`` [m**3]. Dividing by
        the packed solid fraction ``1 - porosity`` [-] gives cake volume
        [m**3].
        """
        self._Phases = list(phases_list)

        classify_phases(self)

        self.porosity = self.Solid_1.getPorosity()  # [-]
        solid_vol = self.Solid_1.kv * self.Solid_1.moments[3]  # [m**3]
        self.cake_vol = solid_vol / (1 - self.porosity)  # [m**3]
        self.alpha = self.get_alpha()

        self.num_species = self.Liquid_1.num_species

    def get_alpha(self) -> np.floating:
        """Calculate the volume-weighted specific cake resistance.

        Returns
        -------
        numpy.floating
            Specific cake resistance on a solid-mass basis [m/kg].

        Notes
        -----
        Crystal sizes are stored in micrometres and converted to metres for
        the resistance calculation. The number-based CSD may use any common
        number basis because only normalized bin weights are used.

        Each bin is weighted by ``n_i * delta_x_i * kv * x_i**3``. The solid
        phase supplies the scalar volumetric shape factor ``kv`` [-]. For any
        nonzero scalar value, it cancels exactly when these weights are divided
        by their sum, so the returned resistance is independent of ``kv``.

        The local resistance follows the Carman--Kozeny form
        ``180 * (1 - porosity) / (porosity**3 * rho_s * x_i**2)`` and assumes
        a positive size grid, nonzero ``kv``, and ``0 < porosity < 1``.
        The dimensionless coefficient 180 follows Carman (1937), Equation 10,
        with the measured Kozeny constant ``k = 5`` for packed granular beds.

        References
        ----------
        Carman, P. C. (1937). Fluid flow through granular beds. *Transactions
        of the Institution of Chemical Engineers*, 15, 150-166,
        https://doi.org/10.1016/S0263-8762(97)80003-2.
        """
        csd = self.Solid_1.distrib  # [common number basis/um]
        porosity = self.porosity  # [-]
        solid_density = self.Solid_1.getDensity()  # [kg/m**3]
        micrometer_to_meter = 1e-6  # [m/um], exact unit conversion
        size_grid = self.Solid_1.x_distrib * micrometer_to_meter  # [m]
        shape_factor = self.Solid_1.kv  # [-]

        bin_widths = np.diff(size_grid)  # [m]
        node_sizes = (size_grid[:-1] + size_grid[1:]) / 2  # [m]
        node_csd = (csd[:-1] + csd[1:]) / 2  # [common number basis/um]

        # The arbitrary common scale of these proportional-volume weights
        # cancels during normalization together with the scalar shape factor.
        volume_weights = (
            node_csd * bin_widths * (shape_factor * node_sizes**3)
        )  # [common proportional-volume basis]
        volume_fractions = volume_weights / np.sum(volume_weights)  # [-]

        # Carman--Kozeny coefficient from Carman (1937), Equation 10, with
        # the measured Kozeny constant k = 5.
        ck_coeff = 180  # [-]
        local_resistance = (
            ck_coeff * (1 - porosity)
            / porosity**3
            / node_sizes**2
            / solid_density
        )  # [m/kg]
        alpha = np.sum(local_resistance * volume_fractions)  # [m/kg]

        return alpha

    def getEnthalpy(self, temp=None, mass_frac=None, distrib=None):
        if temp is None:
            temp = self.Liquid_1.temp

        if mass_frac is None:
            mass_frac = self.Liquid_1.mass_frac

        # Individual phases
        hLiq = self.Liquid_1.getEnthalpy(temp=temp, mass_frac=mass_frac)
        hSol = self.Solid_1.getEnthalpy(temp=temp)

        porosity = self.Solid_1.getPorosity()
        porosities = [porosity, 1 - porosity]

        densities = self.getDensity()

        frac_mass = porosities * densities / np.dot(densities, porosities)

        enthalpy = np.dot(frac_mass, [hLiq, hSol])

        return enthalpy
