# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:43:14 2020

@author: dcasasor
"""

from collections.abc import Sequence
from typing import Union

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
    """Represent a homogeneous liquid-solid mixture with a shared temperature.
    
    Parameters
    ----------
    vol : float, optional
        Volume of the Slurry [m**3]. The default is 0.
    moments : array, optional
        Slurry-volume-specific distribution moments [m**n/m**3], shape
        (num_moments,), ordered from n = 0. The default is None.
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
            If volume-specific moments are supplied for a zero-volume slurry,
            or phase-only input has zero combined liquid and solid volume.

        Notes
        -----
        The physical solid volume fraction is ``kv * moments[3]`` [-].
        For moment-based input, the solid mass [kg] is derived from that
        physical volume [m**3] and the solid mixture density [kg/m**3], then
        reconciled with the solid mole amount [mol]. With phase-only input
        and no size grid, total solid moments [m**n] are divided by the
        combined liquid and solid volume to obtain slurry moments [m**n/m**3].
        A supplied solid number distribution [#/um] is also divided by that
        volume to obtain [#/m**3/um]; an absent distribution remains None.
        Phase-only input requires positive combined volume: an empty mixture
        has neither defined volume-specific moments nor a thermal inventory
        from which to determine its mixture temperature.
        """
        if isinstance(phases_list, (list, tuple)):
            phases_list = list(phases_list)

        self._Phases = phases_list

        for phase in self._Phases:
            if phase.transferred_from_uo:
                self.transferred_from_uo = True
                break

        classify_phases(self)

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

            total_volume = vol_sol + vol_liq  # [m**3], combined phase inventory
            if np.any(total_volume == 0):
                raise ValueError(
                    "Cannot initialize Slurry from zero combined phase volume; "
                    "provide a positive liquid or solid inventory before "
                    "assigning Phases.")
            self.vol = total_volume  # [m**3]
            self.mass_slurry = mass_liq + mass_sol

            self.x_distrib = self.Solid_1.x_distrib
            if self.Solid_1.x_distrib is None:
                self.distrib = (None if self.Solid_1.distrib is None
                                else self.Solid_1.distrib / self.vol)  # [#/m**3/um]
                self.dx = None
                self.moments = self.Solid_1.moments / self.vol  # [m**n/m**3]
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

            dens_liq = self.Liquid_1.getDensity()  # [kg/m**3]
            dens_sol = self.Solid_1.getDensity()  # [kg/m**3]
            dens_phases = np.array([dens_liq, dens_sol])  # [kg/m**3]

            if self.vol > 0:
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

    def getTotalVol(self) -> Union[float, np.ndarray]:
        """Return total liquid-plus-solid slurry volume.

        Returns
        -------
        float or ndarray
            Sum of the reconciled phase volumes [m**3], retaining their
            scalar or array shape. No conversion from stored slurry moments
            is needed.

        Notes
        -----
        ``Slurry.vol`` records the initialized volume and is not refreshed by
        in-place phase updates. This method is the authoritative current
        total, computed from the reconciled liquid and solid phase volumes.
        """
        return self.Liquid_1.vol + self.Solid_1.vol

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

    def getCp(self, temp: float, volfracs=None, density=None,
              times_vliq: bool = False, basis: str = 'mass') -> Union[float, np.ndarray]:
        """Return mixture heat capacitance on the selected volume basis.

        Parameters
        ----------
        temp : float
            Common phase temperature [K].
        volfracs : array-like, optional
            Liquid and solid fractions of slurry volume [-], shape (2,).
            Defaults to the attached slurry fractions.
        density : array-like, optional
            Liquid and solid densities, shape (2,). Defaults to attached
            phase densities in the selected basis: [kg/m**3] for mass or
            [kmol/m**3] (equivalently [mol/L]) for mole.
        times_vliq : bool, optional
            If True, divide the entire slurry-volume capacitance by the
            liquid volume fraction. Multiplication by liquid volume then
            recovers total phase capacitance [J/K] for mass or [kJ/K] for
            mole, when volume is in [m**3]. Default False.
        basis : {'mass', 'mole'}, optional
            Basis passed to the phase Cp providers: [J/kg/K] or [J/mol/K].
            Default mass matches the default density basis.

        Returns
        -------
        float or ndarray
            Capacitance per slurry volume, or per liquid volume when
            ``times_vliq=True``. Mass basis returns [J/m**3/K]; mole basis
            returns [J/L/K] (equivalently [kJ/m**3/K]), since phase providers
            express molar density in [mol/L] and Cp in [J/mol/K].

        Notes
        -----
        Liquid-volume normalization requires a positive liquid fraction.
        Supplied fractions are copied and are not modified in place.
        """

        # Individual phases
        cpLiq = self.Liquid_1.getCp(temp=temp, basis=basis)
        cpSol = self.Solid_1.getCp(temp=temp, basis=basis)

        cpPhases = np.array([cpLiq, cpSol])

        # Mixture
        if volfracs is None:
            volfracs = self.getFractions()
        volfracs = np.array(volfracs, copy=True)

        if density is None:
            density = self.getDensity(basis=basis)
        density = np.asarray(density)

        self.epsilon = volfracs.copy()
        self.densities = density

        if times_vliq:
            volfracs = volfracs / volfracs[0]  # [m**3 phase/m**3 liquid]

        cpSlurry = sum(volfracs * density * cpPhases)  # [J/m**3/K] mass; [J/L/K] mole

        return cpSlurry


class SlurryStream(Slurry):
    """Represent a homogeneous liquid-solid flow with a shared temperature.
    
    Parameters
    ----------
    vol_flow : float, optional
        Volumetric flow rate in which the slurry is transfered [m**3/s]. 
        The default is 0.
    moments : array, optional
        Slurry-volume-specific distribution moments [m**n/m**3], shape
        (num_moments,), ordered from n = 0. The default is None.
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
        After every assignment, ``vol`` and ``vol_flow`` [m**3/s] and
        ``mass_slurry`` and ``mass_flow`` [kg/s] equal the constituent flow
        sums. With no distribution or moments supplied, constituent flows
        determine the total, replacing the constructor's flow value.
        Solid volume uses the inherited ``SolidStream.vol`` [m**3/s]; that
        stream need not expose a ``vol_flow`` alias. Liquid volume and mass
        use flow aliases because liquid updates delete phase amount attributes.
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

            self.Liquid_1.updatePhase(mass_flow=mass_liq)

            self.Solid_1.updatePhase(
                moments=self.moments,
                mass_flow=mass_sol,
            )
            self.Solid_1.vol_flow = vol_phases[1]

        elif self.distrib is None:
            vol_sol = self.Solid_1.vol  # [m**3/s]
            vol_liq = self.Liquid_1.vol_flow  # [m**3/s]

            self.vol = vol_sol + vol_liq  # [m**3/s]

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

                vol_share = self.getFractions()
                vol_phases = vol_share * self.vol

                mass_liq = vol_phases[0] * dens_liq  # [kg/s]

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

            f_distr = self.vol * self.distrib

            self.Liquid_1.updatePhase(mass_flow=mass_liq)

            # The total-population distribution [#/um] is the authoritative
            # solid inventory; its third moment determines mass flow [kg/s].
            self.Solid_1.updatePhase(
                x_distrib=self.x_distrib,
                distrib=f_distr,
            )
            self.Solid_1.vol_flow = vol_phases[1]

        self.vol = self.Liquid_1.vol_flow + self.Solid_1.vol  # [m**3/s]
        self.mass_slurry = (self.Liquid_1.mass_flow
                            + self.Solid_1.mass_flow)  # [kg/s]
        self.vol_flow = self.vol  # [m**3/s]
        self.mass_flow = self.mass_slurry  # [kg/s]

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

    def getEnthalpy(self, temp=None, mass_frac=None,
                    distrib=None) -> Union[float, np.ndarray]:
        """Return the mass-specific enthalpy of liquid and solid in a cake.

        Parameters
        ----------
        temp : float or ndarray, optional
            Common phase temperature [K], scalar or temperature profile;
            defaults to the liquid temperature.
        mass_frac : array-like, optional
            Liquid species mass fractions [-], shape (num_species,).
            Defaults to the attached liquid composition.
        distrib : array-like, optional
            Legacy unused distribution argument [#/um].

        Returns
        -------
        float or ndarray
            Enthalpy [J/kg of liquid plus solid inventory], relative to the
            phase providers' default reference temperature of 298.15 K. Array
            temperatures retain the providers' temperature axis.

        Notes
        -----
        The attached liquid and solid masses set the weights, matching the
        total inventory used by Mixer.energy_balance. Porosity and saturation
        metadata do not change these inventories. Gas mass and enthalpy are
        neglected.
        """
        if temp is None:
            temp = self.Liquid_1.temp  # [K]

        if mass_frac is None:
            mass_frac = self.Liquid_1.mass_frac  # [-]

        hLiq = self.Liquid_1.getEnthalpy(
            temp=temp, mass_frac=mass_frac, basis='mass')  # [J/kg liquid]
        hSol = self.Solid_1.getEnthalpy(temp=temp, basis='mass')  # [J/kg solid]

        mass_liq = self.Liquid_1.mass  # [kg]
        mass_sol = self.Solid_1.mass  # [kg]
        enthalpy = (mass_liq * hLiq + mass_sol * hSol) / (mass_liq + mass_sol)  # [J/kg]

        return enthalpy
