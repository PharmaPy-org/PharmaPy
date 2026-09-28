# -*- coding: utf-8 -*-
"""
Cake physics correlations shared by the solid-liquid separation units.

These four functions carry no state and belong to no class. They were defined
at the top of ``SolidLiquidSep`` and imported from there by ``Drying_Model``,
which made a legacy module part of another module's public API. Keeping them
here lets the legacy units, the refactored ones and the dryer draw on one copy
instead of drifting apart.

``SolidLiquidSep`` re-exports every name below, so the original import path
still works.
"""

import numpy as np

eps = np.finfo(float).eps * 1.1
grav = 9.8  # m/s**2


def high_resolution_fvm(f, boundary_cond, limiter_type='Van Leer'):

    # Ghost cells -1, 0 and N + 1 (see LeVeque 2002, Chapter 9)
    f_extrap = 2*f[-1] - f[-2]
    f_aug = np.concatenate(([boundary_cond]*2, f, [f_extrap]))

    f_diff = np.diff(f_aug, axis=0)

    theta = (f_diff[:-1]) / (f_diff[1:] + eps)

    if limiter_type == 'Van Leer':
        limiter = (np.abs(theta) + theta) / (1 + np.abs(theta))
    else:  # TODO: include more limiters
        pass

    fluxes = f_aug[1:-1] + 0.5 * f_diff[1:] * limiter

    return fluxes


def upwind_fvm(f, boundary_cond):
    f_aug = np.concatenate(([boundary_cond], f))

    return f_aug


def get_alpha(solid_phase, porosity, sphericity, rho_sol, csd=None):
    # if csd is None:
    #     csd = solid_phase.distrib

    # x_grid = solid_phase.x_distrib

    # alpha_x = 180 * (1 - porosity) / \
    #     (porosity**3 * (x_grid*1e-6)**2 * rho_sol * sphericity**2)

    # numerator = trapezoidal_rule(x_grid, csd * alpha_x)
    # denominator = solid_phase.moments[0]

    # alpha = numerator / (denominator + eps)
    csd = solid_phase.distrib
    rho_sol = solid_phase.getDensity()
    x_grid = solid_phase.x_distrib * 1e-6

    kv = 0.524  # converting number based CSD to volume based:

    del_x_dist = np.diff(x_grid)
    node_x_dist = (x_grid[:-1] + x_grid[1:]) / 2
    node_CSD = (csd[:-1] + csd[1:]) / 2

    # Volume of crystals in each bin
    vol_cry = node_CSD * del_x_dist * (kv * node_x_dist**3)
    frac_vol_cry = vol_cry / (np.sum(vol_cry) + eps)

    csd = vol_cry

    # Calculate irreducible saturation in weighted csd (volume based)
    vol_frac = vol_cry/ np.sum(vol_cry)
    x_grid = node_x_dist
    alpha_x = 180 * (1 - porosity) / porosity**3 / x_grid**2 / rho_sol
    alpha = np.sum(alpha_x * vol_frac)

    return alpha


def get_sat_inf(x_vec, csd, deltaP, porosity, height, mu_zero, props):
    surf_tens, rho_liq = props

    kv = 0.524  # converting number based CSD to volume based:

    del_x_dist = np.diff(x_vec)
    node_x_dist = (x_vec[:-1] + x_vec[1:]) / 2
    node_CSD = (csd[:-1] + csd[1:]) / 2

    x_vec = node_x_dist
    if isinstance(surf_tens, float) or isinstance(rho_liq, float):
        capillary_number = porosity**3 * x_vec**2 * \
            (rho_liq*grav*height + deltaP) / (1 - porosity)**2 / height / surf_tens
    else:
        capillary_number = np.outer(
            porosity**3 * x_vec**2,
            (rho_liq*grav*height + deltaP)/(1 - porosity)**2 / height / surf_tens
            )
    # Volume of crystals in each bin
    vol_cry = node_CSD * del_x_dist * (kv * node_x_dist**3)
    frac_vol_cry = vol_cry / (np.sum(vol_cry) + eps)

    csd = vol_cry

    s_inf = 0.155 * (1 + 0.031*capillary_number**(-0.49))
    s_inf = np.where(s_inf > 1, 1, s_inf)

    # Calculate irreducible saturation in weighted csd (volume based)
    vol_frac = vol_cry/ np.sum(vol_cry)

    s_inf = np.sum(vol_frac *s_inf)

    return s_inf
