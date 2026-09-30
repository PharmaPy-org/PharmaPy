"""Declared bioreactor units and conversions into native PharmaPy source bases.

This module converts quantities, not scientific interpretations. Network columns
and kinetic targets must already use compatible reaction bases. Molecular weights
and densities remain supplied by native phases; no chemical data are inferred.
"""

import json
import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np


class BioreactorUnitConverter:
    """Compile biological rate mappings and convert supported physical quantities."""

    TIME = {"s": 1., "min": 60., "h": 3600., "day": 86400.}
    VOLUME = {"m3": 1000., "L": 1., "mL": 1e-3, "uL": 1e-6}
    MASS = {"kg": 1000., "g": 1., "mg": 1e-3, "ug": 1e-6, "ng": 1e-9, "pg": 1e-12}
    MOLAR = {"mol": 1., "mmol": 1e-3, "umol": 1e-6, "nmol": 1e-9, "pmol": 1e-12, "fmol": 1e-15}
    CELLS = {"cell": 1., "10^6 cell": 1e6}

    def __init__(self, definition=None, *, flux_time_unit=None, flux_basis=None,
                 population_basis='10^6 cell'):
        self.definition = definition
        self.population_basis = population_basis
        self.expected_rate_units = {}
        if definition is not None:
            self._configure_flux_basis()
        elif flux_time_unit is not None:
            if flux_time_unit not in {'1/' + unit for unit in self.TIME}:
                raise ValueError("pathway flux_time_unit must use s, min, h, or day")
            self.per_second = 1. / self.TIME[flux_time_unit[2:]]
            self.exchange_scale = 1.
            self.flux_basis = dict(flux_basis or {})
            if self.flux_basis:
                if set(self.flux_basis) != {'exchange', 'growth'}:
                    raise ValueError('pathway flux_basis requires exchange and growth')
                exchange, growth = self.flux_basis['exchange'], self.flux_basis['growth']
                for declaration in (exchange, growth):
                    if not isinstance(declaration, Mapping) or declaration.get('time_unit') != flux_time_unit[2:]:
                        raise ValueError('pathway bases must agree with flux_time_unit')
                factor, _ = self.rate_basis(exchange, normalization='kgDW', time_unit=flux_time_unit[2:])
                if exchange['amount_unit'] not in self.MOLAR:
                    raise ValueError('pathway exchanges require molecular amount units')
                self.exchange_scale = factor * self.MOLAR[exchange['amount_unit']]
                if not np.isfinite(self.exchange_scale) or self.exchange_scale <= 0.:
                    raise ValueError('exchange basis conversion must be finite and positive')
                if growth != dict(amount_unit='1', normalization='none', time_unit=flux_time_unit[2:]):
                    raise ValueError('pathway growth must be specific growth in the declared time unit')

    @classmethod
    def from_inputs(cls, case, mechanism):
        """Read mappings or JSON paths using the same adapter as build_bioreactor."""
        from .input_format import normalize_forward_inputs
        from .culture import CultureModelDefinition
        case = dict(case) if isinstance(case, Mapping) else json.loads(Path(case).read_text())
        mechanism = dict(mechanism) if isinstance(mechanism, Mapping) else json.loads(Path(mechanism).read_text())
        case, mechanism = normalize_forward_inputs(case, mechanism)
        if case['biology']['mechanism'] != mechanism['mechanism']:
            raise ValueError("case and mechanism declarations do not agree")
        if mechanism['mechanism'] == 'pathway-lp':
            return cls(flux_time_unit=mechanism['flux_time_unit'],
                       flux_basis=mechanism['model'].get('flux_basis'))
        if mechanism['mechanism'] == 'configured-kinetics':
            return cls()
        if mechanism['mechanism'] != 'rate-reconciled-culture':
            raise ValueError("unsupported biological mechanism")
        model = dict(mechanism['model'], parameters=mechanism['parameters'])
        basis = 'kgDW' if mechanism['state'].get('biomass_kg') is not None else '10^6 cell'
        return cls(CultureModelDefinition.from_mapping(model), population_basis=basis)

    def validate_rate_units(self, units):
        if any(units.get(name) != unit for name, unit in self.expected_rate_units.items()):
            raise ValueError("provider output units disagree with the declared flux basis")

    @classmethod
    def mass_rates(cls, amount_rates, molecular_weights, *, amount_unit, time_unit):
        """Map accepted molecular sources to native kg/s, with no change to fluxes."""
        if amount_unit not in cls.MOLAR or time_unit not in cls.TIME:
            raise ValueError('source rates require molecular amount and time units')
        rates = np.asarray(amount_rates, dtype=float)
        weights = cls._property(molecular_weights, 'molecular_weights')
        if rates.shape != weights.shape or not np.isfinite(rates).all():
            raise ValueError('source rates and molecular weights must be finite and aligned')
        return rates * weights * (cls.MOLAR[amount_unit] / 1000.) / cls.TIME[time_unit]

    @classmethod
    def rate_basis(cls, declaration, *, growth=False, normalization='10^6 cell', time_unit='day'):
        """Convert a rate denominator; default target is per-million-cell, per-day."""
        required = {"amount_unit", "normalization", "time_unit"}
        optional = {"biomass_g_per_million_cells", "amount_per_million_cells"}
        if (not isinstance(declaration, Mapping) or not required <= set(declaration)
                or set(declaration) - required - optional):
            raise ValueError("rate basis requires amount_unit, normalization and time_unit")
        unit = declaration["amount_unit"]
        if time_unit not in cls.TIME:
            raise ValueError('unsupported target time unit')
        times = {unit: cls.TIME[time_unit] / scale for unit, scale in cls.TIME.items()}
        if declaration["time_unit"] not in times:
            raise ValueError("rate time_unit must be s, min, h, or day")
        factor = times[declaration["time_unit"]]
        target = normalization
        normalization = declaration["normalization"]
        if growth and unit == "1":
            if normalization != "none" or set(declaration) != required:
                raise ValueError("specific growth requires normalization none and no content conversion")
            return factor, "1/" + declaration["time_unit"]
        if unit not in (set(cls.MOLAR) | {unit + "-C" for unit in cls.MOLAR} | set(cls.MASS)):
            raise ValueError("unsupported rate amount_unit")
        mass = {'gDW': 1., 'kgDW': 1000.}
        cells = cls.CELLS
        if normalization not in mass.keys() | cells.keys() or target not in mass.keys() | cells.keys():
            raise ValueError("rate normalization must be cell, 10^6 cell, gDW, or kgDW")
        if normalization in mass and target in mass:
            factor *= mass[target] / mass[normalization]
        elif normalization in cells and target in cells:
            factor *= cells[target] / cells[normalization]
        else:
            content = float(cls._property(declaration.get('biomass_g_per_million_cells'),
                                          'biomass_g_per_million_cells'))
            factor *= (cells[target] / 1e6 * content / mass[normalization]
                       if normalization in mass else mass[target] * 1e6 / cells[normalization] / content)
        if 'biomass_g_per_million_cells' in declaration:
            cls._property(declaration['biomass_g_per_million_cells'], 'biomass_g_per_million_cells')
        if growth:
            if target != '10^6 cell':
                raise ValueError('amount-based growth requires the per-million-cell target basis')
            content = float(declaration.get("amount_per_million_cells", 0.))
            if not np.isfinite(content) or content <= 0.:
                raise ValueError("biomass growth flux requires positive amount_per_million_cells in amount_unit")
            factor /= content
        elif "amount_per_million_cells" in declaration:
            raise ValueError("amount_per_million_cells is only used for growth")
        if not np.isfinite(factor) or factor <= 0.:
            raise ValueError("rate basis conversion must be finite and positive")
        return factor, f"{unit}/({normalization} {declaration['time_unit']})"

    @classmethod
    def rate_quantity(cls, quantity, target_basis, **properties):
        """Convert a positive policy scale into declared network coordinates."""
        if not isinstance(quantity, Mapping) or set(quantity) != {'value', 'basis'}:
            raise ValueError("rate scale requires value and basis (null means source coordinates)")
        value = float(quantity['value'])
        if not np.isfinite(value) or value <= 0.:
            raise ValueError("rate scale must be finite and positive")
        source = quantity['basis']
        if source is None:
            return value
        if not target_basis:
            raise ValueError("unit-bearing scales require a declared network flux basis")
        source, target = dict(source), dict(target_basis)
        # A flux scale measures the declared amount, not the growth it produces.
        source.pop('amount_per_million_cells', None)
        target.pop('amount_per_million_cells', None)
        specific = source.get('amount_unit') == target.get('amount_unit') == '1'
        if (source.get('amount_unit') == '1' or target.get('amount_unit') == '1') and not specific:
            raise ValueError("specific growth and amount flux scales are not interchangeable")
        normalization = target['normalization']
        if 'biomass_g_per_million_cells' not in source and 'biomass_g_per_million_cells' in target:
            source['biomass_g_per_million_cells'] = target['biomass_g_per_million_cells']
        left, _ = cls.rate_basis(source, growth=specific, normalization=normalization,
                                 time_unit=target['time_unit'])
        right, _ = cls.rate_basis(target, growth=specific, normalization=normalization,
                                  time_unit=target['time_unit'])
        amount = 1. if specific else cls.convert(1., source['amount_unit'], target['amount_unit'], **properties)
        converted = value * left / right * amount
        if not np.isfinite(converted) or converted <= 0.:
            raise ValueError("converted rate scale must be finite and positive")
        return converted

    def reconciliation_target_scales(self, policy, reactions):
        """Preserve a declared dimensionless residual metric across flux units."""
        if not isinstance(policy, Mapping) or set(policy) != {'target_scales'}:
            raise ValueError('unit-scaled reconciliation requires target_scales only')
        declared = policy['target_scales']
        if not isinstance(declared, Mapping) or set(declared) != set(reactions):
            raise ValueError('target_scales must cover exactly the targeted reactions')
        scales = {}
        for reaction, group in reactions.items():
            quantity = declared[reaction]
            if not isinstance(quantity, Mapping) or quantity.get('basis') is None:
                raise ValueError('target_scales require explicit reference bases')
            properties = ({key: self.definition.product_mapping[key] for key in
                           ('molecular_weight_g_mol', 'mass_per_cmol_g')
                           if key in self.definition.product_mapping} if group == 'product' else {})
            scales[reaction] = self.rate_quantity(
                quantity, self.definition.flux_basis.get(group), **properties)
        return scales

    def reconciliation_scales(self, policy, reactions, target_floor=None):
        """Compile floors and optional normalization without assigning biology."""
        policy = dict(policy or {})
        if set(policy) - {'target_floor', 'target_floors', 'internal_scale'}:
            raise ValueError("unknown reconciliation scaling fields")
        overrides = policy.get('target_floors', {})
        if not isinstance(overrides, Mapping) or set(overrides) - set(reactions):
            raise ValueError("target_floors must reference declared kinetic reactions")
        basis = self.definition.flux_basis
        default = {'value': 1e-6 if target_floor is None else target_floor, 'basis': None}
        internal_floor = self.rate_quantity(policy.get('target_floor', default),
                                            basis.get('exchange'))
        floors = {}
        for reaction, group in reactions.items():
            declaration = overrides.get(reaction, policy.get('target_floor'))
            floors[reaction] = (internal_floor if declaration is None else
                                self.rate_quantity(declaration, basis.get(group)))
        internal = policy.get('internal_scale')
        included = [group for reaction, group in reactions.items()
                    if reaction not in self.definition.normalization_exclusions]
        coordinates = {tuple(basis[group][key] for key in
                             ('amount_unit', 'normalization', 'time_unit'))
                       for group in included} if basis else set()
        declared = [policy.get('target_floor'), *overrides.values()]
        physical = any(item is not None and item['basis'] is not None for item in declared)
        if physical and len(coordinates) > 1 and internal is None:
            raise ValueError("mixed flux coordinates require an explicit internal_scale")
        internal = None if internal is None else self.rate_quantity(internal, basis.get('exchange'))
        return floors, internal_floor, internal

    def _configure_flux_basis(self):
        """Compile source conversions once; keep undeclared legacy bases explicit."""
        definition = self.definition
        constants, product = definition.constants, definition.product_mapping
        basis = definition.flux_basis
        if self.population_basis not in {'10^6 cell', 'kgDW'}:
            raise ValueError('unsupported population basis')
        if self.population_basis == 'kgDW' and not basis:
            raise ValueError('biomass reconciliation requires declared flux_basis')
        if not basis:
            self.exchange_scale = constants["cell_flux_to_mmol"]
            self.growth_scale = constants.get("growth_flux_to_per_day", 1.)
            self.product_mass = float(product["molecular_weight_g_mol"])
            self.product_scale = constants["cell_product_scale"]
            return
        if set(basis) != {"exchange", "growth", "product"}:
            raise ValueError("flux_basis requires exchange, growth and product declarations")
        if constants["seconds_per_day"] != 86400.:
            raise ValueError("explicit flux bases require seconds_per_day = 86400")
        scales, units = {}, {}
        for name, declaration in basis.items():
            scales[name], units[name] = self.rate_basis(
                declaration, growth=name == "growth", normalization=self.population_basis)
        if len({item["time_unit"] for item in basis.values()}) != 1:
            raise ValueError("network flux bases must share one time unit")
        normalizations = {item["normalization"] for item in basis.values() if item["amount_unit"] != "1"}
        if len(normalizations) != 1:
            raise ValueError("network production fluxes must share one population normalization")
        molar = self.MOLAR
        exchange_unit = basis["exchange"]["amount_unit"]
        if exchange_unit not in molar:
            raise ValueError("extracellular exchange basis must use a supported molecular amount unit")
        self.exchange_scale = scales["exchange"] * (molar[exchange_unit] * 1000.)
        self.growth_scale = scales["growth"]
        product_unit = basis["product"]["amount_unit"]
        if product_unit in self.MASS:
            self.product_mass = 1.
            self.product_scale = scales["product"] * self.MASS[product_unit]
        else:
            carbon = product_unit.endswith("-C")
            key = "mass_per_cmol_g" if carbon else "molecular_weight_g_mol"
            self.product_mass = float(product.get(key, 0.))
            if not np.isfinite(self.product_mass) or self.product_mass <= 0.:
                raise ValueError(f"product basis requires positive {key}")
            self.product_scale = scales["product"] * molar[product_unit.removesuffix("-C")]
        for name, factor in (("cell_flux_to_mmol", self.exchange_scale),
                             ("growth_flux_to_per_day", self.growth_scale),
                             ("cell_product_scale", self.product_scale)):
            if not np.isfinite(factor) or factor <= 0.:
                raise ValueError("flux basis conversion must be finite and positive")
            if name in constants and not np.isclose(constants[name], factor, rtol=1e-12, atol=0.):
                raise ValueError(f"legacy {name} conflicts with declared flux_basis; remove it")
        outputs = definition.kinetic_outputs
        self.expected_rate_units = {outputs["growth"]: "1/day", outputs["death"]: "1/day",
                                    **{item["rate_output"]: "1/day" for item in definition.degradation_mappings}}
        if outputs.get("dead_removal"):
            self.expected_rate_units[outputs["dead_removal"]] = "1/day"
        for reaction, output in outputs["exchange_fluxes"].items():
            group = ("product" if reaction == product["reaction"] else
                     "growth" if reaction == outputs.get("growth_reaction") else "exchange")
            if output in self.expected_rate_units and self.expected_rate_units[output] != units[group]:
                raise ValueError("one provider output cannot represent incompatible flux bases")
            self.expected_rate_units[output] = units[group]

    @staticmethod
    def _property(value, name):
        if value is None:
            raise ValueError(f"conversion requires {name}")
        value = np.asarray(value, dtype=float)
        if not np.isfinite(value).all() or np.any(value <= 0.):
            raise ValueError(f"{name} must be finite and positive")
        return value

    @classmethod
    def convert(cls, value, source, target, *, molecular_weight_g_mol=None,
                mass_per_cmol_g=None, carbon_atoms=None,
                biomass_g_per_million_cells=None):
        """Convert scalars/arrays in supported units, without inferring properties.

        Supports time, volume, mass, molecular/carbon amount, cell count,
        temperature, concentrations, volume flows and amount-per-volume rates.
        Compound biological
        rates use rate_basis and the configured source mappings instead.
        """
        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError("source and target units must be strings")
        value = np.asarray(value, dtype=float)
        if not np.isfinite(value).all():
            raise ValueError("converted quantities must be finite")
        properties = dict(molecular_weight_g_mol=molecular_weight_g_mol,
                          mass_per_cmol_g=mass_per_cmol_g, carbon_atoms=carbon_atoms,
                          biomass_g_per_million_cells=biomass_g_per_million_cells)
        aliases = {"M": "mol/L", "mM": "mmol/L", "uM": "umol/L", "nM": "nmol/L"}
        source, target = aliases.get(source, source), aliases.get(target, target)
        rate_pattern = r'([^/]+)/\(([^ ]+) ([^ ]+)\)'
        source_rate, target_rate = re.fullmatch(rate_pattern, source), re.fullmatch(rate_pattern, target)
        temperatures = {"K", "degC", "degF"}
        if source_rate and target_rate:
            amount_s, volume_s, time_s = source_rate.groups()
            amount_t, volume_t, time_t = target_rate.groups()
            if time_s not in cls.TIME or time_t not in cls.TIME:
                raise ValueError('unsupported rate time units')
            result = cls.convert(value, amount_s + '/' + volume_s,
                                 amount_t + '/' + volume_t, **properties)
            result *= cls.TIME[time_t] / cls.TIME[time_s]
        elif source in temperatures and target in temperatures:
            kelvin = value if source == "K" else (
                value + 273.15 if source == "degC" else (value - 32.) * 5. / 9. + 273.15)
            if np.any(kelvin < 0.):
                raise ValueError("temperature cannot be below absolute zero")
            result = kelvin if target == "K" else (
                kelvin - 273.15 if target == "degC" else (kelvin - 273.15) * 9. / 5. + 32.)
        elif "/" in source and "/" in target:
            left, right = source.split("/"), target.split("/")
            amounts = set(cls.MASS) | set(cls.MOLAR) | set(cls.CELLS) | {unit + '-C' for unit in cls.MOLAR}
            if (len(left) == len(right) == 2 and left[0] in cls.VOLUME and right[0] in cls.VOLUME
                    and left[1] in cls.TIME and right[1] in cls.TIME):
                result = cls.convert(value, left[0], right[0]) * cls.TIME[right[1]] / cls.TIME[left[1]]
            elif (len(left) != 2 or len(right) != 2 or left[0] not in amounts or right[0] not in amounts
                    or left[1] not in cls.VOLUME or right[1] not in cls.VOLUME):
                raise ValueError("unsupported concentration units")
            else:
                result = cls.convert(value, left[0], right[0], **properties) * cls.VOLUME[right[1]] / cls.VOLUME[left[1]]
        else:
            carbon = {unit + '-C': scale for unit, scale in cls.MOLAR.items()}
            families = (cls.TIME, cls.VOLUME, cls.MASS, cls.MOLAR, carbon, cls.CELLS)
            common = next((units for units in families if source in units and target in units), None)
            if common is not None:
                result = value * (common[source] / common[target])
            elif ((source in cls.MOLAR and target in carbon)
                  or (source in carbon and target in cls.MOLAR)):
                count = (cls._property(carbon_atoms, 'carbon_atoms') if carbon_atoms is not None else
                         cls._property(molecular_weight_g_mol, 'molecular_weight_g_mol') /
                         cls._property(mass_per_cmol_g, 'mass_per_cmol_g'))
                result = (value * cls.MOLAR[source] * count / carbon[target] if source in cls.MOLAR else
                          value * carbon[source] / count / cls.MOLAR[target])
            elif source in cls.MASS or target in cls.MASS:
                amount, to_mass = (source, True) if target in cls.MASS else (target, False)
                if amount in cls.MOLAR:
                    grams = cls.MOLAR[amount] * cls._property(molecular_weight_g_mol, 'molecular_weight_g_mol')
                elif amount in carbon:
                    grams = carbon[amount] * cls._property(mass_per_cmol_g, 'mass_per_cmol_g')
                elif amount in cls.CELLS:
                    grams = cls.CELLS[amount] / 1e6 * cls._property(biomass_g_per_million_cells, 'biomass_g_per_million_cells')
                else:
                    raise ValueError(f"unsupported conversion from {source} to {target}")
                result = value * grams / cls.MASS[target] if to_mass else value * cls.MASS[source] / grams
            else:
                raise ValueError(f"unsupported conversion from {source} to {target}")
        if not np.isfinite(result).all():
            raise ValueError("unit conversion produced a nonfinite result")
        return float(result) if np.ndim(result) == 0 else np.array(result, copy=True)
