"""Immutable named definitions for steady-state metabolic networks.

Reaction declarations are validated and assembled into the ordered matrix,
objective, and bounds used in ``S v = 0`` optimization. A canonical content
digest supports lineage and caching.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ReactionDefinition:
    """Represent one bounded reaction column in a metabolic network."""

    reaction_id: str
    stoichiometry: Mapping[str, float]
    lower_bound: float
    upper_bound: float
    name: Optional[str] = None

    def __post_init__(self):
        if not self.reaction_id or not self.reaction_id.strip():
            raise ValueError("reaction_id must be nonempty.")
        values = dict(self.stoichiometry)
        if not values or any(not key for key in values):
            raise ValueError(f"reaction {self.reaction_id!r} requires named stoichiometry.")
        if not np.isfinite(list(values.values())).all():
            raise ValueError(f"reaction {self.reaction_id!r} has nonfinite stoichiometry.")
        if not np.isfinite([self.lower_bound, self.upper_bound]).all():
            raise ValueError(f"reaction {self.reaction_id!r} bounds must be finite.")
        if self.lower_bound > self.upper_bound:
            raise ValueError(f"reaction {self.reaction_id!r} lower bound exceeds upper bound.")
        object.__setattr__(self, "stoichiometry", MappingProxyType(values))


# Assembles reaction columns into the stoichiometric matrix, objective vector, and flux bounds.
class MetabolicNetworkDefinition:
    """Validated immutable `S v = 0` network with stable declared ordering."""

    def __init__(
        self,
        network_id: str,
        version: str,
        internal_metabolite_ids: Sequence[str],
        reactions: Sequence[ReactionDefinition],
        objective: Mapping[str, float],
        exchange_reaction_ids: Sequence[str] = (),
        representation="unspecified",
        exchange_roles=None,
    ):
        self.network_id = str(network_id)
        self.version = str(version)
        self.internal_metabolite_ids = tuple(internal_metabolite_ids)
        self.reactions = tuple(reactions)
        self.reaction_ids = tuple(item.reaction_id for item in self.reactions)
        self.exchange_reaction_ids = tuple(exchange_reaction_ids)
        if not self.network_id or not self.version:
            raise ValueError("network_id and version must be nonempty.")
        for label, identifiers in (
            ("metabolite", self.internal_metabolite_ids),
            ("reaction", self.reaction_ids),
            ("exchange reaction", self.exchange_reaction_ids),
        ):
            if len(set(identifiers)) != len(identifiers):
                raise ValueError(f"{label} IDs must be unique.")
        if not self.internal_metabolite_ids or not self.reactions:
            raise ValueError("network requires metabolites and reactions.")
        known_metabolites = set(self.internal_metabolite_ids)
        for reaction in self.reactions:
            unknown = set(reaction.stoichiometry) - known_metabolites
            if unknown:
                raise ValueError(
                    f"reaction {reaction.reaction_id!r} uses unknown "
                    f"metabolites {sorted(unknown)}."
                )
        reaction_set = set(self.reaction_ids)
        if set(objective) - reaction_set:
            raise ValueError("objective contains an unknown reaction.")
        if not objective or not np.isfinite(list(objective.values())).all():
            raise ValueError("objective must contain finite coefficients.")
        if set(self.exchange_reaction_ids) - reaction_set:
            raise ValueError("exchange_reaction_ids contains an unknown reaction.")
        if representation not in {"unspecified", "reduced", "chemical"}:
            raise ValueError("network representation must be reduced or chemical.")
        roles = dict(exchange_roles or {})
        if (set(roles) - set(self.exchange_reaction_ids)
                or set(roles.values()) - {"tracked", "external-supply", "untracked-exchange", "bookkeeping"}):
            raise ValueError("exchange roles must reference boundary reactions and supported roles.")
        self.representation = representation
        self.exchange_roles = MappingProxyType(roles)
        self.objective = MappingProxyType(dict(objective))
        metabolite_index = {name: index for index, name in enumerate(self.internal_metabolite_ids)}
        matrix = np.zeros((len(self.internal_metabolite_ids), len(self.reactions)))
        for column, reaction in enumerate(self.reactions):
            for name, coefficient in reaction.stoichiometry.items():
                matrix[metabolite_index[name], column] = coefficient
        matrix.setflags(write=False)
        self.stoichiometric_matrix = matrix
        for name, array in (
            ("lower_bounds", np.asarray([r.lower_bound for r in self.reactions], dtype=float)),
            ("upper_bounds", np.asarray([r.upper_bound for r in self.reactions], dtype=float)),
            (
                "objective_vector",
                np.asarray(
                    [objective.get(r, 0.0) for r in self.reaction_ids],
                    dtype=float,
                ),
            ),
        ):
            array.setflags(write=False)
            setattr(self, name, array)
        payload = {
            "network_id": self.network_id, "version": self.version,
            "metabolites": self.internal_metabolite_ids,
            "reactions": [
                {"id": r.reaction_id, "stoichiometry": dict(sorted(r.stoichiometry.items())),
                 "lower": r.lower_bound, "upper": r.upper_bound}
                for r in self.reactions
            ],
            "objective": dict(sorted(self.objective.items())),
            "exchange_reactions": self.exchange_reaction_ids,
        }
        if representation != "unspecified" or roles:
            payload.update(representation=representation, exchange_roles=roles)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        self.content_digest = hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_mapping(cls, payload):
        """Construct a network from a JSON-compatible native input mapping."""
        reactions = tuple(ReactionDefinition(
            reaction_id=item.get("reaction_id", item.get("identifier")),
            stoichiometry=item["stoichiometry"],
            lower_bound=item.get("lower_bound", item.get("lower")),
            upper_bound=item.get("upper_bound", item.get("upper")),
            name=item.get("name"),
        ) for item in payload["reactions"])
        return cls(
            network_id=payload.get("network_id", payload.get("identifier")),
            version=str(payload.get("version", "1.0")),
            internal_metabolite_ids=payload.get(
                "internal_metabolite_ids", payload.get("internal_metabolites")
            ),
            reactions=reactions,
            objective=payload["objective"],
            exchange_reaction_ids=payload.get(
                "exchange_reaction_ids", payload.get("exchange_reactions", ())
            ),
            representation=payload.get("representation") or "unspecified",
            exchange_roles=payload.get("exchange_roles"),
        )

    def audit_conservation(self):
        """Check internal stoichiometry without assuming molecular compositions.

        A positive conserved vector is necessary for mass consistency, not
        proof of elemental balance. Declared boundary exchanges are excluded.
        This diagnostic does not change bounds or repair lumped reactions.
        """
        columns = [i for i, name in enumerate(self.reaction_ids)
                   if name not in self.exchange_reaction_ids]
        if not columns:
            return {"stoichiometric_consistency": "NOT_ASSESSED",
                    "elemental_balance": "NOT_ASSESSED"}
        matrix = self.stoichiometric_matrix[:, columns].T.copy()
        scale = np.max(np.abs(matrix), axis=1)
        matrix /= np.where(scale > 0., scale, 1.)[:, None]
        # Homogeneity permits m >= 1 instead of a strict positivity constraint.
        result = linprog(np.ones(matrix.shape[1]), A_eq=matrix,
                         b_eq=np.zeros(len(columns)), bounds=(1., None),
                         method="highs")
        if result.success:
            residual = np.max(np.abs(matrix @ result.x))
            status = "PASS" if residual <= 1e-7 and np.min(result.x) >= 1. - 1e-7 else "UNRESOLVED"
        else:
            status = "FAIL" if result.status == 2 else "UNRESOLVED"
        return {"stoichiometric_consistency": status,
                "elemental_balance": "NOT_ASSESSED"}

    def audit_closed_uptake(self):
        """Test boundary production with all sign-defined boundary uptake closed.

        Mixed-sign boundary reactions need an explicit transport model and are
        not classified here. This is a feasibility screen, not thermodynamics.
        """
        bounds = list(zip(self.lower_bounds, self.upper_bounds))
        exports = {}
        for i, reaction in enumerate(self.reactions):
            if reaction.reaction_id not in self.exchange_reaction_ids:
                continue
            values = np.array(list(reaction.stoichiometry.values()))
            if np.all(values >= 0.) and np.any(values > 0.):
                bounds[i] = (bounds[i][0], min(0., bounds[i][1]))
                exports[i] = -1.
            elif np.all(values <= 0.) and np.any(values < 0.):
                bounds[i] = (max(0., bounds[i][0]), bounds[i][1])
                exports[i] = 1.
            else:
                return {"status": "NOT_ASSESSED", "reason": "mixed-sign or zero boundary reaction"}
        if not exports or any(lo > hi for lo, hi in bounds):
            return {"status": "NOT_ASSESSED", "reason": "no boundary reactions or mandatory uptake"}
        maxima = {}
        for i, sign in exports.items():
            objective = np.zeros(len(self.reactions))
            objective[i] = -sign
            result = linprog(objective, A_eq=self.stoichiometric_matrix,
                             b_eq=np.zeros(len(self.internal_metabolite_ids)),
                             bounds=bounds, method="highs")
            if not result.success:
                return {"status": "NOT_ASSESSED" if result.status == 2 else "UNRESOLVED",
                        "reason": result.message}
            maxima[self.reaction_ids[i]] = max(0., float(-result.fun))
        return {"status": "FAIL" if max(maxima.values()) > 1e-7 else "PASS",
                "maximum_exports": maxima}

    # Applies named bound overrides to produce aligned lower and upper flux vectors.
    def bounds(self, overrides: Optional[Mapping[str, Tuple[float, float]]] = None):
        lower, upper = self.lower_bounds.copy(), self.upper_bounds.copy()
        index = {name: i for i, name in enumerate(self.reaction_ids)}
        for name, pair in (overrides or {}).items():
            if name not in index:
                raise ValueError(f"bound override references unknown reaction {name!r}.")
            if len(pair) != 2 or not np.isfinite(pair).all() or pair[0] > pair[1]:
                raise ValueError(f"invalid bound override for {name!r}.")
            lower[index[name]], upper[index[name]] = pair
        return lower, upper
