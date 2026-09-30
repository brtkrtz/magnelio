"""Material-preserving insertion with explicit overlap precedence."""

from __future__ import annotations


def insert(*bodies, priorities, voids=()):
    """Trim overlapping material bodies according to explicit priorities.

    Every retained region keeps the material of its source body. A body with
    a larger priority wins any shared volume. Equal-priority bodies may be
    disjoint, but overlapping equal-priority bodies raise an error. Void
    tools remove volume from every body and are absent from the result.

    Parameters
    ----------
    *bodies : Solid
        Physical bodies, each carrying a material.
    priorities : sequence of int
        One priority per body; larger values win. Priorities govern geometry
        independently of the order in which the result enters a model.
    voids : sequence of Solid, optional
        Material-less construction solids subtracted from every body.

    Returns
    -------
    Group
        Material-preserving, non-overlapping retained bodies. Fully removed
        unnamed bodies are omitted. The Group can be added to a GeometryModel.

    Raises
    ------
    ValueError
        If priorities leave an actual overlap unresolved.
    TopologyEvolutionError
        If trimming deletes or ambiguously splits a named selection.

    Examples
    --------
    Give a dielectric insert priority over its metal housing::

        assembly = insert(housing, dielectric, priorities=(0, 1))
        model.add(assembly)
    """
    from magnelio.geo._occ_backend import check_pairwise_overlaps
    from magnelio.geo._scaling import model_scale
    from magnelio.geo.operations import Group, _InsertRegion
    from magnelio.geo.shape import Solid

    if not bodies:
        raise ValueError("insert requires at least one material body.")
    if len({id(body) for body in bodies}) != len(bodies):
        raise ValueError("insert requires distinct body objects.")
    for index, body in enumerate(bodies):
        if not isinstance(body, Solid):
            raise TypeError(f"insert body {index} must be a Solid.")
        if body.material is None:
            raise ValueError(f"insert body {index} needs a material; use voids for tools.")
    try:
        ranks = tuple(priorities)
    except TypeError as exc:
        raise TypeError("insert priorities must be a sequence of integers.") from exc
    if len(ranks) != len(bodies) or any(type(rank) is not int for rank in ranks):
        raise ValueError("insert requires one integer priority per body.")
    if isinstance(voids, Solid):
        raise TypeError("insert voids must be a sequence of material-less Solids.")
    try:
        voids = tuple(voids)
    except TypeError as exc:
        raise TypeError("insert voids must be a sequence of material-less Solids.") from exc
    for index, void in enumerate(voids):
        if not isinstance(void, Solid):
            raise TypeError(f"insert void {index} must be a Solid.")
        if void.material is not None:
            raise ValueError(f"insert void {index} must not carry a material.")
    operands = (*bodies, *voids)
    overlaps = check_pairwise_overlaps(
        list(operands), tolerance=0.0, scale=model_scale(operands), strict=True
    )
    partners = [set() for _ in operands]
    for left, right, _volume in overlaps:
        partners[left].add(right)
        partners[right].add(left)
        if right < len(bodies) and ranks[left] == ranks[right]:
            raise ValueError(
                f"insert bodies {left} and {right} overlap at equal priority {ranks[left]}."
            )
    retained = []
    for index, body in enumerate(bodies):
        cutters = [
            other
            for other in sorted(partners[index])
            if other >= len(bodies) or ranks[other] > ranks[index]
        ]
        if not cutters:
            retained.append(body)
            continue
        region = _InsertRegion(body, *(operands[other] for other in cutters))
        if region.volume() == 0.0:
            continue
        retained.append(region)
    return Group(*retained)
