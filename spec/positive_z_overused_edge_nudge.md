# Positive-Z Overused Edge Nudge

## Scope

This spec describes the nudge behavior for exact `Z>0` overused-edge topology
cases. It is intentionally paired with
`spec/shared_z0_edge_corner_nudge.md`.

Use the same XY nudge geometry from `shared_z0_edge_corner_nudge.md`:

```text
- same affected-corner sets;
- same midpoint locations;
- same one-corner, two-adjacent, two-opposite, and three-corner footprints;
- same clipped-cell polygon operations;
- same exact comparisons; no tolerance.
```

The important difference is ownership. For `Z=0`, the bad edge is a base-plane
contact in the normal mesh. For `Z>0`, the bad edge is an overused contact in
the difference mesh. Therefore the normal and difference operations are
swapped.

Examples of the positive-Z behavior are in:

```text
spec/positive-z-normal-difference-overused-edge-demo.3mf
```

## Problem

A positive-Z overused edge occurs when a lower-normal surface and a difference
mesh surface both emit the same exact positive-Z edge. The edge is not a
base-plane edge, and it should not create `Z=0` fill. It is an interlock
problem between the normal mesh generated from the lower raster and the
difference mesh generated between the lower and upper rasters.

The affected vertices are exact positive contacts:

```text
upper_or_top.z == lower_or_bottom.z
and
upper_or_top.z > 0
```

Exact equality remains exact Python float equality after the current raster to
mesh coordinate conversion. No tolerance is introduced.

## Role Swap From Z=0

The `Z=0` spec defines two operations:

```text
Z=0 normal operation:
    remove affected contact area from the mesh footprint.

Z=0 difference operation:
    keep full top footprint and split the bottom into:
    corrected normal-lower coverage + complementary Z=0 coverage.
```

For positive-Z overused edges, apply those operations with normal and
difference roles swapped:

```text
Z>0 difference operation:
    use the Z=0 normal operation.
    The difference mesh removes the positive contact area from its footprint.

Z>0 normal operation:
    use the Z=0 difference operation.
    The normal mesh keeps full footprint coverage, split into corrected
    nudged coverage plus complementary positive-Z coverage.
```

## Difference Mesh Mode

For `Z>0` overused edges, the difference mesh is the mesh that must be clipped
away from the positive contact.

Rules:

```text
1. Detect affected corners or edges from exact positive top/bottom equality.
2. Build the same nudge keep footprint defined by
   shared_z0_edge_corner_nudge.md.
3. Replace the difference mesh top and bottom emitted footprint with that keep
   footprint.
4. Add midpoint vertices at the same XY locations as the Z=0 spec.
5. Assign midpoint Z from the positive upper/lower surfaces, not from Z=0.
6. Rebuild exterior walls from the final nudged difference footprint.
7. Do not add base-plane fill for the removed positive-Z contact area.
```

In short:

```text
positive-Z difference-after == Z=0 normal-after footprint behavior,
but with positive interpolated Z values.
```

## Normal Mesh Mode

For `Z>0` overused edges, the lower-normal mesh must keep full footprint
coverage so the lower raster remains the interlocking source of truth.

Rules:

```text
1. Use the same normal keep footprint from shared_z0_edge_corner_nudge.md.
2. Add the same complementary patch or patches from the Z=0 difference spec.
3. The union of keep + complementary patches covers the original footprint.
4. Internal seams between keep and complementary patches are surface seams, not
   exterior walls.
5. Midpoint vertices use the same XY locations as the Z=0 spec.
6. Midpoint Z is interpolated from the positive lower-normal surface.
7. The normal mesh bottom/base remains its ordinary normal-mode base surface.
```

In short:

```text
positive-Z normal-after == Z=0 difference-after footprint behavior,
but the complementary patch is positive lower-normal coverage, not Z=0 fill.
```

## Clipped Cells

Clipped cells use the same role swap.

For the positive-Z difference mesh:

```text
final_difference_footprint =
    existing_clipped_footprint intersected with normal_nudge_keep_footprint
```

For the positive-Z normal mesh:

```text
normal_keep_piece =
    existing_clipped_footprint intersected with normal_nudge_keep_footprint

normal_complement_piece_or_pieces =
    existing_clipped_footprint minus normal_keep_piece
```

If polygon operations produce multiple polygons, triangulate and emit each
polygon independently. Split shared keep/complement boundaries so both sides
use matching vertices. Do not generate exterior walls on internal seams.

## Z Assignment

Unlike the `Z=0` fix, inserted midpoint and intersection vertices are not
forced to `0`.

Use the corrected lower-normal top surface as the source of truth for the
shared positive interface:

```text
difference top vertices:      interpolate from the upper raster surface
difference bottom vertices:   interpolate from the corrected lower-normal top
                              surface
normal top vertices:          interpolate from the corrected lower-normal top
                              surface
normal bottom vertices:       use the normal-mode base surface
```

Here, the "lower raster surface" and "lower-normal top surface" refer to the
same DEM data, but not the same topology source. The difference mesh must follow
the corrected lower-normal top surface that the normal mesh actually emits. It
should not independently reconstruct a different lower surface from the raster.

Only vertices whose source surface is exactly `0` should have `Z=0`.

## Covered Cases

Use the footprint definitions from `shared_z0_edge_corner_nudge.md` for:

```text
- one affected corner;
- two adjacent affected corners;
- two opposite affected corners;
- three affected corners;
- clipped-cell intersections of those cases.
```

The most important positive-Z overused-edge cases are:

```text
- two-adjacent shared positive edge contacts;
- two-opposite positive diagonal contacts;
- three-corner positive contacts that create one overused positive edge.
```

One-corner positive contacts use the same geometry but may not create an
overused edge by themselves. They are still included for completeness and
rotational consistency.

## Non-Goals

This spec does not introduce tolerance-based matching. It does not change the
Z=0 base-plane behavior. It does not define a global post-triangle flip repair;
the fix should be expressed in the same local nudge footprint language used by
`shared_z0_edge_corner_nudge.md` so the normal and difference meshes remain
predictably interlocking.
