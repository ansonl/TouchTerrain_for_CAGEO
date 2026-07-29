# nudge_apply.py
# drive nudge repairs across a whole grid of already-built cells

"""Grid-wide positive-Z repair.

An interlocking pair is generated in two passes: both meshes are built, the
plan is confirmed against what they actually emitted, and only then is the
repair applied back onto the existing cells. This module is that second pass.

``apply_positive_z_plan_to_existing_cells`` walks the confirmed plan and calls
the per-cell operations in ``nudge_cell_ops``. Two grid-wide passes follow it
on the difference mesh, because cutting a contact footprint out of one cell can
leave its neighbours open: ``_add_unmatched_cardinal_surface_walls`` closes
cardinal edges left exposed, and ``_close_local_positive_z_boundary_edge_loops``
caps small boundary loops using directed edge order so the winding stays right.
Those two are safety nets, not primary geometry.

These take the grid as an ordinary argument so the module stays below
``grid_tesselate`` in the import order; ``grid`` keeps thin delegating methods.
"""

import itertools

from collections.abc import Iterable, Sequence
from typing import Any, TYPE_CHECKING

import numpy as np
import shapely

from touchterrain.common import nudge_cell_ops
from touchterrain.common.Cell import cell
from touchterrain.common.Quad import quad
from touchterrain.common.Vertex import vertex
from touchterrain.common.nudge_corner import IntermediateCorner
from touchterrain.common.mesh_vocabulary import (
    CARDINAL_DIRECTIONS,
    Coordinate,
    DirectedEdge3D,
    Edge3D,
    PositiveZNudgePlan,
    PositiveZNudgeRecord,
    SerializedVertexCache,
    SurfaceMesh,
    TopFootprintProvider,
    XYEdge,
    _merge_count_map,
    _parallel_range_results,
    _should_parallelize_rows,
    single_job_parallel_workers,
)
from touchterrain.common.mesh_serialization import (
    _serialized_triangle_collapses,
    _serialized_vertex_from_cache,
    boundary_edge_map_from_meshes,
    directed_edges_are_balanced,
    edge_3d_signature,
    edge_xy_signature,
    normalize_vertex_to_match_mesh_serialization,
    triangle_collapses_after_mesh_serialization,
    surface_mesh_edge_usage,
    surface_polygon_normalized_to_match_mesh_serialization,
)
from touchterrain.common.surface_geometry import (
    _surface_wall_requested_lines,
    make_wall_without_exact_duplicate_vertices,
)
from touchterrain.common.nudge_geometry import (
    _nudge_keep_footprint_splits_side,
    _nudge_split_side_endpoint_edges,
    _nudge_split_side_endpoint_xy,
    edge_cardinal_side,
    side_values_from_bounds,
)
from touchterrain.common.nudge_plan import (
    _positive_z_difference_neighbor_split_sides,
    _positive_z_effective_difference_corners,
    _positive_z_neighbor_split_sides_from_plan,
)

if TYPE_CHECKING:
    from touchterrain.common.grid_tesselate import grid


def apply_positive_z_plan_to_existing_cells(
    target: "grid",
    positive_z_nudge_plan: PositiveZNudgePlan,
    positive_contact_top_raster: np.ndarray | None = None,
    positive_z_difference_top_footprints: (
        TopFootprintProvider | None
    ) = None,
) -> None:
    """Apply a confirmed positive-Z plan to already-created cells."""
    if not positive_z_nudge_plan or target.cells is None:
        return

    output_fileformat = target.tile_info.config.fileformat
    split_rotation = target.tile_info.config.split_rotation
    is_difference_mesh = target.tile.bottom_raster_variants is not None
    normal_neighbor_split_sides: dict[tuple[int, int], set[str]] = {}
    difference_neighbor_split_sides: dict[tuple[int, int], set[str]] = {}
    if not is_difference_mesh:
        normal_neighbor_split_sides = (
            _positive_z_neighbor_split_sides_from_plan(
                positive_z_nudge_plan,
                lambda record: list(record.get("corners", [])),
            )
        )
    else:
        difference_neighbor_split_sides = (
            _positive_z_difference_neighbor_split_sides(
                positive_z_nudge_plan,
            )
        )

    def apply_plan_to_cell(row: int, col: int) -> None:
        current_cell = target.cells[row, col]
        if current_cell is None:
            return

        padded_row = row + 1
        padded_col = col + 1
        W, E, N, S = target.cell_bounds(padded_row, padded_col)
        record = positive_z_nudge_plan.get(
            (padded_row, padded_col),
            {},
        )
        corners = list(record.get("corners", []))
        record_split_sides = set(record.get("split_sides", set()))
        split_sides = set(record_split_sides)
        if not is_difference_mesh:
            split_sides.update(
                normal_neighbor_split_sides.get(
                    (padded_row, padded_col),
                    set(),
                )
            )
        contact_corners = list(record.get("contact_corners", []))
        midpoint_z_by_name = dict(
            record.get("midpoint_z_by_name", {}),
        )
        difference_midpoint_z_by_name = dict(
            record.get(
                "difference_midpoint_z_by_name",
                midpoint_z_by_name,
            ),
        )
        preserve_zero_height_xy: set[tuple[float, float]] | None = None
        preserve_zero_height_edges: set[XYEdge] | None = None
        changed = False

        if not is_difference_mesh:
            if 0 < len(corners) < 4:
                difference_footprint = None
                if positive_z_difference_top_footprints is not None:
                    difference_footprint = (
                        positive_z_difference_top_footprints[row][col]
                    )
                changed = current_cell.apply_positive_z_normal_nudge(
                    corners,
                    W=W,
                    E=E,
                    N=N,
                    S=S,
                    split_rotation=split_rotation,
                    output_fileformat=output_fileformat,
                    top_midpoint_z_by_name=midpoint_z_by_name,
                    difference_footprint=difference_footprint,
                ) or changed
            elif split_sides:
                changed = (
                    current_cell.split_surface_boundary_midpoints(
                        split_sides,
                        contact_corners,
                        W=W,
                        E=E,
                        N=N,
                        S=S,
                        split_rotation=split_rotation,
                        output_fileformat=output_fileformat,
                        include_contact_cut_walls=False,
                        top_midpoint_z_by_name=midpoint_z_by_name,
                        side_cut_wall_sides=None,
                    )
                    or changed
                )
        else:
            propagated_split_sides = (
                difference_neighbor_split_sides.get(
                    (padded_row, padded_col),
                    set(),
                )
            )
            split_sides.update(propagated_split_sides)
            corners = _positive_z_effective_difference_corners(
                record,
            )
            flip_edges = set(record.get("flip_edges", set()))

            if (
                flip_edges
                and current_cell.topSurfacePolygons
                and current_cell.bottomSurfacePolygons
            ):
                changed = (
                    current_cell.flip_bottom_positive_z_contact_edges(
                        split_rotation=split_rotation,
                        output_fileformat=output_fileformat,
                        allowed_edges=flip_edges,
                    )
                    or changed
                )

            if corners:
                side_cut_wall_sides: set[str] = set()
                for current_side, neighbor_location, neighbor_side in [
                    ("N", (padded_row - 1, padded_col), "S"),
                    ("S", (padded_row + 1, padded_col), "N"),
                    ("W", (padded_row, padded_col - 1), "E"),
                    ("E", (padded_row, padded_col + 1), "W"),
                ]:
                    if not _nudge_keep_footprint_splits_side(
                        corners,
                        current_side,
                    ):
                        continue
                    neighbor_record = positive_z_nudge_plan.get(
                        neighbor_location,
                        {},
                    )
                    neighbor_corners = (
                        _positive_z_effective_difference_corners(
                            neighbor_record,
                        )
                    )
                    neighbor_matches = (
                        neighbor_side
                        in neighbor_record.get("split_sides", set())
                    ) or (
                        neighbor_side
                        in difference_neighbor_split_sides.get(
                            neighbor_location,
                            set(),
                        )
                    ) or _nudge_keep_footprint_splits_side(
                        neighbor_corners,
                        neighbor_side,
                    )
                    if not neighbor_matches:
                        side_cut_wall_sides.add(current_side)
                changed = (
                    current_cell.apply_positive_z_difference_nudge(
                        corners,
                        W=W,
                        E=E,
                        N=N,
                        S=S,
                        split_rotation=split_rotation,
                        output_fileformat=output_fileformat,
                        top_midpoint_z_by_name=(
                            difference_midpoint_z_by_name
                        ),
                        bottom_midpoint_z_by_name=(
                            difference_midpoint_z_by_name
                        ),
                        side_cut_wall_sides=side_cut_wall_sides,
                    )
                    or changed
                )
            elif split_sides:
                if record_split_sides:
                    preserve_zero_height_xy = (
                        _nudge_split_side_endpoint_xy(
                            record_split_sides,
                            W,
                            E,
                            N,
                            S,
                            output_fileformat,
                        )
                    )
                    preserve_zero_height_edges = (
                        _nudge_split_side_endpoint_edges(
                            record_split_sides,
                            W,
                            E,
                            N,
                            S,
                            output_fileformat,
                        )
                    )
                changed = (
                    current_cell.split_surface_boundary_midpoints(
                        split_sides,
                        contact_corners,
                        W,
                        E,
                        N,
                        S,
                        split_rotation=split_rotation,
                        output_fileformat=output_fileformat,
                        include_contact_cut_walls=False,
                        top_midpoint_z_by_name=midpoint_z_by_name,
                        side_cut_wall_sides=None,
                    )
                    or changed
                )

        if changed or is_difference_mesh:
            current_cell.remove_zero_height_volumes(
                split_rotation=split_rotation,
                output_fileformat=output_fileformat,
                preserve_zero_height_xy=preserve_zero_height_xy,
                preserve_zero_height_edges=preserve_zero_height_edges,
            )
        if changed:
            current_cell.remove_geometry_collapsed_by_mesh_serialization(
                output_fileformat=output_fileformat,
                split_rotation=split_rotation,
            )

    def apply_plan_to_rows(row_start: int, row_end: int) -> None:
        for row in range(row_start, row_end):
            for col in range(col_count):
                apply_plan_to_cell(row, col)

    row_count = target.cells.shape[0]
    col_count = target.cells.shape[1]
    if not is_difference_mesh:
        target_cells = sorted(
            (padded_row - 1, padded_col - 1)
            for padded_row, padded_col in (
                set(positive_z_nudge_plan)
                | set(normal_neighbor_split_sides)
            )
            if (
                1 <= padded_row <= row_count
                and 1 <= padded_col <= col_count
            )
        )

        def apply_plan_to_target_range(
            target_start: int,
            target_end: int,
        ) -> None:
            for row, col in target_cells[target_start:target_end]:
                apply_plan_to_cell(row, col)

        worker_count = 1
        target_count = len(target_cells)
        if output_fileformat != "obj":
            worker_count = single_job_parallel_workers(
                target.tile_info.config,
                target_count,
            )
        if not _should_parallelize_rows(target_count, worker_count):
            apply_plan_to_target_range(0, target_count)
        else:
            for _result in _parallel_range_results(
                0,
                target_count,
                worker_count,
                apply_plan_to_target_range,
            ):
                pass
    else:
        worker_count = 1
        if output_fileformat != "obj":
            worker_count = single_job_parallel_workers(
                target.tile_info.config,
                row_count,
            )
        if not _should_parallelize_rows(row_count, worker_count):
            apply_plan_to_rows(0, row_count)
        else:
            for _result in _parallel_range_results(
                0,
                row_count,
                worker_count,
                apply_plan_to_rows,
            ):
                pass

    if is_difference_mesh:
        target._add_unmatched_cardinal_surface_walls(
            positive_z_nudge_plan=positive_z_nudge_plan,
            split_rotation=split_rotation,
            output_fileformat=output_fileformat,
        )
        target._close_local_positive_z_boundary_edge_loops(
            positive_z_nudge_plan=positive_z_nudge_plan,
            split_rotation=split_rotation,
            output_fileformat=output_fileformat,
        )
        target._add_unmatched_cardinal_surface_walls(
            positive_z_nudge_plan=positive_z_nudge_plan,
            split_rotation=split_rotation,
            output_fileformat=output_fileformat,
        )


def _add_unmatched_cardinal_surface_walls(
    target: "grid",
    positive_z_nudge_plan: PositiveZNudgePlan,
    split_rotation: int,
    output_fileformat: str,
) -> None:
    """Add missing walls where repaired cardinal surface edges are open."""
    if target.cells is None:
        return
    difference_neighbor_split_sides = (
        _positive_z_difference_neighbor_split_sides(
            positive_z_nudge_plan,
        )
    )

    def add_counts(
        counts: dict[Edge3D, int],
        meshes: Iterable[SurfaceMesh],
        directed_counts: dict[DirectedEdge3D, int] | None = None,
        serialized_vertices: SerializedVertexCache | None = None,
    ) -> None:
        mesh_counts, mesh_directed_counts = surface_mesh_edge_usage(
            meshes,
            split_rotation,
            output_fileformat,
            serialized_vertices,
            include_directed=directed_counts is not None,
        )
        for edge_key, count in mesh_counts.items():
            counts[edge_key] = counts.get(edge_key, 0) + count
        if directed_counts is not None:
            for directed_edge, count in mesh_directed_counts.items():
                directed_counts[directed_edge] = (
                    directed_counts.get(directed_edge, 0) + count
                )

    top_counts: dict[Edge3D, int] = {}
    bottom_counts: dict[Edge3D, int] = {}
    all_counts: dict[Edge3D, int] = {}
    all_directed_counts: dict[DirectedEdge3D, int] = {}

    def edge_counts_for_rows(
        row_start: int,
        row_end: int,
    ) -> tuple[
        dict[Edge3D, int],
        dict[Edge3D, int],
        dict[Edge3D, int],
        dict[DirectedEdge3D, int],
    ]:
        row_top_counts: dict[Edge3D, int] = {}
        row_bottom_counts: dict[Edge3D, int] = {}
        row_all_counts: dict[Edge3D, int] = {}
        row_all_directed_counts: dict[DirectedEdge3D, int] = {}
        for row in range(row_start, row_end):
            for col in range(target.cells.shape[1]):
                current_cell = target.cells[row, col]
                if current_cell is None:
                    continue
                top_meshes = current_cell.top_surface_meshes()
                bottom_meshes = current_cell.bottom_surface_meshes()
                serialized_vertices: SerializedVertexCache = {}
                add_counts(
                    row_top_counts,
                    top_meshes,
                    serialized_vertices=serialized_vertices,
                )
                add_counts(
                    row_bottom_counts,
                    bottom_meshes,
                    serialized_vertices=serialized_vertices,
                )
                add_counts(
                    row_all_counts,
                    itertools.chain(top_meshes, bottom_meshes),
                    row_all_directed_counts,
                    serialized_vertices,
                )
                wall_meshes: list[SurfaceMesh] = []
                wall_meshes.extend(current_cell.borders.values())
                wall_meshes.extend(
                    current_cell.surfacePolygonBorders or [],
                )
                add_counts(
                    row_all_counts,
                    wall_meshes,
                    row_all_directed_counts,
                    serialized_vertices,
                )
        return (
            row_top_counts,
            row_bottom_counts,
            row_all_counts,
            row_all_directed_counts,
        )

    def merge_row_counts(
        row_counts: tuple[
            dict[Edge3D, int],
            dict[Edge3D, int],
            dict[Edge3D, int],
            dict[DirectedEdge3D, int],
        ],
    ) -> None:
        (
            row_top_counts,
            row_bottom_counts,
            row_all_counts,
            row_all_directed_counts,
        ) = row_counts
        _merge_count_map(top_counts, row_top_counts)
        _merge_count_map(bottom_counts, row_bottom_counts)
        _merge_count_map(all_counts, row_all_counts)
        _merge_count_map(all_directed_counts, row_all_directed_counts)

    worker_count = single_job_parallel_workers(
        target.tile_info.config,
        target.cells.shape[0],
    )
    if not _should_parallelize_rows(target.cells.shape[0], worker_count):
        merge_row_counts(edge_counts_for_rows(0, target.cells.shape[0]))
    else:
        for row_counts in _parallel_range_results(
            0,
            target.cells.shape[0],
            worker_count,
            edge_counts_for_rows,
        ):
            merge_row_counts(row_counts)

    for row in range(target.cells.shape[0]):
        for col in range(target.cells.shape[1]):
            current_cell = target.cells[row, col]
            if current_cell is None:
                continue
            record = positive_z_nudge_plan.get((row + 1, col + 1), {})
            effective_corners = _positive_z_effective_difference_corners(
                record,
            )
            derived_split_sides = difference_neighbor_split_sides.get(
                (row + 1, col + 1),
                set(),
            )
            if (
                not effective_corners
                and not record.get("split_sides")
                and not derived_split_sides
            ):
                continue

            top_meshes = current_cell.top_surface_meshes()
            bottom_meshes = current_cell.bottom_surface_meshes()
            if not top_meshes or not bottom_meshes:
                continue

            serialized_vertices: SerializedVertexCache = {}
            top_boundary_edges = boundary_edge_map_from_meshes(
                top_meshes,
                output_fileformat=output_fileformat,
                serialized_vertices=serialized_vertices,
            )
            bottom_boundary_edges = boundary_edge_map_from_meshes(
                bottom_meshes,
                output_fileformat=output_fileformat,
                serialized_vertices=serialized_vertices,
            )
            if not top_boundary_edges or not bottom_boundary_edges:
                continue

            W, E, N, S = target.cell_bounds(row + 1, col + 1)
            side_values = side_values_from_bounds(
                W,
                E,
                N,
                S,
                output_fileformat,
            )

            existing_lines = _surface_wall_requested_lines(
                current_cell.surfacePolygonBorders,
                output_fileformat=output_fileformat,
            )
            new_borders: list[quad] = []

            def viable_wall_counts(
                wall_candidate: quad | None,
            ) -> tuple[
                dict[Edge3D, int],
                dict[DirectedEdge3D, int],
            ] | None:
                if wall_candidate is None:
                    return None
                candidate_counts, candidate_directed_counts = (
                    surface_mesh_edge_usage(
                        [wall_candidate],
                        split_rotation,
                        output_fileformat,
                        serialized_vertices,
                    )
                )
                touched_edges = set(candidate_counts)
                final_counts: dict[Edge3D, int] = {}
                final_directed_counts: dict[DirectedEdge3D, int] = {}
                for edge_key in touched_edges:
                    count = candidate_counts[edge_key]
                    final_count = all_counts.get(edge_key, 0) + count
                    if final_count != 2:
                        return None
                    final_counts[edge_key] = final_count
                    for directed_edge in (
                        (edge_key[0], edge_key[1]),
                        (edge_key[1], edge_key[0]),
                    ):
                        final_directed_counts[directed_edge] = (
                            all_directed_counts.get(directed_edge, 0)
                            + candidate_directed_counts.get(
                                directed_edge,
                                0,
                            )
                        )
                if not directed_edges_are_balanced(
                    final_counts,
                    final_directed_counts,
                    touched_edges,
                ):
                    return None
                return candidate_counts, candidate_directed_counts

            def choose_wall(
                candidates: Sequence[quad | None],
            ) -> tuple[
                quad,
                dict[Edge3D, int],
                dict[DirectedEdge3D, int],
            ] | None:
                best: tuple[
                    int,
                    int,
                    quad,
                    dict[Edge3D, int],
                    dict[DirectedEdge3D, int],
                ] | None = None
                for candidate in candidates:
                    candidate_usage = viable_wall_counts(candidate)
                    if candidate_usage is None or candidate is None:
                        continue
                    candidate_counts, candidate_directed_counts = (
                        candidate_usage
                    )
                    closed_edges = sum(
                        1
                        for edge_key in candidate_counts
                        if all_counts.get(edge_key, 0) == 1
                    )
                    score = (closed_edges, -len(candidate_counts))
                    if best is None or score > (best[0], best[1]):
                        best = (
                            score[0],
                            score[1],
                            candidate,
                            candidate_counts,
                            candidate_directed_counts,
                        )
                if best is None:
                    return None
                return best[2], best[3], best[4]

            for footprint, top_edge in top_boundary_edges.items():
                bottom_edge = bottom_boundary_edges.get(footprint)
                if bottom_edge is None:
                    continue
                side = edge_cardinal_side(footprint, side_values)
                if side is None:
                    continue
                if current_cell.borders.get(side):
                    continue
                if (
                    existing_lines is not None
                    and existing_lines.covers(shapely.LineString(footprint))
                ):
                    continue
                top_edge_key = edge_3d_signature(top_edge[0], top_edge[1])
                bottom_edge_key = edge_3d_signature(
                    bottom_edge[0],
                    bottom_edge[1],
                )
                if (
                    top_counts.get(top_edge_key, 0) > 1
                    and bottom_counts.get(bottom_edge_key, 0) > 1
                ):
                    continue
                if (
                    all_counts.get(top_edge_key, 0) != 1
                    and all_counts.get(bottom_edge_key, 0) != 1
                ):
                    continue

                preferred_wall: quad | None = None
                top_all_count = all_counts.get(top_edge_key, 0)
                bottom_all_count = all_counts.get(bottom_edge_key, 0)
                vertical_pairs: list[tuple[int, Coordinate]] = []
                for top_coord in top_edge:
                    for bottom_coord in bottom_edge:
                        if top_coord[:2] != bottom_coord[:2]:
                            continue
                        if top_coord == bottom_coord:
                            vertical_pairs.append((999, top_coord))
                            continue
                        vertical_key = edge_3d_signature(
                            top_coord,
                            bottom_coord,
                        )
                        vertical_pairs.append(
                            (all_counts.get(vertical_key, 0), top_coord)
                        )
                if (
                    len(vertical_pairs) == 2
                    and top_all_count > 1
                    and bottom_all_count <= 1
                ):
                    vertical_pairs.sort(key=lambda item: item[0])
                    preferred_wall = quad(
                        vertex(*vertical_pairs[0][1]),
                        vertex(*bottom_edge[0]),
                        vertex(*bottom_edge[1]),
                        None,
                    )
                elif len(vertical_pairs) == 2:
                    vertical_pairs.sort(key=lambda item: item[0])
                    if (
                        vertical_pairs[0][0] < vertical_pairs[1][0]
                        and vertical_pairs[1][0] > 0
                    ):
                        preferred_wall = quad(
                            vertex(*vertical_pairs[0][1]),
                            vertex(*bottom_edge[0]),
                            vertex(*bottom_edge[1]),
                            None,
                        )

                candidates: list[quad | None] = [preferred_wall]
                for top_order in (
                    (top_edge[0], top_edge[1]),
                    (top_edge[1], top_edge[0]),
                ):
                    for bottom_order in (
                        (bottom_edge[0], bottom_edge[1]),
                        (bottom_edge[1], bottom_edge[0]),
                    ):
                        candidates.append(
                            make_wall_without_exact_duplicate_vertices(
                                vertex(*top_order[0]),
                                vertex(*top_order[1]),
                                vertex(*bottom_order[0]),
                                vertex(*bottom_order[1]),
                                output_fileformat=output_fileformat,
                            )
                        )
                unique_coords = []
                for candidate_coord in (
                    top_edge[0],
                    top_edge[1],
                    bottom_edge[0],
                    bottom_edge[1],
                ):
                    if candidate_coord not in unique_coords:
                        unique_coords.append(candidate_coord)
                if len(unique_coords) == 4:
                    for order in itertools.permutations(unique_coords):
                        candidates.append(
                            make_wall_without_exact_duplicate_vertices(
                                vertex(*order[0]),
                                vertex(*order[1]),
                                vertex(*order[2]),
                                vertex(*order[3]),
                                output_fileformat=output_fileformat,
                            )
                        )

                if not effective_corners:
                    simple_bottom_edge = bottom_edge
                    top_start_xy = (top_edge[0][0], top_edge[0][1])
                    bottom_start_xy = (
                        bottom_edge[0][0],
                        bottom_edge[0][1],
                    )
                    if top_start_xy == bottom_start_xy:
                        simple_bottom_edge = (
                            bottom_edge[1],
                            bottom_edge[0],
                        )
                    simple_wall = preferred_wall
                    if simple_wall is None:
                        simple_wall = (
                            make_wall_without_exact_duplicate_vertices(
                                vertex(*top_edge[1]),
                                vertex(*top_edge[0]),
                                vertex(*simple_bottom_edge[1]),
                                vertex(*simple_bottom_edge[0]),
                                output_fileformat=output_fileformat,
                            )
                        )
                    if simple_wall is not None:
                        candidates.append(simple_wall)

                selected_wall = choose_wall(candidates)

                if selected_wall is not None:
                    wall, wall_counts, wall_directed_counts = selected_wall
                    new_borders.append(wall)
                    for edge_key, count in wall_counts.items():
                        all_counts[edge_key] = (
                            all_counts.get(edge_key, 0) + count
                        )
                    for directed_edge, count in (
                        wall_directed_counts.items()
                    ):
                        all_directed_counts[directed_edge] = (
                            all_directed_counts.get(directed_edge, 0)
                            + count
                        )

            if new_borders:
                current_cell.surfacePolygonBorders = (
                    (current_cell.surfacePolygonBorders or [])
                    + new_borders
                )


def _close_local_positive_z_boundary_edge_loops(
    target: "grid",
    positive_z_nudge_plan: PositiveZNudgePlan,
    split_rotation: int,
    output_fileformat: str,
) -> None:
    """Close small positive-Z loops using local directed boundary order."""
    if target.cells is None:
        return

    target_locations = (
        (row + 1, col + 1)
        for row in range(target.cells.shape[0])
        for col in range(target.cells.shape[1])
    )

    def edge_usage_for_rows(
        row_start: int,
        row_end: int,
    ) -> tuple[dict[Edge3D, int], dict[DirectedEdge3D, int]]:
        row_edge_counts: dict[Edge3D, int] = {}
        row_directed_counts: dict[DirectedEdge3D, int] = {}
        for row in range(row_start, row_end):
            for col in range(target.cells.shape[1]):
                current_cell = target.cells[row, col]
                if current_cell is None:
                    continue
                serialized_vertices: SerializedVertexCache = {}
                edge_counts, directed_counts = surface_mesh_edge_usage(
                    current_cell.iter_meshes_for_model(),
                    split_rotation,
                    output_fileformat,
                    serialized_vertices,
                )
                _merge_count_map(row_edge_counts, edge_counts)
                _merge_count_map(row_directed_counts, directed_counts)
        return row_edge_counts, row_directed_counts

    def merge_edge_usage(
        row_usage: tuple[dict[Edge3D, int], dict[DirectedEdge3D, int]],
    ) -> None:
        row_edge_counts, row_directed_counts = row_usage
        _merge_count_map(global_edge_counts, row_edge_counts)
        _merge_count_map(global_directed_counts, row_directed_counts)

    global_edge_counts: dict[Edge3D, int] = {}
    global_directed_counts: dict[DirectedEdge3D, int] = {}
    tile_info = getattr(target, "tile_info", None)
    worker_count = single_job_parallel_workers(
        getattr(tile_info, "config", None),
        target.cells.shape[0],
    )
    if not _should_parallelize_rows(target.cells.shape[0], worker_count):
        merge_edge_usage(edge_usage_for_rows(0, target.cells.shape[0]))
    else:
        for row_usage in _parallel_range_results(
            0,
            target.cells.shape[0],
            worker_count,
            edge_usage_for_rows,
        ):
            merge_edge_usage(row_usage)

    def ordered_cycle_vertices(
        component: set[Edge3D],
        directed_boundary_edges: dict[Edge3D, DirectedEdge3D],
    ) -> list[Coordinate] | None:
        vertices = {coord for edge in component for coord in edge}
        outgoing: dict[Coordinate, list[Coordinate]] = {
            coord: [] for coord in vertices
        }
        incoming: dict[Coordinate, list[Coordinate]] = {
            coord: [] for coord in vertices
        }
        for edge in component:
            directed_edge = directed_boundary_edges.get(edge)
            if directed_edge is None:
                return None
            outgoing[directed_edge[0]].append(directed_edge[1])
            incoming[directed_edge[1]].append(directed_edge[0])
        if any(len(outgoing[coord]) != 1 for coord in vertices):
            return None
        if any(len(incoming[coord]) != 1 for coord in vertices):
            return None

        start = min(vertices)
        ordered = [start]
        current = start
        while True:
            next_vertex = outgoing[current][0]
            if next_vertex == start:
                break
            if next_vertex in ordered:
                return None

            ordered.append(next_vertex)
            current = next_vertex

        if len(ordered) != len(vertices):
            return None
        return ordered

    def candidate_is_valid(
        candidates: list[quad],
        edge_counts: dict[Edge3D, int],
        directed_edge_counts: dict[DirectedEdge3D, int],
        serialized_vertices: SerializedVertexCache,
    ) -> tuple[dict[Edge3D, int], dict[DirectedEdge3D, int]] | None:
        for candidate in candidates:
            for triangle in candidate.get_triangles(split_rotation):
                if triangle_collapses_after_mesh_serialization(
                    [mesh_vertex.coords for mesh_vertex in triangle],
                    output_fileformat,
                    serialized_vertices=serialized_vertices,
                ):
                    return None

        candidate_counts, candidate_directed_counts = (
            surface_mesh_edge_usage(
                candidates,
                split_rotation,
                output_fileformat,
                serialized_vertices,
            )
        )
        touched_edges = set(candidate_counts)
        combined_counts = {
            edge_key: edge_counts.get(edge_key, 0)
            + candidate_counts.get(edge_key, 0)
            for edge_key in touched_edges
        }
        if any(count != 2 for count in combined_counts.values()):
            return None

        combined_directed: dict[DirectedEdge3D, int] = {}
        for edge_key in touched_edges:
            for directed_edge in (
                (edge_key[0], edge_key[1]),
                (edge_key[1], edge_key[0]),
            ):
                combined_directed[directed_edge] = (
                    directed_edge_counts.get(directed_edge, 0)
                    + candidate_directed_counts.get(directed_edge, 0)
                )
        if not directed_edges_are_balanced(
            combined_counts,
            combined_directed,
            touched_edges,
        ):
            return None
        return candidate_counts, candidate_directed_counts

    def cap_candidates_for_order(
        order: Sequence[Coordinate],
    ) -> list[list[quad]]:
        order_len = len(order)
        candidates: list[list[quad]] = []
        if order_len == 3:
            candidates.append(
                [
                    quad(
                        vertex(*order[1]),
                        vertex(*order[0]),
                        vertex(*order[2]),
                        None,
                    )
                ],
            )
        if order_len == 4:
            candidates.extend(
                [
                    [
                        quad(
                            vertex(*order[1]),
                            vertex(*order[0]),
                            vertex(*order[2]),
                            None,
                        ),
                        quad(
                            vertex(*order[3]),
                            vertex(*order[2]),
                            vertex(*order[0]),
                            None,
                        ),
                    ],
                    [
                        quad(
                            vertex(*order[2]),
                            vertex(*order[1]),
                            vertex(*order[3]),
                            None,
                        ),
                        quad(
                            vertex(*order[0]),
                            vertex(*order[3]),
                            vertex(*order[1]),
                            None,
                        ),
                    ],
                ],
            )

        center = tuple(
            sum(coord[axis] for coord in order) / order_len
            for axis in range(3)
        )
        candidates.append(
            [
                quad(
                    vertex(*order[(index + 1) % order_len]),
                    vertex(*order[index]),
                    vertex(*center),
                    None,
                )
                for index in range(order_len)
            ],
        )
        return candidates

    for padded_row, padded_col in target_locations:
        row = padded_row - 1
        col = padded_col - 1
        if (
            row < 0
            or col < 0
            or row >= target.cells.shape[0]
            or col >= target.cells.shape[1]
        ):
            continue

        current_cell = target.cells[row, col]
        if current_cell is None:
            continue

        serialized_vertices: SerializedVertexCache = {}
        edge_counts, directed_edge_counts = surface_mesh_edge_usage(
            current_cell.iter_meshes_for_model(),
            split_rotation,
            output_fileformat,
            serialized_vertices,
        )
        boundary_edges = {
            edge_key
            for edge_key, count in edge_counts.items()
            if (
                count == 1
                and global_edge_counts.get(edge_key, 0) == 1
                and edge_key[0][2] > 0
                and edge_key[1][2] > 0
            )
        }
        if not boundary_edges:
            continue

        directed_boundary_edges: dict[Edge3D, DirectedEdge3D] = {}
        for directed_edge, count in directed_edge_counts.items():
            edge_key = edge_3d_signature(
                directed_edge[0],
                directed_edge[1],
            )
            if edge_key in boundary_edges and count == 1:
                directed_boundary_edges[edge_key] = directed_edge

        edge_by_vertex: dict[Coordinate, list[Edge3D]] = {}
        for edge_key in boundary_edges:
            edge_by_vertex.setdefault(edge_key[0], []).append(edge_key)
            edge_by_vertex.setdefault(edge_key[1], []).append(edge_key)

        selected_caps: list[quad] = []
        visited: set[Edge3D] = set()
        for start_edge in boundary_edges:
            if start_edge in visited:
                continue
            stack = [start_edge]
            component: set[Edge3D] = set()
            while stack:
                edge_key = stack.pop()
                if edge_key in visited:
                    continue
                visited.add(edge_key)
                component.add(edge_key)
                for coord in edge_key:
                    stack.extend(edge_by_vertex.get(coord, []))

            vertices = {coord for edge_key in component for coord in edge_key}
            if len(component) != len(vertices):
                continue
            if len(component) < 3 or len(component) > 8:
                continue
            if any(coord[2] <= 0 for coord in vertices):
                continue

            ordered_vertices = ordered_cycle_vertices(
                component,
                directed_boundary_edges,
            )
            if ordered_vertices is None:
                continue

            selected_candidate: list[quad] | None = None
            selected_counts: dict[Edge3D, int] | None = None
            selected_directed_counts: dict[DirectedEdge3D, int] | None = None
            for candidate in cap_candidates_for_order(ordered_vertices):
                candidate_counts = candidate_is_valid(
                    candidate,
                    edge_counts,
                    directed_edge_counts,
                    serialized_vertices,
                )
                if candidate_counts is None:
                    continue
                selected_candidate = candidate
                selected_counts, selected_directed_counts = candidate_counts
                break

            if (
                selected_candidate is None
                or selected_counts is None
                or selected_directed_counts is None
            ):
                continue

            selected_caps.extend(selected_candidate)
            for edge_key, count in selected_counts.items():
                edge_counts[edge_key] = edge_counts.get(edge_key, 0) + count
                global_edge_counts[edge_key] = (
                    global_edge_counts.get(edge_key, 0) + count
                )
            for directed_edge, count in selected_directed_counts.items():
                directed_edge_counts[directed_edge] = (
                    directed_edge_counts.get(directed_edge, 0) + count
                )
                global_directed_counts[directed_edge] = (
                    global_directed_counts.get(directed_edge, 0) + count
                )

        if selected_caps:
            current_cell.surfacePolygonBorders = (
                (current_cell.surfacePolygonBorders or []) + selected_caps
            )

    global_boundary_edges = {
        edge_key
        for edge_key, count in global_edge_counts.items()
        if count == 1 and edge_key[0][2] > 0 and edge_key[1][2] > 0
    }
    if not global_boundary_edges:
        return

    edge_owner_cell: dict[Edge3D, cell] = {}
    for current_cell in target.cells.flat:
        if current_cell is None:
            continue
        serialized_vertices: SerializedVertexCache = {}
        local_counts, _unused_directed_counts = surface_mesh_edge_usage(
            current_cell.iter_meshes_for_model(),
            split_rotation,
            output_fileformat,
            serialized_vertices,
            include_directed=False,
        )
        for edge_key, count in local_counts.items():
            if (
                count == 1
                and edge_key in global_boundary_edges
            ):
                edge_owner_cell.setdefault(edge_key, current_cell)

    directed_boundary_edges: dict[Edge3D, DirectedEdge3D] = {}
    outgoing_edges: dict[
        Coordinate,
        list[tuple[Coordinate, Edge3D]],
    ] = {}
    for directed_edge, count in global_directed_counts.items():
        edge_key = edge_3d_signature(directed_edge[0], directed_edge[1])
        if edge_key not in global_boundary_edges or count != 1:
            continue
        directed_boundary_edges[edge_key] = directed_edge
        outgoing_edges.setdefault(directed_edge[0], []).append(
            (directed_edge[1], edge_key),
        )

    def find_small_directed_cycle(
        start_edge: Edge3D,
        unused_edges: set[Edge3D],
    ) -> tuple[list[Coordinate], list[Edge3D]] | None:
        start_directed_edge = directed_boundary_edges.get(start_edge)
        if start_directed_edge is None:
            return None
        start_vertex, next_vertex = start_directed_edge

        def walk(
            current_vertex: Coordinate,
            order: list[Coordinate],
            cycle_edges: list[Edge3D],
        ) -> tuple[list[Coordinate], list[Edge3D]] | None:
            if len(cycle_edges) > 8:
                return None
            next_edges = sorted(
                outgoing_edges.get(current_vertex, []),
                key=lambda item: (item[0], item[1]),
            )
            for candidate_vertex, candidate_edge in next_edges:
                if candidate_edge not in unused_edges:
                    continue
                if candidate_edge in cycle_edges:
                    continue
                if candidate_vertex == start_vertex:
                    if len(order) < 3:
                        continue
                    return order, [*cycle_edges, candidate_edge]
                if candidate_vertex in order:
                    continue
                result = walk(
                    candidate_vertex,
                    [*order, candidate_vertex],
                    [*cycle_edges, candidate_edge],
                )
                if result is not None:
                    return result
            return None

        return walk(next_vertex, [start_vertex, next_vertex], [start_edge])

    def split_collinear_triangle_edge(
        target_cell: cell,
        long_edge: Edge3D,
        split_coord: Coordinate,
    ) -> bool:
        serialized_vertices: SerializedVertexCache = {}

        def split_polygon_list(
            polygons: list[shapely.Polygon] | None,
        ) -> tuple[list[shapely.Polygon] | None, bool]:
            if not polygons:
                return polygons, False
            changed = False
            output: list[shapely.Polygon] = []
            for polygon in polygons:
                coords = [
                    _serialized_vertex_from_cache(
                        coord,
                        output_fileformat,
                        serialized_vertices,
                    )
                    for coord in polygon.exterior.coords[:-1]
                ]
                if len(coords) != 3:
                    output.append(polygon)
                    continue
                split_done = False
                for index, coord0 in enumerate(coords):
                    coord1 = coords[(index + 1) % 3]
                    if edge_3d_signature(coord0, coord1) != long_edge:
                        continue
                    coord2 = coords[(index + 2) % 3]
                    output.extend(
                        [
                            shapely.Polygon(
                                [coord0, split_coord, coord2, coord0],
                            ),
                            shapely.Polygon(
                                [split_coord, coord1, coord2, split_coord],
                            ),
                        ]
                    )
                    changed = True
                    split_done = True
                    break
                if not split_done:
                    output.append(polygon)
            return output, changed

        def split_quad_list(
            meshes: list[quad] | None,
        ) -> tuple[list[quad] | None, bool]:
            if not meshes:
                return meshes, False
            changed = False
            output: list[quad] = []
            for mesh in meshes:
                vertices = [v for v in mesh.vl if v is not None]
                if len(vertices) != 3:
                    output.append(mesh)
                    continue
                coords = [
                    _serialized_vertex_from_cache(
                        mesh_vertex.coords,
                        output_fileformat,
                        serialized_vertices,
                    )
                    for mesh_vertex in vertices
                ]
                split_done = False
                for index, coord0 in enumerate(coords):
                    coord1 = coords[(index + 1) % 3]
                    if edge_3d_signature(coord0, coord1) != long_edge:
                        continue
                    coord2 = coords[(index + 2) % 3]
                    output.extend(
                        [
                            quad(
                                vertex(*coord0),
                                vertex(*split_coord),
                                vertex(*coord2),
                                None,
                            ),
                            quad(
                                vertex(*split_coord),
                                vertex(*coord1),
                                vertex(*coord2),
                                None,
                            ),
                        ]
                    )
                    changed = True
                    split_done = True
                    break
                if not split_done:
                    output.append(mesh)
            return output, changed

        top_polygons, changed = split_polygon_list(
            target_cell.topSurfacePolygons,
        )
        if changed:
            target_cell.topSurfacePolygons = top_polygons
            return True

        bottom_polygons, changed = split_polygon_list(
            target_cell.bottomSurfacePolygons,
        )
        if changed:
            target_cell.bottomSurfacePolygons = bottom_polygons
            return True

        surface_borders, changed = split_quad_list(
            target_cell.surfacePolygonBorders,
        )
        if changed:
            target_cell.surfacePolygonBorders = surface_borders
            return True
        return False

    def split_collinear_three_edge_cycle(
        cycle_edges: list[Edge3D],
        ordered_vertices: list[Coordinate],
    ) -> bool:
        if len(cycle_edges) != 3 or len(ordered_vertices) != 3:
            return False
        if not _serialized_triangle_collapses(ordered_vertices):
            return False
        long_edge = max(
            cycle_edges,
            key=lambda edge_key: sum(
                (edge_key[0][axis] - edge_key[1][axis]) ** 2
                for axis in range(3)
            ),
        )
        split_vertices = [
            coord
            for coord in ordered_vertices
            if coord != long_edge[0] and coord != long_edge[1]
        ]
        if len(split_vertices) != 1:
            return False
        target_cell = edge_owner_cell.get(long_edge)
        if target_cell is None:
            return False
        return split_collinear_triangle_edge(
            target_cell,
            long_edge,
            split_vertices[0],
        )

    unused_edges = set(global_boundary_edges)
    while unused_edges:
        start_edge = min(unused_edges)
        cycle = find_small_directed_cycle(start_edge, unused_edges)
        if cycle is None:
            unused_edges.remove(start_edge)
            continue

        ordered_vertices, cycle_edges = cycle
        cycle_edge_set = set(cycle_edges)
        if len(cycle_edge_set) != len(cycle_edges):
            unused_edges.difference_update(cycle_edge_set)
            continue
        if len(cycle_edges) < 3 or len(cycle_edges) > 8:
            unused_edges.difference_update(cycle_edge_set)
            continue
        if any(coord[2] <= 0 for coord in ordered_vertices):
            unused_edges.difference_update(cycle_edge_set)
            continue

        selected_candidate: list[quad] | None = None
        selected_counts: dict[Edge3D, int] | None = None
        selected_directed_counts: dict[DirectedEdge3D, int] | None = None
        serialized_vertices: SerializedVertexCache = {}
        for candidate in cap_candidates_for_order(ordered_vertices):
            candidate_counts = candidate_is_valid(
                candidate,
                global_edge_counts,
                global_directed_counts,
                serialized_vertices,
            )
            if candidate_counts is None:
                continue
            selected_candidate = candidate
            selected_counts, selected_directed_counts = candidate_counts
            break

        if (
            selected_candidate is None
            or selected_counts is None
            or selected_directed_counts is None
        ):
            if split_collinear_three_edge_cycle(
                cycle_edges,
                ordered_vertices,
            ):
                unused_edges.difference_update(cycle_edge_set)
                continue
            unused_edges.difference_update(cycle_edge_set)
            continue

        target_cell = next(
            (
                edge_owner_cell[edge_key]
                for edge_key in cycle_edges
                if edge_key in edge_owner_cell
            ),
            None,
        )
        if target_cell is None:
            unused_edges.difference_update(cycle_edge_set)
            continue

        target_cell.surfacePolygonBorders = (
            (target_cell.surfacePolygonBorders or []) + selected_candidate
        )
        for edge_key, count in selected_counts.items():
            global_edge_counts[edge_key] = (
                global_edge_counts.get(edge_key, 0) + count
            )
        for directed_edge, count in selected_directed_counts.items():
            global_directed_counts[directed_edge] = (
                global_directed_counts.get(directed_edge, 0) + count
            )
        unused_edges.difference_update(cycle_edge_set)
