# grid_tesselate.py
# create triangles from a top and bottom np 2D array, including walls

'''
@author:     Chris Harding
@license:    GPL
@contact:    charding@iastate.edu

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
# CH: May  2023: modified refactored optimized version (lower memory foot print) by keerl
# CH: Apr. 2019: converted to Python 3
# CH: Feb. 2018: added use of tempfile as file buffer to lower memory footprint
# CH: Feb. 2017: added calculations for normals in stl files
# CH: Jan. 22, 16: putting the vert index behind a comment makes some programs crash
#                  when loading the obj file, so I removed those.
# FIX: (CH, Nov.16,15): make the vertex index a per grid attribute rather than
#  a vertex class attribute as this seem to index not found fail eventually when
#  multiple grids are processed together.
# CH July 2015

import io
import itertools
import multiprocessing
import os
import shutil
import struct # for making binary STL
import sys

# get root logger, will later be redirected into a logfile
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from collections.abc import Iterable, Iterator, Sequence
from typing import Union, Any, Callable

import numpy as np
import shapely

from touchterrain.common.Vertex import vertex
from touchterrain.common.Quad import quad


from touchterrain.common.tile_info import TouchTerrainTileInfo

from touchterrain.common.RasterVariants import RasterVariants
from touchterrain.common.BorderEdge import BorderEdge
from touchterrain.common.nudge_corner import (
    IntermediateCorner,
    z0_nudge_corners_from_source_raster,
)

from touchterrain.common.shapely_utils import flatten_geometries
from touchterrain.common.shapely_polygon_utils import (
    polygon_to_list_of_vertex,
    polygons_equal_3d,
)
from touchterrain.common.interpolate_Z import interpolate_z_planar

from touchterrain.common.mesh_vocabulary import (
    BottomSurfaceProvider,
    CARDINAL_DIRECTIONS,
    CELL_NEIGHBOR_SIDES,
    CardinalWallMap,
    Coordinate,
    DirectedEdge3D,
    Edge3D,
    EmittedBottomSurface,
    MESH_OUTPUT_SERIALIZATION_DECIMAL_PRECISION,
    NUDGE_MIDPOINT_CORNERS_BY_NAME,
    NUDGE_SIDE_MIDPOINT_NAME,
    POSITIVE_Z_OPPOSITE_CORNER_PAIRS,
    POSITIVE_Z_SE_NW_DIAGONAL,
    POSITIVE_Z_SIDE_CONTACT_CHECKS,
    POSITIVE_Z_SW_NE_DIAGONAL,
    PositiveZNudgePlan,
    PositiveZNudgeRecord,
    PositiveZSurfaceValues,
    SerializedVertexCache,
    SurfaceMesh,
    TopFootprintProvider,
    TopFootprintSource,
    XYEdge,
    _empty_borders,
    _empty_side_edge_sets,
    _merge_count_map,
    _parallel_range_results,
    _should_parallelize_rows,
    single_job_parallel_workers,
)
from touchterrain.common.mesh_serialization import (
    _boundary_line_map_by_serialized_xy,
    _line_with_serialized_xy,
    _serialized_triangle_collapses,
    _serialized_vertex_from_cache,
    boundary_edge_map_from_meshes,
    directed_edges_are_balanced,
    edge_3d_signature,
    edge_xy_signature,
    normalize_coordinate_to_match_mesh_serialization,
    normalize_vertex_to_match_mesh_serialization,
    polygon_normalized_to_match_mesh_serialization,
    quad_normalized_to_match_mesh_serialization,
    surface_mesh_edge_counts,
    surface_mesh_edge_usage,
    surface_polygon_normalized_to_match_mesh_serialization,
    triangle_collapses_after_mesh_serialization,
)
from touchterrain.common.nudge_geometry import (
    _nudge_adjusted_surface_planes,
    _nudge_keep_footprint_split_sides,
    _nudge_keep_footprint_splits_side,
    _nudge_keep_vertex_names,
    _nudge_midpoint_z_by_xy,
    _nudge_split_side_endpoint_edges,
    _nudge_split_side_endpoint_xy,
    _surface_polygons_with_midpoint_z,
    _surface_polygons_with_z_overrides,
    _surface_vertex_z_overrides_by_xy,
    _z0_adjusted_keep_surface_planes,
    cell_bounds_for_location,
    cell_corner_points,
    cell_side_values,
    edge_cardinal_side,
    full_cell_footprint,
    nudge_keep_footprint,
    quad_corner_vertices_by_xy,
    rebuild_nudged_surface_polygon_borders,
    side_values_from_bounds,
)
from touchterrain.common.surface_geometry import (
    _build_cardinal_wall_borders,
    _clip_3d_surface_polygons_to_2d_geometry,
    _clipped_cell_surface_polygons,
    _create_cell_bottom_geometry,
    _current_surface_footprint,
    _geometry_boundary_linework,
    _linework_covers_footprint,
    _polygonized_regions_with_shared_boundaries,
    _rebuild_matching_surface_polygon_borders,
    _split_surface_boundary_edges_for_wall_matches,
    _surface_planes_from_current_geometry,
    _surface_wall_requested_lines,
    _triangulate_2d_geometry_to_3d_polygons,
    _union_polygon_footprint,
    get_normal,
    make_wall_without_exact_duplicate_vertices,
)
from touchterrain.common.raster_interpolation import (
    _cell_corner_elevations,
    _interpolated_corner_grid,
    _zero_elevations_below_threshold,
    interpolate_with_NaN,
)


BINARY_STL_FACET = struct.Struct("<12fH")
BINARY_STL_HEADER = struct.Struct("80sI")
ASCII_STL_FACET_TEMPLATE = (
    "facet normal "
    "{face[0]:.{precision}f} "
    "{face[1]:.{precision}f} "
    "{face[2]:.{precision}f}\n"
    "outer loop\n"
    "vertex "
    "{face[3]:.{precision}f} "
    "{face[4]:.{precision}f} "
    "{face[5]:.{precision}f}\n"
    "vertex "
    "{face[6]:.{precision}f} "
    "{face[7]:.{precision}f} "
    "{face[8]:.{precision}f}\n"
    "vertex "
    "{face[9]:.{precision}f} "
    "{face[10]:.{precision}f} "
    "{face[11]:.{precision}f}\n"
    "endloop\n"
    "endfacet\n"
)


def _cleanup_cells_for_mesh_serialization(
    cells: np.ndarray,
    output_fileformat: str,
    split_rotation: int,
    parallel_workers: int = 1,
) -> None:
    """Clean cell geometry before serial mesh writes or topology scans."""
    def cleanup_rows(row_start: int, row_end: int) -> None:
        for row_index in range(row_start, row_end):
            serialized_vertices: SerializedVertexCache = {}
            for current_cell in cells[row_index]:
                if current_cell is not None:
                    current_cell.remove_geometry_collapsed_by_mesh_serialization(
                        output_fileformat=output_fileformat,
                        split_rotation=split_rotation,
                        serialized_vertices=serialized_vertices,
                    )

    row_count = cells.shape[0]
    worker_count = max(1, min(parallel_workers, row_count))
    if not _should_parallelize_rows(row_count, worker_count):
        cleanup_rows(0, row_count)
        return

    for _result in _parallel_range_results(
        0,
        row_count,
        worker_count,
        cleanup_rows,
    ):
        pass


def positive_z_surface_contact_edge_record(
    top_meshes: list[SurfaceMesh | None],
    bottom_meshes: list[SurfaceMesh | None],
    split_rotation: int,
    output_fileformat: str,
    side_values: dict[str, float],
) -> dict[str, Any]:
    """Return actual positive top/bottom contact edges by cell side."""
    serialized_vertices: SerializedVertexCache = {}
    top_edges = surface_mesh_edge_counts(
        top_meshes,
        split_rotation,
        output_fileformat,
        serialized_vertices,
    )
    bottom_edges = surface_mesh_edge_counts(
        bottom_meshes,
        split_rotation,
        output_fileformat,
        serialized_vertices,
    )

    record = {
        "side_edges": _empty_side_edge_sets(),
        "diagonal_edges": set(),
        "has_diagonal": False,
    }
    for edge_key, top_count in top_edges.items():
        bottom_count = bottom_edges.get(edge_key, 0)
        if bottom_count == 0:
            continue
        if edge_key[0][2] <= 0 or edge_key[1][2] <= 0:
            continue
        side = edge_cardinal_side(
            edge_xy_signature(edge_key[0], edge_key[1]),
            side_values,
        )
        if side is None:
            if top_count + bottom_count > 2:
                record["has_diagonal"] = True
            record["diagonal_edges"].add(edge_key)
        else:
            record["side_edges"][side].add(edge_key)
    return record


def _positive_z_effective_difference_corners(
    record: PositiveZNudgeRecord,
) -> list[IntermediateCorner]:
    """Return local difference-removal corners implied by a plan record."""
    difference_corners = list(record.get("difference_corners", []))
    if difference_corners:
        return difference_corners

    corners = list(record.get("corners", []))
    if corners:
        return corners

    contact_corners = list(record.get("contact_corners", []))
    split_sides = set(record.get("split_sides", set()))
    if len(contact_corners) == 3:
        required_split_sides = _nudge_keep_footprint_split_sides(
            contact_corners,
        )
        if required_split_sides and required_split_sides.issubset(split_sides):
            return contact_corners
        return []

    if len(contact_corners) == 2:
        contact_set = frozenset(contact_corners)
        if contact_set not in POSITIVE_Z_OPPOSITE_CORNER_PAIRS:
            return []
        open_corner_by_split_sides = {
            frozenset({"S", "E"}): IntermediateCorner.SE,
            frozenset({"S", "W"}): IntermediateCorner.SW,
            frozenset({"N", "E"}): IntermediateCorner.NE,
            frozenset({"N", "W"}): IntermediateCorner.NW,
        }
        open_corner = open_corner_by_split_sides.get(frozenset(split_sides))
        if open_corner is None and len(split_sides) == 1:
            split_side = next(iter(split_sides))
            if contact_set == POSITIVE_Z_SW_NE_DIAGONAL:
                if split_side in {"S", "E"}:
                    open_corner = IntermediateCorner.SE
                elif split_side in {"N", "W"}:
                    open_corner = IntermediateCorner.NW
            elif contact_set == POSITIVE_Z_SE_NW_DIAGONAL:
                if split_side in {"S", "W"}:
                    open_corner = IntermediateCorner.SW
                elif split_side in {"N", "E"}:
                    open_corner = IntermediateCorner.NE
        if open_corner is None:
            return []
        promoted_corners = [
            corner
            for corner in [
                IntermediateCorner.NW,
                IntermediateCorner.NE,
                IntermediateCorner.SW,
                IntermediateCorner.SE,
            ]
            if corner is not open_corner
        ]
        if contact_set.issubset(set(promoted_corners)):
            return promoted_corners
    return []


def _positive_z_has_opposite_diagonal_corners(
    corners: Sequence[IntermediateCorner],
) -> bool:
    """Return whether corners include either positive-Z diagonal pair."""
    corner_set = set(corners)
    return any(
        diagonal_corners.issubset(corner_set)
        for diagonal_corners in POSITIVE_Z_OPPOSITE_CORNER_PAIRS
    )


def _positive_z_actionable_corners(
    cell_j: int,
    cell_i: int,
    corners_for_location: Callable[
        [int, int],
        Sequence[IntermediateCorner],
    ],
    has_shared_side_pattern: Callable[
        [int, int, Sequence[IntermediateCorner]],
        bool,
    ],
    has_opposite_diagonal_pattern: Callable[
        [int, int, Sequence[IntermediateCorner]],
        bool,
    ],
) -> list[IntermediateCorner]:
    """Return corners that should be nudged due to local or nearby contact."""
    corners = list(corners_for_location(cell_j, cell_i))
    if not corners:
        return []

    def has_actionable_pattern(
        location_j: int,
        location_i: int,
        location_corners: Sequence[IntermediateCorner],
    ) -> bool:
        return (
            has_shared_side_pattern(
                location_j,
                location_i,
                location_corners,
            )
            or has_opposite_diagonal_pattern(
                location_j,
                location_i,
                location_corners,
            )
        )

    if has_actionable_pattern(cell_j, cell_i, corners):
        return corners

    for neighbor_j in range(cell_j - 1, cell_j + 2):
        for neighbor_i in range(cell_i - 1, cell_i + 2):
            if neighbor_j == cell_j and neighbor_i == cell_i:
                continue
            neighbor_corners = list(
                corners_for_location(neighbor_j, neighbor_i),
            )
            if has_actionable_pattern(
                neighbor_j,
                neighbor_i,
                neighbor_corners,
            ):
                return corners
    return []


def _positive_z_neighbor_split_sides_from_plan(
    positive_z_nudge_plan: PositiveZNudgePlan,
    corners_from_record: Callable[
        [PositiveZNudgeRecord],
        Sequence[IntermediateCorner],
    ],
) -> dict[tuple[int, int], set[str]]:
    """Return neighbor splits implied by positive-Z local clipping."""
    neighbor_split_sides: dict[tuple[int, int], set[str]] = {}
    for location, record in positive_z_nudge_plan.items():
        corners = list(corners_from_record(record))
        if not corners or len(corners) == 4:
            continue
        for side, neighbor_delta, neighbor_side in CELL_NEIGHBOR_SIDES:
            if not _nudge_keep_footprint_splits_side(corners, side):
                continue
            neighbor_location = (
                location[0] + neighbor_delta[0],
                location[1] + neighbor_delta[1],
            )
            neighbor_split_sides.setdefault(neighbor_location, set()).add(
                neighbor_side,
            )
    return neighbor_split_sides


def _positive_z_difference_neighbor_split_sides(
    positive_z_nudge_plan: PositiveZNudgePlan,
) -> dict[tuple[int, int], set[str]]:
    """Return difference-only neighbor splits implied by local clipping."""
    return _positive_z_neighbor_split_sides_from_plan(
        positive_z_nudge_plan,
        _positive_z_effective_difference_corners,
    )


def _positive_z_empty_cleaned_edge_record() -> dict[str, Any]:
    """Return a fresh positive-Z cleaned edge record with no contact."""
    return {
        "corners": [],
        "side_edges": _empty_side_edge_sets(),
        "has_diagonal": False,
    }


def _positive_z_cell_surface_values(
    upper_raster: np.ndarray,
    lower_raster: np.ndarray,
    cell_j: int,
    cell_i: int,
    zero_threshold: float,
) -> PositiveZSurfaceValues | None:
    """Return positive-Z top and corrected bottom corner values for a cell."""
    upper_elevs = interpolate_with_NaN(upper_raster, cell_i, cell_j)
    lower_elevs = interpolate_with_NaN(lower_raster, cell_i, cell_j)
    if any(
        elev is None or np.isnan(elev)
        for elev in (*upper_elevs, *lower_elevs)
    ):
        return None

    upper_ne, upper_nw, upper_se, upper_sw = upper_elevs
    lower_ne, lower_nw, lower_se, lower_sw = lower_elevs
    upper_values = {
        IntermediateCorner.NW: upper_nw,
        IntermediateCorner.NE: upper_ne,
        IntermediateCorner.SW: upper_sw,
        IntermediateCorner.SE: upper_se,
    }
    lower_values = {
        IntermediateCorner.NW: lower_nw,
        IntermediateCorner.NE: lower_ne,
        IntermediateCorner.SW: lower_sw,
        IntermediateCorner.SE: lower_se,
    }
    for corner, elev in list(lower_values.items()):
        if elev < zero_threshold:
            lower_values[corner] = 0
    return upper_values, lower_values


def _positive_z_nudge_corners_from_values(
    surface_values: PositiveZSurfaceValues,
    output_fileformat: str,
) -> list[IntermediateCorner]:
    """Return positive-Z nudge corners from top and bottom corner values."""
    upper_values, lower_values = surface_values
    corners: list[IntermediateCorner] = []
    for corner in IntermediateCorner:
        upper_z = normalize_coordinate_to_match_mesh_serialization(
            upper_values[corner],
            output_fileformat,
        )
        lower_z = normalize_coordinate_to_match_mesh_serialization(
            lower_values[corner],
            output_fileformat,
        )
        if upper_z == lower_z and upper_z > 0:
            corners.append(corner)
    if len(corners) == 4:
        return []
    return corners


def _positive_z_cleaned_edge_record_from_values(
    corners: Sequence[IntermediateCorner],
    surface_values: PositiveZSurfaceValues,
    cell_j: int,
    cell_i: int,
    cell_size: float,
    offsetx: float,
    offsety: float,
    split_rotation: int,
    output_fileformat: str,
) -> dict[str, Any]:
    """Return contact edges after applying local zero-height cleanup."""
    if not corners:
        return _positive_z_empty_cleaned_edge_record()

    upper_values, lower_values = surface_values
    cell_w, cell_e, cell_n, cell_s = cell_bounds_for_location(
        cell_j,
        cell_i,
        cell_size,
        offsetx,
        offsety,
    )
    temp_top = quad(
        vertex(cell_w, cell_n, upper_values[IntermediateCorner.NW]),
        vertex(cell_w, cell_s, upper_values[IntermediateCorner.SW]),
        vertex(cell_e, cell_s, upper_values[IntermediateCorner.SE]),
        vertex(cell_e, cell_n, upper_values[IntermediateCorner.NE]),
    )
    temp_bottom = quad(
        vertex(cell_w, cell_n, lower_values[IntermediateCorner.NW]),
        vertex(cell_e, cell_n, lower_values[IntermediateCorner.NE]),
        vertex(cell_e, cell_s, lower_values[IntermediateCorner.SE]),
        vertex(cell_w, cell_s, lower_values[IntermediateCorner.SW]),
    )
    temp_cell = cell(
        temp_top,
        temp_bottom,
        _empty_borders(),
    )
    temp_cell.remove_zero_height_volumes(
        split_rotation=split_rotation,
        output_fileformat=output_fileformat,
    )

    contact_record = positive_z_surface_contact_edge_record(
        [temp_cell.topquad] + (temp_cell.topSurfacePolygons or []),
        [temp_cell.bottomquad] + (temp_cell.bottomSurfacePolygons or []),
        split_rotation,
        output_fileformat,
        cell_side_values(
            cell_j,
            cell_i,
            cell_size,
            offsetx,
            offsety,
            output_fileformat,
        ),
    )
    return {
        "corners": list(corners),
        "side_edges": contact_record["side_edges"],
        "has_diagonal": contact_record["has_diagonal"],
    }


def _build_positive_z_nudge_plan(
    upper_raster: np.ndarray | None,
    lower_raster: np.ndarray | None,
    emit_raster: np.ndarray | None,
    split_emit_raster: np.ndarray | None,
    cell_size: float,
    offsetx: float,
    offsety: float,
    split_rotation: int,
    ymaxidx: int,
    xmaxidx: int,
    zero_threshold: float,
    output_fileformat: str,
    parallel_workers: int = 1,
) -> PositiveZNudgePlan:
    """Return per-cell positive-Z nudge and split records.

    The plan is keyed by padded raster row/col so normal and difference pair
    meshes can consume identical positive-Z nudge decisions while still
    building geometry locally in each cell.
    """
    if upper_raster is None or lower_raster is None or emit_raster is None:
        return {}

    corners_by_location: dict[
        tuple[int, int],
        tuple[IntermediateCorner, ...],
    ] = {}
    surface_values_by_location: dict[
        tuple[int, int],
        PositiveZSurfaceValues,
    ] = {}

    def cell_will_be_created(cell_j: int, cell_i: int) -> bool:
        if (
            cell_j < 1
            or cell_j > ymaxidx
            or cell_i < 1
            or cell_i > xmaxidx
        ):
            return False
        return not np.isnan(emit_raster[cell_j, cell_i])

    def split_cell_will_be_created(cell_j: int, cell_i: int) -> bool:
        if (
            cell_j < 1
            or cell_j > ymaxidx
            or cell_i < 1
            or cell_i > xmaxidx
        ):
            return False
        raster = split_emit_raster if split_emit_raster is not None else emit_raster
        return not np.isnan(raster[cell_j, cell_i])

    def candidate_for_cell(
        cell_j: int,
        cell_i: int,
    ) -> tuple[
        tuple[int, int],
        tuple[IntermediateCorner, ...],
        PositiveZSurfaceValues,
    ] | None:
        if not cell_will_be_created(cell_j, cell_i):
            return None

        surface_values = _positive_z_cell_surface_values(
            upper_raster,
            lower_raster,
            cell_j,
            cell_i,
            zero_threshold,
        )
        if surface_values is None:
            return None

        corners = tuple(
            _positive_z_nudge_corners_from_values(
                surface_values,
                output_fileformat,
            )
        )
        if not corners:
            return None
        return (cell_j, cell_i), corners, surface_values

    def candidates_for_rows(
        row_start: int,
        row_end: int,
    ) -> tuple[
        dict[tuple[int, int], tuple[IntermediateCorner, ...]],
        dict[tuple[int, int], PositiveZSurfaceValues],
    ]:
        row_corners: dict[
            tuple[int, int],
            tuple[IntermediateCorner, ...],
        ] = {}
        row_surface_values: dict[
            tuple[int, int],
            PositiveZSurfaceValues,
        ] = {}
        for cell_j in range(row_start, row_end):
            for cell_i in range(1, xmaxidx + 1):
                candidate = candidate_for_cell(cell_j, cell_i)
                if candidate is None:
                    continue
                location, corners, surface_values = candidate
                row_corners[location] = corners
                row_surface_values[location] = surface_values
        return row_corners, row_surface_values

    def merge_candidates(
        row_candidates: tuple[
            dict[tuple[int, int], tuple[IntermediateCorner, ...]],
            dict[tuple[int, int], PositiveZSurfaceValues],
        ],
    ) -> None:
        row_corners, row_surface_values = row_candidates
        corners_by_location.update(row_corners)
        surface_values_by_location.update(row_surface_values)

    worker_count = max(1, min(parallel_workers, ymaxidx))
    if not _should_parallelize_rows(ymaxidx, worker_count):
        merge_candidates(candidates_for_rows(1, ymaxidx + 1))
    else:
        for row_candidates in _parallel_range_results(
            1,
            ymaxidx + 1,
            worker_count,
            candidates_for_rows,
        ):
            merge_candidates(row_candidates)

    edge_record_cache: dict[tuple[int, int], dict[str, Any]] = {}

    def cleaned_edge_record(cell_j: int, cell_i: int) -> dict[str, Any]:
        location = (cell_j, cell_i)
        if location in edge_record_cache:
            return edge_record_cache[location]
        corners = corners_by_location.get(location)
        surface_values = surface_values_by_location.get(location)
        if not corners or surface_values is None:
            empty_record = _positive_z_empty_cleaned_edge_record()
            edge_record_cache[location] = empty_record
            return empty_record

        record = _positive_z_cleaned_edge_record_from_values(
            corners,
            surface_values,
            cell_j,
            cell_i,
            cell_size,
            offsetx,
            offsety,
            split_rotation,
            output_fileformat,
        )
        edge_record_cache[location] = record
        return record

    def has_shared_side_pattern(
        cell_j: int,
        cell_i: int,
        corners: Sequence[IntermediateCorner],
    ) -> bool:
        record = cleaned_edge_record(cell_j, cell_i)
        corner_set = set(corners)
        for (
            side_corners,
            side,
            neighbor_delta,
            neighbor_opposite_side,
            neighbor_side,
        ) in POSITIVE_Z_SIDE_CONTACT_CHECKS:
            if not side_corners.issubset(corner_set):
                continue
            neighbor = (
                cell_j + neighbor_delta[0],
                cell_i + neighbor_delta[1],
            )
            if not split_cell_will_be_created(neighbor[0], neighbor[1]):
                continue
            neighbor_corners = corners_by_location.get(neighbor, ())
            if not neighbor_side.issubset(neighbor_corners):
                continue
            if not record["side_edges"][side]:
                return True
            if (
                record["side_edges"][side]
                & cleaned_edge_record(
                    neighbor[0],
                    neighbor[1],
                )["side_edges"][neighbor_opposite_side]
            ):
                return True
        return False

    def has_opposite_diagonal_pattern(
        cell_j: int,
        cell_i: int,
        corners: Sequence[IntermediateCorner],
    ) -> bool:
        return (
            cleaned_edge_record(cell_j, cell_i)["has_diagonal"]
            and _positive_z_has_opposite_diagonal_corners(corners)
        )

    def actionable_corners(
        cell_j: int,
        cell_i: int,
    ) -> tuple[IntermediateCorner, ...]:
        return tuple(
            _positive_z_actionable_corners(
                cell_j,
                cell_i,
                lambda location_j, location_i: corners_by_location.get(
                    (location_j, location_i),
                    (),
                ),
                has_shared_side_pattern,
                has_opposite_diagonal_pattern,
            )
        )

    actionable_by_location = {
        location: corners
        for location in corners_by_location
        if (corners := actionable_corners(*location))
    }

    def midpoint_z_by_name_for_location(
        location: tuple[int, int],
        affected_corners: Sequence[IntermediateCorner],
    ) -> dict[str, float]:
        vertex_names = _nudge_keep_vertex_names(affected_corners)
        surface_values = surface_values_by_location.get(location)
        if vertex_names is None or surface_values is None:
            return {}
        upper_values = surface_values[0]
        return {
            midpoint_name: (
                upper_values[midpoint_corners[0]]
                + upper_values[midpoint_corners[1]]
            ) / 2
            for midpoint_name, midpoint_corners
            in NUDGE_MIDPOINT_CORNERS_BY_NAME.items()
            if midpoint_name in vertex_names
        }

    plan: PositiveZNudgePlan = {
        location: {
            "corners": list(corners),
            "split_sides": set(),
            "contact_corners": list(corners_by_location.get(location, ())),
            "midpoint_z_by_name": midpoint_z_by_name_for_location(
                location,
                corners,
            ),
        }
        for location, corners in actionable_by_location.items()
    }
    for (cell_j, cell_i), corners in actionable_by_location.items():
        for current_side, neighbor_delta, neighbor_side in CELL_NEIGHBOR_SIDES:
            neighbor_location = (
                cell_j + neighbor_delta[0],
                cell_i + neighbor_delta[1],
            )
            if not _nudge_keep_footprint_splits_side(corners, current_side):
                continue
            neighbor_j, neighbor_i = neighbor_location
            if not split_cell_will_be_created(neighbor_j, neighbor_i):
                continue
            record = plan.setdefault(
                neighbor_location,
                {
                    "corners": [],
                    "split_sides": set(),
                    "contact_corners": list(
                        corners_by_location.get(neighbor_location, ()),
                    ),
                    "midpoint_z_by_name": {},
                },
            )
            record["split_sides"].add(neighbor_side)

    for location, corners in corners_by_location.items():
        plan.setdefault(
            location,
            {
                "corners": [],
                "split_sides": set(),
                "contact_corners": list(corners),
                "midpoint_z_by_name": {},
            },
        )

    return plan


def _filter_positive_z_nudge_plan_to_actual_overused_edges(
    candidate_plan: PositiveZNudgePlan,
    cells: np.ndarray,
    cell_size: float,
    offsetx: float,
    offsety: float,
    split_rotation: int,
    output_fileformat: str,
    parallel_workers: int = 1,
) -> PositiveZNudgePlan:
    """Keep candidate cells tied to emitted positive-Z contact overuse."""
    def add_edges_to_counts(
        edge_counts: dict[Edge3D, int],
        meshes: list[SurfaceMesh | None],
    ) -> None:
        for edge_key, count in surface_mesh_edge_counts(
            meshes,
            split_rotation,
            output_fileformat,
        ).items():
            edge_counts[edge_key] = edge_counts.get(edge_key, 0) + count

    def edge_counts_for_rows(
        row_start: int,
        row_end: int,
    ) -> dict[Edge3D, int]:
        row_edge_counts: dict[Edge3D, int] = {}
        for row in range(row_start, row_end):
            for col in range(cells.shape[1]):
                current_cell = cells[row, col]
                if current_cell is None:
                    continue
                add_edges_to_counts(
                    row_edge_counts,
                    current_cell.iter_meshes_for_model(),
                )
        return row_edge_counts

    edge_counts: dict[Edge3D, int] = {}
    worker_count = max(1, min(parallel_workers, cells.shape[0]))
    if not _should_parallelize_rows(cells.shape[0], worker_count):
        _merge_count_map(edge_counts, edge_counts_for_rows(0, cells.shape[0]))
    else:
        for row_edge_counts in _parallel_range_results(
            0,
            cells.shape[0],
            worker_count,
            edge_counts_for_rows,
        ):
            _merge_count_map(edge_counts, row_edge_counts)

    overused_edges = {
        edge_key
        for edge_key, count in edge_counts.items()
        if count > 2 and edge_key[0][2] > 0 and edge_key[1][2] > 0
    }
    overused_edges_by_endpoint: dict[tuple[float, ...], set[Edge3D]] = {}
    for edge_key in overused_edges:
        for endpoint in edge_key:
            overused_edges_by_endpoint.setdefault(endpoint, set()).add(
                edge_key,
            )

    def cell_exists(cell_j: int, cell_i: int) -> bool:
        row = cell_j - 1
        col = cell_i - 1
        return (
            0 <= row < cells.shape[0]
            and 0 <= col < cells.shape[1]
            and cells[row, col] is not None
        )

    filtered: PositiveZNudgePlan = {}
    side_corners_by_side = {
        side: list(corners)
        for (
            corners,
            side,
            _delta,
            _opposite_side,
            _neighbor_side,
        ) in POSITIVE_Z_SIDE_CONTACT_CHECKS
    }

    contact_record_cache: dict[tuple[int, int], dict[str, Any]] = {}

    def contact_record_for_location(
        location: tuple[int, int],
    ) -> dict[str, Any]:
        if location not in contact_record_cache:
            current_cell = cells[location[0] - 1, location[1] - 1]
            contact_record_cache[location] = (
                positive_z_surface_contact_edge_record(
                    current_cell.top_surface_meshes(),
                    current_cell.bottom_surface_meshes(),
                    split_rotation,
                    output_fileformat,
                    cell_side_values(
                        *location,
                        cell_size,
                        offsetx,
                        offsety,
                        output_fileformat,
                    ),
                )
            )
        return contact_record_cache[location]

    def contact_edges_from_record(record: dict[str, Any]) -> set[Edge3D]:
        edges: set[Edge3D] = set(record["diagonal_edges"])
        for side_edges in record["side_edges"].values():
            edges.update(side_edges)
        return edges

    def serialized_contact_corner_points(
        location: tuple[int, int],
        corners: Sequence[IntermediateCorner],
    ) -> set[tuple[float, ...]]:
        current_cell = cells[location[0] - 1, location[1] - 1]
        side_values = cell_side_values(
            *location,
            cell_size,
            offsetx,
            offsety,
            output_fileformat,
        )
        corner_xy = {
            IntermediateCorner.NW: (side_values["W"], side_values["N"]),
            IntermediateCorner.NE: (side_values["E"], side_values["N"]),
            IntermediateCorner.SW: (side_values["W"], side_values["S"]),
            IntermediateCorner.SE: (side_values["E"], side_values["S"]),
        }
        target_xy = {corner_xy[corner] for corner in corners}

        serialized_vertices: SerializedVertexCache = {}

        def serialized_positive_points(
            meshes: Iterable[SurfaceMesh | None],
        ) -> set[tuple[float, ...]]:
            points: set[tuple[float, ...]] = set()
            for mesh in meshes:
                if mesh is None:
                    continue
                if isinstance(mesh, quad):
                    coord_iterable = (
                        v.coords for v in mesh.vl if v is not None
                    )
                else:
                    coords = mesh.exterior.coords
                    coord_iterable = (
                        coords[index] for index in range(len(coords) - 1)
                    )
                for coord in coord_iterable:
                    serialized = _serialized_vertex_from_cache(
                        coord,
                        output_fileformat,
                        serialized_vertices,
                    )
                    if serialized[2] > 0 and serialized[:2] in target_xy:
                        points.add(serialized)
            return points

        return serialized_positive_points(current_cell.top_surface_meshes()) & (
            serialized_positive_points(current_cell.bottom_surface_meshes())
        )

    def midpoint_z_by_name_from_cell_top(
        location: tuple[int, int],
        affected_corners: Sequence[IntermediateCorner],
        fallback_record: dict[str, Any],
    ) -> dict[str, float]:
        cell_j, cell_i = location
        current_cell = cells[cell_j - 1, cell_i - 1]
        if current_cell is None or current_cell.topquad is None:
            return dict(fallback_record.get("midpoint_z_by_name", {}))

        vertex_names = _nudge_keep_vertex_names(affected_corners)
        if vertex_names is None:
            return {}

        cell_w, cell_e, cell_n, cell_s = cell_bounds_for_location(
            cell_j,
            cell_i,
            cell_size,
            offsetx,
            offsety,
        )
        points = cell_corner_points(cell_w, cell_e, cell_n, cell_s)
        planes = _surface_planes_from_current_geometry(
            current_cell.topquad,
            current_cell.topSurfacePolygons,
            split_rotation,
        )
        corner_vertices = quad_corner_vertices_by_xy(
            current_cell.topquad,
            cell_w,
            cell_e,
            cell_n,
            cell_s,
        )
        midpoint_z_by_name: dict[str, float] = {}
        for midpoint_name, midpoint_corners in (
            NUDGE_MIDPOINT_CORNERS_BY_NAME.items()
        ):
            if midpoint_name not in vertex_names:
                continue
            if planes:
                try:
                    midpoint_with_z = interpolate_z_planar(
                        shapely.Point(points[midpoint_name]),
                        planes,
                    )
                except (TypeError, ValueError):
                    midpoint_with_z = None
                if isinstance(midpoint_with_z, shapely.Point):
                    midpoint_coord = midpoint_with_z.coords[0]
                    if len(midpoint_coord) >= 3:
                        midpoint_z_by_name[midpoint_name] = midpoint_coord[2]
                        continue

            midpoint_vertices = [
                corner_vertices.get(corner)
                for corner in midpoint_corners
            ]
            if any(corner_vertex is None for corner_vertex in midpoint_vertices):
                return dict(fallback_record.get("midpoint_z_by_name", {}))
            midpoint_z_by_name[midpoint_name] = (
                midpoint_vertices[0].coords[2]
                + midpoint_vertices[1].coords[2]
            ) / 2
        return midpoint_z_by_name

    for location, candidate in candidate_plan.items():
        corners = list(candidate.get("corners", []))
        if location in filtered:
            continue
        if not cell_exists(*location):
            continue
        if not corners:
            continue

        contact_record = contact_record_for_location(location)
        corner_set = set(corners)
        confirmed_overused_edges: set[Edge3D] = set()
        for (
            side_corners,
            side,
            _delta,
            _opposite_side,
            _neighbor_side,
        ) in POSITIVE_Z_SIDE_CONTACT_CHECKS:
            if not side_corners.issubset(corner_set):
                continue
            confirmed_overused_edges.update(
                contact_record["side_edges"][side] & overused_edges,
            )
        confirmed_overused_edges.update(
            contact_record["diagonal_edges"] & overused_edges,
        )
        if not confirmed_overused_edges:
            continue

        filtered[location] = {
            "corners": corners,
            "split_sides": set(),
            "contact_corners": list(candidate.get("contact_corners", corners)),
            "confirmed_overused_edges": confirmed_overused_edges,
            "midpoint_z_by_name": midpoint_z_by_name_from_cell_top(
                location,
                corners,
                candidate,
            ),
        }

    for location, candidate in candidate_plan.items():
        if location in filtered or not cell_exists(*location):
            continue

        corners = list(candidate.get("corners", []))
        contact_corners = list(candidate.get("contact_corners", corners))
        one_corner_contact = (
            corners
            if len(corners) == 1
            else (
                contact_corners
                if not corners and len(contact_corners) == 1
                else []
            )
        )
        if not one_corner_contact:
            continue

        confirmed_overused_edges: set[Edge3D] = set()
        for contact_point in serialized_contact_corner_points(
            location,
            one_corner_contact,
        ):
            confirmed_overused_edges.update(
                overused_edges_by_endpoint.get(contact_point, set()),
            )
        if not confirmed_overused_edges:
            continue
        filtered[location] = {
            "corners": one_corner_contact,
            "split_sides": set(),
            "contact_corners": contact_corners,
            "confirmed_overused_edges": confirmed_overused_edges,
            "midpoint_z_by_name": midpoint_z_by_name_from_cell_top(
                location,
                one_corner_contact,
                candidate,
            ),
        }

    confirmed_nudge_edges: set[Edge3D] = set()
    for record in filtered.values():
        if record.get("corners"):
            confirmed_nudge_edges.update(record["confirmed_overused_edges"])
    confirmed_nudge_endpoints = {
        endpoint
        for edge in confirmed_nudge_edges
        for endpoint in edge
    }
    if confirmed_nudge_endpoints:
        for location, candidate in candidate_plan.items():
            corners = list(candidate.get("corners", []))
            contact_corners = list(candidate.get("contact_corners", corners))
            if location in filtered or not cell_exists(*location):
                continue
            one_corner_contact = (
                corners if len(corners) == 1
                else contact_corners if not corners and len(contact_corners) == 1
                else []
            )
            if one_corner_contact:
                if (
                    serialized_contact_corner_points(
                        location,
                        one_corner_contact,
                    )
                    & confirmed_nudge_endpoints
                ):
                    filtered[location] = {
                        "corners": one_corner_contact,
                        "split_sides": set(),
                        "contact_corners": contact_corners,
                        "confirmed_overused_edges": set(),
                        "midpoint_z_by_name": midpoint_z_by_name_from_cell_top(
                            location,
                            one_corner_contact,
                            candidate,
                        ),
                    }
                continue
            if not corners:
                continue
            contact_edges = contact_edges_from_record(
                contact_record_for_location(location),
            )
            if not any(
                edge[0] in confirmed_nudge_endpoints
                or edge[1] in confirmed_nudge_endpoints
                for edge in contact_edges
            ):
                continue
            filtered[location] = {
                "corners": corners,
                "split_sides": set(),
                "contact_corners": list(
                    candidate.get("contact_corners", corners),
                ),
                "confirmed_overused_edges": set(),
                "midpoint_z_by_name": midpoint_z_by_name_from_cell_top(
                    location,
                    corners,
                    candidate,
                ),
            }

    for row in range(cells.shape[0]):
        for col in range(cells.shape[1]):
            current_cell = cells[row, col]
            if current_cell is None:
                continue

            location = (row + 1, col + 1)
            contact_record = contact_record_for_location(location)
            confirmed_side_edges: set[Edge3D] = set()
            inferred_corners: list[IntermediateCorner] = []
            for side, side_edges in contact_record["side_edges"].items():
                side_overused_edges = side_edges & overused_edges
                if not side_overused_edges:
                    continue
                confirmed_side_edges.update(side_overused_edges)
                for corner in side_corners_by_side[side]:
                    if corner not in inferred_corners:
                        inferred_corners.append(corner)
            if not confirmed_side_edges:
                continue

            candidate = candidate_plan.get(location, {})
            record = filtered.setdefault(
                location,
                {
                    "corners": [],
                    "difference_corners": [],
                    "split_sides": set(),
                    "contact_corners": list(
                        candidate.get("contact_corners", []),
                    ),
                    "confirmed_overused_edges": set(),
                    "midpoint_z_by_name": {},
                },
            )
            merged_corners = list(
                dict.fromkeys(
                    list(record.get("difference_corners", []))
                    + list(inferred_corners),
                ),
            )
            if 0 < len(merged_corners) < 4:
                record["difference_corners"] = merged_corners
                record.setdefault("confirmed_overused_edges", set()).update(
                    confirmed_side_edges,
                )
                record["difference_midpoint_z_by_name"] = (
                    midpoint_z_by_name_from_cell_top(
                        location,
                        merged_corners,
                        candidate,
                    )
                )

    for row in range(cells.shape[0]):
        for col in range(cells.shape[1]):
            current_cell = cells[row, col]
            if (
                current_cell is None
                or not current_cell.topSurfacePolygons
                or not current_cell.bottomSurfacePolygons
            ):
                continue
            location = (row + 1, col + 1)
            contact_record = contact_record_for_location(location)
            confirmed_flip_edges: set[Edge3D] = set()
            for side_edges in contact_record["side_edges"].values():
                confirmed_flip_edges.update(side_edges & overused_edges)
            confirmed_flip_edges.update(
                contact_record["diagonal_edges"] & overused_edges,
            )
            if not confirmed_flip_edges:
                continue

            record = filtered.setdefault(
                location,
                {
                    "corners": [],
                    "split_sides": set(),
                    "contact_corners": list(
                        candidate_plan.get(location, {}).get(
                            "contact_corners",
                            [],
                        )
                    ),
                    "confirmed_overused_edges": set(),
                    "midpoint_z_by_name": dict(
                        candidate_plan.get(location, {}).get(
                            "midpoint_z_by_name",
                            {},
                        ),
                    ),
                },
            )
            record.setdefault("flip_edges", set()).update(
                confirmed_flip_edges,
            )

    for (cell_j, cell_i), record in list(filtered.items()):
        corners = record["corners"]
        for current_side, neighbor_delta, neighbor_side in CELL_NEIGHBOR_SIDES:
            neighbor_location = (
                cell_j + neighbor_delta[0],
                cell_i + neighbor_delta[1],
            )
            if not _nudge_keep_footprint_splits_side(corners, current_side):
                continue
            neighbor_candidate = candidate_plan.get(neighbor_location, {})
            if (
                not cell_exists(*neighbor_location)
                and not neighbor_candidate.get("split_sides")
            ):
                continue
            neighbor_record = filtered.setdefault(
                neighbor_location,
                {
                    "corners": [],
                    "split_sides": set(),
                    "contact_corners": list(
                        neighbor_candidate.get("contact_corners", [])
                    ),
                    "confirmed_overused_edges": set(),
                    "midpoint_z_by_name": dict(
                        neighbor_candidate.get("midpoint_z_by_name", {}),
                    ),
                },
            )
            neighbor_record["split_sides"].add(neighbor_side)
            current_midpoint_name = NUDGE_SIDE_MIDPOINT_NAME[current_side]
            neighbor_midpoint_name = NUDGE_SIDE_MIDPOINT_NAME[neighbor_side]
            current_midpoint_z = record.get(
                "midpoint_z_by_name",
                {},
            ).get(current_midpoint_name)
            if current_midpoint_z is not None:
                neighbor_record.setdefault(
                    "midpoint_z_by_name",
                    {},
                )[neighbor_midpoint_name] = current_midpoint_z

    midpoint_z_by_xy: dict[tuple[float, float], float] = {}
    for (cell_j, cell_i), record in sorted(filtered.items()):
        midpoint_z_by_name = record.get("midpoint_z_by_name", {})
        if not midpoint_z_by_name:
            continue
        cell_w, cell_e, cell_n, cell_s = cell_bounds_for_location(
            cell_j,
            cell_i,
            cell_size,
            offsetx,
            offsety,
        )
        points = cell_corner_points(cell_w, cell_e, cell_n, cell_s)
        for midpoint_name, midpoint_z in list(midpoint_z_by_name.items()):
            if midpoint_name not in points:
                continue
            midpoint_xy = normalize_vertex_to_match_mesh_serialization(
                (*points[midpoint_name], 0.0),
                output_fileformat,
            )[:2]
            serialized_midpoint_z = (
                normalize_coordinate_to_match_mesh_serialization(
                    midpoint_z,
                    output_fileformat,
                )
            )
            shared_z = midpoint_z_by_xy.setdefault(
                midpoint_xy,
                serialized_midpoint_z,
            )
            midpoint_z_by_name[midpoint_name] = shared_z

    return filtered


class cell:
    '''a cell with a top and bottom quad, constructor: uses refs and does NOT copy ...
       except for triangle cells
       '''
    __slots__ = (
        "topquad",
        "bottomquad",
        "borders",
        "is_tri_cell",
        "topSurfacePolygons",
        "bottomSurfacePolygons",
        "surfacePolygonBorders",
    )

    topquad: quad | None
    bottomquad: quad | None
    borders: CardinalWallMap
    is_tri_cell: bool

    topSurfacePolygons: list[shapely.Polygon] | None
    "list of polygons (preferably tris) with X,Y,Z to use for the mesh instead of the topquad"
    bottomSurfacePolygons: list[shapely.Polygon] | None
    "list of polygons (preferably tris) with X,Y,Z to use for the mesh instead of the bottomquad"
    surfacePolygonBorders: list[quad] | None
    # surface polygon borders should be generated using raster polygon edge buckets BorderEdge wall value

    def __init__(
        self,
        topquad: quad | None,
        bottomquad: quad | None,
        borders: CardinalWallMap,
        is_tri_cell: bool = False,
    ) -> None:
        self.topquad = topquad
        self.bottomquad = bottomquad
        self.borders = borders
        self.is_tri_cell = is_tri_cell
        self.topSurfacePolygons = None
        self.bottomSurfacePolygons = None
        self.surfacePolygonBorders = None

    def __str__(self):
        r = hex(id(self)) + "\n top:" + str(self.topquad) + "\n btm:" + str(self.bottomquad) + "\n borders:\n"
        for d in CARDINAL_DIRECTIONS:
            border = self.borders.get(d)
            if border:
                r = r + "  " + d + ": " + str(border) + "\n"
        return r

    def top_surface_meshes(self) -> list[SurfaceMesh]:
        """Return the emitted top surface meshes for this cell."""
        if self.topSurfacePolygons:
            return list(self.topSurfacePolygons)
        return [self.topquad] if self.topquad is not None else []

    def bottom_surface_meshes(self) -> list[SurfaceMesh]:
        """Return the emitted bottom surface meshes for this cell."""
        if self.bottomSurfacePolygons:
            return list(self.bottomSurfacePolygons)
        return [self.bottomquad] if self.bottomquad is not None else []

    def iter_meshes_for_model(
        self,
    ) -> Iterator[Union[quad, shapely.Polygon]]:
        """Yield the meshes to include in the output model for this cell."""
        if self.topSurfacePolygons:
            yield from self.topSurfacePolygons
        elif self.topquad is not None:
            yield self.topquad

        if not self.topSurfacePolygons and self.topquad:
            # if we use topquad, we also use cardinal direction borders
            yield from self.borders.values()

        if self.bottomSurfacePolygons:
            yield from self.bottomSurfacePolygons
        elif self.bottomquad is not None:
            yield self.bottomquad

        if self.surfacePolygonBorders:
            yield from self.surfacePolygonBorders

    def _clear_geometry(self) -> None:
        """Remove all emitted geometry from this cell."""
        self.topquad = None
        self.bottomquad = None
        self.topSurfacePolygons = None
        self.bottomSurfacePolygons = None
        self.surfacePolygonBorders = None
        self.borders = _empty_borders()

    def meshes_for_model(self) -> list[Union[quad, shapely.Polygon]]:
        """Return the meshes to include in the output model for this cell."""
        return list(self.iter_meshes_for_model())

    def remove_geometry_collapsed_by_mesh_serialization(
        self,
        output_fileformat: str,
        split_rotation: int,
        serialized_vertices: SerializedVertexCache | None = None,
    ) -> None:
        """Remove cell meshes that collapse at output precision."""
        def normalize_surface_polygons(
            surface_polygons: list[shapely.Polygon] | None,
        ) -> list[shapely.Polygon] | None:
            if not surface_polygons:
                return None

            output: list[shapely.Polygon] = []
            for surface_polygon in surface_polygons:
                output_polygon = (
                    surface_polygon_normalized_to_match_mesh_serialization(
                        surface_polygon,
                        output_fileformat,
                        serialized_vertices=serialized_vertices,
                    )
                )
                if output_polygon is not None:
                    output.append(output_polygon)
            return output or None

        had_top_surface_polygons = bool(self.topSurfacePolygons)
        had_bottom_surface_polygons = bool(self.bottomSurfacePolygons)
        if serialized_vertices is None:
            serialized_vertices = {}

        if self.topquad is not None:
            self.topquad = quad_normalized_to_match_mesh_serialization(
                self.topquad,
                output_fileformat,
                split_rotation,
                serialized_vertices,
            )

        if self.bottomquad is not None:
            self.bottomquad = quad_normalized_to_match_mesh_serialization(
                self.bottomquad,
                output_fileformat,
                split_rotation,
                serialized_vertices,
            )

        for direction, border in list(self.borders.items()):
            output_border = quad_normalized_to_match_mesh_serialization(
                border,
                output_fileformat,
                split_rotation,
                serialized_vertices,
            )
            if output_border is None:
                self.borders.pop(direction)
            else:
                self.borders[direction] = output_border

        if self.surfacePolygonBorders:
            surface_borders = []
            for surface_border in self.surfacePolygonBorders:
                output_border = quad_normalized_to_match_mesh_serialization(
                    surface_border,
                    output_fileformat,
                    split_rotation,
                    serialized_vertices,
                )
                if output_border is not None:
                    surface_borders.append(output_border)
            self.surfacePolygonBorders = surface_borders or None

        self.topSurfacePolygons = normalize_surface_polygons(
            self.topSurfacePolygons,
        )
        self.bottomSurfacePolygons = normalize_surface_polygons(
            self.bottomSurfacePolygons,
        )
        if had_top_surface_polygons and not self.topSurfacePolygons:
            self.topquad = None
        if had_bottom_surface_polygons and not self.bottomSurfacePolygons:
            self.bottomquad = None

        if self.topquad is None and not self.topSurfacePolygons:
            self._clear_geometry()

    def emitted_top_as_bottom_surfaces(
        self,
    ) -> EmittedBottomSurface:
        """Return emitted top surfaces reoriented for use as a bottom."""
        if self.topSurfacePolygons:
            return (
                None,
                [
                    shapely.orient_polygons(polygon, exterior_cw=True)
                    for polygon in self.topSurfacePolygons
                ],
            )

        if self.topquad is None:
            return None, None

        top_vertices = self.topquad.vl
        if top_vertices[3] is None:
            return (
                quad(
                    top_vertices[0],
                    top_vertices[2],
                    top_vertices[1],
                    None,
                    forced_split_edge=self.topquad.forced_split_edge,
                ),
                None,
            )
        return (
            quad(
                top_vertices[0],
                top_vertices[3],
                top_vertices[2],
                top_vertices[1],
                forced_split_edge=self.topquad.forced_split_edge,
            ),
            None,
        )

    def replace_bottom_surfaces(
        self,
        bottom_surface_quad: quad | None,
        bottom_surface_polygons: list[shapely.Polygon] | None,
        split_rotation: int,
        output_fileformat: str | None = None,
    ) -> None:
        """Replace bottom geometry and rebuild walls against the current top.

        Pair mode uses this after the normal mesh is emitted. The replacement
        preserves the difference top surface and wall footprint decisions, but
        swaps the bottom surface to the exact normal top geometry for the same
        cell.
        """
        replacement_bottom_quad = bottom_surface_quad
        if self.topSurfacePolygons and bottom_surface_quad is not None:
            bottom_surface_polygons = []
            bottom_planes = bottom_surface_quad.get_triangles_in_polygons(
                split_rotation=split_rotation,
            )
            kept_top_surface_polygons: list[shapely.Polygon] = []
            for top_polygon in self.topSurfacePolygons:
                bottom_polygon = interpolate_z_planar(
                    geometry_2d=shapely.orient_polygons(
                        shapely.force_2d(top_polygon),
                        exterior_cw=True,
                    ),
                    planes_3d=bottom_planes,
                )
                if not isinstance(bottom_polygon, shapely.Polygon):
                    raise TypeError(
                        "Shared pair clipped bottom interpolation did not "
                        "return a Polygon."
                    )
                if output_fileformat is not None:
                    bottom_polygon = (
                        polygon_normalized_to_match_mesh_serialization(
                            bottom_polygon,
                            output_fileformat,
                        )
                    )
                    if bottom_polygon is None:
                        continue
                kept_top_surface_polygons.append(top_polygon)
                bottom_surface_polygons.append(bottom_polygon)
            self.topSurfacePolygons = kept_top_surface_polygons
            if not bottom_surface_polygons:
                self._clear_geometry()
                return
            bottom_surface_quad = None

        if (
            bottom_surface_polygons is not None
            and not self.topSurfacePolygons
            and self.topquad is not None
        ):
            top_planes = self.topquad.get_triangles_in_polygons(
                split_rotation=split_rotation,
            )
            promoted_top_polygons: list[shapely.Polygon] = []
            kept_bottom_polygons: list[shapely.Polygon] = []
            for bottom_polygon in bottom_surface_polygons:
                top_polygon = interpolate_z_planar(
                    geometry_2d=shapely.orient_polygons(
                        shapely.force_2d(bottom_polygon),
                        exterior_cw=False,
                    ),
                    planes_3d=top_planes,
                )
                if not isinstance(top_polygon, shapely.Polygon):
                    raise TypeError(
                        "Shared pair top promotion did not return a Polygon."
                    )
                promoted_top_polygons.append(top_polygon)
                kept_bottom_polygons.append(bottom_polygon)

            self.topSurfacePolygons = promoted_top_polygons or None
            bottom_surface_polygons = kept_bottom_polygons or None
            if bottom_surface_polygons is None:
                self._clear_geometry()
                return

        if bottom_surface_polygons is not None:
            if not self.topSurfacePolygons:
                raise RuntimeError(
                    "Shared pair bottom has clipped polygons but the "
                    "difference top does not."
                )
            if replacement_bottom_quad is not None:
                self.bottomquad = replacement_bottom_quad
            self.bottomSurfacePolygons = bottom_surface_polygons
            self.borders = _empty_borders()
            self._rebuild_surface_polygon_borders(output_fileformat)
            return

        if bottom_surface_quad is None:
            raise RuntimeError("Shared pair bottom surface is missing.")
        if self.topquad is None:
            raise RuntimeError("Difference top surface is missing.")

        self._force_top_split_to_bottom_surface(
            bottom_surface_quad,
            None,
            split_rotation,
        )
        self.bottomquad = bottom_surface_quad
        self.bottomSurfacePolygons = None
        self.surfacePolygonBorders = None
        self._rebuild_cardinal_borders(output_fileformat)

    def _force_top_split_to_bottom_surface(
        self,
        bottom_surface_quad: quad | None,
        bottom_surface_polygons: list[shapely.Polygon] | None,
        split_rotation: int,
    ) -> None:
        """Force a full top quad to use the provider bottom diagonal."""
        if self.topquad is None:
            return
        bottom_split_edge = self._bottom_surface_split_edge(
            bottom_surface_quad,
            bottom_surface_polygons,
            split_rotation,
        )
        if bottom_split_edge is None:
            return
        if (
            self.topquad.get_split_edge_indices(split_rotation)
            != bottom_split_edge
        ):
            self.topquad.forced_split_edge = bottom_split_edge

    def _bottom_surface_split_edge(
        self,
        bottom_surface_quad: quad | None,
        bottom_surface_polygons: list[shapely.Polygon] | None,
        split_rotation: int,
    ) -> tuple[int, int] | None:
        """Return the provider split edge when it is a full-cell diagonal."""
        if self.topquad is None or self.topquad.vl[3] is None:
            return None
        if (
            bottom_surface_quad is not None
            and bottom_surface_quad.vl[3] is not None
        ):
            return bottom_surface_quad.get_split_edge_indices(split_rotation)
        if not bottom_surface_polygons:
            return None

        def edge_key(coord0: Coordinate, coord1: Coordinate) -> XYEdge:
            return tuple(
                sorted(
                    (
                        (
                            round(float(coord0[0]), 6),
                            round(float(coord0[1]), 6),
                        ),
                        (
                            round(float(coord1[0]), 6),
                            round(float(coord1[1]), 6),
                        ),
                    )
                )
            )

        top_vertices = self.topquad.vl
        default_edge = edge_key(
            top_vertices[0].coords,
            top_vertices[2].coords,
        )
        rotated_edge = edge_key(
            top_vertices[1].coords,
            top_vertices[3].coords,
        )
        provider_split_edges: set[tuple[int, int]] = set()
        for polygon in bottom_surface_polygons:
            coords = list(polygon.exterior.coords)
            for index, coord0 in enumerate(coords[:-1]):
                coord1 = coords[(index + 1) % (len(coords) - 1)]
                polygon_edge = edge_key(coord0, coord1)
                if polygon_edge == default_edge:
                    provider_split_edges.add(quad._default_split_edge)
                elif polygon_edge == rotated_edge:
                    provider_split_edges.add(quad._rotated_split_edge)
        if len(provider_split_edges) != 1:
            return None
        return next(iter(provider_split_edges))

    def _rebuild_cardinal_borders(
        self,
        output_fileformat: str | None,
    ) -> None:
        """Rebuild existing cardinal walls after replacing a bottom quad."""
        self.borders = _build_cardinal_wall_borders(
            self.borders,
            self.topquad.vl,
            self.bottomquad.vl,
            output_fileformat or "STLb",
        )

    def _rebuild_surface_polygon_borders(
        self,
        output_fileformat: str | None,
    ) -> None:
        """Rebuild existing clipped wall footprints with shared bottom edges."""
        mesh_fileformat = output_fileformat or "STLb"
        requested_lines = _surface_wall_requested_lines(
            self.surfacePolygonBorders,
            output_fileformat=mesh_fileformat,
        )
        if requested_lines is None:
            self.surfacePolygonBorders = None
            return

        self.surfacePolygonBorders = (
            _rebuild_matching_surface_polygon_borders(
                self.topSurfacePolygons,
                self.bottomSurfacePolygons,
                lambda footprint: _linework_covers_footprint(
                    requested_lines,
                    footprint,
                ),
                mesh_fileformat,
            )
            or None
        )

    def split_surface_boundary_midpoints(
        self,
        split_sides: set[str],
        contact_corners: Sequence[IntermediateCorner],
        W: float,
        E: float,
        N: float,
        S: float,
        split_rotation: int,
        output_fileformat: str,
        include_contact_cut_walls: bool = True,
        top_midpoint_corner_vertices: (
            dict[IntermediateCorner, vertex] | None
        ) = None,
        top_midpoint_z_by_name: dict[str, float] | None = None,
        bottom_midpoint_corner_vertices: (
            dict[IntermediateCorner, vertex] | None
        ) = None,
        bottom_midpoint_z_by_name: dict[str, float] | None = None,
        side_cut_wall_sides: set[str] | None = None,
    ) -> bool:
        """Split a cell surface at requested side midpoints."""
        if not split_sides:
            return False

        full_footprint = full_cell_footprint(W, E, N, S)
        current_footprint = _current_surface_footprint(
            self.topSurfacePolygons,
            full_footprint,
            self.topquad,
        )
        if current_footprint.is_empty:
            return False
        bottom_footprint = _current_surface_footprint(
            self.bottomSurfacePolygons,
            full_footprint,
            self.bottomquad,
        )
        if (
            not self.bottomSurfacePolygons
            and bottom_footprint.equals(full_footprint)
            and not current_footprint.equals(full_footprint)
        ):
            bottom_footprint = current_footprint
        if bottom_footprint.is_empty:
            return False
        use_top_footprint_for_bottom = (
            bottom_midpoint_corner_vertices is None
            and bottom_midpoint_z_by_name is None
        )
        if use_top_footprint_for_bottom:
            bottom_footprint = current_footprint

        top_planes = _surface_planes_from_current_geometry(
            self.topquad,
            self.topSurfacePolygons,
            split_rotation,
        )
        bottom_planes = _surface_planes_from_current_geometry(
            self.bottomquad,
            self.bottomSurfacePolygons,
            split_rotation,
        )
        if not top_planes or not bottom_planes:
            return False
        top_existing_z_by_xy = _surface_vertex_z_overrides_by_xy(
            self.topSurfacePolygons,
            output_fileformat,
        )
        bottom_existing_z_by_xy = _surface_vertex_z_overrides_by_xy(
            self.bottomSurfacePolygons,
            output_fileformat,
        )

        points = cell_corner_points(W, E, N, S)
        midpoint_name_by_side = {
            "N": "Nmid",
            "S": "Smid",
            "E": "Emid",
            "W": "Wmid",
        }
        boundary_linework = _geometry_boundary_linework(
            current_footprint,
            output_fileformat,
        )
        bottom_boundary_linework = _geometry_boundary_linework(
            bottom_footprint,
            output_fileformat,
        )
        requested_split_xy: set[tuple[float, float]] = set()
        for side in split_sides:
            midpoint_name = midpoint_name_by_side[side]
            midpoint_xy = normalize_vertex_to_match_mesh_serialization(
                (*points[midpoint_name], 0.0),
                output_fileformat,
            )[:2]
            midpoint = shapely.Point(midpoint_xy)
            if (
                boundary_linework is not None
                and boundary_linework.covers(midpoint)
            ) or (
                bottom_boundary_linework is not None
                and bottom_boundary_linework.covers(midpoint)
            ):
                requested_split_xy.add(midpoint_xy)
        if not requested_split_xy:
            return False

        vertex_names = ["SW"]
        if "S" in split_sides:
            vertex_names.append("Smid")
        vertex_names.append("SE")
        if "E" in split_sides:
            vertex_names.append("Emid")
        vertex_names.append("NE")
        if "N" in split_sides:
            vertex_names.append("Nmid")
        vertex_names.append("NW")
        if "W" in split_sides:
            vertex_names.append("Wmid")
        base_split_footprint = shapely.Polygon(
            [points[name] for name in vertex_names],
        )
        top_split_footprint = (
            base_split_footprint
            if current_footprint.equals(full_footprint)
            else current_footprint
        )
        bottom_split_footprint = (
            base_split_footprint
            if bottom_footprint.equals(full_footprint)
            else bottom_footprint
        )

        if (
            top_midpoint_corner_vertices is not None
            and self.topquad is not None
            and current_footprint.equals(full_footprint)
        ):
            adjusted_top_planes = _nudge_adjusted_surface_planes(
                base_split_footprint,
                quad_corner_vertices_by_xy(self.topquad, W, E, N, S),
                W,
                E,
                N,
                S,
                midpoint_corner_vertices=top_midpoint_corner_vertices,
            )
            if adjusted_top_planes:
                top_planes = adjusted_top_planes
        if (
            bottom_midpoint_corner_vertices is not None
            and self.bottomquad is not None
            and bottom_footprint.equals(full_footprint)
        ):
            adjusted_bottom_planes = _nudge_adjusted_surface_planes(
                base_split_footprint,
                quad_corner_vertices_by_xy(self.bottomquad, W, E, N, S),
                W,
                E,
                N,
                S,
                midpoint_corner_vertices=bottom_midpoint_corner_vertices,
            )
            if adjusted_bottom_planes:
                bottom_planes = adjusted_bottom_planes

        top_fallback_z_by_xy = _nudge_midpoint_z_by_xy(
            W,
            E,
            N,
            S,
            top_midpoint_corner_vertices,
            output_fileformat,
            top_midpoint_z_by_name,
        )
        top_surfaces = _triangulate_2d_geometry_to_3d_polygons(
            top_split_footprint,
            top_planes,
            exterior_cw=False,
            output_fileformat=output_fileformat,
            fallback_z_by_xy=top_fallback_z_by_xy,
        )
        if top_midpoint_corner_vertices is not None or top_midpoint_z_by_name:
            top_surfaces = _surface_polygons_with_midpoint_z(
                top_surfaces,
                W,
                E,
                N,
                S,
                top_midpoint_corner_vertices,
                output_fileformat,
                top_midpoint_z_by_name,
            )
        bottom_fallback_z_by_xy = _nudge_midpoint_z_by_xy(
            W,
            E,
            N,
            S,
            bottom_midpoint_corner_vertices,
            output_fileformat,
            bottom_midpoint_z_by_name,
        )
        if (
            self.bottomSurfacePolygons
            and not bottom_fallback_z_by_xy
            and not use_top_footprint_for_bottom
        ):
            bottom_surfaces = _clip_3d_surface_polygons_to_2d_geometry(
                self.bottomSurfacePolygons,
                bottom_split_footprint,
                exterior_cw=True,
                output_fileformat=output_fileformat,
            )
        else:
            bottom_surfaces = _triangulate_2d_geometry_to_3d_polygons(
                bottom_split_footprint,
                bottom_planes,
                exterior_cw=True,
                output_fileformat=output_fileformat,
                fallback_z_by_xy=bottom_fallback_z_by_xy,
            )
        if not top_surfaces or not bottom_surfaces:
            return False

        def footprint_has_requested_split(footprint: XYEdge) -> bool:
            line = shapely.LineString(footprint)
            return any(
                line.covers(shapely.Point(split_xy))
                for split_xy in requested_split_xy
            )

        serialized_vertices: SerializedVertexCache = {}
        top_boundary_edges = boundary_edge_map_from_meshes(
            top_surfaces,
            output_fileformat=output_fileformat,
            serialized_vertices=serialized_vertices,
        )
        bottom_boundary_edges = boundary_edge_map_from_meshes(
            bottom_surfaces,
            output_fileformat=output_fileformat,
            serialized_vertices=serialized_vertices,
        )
        top_surfaces = _split_surface_boundary_edges_for_wall_matches(
            top_surfaces,
            top_boundary_edges,
            requested_split_xy,
            footprint_has_requested_split,
            output_fileformat,
        )
        bottom_surfaces = _split_surface_boundary_edges_for_wall_matches(
            bottom_surfaces,
            bottom_boundary_edges,
            requested_split_xy,
            footprint_has_requested_split,
            output_fileformat,
        )
        top_surfaces = _surface_polygons_with_z_overrides(
            top_surfaces,
            top_existing_z_by_xy,
            output_fileformat,
        )
        bottom_surfaces = _surface_polygons_with_z_overrides(
            bottom_surfaces,
            bottom_existing_z_by_xy,
            output_fileformat,
        )
        if top_midpoint_corner_vertices is not None or top_midpoint_z_by_name:
            top_surfaces = _surface_polygons_with_midpoint_z(
                top_surfaces,
                W,
                E,
                N,
                S,
                top_midpoint_corner_vertices,
                output_fileformat,
                top_midpoint_z_by_name,
            )
        if (
            bottom_midpoint_corner_vertices is not None
            or bottom_midpoint_z_by_name
        ):
            bottom_surfaces = _surface_polygons_with_midpoint_z(
                bottom_surfaces,
                W,
                E,
                N,
                S,
                bottom_midpoint_corner_vertices,
                output_fileformat,
                bottom_midpoint_z_by_name,
            )

        previous_borders = self.borders
        previous_surface_borders = self.surfacePolygonBorders
        self.topSurfacePolygons = top_surfaces
        self.bottomSurfacePolygons = bottom_surfaces
        exterior_walls = (
            rebuild_nudged_surface_polygon_borders(
                self.topSurfacePolygons,
                self.bottomSurfacePolygons,
                previous_borders,
                previous_surface_borders,
                top_split_footprint,
                include_normal_cut_edges=False,
                W=W,
                E=E,
                N=N,
                S=S,
                output_fileformat=output_fileformat,
                side_cut_wall_sides=side_cut_wall_sides,
            )
            or []
        )
        contact_corner_set = set(contact_corners)
        cut_segments = {
            ("W", IntermediateCorner.SW): ("SW", "Wmid"),
            ("W", IntermediateCorner.NW): ("Wmid", "NW"),
            ("E", IntermediateCorner.SE): ("SE", "Emid"),
            ("E", IntermediateCorner.NE): ("Emid", "NE"),
            ("S", IntermediateCorner.SW): ("SW", "Smid"),
            ("S", IntermediateCorner.SE): ("Smid", "SE"),
            ("N", IntermediateCorner.NW): ("Nmid", "NW"),
            ("N", IntermediateCorner.NE): ("NE", "Nmid"),
        }
        requested_cut_footprints: set[XYEdge] = set()
        def normalized_point_xy(
            point: tuple[float, float],
        ) -> tuple[float, float]:
            return normalize_vertex_to_match_mesh_serialization(
                (point[0], point[1], 0.0),
                output_fileformat,
            )[:2]

        for (side, corner), (start_name, end_name) in cut_segments.items():
            if side not in split_sides or corner not in contact_corner_set:
                continue
            requested_cut_footprints.add(
                edge_xy_signature(
                    normalized_point_xy(points[start_name]),
                    normalized_point_xy(points[end_name]),
                ),
            )

        cut_walls = (
            _rebuild_matching_surface_polygon_borders(
                self.topSurfacePolygons,
                self.bottomSurfacePolygons,
                lambda footprint: footprint in requested_cut_footprints,
                output_fileformat,
            )
            if include_contact_cut_walls and requested_cut_footprints
            else []
        )
        self.surfacePolygonBorders = (
            exterior_walls + (cut_walls or [])
        ) or None
        self.borders = _empty_borders()
        return True

    def apply_positive_z_normal_nudge(
        self,
        affected_corners: Sequence[IntermediateCorner],
        W: float,
        E: float,
        N: float,
        S: float,
        split_rotation: int,
        output_fileformat: str,
        top_midpoint_corner_vertices: (
            dict[IntermediateCorner, vertex] | None
        ) = None,
        top_midpoint_z_by_name: dict[str, float] | None = None,
        difference_footprint: shapely.Geometry | None = None,
    ) -> bool:
        """Split a normal cell into positive-Z keep and complement pieces."""
        if not affected_corners or len(affected_corners) == 4:
            return False

        keep_footprint = nudge_keep_footprint(
            affected_corners,
            W,
            E,
            N,
            S,
        )
        if keep_footprint is None:
            return False

        full_footprint = full_cell_footprint(W, E, N, S)
        current_footprint = _current_surface_footprint(
            self.topSurfacePolygons,
            full_footprint,
            self.topquad,
        )
        pieces = [
            current_footprint.intersection(keep_footprint),
            current_footprint.difference(keep_footprint),
        ]
        if difference_footprint is not None and not difference_footprint.is_empty:
            aligned_difference_footprint = current_footprint.intersection(
                difference_footprint,
            )
            if not aligned_difference_footprint.is_empty:
                shared_boundary_pieces = (
                    _polygonized_regions_with_shared_boundaries(
                        current_footprint,
                        [
                            aligned_difference_footprint.intersection(
                                keep_footprint,
                            ),
                            aligned_difference_footprint.difference(
                                keep_footprint,
                            ),
                            current_footprint.difference(
                                aligned_difference_footprint,
                            ),
                            current_footprint.intersection(
                                keep_footprint,
                            ),
                            current_footprint.difference(
                                keep_footprint,
                            ),
                        ],
                    )
                )
                if shared_boundary_pieces:
                    pieces = shared_boundary_pieces
        top_planes = _surface_planes_from_current_geometry(
            self.topquad,
            self.topSurfacePolygons,
            split_rotation,
        )
        bottom_planes = _surface_planes_from_current_geometry(
            self.bottomquad,
            self.bottomSurfacePolygons,
            split_rotation,
        )
        if not top_planes or not bottom_planes:
            return False

        new_top_polygons: list[shapely.Polygon] = []
        new_bottom_polygons: list[shapely.Polygon] = []
        for piece in pieces:
            new_top_polygons.extend(
                _triangulate_2d_geometry_to_3d_polygons(
                    piece,
                    top_planes,
                    exterior_cw=False,
                    output_fileformat=output_fileformat,
                )
            )
            new_bottom_polygons.extend(
                _triangulate_2d_geometry_to_3d_polygons(
                    piece,
                    bottom_planes,
                    exterior_cw=True,
                    output_fileformat=output_fileformat,
                )
            )

        if top_midpoint_corner_vertices is not None or top_midpoint_z_by_name:
            new_top_polygons = _surface_polygons_with_midpoint_z(
                new_top_polygons,
                W,
                E,
                N,
                S,
                top_midpoint_corner_vertices,
                output_fileformat,
                top_midpoint_z_by_name,
            )

        if not new_top_polygons or not new_bottom_polygons:
            self._clear_geometry()
            return True

        previous_borders = self.borders
        previous_surface_borders = self.surfacePolygonBorders
        self.topSurfacePolygons = new_top_polygons
        self.bottomSurfacePolygons = new_bottom_polygons
        self.surfacePolygonBorders = (
            rebuild_nudged_surface_polygon_borders(
                self.topSurfacePolygons,
                self.bottomSurfacePolygons,
                previous_borders,
                previous_surface_borders,
                current_footprint,
                include_normal_cut_edges=False,
                W=W,
                E=E,
                N=N,
                S=S,
                output_fileformat=output_fileformat,
            )
            or None
        )
        self.borders = _empty_borders()
        return True

    def apply_positive_z_difference_nudge(
        self,
        affected_corners: Sequence[IntermediateCorner],
        W: float,
        E: float,
        N: float,
        S: float,
        split_rotation: int,
        output_fileformat: str,
        top_midpoint_corner_vertices: (
            dict[IntermediateCorner, vertex] | None
        ) = None,
        top_midpoint_z_by_name: dict[str, float] | None = None,
        bottom_midpoint_corner_vertices: (
            dict[IntermediateCorner, vertex] | None
        ) = None,
        bottom_midpoint_z_by_name: dict[str, float] | None = None,
        side_cut_wall_sides: set[str] | None = None,
    ) -> bool:
        """Remove the positive-Z contact footprint from a difference cell."""
        if not affected_corners or len(affected_corners) == 4:
            return False

        keep_footprint = nudge_keep_footprint(
            affected_corners,
            W,
            E,
            N,
            S,
        )
        if keep_footprint is None:
            return False

        cell_footprint_2D = full_cell_footprint(W, E, N, S)
        current_footprint = _current_surface_footprint(
            self.topSurfacePolygons,
            cell_footprint_2D,
            self.topquad,
        )
        keep_geometry = current_footprint.intersection(keep_footprint)
        top_planes = _surface_planes_from_current_geometry(
            self.topquad,
            self.topSurfacePolygons,
            split_rotation,
        )
        bottom_planes = _surface_planes_from_current_geometry(
            self.bottomquad,
            self.bottomSurfacePolygons,
            split_rotation,
        )
        if not top_planes or not bottom_planes:
            return False

        new_top_surfaces = _triangulate_2d_geometry_to_3d_polygons(
            keep_geometry,
            top_planes,
            exterior_cw=False,
            output_fileformat=output_fileformat,
        )
        if (
            top_midpoint_corner_vertices is not None
            or top_midpoint_z_by_name
        ):
            new_top_surfaces = _surface_polygons_with_midpoint_z(
                new_top_surfaces,
                W,
                E,
                N,
                S,
                top_midpoint_corner_vertices,
                output_fileformat,
                top_midpoint_z_by_name,
            )
        if (
            self.bottomSurfacePolygons
            and bottom_midpoint_corner_vertices is None
            and not bottom_midpoint_z_by_name
        ):
            new_bottom_surfaces = _clip_3d_surface_polygons_to_2d_geometry(
                self.bottomSurfacePolygons,
                keep_geometry,
                exterior_cw=True,
                output_fileformat=output_fileformat,
            )
        else:
            bottom_fallback_z_by_xy = (
                _nudge_midpoint_z_by_xy(
                    W,
                    E,
                    N,
                    S,
                    bottom_midpoint_corner_vertices,
                    output_fileformat,
                    bottom_midpoint_z_by_name,
                )
                if bottom_midpoint_corner_vertices is not None
                or bottom_midpoint_z_by_name
                else None
            )
            new_bottom_surfaces = _triangulate_2d_geometry_to_3d_polygons(
                keep_geometry,
                bottom_planes,
                exterior_cw=True,
                output_fileformat=output_fileformat,
                fallback_z_by_xy=bottom_fallback_z_by_xy,
            )
        if (
            bottom_midpoint_corner_vertices is not None
            or bottom_midpoint_z_by_name
        ):
            new_bottom_surfaces = _surface_polygons_with_midpoint_z(
                new_bottom_surfaces,
                W,
                E,
                N,
                S,
                bottom_midpoint_corner_vertices,
                output_fileformat,
                bottom_midpoint_z_by_name,
            )
        if not new_top_surfaces or not new_bottom_surfaces:
            self._clear_geometry()
            return True

        previous_borders = self.borders
        previous_surface_borders = self.surfacePolygonBorders
        self.topSurfacePolygons = new_top_surfaces
        self.bottomSurfacePolygons = new_bottom_surfaces
        self.surfacePolygonBorders = (
            rebuild_nudged_surface_polygon_borders(
                self.topSurfacePolygons,
                self.bottomSurfacePolygons,
                previous_borders,
                previous_surface_borders,
                current_footprint,
                include_normal_cut_edges=True,
                W=W,
                E=E,
                N=N,
                S=S,
                output_fileformat=output_fileformat,
                side_cut_wall_sides=side_cut_wall_sides,
            )
            or None
        )
        self.borders = _empty_borders()
        return True

    def flip_bottom_positive_z_contact_edges(
        self,
        split_rotation: int,
        output_fileformat: str,
        allowed_edges: set[Edge3D] | None = None,
    ) -> bool:
        """Flip bottom triangulation for actual clipped positive-Z contacts."""
        if not self.bottomSurfacePolygons:
            return False

        top_edges = surface_mesh_edge_counts(
            self.top_surface_meshes(),
            split_rotation,
            output_fileformat,
        )
        bottom_edge_polygons: dict[Edge3D, list[int]] = {}
        normalized_bottom_coords: list[list[tuple[float, ...]]] = []
        for polygon_index, polygon in enumerate(self.bottomSurfacePolygons):
            coords = [
                normalize_vertex_to_match_mesh_serialization(
                    coord,
                    output_fileformat,
                )
                for coord in polygon.exterior.coords[:-1]
            ]
            normalized_bottom_coords.append(coords)
            for coord_index, coord0 in enumerate(coords):
                coord1 = coords[(coord_index + 1) % len(coords)]
                edge_key = edge_3d_signature(coord0, coord1)
                bottom_edge_polygons.setdefault(edge_key, []).append(
                    polygon_index,
                )

        replacements: dict[int, list[shapely.Polygon]] = {}
        consumed_polygons: set[int] = set()
        for edge_key, polygon_indices in bottom_edge_polygons.items():
            if allowed_edges is not None and edge_key not in allowed_edges:
                continue
            if len(polygon_indices) != 2:
                continue
            if polygon_indices[0] in consumed_polygons:
                continue
            if polygon_indices[1] in consumed_polygons:
                continue
            if edge_key[0][2] <= 0 or edge_key[1][2] <= 0:
                continue
            if top_edges.get(edge_key, 0) + len(polygon_indices) <= 2:
                continue

            edge_vertices = {edge_key[0], edge_key[1]}
            opposite_vertices: list[tuple[float, ...]] = []
            for polygon_index in polygon_indices:
                for coord in normalized_bottom_coords[polygon_index]:
                    if coord not in edge_vertices and coord not in opposite_vertices:
                        opposite_vertices.append(coord)
            if len(opposite_vertices) != 2:
                continue

            a, b = edge_key
            c, d = opposite_vertices
            new_polygons: list[shapely.Polygon] = []
            for triangle_coords in (
                [c, d, a, c],
                [d, c, b, d],
            ):
                triangle = shapely.orient_polygons(
                    shapely.Polygon(triangle_coords),
                    exterior_cw=True,
                )
                normalized_triangle = (
                    surface_polygon_normalized_to_match_mesh_serialization(
                        triangle,
                        output_fileformat,
                    )
                )
                if normalized_triangle is not None:
                    new_polygons.append(normalized_triangle)
            if len(new_polygons) != 2:
                continue

            replacements[polygon_indices[0]] = new_polygons
            consumed_polygons.update(polygon_indices)

        if not replacements:
            return False

        new_bottom_polygons: list[shapely.Polygon] = []
        for polygon_index, polygon in enumerate(self.bottomSurfacePolygons):
            if polygon_index in replacements:
                new_bottom_polygons.extend(replacements[polygon_index])
            elif polygon_index not in consumed_polygons:
                new_bottom_polygons.append(polygon)
        self.bottomSurfacePolygons = new_bottom_polygons
        return True

    def check_for_tri_cell(self) -> bool:
        """Return whether two adjacent walls allow triangular smoothing."""
        if self.is_tri_cell:
            return False
        border_sides = {
            direction
            for direction, border in self.borders.items()
            if border
        }

        if len(border_sides) != 2:
            return False
        if border_sides in ({"N", "S"}, {"E", "W"}):
            return False

        return True

    def convert_to_tri_cell(self) -> None:
        """Collapse a cell with two adjacent walls into a triangular cell."""
        if self.is_tri_cell:
            return
        if self.topquad is None or self.bottomquad is None:
            raise RuntimeError(
                "Triangular smoothing needs top and bottom quads."
            )

        b = self.borders
        tvl = self.topquad.vl
        bvl = self.bottomquad.vl

        # Keep one outer wall and replace it with the new diagonal wall.

        if b.get("N") and b.get("W"):
            self.topquad = quad(tvl[3], tvl[1], tvl[2], None) # ccw, order doesn't matter
            self.bottomquad = quad(bvl[1], bvl[2], bvl[3], None) # cw!
            b["N"] = quad(tvl[1], tvl[3], bvl[1], bvl[3]) # diagonal wall (ccw!)
            b.pop("W")
        elif b.get("N") and b.get("E"):
            self.topquad = quad(tvl[0], tvl[1], tvl[2], None)
            self.bottomquad = quad(bvl[0], bvl[2], bvl[3], None)
            b["N"] = quad(tvl[0], tvl[2], bvl[2], bvl[0])
            b.pop("E")
        elif b.get("S") and b.get("E"):
            self.topquad = quad(tvl[3], tvl[0], tvl[1], None)
            self.bottomquad = quad(bvl[3], bvl[0], bvl[1], None)
            b["S"] = quad(tvl[3], tvl[1], bvl[3], bvl[1])
            b.pop("E")
        elif b.get("S") and b.get("W"):
            self.topquad = quad(tvl[2], tvl[3], tvl[0], None)
            self.bottomquad = quad(bvl[0], bvl[1], bvl[2], None)
            b["S"] = quad(tvl[2], tvl[0], bvl[0], bvl[2])
            b.pop("W")
        else:
            raise RuntimeError(
                f"Invalid triangular smoothing walls: {self.borders}"
            )

        self.is_tri_cell = True

    def remove_zero_height_volumes(
        self,
        split_rotation: int,
        output_fileformat: str = "STLb",
        preserve_zero_height_xy: set[tuple[float, float]] | None = None,
        preserve_zero_height_edges: set[XYEdge] | None = None,
        serialized_vertices: SerializedVertexCache | None = None,
    ) -> None:
        """Remove zero-height cell geometry in place.

        This mutates quads, cardinal borders, clipped surface polygons, and
        clipped wall borders. Cardinal quads are compared as the triangles
        emitted for ``split_rotation``. Matching clipped top/bottom polygons
        are deleted, then ``surfacePolygonBorders`` is filtered or rebuilt so
        only walls still supported by both remaining clipped-surface boundaries
        are kept.
        """
        # Step 1: compare at serialized mesh precision, not raw float
        # precision, so cleanup matches the STL/OBJ vertices that are written.
        if serialized_vertices is None:
            serialized_vertices = {}

        def output_signature(coord: Coordinate) -> tuple[float, ...]:
            return _serialized_vertex_from_cache(
                coord,
                output_fileformat,
                serialized_vertices,
            )

        def ordinary_quad_zero_height_corner_count() -> int | None:
            """Return matching corner count for simple top/bottom quads."""
            if self.topSurfacePolygons or self.bottomSurfacePolygons:
                return None
            if self.topquad is None or self.bottomquad is None:
                return None
            top_vertices = self.topquad.vl
            bottom_vertices = self.bottomquad.vl
            if top_vertices[3] is None or bottom_vertices[3] is None:
                return None

            corner_pairs = (
                (top_vertices[0], bottom_vertices[0]),  # NW
                (top_vertices[1], bottom_vertices[3]),  # SW
                (top_vertices[2], bottom_vertices[2]),  # SE
                (top_vertices[3], bottom_vertices[1]),  # NE
            )

            def vertices_match(
                top_vertex: vertex,
                bottom_vertex: vertex,
            ) -> bool:
                top_coords = top_vertex.coords
                bottom_coords = bottom_vertex.coords
                if top_coords[:2] != bottom_coords[:2]:
                    return (
                        output_signature(top_coords)
                        == output_signature(bottom_coords)
                    )
                if top_coords[2] == bottom_coords[2]:
                    return True
                return (
                    normalize_coordinate_to_match_mesh_serialization(
                        top_coords[2],
                        output_fileformat,
                    )
                    == normalize_coordinate_to_match_mesh_serialization(
                        bottom_coords[2],
                        output_fileformat,
                    )
                )

            matching_corners = 0
            remaining_corners = len(corner_pairs)
            for top_vertex, bottom_vertex in corner_pairs:
                remaining_corners -= 1
                if vertices_match(top_vertex, bottom_vertex):
                    matching_corners += 1
                if matching_corners + remaining_corners < 3:
                    break
            return matching_corners

        zero_height_corner_count = ordinary_quad_zero_height_corner_count()
        if zero_height_corner_count is not None and zero_height_corner_count < 3:
            return

        protected_zero_height_xy = {
            output_signature((xy[0], xy[1], 0.0))[:2]
            for xy in (preserve_zero_height_xy or set())
        }

        def surface_border_footprint(
            surface_border: quad,
        ) -> shapely.LineString | None:
            # Collapse a clipped wall quad/tri to its two unique XY endpoints.
            xy_coords: list[tuple[float, float]] = []
            for v in surface_border.vl:
                if v is None:
                    continue
                output_coord = output_signature(v.coords)
                xy = (output_coord[0], output_coord[1])
                if xy not in xy_coords:
                    xy_coords.append(xy)
            if len(xy_coords) != 2:
                return None
            return shapely.LineString(xy_coords)

        def edge_xy_signature(
            coord0: Coordinate,
            coord1: Coordinate,
        ) -> XYEdge:
            # Normalize XY edge direction so dictionary lookup is stable.
            return tuple(
                sorted(
                    (
                        output_signature(coord0)[:2],
                        output_signature(coord1)[:2],
                    )
                )
            )

        protected_zero_height_edges = {
            edge_xy_signature(
                (edge[0][0], edge[0][1], 0.0),
                (edge[1][0], edge[1][1], 0.0),
            )
            for edge in (preserve_zero_height_edges or set())
        }

        def polygon_edge_footprints(polygon: shapely.Polygon) -> set[XYEdge]:
            # Collect all XY boundary edges for a removed clipped polygon.
            footprints: set[XYEdge] = set()
            rings = [polygon.exterior, *polygon.interiors]
            for ring in rings:
                coords = list(ring.coords)
                for ci in range(len(coords) - 1):
                    footprints.add(
                        edge_xy_signature(coords[ci], coords[ci + 1])
                    )
            return footprints

        def polygon_has_protected_edge(polygon: shapely.Polygon) -> bool:
            """Return whether a zero-height polygon carries protected edge."""
            return bool(
                protected_zero_height_edges
                and (
                    polygon_edge_footprints(polygon)
                    & protected_zero_height_edges
                )
            )

        def polygon_has_protected_vertex(polygon: shapely.Polygon) -> bool:
            """Return whether a zero-height polygon carries protected XY."""
            if not protected_zero_height_xy:
                return False
            rings = [polygon.exterior, *polygon.interiors]
            for ring in rings:
                for coord in ring.coords:
                    if output_signature(coord)[:2] in protected_zero_height_xy:
                        return True
            return False

        # Step 2: cardinal quads are cleaned up using the triangles that will
        # actually be emitted for the active split_rotation.
        def triangle_signature(
            triangle: tuple[vertex, ...],
        ) -> tuple[tuple[float, ...], ...]:
            """Return a serialized-coordinate triangle signature."""
            # Sort vertices so opposite top/bottom winding still matches.
            return tuple(
                sorted(output_signature(v.coords) for v in triangle)
            )

        def triangle_boundary_footprints(
            triangles: list[tuple[vertex, ...]],
        ) -> set[XYEdge]:
            """Return serialized XY boundary footprints for triangles."""
            # Collect the serialized XY boundary edges still present.
            footprints: set[XYEdge] = set()
            for triangle in triangles:
                for vi, v0 in enumerate(triangle):
                    v1 = triangle[(vi + 1) % len(triangle)]
                    footprints.add(edge_xy_signature(v0.coords, v1.coords))
            return footprints

        def filter_cardinal_borders(
            top_triangles: list[tuple[vertex, ...]],
            bottom_triangles: list[tuple[vertex, ...]],
        ) -> None:
            """Keep cardinal walls still supported by both surfaces."""
            # After a cardinal surface triangle is removed, drop any N/S/E/W
            # wall whose footprint is no longer present on both surfaces.
            top_footprints = triangle_boundary_footprints(top_triangles)
            bottom_footprints = triangle_boundary_footprints(bottom_triangles)
            for direction, border in list(self.borders.items()):
                footprint = surface_border_footprint(border)
                if footprint is None:
                    self.borders.pop(direction)
                    continue
                footprint_key = edge_xy_signature(
                    footprint.coords[0],
                    footprint.coords[1],
                )
                if (
                    footprint_key not in top_footprints
                    or footprint_key not in bottom_footprints
                ):
                    self.borders.pop(direction)

        def remove_matching_cardinal_corners() -> bool:
            """Remove a zero-height corner when split diagonals differ."""
            # This preserves the old 3-corner cleanup for cases where top and
            # bottom choose different diagonals, so triangle signatures miss
            # the zero-height corner.
            if self.topquad is None or self.bottomquad is None:
                return False
            tvl = self.topquad.vl
            bvl = self.bottomquad.vl
            if tvl[3] is None or bvl[3] is None:
                return False

            # Corner index mapping by shared XY position:
            #     position: NW  SW  SE  NE
            #     top:       0   1   2   3
            #     bottom:    0   3   2   1
            corners_match = {
                "NW": output_signature(tvl[0].coords)
                == output_signature(bvl[0].coords),
                "SW": output_signature(tvl[1].coords)
                == output_signature(bvl[3].coords),
                "SE": output_signature(tvl[2].coords)
                == output_signature(bvl[2].coords),
                "NE": output_signature(tvl[3].coords)
                == output_signature(bvl[1].coords),
            }

            if all(corners_match.values()):
                self.topquad = None
                self.bottomquad = None
                self.borders.clear()
                return True

            if corners_match["NW"] and corners_match["NE"] and corners_match["SW"]:
                self.topquad = quad(tvl[3], tvl[1], tvl[2], None)
                self.bottomquad = quad(bvl[1], bvl[2], bvl[3], None)
                self.borders.pop("N", None)
                self.borders.pop("W", None)
                return True

            if corners_match["NW"] and corners_match["NE"] and corners_match["SE"]:
                self.topquad = quad(tvl[0], tvl[1], tvl[2], None)
                self.bottomquad = quad(bvl[0], bvl[2], bvl[3], None)
                self.borders.pop("N", None)
                self.borders.pop("E", None)
                return True

            if corners_match["NE"] and corners_match["SW"] and corners_match["SE"]:
                self.topquad = quad(tvl[3], tvl[0], tvl[1], None)
                self.bottomquad = quad(bvl[3], bvl[0], bvl[1], None)
                self.borders.pop("S", None)
                self.borders.pop("E", None)
                return True

            if corners_match["NW"] and corners_match["SW"] and corners_match["SE"]:
                self.topquad = quad(tvl[2], tvl[3], tvl[0], None)
                self.bottomquad = quad(bvl[0], bvl[1], bvl[2], None)
                self.borders.pop("S", None)
                self.borders.pop("W", None)
                return True

            return False

        def remove_matching_cardinal_triangles() -> None:
            """Remove zero-height cardinal triangles for the active split."""
            # Match top and bottom triangles by serialized coordinates. If
            # only one pair remains, keep that triangle and filter its walls.
            if self.topquad is None or self.bottomquad is None:
                return

            top_triangles = self.topquad.get_triangles(
                split_rotation=split_rotation,
            )
            bottom_triangles = self.bottomquad.get_triangles(
                split_rotation=split_rotation,
            )
            if len(top_triangles) != len(bottom_triangles):
                return

            bottom_signatures = [
                triangle_signature(triangle)
                for triangle in bottom_triangles
            ]
            matched_top_indexes: set[int] = set()
            matched_bottom_indexes: set[int] = set()

            # Match each top triangle to at most one serialized-equal bottom.
            for top_index, top_triangle in enumerate(top_triangles):
                top_signature = triangle_signature(top_triangle)
                for bottom_index, bottom_signature in enumerate(
                    bottom_signatures,
                ):
                    if bottom_index in matched_bottom_indexes:
                        continue
                    if top_signature != bottom_signature:
                        continue
                    matched_top_indexes.add(top_index)
                    matched_bottom_indexes.add(bottom_index)
                    break

            if not matched_top_indexes:
                if (
                    self.topquad.get_split_edge_indices(split_rotation)
                    != self.bottomquad.get_split_edge_indices(split_rotation)
                ):
                    remove_matching_cardinal_corners()
                return

            remaining_top_triangles = [
                triangle
                for index, triangle in enumerate(top_triangles)
                if index not in matched_top_indexes
            ]
            remaining_bottom_triangles = [
                triangle
                for index, triangle in enumerate(bottom_triangles)
                if index not in matched_bottom_indexes
            ]

            if not remaining_top_triangles and not remaining_bottom_triangles:
                self.topquad = None
                self.bottomquad = None
                self.borders.clear()
                return

            if (
                len(remaining_top_triangles) == 1
                and len(remaining_bottom_triangles) == 1
            ):
                top_triangle = remaining_top_triangles[0]
                bottom_triangle = remaining_bottom_triangles[0]
                self.topquad = quad(
                    top_triangle[0],
                    top_triangle[1],
                    top_triangle[2],
                    None,
                )
                self.bottomquad = quad(
                    bottom_triangle[0],
                    bottom_triangle[1],
                    bottom_triangle[2],
                    None,
                )
                filter_cardinal_borders(
                    remaining_top_triangles,
                    remaining_bottom_triangles,
                )

        # Run cardinal cleanup before clipped cleanup; unclipped cells use
        # topquad, bottomquad, and the N/S/E/W wall dictionary.
        remove_matching_cardinal_triangles()

        # Step 3: clipped cells already have explicit triangulated surfaces,
        # so remove exact matching top/bottom polygons directly.
        removed_surface_polygon = False
        removed_surface_edge_footprints: set[XYEdge] = set()
        if self.topSurfacePolygons and self.bottomSurfacePolygons:
            ti = 0
            while ti < len(self.topSurfacePolygons):
                match = False
                bi = 0
                while bi < len(self.bottomSurfacePolygons):
                    top_surface_polygon = self.topSurfacePolygons[ti]
                    bottom_surface_polygon = self.bottomSurfacePolygons[bi]
                    normalized_top_surface = (
                        surface_polygon_normalized_to_match_mesh_serialization(
                            top_surface_polygon,
                            output_fileformat,
                            serialized_vertices=serialized_vertices,
                        )
                    )
                    normalized_bottom_surface = (
                        surface_polygon_normalized_to_match_mesh_serialization(
                            bottom_surface_polygon,
                            output_fileformat,
                            serialized_vertices=serialized_vertices,
                        )
                    )
                    if (
                        normalized_top_surface is not None
                        and normalized_bottom_surface is not None
                        and polygons_equal_3d(
                            normalized_top_surface,
                            normalized_bottom_surface,
                        )
                    ):
                        if polygon_has_protected_vertex(
                            top_surface_polygon,
                        ) or polygon_has_protected_vertex(
                            bottom_surface_polygon,
                        ) or polygon_has_protected_edge(
                            top_surface_polygon,
                        ) or polygon_has_protected_edge(
                            bottom_surface_polygon,
                        ):
                            bi += 1
                            continue
                        removed_surface_edge_footprints.update(
                            polygon_edge_footprints(top_surface_polygon)
                        )
                        removed_surface_edge_footprints.update(
                            polygon_edge_footprints(bottom_surface_polygon)
                        )
                        del self.topSurfacePolygons[ti]
                        del self.bottomSurfacePolygons[bi]
                        removed_surface_polygon = True
                        match = True
                        break
                    bi += 1
                if not match:
                    ti += 1

        if removed_surface_polygon:
            # Step 4: after clipped surface removal, split final top/bottom
            # boundary edges at each other's wall vertices, then rebuild walls
            # from exact serialized XY matches.
            requested_line_parts: list[shapely.Geometry] = [
                shapely.LineString(footprint)
                for footprint in removed_surface_edge_footprints
            ]
            requested_clipped_lines = _surface_wall_requested_lines(
                self.surfacePolygonBorders,
                output_fileformat=output_fileformat,
            )
            if requested_clipped_lines is not None:
                requested_line_parts.append(requested_clipped_lines)
            requested_lines = (
                shapely.union_all(requested_line_parts)
                if requested_line_parts
                else None
            )

            self.surfacePolygonBorders = (
                _rebuild_matching_surface_polygon_borders(
                    self.topSurfacePolygons,
                    self.bottomSurfacePolygons,
                    lambda footprint: _linework_covers_footprint(
                        requested_lines,
                        footprint,
                    ),
                    output_fileformat,
                )
                or None
            )

class ProcessingTile:
    """Raster variants and output state needed to process one mesh tile."""

    __slots__ = (
        "tile_info",
        "top_raster_variants",
        "bottom_raster_variants",
        "bottom_surface_provider",
        "positive_contact_top_raster",
        "positive_z_nudge_plan",
        "return_grid",
        "defer_triangle_writes",
        "defer_serialization_cleanup",
    )

    tile_info: TouchTerrainTileInfo
    top_raster_variants: RasterVariants
    bottom_raster_variants: Union[None, RasterVariants]
    bottom_surface_provider: BottomSurfaceProvider | None
    positive_contact_top_raster: np.ndarray | None
    positive_z_nudge_plan: PositiveZNudgePlan | None
    return_grid: bool
    defer_triangle_writes: bool
    defer_serialization_cleanup: bool

    def __init__(
        self,
        tile_info: TouchTerrainTileInfo,
        top: RasterVariants,
        bottom: Union[None, RasterVariants],
        bottom_surface_provider: BottomSurfaceProvider | None = None,
        positive_contact_top_raster: np.ndarray | None = None,
        positive_z_nudge_plan: PositiveZNudgePlan | None = None,
        return_grid: bool = False,
        defer_triangle_writes: bool = False,
        defer_serialization_cleanup: bool = False,
    ):
        self.tile_info = tile_info
        self.top_raster_variants = top
        self.bottom_raster_variants = bottom
        self.bottom_surface_provider = bottom_surface_provider
        self.positive_contact_top_raster = positive_contact_top_raster
        self.positive_z_nudge_plan = positive_z_nudge_plan
        self.return_grid = return_grid
        self.defer_triangle_writes = defer_triangle_writes
        self.defer_serialization_cleanup = defer_serialization_cleanup

def _requested_cardinal_borders(
    padded_row: int,
    padded_col: int,
    ymaxidx: int,
    xmaxidx: int,
    borders_top_raster: np.ndarray,
    check_nan_neighbors: bool = True,
) -> set[str]:
    """Return N/S/E/W sides that need exterior walls for this cell."""
    borders: set[str] = set()
    if padded_row == 1:
        borders.add("N")
    if padded_row == ymaxidx:
        borders.add("S")
    if padded_col == 1:
        borders.add("W")
    if padded_col == xmaxidx:
        borders.add("E")

    if not check_nan_neighbors:
        return borders

    if np.isnan(borders_top_raster[padded_row - 1, padded_col]):
        borders.add("N")
    if np.isnan(borders_top_raster[padded_row + 1, padded_col]):
        borders.add("S")
    if np.isnan(borders_top_raster[padded_row, padded_col - 1]):
        borders.add("W")
    if np.isnan(borders_top_raster[padded_row, padded_col + 1]):
        borders.add("E")

    return borders


def _wall_border_edges_from_buckets(buckets: Any) -> list[BorderEdge]:
    """Return the clipping edges in a cell bucket that request walls."""
    if not isinstance(buckets, dict):
        return []
    return [
        border_edge
        for bucket in buckets.values()
        if isinstance(bucket, list)
        for border_edge in bucket
        if isinstance(border_edge, BorderEdge) and border_edge.make_wall
    ]


def _surface_polygon_edges(
    surface_polygons: Sequence[shapely.Polygon],
) -> list[shapely.LineString]:
    """Return the individual boundary lines from surface polygons."""
    return [
        geometry
        for surface_polygon in surface_polygons
        for geometry in flatten_geometries(
            geometries=[surface_polygon],
            to_single_lines=True,
        )
        if isinstance(geometry, shapely.LineString)
    ]


class grid:
    """makes cell data structure from two np arrays (top, bottom) of the same shape."""
    tile: ProcessingTile
    tile_info: TouchTerrainTileInfo
    bottom_thru_base: bool
    cells: np.ndarray | None
    positive_z_nudge_plan: PositiveZNudgePlan
    xmaxidx: int
    ymaxidx: int
    cell_size: float
    offsetx: float
    offsety: float
    num_triangles: int

    def __init__(self, tile: ProcessingTile):
        '''tile: Includes Top and Bottom raster variants and tile_info dict
        '''
        self.tile = tile
        self.tile_info = tile.tile_info

        self.bottom_thru_base = tile.tile_info.config.bottom_thru_base

        if self.tile_info.config.fileformat == "obj":
            vertex.vertex_index_dict = {}  # will be filled with vertex indices
        else:
            vertex.vertex_index_dict = -1

        self.cells = None
        self.positive_z_nudge_plan: PositiveZNudgePlan = (
            tile.positive_z_nudge_plan or {}
        )

        if tile.top_raster_variants.dilated is None:
            print("grid.init() error: No prepared top raster. tile.top_raster_variants.dilated is None")
            return None

        if tile.bottom_raster_variants is None:
            print("grid.init() error: No bottom_raster_variants passed in. bottom_raster_variants required to be passed in even if the variants are None.") # We may set bottom_raster_variants to None later to signal that we are not in "difference mesh" mode.
            return None

        # Important: in 2D np arrays, x and y coordinate are "flipped" in the sense that when printing top
        # top[0,0] appears to the upper left (NW) corner and [0,1] (East) of it:
        #[[11  12 13]       top[0,1] => 12
        # [Nan 22 23]       top[2,0] => NaN (Not a Number -> undefined elevation)
        # [31  32 33]       top[2,1] => 32
        # [41  42 43]]
        # Note: the actual array will be edge-padded which is important to be able to interpolate the border cells


        # cell size (x and y delta)
        self.cell_size = self.tile_info.pixel_mm

        # does top have NaNs?
        self.tile_info.have_nan = np.any(np.isnan(tile.top_raster_variants.dilated)) # True => we have NaN values

        bottom_dilated = tile.bottom_raster_variants.dilated
        self.tile_info.have_bot_nan = (
            isinstance(bottom_dilated, np.ndarray)
            and np.any(np.isnan(bottom_dilated))
        )

        # A missing prepared bottom means normal-mesh mode.
        if not isinstance(bottom_dilated, np.ndarray):
            tile.bottom_raster_variants = None
        # can't have a bottom_image and NaNs in top
        elif (
            self.tile_info.config.bottom_image is not None
            and self.tile_info.have_nan
        ):
            tile.bottom_raster_variants = None
            print("Top has NaN values, requested bottom image will be ignored!")

        # need to use the tilewide min/max for each tile, otherwise the boudaries don't line up perfectly!

        #
        # Convert elevation from real word elevation (m) to model print3D height (mm)
        #
        if self.tile_info.config.use_geo_coords is None: # Coordinates need to be in mm

            scz = 1 / self.tile_info.scale * 1000.0 # scale z to mm
            scale_min_elev = self.tile_info.config.min_elev

            if tile.bottom_raster_variants is not None: # Top-Bottom difference mesh mode
                if not self.bottom_thru_base:  # normal case,
                    tile.bottom_raster_variants -= scale_min_elev

                    tile.bottom_raster_variants *= scz * self.tile_info.config.zscale # apply z-scale to bottom
                    tile.bottom_raster_variants += self.tile_info.config.basethick # add base thickness to bottom

                    if tile.bottom_raster_variants.dilated is not None:
                        # Update with per-tile mm min/max
                        self.tile_info.min_bot_elev = np.nanmin(tile.bottom_raster_variants.dilated)
                        self.tile_info.max_bot_elev = np.nanmax(tile.bottom_raster_variants.dilated)
                        print("bottom min/max (mm) for tile:", self.tile_info.min_bot_elev, self.tile_info.max_bot_elev)
                    else:
                        print("tile.bottom_raster_variants.dilated not found")
                        return None
            tile.top_raster_variants -= scale_min_elev
            tile.top_raster_variants *= scz * self.tile_info.config.zscale # apply z-scale to top
            tile.top_raster_variants += self.tile_info.config.basethick # add base thickness to top

            if tile.positive_contact_top_raster is not None:
                positive_contact_top_raster = (
                    tile.positive_contact_top_raster.copy().astype(np.float64)
                )
                positive_contact_top_raster -= scale_min_elev
                positive_contact_top_raster *= scz * self.tile_info.config.zscale
                positive_contact_top_raster += self.tile_info.config.basethick
                tile.positive_contact_top_raster = positive_contact_top_raster

            # post-scale (i.e. in mm) top elevations (for this tile)
            self.tile_info.config.min_elev = np.nanmin(tile.top_raster_variants.dilated)
            self.tile_info.max_elev = np.nanmax(tile.top_raster_variants.dilated)
            print("top min/max for tile (mm):", self.tile_info.config.min_elev, self.tile_info.max_elev)

        else:  # using geo coords (UTM, meter based) - thickness is meters
            # TODO: Just noticed that we don't apply a z-scale to the top. Not sure if we should
            tile.bottom_raster_variants.dilated = self.tile_info.config.min_elev - self.tile_info.config.basethick * 10
            logger.info("Using geo coords with a base thickness of " + str(self.tile_info.config.basethick * 10) + " meters")

        # After this point, all values are in real print3D units (mm) and 0 is the bottom.

        # max index in x and y for "inner" raster
        self.xmaxidx = tile.top_raster_variants.dilated.shape[1]-2
        self.ymaxidx = tile.top_raster_variants.dilated.shape[0]-2
        # offset so that 0/0 is the center of this tile (local) or so that 0/0 is the lower left corner of all tiles (global)
        if not self.tile_info.config.tile_centered: # global offset, best for looking at all tiles together
            self.offsetx = -self.tile_info.tile_width  * (self.tile_info.tile_no_x-1)  # tile_no starts with 1! This is the top end of the tile, not 0!
            self.offsety = -self.tile_info.tile_height * (self.tile_info.tile_no_y-1)  + self.tile_info.tile_height * self.tile_info.config.ntilesy

        else: # local centered for printing
            self.offsetx = self.tile_info.tile_width / 2.0
            self.offsety = self.tile_info.tile_height / 2.0

        # geo coords are in meters (UTM). tile_centered is ignored for geo coords
        if self.tile_info.config.use_geo_coords is not None:

            geo_transform = self.tile_info.geo_transform
            self.cell_size = abs(geo_transform[1]) # rw pixel size of geotiff in m
            tile_width_m  = self.xmaxidx * self.cell_size # number of (unpadded) pixels of current tile
            tile_height_m = self.ymaxidx * self.cell_size

            # Place the tiles so that the center is at 0/0, which is what Blender GIS needs.
            if self.tile_info.config.use_geo_coords == "centered":

                self.offsetx = -tile_width_m  * (self.tile_info.tile_no_x-1)
                self.offsety = tile_height_m  * self.tile_info.config.ntilesy - tile_height_m * (self.tile_info.tile_no_y-1)

                # center by half the total size
                self.offsetx += (self.tile_info.full_raster_width * self.cell_size) / 2
                self.offsety -= (self.tile_info.full_raster_height * self.cell_size) / 2

                # correct for off-by-1 cells
                self.offsetx -= self.cell_size
                self.offsety += self.cell_size

            # size in meters but the UTM zone's origin is used, i.e. each vertex is in full
            # UTM coordinates. Not sure what CAD/modelling system uses that but if needed it's an option.
            else:  # "UTM"

                self.offsetx = -tile_width_m  * (self.tile_info.tile_no_x-1)
                self.offsety = -tile_height_m * (self.tile_info.tile_no_y-1)

                self.offsetx = -geo_transform[0] + self.offsetx # UTM x of upper left corner
                self.offsety =  geo_transform[3] + self.offsety # UTM y


        # put corner coordinates tile info dict (may later be needed for 2 bottom triangles)
        if not self.tile_info.config.tile_centered:
            self.tile_info.W = self.tile_info.tile_width  * (self.tile_info.tile_no_x-1)
            self.tile_info.E = self.tile_info.W + self.tile_info.tile_width
            tot_height = self.tile_info.tile_height * self.tile_info.config.ntilesy
            # y tiles index goes top(0) DOWN to bottom
            self.tile_info.N = tot_height - (self.tile_info.tile_height * (self.tile_info.tile_no_y-1))
            self.tile_info.S = self.tile_info.N - self.tile_info.tile_height
        else:
            self.tile_info.W = -self.tile_info.tile_width / 2
            self.tile_info.E =  self.tile_info.tile_width / 2
            self.tile_info.S = -self.tile_info.tile_height / 2
            self.tile_info.N =  self.tile_info.tile_height / 2

    def extract_emitted_top_bottom_surfaces(self) -> BottomSurfaceProvider:
        """Return each emitted top surface reoriented as bottom geometry."""
        if self.cells is None:
            raise RuntimeError("Cannot extract surfaces before cells exist.")

        def surfaces_for_rows(
            row_start: int,
            row_end: int,
        ) -> BottomSurfaceProvider:
            output: BottomSurfaceProvider = []
            for row_index in range(row_start, row_end):
                surface_row: list[EmittedBottomSurface] = []
                for current_cell in self.cells[row_index]:
                    if current_cell is None:
                        surface_row.append((None, None))
                    else:
                        surface_row.append(
                            current_cell.emitted_top_as_bottom_surfaces()
                        )
                output.append(surface_row)
            return output

        row_count = self.cells.shape[0]
        worker_count = single_job_parallel_workers(
            getattr(self.tile_info, "config", None),
            row_count,
        )
        if not _should_parallelize_rows(row_count, worker_count):
            return surfaces_for_rows(0, row_count)

        surfaces: BottomSurfaceProvider = []
        for surface_rows in _parallel_range_results(
            0,
            row_count,
            worker_count,
            surfaces_for_rows,
        ):
            surfaces.extend(surface_rows)
        return surfaces

    def extract_emitted_top_footprints(self) -> TopFootprintProvider:
        """Return each cell's current emitted top footprint."""
        if self.cells is None:
            raise RuntimeError("Cannot extract footprints before cells exist.")

        def footprints_for_rows(
            row_start: int,
            row_end: int,
        ) -> TopFootprintProvider:
            output: TopFootprintProvider = []
            for row_index in range(row_start, row_end):
                footprint_row: list[TopFootprintSource | None] = []
                for col_index, current_cell in enumerate(
                    self.cells[row_index],
                ):
                    if current_cell is None:
                        footprint_row.append(None)
                        continue

                    W, E, N, S = self.cell_bounds(
                        row_index + 1,
                        col_index + 1,
                    )
                    full_footprint = full_cell_footprint(W, E, N, S)
                    footprint = _current_surface_footprint(
                        current_cell.topSurfacePolygons,
                        full_footprint,
                        current_cell.topquad,
                    )
                    has_surface_planes = bool(
                        _surface_planes_from_current_geometry(
                            current_cell.topquad,
                            current_cell.topSurfacePolygons,
                            self.tile_info.config.split_rotation,
                        )
                    )
                    if footprint.is_empty or not has_surface_planes:
                        footprint_row.append(None)
                    else:
                        footprint_row.append(footprint)
                output.append(footprint_row)
            return output

        row_count = self.cells.shape[0]
        worker_count = single_job_parallel_workers(
            getattr(self.tile_info, "config", None),
            row_count,
        )
        if not _should_parallelize_rows(row_count, worker_count):
            return footprints_for_rows(0, row_count)

        footprints: TopFootprintProvider = []
        for footprint_rows in _parallel_range_results(
            0,
            row_count,
            worker_count,
            footprints_for_rows,
        ):
            footprints.extend(footprint_rows)
        return footprints

    def cell_bounds(
        self,
        padded_row: int,
        padded_col: int,
    ) -> tuple[float, float, float, float]:
        """Return W/E/N/S coordinates for a padded raster cell location."""
        W = (padded_col - 1) * self.cell_size - self.offsetx
        E = W + self.cell_size
        N = -((padded_row - 1) * self.cell_size) + self.offsety
        S = N - self.cell_size
        return W, E, N, S

    def _can_skip_simple_serialization_cleanup(
        self,
        using_difference_mesh: bool,
        nudge_enabled: bool,
    ) -> bool:
        """Return whether ordinary rectangular cells can skip cleanup."""
        return (
            not using_difference_mesh
            and not nudge_enabled
            and not self.tile_info.have_nan
            and self.tile.bottom_surface_provider is None
            and self.tile.top_raster_variants.polygon_intersection_geometry
            is None
            and self.tile.top_raster_variants.polygon_intersection_edge_buckets
            is None
            and (
                self.tile.top_raster_variants
                .polygon_intersection_contains_properly
            )
            is None
            and self.tile_info.config.bottom_image is None
        )

    def _cell_xy_survives_mesh_serialization(
        self,
        W: float,
        E: float,
        N: float,
        S: float,
        output_fileformat: str,
    ) -> bool:
        """Return whether the rectangular cell footprint has output area."""
        output_w = normalize_coordinate_to_match_mesh_serialization(
            W,
            output_fileformat,
        )
        output_e = normalize_coordinate_to_match_mesh_serialization(
            E,
            output_fileformat,
        )
        output_n = normalize_coordinate_to_match_mesh_serialization(
            N,
            output_fileformat,
        )
        output_s = normalize_coordinate_to_match_mesh_serialization(
            S,
            output_fileformat,
        )
        return output_w != output_e and output_n != output_s

    def apply_positive_z_plan_to_existing_cells(
        self,
        positive_z_nudge_plan: PositiveZNudgePlan,
        positive_contact_top_raster: np.ndarray | None = None,
        positive_z_difference_top_footprints: (
            TopFootprintProvider | None
        ) = None,
    ) -> None:
        """Apply a confirmed positive-Z plan to already-created cells."""
        if not positive_z_nudge_plan or self.cells is None:
            return

        output_fileformat = self.tile_info.config.fileformat
        split_rotation = self.tile_info.config.split_rotation
        is_difference_mesh = self.tile.bottom_raster_variants is not None
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
            current_cell = self.cells[row, col]
            if current_cell is None:
                return

            padded_row = row + 1
            padded_col = col + 1
            W, E, N, S = self.cell_bounds(padded_row, padded_col)
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

        row_count = self.cells.shape[0]
        col_count = self.cells.shape[1]
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
                    self.tile_info.config,
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
                    self.tile_info.config,
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
            self._add_unmatched_cardinal_surface_walls(
                positive_z_nudge_plan=positive_z_nudge_plan,
                split_rotation=split_rotation,
                output_fileformat=output_fileformat,
            )
            self._close_local_positive_z_boundary_edge_loops(
                positive_z_nudge_plan=positive_z_nudge_plan,
                split_rotation=split_rotation,
                output_fileformat=output_fileformat,
            )
            self._add_unmatched_cardinal_surface_walls(
                positive_z_nudge_plan=positive_z_nudge_plan,
                split_rotation=split_rotation,
                output_fileformat=output_fileformat,
            )

    def _add_unmatched_cardinal_surface_walls(
        self,
        positive_z_nudge_plan: PositiveZNudgePlan,
        split_rotation: int,
        output_fileformat: str,
    ) -> None:
        """Add missing walls where repaired cardinal surface edges are open."""
        if self.cells is None:
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
                for col in range(self.cells.shape[1]):
                    current_cell = self.cells[row, col]
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
            self.tile_info.config,
            self.cells.shape[0],
        )
        if not _should_parallelize_rows(self.cells.shape[0], worker_count):
            merge_row_counts(edge_counts_for_rows(0, self.cells.shape[0]))
        else:
            for row_counts in _parallel_range_results(
                0,
                self.cells.shape[0],
                worker_count,
                edge_counts_for_rows,
            ):
                merge_row_counts(row_counts)

        for row in range(self.cells.shape[0]):
            for col in range(self.cells.shape[1]):
                current_cell = self.cells[row, col]
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

                W, E, N, S = self.cell_bounds(row + 1, col + 1)
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
        self,
        positive_z_nudge_plan: PositiveZNudgePlan,
        split_rotation: int,
        output_fileformat: str,
    ) -> None:
        """Close small positive-Z loops using local directed boundary order."""
        if self.cells is None:
            return

        target_locations = (
            (row + 1, col + 1)
            for row in range(self.cells.shape[0])
            for col in range(self.cells.shape[1])
        )

        def edge_usage_for_rows(
            row_start: int,
            row_end: int,
        ) -> tuple[dict[Edge3D, int], dict[DirectedEdge3D, int]]:
            row_edge_counts: dict[Edge3D, int] = {}
            row_directed_counts: dict[DirectedEdge3D, int] = {}
            for row in range(row_start, row_end):
                for col in range(self.cells.shape[1]):
                    current_cell = self.cells[row, col]
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
        tile_info = getattr(self, "tile_info", None)
        worker_count = single_job_parallel_workers(
            getattr(tile_info, "config", None),
            self.cells.shape[0],
        )
        if not _should_parallelize_rows(self.cells.shape[0], worker_count):
            merge_edge_usage(edge_usage_for_rows(0, self.cells.shape[0]))
        else:
            for row_usage in _parallel_range_results(
                0,
                self.cells.shape[0],
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
                or row >= self.cells.shape[0]
                or col >= self.cells.shape[1]
            ):
                continue

            current_cell = self.cells[row, col]
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
        for current_cell in self.cells.flat:
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

    def create_cells(self):
        '''Creates a data structure for each raster cell based on quads for top, any walls and possible bottom.
        Once created, each cell is converted into triangles for each file format, which are stored as a stream buffer (self.s)
        If using temp files, this buffer serves as a cache for occasionally writing to disk (self.fo)
        Note that for obj, two streams/files are needed, one for indices that define the vertices for each triangle and one
        for vertex coordinates. Here, only the index part (s[1] and fo[1]) is stored, the vertex coordinates will be
        created and stored later based on the keys of the vertex class attribute vertex_index_dict'''
        if self.tile_info is None:
            print("create_cells: Error: self.tile_info is None")
            return

        # Cells that are not emitted remain explicit None entries.
        self.cells = np.full(
            (self.ymaxidx, self.xmaxidx),
            None,
            dtype=object,
        )

        # report progress in %
        percent = 10
        pc_step = int(self.ymaxidx/percent) + 1
        progress = 0
        print("creating internal triangle data structure for", multiprocessing.current_process(), file=sys.stderr)
        output_fileformat = self.tile_info.config.fileformat
        split_rotation = self.tile_info.config.split_rotation
        top_variants = self.tile.top_raster_variants
        bottom_variants = self.tile.bottom_raster_variants
        using_difference_mesh = bottom_variants is not None
        top_dilated = top_variants.dilated
        top: np.ndarray | None = top_dilated
        if using_difference_mesh and self.tile_info.config.bottom_thru_base:
            top = bottom_variants.nan_close
        borders_top_raster: np.ndarray | None = top_dilated
        if using_difference_mesh and self.bottom_thru_base:
            borders_top_raster = top_variants.nan_close
        polygon_contains_properly = (
            top_variants.polygon_intersection_contains_properly
        )
        polygon_intersection_geometry = (
            top_variants.polygon_intersection_geometry
        )
        polygon_edge_buckets = top_variants.polygon_intersection_edge_buckets
        nudge_enabled = (
            self.tile_info.config.nudge_in_overused_edges_vertex
            and not self.tile_info.config.no_bottom
        )
        skip_simple_serialization_cleanup = (
            self._can_skip_simple_serialization_cleanup(
                using_difference_mesh,
                nudge_enabled,
            )
        )
        emit_cell_bottom = (
            not self.tile_info.config.no_bottom
            and (
                self.tile_info.have_nan
                or using_difference_mesh
                or nudge_enabled
            )
        )

        if not self.tile_info.have_nan:
            top_interpolation_raster = top_dilated
        elif top_variants.edge_interpolation is not None:
            top_interpolation_raster = top_variants.edge_interpolation
        else:
            top_interpolation_raster = top_variants.original
        top_corner_elevations = _interpolated_corner_grid(
            top_interpolation_raster,
        )
        bottom_raster_for_z0_nudge: np.ndarray | None = None
        bottom_corner_elevations: np.ndarray | None = None
        if using_difference_mesh and not self.bottom_thru_base:
            if self.tile_info.have_bot_nan:
                bottom_raster_for_z0_nudge = (
                    bottom_variants.original
                )
            else:
                bottom_raster_for_z0_nudge = (
                    bottom_variants.dilated
                )
            bottom_corner_elevations = _interpolated_corner_grid(
                bottom_raster_for_z0_nudge,
            )

        positive_z_nudge_plan: PositiveZNudgePlan = (
            self.tile.positive_z_nudge_plan or {}
        )
        if (
            nudge_enabled
            and not positive_z_nudge_plan
            and not using_difference_mesh
            and self.tile.positive_contact_top_raster is not None
        ):
            positive_z_nudge_plan = _build_positive_z_nudge_plan(
                upper_raster=self.tile.positive_contact_top_raster,
                lower_raster=top_interpolation_raster,
                emit_raster=self.tile.positive_contact_top_raster,
                split_emit_raster=top_variants.dilated,
                cell_size=self.cell_size,
                offsetx=self.offsetx,
                offsety=self.offsety,
                split_rotation=split_rotation,
                ymaxidx=self.ymaxidx,
                xmaxidx=self.xmaxidx,
                zero_threshold=self.tile_info.config.basethick,
                output_fileformat=output_fileformat,
                parallel_workers=single_job_parallel_workers(
                    self.tile_info.config,
                    self.ymaxidx,
                ),
            )
        self.positive_z_nudge_plan = positive_z_nudge_plan
        positive_z_difference_neighbor_split_sides = (
            _positive_z_difference_neighbor_split_sides(positive_z_nudge_plan)
            if using_difference_mesh
            else {}
        )
        cell_size = self.cell_size
        offsetx = self.offsetx
        offsety = self.offsety
        empty_positive_z_record: PositiveZNudgeRecord = {}

        for j in range(1, self.ymaxidx+1):# y dimension for looping within the +1 padded raster
            cell_row = j - 1
            N = -(cell_row * cell_size) + offsety
            S = N - cell_size
            serialized_row_vertices: SerializedVertexCache = {}
            if j % pc_step == 0:
                progress += percent
                print(progress, "%", multiprocessing.current_process(), file=sys.stderr)

            for i in range(1, self.xmaxidx + 1):# x dim.
                cell_col = i - 1
                # A NaN center is outside the emitted raster footprint.
                if self.tile_info.have_nan and np.isnan(top[j, i]):
                    continue

                # XY cell bounds use the upper-left raster origin.
                W = cell_col * cell_size - offsetx
                E = W + cell_size
                top_elevations = _cell_corner_elevations(
                    top_corner_elevations,
                    i,
                    j,
                )
                if np.isnan(top_elevations).any():
                    continue
                if self.tile_info.have_nan:
                    # Restore the zero base after edge interpolation used a
                    # value just below basethick as its fill elevation.
                    top_elevations = _zero_elevations_below_threshold(
                        top_elevations,
                        self.tile_info.config.basethick,
                    )
                (
                    top_ne_elevation,
                    top_nw_elevation,
                    top_se_elevation,
                    top_sw_elevation,
                ) = top_elevations

                # This vertex order emits counterclockwise top triangles.
                topq = quad(
                    vertex(W, N, top_nw_elevation),
                    vertex(W, S, top_sw_elevation),
                    vertex(E, S, top_se_elevation),
                    vertex(E, N, top_ne_elevation),
                )

                top_bottom_surface_geometries_2D: list[shapely.Geometry] | None = None
                if (
                    polygon_contains_properly is not None
                    and polygon_intersection_geometry is not None
                    and not polygon_contains_properly[cell_row][cell_col]
                ):
                    top_bottom_surface_geometries_2D = (
                        polygon_intersection_geometry[cell_row][cell_col]
                    )

                if not using_difference_mesh or self.bottom_thru_base:
                    bottom_elevations = (0, 0, 0, 0)
                else:
                    if bottom_corner_elevations is None:
                        raise RuntimeError(
                            "Difference mesh corner elevations are missing."
                        )
                    bottom_elevations = _cell_corner_elevations(
                        bottom_corner_elevations,
                        i,
                        j,
                    )
                    if np.isnan(bottom_elevations).any():
                        continue
                    if self.tile_info.have_bot_nan:
                        bottom_elevations = (
                            _zero_elevations_below_threshold(
                                bottom_elevations,
                                self.tile_info.config.basethick,
                            )
                        )
                botq = None
                bottom_corner_vertices: dict[IntermediateCorner, vertex] | None = None

                if not skip_simple_serialization_cleanup:
                    # These vertices may become emitted bottom surfaces, walls,
                    # or source planes for clipped/nudged cells.
                    (
                        botq,
                        bottom_corner_vertices,
                    ) = _create_cell_bottom_geometry(
                        W,
                        E,
                        N,
                        S,
                        *bottom_elevations,
                        nudge_enabled,
                    )

                top_surface_polygons_triangulated_3D = None
                bottom_surface_polygons_triangulated_3D = None
                clipped_surfaces_collapsed_after_output = False
                if top_bottom_surface_geometries_2D is not None:
                    if botq is None:
                        raise RuntimeError(
                            "Clipped cell surface creation needs a bottom quad."
                        )
                    (
                        top_surface_polygons_triangulated_3D,
                        bottom_surface_polygons_triangulated_3D,
                        clipped_surfaces_collapsed_after_output,
                    ) = _clipped_cell_surface_polygons(
                        top_bottom_surface_geometries_2D,
                        topq,
                        botq,
                        split_rotation,
                        output_fileformat,
                    )

                z0_nudged_cell = False
                z0_used_bottom_provider = False
                z0_full_footprint_2D: shapely.Geometry | None = None
                z0_include_normal_cut_edges = False
                positive_z_nudged_cell = False
                positive_z_full_footprint_2D: shapely.Geometry | None = None
                positive_z_difference_corners: (
                    list[IntermediateCorner] | None
                ) = None
                positive_z_split_sides: set[str] | None = None
                positive_z_protected_split_sides: set[str] | None = None
                positive_z_side_cut_wall_sides: set[str] | None = None
                positive_z_split_contact_corners: (
                    Sequence[IntermediateCorner]
                ) = ()
                positive_z_flip_edges: set[Edge3D] | None = None
                positive_z_record = empty_positive_z_record
                if nudge_enabled:
                    positive_z_record = positive_z_nudge_plan.get(
                        (j, i),
                        empty_positive_z_record,
                    )
                    positive_z_flip_edges = positive_z_record.get(
                        "flip_edges",
                    )

                    if not using_difference_mesh:
                        z0_detection_raster = top_interpolation_raster
                    else:
                        z0_detection_raster = bottom_raster_for_z0_nudge

                    z0_corners = (
                        z0_nudge_corners_from_source_raster(
                            z0_detection_raster,
                            (j, i),
                            zero_threshold=self.tile_info.config.basethick,
                        )
                        if z0_detection_raster is not None
                        else []
                    )
                    if z0_corners:
                        NWt, SWt, SEt, NEt = topq.vl
                        if (
                            NWt is None
                            or SWt is None
                            or SEt is None
                            or NEt is None
                        ):
                            raise RuntimeError(
                                "Z0 nudge needs four top corner vertices."
                            )
                        cell_top_corner_vertices = {
                            IntermediateCorner.NW: NWt,
                            IntermediateCorner.NE: NEt,
                            IntermediateCorner.SW: SWt,
                            IntermediateCorner.SE: SEt,
                        }
                        if botq is None:
                            (
                                botq,
                                bottom_corner_vertices,
                            ) = _create_cell_bottom_geometry(
                                W,
                                E,
                                N,
                                S,
                                *bottom_elevations,
                                nudge_enabled,
                            )
                        if bottom_corner_vertices is None:
                            raise RuntimeError(
                                "Z0 nudge needs bottom corner vertices.",
                            )
                        cell_bottom_corner_vertices = bottom_corner_vertices

                        def output_z_is_zero(v: vertex) -> bool:
                            return (
                                normalize_coordinate_to_match_mesh_serialization(
                                    v.coords[2],
                                    output_fileformat,
                                )
                                == 0
                            )

                        if not using_difference_mesh:
                            z0_corners = [
                                corner
                                for corner in z0_corners
                                if output_z_is_zero(
                                    cell_top_corner_vertices[corner],
                                )
                                and output_z_is_zero(
                                    cell_bottom_corner_vertices[corner],
                                )
                            ]
                        else:
                            z0_corners = [
                                corner
                                for corner in z0_corners
                                if output_z_is_zero(
                                    cell_bottom_corner_vertices[corner],
                                )
                            ]
                    if 0 < len(z0_corners) < 4:
                        cell_footprint_2D = full_cell_footprint(
                            W,
                            E,
                            N,
                            S,
                        )
                        keep_footprint = nudge_keep_footprint(
                            z0_corners,
                            W,
                            E,
                            N,
                            S,
                        )
                        if keep_footprint is not None:
                            canonicalize_clipped_nudge_xy = (
                                top_bottom_surface_geometries_2D is not None
                            )
                            z0_full_footprint_2D = _union_polygon_footprint(
                                top_bottom_surface_geometries_2D,
                                cell_footprint_2D,
                            )
                            keep_geometry = z0_full_footprint_2D.intersection(
                                keep_footprint,
                            )
                            complement_geometry = (
                                z0_full_footprint_2D.difference(
                                    keep_footprint,
                                )
                            )
                            top_planes = topq.get_triangles_in_polygons(
                                split_rotation=split_rotation,
                            )
                            z0_top_planes = _z0_adjusted_keep_surface_planes(
                                z0_corners,
                                cell_top_corner_vertices,
                                W,
                                E,
                                N,
                                S,
                            )
                            z0_bottom_planes = _z0_adjusted_keep_surface_planes(
                                z0_corners,
                                cell_bottom_corner_vertices,
                                W,
                                E,
                                N,
                                S,
                            )
                            z0_planes = quad(
                                vertex(W, N, 0),
                                vertex(W, S, 0),
                                vertex(E, S, 0),
                                vertex(E, N, 0),
                            ).get_triangles_in_polygons(
                                split_rotation=split_rotation,
                            )
                            if not using_difference_mesh:
                                top_surface_polygons_triangulated_3D = (
                                    _triangulate_2d_geometry_to_3d_polygons(
                                        keep_geometry,
                                        z0_top_planes,
                                        exterior_cw=False,
                                        output_fileformat=output_fileformat,
                                        canonicalize_serialized_xy=(
                                            canonicalize_clipped_nudge_xy
                                        ),
                                    )
                                )
                                bottom_surface_polygons_triangulated_3D = (
                                    _triangulate_2d_geometry_to_3d_polygons(
                                        keep_geometry,
                                        z0_planes,
                                        exterior_cw=True,
                                        output_fileformat=output_fileformat,
                                        canonicalize_serialized_xy=(
                                            canonicalize_clipped_nudge_xy
                                        ),
                                    )
                                )
                                z0_include_normal_cut_edges = True
                            else:
                                if self.tile.bottom_surface_provider is not None:
                                    z0_used_bottom_provider = True
                                if top_bottom_surface_geometries_2D is not None:
                                    z0_include_normal_cut_edges = True
                                kept_bottom_polygons = (
                                    _triangulate_2d_geometry_to_3d_polygons(
                                        keep_geometry,
                                        z0_bottom_planes,
                                        exterior_cw=True,
                                        output_fileformat=output_fileformat,
                                        canonicalize_serialized_xy=(
                                            canonicalize_clipped_nudge_xy
                                        ),
                                    )
                                )

                                top_surface_polygons_triangulated_3D = []
                                for top_piece in (
                                    keep_geometry,
                                    complement_geometry,
                                ):
                                    top_surface_polygons_triangulated_3D.extend(
                                        _triangulate_2d_geometry_to_3d_polygons(
                                            top_piece,
                                            top_planes,
                                            exterior_cw=False,
                                            output_fileformat=output_fileformat,
                                            canonicalize_serialized_xy=(
                                                canonicalize_clipped_nudge_xy
                                            ),
                                        )
                                    )

                                bottom_surface_polygons_triangulated_3D = (
                                    kept_bottom_polygons
                                    + _triangulate_2d_geometry_to_3d_polygons(
                                        complement_geometry,
                                        z0_planes,
                                        exterior_cw=True,
                                        output_fileformat=output_fileformat,
                                        canonicalize_serialized_xy=(
                                            canonicalize_clipped_nudge_xy
                                        ),
                                    )
                                )

                            if (
                                top_surface_polygons_triangulated_3D
                                and bottom_surface_polygons_triangulated_3D
                            ):
                                z0_nudged_cell = True
                            else:
                                clipped_surfaces_collapsed_after_output = True

                    if not using_difference_mesh:
                        positive_contact_corners = positive_z_record.get(
                            "corners",
                            (),
                        )

                        if 0 < len(positive_contact_corners) < 4:
                            cell_footprint_2D = full_cell_footprint(
                                W,
                                E,
                                N,
                                S,
                            )
                            positive_z_full_footprint_2D = (
                                _current_surface_footprint(
                                    top_surface_polygons_triangulated_3D,
                                    cell_footprint_2D,
                                )
                            )
                            keep_footprint = nudge_keep_footprint(
                                positive_contact_corners,
                                W,
                                E,
                                N,
                                S,
                            )
                            if keep_footprint is not None:
                                canonicalize_clipped_nudge_xy = (
                                    top_bottom_surface_geometries_2D is not None
                                )
                                keep_geometry = (
                                    positive_z_full_footprint_2D.intersection(
                                        keep_footprint,
                                    )
                                )
                                complement_geometry = (
                                    positive_z_full_footprint_2D.difference(
                                        keep_footprint,
                                    )
                                )
                                top_planes = _surface_planes_from_current_geometry(
                                    topq,
                                    top_surface_polygons_triangulated_3D,
                                    split_rotation,
                                )
                                bottom_planes = (
                                    botq.get_triangles_in_polygons(
                                        split_rotation=split_rotation,
                                    )
                                )
                                new_top_polygons: list[shapely.Polygon] = []
                                new_bottom_polygons: list[shapely.Polygon] = []
                                for piece in (
                                    keep_geometry,
                                    complement_geometry,
                                ):
                                    new_top_polygons.extend(
                                        _triangulate_2d_geometry_to_3d_polygons(
                                            piece,
                                            top_planes,
                                            exterior_cw=False,
                                            output_fileformat=output_fileformat,
                                            canonicalize_serialized_xy=(
                                                canonicalize_clipped_nudge_xy
                                            ),
                                        )
                                    )
                                    new_bottom_polygons.extend(
                                        _triangulate_2d_geometry_to_3d_polygons(
                                            piece,
                                            bottom_planes,
                                            exterior_cw=True,
                                            output_fileformat=output_fileformat,
                                            canonicalize_serialized_xy=(
                                                canonicalize_clipped_nudge_xy
                                            ),
                                        )
                                    )

                                if positive_z_record.get("midpoint_z_by_name"):
                                    new_top_polygons = (
                                        _surface_polygons_with_midpoint_z(
                                            new_top_polygons,
                                            W,
                                            E,
                                            N,
                                            S,
                                            None,
                                            output_fileformat,
                                            positive_z_record[
                                                "midpoint_z_by_name"
                                            ],
                                        )
                                    )

                                if new_top_polygons and new_bottom_polygons:
                                    top_surface_polygons_triangulated_3D = (
                                        new_top_polygons
                                    )
                                    bottom_surface_polygons_triangulated_3D = (
                                        new_bottom_polygons
                                    )
                                    positive_z_nudged_cell = True
                                else:
                                    clipped_surfaces_collapsed_after_output = True
                        elif (
                            positive_z_record.get("split_sides")
                            and top_bottom_surface_geometries_2D is None
                        ):
                            positive_z_split_sides = set(
                                positive_z_record["split_sides"],
                            )
                            positive_z_split_contact_corners = (
                                positive_z_record.get(
                                    "contact_corners",
                                    (),
                                )
                            )
                    else:
                        positive_z_difference_corners = (
                            _positive_z_effective_difference_corners(
                                positive_z_record,
                            )
                        )
                        if positive_z_difference_corners:
                            positive_z_side_cut_wall_sides = set()
                            for (
                                current_side,
                                neighbor_delta,
                                neighbor_side,
                            ) in CELL_NEIGHBOR_SIDES:
                                neighbor_location = (
                                    j + neighbor_delta[0],
                                    i + neighbor_delta[1],
                                )
                                if not _nudge_keep_footprint_splits_side(
                                    positive_z_difference_corners,
                                    current_side,
                                ):
                                    continue
                                neighbor_record = positive_z_nudge_plan.get(
                                    neighbor_location,
                                    empty_positive_z_record,
                                )
                                neighbor_corners = (
                                    _positive_z_effective_difference_corners(
                                        neighbor_record,
                                    )
                                )
                                neighbor_matches = (
                                    neighbor_side
                                    in neighbor_record.get("split_sides", ())
                                ) or (
                                    neighbor_side
                                    in (
                                        positive_z_difference_neighbor_split_sides
                                        .get(neighbor_location, ())
                                    )
                                ) or _nudge_keep_footprint_splits_side(
                                    neighbor_corners,
                                    neighbor_side,
                                )
                                if not neighbor_matches:
                                    positive_z_side_cut_wall_sides.add(
                                        current_side,
                                    )
                        if (
                            not positive_z_difference_corners
                            and top_bottom_surface_geometries_2D is None
                        ):
                            positive_z_split_sides = set(
                                positive_z_record.get("split_sides", ()),
                            )
                            positive_z_protected_split_sides = set(
                                positive_z_split_sides,
                            )
                            positive_z_split_sides.update(
                                positive_z_difference_neighbor_split_sides.get(
                                    (j, i),
                                    (),
                                )
                            )
                            if positive_z_split_sides:
                                positive_z_split_contact_corners = (
                                    positive_z_record.get(
                                        "contact_corners",
                                        (),
                                    )
                                )

                if clipped_surfaces_collapsed_after_output:
                    continue

                requested_cardinal_sides = _requested_cardinal_borders(
                    j,
                    i,
                    self.ymaxidx,
                    self.xmaxidx,
                    borders_top_raster,
                    check_nan_neighbors=self.tile_info.have_nan,
                )

                # Materialize only the requested exterior walls.
                if requested_cardinal_sides:
                    if botq is None:
                        (
                            botq,
                            bottom_corner_vertices,
                        ) = _create_cell_bottom_geometry(
                            W,
                            E,
                            N,
                            S,
                            *bottom_elevations,
                            nudge_enabled,
                        )
                    borders = _build_cardinal_wall_borders(
                        requested_cardinal_sides,
                        topq.vl,
                        botq.vl,
                        output_fileformat,
                    )
                else:
                    borders = _empty_borders()

                # create borders if there is a top surface polygon using the edge buckets
                surface_polygon_borders_3D: list[quad] | None = None
                buckets = (
                    polygon_edge_buckets[cell_row][cell_col]
                    if polygon_edge_buckets is not None
                    else None
                )
                if buckets is not None:
                    wall_border_edges = _wall_border_edges_from_buckets(
                        buckets,
                    )

                    if (
                        top_bottom_surface_geometries_2D
                        and top_surface_polygons_triangulated_3D
                        and bottom_surface_polygons_triangulated_3D
                    ):
                        top_edges_by_key = _boundary_line_map_by_serialized_xy(
                            _surface_polygon_edges(
                                top_surface_polygons_triangulated_3D,
                            ),
                            output_fileformat,
                        )
                        bottom_edges_by_key = (
                            _boundary_line_map_by_serialized_xy(
                                _surface_polygon_edges(
                                    bottom_surface_polygons_triangulated_3D,
                                ),
                                output_fileformat,
                            )
                        )

                        serialized_border_lines = [
                            border_line
                            for border_edge in wall_border_edges
                            for border_line in [
                                _line_with_serialized_xy(
                                    border_edge.geometry,
                                    output_fileformat,
                                )
                            ]
                            if border_line is not None
                        ]
                        wall_border_linework = (
                            shapely.union_all(serialized_border_lines)
                            if serialized_border_lines
                            else shapely.GeometryCollection()
                        )

                        # Build one wall for each emitted clipped surface edge
                        # covered by this cell's wall border linework.
                        for edge_key, top_edge_matches in (
                            top_edges_by_key.items()
                        ):
                            edge_line = shapely.LineString(
                                [edge_key[0], edge_key[1]],
                            )
                            if not wall_border_linework.covers(edge_line):
                                continue

                            bottom_edge_matches = bottom_edges_by_key.get(
                                edge_key,
                                [],
                            )
                            if len(top_edge_matches) != len(
                                bottom_edge_matches,
                            ):
                                raise RuntimeError(
                                    "Border creation found different top and "
                                    "bottom edge match counts."
                                )

                            for top_edge_match, bottom_edge_match in zip(
                                top_edge_matches,
                                bottom_edge_matches,
                            ):
                                # Success condition where wall border linework
                                # covers a top/bottom surface edge pair.
                                top_edge_v0 = vertex(*top_edge_match.coords[1])
                                top_edge_v1 = vertex(*top_edge_match.coords[0])
                                bot_edge_v0 = vertex(
                                    *bottom_edge_match.coords[1],
                                )
                                bot_edge_v1 = vertex(
                                    *bottom_edge_match.coords[0],
                                )
                                tb_wall = make_wall_without_exact_duplicate_vertices(
                                    top_edge_v0,
                                    top_edge_v1,
                                    bot_edge_v0,
                                    bot_edge_v1,
                                    output_fileformat=output_fileformat,
                                )
                                if tb_wall is not None:
                                    if surface_polygon_borders_3D is None:
                                        surface_polygon_borders_3D = []
                                    surface_polygon_borders_3D.append(tb_wall)
                            # create border geometry with top and bot edge
                            # top and bot edges are in CW order (viewed from top) from shapely
                if z0_nudged_cell:
                    if (
                        top_surface_polygons_triangulated_3D is None
                        or bottom_surface_polygons_triangulated_3D is None
                    ):
                        raise RuntimeError(
                            "Z0 nudged cell is missing final surface polygons."
                        )
                    if z0_full_footprint_2D is None:
                        z0_full_footprint_2D = full_cell_footprint(
                            W,
                            E,
                            N,
                            S,
                        )
                    surface_polygon_borders_3D = (
                        rebuild_nudged_surface_polygon_borders(
                            top_surface_polygons_triangulated_3D,
                            bottom_surface_polygons_triangulated_3D,
                            borders,
                            surface_polygon_borders_3D,
                            z0_full_footprint_2D,
                            include_normal_cut_edges=(
                                z0_include_normal_cut_edges
                            ),
                            W=W,
                            E=E,
                            N=N,
                            S=S,
                            output_fileformat=output_fileformat,
                        )
                    )
                    borders = _empty_borders()
                elif positive_z_nudged_cell:
                    if (
                        top_surface_polygons_triangulated_3D is None
                        or bottom_surface_polygons_triangulated_3D is None
                    ):
                        raise RuntimeError(
                            "Positive-Z nudged cell is missing final surface "
                            "polygons."
                        )
                    if positive_z_full_footprint_2D is None:
                        positive_z_full_footprint_2D = (
                            full_cell_footprint(
                                W,
                                E,
                                N,
                                S,
                            )
                        )
                    surface_polygon_borders_3D = (
                        rebuild_nudged_surface_polygon_borders(
                            top_surface_polygons_triangulated_3D,
                            bottom_surface_polygons_triangulated_3D,
                            borders,
                            surface_polygon_borders_3D,
                            positive_z_full_footprint_2D,
                            include_normal_cut_edges=False,
                            W=W,
                            E=E,
                            N=N,
                            S=S,
                            output_fileformat=output_fileformat,
                        )
                    )
                    borders = _empty_borders()

                c = cell(
                    topq,
                    botq if emit_cell_bottom else None,
                    borders,
                )

                # set surface polygons for cell if clipping border affects this cell
                if top_surface_polygons_triangulated_3D:
                    c.topSurfacePolygons = top_surface_polygons_triangulated_3D
                if bottom_surface_polygons_triangulated_3D:
                    c.bottomSurfacePolygons = bottom_surface_polygons_triangulated_3D
                if surface_polygon_borders_3D:
                    c.surfacePolygonBorders = surface_polygon_borders_3D

                if (
                    self.tile.bottom_surface_provider is not None
                    and not z0_used_bottom_provider
                    and not positive_z_difference_corners
                    and not positive_z_split_sides
                ):
                    bottom_provider = self.tile.bottom_surface_provider
                    bottom_quad, bottom_polygons = (
                        bottom_provider[cell_row][cell_col]
                    )
                    if bottom_quad is None and bottom_polygons is None:
                        if (
                            c.bottomquad is None
                            and c.bottomSurfacePolygons is None
                        ):
                            raise RuntimeError(
                                "Interlocking pair difference cell needs a "
                                "bottom surface at "
                                f"row={cell_row}, col={cell_col}."
                            )
                    else:
                        c.replace_bottom_surfaces(
                            bottom_surface_quad=bottom_quad,
                            bottom_surface_polygons=bottom_polygons,
                            split_rotation=split_rotation,
                            output_fileformat=output_fileformat,
                        )

                if (
                    positive_z_split_sides
                    and not positive_z_difference_corners
                    and nudge_enabled
                ):
                    c.split_surface_boundary_midpoints(
                        positive_z_split_sides,
                        positive_z_split_contact_corners,
                        W=W,
                        E=E,
                        N=N,
                        S=S,
                        split_rotation=split_rotation,
                        output_fileformat=output_fileformat,
                        include_contact_cut_walls=False,
                        side_cut_wall_sides=None,
                    )

                preserve_zero_height_xy: set[tuple[float, float]] | None = None
                preserve_zero_height_edges: set[XYEdge] | None = None
                if (
                    using_difference_mesh
                    and positive_z_split_sides
                    and not positive_z_difference_corners
                    and nudge_enabled
                    and positive_z_protected_split_sides
                ):
                    preserve_zero_height_xy = _nudge_split_side_endpoint_xy(
                        positive_z_protected_split_sides,
                        W,
                        E,
                        N,
                        S,
                        output_fileformat,
                    )
                    preserve_zero_height_edges = (
                        _nudge_split_side_endpoint_edges(
                            positive_z_protected_split_sides,
                            W,
                            E,
                            N,
                            S,
                            output_fileformat,
                        )
                    )

                if (
                    positive_z_flip_edges
                    and nudge_enabled
                    and using_difference_mesh
                    and c.topSurfacePolygons
                    and c.bottomSurfacePolygons
                ):
                    c.flip_bottom_positive_z_contact_edges(
                        split_rotation=split_rotation,
                        output_fileformat=output_fileformat,
                        allowed_edges=positive_z_flip_edges,
                    )

                if (
                    using_difference_mesh or nudge_enabled
                ):
                    c.remove_zero_height_volumes(
                        split_rotation=split_rotation,
                        output_fileformat=output_fileformat,
                        preserve_zero_height_xy=preserve_zero_height_xy,
                        preserve_zero_height_edges=preserve_zero_height_edges,
                        serialized_vertices=serialized_row_vertices,
                    )

                if (
                    positive_z_difference_corners
                    and nudge_enabled
                    and using_difference_mesh
                ):
                    if self.tile.bottom_surface_provider is not None:
                        bottom_quad, bottom_polygons = (
                            self.tile.bottom_surface_provider[cell_row][
                                cell_col
                            ]
                        )
                        if bottom_quad is not None or bottom_polygons is not None:
                            c.replace_bottom_surfaces(
                                bottom_surface_quad=bottom_quad,
                                bottom_surface_polygons=bottom_polygons,
                                split_rotation=split_rotation,
                                output_fileformat=output_fileformat,
                            )
                    positive_z_difference_midpoints = dict(
                        positive_z_record.get(
                            "difference_midpoint_z_by_name",
                            positive_z_record.get("midpoint_z_by_name", {}),
                        ),
                    )
                    c.apply_positive_z_difference_nudge(
                        positive_z_difference_corners,
                        W=W,
                        E=E,
                        N=N,
                        S=S,
                        split_rotation=split_rotation,
                        output_fileformat=output_fileformat,
                        top_midpoint_z_by_name=(
                            positive_z_difference_midpoints
                        ),
                        bottom_midpoint_z_by_name=(
                            positive_z_difference_midpoints
                        ),
                        side_cut_wall_sides=positive_z_side_cut_wall_sides,
                    )
                    c.remove_zero_height_volumes(
                        split_rotation=split_rotation,
                        output_fileformat=output_fileformat,
                        serialized_vertices=serialized_row_vertices,
                    )

                # if we have nan cells, do some postprocessing on this cell to get rid of stair case patterns
                # This will create special triangle cells that have a triangle of any orientation at top/bottom, which
                # are flagged as is_tri_cell = True, and have only v0, v1 and v2. One border is deleted, the other
                # is set as a diagonal wall.
                # Note: this will not be done if we have a bottom as it will lead to lots of triangle holes!
                if (
                    self.tile_info.have_nan
                    and self.tile_info.config.smooth_borders
                    and not using_difference_mesh
                ):
                    if c.check_for_tri_cell():
                        c.convert_to_tri_cell()

                # Normalize and remove geometry that would collapse during mesh
                # serialization before storing the cell.
                if (
                    not self.tile.defer_serialization_cleanup
                    and not (
                        skip_simple_serialization_cleanup
                        and self._cell_xy_survives_mesh_serialization(
                            W,
                            E,
                            N,
                            S,
                            output_fileformat,
                        )
                    )
                ):
                    c.remove_geometry_collapsed_by_mesh_serialization(
                        output_fileformat=output_fileformat,
                        split_rotation=split_rotation,
                        serialized_vertices=serialized_row_vertices,
                    )

                self.cells[cell_row, cell_col] = c

                if not self.tile.defer_triangle_writes:
                    self.write_cell_meshes_to_buffer(c)

        print("100%", multiprocessing.current_process(), "\n", file=sys.stderr)

    def write_cell_meshes_to_buffer(
        self,
        current_cell: cell,
        coordinates_normalized: bool = False,
    ) -> None:
        """Write one finalized cell's meshes to the current output buffer."""
        if self._uses_fast_binary_stl_no_normals_writer():
            self._write_cell_meshes_to_binary_stl_no_normals(
                current_cell,
                coordinates_normalized,
            )
            return

        def triangle_rounded_to_precision(
            decimals: int,
            triangle: list[vertex],
        ) -> list[vertex]:
            output: list[vertex] = []
            for tv in triangle:
                output.append(
                    tv.vertex_rounded_to_precision(
                        decimals=decimals,
                    )
                )
            return output

        decimal_precision = MESH_OUTPUT_SERIALIZATION_DECIMAL_PRECISION
        for q in current_cell.iter_meshes_for_model():
            if isinstance(q, quad):
                quad_triangles = q.get_triangles(
                    split_rotation=self.tile_info.config.split_rotation,
                )
                for t in quad_triangles:
                    self.write_triangle_to_buffer(
                        tuple(
                            triangle_rounded_to_precision(
                                decimals=decimal_precision,
                                triangle=list(t),
                            )
                        )
                    )
            elif isinstance(q, shapely.Polygon):
                t0 = polygon_to_list_of_vertex(polygon=q)
                if len(t0) == 4 and t0[0].coords == t0[3].coords:
                    self.write_triangle_to_buffer(
                        tuple(
                            triangle_rounded_to_precision(
                                decimals=decimal_precision,
                                triangle=list(t0[:3]),
                            )
                        )
                    )
                else:
                    raise ValueError(
                        "create_cells: found a polygon to write to buffer "
                        "that is not a triangle. Expected a tri of length "
                        "3+1=4 and [0]==[3] vertex. Polygon had vertex "
                        f"count f{len(t0)}."
                    )

    def _uses_fast_binary_stl_no_normals_writer(self) -> bool:
        """Return whether triangles can be serialized without vertex objects."""
        return (
            self.tile_info.config.fileformat == "STLb"
            and self.tile_info.config.no_normals is True
        )

    def _write_triangle_coords_to_binary_stl_no_normals(
        self,
        triangle: Sequence[Coordinate],
        coordinates_normalized: bool = False,
    ) -> None:
        """Write one no-normal binary STL triangle from raw coordinates."""
        self.num_triangles += 1
        write = self.s.write
        pack_facet = BINARY_STL_FACET.pack
        c0, c1, c2 = triangle
        if coordinates_normalized:
            write(
                pack_facet(
                    0.0,
                    0.0,
                    0.0,
                    c0[0] + 0.0,
                    c0[1] + 0.0,
                    c0[2] + 0.0,
                    c1[0] + 0.0,
                    c1[1] + 0.0,
                    c1[2] + 0.0,
                    c2[0] + 0.0,
                    c2[1] + 0.0,
                    c2[2] + 0.0,
                    0,
                )
            )
            if self.tile_info.temp_file is not None:
                self.write_buffer_to_file()
            return

        decimal_precision = MESH_OUTPUT_SERIALIZATION_DECIMAL_PRECISION
        round_coord = round
        to_float = float
        write(
            pack_facet(
                0.0,
                0.0,
                0.0,
                round_coord(to_float(c0[0]), decimal_precision) + 0.0,
                round_coord(to_float(c0[1]), decimal_precision) + 0.0,
                round_coord(to_float(c0[2]), decimal_precision) + 0.0,
                round_coord(to_float(c1[0]), decimal_precision) + 0.0,
                round_coord(to_float(c1[1]), decimal_precision) + 0.0,
                round_coord(to_float(c1[2]), decimal_precision) + 0.0,
                round_coord(to_float(c2[0]), decimal_precision) + 0.0,
                round_coord(to_float(c2[1]), decimal_precision) + 0.0,
                round_coord(to_float(c2[2]), decimal_precision) + 0.0,
                0,
            )
        )
        if self.tile_info.temp_file is not None:
            self.write_buffer_to_file()

    def _write_cell_meshes_to_binary_stl_no_normals(
        self,
        current_cell: cell,
        coordinates_normalized: bool = False,
    ) -> None:
        """Write finalized cell meshes through the fast binary STL path."""
        split_rotation = self.tile_info.config.split_rotation
        for mesh in current_cell.iter_meshes_for_model():
            if isinstance(mesh, quad):
                for triangle in mesh.get_triangles(
                    split_rotation=split_rotation,
                ):
                    self._write_triangle_coords_to_binary_stl_no_normals(
                        tuple(v.coords for v in triangle),
                        coordinates_normalized,
                    )
            elif isinstance(mesh, shapely.Polygon):
                coords = mesh.exterior.coords
                if len(coords) == 4 and coords[0] == coords[3]:
                    self._write_triangle_coords_to_binary_stl_no_normals(
                        (coords[0], coords[1], coords[2]),
                        coordinates_normalized,
                    )
                else:
                    raise ValueError(
                        "create_cells: found a polygon to write to buffer "
                        "that is not a triangle. Expected a tri of length "
                        "3+1=4 and [0]==[3] vertex. Polygon had vertex "
                        f"count f{len(coords)}."
                    )

    def _cleanup_cell_for_mesh_serialization(
        self,
        current_cell: cell,
    ) -> None:
        """Normalize and drop cell geometry that cannot serialize."""
        current_cell.remove_geometry_collapsed_by_mesh_serialization(
            output_fileformat=self.tile_info.config.fileformat,
            split_rotation=self.tile_info.config.split_rotation,
        )

    def _cleanup_existing_cells_for_serialization(
        self,
        parallel_workers: int | None = None,
    ) -> None:
        """Clean deferred cells before serial mesh writes."""
        if self.cells is None:
            raise RuntimeError("Cannot clean cells before create_cells().")

        worker_count = (
            parallel_workers
            if parallel_workers is not None
            else single_job_parallel_workers(
                self.tile_info.config,
                self.cells.shape[0],
            )
        )
        _cleanup_cells_for_mesh_serialization(
            self.cells,
            self.tile_info.config.fileformat,
            self.tile_info.config.split_rotation,
            worker_count,
        )

    def write_existing_cells_to_buffer(
        self,
        parallel_cleanup_workers: int | None = None,
    ) -> None:
        """Write already-created cells to the current output buffer."""
        if self.cells is None:
            raise RuntimeError("Cannot write cells before create_cells().")

        if self.tile_info.config.fileformat != "obj":
            self._cleanup_existing_cells_for_serialization(
                parallel_cleanup_workers,
            )
            for row in self.cells:
                for current_cell in row:
                    if current_cell is not None:
                        self.write_cell_meshes_to_buffer(
                            current_cell,
                            coordinates_normalized=True,
                        )
            return

        for row in self.cells:
            for current_cell in row:
                if current_cell is not None:
                    self._cleanup_cell_for_mesh_serialization(current_cell)
                    self.write_cell_meshes_to_buffer(current_cell)

    def _should_add_simple_bottom(self) -> bool:
        """Return whether this grid should add one tile-wide bottom."""
        if self.tile_info.config.no_bottom:
            return False
        if self.tile_info.have_nan:
            return False
        if (
            self.tile_info.config.bottom_image is not None
            or self.tile_info.config.bottom_elevation is not None
        ):
            return False
        return not self.tile_info.config.nudge_in_overused_edges_vertex

    def _add_simple_bottom_to_buffer(self) -> None:
        """Add the two-triangle tile bottom used by simple normal meshes."""
        v0 = vertex(self.tile_info.W, self.tile_info.S, 0)
        v1 = vertex(self.tile_info.E, self.tile_info.S, 0)
        v2 = vertex(self.tile_info.E, self.tile_info.N, 0)
        v3 = vertex(self.tile_info.W, self.tile_info.N, 0)

        self.write_triangle_to_buffer((v0, v2, v1))
        self.write_triangle_to_buffer((v0, v3, v2))

    def write_triangle_to_buffer(self, t: tuple[vertex, ...]) -> None:
        """Write a triangle to the in-memory output buffer."""
        self.num_triangles += 1

        # Create triangle coords list, for STL including normal coords (no normals for obj)
        if self.tile_info.config.fileformat != "obj":
            tl = (
                get_normal(t)
                if not self.tile_info.config.no_normals
                else [0, 0, 0]
            )
            for v in t:
                coords = v.get() # get() => list of coords [x,y,z]
                # pack 64 bit float to 32 bit and unpack 32 bit back to 64 bit to try to get the same value represented in 32 bit
                #coords = tuple(map(lambda x: struct.unpack('<f', struct.pack('<f', x))[0], coords))
                # add 0.0 to value to force -0 value to +0
                coords = tuple(coord + 0.0 for coord in coords)
                tl.extend(coords) # like append() but extend() unpacks that list!
            tl.append(0) # append attribute byte 0

        if self.tile_info.config.fileformat == "STLb":
            # en.wikipedia.org/wiki/STL_%28file_format%29#Binary_STL
            self.s.write(BINARY_STL_FACET.pack(*tl))  # append to s

        elif self.tile_info.config.fileformat == "STLa":
            self.s.write(
                ASCII_STL_FACET_TEMPLATE.format(
                    face=tl,
                    precision=MESH_OUTPUT_SERIALIZATION_DECIMAL_PRECISION,
                )
            )

        elif self.tile_info.config.fileformat == "obj":
            # add facet indices to index stream buffer
            vl = [v.get_id() + 1 for v in t] # vertex list +1 b/c obj indices start at 1
            self.s[1].write(f"f {vl[0]}, {vl[1]}, {vl[2]}\n")

        # for STL maybe write to temp file. This can't work for obj b/c we need the full list
        # of tri indices first. Once we have that, we can create a buffer/tempfile
        if (
            self.tile_info.config.fileformat != "obj"
            and self.tile_info.temp_file is not None
        ):
            self.write_buffer_to_file()

    def write_buffer_to_file(self, flush=False, chunk_size=100000):
        # write buffer to file every 10k triangles
        # chunksize is the number of triangles that need to have been collected into the buffer in order to actually write to disk. (cache)
        # flusk=True forces a write: use this to flush whatever is in the buffer.  Will NOT close the file!
        # for obj, write only the indices [1], vertices [0] will be done later

        # Only write to file if we're actually using temp files, otherwise just bail out
        if self.tile_info.temp_file is None:
            return

        if self.num_triangles % chunk_size == 0 or flush:
            if self.tile_info.config.fileformat == "STLb":
                self.fo.write(self.s.getbuffer())   # append (partial) binary buffer to file
                self.s.close()
                self.s = io.BytesIO()
            elif self.tile_info.config.fileformat == "STLa":
                self.fo.write(self.s.getvalue())   # append (partial) text buffer to file
                self.s.close()
                self.s = io.StringIO()
            elif self.tile_info.config.fileformat == "obj":
                self.fo[1].write(self.s[1].getvalue())
                self.s[1].close()
                self.s[1] = io.StringIO()

        if flush:
            # close buffers (needed?)
            if self.tile_info.config.fileformat == "obj":
                self.s[1].close()
            else: # STLb and STLa
                self.s.close()


    # Convert grid into a file or memory buffer containing triangles (plus indices for obj)
    def make_file_buffer(
        self,
        create_cells: bool = True,
        parallel_cleanup_workers: int | None = None,
    ):

        if self.tile_info is None:
            print("make_file_buffer: Error: self.tile_info is None")
            return

        # check that we have a valid triangle file format
        if self.tile_info.config.fileformat not in ["obj", "STLa", "STLb"]:
            raise ValueError(f"Invalid file format: {self.tile_info.config.fileformat}. Supported formats are 'obj', 'STLa', and 'STLb'")

        temp_file = self.tile_info.temp_file

        self.num_triangles = 0

        # Open in-memory stream buffers s
        # s is used to collect the data that is eventually written into a proper file
        if self.tile_info.config.fileformat == "STLb":
            self.s = io.BytesIO()
            mode = "ab" if create_cells else "wb"  # for using open() later
        elif self.tile_info.config.fileformat == "STLa":
            self.s = io.StringIO()
            mode = "a" if create_cells else "w"
        elif self.tile_info.config.fileformat == "obj":
            mode = "a" if create_cells else "w"
            # 2 buffers: vertices and indices
            self.s = [io.StringIO(), io.StringIO()]

        # open temp file for appending, file object self.fo will be used in create_cells()
        if temp_file is not None:
            if self.tile_info.config.fileformat == "STLa" or self.tile_info.config.fileformat == "STLb":
                try:
                    self.fo = open(temp_file, mode)
                except Exception as e:
                    print("Error opening:", temp_file, e, file=sys.stderr)
                    return e
            elif self.tile_info.config.fileformat == "obj":
                # for obj we need 2  temp files and file objects, so s and fo are now lists
                try:
                    vertsfo =  open(temp_file, mode)
                except Exception as e:
                    print("Error opening:", temp_file, e, file=sys.stderr)
                    return e
                idx_temp_file = temp_file + ".idx" # index temp file just has .idx at the end

                try:
                    idxfo = open(idx_temp_file, mode)
                except Exception as e:
                    print("Error opening:", idx_temp_file, e, file=sys.stderr)
                    return e
                self.fo = [vertsfo, idxfo]

        # header for STLa and obj
        # (STLb header can only pre-pended later)
        if self.tile_info.config.fileformat == "STLa":
            self.s.write('solid digital_elevation_model\n') # digital_elevation_model is the name of the model
        elif self.tile_info.config.fileformat == "obj":
            self.s[0].write("g vert\n")
            self.s[1].write("g tris\n")

        if create_cells:
            # Populate self.cells and write triangles into buffer/file.
            self.create_cells()
        else:
            self.write_existing_cells_to_buffer(parallel_cleanup_workers)

        # Can we use 2-triangle bottoms?
        add_simple_bottom = self._should_add_simple_bottom()

        # For simple bottom, add 2 triangles based on the corners of the tile
        if add_simple_bottom:
            self._add_simple_bottom_to_buffer()

        # using buffer
        if temp_file is None:

            # finish STLa stream buffer
            if self.tile_info.config.fileformat == "STLa":
                self.s.write('endsolid digital_elevation_model') # append end clause
                buf = self.s.getvalue()

            # For STLb buffer, prepend the header
            if self.tile_info.config.fileformat == "STLb":
                stlb_header = io.BytesIO()
                stlb_header.write(
                    BINARY_STL_HEADER.pack(
                        b'Binary STL Writer',
                        self.num_triangles,
                    )
                )
                stlb_header.write(self.s.getbuffer()) # append body to header
                del self.s # no longer needed
                buf = stlb_header.getvalue()  # CH 5/2025 changed from getbuffer to not return a memory object that c an't be pickled

            # fill s[0] and append s[1]
            elif self.tile_info.config.fileformat == "obj":
                # fill s[0] with all vertices used (keys of vertex class attribute dict)
                print("Appending obj triangle indices\n", file=sys.stderr)
                for vc in vertex.vertex_index_dict:
                    self.s[0].write(f"v {vc[0]}, {vc[1]}, {vc[2]}\n")

                self.s[0].write(self.s[1].getvalue()) # append indices
                del self.s[1]
                buf = self.s[0].getvalue()

            return buf

        # using temp file
        else:
            self.write_buffer_to_file(flush=True) # write leftover buffer to file, will NOT close fo!

            # STLa: append last line
            if self.tile_info.config.fileformat == "STLa":
                self.fo.write('endsolid digital_elevation_model')
                self.fo.close()

            # for binary STL we can only now prepend a header as we didn't have num_triangles until now.
            elif self.tile_info.config.fileformat == "STLb":
                # rename curent file so we can append it to the header file
                self.fo.close()
                body_file = temp_file + ".body"
                os.replace(temp_file, body_file)
                with open(body_file, "rb") as fbody:
                    with open(temp_file, "ab") as fheader: # new temp_file
                        fheader.write(
                            BINARY_STL_HEADER.pack(
                                b'Binary STL Writer',
                                self.num_triangles,
                            )
                        )
                        shutil.copyfileobj(fbody, fheader) # append the body to the header
                os.remove(body_file)

            # For obj the the fo[0] temp file (vertices) must be filled, then the
            # .idx temp file needs to be appended to i
            elif self.tile_info.config.fileformat == "obj":
                # fill vertex temp file
                print("Appending obj triangle indices\n", file=sys.stderr)
                for vc in vertex.vertex_index_dict:
                    self.fo[0].write(f"v {vc[0]}, {vc[1]}, {vc[2]}\n")
                self.fo[0].close()
                self.fo[1].close()

                # append index temp file top vertex temp file
                idx_temp_file = temp_file + ".idx"
                with open(idx_temp_file, "r") as idx_fo:
                    with open(temp_file, "a") as vert_fo:
                        shutil.copyfileobj(idx_fo, vert_fo)
                os.remove(idx_temp_file)

            return temp_file
