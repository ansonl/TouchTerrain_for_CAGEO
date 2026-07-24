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
import warnings # for muting warnings about nan in e.g. nanmean()
import multiprocessing
import os
import shutil
import struct # for making binary STL
import sys

# get root logger, will later be redirected into a logfile
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from concurrent.futures import ThreadPoolExecutor
from collections.abc import Iterable, Iterator, Sequence
from typing import Union, Any, Callable, TypeAlias

import numpy as np
import shapely

from touchterrain.common.Vertex import vertex
from touchterrain.common.Quad import quad

from touchterrain.common.vectors import Vector, Point  # local copy of vectors package which was no longer working in python 3
import touchterrain.common.utils as utils

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


Coordinate: TypeAlias = Sequence[float]
XYEdge: TypeAlias = tuple[tuple[float, float], tuple[float, float]]
Edge3D: TypeAlias = tuple[tuple[float, ...], tuple[float, ...]]
DirectedEdge3D: TypeAlias = tuple[tuple[float, ...], tuple[float, ...]]
SurfaceMesh: TypeAlias = Union[quad, shapely.Polygon]
CornerElevations: TypeAlias = tuple[float, float, float, float]
EmittedBottomSurface: TypeAlias = tuple[
    quad | None,
    list[shapely.Polygon] | None,
]
BottomSurfaceProvider: TypeAlias = list[list[EmittedBottomSurface]]
TopFootprintSource: TypeAlias = shapely.Geometry
TopFootprintProvider: TypeAlias = list[list[TopFootprintSource | None]]
PositiveZNudgeRecord: TypeAlias = dict[str, Any]
PositiveZNudgePlan: TypeAlias = dict[tuple[int, int], PositiveZNudgeRecord]
PositiveZSurfaceValues: TypeAlias = tuple[
    dict[IntermediateCorner, float],
    dict[IntermediateCorner, float],
]
SerializedVertexCache: TypeAlias = dict[tuple[float, ...], tuple[float, ...]]
CellBottomGeometry: TypeAlias = tuple[
    vertex,
    vertex,
    vertex,
    vertex,
    quad,
    dict[IntermediateCorner, vertex] | None,
]
MESH_OUTPUT_SERIALIZATION_DECIMAL_PRECISION = 6
BINARY_STL_FACET = struct.Struct("<12fH")
BINARY_STL_HEADER = struct.Struct("80sI")
BINARY_FLOAT = struct.Struct("<f")
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
CARDINAL_DIRECTIONS = ("N", "S", "E", "W")
NUDGE_SIDE_MIDPOINT_NAME = {
    "N": "Nmid",
    "S": "Smid",
    "E": "Emid",
    "W": "Wmid",
}
NUDGE_SIDE_ENDPOINT_NAMES = {
    "N": ("NW", "NE"),
    "S": ("SW", "SE"),
    "E": ("SE", "NE"),
    "W": ("SW", "NW"),
}
NUDGE_SIDE_SEGMENT_NAMES = {
    "N": (("NW", "Nmid"), ("Nmid", "NE")),
    "S": (("SW", "Smid"), ("Smid", "SE")),
    "E": (("SE", "Emid"), ("Emid", "NE")),
    "W": (("SW", "Wmid"), ("Wmid", "NW")),
}
NUDGE_MIDPOINT_CORNERS_BY_NAME = {
    "Nmid": (IntermediateCorner.NW, IntermediateCorner.NE),
    "Smid": (IntermediateCorner.SW, IntermediateCorner.SE),
    "Emid": (IntermediateCorner.NE, IntermediateCorner.SE),
    "Wmid": (IntermediateCorner.NW, IntermediateCorner.SW),
}
CELL_NEIGHBOR_SIDES = (
    ("N", (-1, 0), "S"),
    ("S", (1, 0), "N"),
    ("W", (0, -1), "E"),
    ("E", (0, 1), "W"),
)
POSITIVE_Z_SIDE_CONTACT_CHECKS = (
    (
        frozenset((IntermediateCorner.NW, IntermediateCorner.NE)),
        "N",
        (-1, 0),
        "S",
        frozenset((IntermediateCorner.SW, IntermediateCorner.SE)),
    ),
    (
        frozenset((IntermediateCorner.SW, IntermediateCorner.SE)),
        "S",
        (1, 0),
        "N",
        frozenset((IntermediateCorner.NW, IntermediateCorner.NE)),
    ),
    (
        frozenset((IntermediateCorner.NW, IntermediateCorner.SW)),
        "W",
        (0, -1),
        "E",
        frozenset((IntermediateCorner.NE, IntermediateCorner.SE)),
    ),
    (
        frozenset((IntermediateCorner.NE, IntermediateCorner.SE)),
        "E",
        (0, 1),
        "W",
        frozenset((IntermediateCorner.NW, IntermediateCorner.SW)),
    ),
)
POSITIVE_Z_SW_NE_DIAGONAL = frozenset({
    IntermediateCorner.SW,
    IntermediateCorner.NE,
})
POSITIVE_Z_SE_NW_DIAGONAL = frozenset({
    IntermediateCorner.SE,
    IntermediateCorner.NW,
})
POSITIVE_Z_OPPOSITE_CORNER_PAIRS = (
    POSITIVE_Z_SW_NE_DIAGONAL,
    POSITIVE_Z_SE_NW_DIAGONAL,
)


def _empty_borders() -> dict[str, quad | bool]:
    """Return a fresh N/S/E/W wall map with no requested walls."""
    return {direction: False for direction in CARDINAL_DIRECTIONS}


def _empty_side_edge_sets() -> dict[str, set[Edge3D]]:
    """Return a fresh N/S/E/W positive-Z edge map."""
    return {direction: set() for direction in CARDINAL_DIRECTIONS}


def _create_cell_bottom_geometry(
    W: float,
    E: float,
    N: float,
    S: float,
    NEelev: float,
    NWelev: float,
    SEelev: float,
    SWelev: float,
    nudge_enabled: bool,
) -> CellBottomGeometry:
    """Create bottom vertices, bottom quad, and optional nudge corner map."""
    NEb = vertex(E, N, NEelev)
    NWb = vertex(W, N, NWelev)
    SEb = vertex(E, S, SEelev)
    SWb = vertex(W, S, SWelev)
    botq = quad(NWb, NEb, SEb, SWb)
    bottom_corner_vertices = None
    if nudge_enabled:
        bottom_corner_vertices = {
            IntermediateCorner.NW: NWb,
            IntermediateCorner.NE: NEb,
            IntermediateCorner.SW: SWb,
            IntermediateCorner.SE: SEb,
        }
    return NEb, NWb, SEb, SWb, botq, bottom_corner_vertices


def _single_job_parallel_workers(config: Any, task_count: int) -> int:
    """Return worker count for independent work inside one mesh job."""
    requested = getattr(config, "CPU_cores_to_use", None)
    if getattr(config, "fileformat", None) == "obj":
        return 1
    if task_count <= 1 or requested in (None, 1):
        return 1
    requested_cores = os.cpu_count() if requested == 0 else requested
    if requested_cores is None:
        return 1
    return max(1, min(task_count, requested_cores))


def _parallel_row_ranges(
    row_start: int,
    row_end: int,
    worker_count: int,
) -> list[tuple[int, int]]:
    """Split a half-open row range into deterministic worker chunks."""
    row_count = max(0, row_end - row_start)
    if row_count == 0:
        return []
    bounded_workers = max(1, worker_count)
    rows_per_worker = (
        row_count + bounded_workers - 1
    ) // bounded_workers
    return [
        (start, min(row_end, start + rows_per_worker))
        for start in range(row_start, row_end, rows_per_worker)
    ]


def _should_parallelize_rows(row_count: int, worker_count: int) -> bool:
    """Return whether row chunking has enough work to pay for threading."""
    return (
        worker_count > 1
        and row_count >= max(384, worker_count * 16)
    )


def _parallel_range_results(
    range_start: int,
    range_end: int,
    worker_count: int,
    range_function: Callable[[int, int], Any],
) -> Iterator[Any]:
    """Yield results from applying range_function to split row ranges."""
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        yield from executor.map(
            lambda row_range: range_function(*row_range),
            _parallel_row_ranges(range_start, range_end, worker_count),
        )


def _merge_count_map(target: dict[Any, int], source: dict[Any, int]) -> None:
    """Add source counts into target in place."""
    for key, count in source.items():
        target[key] = target.get(key, 0) + count


def _cleanup_cells_for_mesh_serialization(
    cells: np.ndarray,
    output_fileformat: str,
    split_rotation: int,
    parallel_workers: int = 1,
) -> None:
    """Clean cell geometry before serial mesh writes or topology scans."""
    def cleanup_rows(row_start: int, row_end: int) -> None:
        for row_index in range(row_start, row_end):
            for current_cell in cells[row_index]:
                if current_cell is not None:
                    current_cell.remove_geometry_collapsed_by_mesh_serialization(
                        output_fileformat=output_fileformat,
                        split_rotation=split_rotation,
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


def edge_xy_signature(coord0: Coordinate, coord1: Coordinate) -> XYEdge:
    """Return an orientation-independent XY signature for an edge."""
    return tuple(
        sorted(
            (
                (coord0[0], coord0[1]),
                (coord1[0], coord1[1]),
            )
        )
    )


def edge_3d_signature(coord0: Coordinate, coord1: Coordinate) -> Edge3D:
    """Return an orientation-independent 3D signature for an edge."""
    coord0_key = (
        coord0
        if isinstance(coord0, tuple) and len(coord0) == 3
        else tuple(coord0[:3])
    )
    coord1_key = (
        coord1
        if isinstance(coord1, tuple) and len(coord1) == 3
        else tuple(coord1[:3])
    )
    return tuple(sorted((coord0_key, coord1_key)))


def boundary_edge_map_from_meshes(
    meshes: Iterable[SurfaceMesh] | None,
    output_fileformat: str | None = None,
    serialized_vertices: SerializedVertexCache | None = None,
) -> dict[XYEdge, Edge3D]:
    """Return emitted boundary edges for quads and triangulated polygons."""
    edge_counts: dict[XYEdge, int] = {}
    edge_coords: dict[XYEdge, Edge3D] = {}
    if serialized_vertices is None:
        serialized_vertices = {}
    if meshes is None:
        return {}

    def add_edge(coord0: Coordinate, coord1: Coordinate) -> None:
        output_coord0 = _serialized_vertex_from_cache(
            coord0,
            output_fileformat,
            serialized_vertices,
        )
        output_coord1 = _serialized_vertex_from_cache(
            coord1,
            output_fileformat,
            serialized_vertices,
        )
        footprint = edge_xy_signature(output_coord0, output_coord1)
        edge_counts[footprint] = edge_counts.get(footprint, 0) + 1
        edge_coords[footprint] = (output_coord0, output_coord1)

    for mesh in meshes:
        if isinstance(mesh, quad):
            coords = [v.coords for v in mesh.vl if v is not None]
            for ci in range(len(coords)):
                add_edge(coords[ci], coords[(ci + 1) % len(coords)])
        elif isinstance(mesh, shapely.Polygon):
            rings = [mesh.exterior, *mesh.interiors]
            for ring in rings:
                coords = ring.coords
                for ci in range(len(coords) - 1):
                    add_edge(coords[ci], coords[ci + 1])

    return {
        footprint: coords
        for footprint, coords in edge_coords.items()
        if edge_counts[footprint] == 1
    }


def normalize_coordinate_to_match_mesh_serialization(
    value: float,
    fileformat: str,
    decimals: int = MESH_OUTPUT_SERIALIZATION_DECIMAL_PRECISION,
) -> float:
    """Return a coordinate as it will be serialized in mesh output."""
    if fileformat == "STLb":
        return _normalize_stlb_coordinate(value, decimals)
    return _normalize_decimal_coordinate(value, decimals)


def _normalize_decimal_coordinate(value: float, decimals: int) -> float:
    """Return a rounded text-format coordinate value."""
    return round(value, decimals) + 0.0


def _normalize_stlb_coordinate(value: float, decimals: int) -> float:
    """Return a rounded binary-STL coordinate value."""
    return (
        round(
            BINARY_FLOAT.unpack(BINARY_FLOAT.pack(value))[0] + 0.0,
            decimals,
        )
        + 0.0
    )


def normalize_vertex_to_match_mesh_serialization(
    coord: Coordinate,
    fileformat: str,
    decimals: int = MESH_OUTPUT_SERIALIZATION_DECIMAL_PRECISION,
) -> tuple[float, ...]:
    """Return a vertex as it will be serialized in mesh output."""
    if len(coord) < 3:
        normalize = (
            _normalize_stlb_coordinate
            if fileformat == "STLb"
            else _normalize_decimal_coordinate
        )
        return tuple(
            normalize(value, decimals)
            for value in coord
        )

    if fileformat == "STLb":
        pack_float = BINARY_FLOAT.pack
        unpack_float = BINARY_FLOAT.unpack
        return (
            round(
                unpack_float(pack_float(coord[0]))[0] + 0.0,
                decimals,
            )
            + 0.0,
            round(
                unpack_float(pack_float(coord[1]))[0] + 0.0,
                decimals,
            )
            + 0.0,
            round(
                unpack_float(pack_float(coord[2]))[0] + 0.0,
                decimals,
            )
            + 0.0,
        )

    return (
        round(coord[0], decimals) + 0.0,
        round(coord[1], decimals) + 0.0,
        round(coord[2], decimals) + 0.0,
    )


def _serialized_vertex_from_cache(
    coord: Coordinate,
    output_fileformat: str | None,
    serialized_vertices: SerializedVertexCache,
) -> tuple[float, ...]:
    """Return a cached coordinate normalized to emitted mesh precision."""
    coord_key = coord if isinstance(coord, tuple) else tuple(coord)
    output_coord = serialized_vertices.get(coord_key)
    if output_coord is not None:
        return output_coord

    if output_fileformat is None:
        output_coord = coord_key[:3]
    else:
        output_coord = normalize_vertex_to_match_mesh_serialization(
            coord_key,
            output_fileformat,
        )
    serialized_vertices[coord_key] = output_coord
    return output_coord


def surface_mesh_edge_counts(
    meshes: Iterable[SurfaceMesh | None],
    split_rotation: int,
    output_fileformat: str,
    serialized_vertices: SerializedVertexCache | None = None,
) -> dict[Edge3D, int]:
    """Count serialized triangle edges emitted by surface meshes."""
    edge_counts, _directed_edge_counts = surface_mesh_edge_usage(
        meshes,
        split_rotation,
        output_fileformat,
        serialized_vertices,
        include_directed=False,
    )
    return edge_counts


def surface_mesh_edge_usage(
    meshes: Iterable[SurfaceMesh | None],
    split_rotation: int,
    output_fileformat: str,
    serialized_vertices: SerializedVertexCache | None = None,
    include_directed: bool = True,
) -> tuple[dict[Edge3D, int], dict[DirectedEdge3D, int]]:
    """Count unordered and directed serialized triangle edges."""
    edge_counts: dict[Edge3D, int] = {}
    directed_edge_counts: dict[DirectedEdge3D, int] = {}
    if serialized_vertices is None:
        serialized_vertices = {}

    def add_triangle_edge(coord0: Coordinate, coord1: Coordinate) -> None:
        output_coord0 = _serialized_vertex_from_cache(
            coord0,
            output_fileformat,
            serialized_vertices,
        )
        output_coord1 = _serialized_vertex_from_cache(
            coord1,
            output_fileformat,
            serialized_vertices,
        )
        edge_key = edge_3d_signature(output_coord0, output_coord1)
        edge_counts[edge_key] = edge_counts.get(edge_key, 0) + 1
        if not include_directed:
            return
        directed_edge = (output_coord0, output_coord1)
        directed_edge_counts[directed_edge] = (
            directed_edge_counts.get(directed_edge, 0) + 1
        )

    for mesh in meshes:
        if mesh is None:
            continue
        if isinstance(mesh, quad):
            for triangle in mesh.get_triangles(split_rotation):
                for index, triangle_vertex in enumerate(triangle):
                    add_triangle_edge(
                        triangle_vertex.coords,
                        triangle[(index + 1) % len(triangle)].coords,
                    )
        elif isinstance(mesh, shapely.Polygon):
            coords = mesh.exterior.coords
            for index in range(len(coords) - 1):
                add_triangle_edge(coords[index], coords[index + 1])
    return edge_counts, directed_edge_counts


def directed_edges_are_balanced(
    edge_counts: dict[Edge3D, int],
    directed_edge_counts: dict[DirectedEdge3D, int],
    edges_to_check: set[Edge3D] | None = None,
) -> bool:
    """Return whether every checked manifold edge has opposite directions."""
    check_edges = edges_to_check if edges_to_check is not None else set(edge_counts)
    for edge_key in check_edges:
        if edge_counts.get(edge_key, 0) != 2:
            continue
        forward = directed_edge_counts.get((edge_key[0], edge_key[1]), 0)
        reverse = directed_edge_counts.get((edge_key[1], edge_key[0]), 0)
        if forward != 1 or reverse != 1:
            return False
    return True


def has_positive_z_overused_surface_contact(
    top_meshes: list[SurfaceMesh | None],
    bottom_meshes: list[SurfaceMesh | None],
    split_rotation: int,
    output_fileformat: str,
) -> bool:
    """Return True when current top/bottom surfaces share an overused Z>0 edge."""
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
    for edge_key, top_count in top_edges.items():
        bottom_count = bottom_edges.get(edge_key, 0)
        if bottom_count == 0:
            continue
        if edge_key[0][2] <= 0 or edge_key[1][2] <= 0:
            continue
        if top_count + bottom_count > 2:
            return True
    return False


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
        side = _edge_cardinal_side(
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


def triangle_collapses_after_mesh_serialization(
    triangle: Sequence[Coordinate],
    fileformat: str,
    decimals: int = MESH_OUTPUT_SERIALIZATION_DECIMAL_PRECISION,
    serialized_vertices: SerializedVertexCache | None = None,
) -> bool:
    """Return whether a triangle is degenerate after mesh serialization."""
    if (
        serialized_vertices is None
        or decimals != MESH_OUTPUT_SERIALIZATION_DECIMAL_PRECISION
    ):
        p0, p1, p2 = [
            normalize_vertex_to_match_mesh_serialization(
                coord=coord,
                fileformat=fileformat,
                decimals=decimals,
            )
            for coord in triangle
        ]
    else:
        p0, p1, p2 = [
            _serialized_vertex_from_cache(
                coord,
                fileformat,
                serialized_vertices,
            )
            for coord in triangle
        ]
    if len({p0, p1, p2}) < 3:
        return True

    a = (
        p1[0] - p0[0],
        p1[1] - p0[1],
        p1[2] - p0[2],
    )
    b = (
        p2[0] - p0[0],
        p2[1] - p0[1],
        p2[2] - p0[2],
    )
    cross = (
        round(a[1] * b[2] - a[2] * b[1], 12),
        round(a[2] * b[0] - a[0] * b[2], 12),
        round(a[0] * b[1] - a[1] * b[0], 12),
    )
    return cross == (0.0, 0.0, 0.0)


def _serialized_triangle_collapses(
    triangle: Sequence[Coordinate],
) -> bool:
    """Return whether already-serialized 3D triangle coordinates collapse."""
    p0, p1, p2 = triangle
    if len({tuple(p0[:3]), tuple(p1[:3]), tuple(p2[:3])}) < 3:
        return True

    a = (
        p1[0] - p0[0],
        p1[1] - p0[1],
        p1[2] - p0[2],
    )
    b = (
        p2[0] - p0[0],
        p2[1] - p0[1],
        p2[2] - p0[2],
    )
    cross = (
        round(a[1] * b[2] - a[2] * b[1], 12),
        round(a[2] * b[0] - a[0] * b[2], 12),
        round(a[0] * b[1] - a[1] * b[0], 12),
    )
    return cross == (0.0, 0.0, 0.0)


def triangle_xy_collapses_after_mesh_serialization(
    triangle: Sequence[Coordinate],
    fileformat: str,
    decimals: int = MESH_OUTPUT_SERIALIZATION_DECIMAL_PRECISION,
    serialized_vertices: SerializedVertexCache | None = None,
) -> bool:
    """Return whether a surface triangle has no serialized XY area."""
    if (
        serialized_vertices is None
        or decimals != MESH_OUTPUT_SERIALIZATION_DECIMAL_PRECISION
    ):
        p0, p1, p2 = [
            normalize_vertex_to_match_mesh_serialization(
                coord=coord,
                fileformat=fileformat,
                decimals=decimals,
            )[:2]
            for coord in triangle
        ]
    else:
        p0, p1, p2 = [
            _serialized_vertex_from_cache(
                coord,
                fileformat,
                serialized_vertices,
            )[:2]
            for coord in triangle
        ]
    if len({p0, p1, p2}) < 3:
        return True

    a = (
        p1[0] - p0[0],
        p1[1] - p0[1],
    )
    b = (
        p2[0] - p0[0],
        p2[1] - p0[1],
    )
    cross = round(a[0] * b[1] - a[1] * b[0], 12)
    return cross == 0.0


def _serialized_triangle_xy_collapses(
    triangle: Sequence[Coordinate],
) -> bool:
    """Return whether already-serialized triangle XY coordinates collapse."""
    p0, p1, p2 = triangle
    p0_xy = tuple(p0[:2])
    p1_xy = tuple(p1[:2])
    p2_xy = tuple(p2[:2])
    if len({p0_xy, p1_xy, p2_xy}) < 3:
        return True

    a = (
        p1_xy[0] - p0_xy[0],
        p1_xy[1] - p0_xy[1],
    )
    b = (
        p2_xy[0] - p0_xy[0],
        p2_xy[1] - p0_xy[1],
    )
    cross = round(a[0] * b[1] - a[1] * b[0], 12)
    return cross == 0.0


def _serialized_xy_key(
    coord: Coordinate,
    output_fileformat: str,
) -> tuple[float, float]:
    """Return the XY coordinate key that mesh serialization will emit."""
    return normalize_vertex_to_match_mesh_serialization(
        (coord[0], coord[1], 0.0),
        output_fileformat,
    )[:2]


def _line_serialized_xy_signature(
    line: shapely.LineString,
    output_fileformat: str,
) -> XYEdge | None:
    """Return an orientation-independent serialized XY line signature."""
    coords = list(line.coords)
    if len(coords) < 2:
        return None

    start_xy = _serialized_xy_key(coords[0], output_fileformat)
    end_xy = _serialized_xy_key(coords[-1], output_fileformat)
    if start_xy == end_xy:
        return None
    return tuple(sorted((start_xy, end_xy)))


def _line_with_serialized_xy(
    line: shapely.LineString,
    output_fileformat: str,
) -> shapely.LineString | None:
    """Return a 2D line with endpoints at serialized XY coordinates."""
    coords = list(line.coords)
    if len(coords) < 2:
        return None

    start_xy = _serialized_xy_key(coords[0], output_fileformat)
    end_xy = _serialized_xy_key(coords[-1], output_fileformat)
    if start_xy == end_xy:
        return None
    return shapely.LineString([start_xy, end_xy])


def _boundary_line_map_by_serialized_xy(
    edges: Sequence[shapely.LineString],
    output_fileformat: str,
) -> dict[XYEdge, list[shapely.LineString]]:
    """Return serialized XY edge map for edges used exactly once."""
    edge_map: dict[XYEdge, list[shapely.LineString]] = {}
    for edge in edges:
        edge_key = _line_serialized_xy_signature(
            edge,
            output_fileformat,
        )
        if edge_key is None:
            continue
        edge_map.setdefault(edge_key, []).append(edge)
    return {
        edge_key: edge_matches
        for edge_key, edge_matches in edge_map.items()
        if len(edge_matches) == 1
    }


def _canonicalize_clipped_triangles_by_serialized_xy(
    triangles: Sequence[shapely.Polygon],
    output_fileformat: str,
) -> list[shapely.Polygon]:
    """Merge local clipped triangle vertices that will serialize to one XY."""
    canonical_xy_by_key: dict[tuple[float, float], tuple[float, float]] = {}
    duplicate_keys: set[tuple[float, float]] = set()

    for triangle in triangles:
        for coord in triangle.exterior.coords[:-1]:
            key = _serialized_xy_key(coord, output_fileformat)
            xy = (coord[0], coord[1])
            if key not in canonical_xy_by_key:
                canonical_xy_by_key[key] = xy
            elif canonical_xy_by_key[key] != xy:
                duplicate_keys.add(key)

    canonicalized: list[shapely.Polygon] = []
    for triangle in triangles:
        coords_2d: list[tuple[float, float]] = []
        for coord in triangle.exterior.coords[:-1]:
            key = _serialized_xy_key(coord, output_fileformat)
            if key in duplicate_keys:
                coords_2d.append(canonical_xy_by_key[key])
            else:
                coords_2d.append((coord[0], coord[1]))

        if len(coords_2d) != 3 or len(set(coords_2d)) < 3:
            continue
        coords_3d = [(coord[0], coord[1], 0.0) for coord in coords_2d]
        if triangle_xy_collapses_after_mesh_serialization(
            coords_3d,
            output_fileformat,
        ):
            continue

        canonical_triangle = shapely.Polygon([*coords_2d, coords_2d[0]])
        if canonical_triangle.is_empty or canonical_triangle.area <= 0:
            continue
        canonicalized.append(canonical_triangle)

    return canonicalized


def polygon_normalized_to_match_mesh_serialization(
    polygon: shapely.Polygon,
    fileformat: str,
    decimals: int = MESH_OUTPUT_SERIALIZATION_DECIMAL_PRECISION,
    serialized_vertices: SerializedVertexCache | None = None,
) -> shapely.Polygon | None:
    """Return a serialized-coordinate polygon, or None if it collapses."""
    coords = polygon.exterior.coords
    if len(coords) != 4 or coords[0] != coords[-1]:
        raise ValueError("Expected a closed triangular Polygon.")

    def normalized_coord(coord: Coordinate) -> tuple[float, ...]:
        if (
            serialized_vertices is not None
            and decimals == MESH_OUTPUT_SERIALIZATION_DECIMAL_PRECISION
        ):
            return _serialized_vertex_from_cache(
                coord,
                fileformat,
                serialized_vertices,
            )
        return normalize_vertex_to_match_mesh_serialization(
            coord=coord,
            fileformat=fileformat,
            decimals=decimals,
        )

    exterior = [
        normalized_coord(coord)
        for coord in polygon.exterior.coords
    ]
    interiors = [
        [
            normalized_coord(coord)
            for coord in ring.coords
        ]
        for ring in polygon.interiors
    ]
    if _serialized_triangle_collapses(exterior[:3]):
        return None
    return shapely.Polygon(exterior, interiors)


def surface_polygon_normalized_to_match_mesh_serialization(
    polygon: shapely.Polygon,
    fileformat: str,
    decimals: int = MESH_OUTPUT_SERIALIZATION_DECIMAL_PRECISION,
    serialized_vertices: SerializedVertexCache | None = None,
) -> shapely.Polygon | None:
    """Return a serialized surface triangle, or None if it has no area."""
    normalized_polygon = polygon_normalized_to_match_mesh_serialization(
        polygon,
        fileformat,
        decimals,
        serialized_vertices,
    )
    if normalized_polygon is None:
        return None

    exterior = normalized_polygon.exterior.coords
    if _serialized_triangle_xy_collapses(
        triangle=(exterior[0], exterior[1], exterior[2]),
    ):
        return None
    return normalized_polygon


def _iter_polygon_parts(geometry: shapely.Geometry) -> list[shapely.Polygon]:
    """Return non-empty polygon parts from a Shapely geometry."""
    if geometry.is_empty:
        return []
    if isinstance(geometry, shapely.Polygon):
        return [geometry] if geometry.area > 0 else []
    polygons: list[shapely.Polygon] = []
    if hasattr(geometry, "geoms"):
        for child in geometry.geoms:
            polygons.extend(_iter_polygon_parts(child))
    return polygons


def _has_valid_z_plane(polygon: shapely.Polygon) -> bool:
    """Return whether a triangular polygon can interpolate Z values."""
    coords = polygon.exterior.coords
    if len(coords) != 4 or len(coords[0]) < 3:
        return False
    p1 = coords[0]
    p2 = coords[1]
    p3 = coords[2]
    xy_cross = (
        (p2[0] - p1[0]) * (p3[1] - p1[1])
        - (p2[1] - p1[1]) * (p3[0] - p1[0])
    )
    return abs(xy_cross) >= 1e-6


def _nudge_cell_points(
    W: float,
    E: float,
    N: float,
    S: float,
) -> dict[str, tuple[float, float]]:
    mid_x = (W + E) / 2
    mid_y = (N + S) / 2
    return {
        "SW": (W, S),
        "SE": (E, S),
        "NE": (E, N),
        "NW": (W, N),
        "Smid": (mid_x, S),
        "Emid": (E, mid_y),
        "Nmid": (mid_x, N),
        "Wmid": (W, mid_y),
    }


def _cell_bounds_for_location(
    cell_j: int,
    cell_i: int,
    cell_size: float,
    offsetx: float,
    offsety: float,
) -> tuple[float, float, float, float]:
    """Return W, E, N, S bounds for a padded raster cell location."""
    cell_w = (cell_i - 1) * cell_size - offsetx
    cell_e = cell_w + cell_size
    cell_n = -((cell_j - 1) * cell_size) + offsety
    cell_s = cell_n - cell_size
    return cell_w, cell_e, cell_n, cell_s


def _cell_side_values(
    cell_j: int,
    cell_i: int,
    cell_size: float,
    offsetx: float,
    offsety: float,
    output_fileformat: str,
) -> dict[str, float]:
    """Return serialized side coordinates for a padded raster cell."""
    cell_w, cell_e, cell_n, cell_s = _cell_bounds_for_location(
        cell_j,
        cell_i,
        cell_size,
        offsetx,
        offsety,
    )
    return _side_values_from_bounds(
        cell_w,
        cell_e,
        cell_n,
        cell_s,
        output_fileformat,
    )


def _side_values_from_bounds(
    W: float,
    E: float,
    N: float,
    S: float,
    output_fileformat: str,
) -> dict[str, float]:
    """Return serialized side coordinates from W/E/N/S bounds."""
    return {
        "N": normalize_coordinate_to_match_mesh_serialization(
            N,
            output_fileformat,
        ),
        "S": normalize_coordinate_to_match_mesh_serialization(
            S,
            output_fileformat,
        ),
        "E": normalize_coordinate_to_match_mesh_serialization(
            E,
            output_fileformat,
        ),
        "W": normalize_coordinate_to_match_mesh_serialization(
            W,
            output_fileformat,
        ),
    }


def _edge_cardinal_side(
    footprint: XYEdge,
    side_values: dict[str, float],
) -> str | None:
    """Return the cardinal side occupied by a serialized XY edge."""
    (x0, y0), (x1, y1) = footprint
    if y0 == side_values["N"] and y1 == side_values["N"]:
        return "N"
    if y0 == side_values["S"] and y1 == side_values["S"]:
        return "S"
    if x0 == side_values["E"] and x1 == side_values["E"]:
        return "E"
    if x0 == side_values["W"] and x1 == side_values["W"]:
        return "W"
    return None


def _nudge_full_cell_footprint(
    W: float,
    E: float,
    N: float,
    S: float,
) -> shapely.Polygon:
    points = _nudge_cell_points(W, E, N, S)
    return shapely.Polygon(
        [
            points["SW"],
            points["SE"],
            points["NE"],
            points["NW"],
            points["SW"],
        ]
    )


def _nudge_keep_footprint(
    affected_corners: Sequence[IntermediateCorner],
    W: float,
    E: float,
    N: float,
    S: float,
) -> shapely.Polygon | None:
    """Return the shared nudge keep footprint for an affected-corner set."""
    vertex_names = _nudge_keep_vertex_names(affected_corners)
    if vertex_names is None:
        return None
    points = _nudge_cell_points(W, E, N, S)
    return shapely.Polygon([points[name] for name in vertex_names])


def _nudge_keep_vertex_names(
    affected_corners: Sequence[IntermediateCorner],
) -> list[str] | None:
    """Return vertex names for the shared nudge keep footprint."""
    affected = frozenset(affected_corners)
    c = IntermediateCorner
    footprints = {
        frozenset([c.SW]): ["Smid", "SE", "NE", "NW", "Wmid"],
        frozenset([c.SE]): ["SW", "Smid", "Emid", "NE", "NW"],
        frozenset([c.NE]): ["SW", "SE", "Emid", "Nmid", "NW"],
        frozenset([c.NW]): ["SW", "SE", "NE", "Nmid", "Wmid"],
        frozenset([c.SW, c.SE]): ["Wmid", "Emid", "NE", "NW"],
        frozenset([c.SE, c.NE]): ["SW", "Smid", "Nmid", "NW"],
        frozenset([c.NW, c.NE]): ["SW", "SE", "Emid", "Wmid"],
        frozenset([c.SW, c.NW]): ["Smid", "SE", "NE", "Nmid"],
        frozenset([c.SW, c.NE]): [
            "Smid",
            "SE",
            "Emid",
            "Nmid",
            "NW",
            "Wmid",
        ],
        frozenset([c.SE, c.NW]): [
            "SW",
            "Smid",
            "Emid",
            "NE",
            "Nmid",
            "Wmid",
        ],
        frozenset([c.SW, c.NW, c.NE]): ["Smid", "SE", "Emid"],
        frozenset([c.NW, c.SW, c.SE]): ["Nmid", "Emid", "NE"],
        frozenset([c.SW, c.SE, c.NE]): ["Wmid", "Nmid", "NW"],
        frozenset([c.SE, c.NE, c.NW]): ["SW", "Smid", "Wmid"],
    }
    return footprints.get(affected)


def _nudge_keep_footprint_splits_side(
    affected_corners: Sequence[IntermediateCorner],
    side: str,
) -> bool:
    """Return whether a nudge keep footprint inserts a midpoint on a side."""
    vertex_names = _nudge_keep_vertex_names(affected_corners)
    return (
        vertex_names is not None
        and NUDGE_SIDE_MIDPOINT_NAME[side] in vertex_names
    )


def _nudge_keep_footprint_split_sides(
    affected_corners: Sequence[IntermediateCorner],
) -> set[str]:
    """Return sides split by the nudge keep footprint."""
    return {
        side
        for side in CARDINAL_DIRECTIONS
        if _nudge_keep_footprint_splits_side(affected_corners, side)
    }


def _nudge_split_side_endpoint_xy(
    split_sides: set[str],
    W: float,
    E: float,
    N: float,
    S: float,
    output_fileformat: str,
) -> set[tuple[float, float]]:
    """Return serialized XY endpoints for split-only side preservation."""
    points = _nudge_cell_points(W, E, N, S)
    endpoints: set[tuple[float, float]] = set()
    for side in split_sides:
        for point_name in NUDGE_SIDE_ENDPOINT_NAMES[side]:
            endpoints.add(
                normalize_vertex_to_match_mesh_serialization(
                    (*points[point_name], 0.0),
                    output_fileformat,
                )[:2]
            )
    return endpoints


def _nudge_split_side_endpoint_edges(
    split_sides: set[str],
    W: float,
    E: float,
    N: float,
    S: float,
    output_fileformat: str,
) -> set[XYEdge]:
    """Return serialized XY half-edges for split-only side preservation."""
    points = _nudge_cell_points(W, E, N, S)
    edges: set[XYEdge] = set()
    for side in split_sides:
        for start_name, end_name in NUDGE_SIDE_SEGMENT_NAMES[side]:
            start_xy = normalize_vertex_to_match_mesh_serialization(
                (*points[start_name], 0.0),
                output_fileformat,
            )[:2]
            end_xy = normalize_vertex_to_match_mesh_serialization(
                (*points[end_name], 0.0),
                output_fileformat,
            )[:2]
            edges.add(tuple(sorted((start_xy, end_xy))))
    return edges


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


def _nudge_adjusted_keep_surface_planes(
    affected_corners: Sequence[IntermediateCorner],
    corner_vertices: dict[IntermediateCorner, vertex],
    W: float,
    E: float,
    N: float,
    S: float,
    forced_midpoint_z: float | None = None,
    midpoint_corner_vertices: dict[IntermediateCorner, vertex] | None = None,
) -> list[shapely.Polygon]:
    """Return full-precision nudge-adjusted keep-footprint triangles."""
    vertex_names = _nudge_keep_vertex_names(affected_corners)
    if vertex_names is None:
        return []

    footprint_2d = shapely.Polygon(
        [_nudge_cell_points(W, E, N, S)[name] for name in vertex_names]
    )
    return _nudge_adjusted_surface_planes(
        footprint_2d,
        corner_vertices,
        W,
        E,
        N,
        S,
        forced_midpoint_z=forced_midpoint_z,
        midpoint_corner_vertices=midpoint_corner_vertices,
    )


def _nudge_adjusted_surface_planes(
    geometry: shapely.Geometry,
    corner_vertices: dict[IntermediateCorner, vertex | None],
    W: float,
    E: float,
    N: float,
    S: float,
    forced_midpoint_z: float | None = None,
    midpoint_corner_vertices: dict[IntermediateCorner, vertex] | None = None,
) -> list[shapely.Polygon]:
    """Return nudge surface planes with optional alternate midpoint Z."""
    points = _nudge_cell_points(W, E, N, S)
    name_by_point = {point: name for name, point in points.items()}
    corner_by_name = {
        "NW": IntermediateCorner.NW,
        "NE": IntermediateCorner.NE,
        "SW": IntermediateCorner.SW,
        "SE": IntermediateCorner.SE,
    }
    midpoint_source = midpoint_corner_vertices or corner_vertices

    def z_for_name(name: str) -> float | None:
        if name in corner_by_name:
            corner_vertex = corner_vertices.get(corner_by_name[name])
            return None if corner_vertex is None else corner_vertex.coords[2]
        if name in NUDGE_MIDPOINT_CORNERS_BY_NAME:
            if forced_midpoint_z is not None:
                return forced_midpoint_z
            midpoint_vertices = [
                midpoint_source.get(corner)
                for corner in NUDGE_MIDPOINT_CORNERS_BY_NAME[name]
            ]
            if any(corner_vertex is None for corner_vertex in midpoint_vertices):
                return None
            return (
                midpoint_vertices[0].coords[2]
                + midpoint_vertices[1].coords[2]
            ) / 2
        return None

    planes: list[shapely.Polygon] = []
    for polygon in _iter_polygon_parts(geometry):
        triangles = shapely.constrained_delaunay_triangles(polygon)
        for triangle in _iter_polygon_parts(triangles):
            coords_3d = []
            coords = triangle.exterior.coords
            for index in range(len(coords) - 1):
                x, y = coords[index]
                point_name = name_by_point.get((x, y))
                if point_name is None:
                    raise ValueError(
                        "Nudge adjusted plane includes an unexpected vertex."
                    )
                z = z_for_name(point_name)
                if z is None:
                    return []
                coords_3d.append((x, y, z))
            coords_3d.append(coords_3d[0])
            planes.append(shapely.Polygon(coords_3d))
    return planes


def _nudge_midpoint_z_by_xy(
    W: float,
    E: float,
    N: float,
    S: float,
    midpoint_corner_vertices: dict[IntermediateCorner, vertex] | None,
    output_fileformat: str,
    midpoint_z_by_name: dict[str, float] | None = None,
) -> dict[tuple[float, float], float]:
    """Return serialized midpoint XY keys mapped to their requested Z."""
    points = _nudge_cell_points(W, E, N, S)
    z_by_xy: dict[tuple[float, float], float] = {}
    for midpoint_name, midpoint_corners in NUDGE_MIDPOINT_CORNERS_BY_NAME.items():
        midpoint_z = None
        if midpoint_z_by_name and midpoint_name in midpoint_z_by_name:
            midpoint_z = midpoint_z_by_name[midpoint_name]
        elif midpoint_corner_vertices is not None:
            midpoint_vertices = [
                midpoint_corner_vertices.get(corner)
                for corner in midpoint_corners
            ]
            if any(corner_vertex is None for corner_vertex in midpoint_vertices):
                continue
            midpoint_z = (
                midpoint_vertices[0].coords[2]
                + midpoint_vertices[1].coords[2]
            ) / 2
        else:
            continue
        midpoint_xy = normalize_vertex_to_match_mesh_serialization(
            (*points[midpoint_name], 0.0),
            output_fileformat,
        )[:2]
        z_by_xy[midpoint_xy] = normalize_coordinate_to_match_mesh_serialization(
            midpoint_z,
            output_fileformat,
        )

    return z_by_xy


def _surface_polygons_with_midpoint_z(
    surface_polygons: list[shapely.Polygon],
    W: float,
    E: float,
    N: float,
    S: float,
    midpoint_corner_vertices: dict[IntermediateCorner, vertex] | None,
    output_fileformat: str,
    midpoint_z_by_name: dict[str, float] | None = None,
) -> list[shapely.Polygon]:
    """Return surfaces with nudge midpoint vertices set from a source surface."""
    z_by_xy = _nudge_midpoint_z_by_xy(
        W,
        E,
        N,
        S,
        midpoint_corner_vertices,
        output_fileformat,
        midpoint_z_by_name,
    )
    if not z_by_xy:
        return surface_polygons

    return _surface_polygons_with_z_overrides(
        surface_polygons,
        z_by_xy,
        output_fileformat,
    )


def _surface_polygons_with_z_overrides(
    surface_polygons: list[shapely.Polygon],
    z_by_xy: dict[tuple[float, float], float],
    output_fileformat: str,
) -> list[shapely.Polygon]:
    """Return surfaces with vertices at matching serialized XY set to Z."""
    if not z_by_xy:
        return surface_polygons

    def adjusted_ring(ring: shapely.LinearRing) -> list[tuple[float, ...]]:
        coords = []
        for coord in ring.coords:
            normalized_xy = normalize_vertex_to_match_mesh_serialization(
                coord,
                output_fileformat,
            )[:2]
            coords.append(
                (coord[0], coord[1], z_by_xy.get(normalized_xy, coord[2]))
            )
        return coords

    adjusted_polygons = []
    for surface_polygon in surface_polygons:
        interiors = [
            adjusted_ring(interior)
            for interior in surface_polygon.interiors
        ]
        adjusted_polygon = shapely.Polygon(
            adjusted_ring(surface_polygon.exterior),
            interiors,
        )
        normalized_polygon = (
            surface_polygon_normalized_to_match_mesh_serialization(
                adjusted_polygon,
                output_fileformat,
            )
        )
        if normalized_polygon is not None:
            adjusted_polygons.append(normalized_polygon)
    return adjusted_polygons


def _surface_vertex_z_overrides_by_xy(
    surface_polygons: list[shapely.Polygon] | None,
    output_fileformat: str,
) -> dict[tuple[float, float], float]:
    """Return serialized Z values for vertices already on a surface."""
    z_by_xy: dict[tuple[float, float], float] = {}
    for surface_polygon in surface_polygons or []:
        for coord in surface_polygon.exterior.coords[:-1]:
            normalized = normalize_vertex_to_match_mesh_serialization(
                coord,
                output_fileformat,
            )
            z_by_xy.setdefault(normalized[:2], normalized[2])
    return z_by_xy


def _z0_adjusted_keep_surface_planes(
    affected_corners: Sequence[IntermediateCorner],
    corner_vertices: dict[IntermediateCorner, vertex],
    W: float,
    E: float,
    N: float,
    S: float,
) -> list[shapely.Polygon]:
    """Return full-precision nudge-adjusted triangles with Z0 midpoints."""
    return _nudge_adjusted_keep_surface_planes(
        affected_corners,
        corner_vertices,
        W,
        E,
        N,
        S,
        forced_midpoint_z=0.0,
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
    cell_w, cell_e, cell_n, cell_s = _cell_bounds_for_location(
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
        _cell_side_values(
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


def _quad_corner_vertices_by_xy(
    surface_quad: quad,
    W: float,
    E: float,
    N: float,
    S: float,
) -> dict[IntermediateCorner, vertex | None]:
    """Return cell corner vertices by XY, independent of quad vertex order."""
    corner_xy = {
        IntermediateCorner.NW: (W, N),
        IntermediateCorner.NE: (E, N),
        IntermediateCorner.SW: (W, S),
        IntermediateCorner.SE: (E, S),
    }
    corner_vertices: dict[IntermediateCorner, vertex | None] = {
        corner: None for corner in IntermediateCorner
    }
    for quad_vertex in surface_quad.vl:
        if quad_vertex is None:
            continue
        for corner, xy in corner_xy.items():
            if (
                abs(quad_vertex.coords[0] - xy[0]) <= 1e-9
                and abs(quad_vertex.coords[1] - xy[1]) <= 1e-9
            ):
                corner_vertices[corner] = quad_vertex
                break
    return corner_vertices


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
                    _cell_side_values(
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
        side_values = _cell_side_values(
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

        cell_w, cell_e, cell_n, cell_s = _cell_bounds_for_location(
            cell_j,
            cell_i,
            cell_size,
            offsetx,
            offsety,
        )
        points = _nudge_cell_points(cell_w, cell_e, cell_n, cell_s)
        planes = _surface_planes_from_current_geometry(
            current_cell.topquad,
            current_cell.topSurfacePolygons,
            split_rotation,
        )
        corner_vertices = _quad_corner_vertices_by_xy(
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
        cell_w, cell_e, cell_n, cell_s = _cell_bounds_for_location(
            cell_j,
            cell_i,
            cell_size,
            offsetx,
            offsety,
        )
        points = _nudge_cell_points(cell_w, cell_e, cell_n, cell_s)
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


def _surface_planes_from_current_geometry(
    surface_quad: quad | None,
    surface_polygons: list[shapely.Polygon] | None,
    split_rotation: int,
) -> list[shapely.Polygon]:
    """Return interpolation planes for the current surface representation."""
    if surface_polygons:
        planes = [
            surface_polygon
            for surface_polygon in surface_polygons
            if _has_valid_z_plane(surface_polygon)
        ]
        if planes:
            return planes
    if surface_quad is None:
        return []
    return surface_quad.get_triangles_in_polygons(split_rotation)


def _current_surface_footprint(
    surface_polygons: list[shapely.Polygon] | None,
    fallback: shapely.Polygon,
    surface_quad: quad | None = None,
) -> shapely.Geometry:
    """Return the emitted XY footprint for the current surface."""
    if surface_polygons:
        return _union_polygon_footprint(surface_polygons, fallback)
    if surface_quad is not None:
        coords = [
            surface_vertex.coords[:2]
            for surface_vertex in surface_quad.vl
            if surface_vertex is not None
        ]
        if len(coords) >= 3:
            return shapely.Polygon(coords)
    return fallback


def _union_polygon_footprint(
    geometries: list[shapely.Geometry] | None,
    fallback: shapely.Polygon,
) -> shapely.Geometry:
    if not geometries:
        return fallback
    polygons: list[shapely.Polygon] = []
    for geometry in geometries:
        polygons.extend(_iter_polygon_parts(shapely.force_2d(geometry)))
    if not polygons:
        return shapely.GeometryCollection()
    return shapely.union_all(polygons)


def _triangulated_clipped_surface_triangles_2d(
    geometries: list[shapely.Geometry] | None,
    output_fileformat: str,
) -> list[shapely.Polygon]:
    """Return canonical 2D triangles for a clipped cell footprint."""
    clipped_triangles: list[shapely.Polygon] = []
    for polygon in geometries or []:
        if not isinstance(polygon, shapely.Polygon):
            continue
        clipped_triangles.extend(
            _iter_polygon_parts(
                shapely.constrained_delaunay_triangles(polygon),
            )
        )
    return _canonicalize_clipped_triangles_by_serialized_xy(
        clipped_triangles,
        output_fileformat,
    )


def _surface_polygons_from_2d_triangles(
    triangles_2d: Sequence[shapely.Polygon],
    planes_3d: list[shapely.Polygon],
    exterior_cw: bool,
    output_fileformat: str,
    keep_collapsed_placeholders: bool = False,
) -> list[shapely.Polygon | None]:
    """Interpolate 2D triangles onto 3D planes and normalize output."""
    surface_polygons: list[shapely.Polygon | None] = []
    for triangle in triangles_2d:
        oriented_triangle = shapely.orient_polygons(
            triangle,
            exterior_cw=exterior_cw,
        )
        triangle_3d = interpolate_z_planar(
            geometry_2d=oriented_triangle,
            planes_3d=planes_3d,
        )
        if not isinstance(triangle_3d, shapely.Polygon):
            raise TypeError(
                "Surface interpolation did not return a Polygon; got "
                f"{type(triangle_3d)}."
            )
        normalized_triangle = polygon_normalized_to_match_mesh_serialization(
            triangle_3d,
            output_fileformat,
        )
        if normalized_triangle is not None or keep_collapsed_placeholders:
            surface_polygons.append(normalized_triangle)
    return surface_polygons


def _triangulate_2d_geometry_to_3d_polygons(
    geometry: shapely.Geometry,
    planes_3d: list[shapely.Polygon],
    exterior_cw: bool,
    output_fileformat: str,
    fallback_z_by_xy: dict[tuple[float, float], float] | None = None,
    canonicalize_serialized_xy: bool = False,
) -> list[shapely.Polygon]:
    output: list[shapely.Polygon] = []

    def interpolate_triangle(
        oriented_triangle: shapely.Polygon,
    ) -> shapely.Polygon:
        try:
            triangle_3d = interpolate_z_planar(
                geometry_2d=oriented_triangle,
                planes_3d=planes_3d,
            )
        except ValueError:
            if not fallback_z_by_xy:
                raise

            coords_3d = []
            for coord in oriented_triangle.exterior.coords[:-1]:
                normalized_xy = normalize_vertex_to_match_mesh_serialization(
                    (coord[0], coord[1], 0.0),
                    output_fileformat,
                )[:2]
                if normalized_xy in fallback_z_by_xy:
                    coords_3d.append(
                        (
                            coord[0],
                            coord[1],
                            fallback_z_by_xy[normalized_xy],
                        )
                    )
                    continue

                point_3d = interpolate_z_planar(
                    geometry_2d=shapely.Point(coord),
                    planes_3d=planes_3d,
                )
                if not isinstance(point_3d, shapely.Point):
                    raise TypeError(
                        "Fallback interpolation did not return a Point."
                    )
                coords_3d.append(point_3d.coords[0])
            coords_3d.append(coords_3d[0])
            triangle_3d = shapely.Polygon(coords_3d)

        if not isinstance(triangle_3d, shapely.Polygon):
            raise TypeError("Z0 nudge interpolation did not return a Polygon.")
        return triangle_3d

    triangles_2d: list[shapely.Polygon] = []
    for polygon in _iter_polygon_parts(geometry):
        triangles = shapely.constrained_delaunay_triangles(polygon)
        triangles_2d.extend(_iter_polygon_parts(triangles))
    if canonicalize_serialized_xy:
        triangles_2d = _canonicalize_clipped_triangles_by_serialized_xy(
            triangles_2d,
            output_fileformat,
        )

    for triangle in triangles_2d:
        oriented_triangle = shapely.orient_polygons(
            triangle,
            exterior_cw=exterior_cw,
        )
        triangle_3d = interpolate_triangle(oriented_triangle)
        normalized_triangle = polygon_normalized_to_match_mesh_serialization(
            triangle_3d,
            output_fileformat,
        )
        if normalized_triangle is not None:
            output.append(normalized_triangle)
    return output


def _polygonized_regions_with_shared_boundaries(
    base_footprint: shapely.Geometry,
    regions: Sequence[shapely.Geometry],
) -> list[shapely.Polygon]:
    """Return polygon regions noded at all shared boundary intersections."""
    linework: list[shapely.Geometry] = []
    for region in regions:
        for polygon in _iter_polygon_parts(region):
            if polygon.area > 0:
                linework.append(polygon.boundary)

    if not linework:
        return []

    noded_linework = shapely.union_all(linework)
    polygons = shapely.polygonize([noded_linework])
    output: list[shapely.Polygon] = []
    for polygon in _iter_polygon_parts(polygons):
        if polygon.area <= 0 or not base_footprint.covers(
            polygon.representative_point(),
        ):
            continue
        clipped = polygon.intersection(base_footprint)
        output.extend(
            part
            for part in _iter_polygon_parts(clipped)
            if part.area > 0
        )
    return output


def _inserted_vertex_z_by_xy_from_planes(
    geometries: Sequence[shapely.Geometry],
    source_planes: list[shapely.Polygon],
    original_xy: set[tuple[float, float]],
    output_fileformat: str,
) -> dict[tuple[float, float], float]:
    """Return source-surface Z for new vertices introduced by split geometry."""
    z_by_xy: dict[tuple[float, float], float] = {}
    if not source_planes:
        return z_by_xy

    for geometry in geometries:
        for polygon in _iter_polygon_parts(geometry):
            for ring in [polygon.exterior, *polygon.interiors]:
                for coord in ring.coords[:-1]:
                    xy = normalize_vertex_to_match_mesh_serialization(
                        (coord[0], coord[1], 0.0),
                        output_fileformat,
                    )[:2]
                    if xy in original_xy or xy in z_by_xy:
                        continue
                    try:
                        point_3d = interpolate_z_planar(
                            geometry_2d=shapely.Point(coord[0], coord[1]),
                            planes_3d=source_planes,
                        )
                    except ValueError:
                        continue
                    if not isinstance(point_3d, shapely.Point):
                        continue
                    z_by_xy[xy] = (
                        normalize_coordinate_to_match_mesh_serialization(
                            point_3d.coords[0][2],
                            output_fileformat,
                        )
                    )

    return z_by_xy


def _clip_3d_surface_polygons_to_2d_geometry(
    surface_polygons: list[shapely.Polygon],
    geometry: shapely.Geometry,
    exterior_cw: bool,
    output_fileformat: str,
) -> list[shapely.Polygon]:
    """Clip existing 3D surface polygons and keep their clipped Z plane."""
    output: list[shapely.Polygon] = []
    geometry_2d = shapely.force_2d(geometry)
    for surface_polygon in surface_polygons:
        if not _has_valid_z_plane(surface_polygon):
            continue
        clipped = surface_polygon.intersection(geometry_2d)
        for clipped_part in _iter_polygon_parts(clipped):
            triangles = shapely.constrained_delaunay_triangles(
                shapely.force_2d(clipped_part),
            )
            for triangle in _iter_polygon_parts(triangles):
                oriented_triangle = shapely.orient_polygons(
                    triangle,
                    exterior_cw=exterior_cw,
                )
                triangle_3d = interpolate_z_planar(
                    geometry_2d=oriented_triangle,
                    planes_3d=[surface_polygon],
                )
                if not isinstance(triangle_3d, shapely.Polygon):
                    raise TypeError(
                        "Clipped surface interpolation did not return a "
                        "Polygon."
                    )
                normalized_triangle = (
                    polygon_normalized_to_match_mesh_serialization(
                        triangle_3d,
                        output_fileformat,
                    )
                )
                if normalized_triangle is not None:
                    output.append(normalized_triangle)
    return output


def _surface_wall_requested_lines(
    surface_polygon_borders: list[quad] | None,
    output_fileformat: str | None = None,
) -> shapely.Geometry | None:
    lines: list[shapely.LineString] = []
    if not surface_polygon_borders:
        return None
    for surface_border in surface_polygon_borders:
        xy_coords: list[tuple[float, float]] = []
        for border_vertex in surface_border.vl:
            if border_vertex is None:
                continue
            coords = border_vertex.coords
            if output_fileformat is not None:
                coords = normalize_vertex_to_match_mesh_serialization(
                    coords,
                    output_fileformat,
                )
            xy = (coords[0], coords[1])
            if xy not in xy_coords:
                xy_coords.append(xy)
        if len(xy_coords) == 2:
            lines.append(shapely.LineString(xy_coords))
    if not lines:
        return None
    return shapely.union_all(lines)


def _geometry_boundary_linework(
    geometry: shapely.Geometry,
    output_fileformat: str,
) -> shapely.Geometry | None:
    lines: list[shapely.LineString] = []
    for polygon in _iter_polygon_parts(shapely.force_2d(geometry)):
        for ring in (polygon.exterior, *polygon.interiors):
            coords = list(ring.coords)
            for index in range(len(coords) - 1):
                start = (
                    normalize_coordinate_to_match_mesh_serialization(
                        coords[index][0],
                        output_fileformat,
                    ),
                    normalize_coordinate_to_match_mesh_serialization(
                        coords[index][1],
                        output_fileformat,
                    ),
                )
                end = (
                    normalize_coordinate_to_match_mesh_serialization(
                        coords[index + 1][0],
                        output_fileformat,
                    ),
                    normalize_coordinate_to_match_mesh_serialization(
                        coords[index + 1][1],
                        output_fileformat,
                    ),
                )
                if start != end:
                    lines.append(shapely.LineString((start, end)))
    if not lines:
        return None
    return shapely.union_all(lines)


def _linework_covers_footprint(
    linework: shapely.Geometry | None,
    footprint: XYEdge,
) -> bool:
    return linework is not None and linework.covers(shapely.LineString(footprint))


def _split_surface_boundary_edges_for_wall_matches(
    surfaces: list[shapely.Polygon],
    boundary_edges: dict[XYEdge, Edge3D],
    split_vertices_xy: set[tuple[float, float]],
    footprint_is_requested: Callable[[XYEdge], bool],
    output_fileformat: str,
) -> list[shapely.Polygon]:
    split_surfaces: list[shapely.Polygon] = []

    for surface in surfaces:
        coords = list(surface.exterior.coords)
        if len(coords) < 4:
            continue
        if (
            len(coords) == 4
            and triangle_xy_collapses_after_mesh_serialization(
                coords[:-1],
                output_fileformat,
            )
        ):
            split_surfaces.append(surface)
            continue

        new_coords: list[tuple[float, ...]] = []
        inserted_point = False
        for index in range(len(coords) - 1):
            start = normalize_vertex_to_match_mesh_serialization(
                coords[index],
                output_fileformat,
            )
            end = normalize_vertex_to_match_mesh_serialization(
                coords[index + 1],
                output_fileformat,
            )
            footprint = edge_xy_signature(start, end)
            new_coords.append(start)
            if footprint not in boundary_edges or not footprint_is_requested(
                footprint,
            ):
                continue

            line = shapely.LineString((start[:2], end[:2]))
            line_length = line.length
            if line_length == 0:
                continue

            split_points: list[tuple[float, tuple[float, float]]] = []
            for split_xy in split_vertices_xy:
                if split_xy == start[:2] or split_xy == end[:2]:
                    continue
                point = shapely.Point(split_xy)
                if not line.covers(point):
                    continue
                distance = line.project(point)
                if distance <= 0 or distance >= line_length:
                    continue
                split_points.append((distance, split_xy))

            if not split_points:
                continue

            inserted_point = True
            for distance, split_xy in sorted(split_points):
                ratio = distance / line_length
                z = start[2] + (end[2] - start[2]) * ratio
                new_coords.append((split_xy[0], split_xy[1], z))

        if not inserted_point:
            split_surfaces.append(surface)
            continue

        new_coords.append(new_coords[0])
        split_footprint = shapely.Polygon([coord[:2] for coord in new_coords])
        exterior_cw = not shapely.is_ccw(shapely.LinearRing(new_coords))
        try:
            split_surfaces.extend(
                _triangulate_2d_geometry_to_3d_polygons(
                    split_footprint,
                    [surface],
                    exterior_cw=exterior_cw,
                    output_fileformat=output_fileformat,
                )
            )
        except ValueError as exc:
            if "near-vertical or collinear" not in str(exc):
                raise
            split_surfaces.append(surface)
            continue

    return split_surfaces


def _rebuild_matching_surface_polygon_borders(
    top_surfaces: list[shapely.Polygon],
    bottom_surfaces: list[shapely.Polygon],
    footprint_is_requested: Callable[[XYEdge], bool],
    output_fileformat: str,
) -> list[quad]:
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
    split_vertices_xy: set[tuple[float, float]] = set()
    for boundary_map in (top_boundary_edges, bottom_boundary_edges):
        for footprint in boundary_map:
            if footprint_is_requested(footprint):
                split_vertices_xy.update(footprint)

    if split_vertices_xy:
        top_surfaces[:] = _split_surface_boundary_edges_for_wall_matches(
            top_surfaces,
            top_boundary_edges,
            split_vertices_xy,
            footprint_is_requested,
            output_fileformat,
        )
        bottom_surfaces[:] = _split_surface_boundary_edges_for_wall_matches(
            bottom_surfaces,
            bottom_boundary_edges,
            split_vertices_xy,
            footprint_is_requested,
            output_fileformat,
        )
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

    rebuilt_borders: list[quad] = []
    for footprint, top_edge in top_boundary_edges.items():
        if not footprint_is_requested(footprint):
            continue
        if footprint not in bottom_boundary_edges:
            continue

        bottom_edge = bottom_boundary_edges[footprint]
        top_start_xy = (top_edge[0][0], top_edge[0][1])
        bottom_start_xy = (bottom_edge[0][0], bottom_edge[0][1])
        if top_start_xy == bottom_start_xy:
            bottom_edge = (bottom_edge[1], bottom_edge[0])

        wall = make_wall_without_exact_duplicate_vertices(
            vertex(*top_edge[1]),
            vertex(*top_edge[0]),
            vertex(*bottom_edge[1]),
            vertex(*bottom_edge[0]),
            output_fileformat=output_fileformat,
        )
        if wall is not None:
            rebuilt_borders.append(wall)
    return rebuilt_borders


def _rebuild_nudged_surface_polygon_borders(
    top_surfaces: list[shapely.Polygon],
    bottom_surfaces: list[shapely.Polygon],
    borders: dict[str, quad],
    previous_surface_polygon_borders: list[quad] | None,
    full_footprint: shapely.Geometry,
    include_normal_cut_edges: bool,
    W: float,
    E: float,
    N: float,
    S: float,
    output_fileformat: str,
    side_cut_wall_sides: set[str] | None = None,
) -> list[quad]:
    """Build current-cell walls requested by existing borders after nudging."""
    requested_clipped_lines = _surface_wall_requested_lines(
        previous_surface_polygon_borders,
        output_fileformat=output_fileformat,
    )
    footprint_boundary = _geometry_boundary_linework(
        full_footprint,
        output_fileformat,
    )

    side_values = _side_values_from_bounds(
        W,
        E,
        N,
        S,
        output_fileformat,
    )

    def footprint_is_requested(footprint: XYEdge) -> bool:
        line = shapely.LineString(footprint)
        side = _edge_cardinal_side(footprint, side_values)
        on_footprint_boundary = _linework_covers_footprint(
            footprint_boundary,
            footprint,
        )
        if side is not None and borders.get(side) is not False:
            return True
        if (
            requested_clipped_lines is not None
            and requested_clipped_lines.covers(line)
        ):
            return True
        if side is None and on_footprint_boundary:
            return True
        if (
            include_normal_cut_edges
            and side is None
            and not on_footprint_boundary
        ):
            return True
        if (
            side_cut_wall_sides is not None
            and side in side_cut_wall_sides
        ):
            return True
        return False

    return _rebuild_matching_surface_polygon_borders(
        top_surfaces,
        bottom_surfaces,
        footprint_is_requested,
        output_fileformat,
    )


# function to calculate the normal for a triangle
def get_normal(tri):
    "in: 3 verts, out normal (nx, ny,nz) with length 1"

    (v0, v1, v2) = tri
    p0 = Point.from_list(v0.get())
    p1 = Point.from_list(v1.get())
    p2 = Point.from_list(v2.get())
    a = Vector.from_points(p1, p0)
    b = Vector.from_points(p1, p2)
    #print p0,p1, p2
    #print a,b
    c = a.cross(b)
    #print c
    m = float(c.magnitude())
    if m == 0:
        normal = [0, 0, 0]
    else:
        normal = [c.x/m, c.y/m, c.z/m]
    return normal

def make_wall_without_exact_duplicate_vertices(
    v0: vertex,
    v1: vertex,
    v2: vertex,
    v3: vertex,
    output_fileformat: str = "STLb",
) -> quad | None:
    """Create a wall mesh, dropping duplicate vertices.

    Difference meshes can produce wall endpoints where top and bottom serialize
    to the same coordinate. In that case the wall should be a triangle, or
    omitted when the whole wall has zero height.
    """
    unique_vertices: list[vertex] = []
    unique_signatures: set[tuple[float, ...]] = set()

    for v in (v0, v1, v2, v3):
        signature = normalize_vertex_to_match_mesh_serialization(
            v.coords,
            output_fileformat,
        )
        if signature not in unique_signatures:
            unique_vertices.append(v)
            unique_signatures.add(signature)

    if len(unique_vertices) < 3:
        return None
    if len(unique_vertices) == 3:
        return quad(
            unique_vertices[0],
            unique_vertices[1],
            unique_vertices[2],
            None,
        )
    return quad(
        unique_vertices[0],
        unique_vertices[1],
        unique_vertices[2],
        unique_vertices[3],
    )


def _cardinal_wall_vertices(
    side: str,
    top_vertices: Sequence[vertex],
    bottom_vertices: Sequence[vertex],
) -> tuple[vertex, vertex, vertex, vertex]:
    """Return wall vertices for one side using cell quad vertex order."""
    wall_vertices = {
        "N": (
            bottom_vertices[0],
            top_vertices[0],
            top_vertices[3],
            bottom_vertices[1],
        ),
        "S": (
            bottom_vertices[2],
            top_vertices[2],
            top_vertices[1],
            bottom_vertices[3],
        ),
        "E": (
            top_vertices[3],
            top_vertices[2],
            bottom_vertices[2],
            bottom_vertices[1],
        ),
        "W": (
            top_vertices[1],
            top_vertices[0],
            bottom_vertices[0],
            bottom_vertices[3],
        ),
    }
    return wall_vertices[side]


def _build_cardinal_wall_borders(
    requested_borders: dict[str, quad | bool],
    top_vertices: Sequence[vertex],
    bottom_vertices: Sequence[vertex],
    output_fileformat: str,
) -> dict[str, quad | bool]:
    """Return requested N/S/E/W walls as wall quads."""
    borders = _empty_borders()
    for side in CARDINAL_DIRECTIONS:
        if requested_borders.get(side) is False:
            continue
        borders[side] = (
            make_wall_without_exact_duplicate_vertices(
                *_cardinal_wall_vertices(
                    side,
                    top_vertices,
                    bottom_vertices,
                ),
                output_fileformat=output_fileformat,
            ) or False
        )
    return borders


def quad_normalized_to_match_mesh_serialization(
    mesh: quad,
    fileformat: str,
    split_rotation: int,
    serialized_vertices: SerializedVertexCache | None = None,
) -> quad | None:
    """Return a serialized-coordinate quad, or None if it collapses."""
    if serialized_vertices is None:
        serialized_vertices = {}

    unique_vertices: list[vertex] = []
    unique_coords: set[tuple[float, ...]] = set()
    for mesh_vertex in mesh.vl:
        if mesh_vertex is None:
            continue

        output_coords = _serialized_vertex_from_cache(
            mesh_vertex.coords,
            fileformat,
            serialized_vertices,
        )
        if output_coords not in unique_coords:
            output_vertex = vertex(*output_coords)
            unique_vertices.append(output_vertex)
            unique_coords.add(output_coords)

    if len(unique_vertices) < 3:
        return None

    if len(unique_vertices) == 3:
        if _serialized_triangle_collapses(
            triangle=[v.coords for v in unique_vertices],
        ):
            return None
        return quad(unique_vertices[0], unique_vertices[1], unique_vertices[2])

    output_quad = quad(
        unique_vertices[0],
        unique_vertices[1],
        unique_vertices[2],
        unique_vertices[3],
        forced_split_edge=mesh.forced_split_edge,
    )
    valid_triangles: list[tuple[vertex, ...]] = []
    for triangle in output_quad.get_triangles(split_rotation=split_rotation):
        if not _serialized_triangle_collapses(
            triangle=[v.coords for v in triangle],
        ):
            valid_triangles.append(triangle)

    if len(valid_triangles) == 2:
        return output_quad
    if len(valid_triangles) == 1:
        triangle = valid_triangles[0]
        return quad(triangle[0], triangle[1], triangle[2])
    return None


class cell:
    '''a cell with a top and bottom quad, constructor: uses refs and does NOT copy ...
       except for triangle cells
       '''
    topquad: quad # 4 corner square quad with X,Y,Z
    bottomquad: quad | None
    borders: dict[str, quad]

    topSurfacePolygons: list[shapely.Polygon] | None = None
    "list of polygons (preferably tris) with X,Y,Z to use for the mesh instead of the topquad"
    bottomSurfacePolygons: list[shapely.Polygon] | None = None
    "list of polygons (preferably tris) with X,Y,Z to use for the mesh instead of the bottomquad"
    surfacePolygonBorders: list[quad] | None = None
    # surface polygon borders should be generated using raster polygon edge buckets BorderEdge wall value

    def __init__(self, topquad, bottomquad, borders, is_tri_cell=False):
        self.topquad = topquad
        self.bottomquad = bottomquad
        self.borders = borders
        self.is_tri_cell = is_tri_cell

        # Debug: keep the original quads to see what the cell functions changed
        # self.topquadoriginal = topquad
        # self.bottomquadtoriginal = bottomquad

    def __str__(self):
        r = hex(id(self)) + "\n top:" + str(self.topquad) + "\n btm:" + str(self.bottomquad) + "\n borders:\n"
        for d in CARDINAL_DIRECTIONS:
            if self.borders[d] is not False:
                r = r + "  " + d + ": " + str(self.borders[d]) + "\n"
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
            for k in self.borders:  # k is N, S, E, W
                if self.borders[k] is not False:
                    yield self.borders[k]
        # else:
        # It is possible to have a cell with no top quad or topSurfacePolygon because all volumes in the cell were removed in zero volume check
        #     raise AttributeError("cell has no top quad or topSurfacePolygons")

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
        serialized_vertices: SerializedVertexCache = {}

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

        for direction, border in self.borders.items():
            if border is not False:
                self.borders[direction] = (
                    quad_normalized_to_match_mesh_serialization(
                        border,
                        output_fileformat,
                        split_rotation,
                        serialized_vertices,
                    ) or False
                )

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

        full_footprint = _nudge_full_cell_footprint(W, E, N, S)
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

        points = _nudge_cell_points(W, E, N, S)
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
                _quad_corner_vertices_by_xy(self.topquad, W, E, N, S),
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
                _quad_corner_vertices_by_xy(self.bottomquad, W, E, N, S),
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
            _rebuild_nudged_surface_polygon_borders(
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

        keep_footprint = _nudge_keep_footprint(
            affected_corners,
            W,
            E,
            N,
            S,
        )
        if keep_footprint is None:
            return False

        full_footprint = _nudge_full_cell_footprint(W, E, N, S)
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
            _rebuild_nudged_surface_polygon_borders(
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

        keep_footprint = _nudge_keep_footprint(
            affected_corners,
            W,
            E,
            N,
            S,
        )
        if keep_footprint is None:
            return False

        full_cell_footprint = _nudge_full_cell_footprint(W, E, N, S)
        current_footprint = _current_surface_footprint(
            self.topSurfacePolygons,
            full_cell_footprint,
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
            _rebuild_nudged_surface_polygon_borders(
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

    def check_for_tri_cell(self):
        """Returns True if cell has borders on 2 consecutive sides False otherwise.
           Returns False is cell is already a tri-cell"""
        if self.is_tri_cell == True: return None
        b = self.borders

        # Count borders (non-False will be a pointer to a wall quad, i.e. True is not used here!
        num_borders = 0
        for d in CARDINAL_DIRECTIONS:
            if b[d] is not False: num_borders += 1

        if num_borders == 2:
            if b["N"] is not False and b["S"] is not False: return False
            if b["E"] is not False and b["W"] is not False: return False
        else:
            return False # cannot be triangelized

        #print("tricell:", num_borders, b)
        return True # 2 touching sides

    def convert_to_tri_cell(self):
        """Collapses the top and bottom quad into a triangle based on its 2 border walls,
        replaces one of the 2 border walls with a diagonal wall and the other with False.
        returns None, sets is_tri_cell to True"""
        if self.is_tri_cell == True: return None

        b = self.borders
        tq =  self.topquad.get_copy()
        bq =  self.bottomquad.get_copy()     # NW SE SW NE
        tvl = tq.vl #                           0  1  2  3
        bvl = bq.vl # vertex order in quad is   0  3  2  1

        # Collapse the quad into a triangle depending on where the 2 borders are
        # In addition we need to get rid of one wall and overwrite the other
        # with a new diagonal wall

        if b["N"] is not False and b["W"] is not False:
            self.topquad = quad(tvl[3], tvl[1], tvl[2], None) # ccw, order doesn't matter
            self.bottomquad = quad(bvl[1], bvl[2], bvl[3], None) # cw!
            b["N"] = quad(tvl[1], tvl[3], bvl[1], bvl[3]) # diagonal wall (ccw!)
            b["W"] = False # no used anymore
        elif b["N"] is not False and b["E"] is not False:
            self.topquad = quad(tvl[0], tvl[1], tvl[2], None)
            self.bottomquad = quad(bvl[0], bvl[2], bvl[3], None)
            b["N"] = quad(tvl[0], tvl[2], bvl[2], bvl[0])
            b["E"] = False
        elif b["S"] is not False and b["E"] is not False:
            self.topquad = quad(tvl[3], tvl[0], tvl[1], None)
            self.bottomquad = quad(bvl[3], bvl[0], bvl[1], None)
            b["S"] = quad(tvl[3], tvl[1], bvl[3], bvl[1])
            b["E"] = False
        elif b["S"] is not False and b["W"] is not False:
            self.topquad = quad(tvl[2], tvl[3], tvl[0], None)
            self.bottomquad = quad(bvl[0], bvl[1], bvl[2], None)
            b["S"] = quad(tvl[2], tvl[0], bvl[0], bvl[2])
            b["W"] = False
        else:
            print("convert_to_tri_cell() got invalid border config:", (self.borders), " - aborting")
            sys.exit()

        self.is_tri_cell = True

        return None

    def remove_zero_height_volumes(
        self,
        split_rotation: int,
        output_fileformat: str = "STLb",
        preserve_zero_height_xy: set[tuple[float, float]] | None = None,
        preserve_zero_height_edges: set[XYEdge] | None = None,
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
        serialized_vertices: SerializedVertexCache = {}

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
            matching_corners = 0
            remaining_corners = len(corner_pairs)
            for top_vertex, bottom_vertex in corner_pairs:
                remaining_corners -= 1
                if (
                    output_signature(top_vertex.coords)
                    == output_signature(bottom_vertex.coords)
                ):
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
            for direction, border in self.borders.items():
                if border is False:
                    continue
                footprint = surface_border_footprint(border)
                if footprint is None:
                    self.borders[direction] = False
                    continue
                footprint_key = edge_xy_signature(
                    footprint.coords[0],
                    footprint.coords[1],
                )
                if (
                    footprint_key not in top_footprints
                    or footprint_key not in bottom_footprints
                ):
                    self.borders[direction] = False

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
                for direction in self.borders:
                    self.borders[direction] = False
                return True

            if corners_match["NW"] and corners_match["NE"] and corners_match["SW"]:
                self.topquad = quad(tvl[3], tvl[1], tvl[2], None)
                self.bottomquad = quad(bvl[1], bvl[2], bvl[3], None)
                self.borders["N"] = False
                self.borders["W"] = False
                return True

            if corners_match["NW"] and corners_match["NE"] and corners_match["SE"]:
                self.topquad = quad(tvl[0], tvl[1], tvl[2], None)
                self.bottomquad = quad(bvl[0], bvl[2], bvl[3], None)
                self.borders["N"] = False
                self.borders["E"] = False
                return True

            if corners_match["NE"] and corners_match["SW"] and corners_match["SE"]:
                self.topquad = quad(tvl[3], tvl[0], tvl[1], None)
                self.bottomquad = quad(bvl[3], bvl[0], bvl[1], None)
                self.borders["S"] = False
                self.borders["E"] = False
                return True

            if corners_match["NW"] and corners_match["SW"] and corners_match["SE"]:
                self.topquad = quad(tvl[2], tvl[3], tvl[0], None)
                self.bottomquad = quad(bvl[0], bvl[1], bvl[2], None)
                self.borders["S"] = False
                self.borders["W"] = False
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
                self.borders["N"] = False
                self.borders["W"] = False
                self.borders["S"] = False
                self.borders["E"] = False
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

'''
#profiling decorator
# https://medium.com/fintechexplained/advanced-python-learn-how-to-profile-python-code-1068055460f9
import cProfile
import functools
import pstats
import tempfile
def profile_me(func):
    @functools.wraps(func)
    def wraps(*args, **kwargs):
        print("profiling started")
        file = tempfile.mktemp()
        profiler = cProfile.Profile()
        profiler.runcall(func, *args, **kwargs)
        profiler.dump_stats(file)
        metrics = pstats.Stats(file)
        metrics.strip_dirs().sort_stats('time').print_stats(100)
    return wraps
'''

class ProcessingTile:
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

def interpolate_corner_with_canonical_order(
    elev: np.ndarray,
    top_left_row: int,
    top_left_col: int,
) -> float:
    """Return a corner average from a canonical 2x2 operand order."""
    top_left = elev[top_left_row, top_left_col]
    top_right = elev[top_left_row, top_left_col + 1]
    bottom_left = elev[top_left_row + 1, top_left_col]
    bottom_right = elev[top_left_row + 1, top_left_col + 1]

    if (
        not np.isnan(top_left)
        and not np.isnan(top_right)
        and not np.isnan(bottom_left)
        and not np.isnan(bottom_right)
    ):
        return (top_left + top_right + bottom_left + bottom_right) / 4.0

    total = np.float64(0.0)
    count = 0
    for value in (top_left, top_right, bottom_left, bottom_right):
        if not np.isnan(value):
            total += value
            count += 1
    if count == 0:
        return np.nan
    return total / count


def interpolate_with_NaN(
    elev: np.ndarray,
    i: int,
    j: int,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Return NE, NW, SE, and SW cell corner elevations.

    The same shared raster corner is always averaged in the same operand order
    so adjacent cells produce identical floating-point values before mesh
    serialization.
    """

    NEelev = interpolate_corner_with_canonical_order(elev, j - 1, i)
    NWelev = interpolate_corner_with_canonical_order(elev, j - 1, i - 1)
    SEelev = interpolate_corner_with_canonical_order(elev, j, i)
    SWelev = interpolate_corner_with_canonical_order(elev, j, i - 1)

    if (
        np.isnan(NEelev)
        or np.isnan(NWelev)
        or np.isnan(SEelev)
        or np.isnan(SWelev)
    ):
        return None, None, None, None

    return NEelev, NWelev, SEelev, SWelev


def _interpolate_cell_corner_elevations(
    elev: np.ndarray,
    i: int,
    j: int,
) -> CornerElevations:
    """Return NE, NW, SE, and SW elevations for a non-NaN cell."""
    return (
        (
            elev[j - 1, i]
            + elev[j - 1, i + 1]
            + elev[j, i]
            + elev[j, i + 1]
        )
        / 4.0,
        (
            elev[j - 1, i - 1]
            + elev[j - 1, i]
            + elev[j, i - 1]
            + elev[j, i]
        )
        / 4.0,
        (
            elev[j, i]
            + elev[j, i + 1]
            + elev[j + 1, i]
            + elev[j + 1, i + 1]
        )
        / 4.0,
        (
            elev[j, i - 1]
            + elev[j, i]
            + elev[j + 1, i - 1]
            + elev[j + 1, i]
        )
        / 4.0,
    )


def _zero_elevations_below_threshold(
    elevations: CornerElevations,
    threshold: float,
) -> CornerElevations:
    """Replace elevations below the model base threshold with exact zero."""
    ne_elev, nw_elev, se_elev, sw_elev = elevations
    return (
        0 if ne_elev < threshold else ne_elev,
        0 if nw_elev < threshold else nw_elev,
        0 if se_elev < threshold else se_elev,
        0 if sw_elev < threshold else sw_elev,
    )


def _requested_cardinal_borders(
    padded_row: int,
    padded_col: int,
    ymaxidx: int,
    xmaxidx: int,
    borders_top_raster: np.ndarray,
    check_nan_neighbors: bool = True,
) -> dict[str, quad | bool]:
    """Return N/S/E/W sides that need exterior walls for this cell."""
    borders = _empty_borders()
    if padded_row == 1:
        borders["N"] = True
    if padded_row == ymaxidx:
        borders["S"] = True
    if padded_col == 1:
        borders["W"] = True
    if padded_col == xmaxidx:
        borders["E"] = True

    if not check_nan_neighbors:
        return borders

    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try:
            if np.isnan(borders_top_raster[padded_row - 1, padded_col]):
                borders["N"] = True
            if np.isnan(borders_top_raster[padded_row + 1, padded_col]):
                borders["S"] = True
            if np.isnan(borders_top_raster[padded_row, padded_col - 1]):
                borders["W"] = True
            if np.isnan(borders_top_raster[padded_row, padded_col + 1]):
                borders["E"] = True
        except RuntimeWarning:
            pass

    return borders


class grid:
    """makes cell data structure from two np arrays (top, bottom) of the same shape."""
    #@profile # https://pypi.org/project/memory-profiler/

    # I'm unclear why these class attributes need to be created here (added by keerl)
    tile: ProcessingTile = None
    # Check self.tile.bottom_raster_variants is not None to check if doing a "difference mesh" mode generation with a bottom array present.
    # If bottom_raster_variants is None, we only generate from top to flat bottom which is "top mesh" mode.

    bottom_thru_base: bool = False # Indicates if generating the "thru" mode

    tile_info = None
    xmaxidx = None
    ymaxidx = None
    cell_size = None
    offsetx = None
    offsety = None
    num_triangles = 0
    fo = None


    def __init__(self, tile: ProcessingTile):
        '''tile: Includes Top and Bottom raster variants and tile_info dict
        '''
        self.tile = tile
        self.tile_info = tile.tile_info

        self.bottom_thru_base = tile.tile_info.config.bottom_thru_base    # Anson's all-the-way-through case
        self.tile_info = tile.tile_info


        if self.tile_info.config.fileformat == "obj":
            vertex.vertex_index_dict = {}  # will be filled with vertex indices
        else:
            vertex.vertex_index_dict = -1

        self.cells = None # stores the cells in  a 2D array of cells
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


        # DEBUG: normalized (0 - 1) xy coord increment per cell
        #y_norm_delta  = 1 / float(top.shape[0]) # y (north-south) direction
        #x_norm_delta  = 1 / float(top.shape[1]) # x (east-west) direction
        #print "normalized x/y delta:", x_norm_delta, y_norm_delta

        # cell size (x and y delta)
        self.cell_size = self.tile_info.pixel_mm

        # does top have NaNs?
        #self.tile_info.have_nan = np.any(np.isnan(self.top)) # True => we have NaN values
        self.tile_info.have_nan = np.any(np.isnan(tile.top_raster_variants.dilated)) # True => we have NaN values

        # same for bottom, if we have one
        if self.tile_info.config.bottom_elevation is not None and tile.bottom_raster_variants.dilated is not None:
            self.tile_info.have_bot_nan = np.any(np.isnan(tile.bottom_raster_variants.dilated))# True => we have NaN values,

        # Jan 2019: no idea why, but sometimes changing top also changes the elevation
        # array of another tile in the tile list
        # for now I make a copy of all rasters and convert them to float
        # self.top = tile.top_raster_variants.dilated.copy().astype(np.float64) # writeable

        # if tile.top_raster_variants.original is not None:
        #     tile.top_raster_variants.original = tile.top_raster_variants.original.copy().astype(np.float64) # writeable

        # if tile.top_raster_variants.nan_close is not None:
        #     tile.top_raster_variants.nan_close = tile.top_raster_variants.nan_close.copy().astype(np.float64)

        # if tile.bottom_raster_variants is not None:
        #     if tile.bottom_raster_variants.dilated is not None:
        #         tile.bottom_raster_variants.dilated = tile.bottom_raster_variants.dilated.copy().astype(np.float64) # writeable

        #     if tile.bottom_raster_variants.original is not None:
        #         tile.bottom_raster_variants.original = tile.bottom_raster_variants.original.copy().astype(np.float64) # writeable

        #
        # Some sanity checks
        #

        # if bottom_raster_variants.dilated (last processed variant) is not an ndarray, we don't have a bottom raster, so bottom_raster_variants is set to None
        if isinstance(tile.bottom_raster_variants.dilated, np.ndarray) == False:
            tile.bottom_raster_variants = None
        # can't have a bottom_image and NaNs in top
        elif self.tile_info.config.bottom_image is not None and isinstance(tile.bottom_raster_variants.dilated, np.ndarray) == True and self.tile_info.have_nan == True:
            tile.bottom_raster_variants = None
            print("Top has NaN values, requested bottom image will be ignored!")
        # bottom is a elevation raster. It's ok to have NaNs in the bottom raster and/or top raster
        elif self.tile_info.config.bottom_elevation is not None and isinstance(tile.bottom_raster_variants.dilated, np.ndarray) == True:
            tile.bottom_raster_variants = tile.bottom_raster_variants

        # need to use the tilewide min/max for each tile, otherwise the boudaries don't line up perfectly!

        #
        # Convert elevation from real word elevation (m) to model print3D height (mm)
        #
        if self.tile_info.config.use_geo_coords is None: # Coordinates need to be in mm

            scz = 1 / self.tile_info.scale * 1000.0 # scale z to mm
            scale_min_elev = self.tile_info.config.min_elev

            if tile.bottom_raster_variants is not None: # Top-Bottom difference mesh mode
                if self.bottom_thru_base == False:  # normal case,
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
                #else:
                    # do nothing in the bottom_thru_base case because we previously set bottom raster to 0

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
        #print range(1, xmaxidx+1), range(1, ymaxidx+1)

        # offset so that 0/0 is the center of this tile (local) or so that 0/0 is the lower left corner of all tiles (global)
        if self.tile_info.config.tile_centered == False: # global offset, best for looking at all tiles together
            self.offsetx = -self.tile_info.tile_width  * (self.tile_info.tile_no_x-1)  # tile_no starts with 1! This is the top end of the tile, not 0!
            self.offsety = -self.tile_info.tile_height * (self.tile_info.tile_no_y-1)  + self.tile_info.tile_height * self.tile_info.config.ntilesy

        else: # local centered for printing
            self.offsetx = self.tile_info.tile_width / 2.0
            self.offsety = self.tile_info.tile_height / 2.0

        # geo coords are in meters (UTM). tile_centered is ignored for geo coords
        if self.tile_info.config.use_geo_coords != None:

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
        if self.tile_info.config.tile_centered == False:
            #print("tile width", self.tile_info.tile_width)
            #print("tile_no_x", self.tile_info.tile_no_x)
            #print("tile_no_y", self.tile_info.tile_no_y)
            #print("tile_height", self.tile_info.tile_height)
            #print("ntilesy", self.tile_info.ntilesy)
            self.tile_info.W = self.tile_info.tile_width  * (self.tile_info.tile_no_x-1)
            self.tile_info.E = self.tile_info.W + self.tile_info.tile_width
            tot_height = self.tile_info.tile_height * self.tile_info.config.ntilesy
            # y tiles index goes top(0) DOWN to bottom
            self.tile_info.N = tot_height - (self.tile_info.tile_height * (self.tile_info.tile_no_y-1))
            self.tile_info.S = self.tile_info.N - self.tile_info.tile_height
            #print("WENS", self.tile_info.W , self.tile_info.E, self.tile_info.N ,self.tile_info.S )
        else:
            self.tile_info.W = -self.tile_info.tile_width / 2
            self.tile_info.E =  self.tile_info.tile_width / 2
            self.tile_info.S = -self.tile_info.tile_height / 2
            self.tile_info.N =  self.tile_info.tile_height / 2

    def clean_up_diags_check(self, ras):
        """Check for NaNs in the raster and clean diagonal NaNs if requested."""
        if np.any(np.isnan(ras)) == True: # do we have any NaNs?
            if self.tile_info.config.clean_diags == True: # cleanup requested?
                ras = utils.clean_up_diags(ras)

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
        worker_count = _single_job_parallel_workers(
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
                    full_footprint = _nudge_full_cell_footprint(W, E, N, S)
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
        worker_count = _single_job_parallel_workers(
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

    def _positive_contact_top_corner_vertices_at(
        self,
        positive_contact_top_raster: np.ndarray | None,
        padded_row: int,
        padded_col: int,
        W: float,
        E: float,
        N: float,
        S: float,
    ) -> dict[IntermediateCorner, vertex] | None:
        """Return upper-raster corner vertices for positive-Z midpoint Z."""
        if positive_contact_top_raster is None:
            return None
        elevations = interpolate_with_NaN(
            positive_contact_top_raster,
            padded_col,
            padded_row,
        )
        if any(elev is None or np.isnan(elev) for elev in elevations):
            return None
        ne_elev, nw_elev, se_elev, sw_elev = elevations
        return {
            IntermediateCorner.NW: vertex(W, N, nw_elev),
            IntermediateCorner.NE: vertex(E, N, ne_elev),
            IntermediateCorner.SW: vertex(W, S, sw_elev),
            IntermediateCorner.SE: vertex(E, S, se_elev),
        }

    def _cell_has_clipped_surface(
        self,
        padded_row: int,
        padded_col: int,
    ) -> bool:
        """Return whether this source cell used a clipped 2D footprint."""
        contains_properly = (
            self.tile.top_raster_variants
            .polygon_intersection_contains_properly
        )
        intersection_geometry = (
            self.tile.top_raster_variants.polygon_intersection_geometry
        )
        return (
            contains_properly is not None
            and intersection_geometry is not None
            and contains_properly[padded_row - 1][padded_col - 1] is False
        )

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
                worker_count = _single_job_parallel_workers(
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
                worker_count = _single_job_parallel_workers(
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
                    wall_meshes.extend(
                        border
                        for border in current_cell.borders.values()
                        if border is not False
                    )
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

        worker_count = _single_job_parallel_workers(
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
                side_values = _side_values_from_bounds(
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
                    side = _edge_cardinal_side(footprint, side_values)
                    if side is None:
                        continue
                    if current_cell.borders.get(side) is not False:
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
        worker_count = _single_job_parallel_workers(
            getattr(self.tile_info, "config", None),
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

        # store cells in an array, init to None
        self.cells = np.empty([self.ymaxidx, self.xmaxidx], dtype=cell)

        # TODO: not sure we need this any more, given that this was done on the full raster
        # and after the operations that could have changed the raster
        # if self.tile_info.config.clean_diags == True:
        #     self.tile.top_raster_variants.dilated = utils.fillHoles(self.tile.top_raster_variants.dilated, 1, 8, True) # fill single holes
        #     self.tile.top_raster_variants.dilated = utils.clean_up_diags(self.tile.top_raster_variants.dilated)
        #     if self.tile.top_raster_variants.nan_close is not None:
        #         self.tile.top_raster_variants.nan_close = utils.clean_up_diags(self.tile.top_raster_variants.nan_close)

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

        if not self.tile_info.have_nan:
            top_interpolation_raster = top_dilated
        elif top_variants.edge_interpolation is not None:
            top_interpolation_raster = top_variants.edge_interpolation
        else:
            top_interpolation_raster = top_variants.original
        bottom_raster_for_z0_nudge: np.ndarray | None = None
        if using_difference_mesh and not self.bottom_thru_base:
            if self.tile_info.have_bot_nan:
                bottom_raster_for_z0_nudge = (
                    bottom_variants.original
                )
            else:
                bottom_raster_for_z0_nudge = (
                    bottom_variants.dilated
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
                parallel_workers=_single_job_parallel_workers(
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

        for j in range(1, self.ymaxidx+1):# y dimension for looping within the +1 padded raster
            cell_row = j - 1
            N = -(cell_row * cell_size) + offsety
            S = N - cell_size
            if j % pc_step == 0:
                progress += percent
                print(progress, "%", multiprocessing.current_process(), file=sys.stderr)

            for i in range(1, self.xmaxidx + 1):# x dim.
                cell_col = i - 1
                #print("y=",j," x=",i, " elev=",top[j,i])

                # if center elevation of current top cell is NaN, set its cell to None and skip the rest
                if self.tile_info.have_nan and np.isnan(top[j, i]):
                    self.cells[cell_row, cell_col] = None
                    continue

                # x/y coords of cell "walls", origin is upper left
                W = cell_col * cell_size - offsetx
                E = W + cell_size
                #print(i,j, " ", E,W, " ",  N,S, " ", top[j,i])






                #region Make top cell vertices' heights
                if not self.tile_info.have_nan:
                    # non NaNs: interpolate elevation of four corners (array order is top[y,x]!)
                    NEelev, NWelev, SEelev, SWelev = (
                        _interpolate_cell_corner_elevations(
                            top_interpolation_raster,
                            i,
                            j,
                        )
                    )
                else:
                    # get values for current cell i, j, NEelev, NWelev, SEelev, SWelev
                    NEelev, NWelev, SEelev, SWelev = interpolate_with_NaN(
                        top_interpolation_raster,
                        i,
                        j,
                    )

                    if NEelev is None: # if any of the corners is NaN, we have set the cell to None and can skip it
                        continue

                    # compare values with real print3D heights at this point
                    # Pull values set to bottom_floor_elev (which will be just below basethick) to actual 0 because we added basethick to all raster.
                    NEelev, NWelev, SEelev, SWelev = (
                        _zero_elevations_below_threshold(
                            (NEelev, NWelev, SEelev, SWelev),
                            self.tile_info.config.basethick,
                        )
                    )

                #
                # Note that here we flip x and y coordinate axis to the system used in 3D graphics
                #

                # make top quad (x,y,z) vertices   vi is the vertex index dict of the grids
                NEt = vertex(E, N, NEelev)
                NWt = vertex(W, N, NWelev)
                SEt = vertex(E, S, SEelev)
                SWt = vertex(W, S, SWelev)
                # a certain vertex order is needed to make the 2 triangles be counter clockwise and so point outwards
                # top quad vertex order is so that the normal points up
                topq = quad(NWt, SWt, SEt, NEt)
                #print(i, j, topq)

                top_bottom_surface_geometries_2D: list[shapely.Geometry] | None = None
                top_bottom_surface_polygons_triangulated_2D: (
                    list[shapely.Polygon] | None
                ) = None
                # Check if non-quad top_surface polygon should be used
                top_surface_polygons_triangulated_3D: (
                    list[shapely.Polygon | None] | None
                ) = None
                clipped_surfaces_collapsed_after_output = False
                # by checking if the cell is NOT contains_properly and if it has polygon_intersection_geometry
                if (
                    polygon_contains_properly is not None
                    and polygon_intersection_geometry is not None
                    and not polygon_contains_properly[cell_row][cell_col]
                ):
                    top_bottom_surface_geometries_2D = (
                        polygon_intersection_geometry[cell_row][cell_col]
                    )
                    if top_bottom_surface_geometries_2D is not None:
                        # We can verify if our shapely utils coordinate converter matches the N W S E made in create_cells. (it does if you adjust for the padding difference)
                        #quadPrint2DCoords = utils.arrayCellCoordToQuadPrint2DCoords(array_coord_2D=(i-1,j-1), cell_size=self.cell_size, tile_y_shape=self.tile.top_raster_variants.polygon_intersection_geometry.shape[0])
                        top_bottom_surface_polygons_triangulated_2D = (
                            _triangulated_clipped_surface_triangles_2d(
                                top_bottom_surface_geometries_2D,
                                output_fileformat,
                            )
                        )
                        top_surface_polygons_triangulated_3D = (
                            _surface_polygons_from_2d_triangles(
                                top_bottom_surface_polygons_triangulated_2D,
                                topq.get_triangles_in_polygons(
                                    split_rotation=split_rotation,
                                ),
                                exterior_cw=False,
                                output_fileformat=output_fileformat,
                                keep_collapsed_placeholders=True,
                            )
                        )

                #endregion

                #
                #region Make bottom quad
                #

                # get corners for bottom array
                if skip_simple_serialization_cleanup:
                    # Simple normal meshes only need per-cell bottom vertices
                    # when an outer wall is emitted for this cell.
                    NEelev = NWelev = SEelev = SWelev = 0
                elif not using_difference_mesh:
                    # Normal mode
                    NEelev = NWelev = SEelev = SWelev = 0
                else:
                    # Difference mode
                    # for the through water case, simply set the bottom to 0
                    if self.bottom_thru_base:
                        NEelev = NWelev = SEelev = SWelev = 0
                    else:
                        # simple interpolation
                        if not self.tile_info.have_bot_nan:
                            NEelev, NWelev, SEelev, SWelev = (
                                _interpolate_cell_corner_elevations(
                                    bottom_raster_for_z0_nudge,
                                    i,
                                    j,
                                )
                            )
                        else:
                            # Nan aware interpolation
                            NEelev, NWelev, SEelev, SWelev = (
                                interpolate_with_NaN(
                                    bottom_raster_for_z0_nudge,
                                    i,
                                    j,
                                )
                            )

                            if NEelev is None: # if any of the corners is NaN, we have set the cell to None and are skippping it
                                continue # skip this cell

                            # Pull values set to bottom_floor_elev to actual 0
                            # compare values with real print3D heights at this point
                            NEelev, NWelev, SEelev, SWelev = (
                                _zero_elevations_below_threshold(
                                    (NEelev, NWelev, SEelev, SWelev),
                                    self.tile_info.config.basethick,
                                )
                            )

                NEb = NWb = SEb = SWb = None
                botq = None
                bottom_corner_vertices: dict[IntermediateCorner, vertex] | None = None

                if not skip_simple_serialization_cleanup:
                    # These vertices may become emitted bottom surfaces, walls,
                    # or source planes for clipped/nudged cells.
                    (
                        NEb,
                        NWb,
                        SEb,
                        SWb,
                        botq,
                        bottom_corner_vertices,
                    ) = _create_cell_bottom_geometry(
                        W,
                        E,
                        N,
                        S,
                        NEelev,
                        NWelev,
                        SEelev,
                        SWelev,
                        nudge_enabled,
                    )

                top_corner_vertices: dict[IntermediateCorner, vertex] | None = None

                # Check if non-quad top_surface polygon should be used for bottom quad
                bottom_surface_polygons_triangulated_3D: (
                    list[shapely.Polygon | None] | None
                ) = None
                if top_bottom_surface_polygons_triangulated_2D is not None:
                    # We can verify if our shapely utils coordinate converter matches the N W S E made in create_cells. (it does if you adjust for the padding difference)
                    #quadPrint2DCoords = utils.arrayCellCoordToQuadPrint2DCoords(array_coord_2D=(i-1,j-1), cell_size=self.cell_size, tile_y_shape=self.tile.top_raster_variants.polygon_intersection_geometry.shape[0])

                    bottom_surface_polygons_triangulated_3D = (
                        _surface_polygons_from_2d_triangles(
                            top_bottom_surface_polygons_triangulated_2D,
                            botq.get_triangles_in_polygons(
                                split_rotation=split_rotation,
                            ),
                            exterior_cw=True,
                            output_fileformat=output_fileformat,
                            keep_collapsed_placeholders=True,
                        )
                    )

                    if top_surface_polygons_triangulated_3D is not None:
                        if len(top_surface_polygons_triangulated_3D) != len(
                            bottom_surface_polygons_triangulated_3D
                        ):
                            raise RuntimeError(
                                "Top and bottom clipped surface triangle "
                                "counts differ during output cleanup."
                            )

                        original_polygon_count = len(
                            top_surface_polygons_triangulated_3D
                        )
                        kept_top_polygons: list[shapely.Polygon] = []
                        kept_bottom_polygons: list[shapely.Polygon] = []
                        for top_polygon, bottom_polygon in zip(
                            top_surface_polygons_triangulated_3D,
                            bottom_surface_polygons_triangulated_3D,
                        ):
                            if top_polygon is None or bottom_polygon is None:
                                continue
                            kept_top_polygons.append(top_polygon)
                            kept_bottom_polygons.append(bottom_polygon)

                        if original_polygon_count and not kept_top_polygons:
                            clipped_surfaces_collapsed_after_output = True

                        top_surface_polygons_triangulated_3D = (
                            kept_top_polygons
                        )
                        bottom_surface_polygons_triangulated_3D = (
                            kept_bottom_polygons
                        )

                #endregion

                #print(topq)
                #print(botq)
                z0_nudged_cell = False
                z0_used_bottom_provider = False
                z0_full_footprint_2D: shapely.Geometry | None = None
                z0_include_normal_cut_edges = False
                positive_z_nudged_cell = False
                positive_z_full_footprint_2D: shapely.Geometry | None = None
                positive_z_difference_corners: list[IntermediateCorner] = []
                positive_z_split_sides: set[str] = set()
                positive_z_protected_split_sides: set[str] = set()
                positive_z_side_cut_wall_sides: set[str] = set()
                positive_z_split_contact_corners: list[IntermediateCorner] = []
                positive_z_flip_edges: set[Edge3D] = set()
                positive_z_record: PositiveZNudgeRecord = {}
                if nudge_enabled:
                    positive_z_record = positive_z_nudge_plan.get((j, i), {})
                    positive_z_flip_edges = set(
                        positive_z_record.get("flip_edges", set()),
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
                        if top_corner_vertices is None:
                            top_corner_vertices = {
                                IntermediateCorner.NW: NWt,
                                IntermediateCorner.NE: NEt,
                                IntermediateCorner.SW: SWt,
                                IntermediateCorner.SE: SEt,
                            }
                        cell_top_corner_vertices = top_corner_vertices
                        if botq is None:
                            (
                                NEb,
                                NWb,
                                SEb,
                                SWb,
                                botq,
                                bottom_corner_vertices,
                            ) = _create_cell_bottom_geometry(
                                W,
                                E,
                                N,
                                S,
                                NEelev,
                                NWelev,
                                SEelev,
                                SWelev,
                                nudge_enabled,
                            )
                        if bottom_corner_vertices is None:
                            raise RuntimeError(
                                "Z0 nudge needs bottom corner vertices.",
                            )
                        cell_bottom_corner_vertices = bottom_corner_vertices

                        def output_z_is_zero(v: vertex) -> bool:
                            return (
                                normalize_vertex_to_match_mesh_serialization(
                                    v.coords,
                                    output_fileformat,
                                )[2]
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
                        full_cell_footprint = _nudge_full_cell_footprint(
                            W,
                            E,
                            N,
                            S,
                        )
                        keep_footprint = _nudge_keep_footprint(
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
                                full_cell_footprint,
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
                        positive_contact_corners = list(
                            positive_z_record.get("corners", []),
                        )

                        if 0 < len(positive_contact_corners) < 4:
                            full_cell_footprint = _nudge_full_cell_footprint(
                                W,
                                E,
                                N,
                                S,
                            )
                            positive_z_full_footprint_2D = (
                                _current_surface_footprint(
                                    top_surface_polygons_triangulated_3D,
                                    full_cell_footprint,
                                )
                            )
                            keep_footprint = _nudge_keep_footprint(
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
                            positive_z_split_contact_corners = list(
                                positive_z_record.get("contact_corners", []),
                            )
                    else:
                        positive_z_difference_corners = (
                            _positive_z_effective_difference_corners(
                                positive_z_record,
                            )
                        )
                        if positive_z_difference_corners:
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
                                    in (
                                        positive_z_difference_neighbor_split_sides
                                        .get(neighbor_location, set())
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
                                positive_z_record.get("split_sides", set()),
                            )
                            positive_z_protected_split_sides = set(
                                positive_z_split_sides,
                            )
                            positive_z_split_sides.update(
                                positive_z_difference_neighbor_split_sides.get(
                                    (j, i),
                                    set(),
                                )
                            )
                            if positive_z_split_sides:
                                positive_z_split_contact_corners = list(
                                    positive_z_record.get(
                                        "contact_corners",
                                        [],
                                    ),
                                )

                if clipped_surfaces_collapsed_after_output:
                    self.cells[cell_row, cell_col] = None
                    continue

                #
                #region Make borders
                #

                borders = _requested_cardinal_borders(
                    j,
                    i,
                    self.ymaxidx,
                    self.xmaxidx,
                    borders_top_raster,
                    check_nan_neighbors=self.tile_info.have_nan,
                )

                # Quads for walls: in borders dict, replace any True with a quad of that wall
                if any(borders[side] for side in CARDINAL_DIRECTIONS):
                    if botq is None:
                        (
                            NEb,
                            NWb,
                            SEb,
                            SWb,
                            botq,
                            bottom_corner_vertices,
                        ) = _create_cell_bottom_geometry(
                            W,
                            E,
                            N,
                            S,
                            NEelev,
                            NWelev,
                            SEelev,
                            SWelev,
                            nudge_enabled,
                        )
                    if (
                        NWb is None
                        or NEb is None
                        or SWb is None
                        or SEb is None
                    ):
                        raise RuntimeError(
                            "Border creation needs bottom vertices.",
                        )
                    borders = _build_cardinal_wall_borders(
                        borders,
                        topq.vl,
                        botq.vl,
                        output_fileformat,
                    )

                # create borders if there is a top surface polygon using the edge buckets
                surface_polygon_borders_3D: list[quad] = []
                buckets = (
                    polygon_edge_buckets[cell_row][cell_col]
                    if polygon_edge_buckets is not None
                    else None
                )
                if buckets is not None:
                    # Get list of all BorderEdges with edge geometry that should be walls
                    wall_borderEdges = []
                    if isinstance(buckets, dict):
                        for bucket in buckets.values():
                            if isinstance(bucket, list):
                                for be in bucket:
                                    if isinstance(be, BorderEdge):
                                        if be.make_wall:
                                            wall_borderEdges.append(be)

                    if (
                        top_bottom_surface_geometries_2D
                        and top_surface_polygons_triangulated_3D
                        and bottom_surface_polygons_triangulated_3D
                    ):
                        top_surface_edges_3D: list[shapely.LineString] = []
                        bot_surface_edges_3D: list[shapely.LineString] = []
                        for geom in top_surface_polygons_triangulated_3D:
                            flattened_top_geom = flatten_geometries(
                                geometries=[geom],
                                to_single_lines=True,
                            )
                            top_surface_edges_3D.extend(
                                [
                                    item
                                    for item in flattened_top_geom
                                    if isinstance(item, shapely.LineString)
                                ]
                            )
                        for geom in bottom_surface_polygons_triangulated_3D:
                            flattened_bot_geom = flatten_geometries(
                                geometries=[geom],
                                to_single_lines=True,
                            )
                            bot_surface_edges_3D.extend(
                                [
                                    item
                                    for item in flattened_bot_geom
                                    if isinstance(item, shapely.LineString)
                                ]
                            )

                        top_edges_by_key = _boundary_line_map_by_serialized_xy(
                            top_surface_edges_3D,
                            output_fileformat,
                        )
                        bottom_edges_by_key = (
                            _boundary_line_map_by_serialized_xy(
                                bot_surface_edges_3D,
                                output_fileformat,
                            )
                        )

                        serialized_border_lines = [
                            border_line
                            for be in wall_borderEdges
                            for border_line in [
                                _line_with_serialized_xy(
                                    be.geometry,
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
                        for edge_key in list(top_edges_by_key):
                            edge_line = shapely.LineString(
                                [edge_key[0], edge_key[1]],
                            )
                            if not wall_border_linework.covers(edge_line):
                                continue

                            top_edge_matches = top_edges_by_key.get(edge_key)
                            bot_edge_matches = bottom_edges_by_key.get(edge_key)
                            while top_edge_matches or bot_edge_matches:
                                topEdgeMatch = (
                                    top_edge_matches.pop(0)
                                    if top_edge_matches
                                    else None
                                )
                                botEdgeMatch = (
                                    bot_edge_matches.pop(0)
                                    if bot_edge_matches
                                    else None
                                )
                                if top_edge_matches == []:
                                    del top_edges_by_key[edge_key]
                                if bot_edge_matches == []:
                                    del bottom_edges_by_key[edge_key]

                                if topEdgeMatch and not botEdgeMatch:
                                    raise RuntimeError(
                                        "Border creation: top edge match found "
                                        "but no bot edge match.",
                                    )
                                if botEdgeMatch and not topEdgeMatch:
                                    raise RuntimeError(
                                        "Border creation: bot edge match found "
                                        "but no top edge match.",
                                    )
                                if not topEdgeMatch or not botEdgeMatch:
                                    continue

                                # Success condition where wall border linework
                                # covers a top/bottom surface edge pair.
                                top_edge_v0 = vertex(*topEdgeMatch.coords[1])
                                top_edge_v1 = vertex(*topEdgeMatch.coords[0])
                                bot_edge_v0 = vertex(*botEdgeMatch.coords[1])
                                bot_edge_v1 = vertex(*botEdgeMatch.coords[0])
                                tb_wall = make_wall_without_exact_duplicate_vertices(
                                    top_edge_v0,
                                    top_edge_v1,
                                    bot_edge_v0,
                                    bot_edge_v1,
                                    output_fileformat=output_fileformat,
                                )
                                if tb_wall is not None:
                                    surface_polygon_borders_3D.append(tb_wall)
                            # create border geometry with top and bot edge
                            # top and bot edges are in CW order (viewed from top) from shapely




                #endregion

                if z0_nudged_cell:
                    if (
                        top_surface_polygons_triangulated_3D is None
                        or bottom_surface_polygons_triangulated_3D is None
                    ):
                        raise RuntimeError(
                            "Z0 nudged cell is missing final surface polygons."
                        )
                    if z0_full_footprint_2D is None:
                        z0_full_footprint_2D = _nudge_full_cell_footprint(
                            W,
                            E,
                            N,
                            S,
                        )
                    surface_polygon_borders_3D = (
                        _rebuild_nudged_surface_polygon_borders(
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
                            _nudge_full_cell_footprint(
                                W,
                                E,
                                N,
                                S,
                            )
                        )
                    surface_polygon_borders_3D = (
                        _rebuild_nudged_surface_polygon_borders(
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

                #region Make cell
                if self.tile_info.config.no_bottom:
                    c = cell(topq, None, borders) # omit bottom - do not fill with 2 tris later (may have NaNs)
                else:
                    if self.tile_info.have_nan or using_difference_mesh or nudge_enabled: #self.tile_info.have_bottom_array == True:
                        # for through water case make sure this in not one of the dilated cells

                        c = cell(topq, botq, borders) # full cell: top quad, bottom quad and wall quads
                    else:
                        c = cell(topq, None, borders) # omit bottom, will fill with 2 tris later

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

                # DEBUG: store i,j, and central elev
                #c.iy = j-1
                #c.ix = i-1
                #c.central_elev = top[j-1,i-1]

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
                    #print(i,j, c.borders)
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
                    )

                self.cells[cell_row, cell_col] = c

                #endregion

                if not self.tile.defer_triangle_writes:
                    self.write_cell_meshes_to_buffer(c)

        print("100%", multiprocessing.current_process(), "\n", file=sys.stderr)

    def write_cell_meshes_to_buffer(self, current_cell: cell) -> None:
        """Write one finalized cell's meshes to the current output buffer."""
        if self._uses_fast_binary_stl_no_normals_writer():
            self._write_cell_meshes_to_binary_stl_no_normals(current_cell)
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
    ) -> None:
        """Write one no-normal binary STL triangle from raw coordinates."""
        self.num_triangles += 1
        c0, c1, c2 = triangle
        decimal_precision = MESH_OUTPUT_SERIALIZATION_DECIMAL_PRECISION
        round_coord = round
        to_float = float
        write = self.s.write
        pack_facet = BINARY_STL_FACET.pack
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
                    )
            elif isinstance(mesh, shapely.Polygon):
                coords = mesh.exterior.coords
                if len(coords) == 4 and coords[0] == coords[3]:
                    self._write_triangle_coords_to_binary_stl_no_normals(
                        (coords[0], coords[1], coords[2]),
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
            else _single_job_parallel_workers(
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
                        self.write_cell_meshes_to_buffer(current_cell)
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
        return not (
            self.tile_info.config.nudge_in_overused_edges_vertex
            and not self.tile_info.config.no_bottom
        )

    def _add_simple_bottom_to_buffer(self) -> None:
        """Add the two-triangle tile bottom used by simple normal meshes."""
        v0 = vertex(self.tile_info.W, self.tile_info.S, 0)
        v1 = vertex(self.tile_info.E, self.tile_info.S, 0)
        v2 = vertex(self.tile_info.E, self.tile_info.N, 0)
        v3 = vertex(self.tile_info.W, self.tile_info.N, 0)

        self.write_triangle_to_buffer((v0, v2, v1))
        self.write_triangle_to_buffer((v0, v3, v2))

    def write_triangle_to_buffer(self, t: tuple[vertex, ...]):
        '''write triangle vertices for triangle t to stream buffer self.s for caching.
        Once the cache is full, is is writting to disk (self.fo)'''

        if t is None: return # just for the case that one of the two triangle was removed by smoothing

        #print(self.num_triangles, end=", ")
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


    '''
    def create_zigzag_borders(self, num_cells_per_zig = 100, zig_dist_mm = 0.15, zig_undershoot_mm = 0.05):
        """ post process the border quads so that it follows a zig-zag pattern """

        assert num_cells_per_zig > 1, "create_zigzag_borders() error: num_cells_per_zig =" + str(num_cells_per_zig)

        # number of cells in x and y     grid is cells[y,x]
        ncells_x = self.cells.shape[1]
        ncells_y = self.cells.shape[0]

        ncpz = num_cells_per_zig

        # north and south border
        ncells = ncells_x

        # figure out how many full and partial zigs we need
        num_full_zigs = ncells // ncpz
        num_leftover_cells = ncells % ncpz

        offset = -abs(zig_undershoot_mm) # in mm

        # very first full zig
        rise_first_full = 1 / float(ncpz-1)

        # full width zig, after the very first
        rise_full = (1 + abs(offset)) / float(ncpz-1)

        # partial zig, made from leftovers
        rise_partial = 0 # 0 means no partials
        if num_leftover_cells > 1:
            rise_partial = (1 + abs(offset)) / float(num_leftover_cells-1)


        #print ncells, ncpz, num_full_zigs, num_leftover_cells, rise_full, brief_text

        # As I have to do 4 passes, I'm wrapping the calculation of the zig "height" into a local function
        def getzigvalue(ci, zig_dist_mm, ncells, ncpz, num_full_zigs, num_leftover_cells, rise_full, rise_partial):
            c_in_zig = ci % ncpz
            #print ncpz, ci, c_in_zig

            # very first zig has to start at 0, not offset
            if ci <= c_in_zig:

                yl = rise_first_full * c_in_zig + 0
                if c_in_zig < ncpz-1:
                    yr = rise_first_full * (c_in_zig+1)
                else:
                    yr = offset # done with first zig, go to offset

            # subsequent zigs, full or partial
            else:

                # full or partial zig
                if ncells - ci > num_leftover_cells:

                    # full cell, go to offset
                    yl = rise_full * c_in_zig + offset
                    if c_in_zig < ncpz-1:
                        yr = rise_full * (c_in_zig+1) + offset
                    else:
                        yr = offset
                else: # partial
                    yl = rise_partial * c_in_zig + offset
                    if c_in_zig < ncpz-1:
                        yr = rise_partial * (c_in_zig+1) + offset
                    else:
                        yr = offset

            # very last cell must set yr as 0
            if ci == ncells-1:
                yr = 0

            #print ci, c_in_zig, yl, yr
            return yl * zig_dist_mm, yr * zig_dist_mm

        # North  and south border
        for ci in range(0, ncells_x):
            yl,yr = getzigvalue(ci,zig_dist_mm, ncells_x, ncpz, num_full_zigs, num_leftover_cells, rise_full, rise_partial)

            # get vertex lists for the 2 quads for the north cell
            nrthcell = self.cells[0,ci]
            topverts = nrthcell.topquad.vl
            botverts = nrthcell.bottomquad.vl

            # note that the vertex order in top or bottom quads are different b/c top has normals up, bottom has normals down

            # order: NEb, NWb, SWb, SEb
            botverts[0].coords[1] += yl  # move y coord up a bit
            botverts[1].coords[1] += yr

            # order: NEt, SEt, SWt, NWt
            topverts[0].coords[1] += yl
            topverts[3].coords[1] += yr



            # get vertex lists for the 2 quads for the south cell
            sthcell = self.cells[-1,ci]
            topverts = sthcell.topquad.vl
            botverts = sthcell.bottomquad.vl


            # order: NEb, NWb, SWb, SEb
            botverts[2].coords[1] += yr  # WTH???? why yr?
            botverts[3].coords[1] += yl

            # order: NEt, SEt, SWt, NWt
            topverts[1].coords[1] += yl
            topverts[2].coords[1] += yr


        # West and east border
        for ci in range(0, ncells_y):
            yl,yr = getzigvalue(ci, zig_dist_mm, ncells_x, ncpz, num_full_zigs, num_leftover_cells, rise_full, rise_partial)


            wcell = self.cells[ci,0]
            topverts = wcell.topquad.vl
            botverts = wcell.bottomquad.vl


            # WTH? why east here?

            # order: NEb, NWb, SWb, SEb
            botverts[0].coords[0] -= yl
            botverts[3].coords[0] -= yr

            # order: NEt, SEt, SWt, NWt
            topverts[0].coords[0] -= yl
            topverts[1].coords[0] -= yr

            # East border
            wcell = self.cells[ci,-1]
            topverts = wcell.topquad.vl
            botverts = wcell.bottomquad.vl


            # WTH? why west here?

            # order: NEb, NWb, SWb, SEb
            botverts[1].coords[0] -= yl
            botverts[2].coords[0] -= yr

            # order: NEt, SEt, SWt, NWt
            topverts[3].coords[0] -= yl
            topverts[2].coords[0] -= yr


    # version that splits skinny triangles - didn't turn out to be a problem but maybe useful later
    def make_STLfile_buffer(self, ascii=False, no_bottom=False, temp_file=None):
        """returns buffer of ASCII or binary STL file from a list of triangles, each triangle must have 9 floats (3 verts, each xyz)
            if no_bottom is True, bottom triangles are omitted
            if temp_file is not None, write STL into it (instead of a buffer) and return it
        """
        # Example: list of 2 triangles
        #[
        # [ 1.0,  1.0,  1.0, # vertex1 xyz
        #  -1.0,  1.0, -1.0, # vertex2 xyz
        #  -1.0, -1.0,  1.0] # vertex3 xyz
        # [ 1.0,  1.0,  1.0,
        #  -1.0, -1.0,  1.0,
        #   1.0, -1.0, -1.0]
        #]
        # Normal for each facet is set to 0,0,0

        triangles = [] # list of triangles

        # number of cells in x and y     grid is cells[y,x]
        ncells_x = self.cells.shape[1]
        ncells_y = self.cells.shape[0]

        # go through all cells, get all its quads and split into triangles
        for ix in range(0, ncells_x):
          for iy in range(0, ncells_y):
            cell = self.cells[iy,ix] # get cell from 2D array of cells (grid)

            if cell != None:
                #print "cell", ix, iy

                # list of top/bottom quads for this cell,
                if no_bottom == False:
                    quads = [cell.topquad, cell.bottomquad]
                else:
                    quads = [cell.topquad] # no bottom quads, only top

                # get tris for top and bottom
                for q in quads:
                    tl = q.get_triangles() # triangle list
                    triangles.append(tl[0])
                    triangles.append(tl[1])

                # add tris for border quads     cell.borders is a dict with S E W N as keys and a quad as value (if there's a border in that direction, False, otherwise)
                for k in cell.borders.keys():
                    border_quad = cell.borders[k]
                    if  border_quad != False:
                        #print k,
                        # run a check if wall is too skinny, this will set an quad internal value for how much to subdivide
                        # the subdivision will happen later when we ask for the skinny wall's triangles
                        # we need the direction (k) b/c the order of verts is different for n/s vs e/w!
                        border_quad.check_if_too_skinny(k)
                        #print border_quad, border_quad.subdivide_by
                        tl = border_quad.get_triangles(k) # triangle list
                        for t in tl:
                            triangles.append(t)

        #for n,t in enumerate(triangles): print n, t[0], t[1], t[2]

        buf = None
        if ascii:
            buf_as_list = self._build_ascii_stl(triangles)
            buf = "\n".join(buf_as_list).encode("UTF-8") # single utf8 string
        else:
            buf_as_list = self._build_binary_stl(triangles)
            buf = b"".join(buf_as_list)  # single "binary string"/buffer

        #print len(buf)

        if temp_file ==  None: return buf

        # Write string into temp file and return it
        temp_file.write(buf)
        return temp_file
    '''
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

        # get file name for temp file (or None if using memory)
        if self.tile_info.temp_file != None:  # contains None or a file name.
            temp_file = self.tile_info.temp_file
        else:
            temp_file = None # means: use memory

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
        if temp_file != None:
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






# MAIN  (left this in so I can test stuff, most of it is however outdated and would need to be fixed ...)

#@profile # https://pypi.org/project/memory-profiler/
def main():
    nn = np.nan
    """
    top = np.array([[11,12,13,14],
                    [21,nn,nn,24],
                    [31,nn,nn,34],
                    [41,42,43,44],
                   ])

    top = np.array([[11,12,13],
                     [21,nn,23],
                     [31,32,33],
                    ])

    top = np.array([[np.nan, np.nan],
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [1,1],
                   ])
    top = np.array([[0.3,0.5,0.4],
                    [0.4,np.nan,0.6],
                    [0.3,0.6,0.7],
                   ])

    top = np.array([[1],
                      ])

    top = np.array([[1.0,1.1, 1.2],
                    [1.4,1.2, 1.3],
                    [1.5,2.6, 1.0],
                    [1.2,1.6, 1.7],
                   ])

    top =  np.array([
                        [nn, nn, nn, 11, 11, nn, nn],
                        [nn, nn, 17, 22, 24, nn, nn],
                        [nn, 13, 33, 44, 33, 24, nn],
                        [11, 22, 55, 70, 25, 30, nn],
                        [14, 17, 33, 39, nn, 22, 12],
                        [nn, 10, 23, 10, nn, 10, nn],
                        [nn, nn, 11,  6, nn, nn, nn],
                     ])

    top =  np.array([
                    [10, 10, 10, 10, 10, 10, 10],
                    [10, 10, 10, 10, 10, 10, 10],
                    [10, 10, 10, 10, 10, 10, 10],
                    [10, 10, 10, 100, 10, 10, 10],
                    [10, 10, 10, 10, 10, 10, 10],
                    [10, 10, 10, 10, 10, 10, 10],
                    [10, 10, 10, 10, 10, 10, 10],
                    ])

    top =  np.array([ [nn, nn, 11],
                      [11, nn, nn],
                      [11, 11, nn],
                 ])

    top =  np.array([
                         [ 1, 5, 10, 50, 20, 10, 1],
                         [ 1, 10, 10, 50, 20, 10, 2],
                         [ 1, 11, 150, 30, 30, 10, 5],
                         [ 1, 23, 100, 40, 20, 10, 2 ],
                         [ 1, 50, 10, 10, 20, 10 , 1 ],

                   ])
    top = np.array([ [1]])
    """

    top =  np.array([ [2, 3, 4],
                      [3, 2, 3],
                      [3, 2, 1],
                 ])
    bot_elev = np.array([ [2, 3, 4],
                          [3, 1, 3],
                          [3, 2, 1],
                        ])



    """
    import matplotlib.pyplot as plt
    #plt.ion()
    fig = plt.figure(figsize=(7,10))
    npim = top
    imgplot = plt.imshow(npim, aspect=u"equal", interpolation=u"none")
    cmap_name = 'nipy_spectral' # gist_earth or terrain or nipy_spectral
    imgplot.set_cmap(cmap_name)
    #a = fig.add_axes()
    #plt.title(DEM_name + " " + str(center))
    plt.colorbar(orientation="horizontal")
    plt.show()
    """


    tile_info_dict = {
        #"scale"  : 10000, # horizontal scale number, defines the size of the model (= 3D map): 1000 => 1m (real) = 1000m in model
        "scale"  : 1,
        "pixel_mm" : 1, # lateral (x/y) size of a pixel in mm
        "max_elev" : np.nanmax(top), # tilewide minimum/maximum elevation (in meter), either int or float, depending on raster
        "min_elev" : np.nanmin(top),
        "z_scale" :  1,     # z (vertical) scale (elevation exageration) factor, float
        "tile_no_x": 1, # current tile number in x, int, starting with 1, at upper left corner
        "tile_no_y": 1,
        "ntilesx": 1,
        "ntilesy": 1,
        "tile_centered" : False, # True: each tile's center is 0/0, False: global (all-tile) 0/0
        "fileformat": "stlb",  # folder/zip file name for all tiles
        #"fileformat": "obj",
        "base_thickness_mm": 0, # thickness between bottom and lowest elevation, NOT including the bottom relief.
        "tile_width": 100,
        "use_geo_coords": None,
        "no_bottom": False,
        "no_normals": True,
        "CPU_cores_to_use" : 1,
        "bottom_elevation": "bot.tif",
        "bottom_image": None
    }

    whratio = top.shape[0] / top.shape[1]
    tile_info_dict["tile_height"] = int(tile_info_dict["tile_width"]  * whratio)

    top = np.pad(top, (1,1), 'edge')

    bot_elev = np.pad(bot_elev, (1,1), 'edge')
    g = grid(top, bot_elev, tile_info_dict)



    #b = g.make_STLfile_buffer(ascii=True, no_normals=True, temp_file="STLtest_asc6.stl")
    #b = g.make_STLfile_buffer(ascii=False, no_normals=False, temp_file="STLtest_new_b3.stl")
    b = g.make_STLfile_buffer(tile_info_dict, ascii=False, temp_file="STLtest.stl")
    #f = open("STLtest_new.stl", 'wb');f.write(b);f.close()

    #b = g.make_OBJfile_buffer(no_bottom=False, temp_file="OBJtest2.obj", no_normals=False)
    print("done")


if __name__ == "__main__":
    main()
