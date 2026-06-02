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

from collections.abc import Sequence
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
SurfaceMesh: TypeAlias = Union[quad, shapely.Polygon]
EmittedBottomSurface: TypeAlias = tuple[
    quad | None,
    list[shapely.Polygon] | None,
]
BottomSurfaceProvider: TypeAlias = list[list[EmittedBottomSurface]]
MESH_OUTPUT_SERIALIZATION_DECIMAL_PRECISION = 6


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
    return tuple(sorted((tuple(coord0[:3]), tuple(coord1[:3]))))


def boundary_edge_map_from_meshes(
    meshes: list[SurfaceMesh] | None,
    output_fileformat: str | None = None,
) -> dict[XYEdge, Edge3D]:
    """Return emitted boundary edges for quads and triangulated polygons."""
    edge_counts: dict[XYEdge, int] = {}
    edge_coords: dict[XYEdge, Edge3D] = {}
    if not meshes:
        return {}

    def add_edge(coord0: Coordinate, coord1: Coordinate) -> None:
        if output_fileformat is None:
            output_coord0 = tuple(coord0[:3])
            output_coord1 = tuple(coord1[:3])
        else:
            output_coord0 = normalize_vertex_to_match_mesh_serialization(
                coord0,
                output_fileformat,
            )
            output_coord1 = normalize_vertex_to_match_mesh_serialization(
                coord1,
                output_fileformat,
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
                coords = list(ring.coords)
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
        value = (
            struct.unpack(
                "<f",
                struct.pack("<f", value),
            )[0]
            + 0.0
        )
    return round(value, decimals) + 0.0


def normalize_vertex_to_match_mesh_serialization(
    coord: Coordinate,
    fileformat: str,
    decimals: int = MESH_OUTPUT_SERIALIZATION_DECIMAL_PRECISION,
) -> tuple[float, ...]:
    """Return a vertex as it will be serialized in mesh output."""
    return tuple(
        normalize_coordinate_to_match_mesh_serialization(
            value=value,
            fileformat=fileformat,
            decimals=decimals,
        )
        for value in coord[:3]
    )


def triangle_collapses_after_mesh_serialization(
    triangle: Sequence[Coordinate],
    fileformat: str,
    decimals: int = MESH_OUTPUT_SERIALIZATION_DECIMAL_PRECISION,
) -> bool:
    """Return whether a triangle is degenerate after mesh serialization."""
    p0, p1, p2 = [
        normalize_vertex_to_match_mesh_serialization(
            coord=coord,
            fileformat=fileformat,
            decimals=decimals,
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


def triangle_xy_collapses_after_mesh_serialization(
    triangle: Sequence[Coordinate],
    fileformat: str,
    decimals: int = MESH_OUTPUT_SERIALIZATION_DECIMAL_PRECISION,
) -> bool:
    """Return whether a surface triangle has no serialized XY area."""
    p0, p1, p2 = [
        normalize_vertex_to_match_mesh_serialization(
            coord=coord,
            fileformat=fileformat,
            decimals=decimals,
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


def polygon_normalized_to_match_mesh_serialization(
    polygon: shapely.Polygon,
    fileformat: str,
    decimals: int = MESH_OUTPUT_SERIALIZATION_DECIMAL_PRECISION,
) -> shapely.Polygon | None:
    """Return a serialized-coordinate polygon, or None if it collapses."""
    coords = list(polygon.exterior.coords)
    if len(coords) != 4 or coords[0] != coords[-1]:
        raise ValueError("Expected a closed triangular Polygon.")

    exterior = [
        normalize_vertex_to_match_mesh_serialization(
            coord=coord,
            fileformat=fileformat,
            decimals=decimals,
        )
        for coord in polygon.exterior.coords
    ]
    interiors = [
        [
            normalize_vertex_to_match_mesh_serialization(
                coord=coord,
                fileformat=fileformat,
                decimals=decimals,
            )
            for coord in ring.coords
        ]
        for ring in polygon.interiors
    ]
    if triangle_collapses_after_mesh_serialization(
        triangle=exterior[:3],
        fileformat=fileformat,
        decimals=decimals,
    ):
        return None
    return shapely.Polygon(exterior, interiors)


def surface_polygon_normalized_to_match_mesh_serialization(
    polygon: shapely.Polygon,
    fileformat: str,
    decimals: int = MESH_OUTPUT_SERIALIZATION_DECIMAL_PRECISION,
) -> shapely.Polygon | None:
    """Return a serialized surface triangle, or None if it has no area."""
    normalized_polygon = polygon_normalized_to_match_mesh_serialization(
        polygon,
        fileformat,
        decimals,
    )
    if normalized_polygon is None:
        return None

    if triangle_xy_collapses_after_mesh_serialization(
        triangle=list(normalized_polygon.exterior.coords)[:3],
        fileformat=fileformat,
        decimals=decimals,
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


def _z0_cell_points(
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


def _z0_full_cell_footprint(
    W: float,
    E: float,
    N: float,
    S: float,
) -> shapely.Polygon:
    points = _z0_cell_points(W, E, N, S)
    return shapely.Polygon(
        [
            points["SW"],
            points["SE"],
            points["NE"],
            points["NW"],
            points["SW"],
        ]
    )


def _z0_normal_keep_footprint(
    affected_corners: Sequence[IntermediateCorner],
    W: float,
    E: float,
    N: float,
    S: float,
) -> shapely.Polygon | None:
    """Return the normal-mode keep footprint for a Z0 affected-corner set."""
    vertex_names = _z0_normal_keep_vertex_names(affected_corners)
    if vertex_names is None:
        return None
    points = _z0_cell_points(W, E, N, S)
    return shapely.Polygon([points[name] for name in vertex_names])


def _z0_normal_keep_vertex_names(
    affected_corners: Sequence[IntermediateCorner],
) -> list[str] | None:
    """Return vertex names for the normal-mode keep footprint."""
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


def _z0_adjusted_keep_surface_planes(
    affected_corners: Sequence[IntermediateCorner],
    corner_vertices: dict[IntermediateCorner, vertex],
    W: float,
    E: float,
    N: float,
    S: float,
) -> list[shapely.Polygon]:
    """Return full-precision nudge-adjusted triangles with Z0 midpoints."""
    vertex_names = _z0_normal_keep_vertex_names(affected_corners)
    if vertex_names is None:
        return []

    points = _z0_cell_points(W, E, N, S)
    corner_z = {
        "NW": corner_vertices[IntermediateCorner.NW].coords[2],
        "NE": corner_vertices[IntermediateCorner.NE].coords[2],
        "SW": corner_vertices[IntermediateCorner.SW].coords[2],
        "SE": corner_vertices[IntermediateCorner.SE].coords[2],
        "Nmid": 0.0,
        "Smid": 0.0,
        "Emid": 0.0,
        "Wmid": 0.0,
    }
    z_by_xy = {
        points[name]: corner_z[name]
        for name in vertex_names
    }
    footprint_2d = shapely.Polygon([points[name] for name in vertex_names])
    planes: list[shapely.Polygon] = []
    triangles = shapely.constrained_delaunay_triangles(footprint_2d)
    for triangle in _iter_polygon_parts(triangles):
        coords_3d = []
        for x, y in list(triangle.exterior.coords)[:-1]:
            z = z_by_xy.get((x, y))
            if z is None:
                raise ValueError(
                    "Z0 nudge adjusted plane includes an unexpected vertex."
                )
            coords_3d.append((x, y, z))
        coords_3d.append(coords_3d[0])
        planes.append(shapely.Polygon(coords_3d))
    return planes


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


def _triangulate_2d_geometry_to_3d_polygons(
    geometry: shapely.Geometry,
    planes_3d: list[shapely.Polygon],
    exterior_cw: bool,
    output_fileformat: str,
) -> list[shapely.Polygon]:
    output: list[shapely.Polygon] = []
    for polygon in _iter_polygon_parts(geometry):
        triangles = shapely.constrained_delaunay_triangles(polygon)
        for triangle in _iter_polygon_parts(triangles):
            oriented_triangle = shapely.orient_polygons(
                triangle,
                exterior_cw=exterior_cw,
            )
            triangle_3d = interpolate_z_planar(
                geometry_2d=oriented_triangle,
                planes_3d=planes_3d,
            )
            if not isinstance(triangle_3d, shapely.Polygon):
                raise TypeError("Z0 nudge interpolation did not return a Polygon.")
            normalized_triangle = polygon_normalized_to_match_mesh_serialization(
                triangle_3d,
                output_fileformat,
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
        split_surfaces.extend(
            _triangulate_2d_geometry_to_3d_polygons(
                split_footprint,
                [surface],
                exterior_cw=exterior_cw,
                output_fileformat=output_fileformat,
            )
        )

    return split_surfaces


def _rebuild_matching_surface_polygon_borders(
    top_surfaces: list[shapely.Polygon],
    bottom_surfaces: list[shapely.Polygon],
    footprint_is_requested: Callable[[XYEdge], bool],
    output_fileformat: str,
) -> list[quad]:
    top_boundary_edges = boundary_edge_map_from_meshes(
        top_surfaces,
        output_fileformat=output_fileformat,
    )
    bottom_boundary_edges = boundary_edge_map_from_meshes(
        bottom_surfaces,
        output_fileformat=output_fileformat,
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
        )
        bottom_boundary_edges = boundary_edge_map_from_meshes(
            bottom_surfaces,
            output_fileformat=output_fileformat,
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


def _z0_rebuild_surface_polygon_borders(
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

    side_values = {
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

    def edge_cardinal_side(footprint: XYEdge) -> str | None:
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

    def footprint_is_requested(footprint: XYEdge) -> bool:
        line = shapely.LineString(footprint)
        side = edge_cardinal_side(footprint)
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

    def same_vertex(a: vertex, b: vertex) -> bool:
        return (
            normalize_vertex_to_match_mesh_serialization(
                a.coords,
                output_fileformat,
            )
            == normalize_vertex_to_match_mesh_serialization(
                b.coords,
                output_fileformat,
            )
        )

    for v in (v0, v1, v2, v3):
        if not any(same_vertex(v, existing) for existing in unique_vertices):
            unique_vertices.append(v)

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


def quad_normalized_to_match_mesh_serialization(
    mesh: quad,
    fileformat: str,
    split_rotation: int,
) -> quad | None:
    """Return a serialized-coordinate quad, or None if it collapses."""
    unique_vertices: list[vertex] = []
    for mesh_vertex in mesh.vl:
        if mesh_vertex is None:
            continue

        output_vertex = vertex(
            *normalize_vertex_to_match_mesh_serialization(
                mesh_vertex.coords,
                fileformat,
            ),
        )
        if not any(
            output_vertex.coords == existing.coords
            for existing in unique_vertices
        ):
            unique_vertices.append(output_vertex)

    if len(unique_vertices) < 3:
        return None

    if len(unique_vertices) == 3:
        if triangle_collapses_after_mesh_serialization(
            triangle=[v.coords for v in unique_vertices],
            fileformat=fileformat,
        ):
            return None
        return quad(unique_vertices[0], unique_vertices[1], unique_vertices[2])

    output_quad = quad(
        unique_vertices[0],
        unique_vertices[1],
        unique_vertices[2],
        unique_vertices[3],
    )
    valid_triangles: list[tuple[vertex, ...]] = []
    for triangle in output_quad.get_triangles(split_rotation=split_rotation):
        if not triangle_collapses_after_mesh_serialization(
            triangle=[v.coords for v in triangle],
            fileformat=fileformat,
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
        for d in ["N", "S", "E", "W"]:
            if self.borders[d] != False:
                r = r + "  " + d + ": " + str(self.borders[d]) + "\n"
        return r

    def meshes_for_model(self) -> list[Union[quad, shapely.Polygon]]:
        """Return the meshes to include in the output model for this cell."""
        meshes = []
        if self.topSurfacePolygons:
            meshes.extend(self.topSurfacePolygons)
        elif self.topquad:
            meshes.append(self.topquad)
            # if we use topquad, we also use cardinal direction borders
            for k in self.borders:  # k is N, S, E, W
                if self.borders[k] is not False: meshes.append(self.borders[k])
        # else:
        # It is possible to have a cell with no top quad or topSurfacePolygon because all volumes in the cell were removed in zero volume check
        #     raise AttributeError("cell has no top quad or topSurfacePolygons")

        if self.bottomSurfacePolygons:
            meshes.extend(self.bottomSurfacePolygons)
        elif self.bottomquad:
            meshes.append(self.bottomquad)

        if self.surfacePolygonBorders:
            meshes.extend(self.surfacePolygonBorders)

        return meshes

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
                    )
                )
                if output_polygon is not None:
                    output.append(output_polygon)
            return output or None

        had_top_surface_polygons = bool(self.topSurfacePolygons)
        had_bottom_surface_polygons = bool(self.bottomSurfacePolygons)

        if self.topquad is not None:
            self.topquad = quad_normalized_to_match_mesh_serialization(
                self.topquad,
                output_fileformat,
                split_rotation,
            )

        if self.bottomquad is not None:
            self.bottomquad = quad_normalized_to_match_mesh_serialization(
                self.bottomquad,
                output_fileformat,
                split_rotation,
            )

        for direction, border in self.borders.items():
            if border is not False:
                self.borders[direction] = (
                    quad_normalized_to_match_mesh_serialization(
                        border,
                        output_fileformat,
                        split_rotation,
                    ) or False
                )

        if self.surfacePolygonBorders:
            surface_borders = []
            for surface_border in self.surfacePolygonBorders:
                output_border = quad_normalized_to_match_mesh_serialization(
                    surface_border,
                    output_fileformat,
                    split_rotation,
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
            self.bottomquad = None
            self.bottomSurfacePolygons = None
            self.surfacePolygonBorders = None
            for direction in self.borders:
                self.borders[direction] = False

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
            return quad(top_vertices[0], top_vertices[2], top_vertices[1], None), None
        return (
            quad(
                top_vertices[0],
                top_vertices[3],
                top_vertices[2],
                top_vertices[1],
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
                self.topquad = None
                self.bottomquad = None
                self.bottomSurfacePolygons = None
                self.surfacePolygonBorders = None
                self.borders = {drct: False for drct in ["N", "S", "E", "W"]}
                return
            bottom_surface_quad = None

        if bottom_surface_polygons is not None:
            if not self.topSurfacePolygons:
                raise RuntimeError(
                    "Shared pair bottom has clipped polygons but the "
                    "difference top does not."
                )
            if replacement_bottom_quad is not None:
                self.bottomquad = replacement_bottom_quad
            self.bottomSurfacePolygons = bottom_surface_polygons
            self.borders = {drct: False for drct in ["N", "S", "E", "W"]}
            self._rebuild_surface_polygon_borders(output_fileformat)
            return

        if bottom_surface_quad is None:
            raise RuntimeError("Shared pair bottom surface is missing.")
        if self.topquad is None:
            raise RuntimeError("Difference top surface is missing.")

        self.bottomquad = bottom_surface_quad
        self.bottomSurfacePolygons = None
        self.surfacePolygonBorders = None
        self._rebuild_cardinal_borders(output_fileformat)

    def _rebuild_cardinal_borders(
        self,
        output_fileformat: str | None,
    ) -> None:
        """Rebuild existing cardinal walls after replacing a bottom quad."""
        top_vertices = self.topquad.vl
        bottom_vertices = self.bottomquad.vl
        rebuilt_borders = {drct: False for drct in ["N", "S", "E", "W"]}

        if self.borders.get("N") is not False:
            rebuilt_borders["N"] = (
                make_wall_without_exact_duplicate_vertices(
                    bottom_vertices[0],
                    top_vertices[0],
                    top_vertices[3],
                    bottom_vertices[1],
                    output_fileformat=output_fileformat or "STLb",
                ) or False
            )
        if self.borders.get("S") is not False:
            rebuilt_borders["S"] = (
                make_wall_without_exact_duplicate_vertices(
                    bottom_vertices[2],
                    top_vertices[2],
                    top_vertices[1],
                    bottom_vertices[3],
                    output_fileformat=output_fileformat or "STLb",
                ) or False
            )
        if self.borders.get("E") is not False:
            rebuilt_borders["E"] = (
                make_wall_without_exact_duplicate_vertices(
                    top_vertices[3],
                    top_vertices[2],
                    bottom_vertices[2],
                    bottom_vertices[1],
                    output_fileformat=output_fileformat or "STLb",
                ) or False
            )
        if self.borders.get("W") is not False:
            rebuilt_borders["W"] = (
                make_wall_without_exact_duplicate_vertices(
                    top_vertices[1],
                    top_vertices[0],
                    bottom_vertices[0],
                    bottom_vertices[3],
                    output_fileformat=output_fileformat or "STLb",
                ) or False
            )

        self.borders = rebuilt_borders

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

    def check_for_tri_cell(self):
        """Returns True if cell has borders on 2 consecutive sides False otherwise.
           Returns False is cell is already a tri-cell""" 
        if self.is_tri_cell == True: return None
        b = self.borders

        # Count borders (non-False will be a pointer to a wall quad, i.e. True is not used here!
        num_borders = 0
        for d in ["N", "S", "E", "W"]:
            if b[d] != False: num_borders += 1

        if num_borders == 2:
            if b["N"] != False and b["S"] != False: return False
            if b["E"] != False and b["W"] != False: return False
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
        
        if b["N"] != False and b["W"] != False:
            self.topquad = quad(tvl[3], tvl[1], tvl[2], None) # ccw, order doesn't matter
            self.bottomquad = quad(bvl[1], bvl[2], bvl[3], None) # cw!
            b["N"] = quad(tvl[1], tvl[3], bvl[1], bvl[3]) # diagonal wall (ccw!)
            b["W"] = False # no used anymore
        elif b["N"] != False and b["E"] != False: 
            self.topquad = quad(tvl[0], tvl[1], tvl[2], None)
            self.bottomquad = quad(bvl[0], bvl[2], bvl[3], None) 
            b["N"] = quad(tvl[0], tvl[2], bvl[2], bvl[0])
            b["E"] = False 
        elif b["S"] != False and b["E"] != False: 
            self.topquad = quad(tvl[3], tvl[0], tvl[1], None)
            self.bottomquad = quad(bvl[3], bvl[0], bvl[1], None)
            b["S"] = quad(tvl[3], tvl[1], bvl[3], bvl[1])
            b["E"] = False
        elif b["S"]!= False and b["W"] != False: 
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
        def output_signature(coord: Coordinate) -> tuple[float, ...]:
            return normalize_vertex_to_match_mesh_serialization(
                coord,
                output_fileformat,
            )

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
                    if polygons_equal_3d(
                        top_surface_polygon,
                        bottom_surface_polygon,
                    ):
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
    return_grid: bool

    def __init__(
        self,
        tile_info: TouchTerrainTileInfo,
        top: RasterVariants,
        bottom: Union[None, RasterVariants],
        bottom_surface_provider: BottomSurfaceProvider | None = None,
        return_grid: bool = False,
    ):
        self.tile_info = tile_info
        self.top_raster_variants = top
        self.bottom_raster_variants = bottom
        self.bottom_surface_provider = bottom_surface_provider
        self.return_grid = return_grid
   
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
        np.isnan(top_left)
        or np.isnan(top_right)
        or np.isnan(bottom_left)
        or np.isnan(bottom_right)
    ):
        return np.nanmean(
            np.array(
                [top_left, top_right, bottom_left, bottom_right],
                dtype=np.float64,
            ),
        )

    return (top_left + top_right + bottom_left + bottom_right) / 4.0


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

    # Interpolate each corner with possible NaNs. If any corner is surrounded
    # by all NaNs, skip the whole cell.
    with warnings.catch_warnings():
        warnings.filterwarnings('error')

        try:
            # init all elevs with NaN
            NEelev = NWelev = SEelev = SWelev = np.nan

            NEelev = interpolate_corner_with_canonical_order(elev, j - 1, i)
            NWelev = interpolate_corner_with_canonical_order(elev, j - 1, i - 1)
            SEelev = interpolate_corner_with_canonical_order(elev, j, i)
            SWelev = interpolate_corner_with_canonical_order(elev, j, i - 1)

        except RuntimeWarning:  # corner is surrounded by NaN elevations
            num_nans = sum(
                np.isnan(np.array([NEelev, NWelev, SEelev, SWelev])),
            )
            if num_nans > 0:  # yes, set cell to None and skip it
                return None, None, None, None
        else:
            
            '''
            print("\n", i,j)
            print("NE", elev[j+0,i+0], elev[j-1,i-0], elev[j-1,i+1], elev[j-0,i+1], NEelev)
            print("NW", elev[j+0,i+0], elev[j+0,i-1], elev[j-1,i-1], elev[j-1,i+0], NWelev)
            print("SE", elev[j+0,i+0], elev[j-0,i+1], elev[j+1,i+1], elev[j+1,i+0], SEelev)
            print("SW", elev[j+0,i+0], elev[j+1,i+0], elev[j+1,i-1], elev[j+0,i-1], SWelev)
            '''
            return NEelev, NWelev, SEelev, SWelev    
        
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


        if self.tile_info.config.fileformat == 'obj':
            vertex.vertex_index_dict = {} # will be filled with vertex indices

        self.cells = None # stores the cells in  a 2D array of cells

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
                
            if tile.bottom_raster_variants is not None: # Top-Bottom difference mesh mode
                if self.bottom_thru_base == False:  # normal case,  
                    tile.bottom_raster_variants -= self.tile_info.config.min_elev
                    
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

            tile.top_raster_variants -= self.tile_info.config.min_elev
            tile.top_raster_variants *= scz * self.tile_info.config.zscale # apply z-scale to top
            tile.top_raster_variants += self.tile_info.config.basethick # add base thickness to top

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

        surfaces: BottomSurfaceProvider = []
        for row in self.cells:
            surface_row: list[EmittedBottomSurface] = []
            for current_cell in row:
                if current_cell is None:
                    surface_row.append((None, None))
                else:
                    surface_row.append(
                        current_cell.emitted_top_as_bottom_surfaces()
                    )
            surfaces.append(surface_row)
        return surfaces

    def create_cells(self):
        '''Creates a data structure for each raster cell based on quads for top, any walls and possible bottom.
        Once created, each cell is converted into triangles for each file format, which are stored as a stream buffer (self.s)
        If using temp files, this buffer serves as a cache for occasionally writing to disk (self.fo)
        Note that for obj, two streams/files are needed, one for indices that define the vertices for each triangle and one
        for vertex coordinates. Here, only the index part (s[1] and fo[1]) is stored, the vertex coordinates will be
        created and stored later based on the keys of the vertex class attribute vertex_index_dict'''
        normalize_polygon_for_serialization = (
            polygon_normalized_to_match_mesh_serialization
        )

        if self.tile_info is None:
            print("create_cells: Error: self.tile_info is None")
            return
        
        top: Union[None, np.ndarray] = None
        
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
        nudge_enabled = (
            self.tile_info.config.nudge_in_overused_edges_vertex
            and not self.tile_info.config.no_bottom
        )

        for j in range(1, self.ymaxidx+1):# y dimension for looping within the +1 padded raster
            if j % pc_step == 0:
                progress += percent
                print(progress, "%", multiprocessing.current_process(), file=sys.stderr)

            for i in range(1, self.xmaxidx + 1):# x dim.
                #print("y=",j," x=",i, " elev=",top[j,i])

                # for bottom_thru_base we must use the pre-dilated, but for NaN'd top only use this check 
                # same for top with NaNs which have been 3x3 dilated
                # dirty_trianglescreates a technically better fit fit of the water into the terrain but will create triangles
                # that are collapsed into a line or a point. This should not be a problem for a modern slicer but will
                # lead to issues when using the model in a 3D mesh modeling program
                # Top set here determines which cells to skip based on the cells' values
                if self.tile_info.have_nan == True and self.tile_info.config.dirty_triangles == False:
                    top = self.tile.top_raster_variants.dilated
                else:
                    top = self.tile.top_raster_variants.dilated
                    
                # For Difference Mesh mode + bottom_thru_base
                if self.tile.bottom_raster_variants is not None and self.tile_info.config.bottom_thru_base:
                        top = self.tile.bottom_raster_variants.nan_close

                # if center elevation of current top cell is NaN, set its cell to None and skip the rest
                if self.tile_info.have_nan and np.isnan(top[j, i]):
                    self.cells[j-1, i-1] = None
                    continue
                
                # x/y coords of cell "walls", origin is upper left
                W = (i-1) * self.cell_size - self.offsetx # index -1 as it's ref'ing to top, not ptop
                E = W + self.cell_size
                N = -(j-1) * self.cell_size + self.offsety # y is flipped to negative
                S = N - self.cell_size
                #print(i,j, " ", E,W, " ",  N,S, " ", top[j,i])
                
                

                
                
                
                #region Make top cell vertices' heights
                interpolation_top_raster: Union[np.ndarray, None]
                if not self.tile_info.have_nan:
                    interpolation_top_raster = self.tile.top_raster_variants.dilated
                    # non NaNs: interpolate elevation of four corners (array order is top[y,x]!)
                    NEelev = interpolate_corner_with_canonical_order(
                        interpolation_top_raster,
                        j - 1,
                        i,
                    )
                    NWelev = interpolate_corner_with_canonical_order(
                        interpolation_top_raster,
                        j - 1,
                        i - 1,
                    )
                    SEelev = interpolate_corner_with_canonical_order(
                        interpolation_top_raster,
                        j,
                        i,
                    )
                    SWelev = interpolate_corner_with_canonical_order(
                        interpolation_top_raster,
                        j,
                        i - 1,
                    )
                else:
                    # NaNs: set borders to True if we have any NaNs in any of the adjacent cells
                    # Do this only for top as we assume that any bottom raster NaNs are the same as on top

                    # Interpolate with edge_interpolation raster variant if available. 
                    interpolation_top_raster = self.tile.top_raster_variants.edge_interpolation
                    if interpolation_top_raster is None:
                        if self.tile.bottom_raster_variants is None:
                            # Normal (not difference mesh) mode
                            # Otherwise use "original" top raster (it's only modified at top_hint mask locs to bottom_floor_elev
                            interpolation_top_raster = self.tile.top_raster_variants.original
                            # Use top.dilated for borders
                        else:
                            # Difference mesh mode
                            interpolation_top_raster = self.tile.top_raster_variants.original
                            if self.tile_info.config.bottom_thru_base:
                                # Use original top raster so we get accurate NaN location and borders
                                interpolation_top_raster = self.tile.top_raster_variants.original

                    # get values for current cell i, j, NEelev, NWelev, SEelev, SWelev
                    NEelev, NWelev, SEelev, SWelev = interpolate_with_NaN(interpolation_top_raster, i, j)

                    # top
                    # set breakpoint for specific points for debugging
                    # if j == 10 and i ==9:
                    #     0==0

                    if NEelev is None: # if any of the corners is NaN, we have set the cell to None and can skip it
                        continue
                    
                    # compare values with real print3D heights at this point
                    # Pull values set to bottom_floor_elev (which will be just below basethick) to actual 0 because we added basethick to all raster.
                    if NEelev < self.tile_info.config.basethick:
                        NEelev = 0
                    if NWelev < self.tile_info.config.basethick:
                        NWelev = 0
                    if SEelev < self.tile_info.config.basethick:
                        SEelev = 0
                    if SWelev < self.tile_info.config.basethick:
                        SWelev = 0
                
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
                top_bottom_surface_polygons_triangulated_2D: list[shapely.GeometryCollection] | None = None # tris for top and bottom surfaces
                # Check if non-quad top_surface polygon should be used
                top_surface_polygons_triangulated_3D: (
                    list[shapely.Polygon | None] | None
                ) = None
                clipped_surfaces_collapsed_after_output = False
                # by checking if the cell is NOT contains_properly and if it has polygon_intersection_geometry
                if (self.tile.top_raster_variants.polygon_intersection_contains_properly is not None and self.tile.top_raster_variants.polygon_intersection_contains_properly[j-1][i-1] == False) and self.tile.top_raster_variants.polygon_intersection_geometry is not None:
                    top_bottom_surface_polygons_triangulated_2D = []
                    
                    top_bottom_surface_geometries_2D = self.tile.top_raster_variants.polygon_intersection_geometry[j-1][i-1]
                    if top_bottom_surface_geometries_2D is not None:
                        # We can verify if our shapely utils coordinate converter matches the N W S E made in create_cells. (it does if you adjust for the padding difference)
                        #quadPrint2DCoords = utils.arrayCellCoordToQuadPrint2DCoords(array_coord_2D=(i-1,j-1), cell_size=self.cell_size, tile_y_shape=self.tile.top_raster_variants.polygon_intersection_geometry.shape[0])
                        top_bottom_surface_geometries_2D = self.tile.top_raster_variants.polygon_intersection_geometry[j-1][i-1]
                        top_surface_polygons_triangulated_3D = [] #init array
                        # Only interpolate the Polygons for the top surface
                        for polygon in [item for item in (top_bottom_surface_geometries_2D if top_bottom_surface_geometries_2D else []) if isinstance(item, shapely.Polygon)]:
                            top_bottom_surface_polygons_triangulated_2D.append(shapely.constrained_delaunay_triangles(polygon))
                        for gc in top_bottom_surface_polygons_triangulated_2D:
                            for tri in [item for item in gc.geoms if isinstance(item, shapely.Polygon)]:
                                tri_ccw_order = shapely.orient_polygons(tri, exterior_cw=False)
                                tri_with_z = interpolate_z_planar(geometry_2d=tri_ccw_order, planes_3d=topq.get_triangles_in_polygons(split_rotation=self.tile_info.config.split_rotation))
                                #tri_with_z = interpolate_geometry_with_quad(geometry=tri_ccw_order, quad=topq, split_rotation=self.tile_info.config.split_rotation)
                                if isinstance(tri_with_z, shapely.Polygon):
                                    top_surface_polygons_triangulated_3D.append(
                                        normalize_polygon_for_serialization(
                                            tri_with_z,
                                            output_fileformat,
                                        )
                                    )
                                else:
                                    raise ValueError(f"tri_with_z is not Polygon, it is {type(tri_with_z)}")
                
                #endregion
                
                #
                #region Make bottom quad  
                #

                # get corners for bottom array
                bottom_raster_for_z0_nudge: np.ndarray | None = None
                if self.tile.bottom_raster_variants is None:
                    # Normal mode
                    NEelev = NWelev = SEelev = SWelev = 0
                else:
                    # Difference mode
                    # for the through water case, simply set the bottom to 0
                    if self.bottom_thru_base == True:
                        NEelev = NWelev = SEelev = SWelev = 0
                    else:
                        # simple interpolation
                        if not self.tile_info.have_bot_nan:
                            bottom_raster = (
                                self.tile.bottom_raster_variants.dilated
                            )
                            bottom_raster_for_z0_nudge = bottom_raster
                            NEelev = interpolate_corner_with_canonical_order(
                                bottom_raster,
                                j - 1,
                                i,
                            )
                            NWelev = interpolate_corner_with_canonical_order(
                                bottom_raster,
                                j - 1,
                                i - 1,
                            )
                            SEelev = interpolate_corner_with_canonical_order(
                                bottom_raster,
                                j,
                                i,
                            )
                            SWelev = interpolate_corner_with_canonical_order(
                                bottom_raster,
                                j,
                                i - 1,
                            )
                        else:
                            # Nan aware interpolation 
                            bottom_raster_for_z0_nudge = (
                                self.tile.bottom_raster_variants.original
                            )
                            NEelev, NWelev, SEelev, SWelev = interpolate_with_NaN(self.tile.bottom_raster_variants.original, i, j)
                            
                            # bottom
                            # set breakpoint for specific points for debugging
                            # if j == 10 and i ==9:
                            #     0==0
                            
                            if NEelev is None: # if any of the corners is NaN, we have set the cell to None and are skippping it
                                continue # skip this cell
                            
                            # Pull values set to bottom_floor_elev to actual 0
                            # compare values with real print3D heights at this point
                            if NEelev < self.tile_info.config.basethick:
                                NEelev = 0
                            if NWelev < self.tile_info.config.basethick:
                                NWelev = 0
                            if SEelev < self.tile_info.config.basethick:
                                SEelev = 0
                            if SWelev < self.tile_info.config.basethick:
                                SWelev = 0

                # from whatever bottom values we have now, make the bottom quad
                # (if we do the 2 tri bottom, these will end up not be used for the bottom but they may be used for any walls ...)
                NEb = vertex(E, N, NEelev)
                NWb = vertex(W, N, NWelev)
                SEb = vertex(E, S, SEelev)
                SWb = vertex(W, S, SWelev)
                botq = quad(NWb, NEb, SEb, SWb)

                # Check if non-quad top_surface polygon should be used for bottom quad
                bottom_surface_polygons_triangulated_3D: (
                    list[shapely.Polygon | None] | None
                ) = None
                if top_bottom_surface_polygons_triangulated_2D is not None:
                    # We can verify if our shapely utils coordinate converter matches the N W S E made in create_cells. (it does if you adjust for the padding difference)
                    #quadPrint2DCoords = utils.arrayCellCoordToQuadPrint2DCoords(array_coord_2D=(i-1,j-1), cell_size=self.cell_size, tile_y_shape=self.tile.top_raster_variants.polygon_intersection_geometry.shape[0])
                    
                    bottom_surface_polygons_triangulated_3D = []
                    # Only interpolate the Polygons for the top surface
                    for gc in top_bottom_surface_polygons_triangulated_2D:
                        for tri in [item for item in gc.geoms if isinstance(item, shapely.Polygon)]:
                            tri_cw_order = shapely.orient_polygons(tri, exterior_cw=True)
                            tri_with_z = interpolate_z_planar(geometry_2d=tri_cw_order, planes_3d=botq.get_triangles_in_polygons(split_rotation=self.tile_info.config.split_rotation))
                            if isinstance(tri_with_z, shapely.Polygon):
                                bottom_surface_polygons_triangulated_3D.append(
                                    normalize_polygon_for_serialization(
                                        tri_with_z,
                                        output_fileformat,
                                    )
                                )
                            else:
                                raise TypeError('tri_with_z is not Polygon. interpolate_z_planar did not return the same Polygon type passed in')

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
                if nudge_enabled:
                    if self.tile.bottom_raster_variants is None:
                        z0_detection_raster = interpolation_top_raster
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
                        top_corner_vertices = {
                            IntermediateCorner.NW: NWt,
                            IntermediateCorner.NE: NEt,
                            IntermediateCorner.SW: SWt,
                            IntermediateCorner.SE: SEt,
                        }
                        bottom_corner_vertices = {
                            IntermediateCorner.NW: NWb,
                            IntermediateCorner.NE: NEb,
                            IntermediateCorner.SW: SWb,
                            IntermediateCorner.SE: SEb,
                        }

                        def output_z_is_zero(v: vertex) -> bool:
                            return (
                                normalize_vertex_to_match_mesh_serialization(
                                    v.coords,
                                    output_fileformat,
                                )[2]
                                == 0
                            )

                        if self.tile.bottom_raster_variants is None:
                            z0_corners = [
                                corner
                                for corner in z0_corners
                                if output_z_is_zero(top_corner_vertices[corner])
                                and output_z_is_zero(
                                    bottom_corner_vertices[corner],
                                )
                            ]
                        else:
                            z0_corners = [
                                corner
                                for corner in z0_corners
                                if output_z_is_zero(
                                    bottom_corner_vertices[corner],
                                )
                            ]
                    if 0 < len(z0_corners) < 4:
                        full_cell_footprint = _z0_full_cell_footprint(
                            W,
                            E,
                            N,
                            S,
                        )
                        keep_footprint = _z0_normal_keep_footprint(
                            z0_corners,
                            W,
                            E,
                            N,
                            S,
                        )
                        if keep_footprint is not None:
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
                                split_rotation=self.tile_info.config.split_rotation,
                            )
                            z0_top_planes = _z0_adjusted_keep_surface_planes(
                                z0_corners,
                                top_corner_vertices,
                                W,
                                E,
                                N,
                                S,
                            )
                            z0_bottom_planes = _z0_adjusted_keep_surface_planes(
                                z0_corners,
                                bottom_corner_vertices,
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
                                split_rotation=self.tile_info.config.split_rotation,
                            )
                            if self.tile.bottom_raster_variants is None:
                                top_surface_polygons_triangulated_3D = (
                                    _triangulate_2d_geometry_to_3d_polygons(
                                        keep_geometry,
                                        z0_top_planes,
                                        exterior_cw=False,
                                        output_fileformat=output_fileformat,
                                    )
                                )
                                bottom_surface_polygons_triangulated_3D = (
                                    _triangulate_2d_geometry_to_3d_polygons(
                                        keep_geometry,
                                        z0_planes,
                                        exterior_cw=True,
                                        output_fileformat=output_fileformat,
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
                                        )
                                    )

                                bottom_surface_polygons_triangulated_3D = (
                                    kept_bottom_polygons
                                    + _triangulate_2d_geometry_to_3d_polygons(
                                        complement_geometry,
                                        z0_planes,
                                        exterior_cw=True,
                                        output_fileformat=output_fileformat,
                                    )
                                )

                            if (
                                top_surface_polygons_triangulated_3D
                                and bottom_surface_polygons_triangulated_3D
                            ):
                                z0_nudged_cell = True
                            else:
                                clipped_surfaces_collapsed_after_output = True

                if clipped_surfaces_collapsed_after_output:
                    self.cells[j - 1, i - 1] = None
                    continue

                #
                #region Make borders
                #

                # Simple rectangular mesh case with no NaN
                # Which directions will need to have a wall?
                # True means: we have an adjacent cell and need a wall in that direction
                borders =   dict([[drct, False] for drct in ["N", "S", "E", "W"]]) # init with no walls                   
                # set walls for fringe cells
                if j == 1             : borders["N"] = True
                if j == self.ymaxidx  : borders["S"] = True
                if i == 1             : borders["W"] = True
                if i == self.xmaxidx  : borders["E"] = True
                
                #cell_clipping_intersection_geometry = self.tile.top_raster_variants.polygon_intersection_geometry[j][i] if self.tile.top_raster_variants.polygon_intersection_geometry else None
                #if cell_clipping_intersection_geometry and len(cell_clipping_intersection_geometry) > 0: # Cell is partially intersecting or sharing edges with polygon
                if False:
                    pass
                else: # Cell contained properly in polygon
                    # Figure out which raster array to use when determining borders
                    borders_top_raster: Union[np.ndarray, None] = None
                    if self.tile.bottom_raster_variants is None:
                        # Normal mode
                        borders_top_raster = self.tile.top_raster_variants.dilated

                    else:
                        # Difference mesh mode
                        #force dilated top because using predilated version has NaNs at edge which makes extra walls
                        borders_top_raster = self.tile.top_raster_variants.dilated
                        
                        #for difference mesh in bottom_thru_base case, check for walls with the nan_close version before dilation
                        if self.bottom_thru_base == True:
                            borders_top_raster = self.tile.top_raster_variants.nan_close
                    
                    with warnings.catch_warnings():
                        warnings.filterwarnings('error')
                        try:
                            if np.isnan(borders_top_raster[j-1,i]): borders["N"] = True
                            if np.isnan(borders_top_raster[j+1,i]): borders["S"] = True
                            if np.isnan(borders_top_raster[j,i-1]): borders["W"] = True
                            if np.isnan(borders_top_raster[j,i+1]): borders["E"] = True
                        except RuntimeWarning:
                            pass # nothing wrong - just here to ignore the warning

                    # Quads for walls: in borders dict, replace any True with a quad of that wall
                    if borders["N"] == True:
                        borders["N"] = (
                            make_wall_without_exact_duplicate_vertices(
                                NWb,
                                NWt,
                                NEt,
                                NEb,
                                output_fileformat=output_fileformat,
                            ) or False
                        )
                    if borders["S"] == True:
                        borders["S"] = (
                            make_wall_without_exact_duplicate_vertices(
                                SEb,
                                SEt,
                                SWt,
                                SWb,
                                output_fileformat=output_fileformat,
                            ) or False
                        )
                    if borders["E"] == True:
                        borders["E"] = (
                            make_wall_without_exact_duplicate_vertices(
                                NEt,
                                SEt,
                                SEb,
                                NEb,
                                output_fileformat=output_fileformat,
                            ) or False
                        )
                    if borders["W"] == True:
                        borders["W"] = (
                            make_wall_without_exact_duplicate_vertices(
                                SWt,
                                NWt,
                                NWb,
                                SWb,
                                output_fileformat=output_fileformat,
                            ) or False
                        )

                # create borders if there is a top surface polygon using the edge buckets
                surface_polygon_borders_3D: list[quad] = []
                if self.tile.top_raster_variants.polygon_intersection_edge_buckets is not None and self.tile.top_raster_variants.polygon_intersection_edge_buckets[j-1][i-1] is not None:
                    # Get list of all BorderEdges with edge geometry that should be walls
                    wall_borderEdges = []
                    buckets = self.tile.top_raster_variants.polygon_intersection_edge_buckets[j-1][i-1]
                    if isinstance(buckets, dict):
                        for bucket in buckets.values():
                            if isinstance(bucket, list):
                                for be in bucket:
                                    if isinstance(be, BorderEdge):
                                        if be.make_wall:
                                            wall_borderEdges.append(be)
                    
                    if top_bottom_surface_geometries_2D:
                        top_surface_edges_3D: list[shapely.LineString] = []
                        bot_surface_edges_3D: list[shapely.LineString] = []
                        for geom in top_bottom_surface_geometries_2D:
                            if isinstance(geom, shapely.Polygon):
                                flattened_top_geom = flatten_geometries(geometries=[interpolate_z_planar(geometry_2d=shapely.orient_polygons(geom, exterior_cw=False), planes_3d=topq.get_triangles_in_polygons(split_rotation=self.tile_info.config.split_rotation))], to_single_lines=True)
                                top_surface_edges_3D.extend([item for item in flattened_top_geom if isinstance(item, shapely.LineString)])
                                flattened_bot_geom = flatten_geometries(geometries=[interpolate_z_planar(geometry_2d=shapely.orient_polygons(geom, exterior_cw=True), planes_3d=botq.get_triangles_in_polygons(split_rotation=self.tile_info.config.split_rotation))], to_single_lines=True)
                                bot_surface_edges_3D.extend([item for item in flattened_bot_geom if isinstance(item, shapely.LineString)])
                        
                        # Find matching interpolated top and bottom surface edges for all BorderEdges that should be a wall
                        # This match is O(n^2) and optimized to O(n^2/2) in the optimal case by removing matched top/bottom surface edges from the array so matched edges are looked through again
                        for be in wall_borderEdges:
                            topEdgeMatch: shapely.LineString | None = None
                            botEdgeMatch: shapely.LineString | None = None
                            for ti in range(0, len(top_surface_edges_3D)):
                                if be.geometry.equals(top_surface_edges_3D[ti]):
                                    topEdgeMatch = top_surface_edges_3D[ti]
                                    del top_surface_edges_3D[ti]
                                    break
                            for bi in range(0, len(bot_surface_edges_3D)):
                                if be.geometry.equals(bot_surface_edges_3D[bi]):
                                    botEdgeMatch = bot_surface_edges_3D[bi]
                                    del bot_surface_edges_3D[bi]
                                    break
                            if topEdgeMatch and botEdgeMatch:
                                # Success condition where wall BorderEdge and top edge and bottom edge share the same X,Y endpoints
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
                                pass
                            elif topEdgeMatch:
                                raise RuntimeError('Border creation: top edge match found but no bot edge match.')
                            elif botEdgeMatch:
                                raise RuntimeError('Border creation: bot edge match found but no top edge match.')
                            else:
                                raise RuntimeError('Border creation: no top edge or bot edge matches')
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
                        z0_full_footprint_2D = _z0_full_cell_footprint(
                            W,
                            E,
                            N,
                            S,
                        )
                    surface_polygon_borders_3D = (
                        _z0_rebuild_surface_polygon_borders(
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
                    borders = {drct: False for drct in ["N", "S", "E", "W"]}

                #region Make cell
                if self.tile_info.config.no_bottom == True:
                    c = cell(topq, None, borders) # omit bottom - do not fill with 2 tris later (may have NaNs)
                else:
                    if self.tile_info.have_nan == True or self.tile.bottom_raster_variants is not None or nudge_enabled: #self.tile_info.have_bottom_array == True: 
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
                ):
                    bottom_provider = self.tile.bottom_surface_provider
                    bottom_quad, bottom_polygons = bottom_provider[j - 1][
                        i - 1
                    ]
                    if bottom_quad is None and bottom_polygons is None:
                        if (
                            c.bottomquad is None
                            and c.bottomSurfacePolygons is None
                        ):
                            raise RuntimeError(
                                "Interlocking pair difference cell needs a "
                                f"bottom surface at row={j - 1}, col={i - 1}."
                            )
                    else:
                        c.replace_bottom_surfaces(
                            bottom_surface_quad=bottom_quad,
                            bottom_surface_polygons=bottom_polygons,
                            split_rotation=(
                                self.tile_info.config.split_rotation
                            ),
                            output_fileformat=output_fileformat,
                        )

                # DEBUG: store i,j, and central elev
                #c.iy = j-1
                #c.ix = i-1
                #c.central_elev = top[j-1,i-1]

                if j == 10 and i == 10:
                    pass

                if self.tile.bottom_raster_variants is not None or nudge_enabled:
                    c.remove_zero_height_volumes(
                        split_rotation=self.tile_info.config.split_rotation,
                        output_fileformat=output_fileformat,
                    )

                # if we have nan cells, do some postprocessing on this cell to get rid of stair case patterns
                # This will create special triangle cells that have a triangle of any orientation at top/bottom, which 
                # are flagged as is_tri_cell = True, and have only v0, v1 and v2. One border is deleted, the other
                # is set as a diagonal wall.
                # Note: this will not be done if we have a bottom as it will lead to lots of triangle holes! 
                if (
                    self.tile_info.have_nan == True
                    and self.tile_info.config.smooth_borders == True
                    and self.tile.bottom_raster_variants is None
                ):
                    #print(i,j, c.borders)
                    if c.check_for_tri_cell():
                        c.convert_to_tri_cell()

                # Normalize and remove geometry that would collapse during mesh
                # serialization before storing the cell.
                c.remove_geometry_collapsed_by_mesh_serialization(
                    output_fileformat=output_fileformat,
                    split_rotation=self.tile_info.config.split_rotation,
                )

                self.cells[j - 1, i - 1] = c

                #endregion

                #
                # Make quads for top, bottom and walls
                #

                # list of meshes for this cell,
                meshes = c.meshes_for_model()
                
                # # list of quads for this cell,
                # if no_bottom == False and (self.tile_info.have_nan or self.tile.bottom_raster_variants is not None): #self.tile_info.have_bottom_array): #  
                #     quads = [c.topquad, c.bottomquad]
                # else:
                #     quads = [c.topquad] # no bottom quads, only top

                # add border quads if we have any (False means no border quad) 
                # for k in c.borders:  # k is N, S, E, W
                #     if c.borders[k] is not False: meshes.append(c.borders[k])
                        
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

                # Debug: inspect cell
                if j == 6 and i == 5:
                    pass

                # write the triangles of the meshes to buffer
                for q in meshes:
                    mesh_triangles: list[list[vertex]] = []
                    if isinstance(q, quad):
                        quad_triangles = q.get_triangles(
                            split_rotation=(
                                self.tile_info.config.split_rotation
                            ),
                        )

                        # For STL this writes triangle vertices. For OBJ it
                        # writes indices into s[1]/fo[1].
                        for t in quad_triangles:
                            #mesh_triangles.append(list(t))
                            mesh_triangles.append(
                                triangle_rounded_to_precision(
                                    decimals=decimal_precision,
                                    triangle=list(t),
                                )
                            )
                        # if any(t0):
                        #     self.write_triangle_to_buffer(t0)
                        #     self.write_triangle_to_buffer(t1) # could be empty ...
                    elif isinstance(q, shapely.Polygon):
                        pass
                        t0 = tuple(polygon_to_list_of_vertex(polygon=q))
                        if len(t0) == 4 and t0[0].coords == t0[3].coords:
                           #mesh_triangles.append(list(t0[:3]))
                           mesh_triangles.append(
                               triangle_rounded_to_precision(
                                   decimals=decimal_precision,
                                   triangle=list(t0[:3]),
                               )
                           )
                        else:
                           raise ValueError(
                               "create_cells: found a polygon to write to "
                               "buffer that is not a triangle. Expected a "
                               "tri of length 3+1=4 and [0]==[3] vertex. "
                               f"Polygon had vertex count f{len(t0)}."
                           )

                    for mt in mesh_triangles:
                        self.write_triangle_to_buffer(tuple(mt))
        
        print("100%", multiprocessing.current_process(), "\n", file=sys.stderr)
    
    def write_triangle_to_buffer(self, t: tuple[vertex, ...]):
        '''write triangle vertices for triangle t to stream buffer self.s for caching.
        Once the cache is full, is is writting to disk (self.fo)'''
        
        if t is None: return # just for the case that one of the two triangle was removed by smoothing
        
        #print(self.num_triangles, end=", ")
        self.num_triangles += 1

        # Create triangle coords list, for STL including normal coords (no normals for obj)
        if self.tile_info.config.fileformat != "obj":
            tl = get_normal(t) if self.tile_info.config.no_normals == False else [0,0,0]
            for v in t:
                coords = v.get() # get() => list of coords [x,y,z]
                # pack 64 bit float to 32 bit and unpack 32 bit back to 64 bit to try to get the same value represented in 32 bit
                #coords = tuple(map(lambda x: struct.unpack('<f', struct.pack('<f', x))[0], coords))
                # add 0.0 to value to force -0 value to +0
                coords = tuple(map(lambda x: x+0.0, coords))
                tl.extend(coords) # like append() but extend() unpacks that list!
            tl.append(0) # append attribute byte 0

        if self.tile_info.config.fileformat == "STLb":
            # en.wikipedia.org/wiki/STL_%28file_format%29#Binary_STL
            BINARY_FACET = "<12fH" # little endian order 12 32-bit floating-point numbers + 2-byte ("short") unsigned integer ("attribute byte count" -> use 0)
            self.s.write(struct.pack(BINARY_FACET, *tl)) # append to s

        elif self.tile_info.config.fileformat == "STLa":
            ASCII_FACET = (
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
            self.s.write(
                ASCII_FACET.format(
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
        if self.tile_info.config.fileformat != "obj":  
            self.write_buffer_to_file()
            
    def write_buffer_to_file(self, flush=False, chunk_size=100000):
        # write buffer to file every 10k triangles
        # chunksize is the number of triangles that need to have been collected into the buffer in order to actually write to disk. (cache)
        # flusk=True forces a write: use this to flush whatever is in the buffer.  Will NOT close the file!
        # for obj, write only the indices [1], vertices [0] will be done later
        
        # Only write to file if we're actually using temp files, otherwise just bail out
        if self.tile_info.temp_file is None:
            return
        
        if self.num_triangles % chunk_size == 0  or flush == True:
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

        if flush == True:
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
    def make_file_buffer(self):
        
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

        # Open in-memory stream buffers s 
        # s is used to collect the data that is eventually written into a proper file
        if self.tile_info.config.fileformat == "STLb":
            self.s = io.BytesIO()
            mode = "ab"  # for using open() later
        elif self.tile_info.config.fileformat == "STLa":
            self.s = io.StringIO() 
            mode = "a"
        elif self.tile_info.config.fileformat == "obj":
            mode = "a"   
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

        # populate self.cells, will write triangles into buffer/file
        self.create_cells()

        # Can we use 2-triangle bottoms?
        add_simple_bottom = True # True by default, set to False if we can't create a 2-triangle bottom
        
        # We don't have bottom tris but that's OK as we don't them anyway (no_bottom option was set)
        if self.tile_info.config.no_bottom == True: add_simple_bottom = False # 
        
        # With a NaN (masked) top array, we already have the corresponding full bottom
        if self.tile_info.have_nan == True: add_simple_bottom = False 
        
        # with a bottom image/elevation, we also already need a full bottom
        if self.tile_info.config.bottom_image != None or self.tile_info.config.bottom_elevation != None: 
            add_simple_bottom = False

        # Z0 nudging changes per-cell footprints, so flagged normal tiles use
        # per-cell bottoms instead of one tile-wide simple bottom.
        if (
            self.tile_info.config.nudge_in_overused_edges_vertex
            and not self.tile_info.config.no_bottom
        ):
            add_simple_bottom = False

        # obj files currently don't support simple bottoms
        #if self.tile_info.fileformat == 'obj': add_simple_bottom = False

        # For simple bottom, add 2 triangles based on the corners of the tile
        if add_simple_bottom:
            v0 = vertex(self.tile_info.W, self.tile_info.S, 0)
            v1 = vertex(self.tile_info.E, self.tile_info.S, 0)
            v2 = vertex(self.tile_info.E, self.tile_info.N, 0)
            v3 = vertex(self.tile_info.W, self.tile_info.N, 0)

            t0 = (v0, v2, v1) #A
            t1 = (v0, v3, v2) #B

            self.write_triangle_to_buffer(t0) #
            self.write_triangle_to_buffer(t1)

        # using buffer 
        if temp_file is None: 
        
            # finish STLa stream buffer
            if self.tile_info.config.fileformat == "STLa":
                self.s.write('endsolid digital_elevation_model') # append end clause
                buf = self.s.getvalue()

            # For STLb buffer, prepend the header
            if self.tile_info.config.fileformat == "STLb":
                BINARY_HEADER = "80sI" # up to 80 chars do NOT start with the word solid + number of faces as UINT32
                stlb_header = io.BytesIO()
                stlb_header.write(struct.pack(BINARY_HEADER, b'Binary STL Writer', self.num_triangles))
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
                        BINARY_HEADER = "80sI" # up to 80 chars do NOT start with the word solid + number of faces as UINT32
                        fheader.write(struct.pack(BINARY_HEADER, b'Binary STL Writer', self.num_triangles))
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

