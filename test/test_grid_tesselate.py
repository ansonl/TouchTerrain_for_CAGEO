import unittest

import shapely

from touchterrain.common.nudge_corner import IntermediateCorner
from touchterrain.common.Quad import quad
from touchterrain.common.Vertex import vertex
from touchterrain.common.grid_tesselate import (
    _rebuild_matching_surface_polygon_borders,
    _triangulate_2d_geometry_to_3d_polygons,
    _z0_adjusted_keep_surface_planes,
    _z0_normal_keep_footprint,
    boundary_edge_map_from_meshes,
    cell,
    edge_xy_signature,
    make_wall_without_exact_duplicate_vertices,
    normalize_vertex_to_match_mesh_serialization,
)


class TestWallMeshes(unittest.TestCase):
    def test_wall_with_no_duplicate_vertices_stays_quad(self):
        wall = make_wall_without_exact_duplicate_vertices(
            vertex(0, 0, 0),
            vertex(0, 0, 1),
            vertex(1, 0, 1),
            vertex(1, 0, 0),
        )

        self.assertIsInstance(wall, quad)
        self.assertIsNotNone(wall.vl[3])

    def test_wall_with_one_exact_duplicate_endpoint_becomes_triangle(self):
        wall = make_wall_without_exact_duplicate_vertices(
            vertex(0, 0, 0),
            vertex(0, 0, 0),
            vertex(1, 0, 1),
            vertex(1, 0, 0),
        )

        self.assertIsInstance(wall, quad)
        self.assertIsNone(wall.vl[3])
        self.assertEqual([v.coords for v in wall.vl[:3]], [(0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (1.0, 0.0, 0.0)])

    def test_wall_with_both_exact_duplicate_endpoints_is_omitted(self):
        wall = make_wall_without_exact_duplicate_vertices(
            vertex(0, 0, 0),
            vertex(0, 0, 0),
            vertex(1, 0, 1),
            vertex(1, 0, 1),
        )

        self.assertIsNone(wall)

    def test_wall_with_nonadjacent_exact_duplicate_endpoint_becomes_triangle(self):
        wall = make_wall_without_exact_duplicate_vertices(
            vertex(0, 0, 1),
            vertex(1, 0, 1),
            vertex(0, 0, 1),
            vertex(1, 0, 0),
        )

        self.assertIsInstance(wall, quad)
        self.assertIsNone(wall.vl[3])
        self.assertEqual([v.coords for v in wall.vl[:3]], [(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 0.0, 0.0)])

    def test_rebuild_clipped_wall_splits_surface_boundary_edges(self):
        fileformat = "STLb"
        top_surfaces = [
            shapely.Polygon(
                [
                    (0.0, 0.0, 1.0),
                    (1.0, 0.0, 2.0),
                    (0.0, 1.0, 1.2),
                    (0.0, 0.0, 1.0),
                ]
            )
        ]
        bottom_surfaces = [
            shapely.Polygon(
                [
                    (0.0, 0.0, 0.0),
                    (0.5, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0),
                ]
            ),
            shapely.Polygon(
                [
                    (0.5, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.5, 0.0, 0.0),
                ]
            ),
        ]
        requested = shapely.LineString([(0.0, 0.0), (1.0, 0.0)])

        walls = _rebuild_matching_surface_polygon_borders(
            top_surfaces,
            bottom_surfaces,
            lambda footprint: requested.covers(shapely.LineString(footprint)),
            fileformat,
        )

        expected_footprints = {
            edge_xy_signature((0.0, 0.0, 0.0), (0.5, 0.0, 0.0)),
            edge_xy_signature((0.5, 0.0, 0.0), (1.0, 0.0, 0.0)),
        }
        self.assertEqual(len(walls), 2)
        self.assertTrue(
            expected_footprints.issubset(
                set(
                    boundary_edge_map_from_meshes(
                        top_surfaces,
                        output_fileformat=fileformat,
                    )
                )
            )
        )
        self.assertTrue(
            expected_footprints.issubset(
                set(
                    boundary_edge_map_from_meshes(
                        bottom_surfaces,
                        output_fileformat=fileformat,
                    )
                )
            )
        )
        wall_footprints = {
            edge_xy_signature(wall.vl[0].coords, wall.vl[1].coords)
            for wall in walls
        }
        self.assertEqual(wall_footprints, expected_footprints)


class TestSerializedSurfaceCleanup(unittest.TestCase):
    def test_removes_surface_triangle_with_serialized_xy_line(self):
        current_cell = cell(
            quad(
                vertex(0.0, 0.0, 1.0),
                vertex(0.0, 1.0, 1.0),
                vertex(1.0, 1.0, 1.0),
                vertex(1.0, 0.0, 1.0),
            ),
            quad(
                vertex(0.0, 0.0, 0.0),
                vertex(1.0, 0.0, 0.0),
                vertex(1.0, 1.0, 0.0),
                vertex(0.0, 1.0, 0.0),
            ),
            {direction: False for direction in ["N", "S", "E", "W"]},
        )
        current_cell.topSurfacePolygons = [
            shapely.Polygon(
                [
                    (0.0, 0.0, 1.0),
                    (1.0, 0.0, 2.0),
                    (0.5, 0.0, 1.5),
                    (0.0, 0.0, 1.0),
                ]
            )
        ]

        current_cell.remove_geometry_collapsed_by_mesh_serialization(
            output_fileformat="STLb",
            split_rotation=0,
        )

        self.assertIsNone(current_cell.topSurfacePolygons)
        self.assertIsNone(current_cell.topquad)
        self.assertIsNone(current_cell.bottomquad)

    def test_keeps_surface_triangle_with_serialized_xy_area(self):
        current_cell = cell(
            quad(
                vertex(0.0, 0.0, 1.0),
                vertex(0.0, 1.0, 1.0),
                vertex(1.0, 1.0, 1.0),
                vertex(1.0, 0.0, 1.0),
            ),
            None,
            {direction: False for direction in ["N", "S", "E", "W"]},
        )
        current_cell.topSurfacePolygons = [
            shapely.Polygon(
                [
                    (0.0, 0.0, 1.0),
                    (1.0, 0.0, 1.2),
                    (0.0, 1.0, 1.4),
                    (0.0, 0.0, 1.0),
                ]
            )
        ]

        current_cell.remove_geometry_collapsed_by_mesh_serialization(
            output_fileformat="STLb",
            split_rotation=0,
        )

        self.assertEqual(len(current_cell.topSurfacePolygons), 1)

    def test_keeps_vertical_wall_with_serialized_xy_line(self):
        current_cell = cell(
            quad(
                vertex(0.0, 0.0, 1.0),
                vertex(0.0, 1.0, 1.0),
                vertex(1.0, 1.0, 1.0),
                vertex(1.0, 0.0, 1.0),
            ),
            None,
            {direction: False for direction in ["N", "S", "E", "W"]},
        )
        current_cell.surfacePolygonBorders = [
            quad(
                vertex(0.0, 0.0, 1.0),
                vertex(1.0, 0.0, 1.0),
                vertex(1.0, 0.0, 0.0),
                vertex(0.0, 0.0, 0.0),
            )
        ]

        current_cell.remove_geometry_collapsed_by_mesh_serialization(
            output_fileformat="STLb",
            split_rotation=0,
        )

        self.assertEqual(len(current_cell.surfacePolygonBorders), 1)


class TestZ0NudgePrecision(unittest.TestCase):
    def test_nudged_shared_edge_keeps_neighbor_precision_until_output(self):
        fileformat = "STLb"
        w, e, n, s = 52.9, 53.0, 153.3, 153.2
        corner_vertices = {
            IntermediateCorner.NW: vertex(w, n, 1.432),
            IntermediateCorner.NE: vertex(e, n, 0.929),
            IntermediateCorner.SW: vertex(w, s, 1.189),
            IntermediateCorner.SE: vertex(e, s, 0.0),
        }
        affected_corners = [IntermediateCorner.SE]

        keep_footprint = _z0_normal_keep_footprint(
            affected_corners,
            w,
            e,
            n,
            s,
        )
        surface_planes = _z0_adjusted_keep_surface_planes(
            affected_corners,
            corner_vertices,
            w,
            e,
            n,
            s,
        )
        nudged_surfaces = _triangulate_2d_geometry_to_3d_polygons(
            keep_footprint,
            surface_planes,
            exterior_cw=False,
            output_fileformat=fileformat,
        )

        boundary_edges = boundary_edge_map_from_meshes(
            nudged_surfaces,
            output_fileformat=fileformat,
        )
        north_footprint = edge_xy_signature(
            normalize_vertex_to_match_mesh_serialization((w, n, 0), fileformat),
            normalize_vertex_to_match_mesh_serialization((e, n, 0), fileformat),
        )
        expected_edge = tuple(
            sorted(
                (
                    normalize_vertex_to_match_mesh_serialization(
                        (w, n, 1.432),
                        fileformat,
                    ),
                    normalize_vertex_to_match_mesh_serialization(
                        (e, n, 0.929),
                        fileformat,
                    ),
                )
            )
        )

        self.assertEqual(
            tuple(sorted(boundary_edges[north_footprint])),
            expected_edge,
        )
