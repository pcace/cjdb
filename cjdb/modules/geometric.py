import copy
from statistics import mean
from typing import Any, Dict, List, Optional, Union

import numpy as np
from cjio.geom_help import get_normal_newell
from pyproj import CRS, Transformer
from shapely import force_2d
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union
import pyclipper

from cjdb.logger import logger
from cjdb.modules.exceptions import InvalidLodException


# get srid from a CRS string definition
def get_srid(crs):
    if crs:
        proj = CRS.from_string(crs)
        srid = proj.to_epsg()

        return srid


def transform_vertex(vertex, transform):
    new_v = vertex.copy()
    new_v[0] = (new_v[0] * transform["scale"][0]) + transform["translate"][0]
    new_v[1] = (new_v[1] * transform["scale"][1]) + transform["translate"][1]
    new_v[2] = (new_v[2] * transform["scale"][2]) + transform["translate"][2]

    return new_v


def transform_with_rotation(vertex, transform):
    # matrix multiplication as in https://www.cityjson.org/dev/geom-templates/
    homo_vertex = np.array([vertex + [1]]).T
    t_matrix = np.reshape(transform, (4, 4))

    transformed_vertex = np.dot(t_matrix, homo_vertex)
    return list(transformed_vertex.T[0])[:-1]


def reproject_vertex_list(vertices, srid_from, srid_to):
    source_proj = CRS.from_epsg(srid_from)
    target_proj = CRS.from_epsg(srid_to)

    # prepare transformer from crs to crs
    transformer = Transformer.from_crs(source_proj, target_proj, always_xy=True)

    # transform all the coordinates
    reprojected_xyz = transformer.transform(*zip(*vertices))
    reprojected_xyz = [list(i) for i in zip(*reprojected_xyz)]

    return reprojected_xyz


def resolve(lod_level, vertices, inplace=True):
    if inplace:
        resolvable = lod_level
    else:
        resolvable = copy.deepcopy(lod_level)

    for boundary in resolvable["boundaries"]:
        for i, shell in enumerate(boundary):
            if type(shell[0]) is list:
                for j, ring in enumerate(shell):
                    new_ring = []
                    for vertex_id in ring:
                        xyz = vertices[vertex_id]
                        new_ring.append(xyz)
                    shell[j] = new_ring
            else:
                new_shell = []
                for vertex_id in shell:
                    xyz = vertices[vertex_id]
                    new_shell.append(xyz)
                boundary[i] = new_shell

    return resolvable


def resolve_template(lod_level, vertices, geometry_templates, source_target_srid):
    # get anchor point
    vertex_id = lod_level["boundaries"][0]
    anchor = vertices[vertex_id]

    # apply transformation matrix to the template vertices
    template_vertices = [
        transform_with_rotation(v, lod_level["transformationMatrix"])
        for v in geometry_templates["vertices-templates"]
    ]

    # add anchor point to the vertices
    template_vertices = [list(np.array(v) + anchor) for v in template_vertices]

    # reproject vertices if needed
    if source_target_srid:
        template_vertices = reproject_vertex_list(
            template_vertices, *source_target_srid
        )

    # dereference template vertices
    template_id = lod_level["template"]
    template = geometry_templates["templates"][template_id]

    # inplace=False, because the template can be resolved differently
    # for some other cityobject
    resolved_template = resolve(template, template_vertices, inplace=False)
    return resolved_template


def resolve_geometry_vertices(
    geometry, vertices, geometry_templates, source_target_srid
):
    # use ready vertices to resolve coordinate values
    # for the geometry (or geometry template)
    for i, lod_level in enumerate(geometry):
        if lod_level["type"] == "GeometryInstance":
            resolved_template = resolve_template(
                lod_level, vertices, geometry_templates, source_target_srid
            )
            geometry[i] = resolved_template
        else:
            # resolve without geometry template
            resolve(lod_level, vertices)

    return geometry


def get_geometry_with_minimum_lod(
    geometries: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Receives a list of Geometry objects and returns
    the geometry with the minimum LoD."""
    if len(geometries) == 0:
        return None
    elif len(geometries) == 1:
        return geometries[0]
    else:
        try:
            lods = [float(geom["lod"]) for geom in geometries]
        except ValueError:
            raise InvalidLodException()
        index_of_min = lods.index(min(lods))
        return geometries[index_of_min]


def get_flattened_polygons_from_boundaries(
    boundaries: List, polygons: Optional[List] = None
) -> List[Union[Polygon, MultiPolygon]]:
    if polygons is None:
        polygons = []
    if (
        isinstance(boundaries[0], list)
        and len(boundaries[0]) == 3
        and all(isinstance(p, float) for p in boundaries[0])
    ):
        surface_points = []
        for point in boundaries:
            surface_points.append(Point(point[0], point[1], point[2]))
        polygons.append(Polygon(surface_points))
        return polygons
    else:
        for shell in boundaries:
            polygons = get_flattened_polygons_from_boundaries(shell, polygons=polygons)
        return polygons


def is_surface_vertical(normal: np.ndarray) -> bool:
    """
    Given the surface normal as input, if it is (almost)
    perpendicular to the "ground" (xy) normal (0,0,1) then
    the surface can be considered vertical and the function
    will return True, otherwise False.
    We check if the vectors are perpendicular to each other
    by calculating their dot product. If the dot product is
    close to 0 then the vectors are perpendicular.
    """
    dot_prd = 0 * normal[0] + 0 * normal[1] + 1 * normal[2]

    if abs(dot_prd) < 0.6:
        return True
    else:
        return False


def get_ground_surfaces(polygons: List[Polygon]) -> List[Polygon]:

    clipper = pyclipper.Pyclipper()
    toBeClipped = []
    solution = []

    for polygon in polygons:
        polygon_2d = [(x, y) for x, y, z in polygon.exterior.coords]

        # Ensure the polygon is closed
        if polygon_2d[0] != polygon_2d[-1]:
            polygon_2d.append(polygon_2d[0])

        area = pyclipper.Area(polygon_2d)

        # Only add the polygon to the clipper if it has at least 3 points,
        # has an area (remove vertical polygons), and is CCW
        if len(polygon_2d) >= 3 and abs(area) > 0 and pyclipper.Orientation(polygon_2d):
            toBeClipped.append(list(reversed(polygon_2d)))

    if len(toBeClipped) > 0:
        for path in toBeClipped:
            clipper.AddPath(list(reversed(path)), pyclipper.PT_SUBJECT, True)
        solution = clipper.Execute(pyclipper.CT_UNION)
    return [Polygon(polygon) for polygon in solution]

def merge_into_a_multipolygon(
    ground_surfaces: List[Union[Polygon, MultiPolygon]]
) -> MultiPolygon:
    valid_ground_surfaces = []
    for gs in ground_surfaces:
        if not gs.is_valid:
            gs = gs.buffer(0)
        if gs.is_valid:
            valid_ground_surfaces.append(gs)
        else:
            print(f"Invalid geometry at {gs.wkt}")
    polygon = unary_union(force_2d(valid_ground_surfaces))
    if isinstance(polygon, MultiPolygon):
        return polygon
    else:
        return MultiPolygon([polygon])

def get_ground_geometry(geometries: List[Dict[str, Any]], obj_id: str) -> MultiPolygon:
    """Receives a list of transformed boundary coordinates
    of the city object
    and extracts only the ground surface.
    If there is an LoD 0, then all the available surfaces
    are merged and returned.
    If not, then only the non-vertical surfaces with the lowest
    height are merged and returned.
    """
    geometry = get_geometry_with_minimum_lod(geometries)

    if geometry is None:
        logger.warning(f"No geometry for object ID=({obj_id}) ")
        return None

    if geometry["type"] == "MultiPoint" or geometry["type"] == "MultiLineString":
        logger.warning(
            f"""MultiPoint or MultiLineString type has no ground geometry,
            object ID=({obj_id})"""
        )
        # TODO return convex hull of the points as ground geometry.
        return None

    if float(geometry["lod"]) < 1:
        flattented_polygons = get_flattened_polygons_from_boundaries(
            geometry["boundaries"]
        )
        return merge_into_a_multipolygon(flattented_polygons)
    else:
        # TODO: check if there are surface types available
        # to choose the ground surfaces
        surfaces = get_flattened_polygons_from_boundaries(geometry["boundaries"])
        ground_surfaces = get_ground_surfaces(surfaces)
        return merge_into_a_multipolygon(ground_surfaces)
