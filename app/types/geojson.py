"""GeoJSON types conforming to RFC 7946."""

from typing import Any, Literal, Self, cast

from pydantic import BaseModel

type Point = tuple[float, float]  # (longitude, latitude)
type LinearRing = list[Point]  # A closed ring of points
type GeomType = Literal[
    "Point",
    "MultiPoint",
    "LineString",
    "MultiLineString",
    "Polygon",
    "MultiPolygon",
]


class GeoJSONGeometry(BaseModel):
    """A GeoJSON Geometry."""

    type: GeomType
    coordinates: (
        Point  # Point
        | list[Point]  # MultiPoint, LineString
        | list[LinearRing]  # Polygon, MultiLineString
        | list[list[LinearRing]]  # MultiPolygon
    )
    bbox: tuple[float, float, float, float] | None = None

    def as_feature_collection(self) -> "GeoJSONFeatureCollection":
        """Convert the geometry to a GeoJSON FeatureCollection containing a single feature."""
        feature = GeoJSONFeature(type="Feature", geometry=self)
        return GeoJSONFeatureCollection(features=[feature])

    def __repr_args__(self) -> list[tuple[str | None, Any]]:
        match self.type:
            case "Point":
                coords = cast("Point", self.coordinates)
                coord_repr = f"({coords[0]}, {coords[1]})"
            case "MultiPoint" | "LineString":
                coords = cast("list[Point]", self.coordinates)
                coord_repr = f"{len(coords)} points"
            case "MultiLineString":
                coords = cast("list[LinearRing]", self.coordinates)
                coord_repr = f"{len(coords)} lines"
            case "Polygon":
                coords = cast("list[LinearRing]", self.coordinates)
                num_rings = len(coords)
                num_points = sum(len(ring) for ring in coords)
                coord_repr = f"{num_rings} rings with {num_points} points total"
            case "MultiPolygon":
                coords = cast("list[list[LinearRing]]", self.coordinates)
                coord_repr = f"{len(coords)} polygons"
        return [("type", self.type), ("coordinates", f"<{coord_repr}>")]


class GeoJSONGeometryCollection(BaseModel):
    """A GeoJSON GeometryCollection."""

    type: Literal["GeometryCollection"] = "GeometryCollection"
    geometries: list["GeoJSONGeometry | GeoJSONGeometryCollection"]
    bbox: tuple[float, float, float, float] | None = None

    @property
    def geom_types(self) -> list[GeomType]:
        """Return a list of geometry types contained in the collection."""
        types: list[GeomType] = []
        for geom in self.geometries:
            if isinstance(geom, GeoJSONGeometry):
                types.append(geom.type)
            else:
                types.extend(geom.geom_types)
        return types


class GeoJSONFeature(BaseModel):
    """A GeoJSON Feature."""

    type: Literal["Feature"] = "Feature"
    geometry: GeoJSONGeometry | GeoJSONGeometryCollection | None = None
    properties: dict[str, Any] | None = None
    bbox: tuple[float, float, float, float] | None = None

    @property
    def geom_types(self) -> list[GeomType]:
        """Return a list of geometry types contained in the feature's geometry."""
        match self.geometry:
            case GeoJSONGeometry():
                return [self.geometry.type]
            case GeoJSONGeometryCollection():
                return self.geometry.geom_types
            case None:
                return []


class GeoJSONFeatureCollection(BaseModel):
    """A GeoJSON FeatureCollection."""

    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[GeoJSONFeature]
    bbox: tuple[float, float, float, float] | None = None

    @classmethod
    def from_serialized_geometry(cls, geometry: dict[str, Any]) -> Self:
        """Create GeoJSONFeatureCollection from serialized geometry dict."""
        feature = GeoJSONFeature(type="Feature", geometry=GeoJSONGeometry.model_validate(geometry))
        return cls(features=[feature])

    @property
    def geom_types(self) -> list[GeomType]:
        """Return a list of geometry types contained in the feature collection."""
        types: list[GeomType] = []
        for feature in self.features:
            types.extend(feature.geom_types)
        return types
