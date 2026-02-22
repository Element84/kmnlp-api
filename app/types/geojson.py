"""GeoJSON types conforming to RFC 7946."""

from typing import Any, Literal, Self, cast

from pydantic import BaseModel

type Point = tuple[float, float]  # (longitude, latitude)
type LinearRing = list[Point]  # A closed ring of points


class GeoJSONGeometry(BaseModel):
    """A GeoJSON Geometry."""

    type: Literal[
        "Point",
        "MultiPoint",
        "LineString",
        "MultiLineString",
        "Polygon",
        "MultiPolygon",
    ]
    coordinates: (
        Point  # Point
        | list[Point]  # MultiPoint, LineString
        | list[LinearRing]  # Polygon, MultiLineString
        | list[list[LinearRing]]  # MultiPolygon
    )
    bbox: tuple[float, float, float, float] | None = None

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


class GeoJSONFeature(BaseModel):
    """A GeoJSON Feature."""

    type: Literal["Feature"] = "Feature"
    geometry: GeoJSONGeometry | GeoJSONGeometryCollection | None = None
    properties: dict[str, Any] | None = None
    bbox: tuple[float, float, float, float] | None = None


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
