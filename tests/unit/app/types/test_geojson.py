from shapely.geometry import Polygon, mapping

from app.types.geojson import GeoJSONFeatureCollection


def test_from_serialized_geometry():
    polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    geom_dict = mapping(polygon)
    feature_collection = GeoJSONFeatureCollection.from_serialized_geometry(geom_dict)
    assert feature_collection.model_dump(mode="json") == {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
                    ],
                    "bbox": None,
                },
                "properties": None,
                "bbox": None,
            }
        ],
        "bbox": None,
    }
