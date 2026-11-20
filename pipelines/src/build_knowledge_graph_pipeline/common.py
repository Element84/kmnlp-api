"""Common constants for building the knowledge graph."""

from rdflib import Namespace

BASE = Namespace("http://kmnlp-demo.noaa.gov/")
QUDT = Namespace("http://qudt.org/3.1.1/schema/qudt/")
UNIT = Namespace("http://qudt.org/3.1.1/vocab/unit/")
CF = Namespace("http://mmisw.org/ont/cf/parameter/")

DIM_ALT_NAMES = {
    "latitude": "lat",
    "longitude": "lon",
    "radiation_wavelength": "wavelength",
}
