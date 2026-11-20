from .._tools import get_wind_components as get_wind_components
from ..http_util import HTTPEndPoint as HTTPEndPoint

class WyomingUpperAir(HTTPEndPoint):
    def __init__(self) -> None: ...
    @classmethod
    def request_data(cls, time, site_id, recalc: bool = False, **kwargs): ...
