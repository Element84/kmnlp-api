from .._tools import get_wind_components as get_wind_components
from ..http_util import HTTPEndPoint as HTTPEndPoint, HTTPError as HTTPError

class IGRAUpperAir(HTTPEndPoint):
    suffix: str
    begin_date: str
    end_date: str
    site_id: str
    folder: str
    def __init__(self) -> None: ...
    @classmethod
    def request_data(cls, time, site_id, derived: bool = False): ...
