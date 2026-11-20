from .cdmremotefeature import CDMRemoteFeature as CDMRemoteFeature
from .dataset import AttributeContainer as AttributeContainer
from _typeshed import Incomplete

log: Incomplete

def reindent_lines(new_leader, source): ...

class CoverageDataset(AttributeContainer):
    cdmrf: Incomplete
    name: str
    lon_lat_domain: Incomplete
    proj_domain: Incomplete
    date_range: Incomplete
    type: Incomplete
    axes: Incomplete
    coord_systems: Incomplete
    grids: Incomplete
    transforms: Incomplete
    def __init__(self, url) -> None: ...
    def load_from_stream(self, header) -> None: ...
