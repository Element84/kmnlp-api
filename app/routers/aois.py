from typing import Annotated

from e84_geoai_common.tracing import observe
from fastapi import APIRouter, Depends, HTTPException, status

from app.dao.aois_dao import AOIsDAO
from app.dependencies import get_current_user
from app.providers import aois_dao_provider
from app.transformers.aois import domain_to_output_aoi_list_response
from app.transformers.cursor_encoding import (
    CursorValueNotConvertibleToFloatError,
    InvalidCursorError,
    MalformedCursorError,
    UnexpectedCursorKeyError,
    decode_datetime_cursor,
)
from app.types.geojson import GeoJSONFeatureCollection
from app.types.input import AOI as InputAOI  # noqa: N811
from app.types.output import AOI, AOIListResponse, UUIDDict

router = APIRouter(
    prefix="/aois",
    tags=["AOIs"],
)


@router.post("", status_code=status.HTTP_201_CREATED)
@observe(name="api.create_aoi")
async def create_area_of_interest(
    aoi: InputAOI,
    user_id: Annotated[str, Depends(get_current_user)],
    aois_dao: Annotated[AOIsDAO, Depends(aois_dao_provider)],
) -> UUIDDict:
    """Create an AOI in the backend and return its UUID."""
    result = aois_dao.create_aoi(user_id, aoi.name, aoi.geometry)
    return UUIDDict(uuid=str(result.uuid))


@router.get("")
@observe(name="api.list_aois")
async def list_aois(
    user_id: Annotated[str, Depends(get_current_user)],
    aois_dao: Annotated[AOIsDAO, Depends(aois_dao_provider)],
    after_cursor: str | None = None,
) -> AOIListResponse:
    """List AOI summaries for user."""
    decoded_cursor = None

    if after_cursor:
        try:
            decoded_cursor = decode_datetime_cursor(after_cursor)
        except (
            InvalidCursorError,
            MalformedCursorError,
            UnexpectedCursorKeyError,
            CursorValueNotConvertibleToFloatError,
        ) as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Malformed after_cursor"
            ) from e

    result = aois_dao.list_aois(
        user_id, created_at_after=decoded_cursor.value if decoded_cursor else None
    )

    return domain_to_output_aoi_list_response(result)


@router.get("/{aoi_id}")
@observe(name="api.get_aoi")
async def get_aoi(
    aoi_id: str,
    requesting_user_id: Annotated[str, Depends(get_current_user)],
    aois_dao: Annotated[AOIsDAO, Depends(aois_dao_provider)],
) -> AOI:
    """Get single AOI for user."""
    aoi = aois_dao.get_aoi(aoi_id, requesting_user_id)
    if aoi:
        geojson = (
            aoi.geojson
            if isinstance(aoi.geojson, GeoJSONFeatureCollection)
            else aoi.geojson.as_feature_collection()
        )
        return AOI(uuid=str(aoi.uuid), name=aoi.name, geojson=geojson, created_at=aoi.created_at)

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="AOI not found")
