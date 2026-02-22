from app.transformers.cursor_encoding import encode_datetime_cursor
from app.types import internal, output


def domain_to_output_aoi_summary(aoi: internal.AOISummary) -> output.AOISummary:
    """Convert a domain AOI Summary to an output AOI Summary."""
    return output.AOISummary(uuid=str(aoi.uuid), name=aoi.name, created_at=aoi.created_at)


def domain_to_output_aoi_list_response(
    response: internal.AOIListResponse,
) -> output.AOIListResponse:
    """Convert a domain AOI list response to an output one."""
    return output.AOIListResponse(
        aois=[domain_to_output_aoi_summary(aoi) for aoi in response.aois],
        pagination_info=output.PaginationInfo(
            total_count=response.total_count,
            next_cursor=encode_datetime_cursor(response.cursor) if response.cursor else None,
        ),
    )
