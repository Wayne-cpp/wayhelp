from fastapi import APIRouter, Request

from app.chains.extract_chain import run_extraction
from app.errors import MessageTooLongError
from app.schemas import AfterSaleExtraction, ExtractRequest

router = APIRouter()


@router.post("/v1/extract", response_model=AfterSaleExtraction)
async def extract(body: ExtractRequest, request: Request) -> AfterSaleExtraction:
    settings = request.app.state.settings
    if len(body.text) > settings.max_message_chars:
        raise MessageTooLongError("text exceeds MAX_MESSAGE_CHARS")
    return await run_extraction(
        request.app.state.model,
        settings.structured_output_method,
        body.text,
        settings.max_input_tokens,
    )
