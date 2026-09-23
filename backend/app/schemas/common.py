from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, PlainSerializer


def _format_decimal(value: Decimal) -> str:
    # Plain notation without exponent or trailing zeros, e.g. "0.000048" rather than "4.8E-5".
    text = format(value.normalize(), "f")
    return "0" if text in ("-0", "") else text


# Monetary and credit amounts are sent as strings to avoid float rounding in JSON clients.
DecimalStr = Annotated[Decimal, PlainSerializer(_format_decimal, return_type=str)]


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody
