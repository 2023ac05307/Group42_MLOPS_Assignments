# app/schemas.py
from pydantic import BaseModel, Field, validator, confloat, conint
from typing import ClassVar, Set


class HousingFeatures(BaseModel):
    # Numbers roughly match the California Housing dataset’s plausible ranges
    MedInc: confloat(gt=0, lt=20) = Field(
        ..., description="Median income (10k USD units)"
    )
    HouseAge: confloat(ge=0, le=100) = Field(
        ..., description="Median house age (years)"
    )
    AveRooms: confloat(gt=0, le=50) = Field(
        ..., description="Average rooms per household"
    )
    AveBedrms: confloat(gt=0, le=15) = Field(
        ..., description="Average bedrooms per household"
    )
    Population: conint(ge=1, le=50000) = Field(
        ..., description="Block group population"
    )
    AveOccup: confloat(gt=0, le=50) = Field(
        ..., description="Average occupants per household"
    )
    Latitude: confloat(ge=32.0, le=42.5) = Field(
        ..., description="Latitude in California"
    )
    Longitude: confloat(ge=-125.0, le=-114.0) = Field(
        ..., description="Longitude in California"
    )
    OceanProximity: str = Field(
        ..., description="One of: INLAND, NEAR OCEAN, <1H OCEAN, NEAR BAY, ISLAND"
    )

    _allowed_ops: ClassVar[Set[str]] = {
        "INLAND",
        "NEAR OCEAN",
        "<1H OCEAN",
        "NEAR BAY",
        "ISLAND",
    }

    @validator("OceanProximity")
    def normalize_ocean_proximity(cls, v: str) -> str:
        s = v.strip().upper().replace("_", " ")
        # normalize a couple of common typos
        if s in {"< 1H OCEAN", "LESS THAN 1H OCEAN", "LT 1H OCEAN"}:
            s = "<1H OCEAN"
        if s not in cls._allowed_ops:
            raise ValueError(
                f"OceanProximity must be one of {sorted(cls._allowed_ops)}"
            )
        return s
