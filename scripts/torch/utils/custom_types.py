from enum import Enum


class FinetuningMode(Enum):
    FULL = "full"
    DECODER = "decoder"
    FULL_RESETLASTLAYER = "full_rll"
