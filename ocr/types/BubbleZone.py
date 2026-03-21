from typing import TypedDict


class BubbleZone(TypedDict, total=False):
    id:              int
    x:               int
    y:               int
    w:               int
    h:               int
    crop:            str
    jp_text:         str
    en_text:         str
    translated_text: str
    polygon:         list[list[int]]  # [[x, y], ...] in crop-local coordinates