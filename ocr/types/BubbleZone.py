from typing import TypedDict


class BubbleZone(TypedDict, total=False):
    id:              int
    x:               int
    y:               int
    w:               int
    h:               int
    crop:            str
    jp_text:         str
    translated_text: str