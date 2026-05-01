from typing import TypedDict

class BubbleZone(TypedDict):
    id:              int
    x:               int
    y:               int
    w:               int
    h:               int
    crop:            str
    jp_text:         str
    translated_text: str
    more_context:    str

    area:            int
    detection_method:        str
    detection_rects_total:   int
    detection_rects_kept:    int
    detection_time_s:        float
    grouping_time_s:         float
    
    extraction_symbols: int
    extraction_time_s: float
    
    translation_symbols: int
    translating_time_s: float
    
    typesetting_characters: int
    typesetting_time_s: float
