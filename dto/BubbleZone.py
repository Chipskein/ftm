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
    detection_method:        str    # 'yolo' | 'easyocr' | 'paddleocr'
    detection_rects_total:   int    # total de retângulos detectados antes do filtro
    detection_rects_kept:    int    # retângulos após agrupamento/filtragem
    detection_time_s:        float  # tempo da etapa de detecção
    grouping_time_s:         float  # tempo do agrupamento
    
    extraction_symbols: int
    extraction_time_s: float
    
    translation_symbols: int
    translating_time_s: float
    
    typesetting_characters: int
    typesetting_time_s: float
