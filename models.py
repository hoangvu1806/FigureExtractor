from dataclasses import dataclass, field

@dataclass(frozen=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def to_normalized(self, page_width: int, page_height: int) -> dict:
        return {
            "cx": round((self.x1 + self.x2) / 2.0 / page_width, 6),
            "cy": round((self.y1 + self.y2) / 2.0 / page_height, 6),
            "w": round(self.width / page_width, 6),
            "h": round(self.height / page_height, 6),
        }

    def to_dict(self) -> dict:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


@dataclass(frozen=True)
class FigureMetadata:
    source_pdf: str
    page_number: int           # 1-indexed
    figure_index_in_page: int  # 1-indexed, reading order within page
    figure_index_global: int   # 1-indexed across the full document
    confidence: float
    bbox: BoundingBox
    page_width_px: int
    page_height_px: int
    crop_filename: str

    def to_dict(self) -> dict:
        return {
            "source_pdf": self.source_pdf,
            "page_number": self.page_number,
            "figure_index_in_page": self.figure_index_in_page,
            "figure_index_global": self.figure_index_global,
            "confidence": self.confidence,
            "bbox_pixels": self.bbox.to_dict(),
            "bbox_normalized": self.bbox.to_normalized(self.page_width_px, self.page_height_px),
            "page_width_px": self.page_width_px,
            "page_height_px": self.page_height_px,
            "crop_filename": self.crop_filename,
        }


@dataclass
class ExtractionResult:
    source_pdf: str
    total_pages: int
    total_figures: int
    dpi: int
    conf_threshold: float
    iou_threshold: float
    figures: list[FigureMetadata] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source_pdf": self.source_pdf,
            "total_pages": self.total_pages,
            "total_figures": self.total_figures,
            "dpi": self.dpi,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "figures": [f.to_dict() for f in self.figures],
        }
