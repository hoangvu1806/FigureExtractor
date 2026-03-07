"""
FigureExtractor: detects and crops figures from PDF pages using a YOLOv8n model.

Responsibilities:
- Render PDF pages to images at a configurable DPI.
- Run the YOLO detector and apply NMS.
- Sort detected boxes in reading order.
- Crop and persist figure images.
- Return a structured ExtractionResult with full metadata.
"""

import json
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from ultralytics import YOLO

from .models import BoundingBox, ExtractionResult, FigureMetadata

_DEFAULT_WEIGHTS = Path(__file__).parent / "weights" / "figure_detect.pt"
_DEFAULT_DPI = 300
_DEFAULT_CONF = 0.35
_DEFAULT_IOU = 0.45
_INFERENCE_IMGSZ = 640
_READING_ORDER_BAND_PX = 50


def _render_page(page: fitz.Page, dpi: int) -> Image.Image:
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _sort_reading_order(
    boxes: list[tuple[float, float, float, float, float]],
) -> list[tuple[float, float, float, float, float]]:
    """
    Groups boxes into horizontal bands then sorts left-to-right within each band.
    Produces natural reading order for document layouts.
    """
    if not boxes:
        return []

    sorted_by_y = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2.0)
    bands: list[list[tuple]] = []

    for box in sorted_by_y:
        cy = (box[1] + box[3]) / 2.0
        for band in bands:
            band_cy = sum((b[1] + b[3]) / 2.0 for b in band) / len(band)
            if abs(cy - band_cy) < _READING_ORDER_BAND_PX:
                band.append(box)
                break
        else:
            bands.append([box])

    result: list[tuple] = []
    for band in bands:
        result.extend(sorted(band, key=lambda b: b[0]))
    return result


class FigureExtractor:
    """
    Stateless-per-call extractor. The YOLO model is loaded once at construction
    and reused across multiple extract() calls for efficiency.
    """

    def __init__(
        self,
        model_path: Path = _DEFAULT_WEIGHTS,
        conf_threshold: float = _DEFAULT_CONF,
        iou_threshold: float = _DEFAULT_IOU,
        dpi: int = _DEFAULT_DPI,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")

        self._model = YOLO(str(model_path))
        self._conf = conf_threshold
        self._iou = iou_threshold
        self._dpi = dpi

    def extract(self, pdf_path: Path, output_dir: Path) -> ExtractionResult:
        """
        Runs the full pipeline on a single PDF.

        Args:
            pdf_path:   Absolute path to the input PDF.
            output_dir: Directory where cropped figures and metadata.json are saved.

        Returns:
            ExtractionResult containing all figure metadata.
        """
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        pdf_name = pdf_path.name

        figures: list[FigureMetadata] = []
        global_index = 0

        for page_idx in range(total_pages):
            page_image = _render_page(doc[page_idx], self._dpi)
            page_w, page_h = page_image.size

            raw_boxes = self._detect(page_image)
            sorted_boxes = _sort_reading_order(raw_boxes)

            for fig_idx, (x1, y1, x2, y2, conf) in enumerate(sorted_boxes):
                global_index += 1
                bbox = BoundingBox(
                    x1=max(0, int(x1)),
                    y1=max(0, int(y1)),
                    x2=min(page_w, int(x2)),
                    y2=min(page_h, int(y2)),
                )
                crop_filename = f"page_{page_idx + 1:03d}_fig_{fig_idx + 1:02d}.png"
                page_image.crop((bbox.x1, bbox.y1, bbox.x2, bbox.y2)).save(
                    figures_dir / crop_filename, format="PNG"
                )

                figures.append(
                    FigureMetadata(
                        source_pdf=pdf_name,
                        page_number=page_idx + 1,
                        figure_index_in_page=fig_idx + 1,
                        figure_index_global=global_index,
                        confidence=round(conf, 4),
                        bbox=bbox,
                        page_width_px=page_w,
                        page_height_px=page_h,
                        crop_filename=crop_filename,
                    )
                )

        doc.close()

        result = ExtractionResult(
            source_pdf=pdf_name,
            total_pages=total_pages,
            total_figures=global_index,
            dpi=self._dpi,
            conf_threshold=self._conf,
            iou_threshold=self._iou,
            figures=figures,
        )

        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return result

    def _detect(
        self, image: Image.Image
    ) -> list[tuple[float, float, float, float, float]]:
        results = self._model.predict(
            source=np.array(image),
            imgsz=_INFERENCE_IMGSZ,
            conf=self._conf,
            iou=self._iou,
            verbose=False,
        )
        boxes: list[tuple[float, float, float, float, float]] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes.append((x1, y1, x2, y2, float(box.conf[0])))
        return boxes
