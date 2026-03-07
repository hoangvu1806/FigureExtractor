"""
CLI entry point for figure extraction.

Usage:
    python -m figure_extract.cli <pdf_path> <output_dir> [options]
"""

import argparse
from pathlib import Path

from .extractor import FigureExtractor, _DEFAULT_CONF, _DEFAULT_DPI, _DEFAULT_IOU, _DEFAULT_WEIGHTS


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="figure_extract",
        description="Detect and crop figures from a PDF using YOLOv8n.",
    )
    parser.add_argument("pdf", type=Path, help="Path to the input PDF file.")
    parser.add_argument("output", type=Path, help="Output directory for crops and metadata.json.")
    parser.add_argument(
        "--model",
        type=Path,
        default=_DEFAULT_WEIGHTS,
        help="Path to YOLOv8 weights (.pt). Default: bundled figure_detect.pt",
    )
    parser.add_argument("--conf", type=float, default=_DEFAULT_CONF, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=_DEFAULT_IOU, help="NMS IoU threshold.")
    parser.add_argument("--dpi", type=int, default=_DEFAULT_DPI, help="PDF page rendering DPI.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    extractor = FigureExtractor(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        dpi=args.dpi,
    )

    result = extractor.extract(pdf_path=args.pdf, output_dir=args.output)

    print(f"Extracted {result.total_figures} figures from {result.source_pdf}")
    print(f"Output    : {args.output}")
    print(f"Metadata  : {args.output / 'metadata.json'}")


if __name__ == "__main__":
    main()
