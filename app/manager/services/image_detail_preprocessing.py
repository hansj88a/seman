"""상품 상세 이미지 전처리: 배경 제거 후 구간 병합 · 여러 장 세로 합성."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

try:
    from PIL import Image, ImageChops, ImageOps
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Pillow가 필요합니다. `pip install Pillow` 후 다시 실행하세요."
    ) from exc

# 이 파일: app/manager/services/ → 저장소 루트는 parents[3]
_REPO_ROOT: Path = Path(__file__).resolve().parents[3]
_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
)


def default_data_image_dir() -> Path:
    """기본 입력 디렉터리: app/data/image"""
    return _REPO_ROOT / "app" / "data" / "image"


def default_data_image_output_dir() -> Path:
    """기본 출력 디렉터리: app/data/image/output"""
    return default_data_image_dir() / "output"


def list_images_in_dir(
    input_dir: Path | None = None,
    *,
    include_s3_uri: str | None = None,
    s3_cache_dir: Path | None = None,
) -> list[Path]:
    """
     디렉터리 바로 아래의 이미지 파일만 수집한 뒤, 옵션으로 S3 prefix 이미지를
    내려받은 경로와 합쳐 파일명 기준 정렬해 반환(하위 폴더는 순회하지 않음).

    S3를 섞을 때: `include_s3_uri="s3://bucket/detail/"` 를 주면
    `list_objects_v2`로 이미지 키를 모은 뒤 `s3_cache_dir`(기본 `input_dir/_s3_include_cache`)에
    저장하고, 그 로컬 경로를 로컬 파일 목록과 합칩니다.

    코드에서만 쓸 때 예:

        from app.manager.services.s3_client import download_s3_prefix_to_directory, parse_s3_uri
        paths = list_images_in_dir(Path("./local"))
        b, p = parse_s3_uri("s3://my-bucket/detail/")
        paths += download_s3_prefix_to_directory(b, p, Path("./.s3_cache"))
        paths = sorted(paths, key=lambda x: x.name.lower())
    """
    root = (input_dir or default_data_image_dir()).resolve()
    files: list[Path] = []
    if root.is_dir():
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS:
                files.append(p.resolve())
    if include_s3_uri:
        from app.manager.services.s3_client import (
            download_s3_prefix_to_directory,
            parse_s3_uri,
        )

        bucket, prefix = parse_s3_uri(include_s3_uri)
        cache = (s3_cache_dir or (root / "_s3_include_cache")).resolve()
        cache.mkdir(parents=True, exist_ok=True)
        files.extend(download_s3_prefix_to_directory(bucket, prefix, cache))
    return sorted(files, key=lambda x: x.name.lower())


@dataclass(slots=True)
class DetailPreprocessConfig:
    """상세 이미지 전처리 옵션."""

    vertical_margin_px: int = 5
    white_threshold: int = 245
    min_segment_height: int = 8
    min_content_pixels_per_row: int = 3
    horizontal_margin_px: int = 0
    """여러 장 합칠 때 이미지 사이 세로 간격(px)."""
    between_images_margin_px: int = 5
    """
    True면 가장자리 픽셀의 중앙값을 배경색으로 보고, 그 색과 비슷한 픽셀은 공백으로 취급.
    (흰색이 아닌 단색 배경에도 대응)
    """
    use_edge_background: bool = True
    background_border_px: int = 2
    """배경색과의 채널별 최대 허용 차이. 이보다 크면 내용 픽셀."""
    background_tolerance: int = 18
    """True면 단계별 진행을 콘솔에 출력."""
    verbose: bool = False


def _log(cfg: DetailPreprocessConfig, msg: str) -> None:
    if cfg.verbose:
        print(f"[detail_preprocess] {msg}", flush=True)


def _median_int(values: list[int]) -> int:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 255
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) // 2


def _sample_edge_rgb_pixels(image: Image.Image, border_px: int) -> list[tuple[int, int, int]]:
    w, h = image.size
    b = max(1, min(border_px, w // 2, h // 2))
    px = image.load()
    assert px is not None
    out: list[tuple[int, int, int]] = []
    for y in range(b):
        for x in range(w):
            out.append(px[x, y])
            out.append(px[x, h - 1 - y])
    for x in range(b):
        for y in range(h):
            out.append(px[x, y])
            out.append(px[w - 1 - x, y])
    return out


def _median_edge_background(image: Image.Image, border_px: int) -> tuple[int, int, int]:
    samples = _sample_edge_rgb_pixels(image, border_px)
    rs = [p[0] for p in samples]
    gs = [p[1] for p in samples]
    bs_ = [p[2] for p in samples]
    return _median_int(rs), _median_int(gs), _median_int(bs_)


def _content_mask_rgb(
    image: Image.Image,
    bg: tuple[int, int, int],
    tolerance: int,
) -> Image.Image:
    """배경과 채널 차이가 tolerance를 넘으면 내용(255)."""
    r0, g0, b0 = bg
    r, g, b = image.split()
    mr = r.point(lambda v, _r0=r0, t=tolerance: 255 if abs(int(v) - _r0) > t else 0)
    mg = g.point(lambda v, _g0=g0, t=tolerance: 255 if abs(int(v) - _g0) > t else 0)
    mb = b.point(lambda v, _b0=b0, t=tolerance: 255 if abs(int(v) - _b0) > t else 0)
    return ImageChops.lighter(ImageChops.lighter(mr, mg), mb)


def _content_mask_luminance(image: Image.Image, white_threshold: int) -> Image.Image:
    """기존 방식: 명도가 white_threshold 미만이면 내용."""
    return image.convert("L").point(lambda p: 255 if p < white_threshold else 0, mode="L")


def _build_content_mask(image: Image.Image, cfg: DetailPreprocessConfig) -> Image.Image:
    if cfg.use_edge_background:
        bg = _median_edge_background(image, cfg.background_border_px)
        return _content_mask_rgb(image, bg, cfg.background_tolerance)
    return _content_mask_luminance(image, cfg.white_threshold)


def _row_content_counts(mask: Image.Image) -> list[int]:
    w, h = mask.size
    data = mask.tobytes()
    return [data[y * w : (y + 1) * w].count(255) for y in range(h)]


def _row_segments(
    row_content_counts: list[int],
    min_content_pixels_per_row: int,
    min_segment_height: int,
) -> list[tuple[int, int]]:
    """내용이 있는 행 구간(start, end). end는 exclusive."""
    meaningful = [c >= min_content_pixels_per_row for c in row_content_counts]
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for idx, has_content in enumerate(meaningful):
        if has_content and start is None:
            start = idx
            continue
        if not has_content and start is not None:
            if idx - start >= min_segment_height:
                segments.append((start, idx))
            start = None
    if start is not None and len(meaningful) - start >= min_segment_height:
        segments.append((start, len(meaningful)))
    return segments


def _crop_segment_horizontally(
    image_rgb: Image.Image,
    content_mask: Image.Image,
    start_row: int,
    end_row: int,
    horizontal_margin_px: int,
) -> Image.Image:
    segment = image_rgb.crop((0, start_row, image_rgb.width, end_row))
    segment_mask = content_mask.crop((0, start_row, image_rgb.width, end_row))
    bbox = segment_mask.getbbox()
    if bbox is None:
        return segment
    left = max(bbox[0] - horizontal_margin_px, 0)
    right = min(bbox[2] + horizontal_margin_px, image_rgb.width)
    return segment.crop((left, 0, right, segment.height))


def _merge_internal_segments(
    image: Image.Image,
    cfg: DetailPreprocessConfig,
    *,
    source_label: str = "",
) -> Image.Image:
    """한 장 안에서 세로 공백 제거 후 세로로 붙인 단일 이미지."""
    prefix = f"{source_label}: " if source_label else ""
    iw, ih = image.size
    _log(cfg, f"{prefix}[1/5] 원본 크기 {iw}x{ih}px")

    if cfg.use_edge_background:
        bg = _median_edge_background(image, cfg.background_border_px)
        _log(
            cfg,
            f"{prefix}[2/5] 배경색 추정(가장자리 중앙값) RGB={bg}, "
            f"tolerance={cfg.background_tolerance}, border={cfg.background_border_px}px",
        )
    else:
        _log(
            cfg,
            f"{prefix}[2/5] 마스크 모드: 명도 임계값 white_threshold={cfg.white_threshold}",
        )

    content_mask = _build_content_mask(image, cfg)
    row_counts = _row_content_counts(content_mask)
    meaningful_rows = sum(1 for c in row_counts if c >= cfg.min_content_pixels_per_row)
    _log(
        cfg,
        f"{prefix}[3/5] 내용 마스크 — 내용으로 잡힌 행 수 {meaningful_rows}/{ih}, "
        f"min_pixels_per_row={cfg.min_content_pixels_per_row}, "
        f"min_segment_height={cfg.min_segment_height}",
    )

    segments = _row_segments(
        row_counts,
        cfg.min_content_pixels_per_row,
        cfg.min_segment_height,
    )
    if not segments:
        _log(
            cfg,
            f"{prefix}[4/5] 세로 구간 없음 → 이 파일은 자르지 않고 원본 그대로 사용",
        )
        _log(cfg, f"{prefix}[5/5] 결과 크기 {iw}x{ih}px (원본)")
        return image

    _log(cfg, f"{prefix}[4/5] 세로 내용 구간 {len(segments)}개")
    max_detail = 8
    for seg_i, (start, end) in enumerate(segments):
        h_seg = end - start
        if seg_i < max_detail:
            _log(cfg, f"        구간 {seg_i + 1}: 행 y=[{start}, {end}) 높이={h_seg}px")
        elif seg_i == max_detail:
            _log(cfg, f"        … 이하 {len(segments) - max_detail}개 구간 생략")

    parts: list[Image.Image] = []
    for start, end in segments:
        cropped = _crop_segment_horizontally(
            image,
            content_mask,
            start,
            end,
            cfg.horizontal_margin_px,
        )
        padded = ImageOps.expand(
            cropped,
            border=(0, cfg.vertical_margin_px, 0, cfg.vertical_margin_px),
            fill="white",
        )
        parts.append(padded)

    w = max(p.width for p in parts)
    h = sum(p.height for p in parts)
    _log(
        cfg,
        f"{prefix}[5/5] 구간 병합 완료 → 캔버스 {w}x{h}px (블록당 상하 여백 {cfg.vertical_margin_px}px)",
    )
    canvas = Image.new("RGB", (w, h), "white")
    y = 0
    for p in parts:
        canvas.paste(p, ((w - p.width) // 2, y))
        y += p.height
    return canvas


def preprocess_detail_images(
    input_image_paths: Sequence[str | Path],
    output_image_path: str | Path,
    config: DetailPreprocessConfig | None = None,
) -> Path:
    """
    여러 상세 이미지를 각각 전처리한 뒤 세로로 하나로 합쳐 저장.

    각 파일에 대해: 배경/공백 제거 → 구간별 여백 적용 후 세로 병합,
    그 다음 모든 파일 결과를 `between_images_margin_px` 간격으로 세로 합침.
    """
    cfg = config or DetailPreprocessConfig()
    paths = [Path(p) for p in input_image_paths]
    if not paths:
        raise ValueError("입력 이미지 경로가 비어 있습니다.")

    out = Path(output_image_path)
    _log(
        cfg,
        f"시작 — 입력 {len(paths)}개 → 출력 {out.resolve()} | "
        f"edge_bg={cfg.use_edge_background} between={cfg.between_images_margin_px}px",
    )

    blocks: list[Image.Image] = []
    for idx, p in enumerate(paths):
        label = f"[{idx + 1}/{len(paths)}] {p.name}"
        _log(cfg, f"--- 파일 처리 {label} ---")
        with Image.open(p) as im:
            rgb = im.convert("RGB")
            blocks.append(_merge_internal_segments(rgb.copy(), cfg, source_label=label))

    sep = max(0, cfg.between_images_margin_px)
    w = max(b.width for b in blocks)
    h = sum(b.height for b in blocks) + sep * max(0, len(blocks) - 1)
    _log(
        cfg,
        f"최종 합성 — 캔버스 {w}x{h}px (이미지 간 간격 {sep}px), 블록 수 {len(blocks)}",
    )
    canvas = Image.new("RGB", (w, h), "white")
    y = 0
    for i, b in enumerate(blocks):
        if i > 0:
            y += sep
        canvas.paste(b, ((w - b.width) // 2, y))
        y += b.height

    out.parent.mkdir(parents=True, exist_ok=True)
    _log(cfg, f"저장 중… {out.resolve()}")
    canvas.save(out)
    _log(cfg, "완료")
    return out


def preprocess_detail_image(
    input_image_path: str | Path,
    output_image_path: str | Path,
    config: DetailPreprocessConfig | None = None,
) -> Path:
    """
    단일 긴 상품 상세 이미지를 전처리해 저장.
    (내부적으로 `preprocess_detail_images` 한 장 분기와 동일 로직)
    """
    return preprocess_detail_images([input_image_path], output_image_path, config)


if __name__ == "__main__":
    import argparse

    _default_in = default_data_image_dir()
    _default_out_dir = default_data_image_output_dir()

    parser = argparse.ArgumentParser(
        description="상품 상세 이미지: 배경·공백 제거 후 세로 병합 (여러 장 입력 가능)"
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        metavar="INPUT",
        help=(
            "입력 이미지 경로 (복수 가능). 생략 시 --input-dir(기본 app/data/image) "
            "바로 아래의 이미지 파일을 모두 사용"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            f"출력 이미지 파일 경로. 생략 시 {_default_out_dir / 'merged.jpg'} "
            "(디렉터리를 주면 그 아래 merged.jpg)"
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=f"입력 디렉터리 (inputs 생략 시 이 경로에서 수집, 기본: {_default_in})",
    )
    parser.add_argument(
        "--include-s3",
        default=None,
        metavar="S3_URI",
        help="로컬 목록과 합칠 S3 prefix (예: s3://bucket/detail/). list_objects_v2 후 캐시에 저장",
    )
    parser.add_argument(
        "--s3-cache-dir",
        type=Path,
        default=None,
        help="S3 이미지 다운로드 폴더 (미지정 시 input-dir/_s3_include_cache)",
    )
    parser.add_argument("--vertical-margin", type=int, default=5, help="블록 내 상하 여백(px)")
    parser.add_argument(
        "--between-margin",
        type=int,
        default=5,
        help="여러 입력 파일 사이 세로 간격(px)",
    )
    parser.add_argument("--white-threshold", type=int, default=245, help="use_edge_background=off 시 명도 임계값")
    parser.add_argument("--min-segment-height", type=int, default=8, help="최소 구간 높이(px)")
    parser.add_argument(
        "--min-content-pixels-per-row",
        type=int,
        default=3,
        help="내용 행으로 볼 최소 마스크 픽셀 수",
    )
    parser.add_argument("--horizontal-margin", type=int, default=0, help="좌우 여백(px)")
    parser.add_argument(
        "--no-edge-background",
        action="store_true",
        help="가장자리 배경 추정 대신 밝기 임계값(흰 배경)만 사용",
    )
    parser.add_argument("--background-tolerance", type=int, default=18, help="배경색과의 채널 차 허용치")
    parser.add_argument("--background-border", type=int, default=2, help="배경 샘플링 테두리 두께(px)")
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="단계별 콘솔 로그 끄기",
    )
    args = parser.parse_args()

    input_dir = (args.input_dir or _default_in).resolve()
    if args.inputs:
        input_paths = [Path(p).resolve() for p in args.inputs]
        if args.include_s3:
            from app.manager.services.s3_client import (
                download_s3_prefix_to_directory,
                parse_s3_uri,
            )

            sb, sp = parse_s3_uri(args.include_s3)
            cache = (args.s3_cache_dir or (input_dir / "_s3_include_cache")).resolve()
            cache.mkdir(parents=True, exist_ok=True)
            input_paths.extend(download_s3_prefix_to_directory(sb, sp, cache))
            input_paths = sorted(input_paths, key=lambda p: p.name.lower())
    else:
        input_paths = list_images_in_dir(
            input_dir,
            include_s3_uri=args.include_s3,
            s3_cache_dir=args.s3_cache_dir,
        )
        if not input_paths:
            parser.error(
                f"입력 이미지가 없습니다. 경로 확인: {input_dir} "
                f"(확장자: {', '.join(sorted(_IMAGE_EXTENSIONS))}), "
                f"또는 --include-s3 URI 확인"
            )

    if args.output is None:
        _default_out_dir.mkdir(parents=True, exist_ok=True)
        output_path: Path = _default_out_dir / "merged.jpg"
    else:
        output_path = Path(args.output)
        if output_path.exists() and output_path.is_dir():
            output_path = output_path / "merged.jpg"
        elif output_path.suffix.lower() not in _IMAGE_EXTENSIONS and (
            not output_path.exists() or output_path.is_dir()
        ):
            output_path.mkdir(parents=True, exist_ok=True)
            output_path = output_path / "merged.jpg"
        else:
            output_path = output_path.resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = DetailPreprocessConfig(
        vertical_margin_px=args.vertical_margin,
        between_images_margin_px=args.between_margin,
        white_threshold=args.white_threshold,
        min_segment_height=args.min_segment_height,
        min_content_pixels_per_row=args.min_content_pixels_per_row,
        horizontal_margin_px=args.horizontal_margin,
        use_edge_background=not args.no_edge_background,
        background_tolerance=args.background_tolerance,
        background_border_px=args.background_border,
        verbose=not args.quiet,
    )
    out = preprocess_detail_images(input_paths, output_path, cfg)
    print(f"saved: {out}", flush=True)
