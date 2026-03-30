"""S3 이미지 로드 후 여백 제거/병합/세로 분할 샘플 스크립트."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.core.config import get_settings
from app.manager.services.s3_client import get_s3_client, list_s3_image_keys, parse_s3_uri


# Global runtime configuration (런타임에 set_runtime_config(...)로 덮어쓸 수 있음)
VALID_IMAGE_TILE_WIDTH = 800  # 타일(콘텐츠)의 목표 가로 픽셀 — 폭 맞춤/리사이즈 기준
CONTENT_TILE_HEIGHT = 900  # 세로 분할 시 한 조각의 기준 높이(px)
WHITE_THRESHOLD = 245  # 명도 이상이면 배경/흰 여백에 가깝다고 보는 임계값(0~255, _content_mask 및 좌우 트림)
USE_EDGE_BACKGROUND = True  # True면 가장자리 색으로 배경 추정, False면 WHITE_THRESHOLD만으로 전경 마스크
BACKGROUND_BORDER_PX = 2  # 엣지 색 중앙값 계산 시 사용하는 테두리 두께(px)
BACKGROUND_TOLERANCE = 18  # 엣지 배경색과 BGR 채널 차가 이 값 초과면 전경으로 보는 허용 오차
# 상·하 트림: 위/아래에서 "배경 행"이 최소 몇 행(높이 px) 연속이어야 잘라낼지.
# - 기본 40이면: 맨 위(또는 맨 아래)에서 배경으로 인정되는 행이 40행 연속일 때만 트림 시작.
# - 그 후 같은 방향으로 배경 행이 이어지면 연속 배경 구간 끝까지 잘라냄(첫 비배경 행에서 중단).
# - 연속 배경이 이 값 미만이면 해당 변은 전혀 자르지 않음.
MIN_REPEATED_PATTERN_PX_VERTICAL = 40
VERTICAL_TRIM_MAX_GRAY_STD = 4.0  # 상·하: "흰 띠" 판별 시 행 명도 표준편차 상한(이하면 균일로 봄)
VERTICAL_TRIM_EDGE_TOLERANCE = 18  # 상·하: 행 평균 BGR과 엣지 중앙값 BGR 채널 차 허용값
VERTICAL_TRIM_PATTERN_FLAT_STD = 6.0  # 상·하: 엣지 색과 맞을 때 "균일 띠"로 볼 명도 표준편차 상한
VERTICAL_TRIM_MAX_HORIZ_MEAN_ABS_DIFF = 14.0  # 상·하: 엣지 색 일치 시 행 내 가로 픽셀 차 평균 상한(저텍스처/미세 패턴)
# 좌·우 트림: 왼쪽/오른쪽에서 밝은 여백 "열"이 최소 몇 열(폭 px) 연속이어야 잘라낼지(상·하와 동일한 run 로직)
MIN_REPEATED_PATTERN_PX_HORIZONTAL = 40
HORIZONTAL_TRIM_MAX_GRAY_STD = 5.0  # 좌·우: 여백으로 볼 열의 명도 표준편차 상한(밝고 거의 균일)
PATTERN_FLAT_STD_THRESHOLD_HORIZONTAL = 2.0  # 레거시 전역 유지용(가장자리 트림 로직에서는 미사용)
BACKGROUND_TOLERANCE_HORIZONTAL = 10  # 레거시 전역 유지용(가장자리 트림 로직에서는 미사용)
MAX_EDGE_TRIM_RATIO = 0.18  # 상·하 한쪽에서 잘라낼 수 있는 최대 비율(이미지 높이 대비 상한)
MAX_EDGE_TRIM_RATIO_HORIZONTAL = 0.08  # 좌·우 한쪽에서 잘라낼 수 있는 최대 비율(이미지 폭 대비)
CONTENT_BBOX_MIN_AREA_RATIO = 0.05  # 콘텐츠 bbox가 이미지 면적 대비 이 비율 미만이면 bbox 무시
CONTENT_BBOX_PADDING_PX = 12  # 콘텐츠 bbox에 추가할 여백(px)
ROW_CONTENT_THRESHOLD = 1  # 한 행을 "전경이 있다"고 볼 최소 전경 픽셀 수
MIN_CONTENT_SEGMENT_HEIGHT = 2  # 의미 있는 세로 구간으로 인정할 최소 높이(행)
INTER_TILE_MARGIN = 0  # 타일 사이 간격(현재 파이프라인에서 사용 값)
SPLIT_OVERLAP_PX = 30  # 세로 분할 시 다음 타일과 겹치게 가져올 픽셀 수
MIN_LAST_TILE_HEIGHT_PX = 200  # 마지막 타일 높이가 이 미만이면 직전 타일에 흡수
SAVE_CONTENT_DEBUG_PREVIEW = True  # True면 콘텐츠 디버그 미리보기 이미지 저장
CONTENT_DEBUG_DIR_NAME = "content_debug"  # 디버그 출력 하위 디렉터리 이름
OUTPUT_DIR = Path("app/data/image/output")  # 결과 저장 루트
MERGED_FILE_NAME = "merged_from_s3.jpg"  # 병합 단일 파일 이름
TILE_STEM = "content_tile"  # 분할 타일 파일명 접두 (예: content_tile_001.jpg)


def set_runtime_config(**kwargs: object) -> None:
    """전역 설정값을 런타임에 덮어쓴다."""
    for key, value in kwargs.items():
        if key not in globals():
            raise ValueError(f"Unknown config key: {key}")
        globals()[key] = value


def log(msg: str) -> None:
    print(f"[image_split_test] {msg}", flush=True)


def default_local_image_dir() -> Path:
    return Path("app/data/image")


def _median_edge_bgr(image: np.ndarray, border_px: int) -> tuple[int, int, int]:
    """가장자리 strip에서 BGR 채널별 중앙값(배경색 추정)."""
    h, w = image.shape[:2]
    b = max(1, min(int(border_px), w // 2, h // 2))
    parts = [
        image[:b, :].reshape(-1, 3),
        image[h - b :, :].reshape(-1, 3),
        image[:, :b].reshape(-1, 3),
        image[:, w - b :].reshape(-1, 3),
    ]
    samples = np.vstack(parts)
    return (
        int(np.median(samples[:, 0])),
        int(np.median(samples[:, 1])),
        int(np.median(samples[:, 2])),
    )


def _row_is_vertical_trim_background(
    row_bgr: np.ndarray,
    row_gray: np.ndarray,
    *,
    white_threshold: int,
    max_white_gray_std: float,
    edge_bgr: tuple[int, int, int],
    edge_tolerance: float,
    pattern_flat_gray_std: float,
    max_horiz_mean_abs_diff: float,
) -> bool:
    """상/하 트림: 흰 균일 띠, 또는 엣지 색과 유사한 균일·저텍스처(가로) 행을 배경으로 본다."""
    mean_g = float(np.mean(row_gray))
    std_g = float(np.std(row_gray))
    if mean_g >= float(white_threshold) and std_g <= max_white_gray_std:
        return True
    rm = row_bgr.mean(axis=0).astype(np.float64)
    bg = np.array(edge_bgr, dtype=np.float64)
    slack = 2.0
    if not bool(np.all(np.abs(rm - bg) <= float(edge_tolerance) + slack)):
        return False
    if std_g <= float(pattern_flat_gray_std):
        return True
    g64 = row_gray.astype(np.float64)
    if g64.size > 1:
        horiz_delta = float(np.mean(np.abs(np.diff(g64))))
    else:
        horiz_delta = 0.0
    return horiz_delta <= float(max_horiz_mean_abs_diff)


def _column_is_bright_margin_strip(
    strip_gray: np.ndarray,
    *,
    white_threshold: int,
    max_gray_std: float,
) -> bool:
    """좌/우 트림: 밝고 거의 균일한 열만 여백으로 본다."""
    mean_v = float(np.mean(strip_gray))
    std_v = float(np.std(strip_gray))
    return mean_v >= float(white_threshold) and std_v <= float(max_gray_std)


def _trim_repeated_background_edges(
    image: np.ndarray,
) -> np.ndarray:
    """가장자리 반복 배경 제거. 상·하는 엣지 유사 색·저텍스처 띠, 좌·우는 밝은 균일 여백만.

    상·하 트림 높이(반복 판단):
    - MIN_REPEATED_PATTERN_PX_VERTICAL 은 "패턴 주기"가 아니라 **연속 배경 행 개수** 기준이다.
    - 예: 기본값 40이면 윗쪽에서 배경 행이 **40행 연속**이어야 그 이후(연속 배경 전체)를 잘라낸다.
    - 39행만 배경이고 바로 본문이면 상단은 자르지 않는다.
    """
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_bgr_trim = _median_edge_bgr(image, BACKGROUND_BORDER_PX)
    white_threshold = WHITE_THRESHOLD
    min_repeat_v = max(1, int(MIN_REPEATED_PATTERN_PX_VERTICAL))
    min_repeat_h = max(1, int(MIN_REPEATED_PATTERN_PX_HORIZONTAL))
    std_v_white = max(0.1, float(VERTICAL_TRIM_MAX_GRAY_STD))
    edge_tol_v = max(0.0, float(VERTICAL_TRIM_EDGE_TOLERANCE))
    pattern_flat_v = max(0.1, float(VERTICAL_TRIM_PATTERN_FLAT_STD))
    horiz_diff_v = max(0.0, float(VERTICAL_TRIM_MAX_HORIZ_MEAN_ABS_DIFF))
    std_h_margin = max(0.1, float(HORIZONTAL_TRIM_MAX_GRAY_STD))

    top = 0
    run = 0
    for y in range(h):
        if _row_is_vertical_trim_background(
            image[y, :],
            gray[y, :],
            white_threshold=white_threshold,
            max_white_gray_std=std_v_white,
            edge_bgr=edge_bgr_trim,
            edge_tolerance=edge_tol_v,
            pattern_flat_gray_std=pattern_flat_v,
            max_horiz_mean_abs_diff=horiz_diff_v,
        ):
            run += 1
            if run >= min_repeat_v:
                top = y + 1
        else:
            break

    bottom = h
    run = 0
    for y in range(h - 1, -1, -1):
        if _row_is_vertical_trim_background(
            image[y, :],
            gray[y, :],
            white_threshold=white_threshold,
            max_white_gray_std=std_v_white,
            edge_bgr=edge_bgr_trim,
            edge_tolerance=edge_tol_v,
            pattern_flat_gray_std=pattern_flat_v,
            max_horiz_mean_abs_diff=horiz_diff_v,
        ):
            run += 1
            if run >= min_repeat_v:
                bottom = y
        else:
            break

    left = 0
    run = 0
    for x in range(w):
        if _column_is_bright_margin_strip(
            gray[:, x],
            white_threshold=white_threshold,
            max_gray_std=std_h_margin,
        ):
            run += 1
            if run >= min_repeat_h:
                left = x + 1
        else:
            break

    right = w
    run = 0
    for x in range(w - 1, -1, -1):
        if _column_is_bright_margin_strip(
            gray[:, x],
            white_threshold=white_threshold,
            max_gray_std=std_h_margin,
        ):
            run += 1
            if run >= min_repeat_h:
                right = x
        else:
            break

    # 과도한 잘림 방지: 각 변 트림량 제한 (좌우는 더 엄격하게 제한)
    max_trim_h = int(h * max(0.0, min(0.45, MAX_EDGE_TRIM_RATIO)))
    max_trim_w = int(w * max(0.0, min(0.45, MAX_EDGE_TRIM_RATIO_HORIZONTAL)))
    top = min(top, max_trim_h)
    left = min(left, max_trim_w)
    bottom = max(bottom, h - max_trim_h)
    right = max(right, w - max_trim_w)

    if top >= bottom or left >= right:
        return image
    return image[top:bottom, left:right]


def _content_mask_foreground_u8(image: np.ndarray) -> np.ndarray:
    """전경 1, 배경 0 (uint8)."""
    if USE_EDGE_BACKGROUND:
        b0, g0, r0 = _median_edge_bgr(image, BACKGROUND_BORDER_PX)
        b, g, r = cv2.split(image)
        db = np.abs(b.astype(np.int16) - b0)
        dg = np.abs(g.astype(np.int16) - g0)
        dr = np.abs(r.astype(np.int16) - r0)
        diff = np.maximum(np.maximum(db, dg), dr)
        return (diff > int(BACKGROUND_TOLERANCE)).astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return (gray < WHITE_THRESHOLD).astype(np.uint8)


def _find_content_bbox(image: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    m = _content_mask_foreground_u8(image)
    mask255 = (m * 255).astype(np.uint8)
    coords = cv2.findNonZero(mask255)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    ih, iw = image.shape[:2]
    if iw <= 0 or ih <= 0:
        return None

    # 과도한 축소 bbox 방지
    area_ratio = float(w * h) / float(iw * ih)
    if area_ratio < max(0.0, min(1.0, CONTENT_BBOX_MIN_AREA_RATIO)):
        return None

    pad = max(0, int(CONTENT_BBOX_PADDING_PX))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(iw, x + w + pad)
    y1 = min(ih, y + h + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _extract_meaningful_tiles(
    image: np.ndarray,
    *,
    row_content_threshold: int,
    min_segment_height: int,
) -> list[np.ndarray]:
    """한 장 이미지 내 의미있는 세로 구간만 추출."""
    mask = _content_mask_foreground_u8(image)
    row_counts = np.sum(mask, axis=1)

    threshold = max(1, int(row_content_threshold))
    min_h = max(1, int(min_segment_height))
    segments: list[tuple[int, int]] = []
    start: Optional[int] = None
    for i, count in enumerate(row_counts):
        if int(count) >= threshold and start is None:
            start = i
        elif int(count) < threshold and start is not None:
            if i - start >= min_h:
                segments.append((start, i))
            start = None
    if start is not None and len(row_counts) - start >= min_h:
        segments.append((start, len(row_counts)))

    if not segments:
        return [image]
    return [image[s:e, :] for s, e in segments if e > s]


def _find_meaningful_row_segments(
    image: np.ndarray,
    *,
    row_content_threshold: int,
    min_segment_height: int,
) -> list[tuple[int, int]]:
    """_extract_meaningful_tiles와 동일 기준으로 세로 구간만 반환."""
    mask = _content_mask_foreground_u8(image)
    row_counts = np.sum(mask, axis=1)
    threshold = max(1, int(row_content_threshold))
    min_h = max(1, int(min_segment_height))

    segments: list[tuple[int, int]] = []
    start: Optional[int] = None
    for i, count in enumerate(row_counts):
        if int(count) >= threshold and start is None:
            start = i
        elif int(count) < threshold and start is not None:
            if i - start >= min_h:
                segments.append((start, i))
            start = None
    if start is not None and len(row_counts) - start >= min_h:
        segments.append((start, len(row_counts)))
    if not segments:
        return [(0, image.shape[0])]
    return segments


def _save_content_debug_preview(
    source_image: np.ndarray,
    trimmed_image: np.ndarray,
    bbox_on_source: Optional[tuple[int, int, int, int]],
    trimmed_segments: list[tuple[int, int]],
    out_path: Path,
) -> None:
    """
    초기 콘텐츠 판단 시각화:
    - source: 콘텐츠 bbox(녹색)
    - trimmed: 세로 콘텐츠 구간(빨간색 채우기+테두리)
    """
    left = source_image.copy()
    right = trimmed_image.copy()

    if bbox_on_source is not None:
        x0, y0, x1, y1 = bbox_on_source
        cv2.rectangle(left, (x0, y0), (x1 - 1, y1 - 1), (0, 200, 0), 2)

    overlay = right.copy()
    for (s, e) in trimmed_segments:
        if e <= s:
            continue
        cv2.rectangle(overlay, (0, s), (right.shape[1] - 1, e - 1), (0, 0, 255), -1)
        cv2.rectangle(right, (0, s), (right.shape[1] - 1, e - 1), (0, 0, 255), 2)
    right = cv2.addWeighted(overlay, 0.12, right, 0.88, 0)

    h = max(left.shape[0], right.shape[0])
    w = left.shape[1] + right.shape[1] + 20
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] + 20 :] = right
    cv2.putText(canvas, "source(bbox)", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "trimmed(segments)",
        (left.shape[1] + 28, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (60, 60, 60),
        2,
        cv2.LINE_AA,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def _resize_to_width(image: np.ndarray, target_width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= 0 or h <= 0:
        return image
    if w == target_width:
        return image
    new_h = max(1, int(round(h * (target_width / w))))
    return cv2.resize(image, (target_width, new_h), interpolation=cv2.INTER_AREA)


def get_image_from_s3(bucket: str, key: str) -> np.ndarray:
    """요구사항: cv2.imdecode로 S3 바이트를 이미지로 변환해 반환."""
    client = get_s3_client()
    resp = client.get_object(Bucket=bucket, Key=key)
    body = resp["Body"].read()
    arr = np.frombuffer(body, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"이미지 디코딩 실패: s3://{bucket}/{key}")
    return image


def get_image_list(s3_key_path: str, bucket: Optional[str] = None) -> tuple[str, list[str], list[np.ndarray]]:
    """
    요구사항: key 경로 목록을 가져오고 get_image_from_s3(key)로 image[] 생성.
    - s3_key_path: `s3://bucket/prefix/` 또는 `prefix/`
    """
    if s3_key_path.lower().startswith("s3://"):
        parsed_bucket, prefix = parse_s3_uri(s3_key_path)
        use_bucket = parsed_bucket
    else:
        if not bucket:
            settings = get_settings()
            if not settings.s3_bucket:
                raise ValueError("bucket 미지정 + S3_BUCKET 설정 없음")
            use_bucket = settings.s3_bucket
        else:
            use_bucket = bucket
        prefix = s3_key_path

    keys = list_s3_image_keys(use_bucket, prefix)
    image_list: list[np.ndarray] = []
    for idx, key in enumerate(keys, start=1):
        log(f"S3 이미지 로드 {idx}/{len(keys)}: s3://{use_bucket}/{key}")
        image_list.append(get_image_from_s3(use_bucket, key))
    return use_bucket, keys, image_list


def get_local_image_list(local_dir: str | Path) -> tuple[list[str], list[np.ndarray]]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
    d = Path(local_dir)
    if not d.exists() or not d.is_dir():
        raise ValueError(f"로컬 이미지 디렉터리가 없습니다: {d}")
    files = sorted(
        [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts],
        key=lambda x: x.name.lower(),
    )
    if not files:
        raise ValueError(f"로컬 이미지가 없습니다: {d}")
    images: list[np.ndarray] = []
    for idx, p in enumerate(files, start=1):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            log(f"로컬 이미지 로드 실패(건너뜀): {p}")
            continue
        images.append(img)
        log(f"로컬 이미지 로드 {idx}/{len(files)}: {p}")
    return [str(p) for p in files], images


def process_images(image_list: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """요구사항 전체 파이프라인 수행."""
    if not image_list:
        raise ValueError("image[]가 비어 있습니다.")

    log(
        "1) 이미지별 의미있는 컨텐츠 타일 추출 시작 "
        f"(edge_bg={USE_EDGE_BACKGROUND}, tol={BACKGROUND_TOLERANCE})"
    )
    processed_tiles: list[np.ndarray] = []
    debug_dir = OUTPUT_DIR / CONTENT_DEBUG_DIR_NAME
    for idx, image in enumerate(image_list, start=1):
        h0, w0 = image.shape[:2]
        log(f" - 원본[{idx}] 크기: {w0}x{h0}")

        # 2-1) 가장자리 25px 이상 반복되는 배경/패턴 제거
        trimmed = _trim_repeated_background_edges(image)

        # 콘텐츠 마스크 기반 bbox로 한 번 더 정리
        edge_trimmed = trimmed.copy()
        bbox = _find_content_bbox(edge_trimmed)
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            trimmed = edge_trimmed[y0:y1, x0:x1]
        else:
            trimmed = edge_trimmed

        # 1) 의미있는 세로 구간만 추출
        trimmed_segments = _find_meaningful_row_segments(
            trimmed,
            row_content_threshold=ROW_CONTENT_THRESHOLD,
            min_segment_height=MIN_CONTENT_SEGMENT_HEIGHT,
        )
        if SAVE_CONTENT_DEBUG_PREVIEW:
            dbg_path = debug_dir / f"content_detect_{idx:03d}.jpg"
            _save_content_debug_preview(edge_trimmed, trimmed, bbox, trimmed_segments, dbg_path)
            log(f"   > 콘텐츠 영역 시각화 저장: {dbg_path}")

        tiles = _extract_meaningful_tiles(
            trimmed,
            row_content_threshold=ROW_CONTENT_THRESHOLD,
            min_segment_height=MIN_CONTENT_SEGMENT_HEIGHT,
        )
        log(f" - 의미 타일 수[{idx}]: {len(tiles)}")

        # 2) 유효 타일을 지정 폭(px)으로 맞춤
        for t_i, tile in enumerate(tiles, start=1):
            resized = _resize_to_width(tile, VALID_IMAGE_TILE_WIDTH)
            processed_tiles.append(resized)
            th, tw = resized.shape[:2]
            log(f"   > tile[{idx}-{t_i}] 리사이즈 완료: {tw}x{th}")

    if not processed_tiles:
        raise ValueError("가공 후 유효 타일이 없습니다.")

    log("3) 가공된 타일을 세로로 이어 붙이는 중")
    target_w = VALID_IMAGE_TILE_WIDTH
    total_h = sum(t.shape[0] for t in processed_tiles) + INTER_TILE_MARGIN * (
        len(processed_tiles) - 1
    )
    merged = np.full((total_h, target_w, 3), 255, dtype=np.uint8)

    y = 0
    for i, tile in enumerate(processed_tiles):
        h, w = tile.shape[:2]
        merged[y : y + h, 0:w] = tile
        y += h
        if i < len(processed_tiles) - 1:
            if INTER_TILE_MARGIN > 0:
                y += INTER_TILE_MARGIN

    log(f" - merged 크기: {merged.shape[1]}x{merged.shape[0]}")
    return merged, processed_tiles


def _split_uniform_with_overlap(
    merged: np.ndarray,
    *,
    tile_height: int,
    overlap_px: int,
    min_last_tile_height_px: int,
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """
    병합 이미지를 고정 높이로 분할.
    - 다음 타일 시작은 이전 타일 끝에서 overlap_px 만큼 위(겹침)
    - 마지막 타일 높이가 min_last_tile_height_px 미만이면 이전 타일에 포함
    """
    h_total = merged.shape[0]
    if h_total <= 0:
        return [], []
    h = max(1, int(tile_height))
    overlap = max(0, int(overlap_px))
    if overlap >= h:
        overlap = h - 1

    ranges: list[tuple[int, int]] = []
    start = 0
    while start < h_total:
        end = min(start + h, h_total)
        ranges.append((start, end))
        if end >= h_total:
            break
        start = max(0, end - overlap)

    if len(ranges) >= 2:
        last_s, last_e = ranges[-1]
        last_h = last_e - last_s
        if last_e == h_total and last_h < max(1, int(min_last_tile_height_px)):
            prev_s, _prev_e = ranges[-2]
            ranges[-2] = (prev_s, h_total)
            ranges.pop()

    valid_ranges = [(s, e) for s, e in ranges if e > s]
    tiles = [merged[s:e, :] for s, e in valid_ranges]
    return tiles, valid_ranges


def save_outputs(merged: np.ndarray) -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged_path = OUTPUT_DIR / MERGED_FILE_NAME
    ok = cv2.imwrite(str(merged_path), merged)
    if not ok:
        raise ValueError(f"병합 이미지 저장 실패: {merged_path}")
    log(f" - 병합 이미지 저장: {merged_path}")

    log(
        "4) 병합 이미지 세로 분할 시작 "
        f"(기준 높이={CONTENT_TILE_HEIGHT}px, 다음 타일 시작 오버랩={SPLIT_OVERLAP_PX}px)"
    )
    split_tiles, split_ranges = _split_uniform_with_overlap(
        merged,
        tile_height=CONTENT_TILE_HEIGHT,
        overlap_px=SPLIT_OVERLAP_PX,
        min_last_tile_height_px=MIN_LAST_TILE_HEIGHT_PX,
    )

    out_paths: list[Path] = [merged_path]
    for i, (tile, (y0, y1)) in enumerate(zip(split_tiles, split_ranges), start=1):
        tile_path = OUTPUT_DIR / f"{TILE_STEM}_{i:03d}.jpg"
        ok = cv2.imwrite(str(tile_path), tile)
        if not ok:
            raise ValueError(f"타일 저장 실패: {tile_path}")
        out_paths.append(tile_path)
        log(
            f"5) 분할 타일 저장[{i}]: {tile_path} "
            f"(y={y0}:{y1}, size={tile.shape[1]}x{tile.shape[0]})"
        )
    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="S3/로컬 이미지 로드 -> 여백 제거/폭 맞춤/병합/세로분할 샘플"
    )
    parser.add_argument(
        "--key",
        default=None,
        help="S3 prefix 경로 (예: s3://my-bucket/path/to/detail/ 또는 path/to/detail/)",
    )
    parser.add_argument("--bucket", default=None, help="--key가 prefix일 때 사용할 버킷")
    parser.add_argument(
        "--local-input-dir",
        default=None,
        help="로컬 이미지 디렉터리 (예: app/data/image). 지정 시 S3 대신 로컬 모드 실행",
    )
    parser.add_argument("--valid-image-tile-width", type=int, default=VALID_IMAGE_TILE_WIDTH)
    parser.add_argument("--content-tile-height", type=int, default=CONTENT_TILE_HEIGHT)
    parser.add_argument("--split-overlap-px", type=int, default=SPLIT_OVERLAP_PX)
    parser.add_argument(
        "--min-last-tile-height-px",
        type=int,
        default=MIN_LAST_TILE_HEIGHT_PX,
        help="마지막 타일 높이가 이 값 미만이면 직전 타일에 포함",
    )
    parser.add_argument(
        "--no-edge-background",
        action="store_true",
        help="엣지 기반 배경 추정 비활성화(명도 white_threshold만 사용)",
    )
    parser.add_argument("--background-border-px", type=int, default=BACKGROUND_BORDER_PX)
    parser.add_argument(
        "--background-tolerance",
        type=int,
        default=BACKGROUND_TOLERANCE,
        help="엣지 배경색과 채널 차이가 이 값 초과면 전경(0~255)",
    )
    parser.add_argument("--min-repeated-pattern-px-vertical", type=int, default=MIN_REPEATED_PATTERN_PX_VERTICAL)
    parser.add_argument(
        "--vertical-trim-max-gray-std",
        type=float,
        default=VERTICAL_TRIM_MAX_GRAY_STD,
        help="상/하 트림: 흰 띠로 볼 때 행 명도 표준편차 상한",
    )
    parser.add_argument(
        "--vertical-trim-edge-tolerance",
        type=float,
        default=VERTICAL_TRIM_EDGE_TOLERANCE,
        help="상/하 트림: 행 평균 BGR과 엣지 중앙값 채널 차 상한",
    )
    parser.add_argument(
        "--vertical-trim-pattern-flat-gray-std",
        type=float,
        default=VERTICAL_TRIM_PATTERN_FLAT_STD,
        help="상/하 트림: 엣지 색과 맞을 때 균일 띠로 볼 명도 표준편차 상한",
    )
    parser.add_argument(
        "--vertical-trim-max-horiz-mean-abs-diff",
        type=float,
        default=VERTICAL_TRIM_MAX_HORIZ_MEAN_ABS_DIFF,
        help="상/하 트림: 엣지 색 일치 시 가로 방향 평균 픽셀 차(저텍스처) 상한",
    )
    parser.add_argument("--min-repeated-pattern-px-horizontal", type=int, default=MIN_REPEATED_PATTERN_PX_HORIZONTAL)
    parser.add_argument(
        "--horizontal-trim-max-gray-std",
        type=float,
        default=HORIZONTAL_TRIM_MAX_GRAY_STD,
        help="좌/우 트림: 밝은 여백 열의 명도 표준편차 상한",
    )
    parser.add_argument("--max-edge-trim-ratio", type=float, default=MAX_EDGE_TRIM_RATIO)
    parser.add_argument(
        "--max-edge-trim-ratio-horizontal",
        type=float,
        default=MAX_EDGE_TRIM_RATIO_HORIZONTAL,
        help="좌/우 한쪽에서 최대 트림 가능한 비율(0~0.45)",
    )
    parser.add_argument("--content-bbox-min-area-ratio", type=float, default=CONTENT_BBOX_MIN_AREA_RATIO)
    parser.add_argument("--content-bbox-padding-px", type=int, default=CONTENT_BBOX_PADDING_PX)
    parser.add_argument("--white-threshold", type=int, default=WHITE_THRESHOLD)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    set_runtime_config(
        VALID_IMAGE_TILE_WIDTH=max(1, args.valid_image_tile_width),
        CONTENT_TILE_HEIGHT=max(1, args.content_tile_height),
        SPLIT_OVERLAP_PX=max(0, args.split_overlap_px),
        MIN_LAST_TILE_HEIGHT_PX=max(1, args.min_last_tile_height_px),
        USE_EDGE_BACKGROUND=not args.no_edge_background,
        BACKGROUND_BORDER_PX=max(1, args.background_border_px),
        BACKGROUND_TOLERANCE=max(0, args.background_tolerance),
        MIN_REPEATED_PATTERN_PX_VERTICAL=max(1, args.min_repeated_pattern_px_vertical),
        VERTICAL_TRIM_MAX_GRAY_STD=max(0.1, args.vertical_trim_max_gray_std),
        VERTICAL_TRIM_EDGE_TOLERANCE=max(0.0, args.vertical_trim_edge_tolerance),
        VERTICAL_TRIM_PATTERN_FLAT_STD=max(0.1, args.vertical_trim_pattern_flat_gray_std),
        VERTICAL_TRIM_MAX_HORIZ_MEAN_ABS_DIFF=max(0.0, args.vertical_trim_max_horiz_mean_abs_diff),
        MIN_REPEATED_PATTERN_PX_HORIZONTAL=max(1, args.min_repeated_pattern_px_horizontal),
        HORIZONTAL_TRIM_MAX_GRAY_STD=max(0.1, args.horizontal_trim_max_gray_std),
        MAX_EDGE_TRIM_RATIO=max(0.0, min(0.45, args.max_edge_trim_ratio)),
        MAX_EDGE_TRIM_RATIO_HORIZONTAL=max(0.0, min(0.45, args.max_edge_trim_ratio_horizontal)),
        CONTENT_BBOX_MIN_AREA_RATIO=max(0.0, min(1.0, args.content_bbox_min_area_ratio)),
        CONTENT_BBOX_PADDING_PX=max(0, args.content_bbox_padding_px),
        WHITE_THRESHOLD=max(0, min(255, args.white_threshold)),
        OUTPUT_DIR=Path(args.output_dir),
    )

    log("파이프라인 시작")
    if args.local_input_dir:
        files, images = get_local_image_list(args.local_input_dir)
        log(f"로드 완료(LOCAL): 파일 개수={len(files)}, image[] 길이={len(images)}")
    elif args.key:
        bucket, keys, images = get_image_list(args.key, bucket=args.bucket)
        log(f"로드 완료(S3): bucket={bucket}, key 개수={len(keys)}, image[] 길이={len(images)}")
    else:
        default_dir = default_local_image_dir()
        files, images = get_local_image_list(default_dir)
        log(
            f"로드 완료(LOCAL-DEFAULT): dir={default_dir}, "
            f"파일 개수={len(files)}, image[] 길이={len(images)}"
        )
    merged, _ = process_images(images)
    out_paths = save_outputs(merged)
    log(f"완료: 출력 파일 수={len(out_paths)}")


def run_local_sample() -> None:
    """
    app/data/image 경로 대상 실행 샘플 코드.
    필요시 다른 코드에서 import 해서 호출 가능.
    """
    set_runtime_config(
        VALID_IMAGE_TILE_WIDTH=800,
        CONTENT_TILE_HEIGHT=900,
        OUTPUT_DIR=Path("app/data/image/output"),
    )
    log("로컬 샘플 실행 시작")
    files, images = get_local_image_list(default_local_image_dir())
    log(f"샘플 로드 완료: 파일={len(files)}장")
    merged, _ = process_images(images)
    out_paths = save_outputs(merged)
    log(f"샘플 실행 완료: 생성 파일={len(out_paths)}개")


if __name__ == "__main__":
    main()
