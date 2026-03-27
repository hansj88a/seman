"""S3 이미지 로드 후 여백 제거/병합/세로 분할 샘플 스크립트."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.core.config import get_settings
from app.manager.services.s3_client import get_s3_client, list_s3_image_keys, parse_s3_uri


@dataclass(slots=True)
class SplitConfig:
    valid_image_tile_width: int = 800
    content_tile_height: int = 900
    min_repeated_pattern_px: int = 25
    white_threshold: int = 245
    row_content_threshold: int = 10
    min_content_segment_height: int = 12
    inter_tile_margin: int = 4
    output_dir: Path = Path("app/data/image/output")
    merged_file_name: str = "merged_from_s3.jpg"
    tile_stem: str = "content_tile"


def log(msg: str) -> None:
    print(f"[image_split_test] {msg}", flush=True)


def _is_mostly_white_or_flat(row_or_col_gray: np.ndarray, white_threshold: int) -> bool:
    """흰 배경 또는 단순 패턴(저분산)을 배경으로 판단."""
    mean_v = float(np.mean(row_or_col_gray))
    std_v = float(np.std(row_or_col_gray))
    return mean_v >= white_threshold or std_v <= 3.5


def _trim_repeated_background_edges(
    image: np.ndarray,
    *,
    min_repeat_px: int,
    white_threshold: int,
) -> np.ndarray:
    """가장자리에서 25px 이상 반복되는 단순 배경/패턴을 제거."""
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    top = 0
    run = 0
    for y in range(h):
        if _is_mostly_white_or_flat(gray[y, :], white_threshold):
            run += 1
            if run >= min_repeat_px:
                top = y + 1
        else:
            break

    bottom = h
    run = 0
    for y in range(h - 1, -1, -1):
        if _is_mostly_white_or_flat(gray[y, :], white_threshold):
            run += 1
            if run >= min_repeat_px:
                bottom = y
        else:
            break

    left = 0
    run = 0
    for x in range(w):
        if _is_mostly_white_or_flat(gray[:, x], white_threshold):
            run += 1
            if run >= min_repeat_px:
                left = x + 1
        else:
            break

    right = w
    run = 0
    for x in range(w - 1, -1, -1):
        if _is_mostly_white_or_flat(gray[:, x], white_threshold):
            run += 1
            if run >= min_repeat_px:
                right = x
        else:
            break

    if top >= bottom or left >= right:
        return image
    return image[top:bottom, left:right]


def _find_content_bbox(image: np.ndarray, white_threshold: int) -> Optional[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = (gray < white_threshold).astype(np.uint8) * 255
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    return x, y, x + w, y + h


def _extract_meaningful_tiles(
    image: np.ndarray,
    *,
    white_threshold: int,
    row_content_threshold: int,
    min_segment_height: int,
) -> list[np.ndarray]:
    """한 장 이미지 내 의미있는 세로 구간만 추출."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = (gray < white_threshold).astype(np.uint8)
    row_counts = np.sum(mask, axis=1)

    segments: list[tuple[int, int]] = []
    start: Optional[int] = None
    for i, count in enumerate(row_counts):
        if int(count) >= row_content_threshold and start is None:
            start = i
        elif int(count) < row_content_threshold and start is not None:
            if i - start >= min_segment_height:
                segments.append((start, i))
            start = None
    if start is not None and len(row_counts) - start >= min_segment_height:
        segments.append((start, len(row_counts)))

    if not segments:
        return [image]
    return [image[s:e, :] for s, e in segments if e > s]


def _resize_to_width(image: np.ndarray, target_width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= 0 or h <= 0:
        return image
    if w == target_width:
        return image
    new_h = max(1, int(round(h * (target_width / w))))
    return cv2.resize(image, (target_width, new_h), interpolation=cv2.INTER_AREA)


def _build_content_mask(image: np.ndarray, white_threshold: int) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return (gray < white_threshold).astype(np.uint8) * 255


def _split_with_content_extension(
    merged: np.ndarray,
    *,
    content_tile_height: int,
    white_threshold: int,
) -> list[np.ndarray]:
    """분할 높이에 걸치는 컨텐츠가 있으면 컨텐츠 끝 경계까지 확장."""
    mask = _build_content_mask(merged, white_threshold)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes: list[tuple[int, int]] = []
    for i in range(1, num_labels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area <= 40:
            continue
        boxes.append((y, y + h))

    h_total = merged.shape[0]
    y0 = 0
    out: list[np.ndarray] = []
    while y0 < h_total:
        y1 = min(y0 + content_tile_height, h_total)
        while True:
            extended = False
            for by0, by1 in boxes:
                intersects = not (by1 <= y0 or by0 >= y1)
                if not intersects:
                    continue
                if by1 > y1 and by1 <= h_total:
                    y1 = by1
                    extended = True
            if not extended:
                break
        out.append(merged[y0:y1, :])
        y0 = y1
    return out


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


def process_images(image_list: list[np.ndarray], cfg: SplitConfig) -> tuple[np.ndarray, list[np.ndarray]]:
    """요구사항 전체 파이프라인 수행."""
    if not image_list:
        raise ValueError("image[]가 비어 있습니다.")

    log("1) 이미지별 의미있는 컨텐츠 타일 추출 시작")
    processed_tiles: list[np.ndarray] = []
    for idx, image in enumerate(image_list, start=1):
        h0, w0 = image.shape[:2]
        log(f" - 원본[{idx}] 크기: {w0}x{h0}")

        # 2-1) 가장자리 25px 이상 반복되는 배경/패턴 제거
        trimmed = _trim_repeated_background_edges(
            image,
            min_repeat_px=cfg.min_repeated_pattern_px,
            white_threshold=cfg.white_threshold,
        )

        # 흰 여백 기반 bbox로 한 번 더 정리
        bbox = _find_content_bbox(trimmed, cfg.white_threshold)
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            trimmed = trimmed[y0:y1, x0:x1]

        # 1) 의미있는 세로 구간만 추출
        tiles = _extract_meaningful_tiles(
            trimmed,
            white_threshold=cfg.white_threshold,
            row_content_threshold=cfg.row_content_threshold,
            min_segment_height=cfg.min_content_segment_height,
        )
        log(f" - 의미 타일 수[{idx}]: {len(tiles)}")

        # 2) 유효 타일을 지정 폭(px)으로 맞춤
        for t_i, tile in enumerate(tiles, start=1):
            resized = _resize_to_width(tile, cfg.valid_image_tile_width)
            processed_tiles.append(resized)
            th, tw = resized.shape[:2]
            log(f"   > tile[{idx}-{t_i}] 리사이즈 완료: {tw}x{th}")

    if not processed_tiles:
        raise ValueError("가공 후 유효 타일이 없습니다.")

    log("3) 가공된 타일을 세로로 이어 붙이는 중")
    target_w = cfg.valid_image_tile_width
    total_h = sum(t.shape[0] for t in processed_tiles) + cfg.inter_tile_margin * (
        len(processed_tiles) - 1
    )
    merged = np.full((total_h, target_w, 3), 255, dtype=np.uint8)

    y = 0
    for i, tile in enumerate(processed_tiles):
        h, w = tile.shape[:2]
        merged[y : y + h, 0:w] = tile
        y += h
        if i < len(processed_tiles) - 1:
            y += cfg.inter_tile_margin

    log(f" - merged 크기: {merged.shape[1]}x{merged.shape[0]}")
    return merged, processed_tiles


def save_outputs(merged: np.ndarray, cfg: SplitConfig) -> list[Path]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = cfg.output_dir / cfg.merged_file_name
    ok = cv2.imwrite(str(merged_path), merged)
    if not ok:
        raise ValueError(f"병합 이미지 저장 실패: {merged_path}")
    log(f" - 병합 이미지 저장: {merged_path}")

    log(
        "4) 병합 이미지 세로 분할 시작 "
        f"(기준 높이={cfg.content_tile_height}px, 컨텐츠 경계 연장 활성)"
    )
    split_tiles = _split_with_content_extension(
        merged,
        content_tile_height=cfg.content_tile_height,
        white_threshold=cfg.white_threshold,
    )

    out_paths: list[Path] = [merged_path]
    for i, tile in enumerate(split_tiles, start=1):
        tile_path = cfg.output_dir / f"{cfg.tile_stem}_{i:03d}.jpg"
        ok = cv2.imwrite(str(tile_path), tile)
        if not ok:
            raise ValueError(f"타일 저장 실패: {tile_path}")
        out_paths.append(tile_path)
        log(f"5) 분할 타일 저장[{i}]: {tile_path} ({tile.shape[1]}x{tile.shape[0]})")
    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="S3 이미지 로드 -> 여백 제거/폭 맞춤/병합/세로분할 샘플"
    )
    parser.add_argument(
        "--key",
        required=True,
        help="S3 prefix 경로 (예: s3://my-bucket/path/to/detail/ 또는 path/to/detail/)",
    )
    parser.add_argument("--bucket", default=None, help="--key가 prefix일 때 사용할 버킷")
    parser.add_argument("--valid-image-tile-width", type=int, default=800)
    parser.add_argument("--content-tile-height", type=int, default=900)
    parser.add_argument("--output-dir", default="app/data/image/output")
    args = parser.parse_args()

    cfg = SplitConfig(
        valid_image_tile_width=max(1, args.valid_image_tile_width),
        content_tile_height=max(1, args.content_tile_height),
        output_dir=Path(args.output_dir),
    )

    log("파이프라인 시작")
    bucket, keys, images = get_image_list(args.key, bucket=args.bucket)
    log(f"로드 완료: bucket={bucket}, key 개수={len(keys)}, image[] 길이={len(images)}")
    merged, _ = process_images(images, cfg)
    out_paths = save_outputs(merged, cfg)
    log(f"완료: 출력 파일 수={len(out_paths)}")


if __name__ == "__main__":
    main()
