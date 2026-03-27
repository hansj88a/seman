"""image_split_test.py import 실행 샘플."""

from app.manager.services.image_split_test import (
    get_image_list,
    get_local_image_list,
    process_images,
    save_outputs,
    set_runtime_config,
)


def run_local_sample() -> None:
    """app/data/image 로컬 이미지 대상으로 실행."""
    set_runtime_config(
        VALID_IMAGE_TILE_WIDTH=800,
        CONTENT_TILE_HEIGHT=900,
        SPLIT_OVERLAP_PX=30,
        MIN_LAST_TILE_HEIGHT_PX=200,
    )
    _, images = get_local_image_list("app/data/image")
    merged, _ = process_images(images)
    out_paths = save_outputs(merged)
    print(f"[test.py] local done: {len(out_paths)} files")


def run_s3_sample() -> None:
    """S3 prefix 대상으로 실행."""
    set_runtime_config(
        VALID_IMAGE_TILE_WIDTH=800,
        CONTENT_TILE_HEIGHT=900,
        SPLIT_OVERLAP_PX=30,
        MIN_LAST_TILE_HEIGHT_PX=200,
    )
    # 예: s3://my-bucket/path/to/detail/
    _, _, images = get_image_list("s3://YOUR_BUCKET/YOUR_PREFIX/")
    merged, _ = process_images(images)
    out_paths = save_outputs(merged)
    print(f"[test.py] s3 done: {len(out_paths)} files")


if __name__ == "__main__":
    # 필요에 따라 하나만 호출하세요.
    run_local_sample()
    # run_s3_sample()
