"""Image rotation utilities using lossless PIL transpose operations.

Uses PIL transpose (not rotate) to avoid interpolation artifacts.
All angles are clockwise; PIL uses counter-clockwise internally.
"""

from PIL import Image

# CW angle -> PIL CCW transpose operation
_ROTATION_TO_TRANSPOSE = {
    0: None,
    90: Image.Transpose.ROTATE_270,
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_90,
}

VALID_ANGLES = frozenset(_ROTATION_TO_TRANSPOSE.keys())


def rotate_image(img: Image.Image, angle: int) -> Image.Image:
    """Rotate image clockwise by 0/90/180/270 degrees (lossless)."""
    if angle not in VALID_ANGLES:
        raise ValueError(
            f"Angle must be one of {sorted(VALID_ANGLES)}, got {angle}"
        )
    transpose_op = _ROTATION_TO_TRANSPOSE[angle]
    if transpose_op is None:
        return img.copy()
    return img.transpose(transpose_op)


def correct_rotation(img: Image.Image, predicted_angle: int) -> Image.Image:
    """Undo a detected rotation by applying the inverse rotation."""
    correction_angle = (360 - predicted_angle) % 360
    return rotate_image(img, correction_angle)


def angle_from_index(idx: int) -> int:
    """Deterministic angle assignment: index % 4 -> {0, 90, 180, 270}."""
    return [0, 90, 180, 270][idx % 4]
