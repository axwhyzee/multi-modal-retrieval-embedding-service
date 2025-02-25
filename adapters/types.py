from enum import StrEnum
from pathlib import Path
from typing import Dict, cast


class Modal(StrEnum):
    IMAGE = "IMAGE"
    TEXT = "TEXT"
    VIDEO = "VIDEO"


class FileExt(StrEnum):
    TXT = ".txt"
    PNG = ".png"
    JPG = ".jpg"
    JPEG = ".jpeg"
    MP4 = ".mp4"


_FileExtToModal: Dict[FileExt, Modal] = {
    FileExt.JPG: Modal.IMAGE,
    FileExt.JPEG: Modal.IMAGE,
    FileExt.PNG: Modal.IMAGE,
    FileExt.MP4: Modal.VIDEO,
    FileExt.TXT: Modal.TEXT,
}


def get_modal(key: str) -> Modal:
    suffix = Path(key).suffix
    file_ext = cast(FileExt, FileExt._value2member_map_[suffix])
    return _FileExtToModal[file_ext]
