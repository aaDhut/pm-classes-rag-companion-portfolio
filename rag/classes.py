"""Class numbering. A class's number is its lecture video's number ("Academic Session N").

The Drive notes folders are numbered differently: they skip Instructor B's three guest
lectures (Sessions 1, 8 and 14) and the telemetry lecture (Session 16), so from Session 2
on they run behind the videos until both reach 20. Matched by comparing each lecture's
content with each folder's files (2026-10-24). Folders not listed keep their number.
"""

# notes folder number -> class (video session) number
NOTES_TO_CLASS = {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 9, 8: 10, 9: 11, 10: 12,
                  11: 13, 12: 15, 13: 17, 15: 18, 16: 19}


def class_no(folder_no: int | None) -> int | None:
    """Class number for files in notes folder "Class <folder_no> - …"."""
    return None if folder_no is None else NOTES_TO_CLASS.get(folder_no, folder_no)


def label(no: int | None, title: str) -> str:
    return title if no is None else f"Class {no} - {title}"


def folder_label(entry: dict) -> str:
    """The notes folder's own name as on Drive, e.g. "Class 12 - Product Vision, Market Analysis"."""
    return label(entry["class_no"], entry["class_title"])
