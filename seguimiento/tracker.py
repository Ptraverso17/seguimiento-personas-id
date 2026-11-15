"""Detección (YOLO) + seguimiento (ByteTrack / BoT-SORT) de personas."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

PERSON_CLASS = 0  # "person" en COCO
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class TrackedPerson:
    person_id: int  # ID que se muestra (1, 2, 3...)
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 en píxeles
    confidence: float

    @property
    def feet(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.box
        return ((x1 + x2) / 2, y2)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class PeopleTracker:
    """Envuelve el modelo de Ultralytics y devuelve personas con un ID estable.

    El tracker interno numera las pistas con huecos (1, 4, 9...) porque crea
    pistas provisionales que luego descarta. Aquí se renumeran en el orden
    en que aparecen para que la primera persona sea la 1, la segunda la 2, etc.
    """

    def __init__(
        self,
        model: str = "yolo11n.pt",
        tracker: str = "bytetrack",
        conf: float = 0.3,
        iou: float = 0.5,
        imgsz: int = 640,
        device: str | None = None,
    ):
        from ultralytics import YOLO  # import diferido: los tests no lo necesitan

        self.model = YOLO(model)
        self.tracker_cfg = tracker if tracker.endswith(".yaml") else f"{tracker}.yaml"
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self._id_map: dict[int, int] = {}

    def _display_id(self, raw_id: int) -> int:
        if raw_id not in self._id_map:
            self._id_map[raw_id] = len(self._id_map) + 1
        return self._id_map[raw_id]

    def update(self, frame: np.ndarray) -> list[TrackedPerson]:
        result = self.model.track(
            frame,
            persist=True,
            classes=[PERSON_CLASS],
            tracker=self.tracker_cfg,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.int().cpu().tolist()
        confs = boxes.conf.cpu().numpy()
        return [
            TrackedPerson(self._display_id(raw), tuple(map(float, box)), float(c))
            for raw, box, c in zip(ids, xyxy, confs)
        ]


def open_source(source: str) -> tuple[Iterator[np.ndarray], float, int, int, int]:
    """Abre un vídeo, una webcam ("0") o una carpeta de imágenes (secuencias MOT).

    Devuelve (generador de frames, fps, ancho, alto, nº de frames o -1).
    """
    path = Path(source)
    if path.is_dir():
        # Las secuencias de MOTChallenge vienen como img1/000001.jpg, ...
        if (path / "img1").is_dir():
            path = path / "img1"
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not files:
            raise FileNotFoundError(f"No hay imágenes en {path}")
        first = cv2.imread(str(files[0]))
        h, w = first.shape[:2]
        fps = _read_seqinfo_fps(path.parent) or 30.0

        def gen_images() -> Iterator[np.ndarray]:
            for f in files:
                yield cv2.imread(str(f))

        return gen_images(), fps, w, h, len(files)

    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se puede abrir la fuente de vídeo: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1

    def gen_video() -> Iterator[np.ndarray]:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        finally:
            cap.release()

    return gen_video(), fps, w, h, n


def _read_seqinfo_fps(seq_dir: Path) -> float | None:
    info = seq_dir / "seqinfo.ini"
    if not info.exists():
        return None
    for line in info.read_text().splitlines():
        if line.lower().startswith("framerate"):
            return float(line.split("=")[1])
    return None
