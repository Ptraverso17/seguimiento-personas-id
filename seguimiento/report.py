"""Estadísticas por persona y exportación a CSV / JSON / formato MOT."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class PersonStats:
    person_id: int
    first_frame: int
    last_frame: int
    frames_visible: int = 1
    mean_confidence: float = 0.0
    crossings: list[str] = field(default_factory=list)

    def duration_seconds(self, fps: float) -> float:
        return (self.last_frame - self.first_frame + 1) / fps if fps > 0 else 0.0


class Report:
    """Acumula lo que ocurre con cada ID a lo largo del vídeo."""

    def __init__(self, fps: float):
        self.fps = fps
        self.people: dict[int, PersonStats] = {}
        self.max_simultaneous = 0
        self.frames_processed = 0
        self._mot_rows: list[tuple] = []

    def add_frame(self, frame: int, detections: list[tuple[int, tuple[float, float, float, float], float]]) -> None:
        """``detections`` es una lista de (id, (x1, y1, x2, y2), confianza)."""
        self.frames_processed += 1
        self.max_simultaneous = max(self.max_simultaneous, len(detections))
        for pid, (x1, y1, x2, y2), conf in detections:
            stats = self.people.get(pid)
            if stats is None:
                self.people[pid] = PersonStats(pid, frame, frame, 1, conf)
            else:
                n = stats.frames_visible
                stats.mean_confidence = (stats.mean_confidence * n + conf) / (n + 1)
                stats.frames_visible = n + 1
                stats.last_frame = frame
            # Formato MOTChallenge: frame (desde 1), id, x, y, w, h, conf, -1, -1, -1
            self._mot_rows.append((frame + 1, pid, x1, y1, x2 - x1, y2 - y1, conf))

    def add_crossing(self, pid: int, direction: str) -> None:
        if pid in self.people:
            self.people[pid].crossings.append(direction)

    @property
    def unique_people(self) -> int:
        return len(self.people)

    def summary(self, entradas: int, salidas: int, min_frames: int = 1) -> dict:
        stable = [p for p in self.people.values() if p.frames_visible >= min_frames]
        dwell = [p.duration_seconds(self.fps) for p in stable]
        return {
            "frames_procesados": self.frames_processed,
            "fps": round(self.fps, 2),
            "personas_unicas": len(stable),
            "ids_totales_sin_filtrar": self.unique_people,
            "max_personas_simultaneas": self.max_simultaneous,
            "cruces_entrada": entradas,
            "cruces_salida": salidas,
            "tiempo_medio_en_escena_s": round(sum(dwell) / len(dwell), 2) if dwell else 0.0,
        }

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "id", "primer_frame", "ultimo_frame", "frames_visible",
                "segundos_en_escena", "confianza_media", "cruces",
            ])
            for p in sorted(self.people.values(), key=lambda s: s.person_id):
                writer.writerow([
                    p.person_id, p.first_frame, p.last_frame, p.frames_visible,
                    round(p.duration_seconds(self.fps), 2), round(p.mean_confidence, 3),
                    "|".join(p.crossings),
                ])

    def write_json(self, path: Path, summary: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"resumen": summary, "personas": [asdict(p) for p in self.people.values()]}
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def write_mot(self, path: Path) -> None:
        """Resultados en el formato de MOTChallenge, para evaluarlos con TrackEval."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for f, pid, x, y, w, h, conf in self._mot_rows:
                fh.write(f"{f},{pid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.3f},-1,-1,-1\n")
