"""Punto de entrada: python -m seguimiento --source video.mp4"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from .line_counter import LineCounter, parse_line
from .report import Report
from .tracker import PeopleTracker, open_source
from .visualizer import Visualizer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m seguimiento",
        description="Cuenta las personas que pasan en un vídeo y asigna un ID fijo a cada una.",
    )
    p.add_argument("--source", required=True,
                   help="Vídeo, carpeta de imágenes (p. ej. una secuencia de MOT17) o '0' para la webcam")
    p.add_argument("--output", default="resultados", help="Carpeta de salida (por defecto: resultados/)")
    p.add_argument("--model", default="yolo11n.pt",
                   help="Pesos de YOLO. n = rápido, s/m = más preciso (se descargan solos)")
    p.add_argument("--tracker", default="bytetrack", choices=["bytetrack", "botsort"],
                   help="Algoritmo de seguimiento")
    p.add_argument("--line", default="0,0.5,1,0.5",
                   help="Línea de conteo x1,y1,x2,y2 en píxeles o en fracciones 0-1 "
                        "(por defecto, horizontal a media altura). 'none' para desactivarla")
    p.add_argument("--anchor", default="feet", choices=["feet", "center"],
                   help="Punto de la persona que debe cruzar la línea")
    p.add_argument("--conf", type=float, default=0.3, help="Confianza mínima de detección")
    p.add_argument("--imgsz", type=int, default=960,
                   help="Tamaño de entrada del modelo. Más grande = detecta mejor a personas lejanas, pero va más lento")
    p.add_argument("--min-frames", type=int, default=5,
                   help="Frames mínimos para contar un ID como persona real (filtra falsos positivos)")
    p.add_argument("--device", default=None, help="'cpu', '0' (GPU), 'mps'... Por defecto, automático")
    p.add_argument("--max-frames", type=int, default=None, help="Procesar solo los N primeros frames")
    p.add_argument("--show", action="store_true", help="Mostrar el vídeo en una ventana mientras se procesa")
    p.add_argument("--no-video", action="store_true", help="No guardar el vídeo anotado")
    p.add_argument("--no-trails", action="store_true", help="No dibujar la trayectoria de cada persona")
    p.add_argument("--mot", action="store_true",
                   help="Guardar también los resultados en formato MOTChallenge (para TrackEval)")
    return p


def run(args: argparse.Namespace) -> dict:
    frames, fps, width, height, total = open_source(args.source)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker = PeopleTracker(args.model, args.tracker, args.conf, imgsz=args.imgsz, device=args.device)
    counter = None
    if args.line.lower() != "none":
        start, end = parse_line(args.line, width, height)
        counter = LineCounter(start, end)
    report = Report(fps)
    viz = Visualizer(show_trails=not args.no_trails)

    writer = None
    if not args.no_video:
        writer = cv2.VideoWriter(str(out_dir / "video_anotado.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    t0 = time.perf_counter()
    n_frames = 0
    for i, frame in enumerate(frames):
        if args.max_frames is not None and i >= args.max_frames:
            break
        people = tracker.update(frame)
        report.add_frame(i, [(p.person_id, p.box, p.confidence) for p in people])

        flash: set[int] = set()
        if counter is not None:
            for p in people:
                point = p.feet if args.anchor == "feet" else p.center
                direction = counter.update(p.person_id, point, frame=i)
                if direction:
                    report.add_crossing(p.person_id, direction)
                    flash.add(p.person_id)

        stats = {"En escena": len(people), "IDs asignados": report.unique_people}
        if counter is not None:
            stats["Entradas"] = counter.entradas
            stats["Salidas"] = counter.salidas
        annotated = viz.draw(frame, people, (counter.start, counter.end) if counter else None, stats, flash)

        if writer is not None:
            writer.write(annotated)
        if args.show:
            cv2.imshow("Seguimiento de personas (q para salir)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        n_frames += 1
        if n_frames % 50 == 0:
            done = f"{n_frames}/{total}" if total > 0 else str(n_frames)
            print(f"  frame {done}  ·  {n_frames / (time.perf_counter() - t0):.1f} FPS", flush=True)

    elapsed = time.perf_counter() - t0
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    summary = report.summary(
        counter.entradas if counter else 0,
        counter.salidas if counter else 0,
        min_frames=args.min_frames,
    )
    summary["fps_procesamiento"] = round(n_frames / elapsed, 1) if elapsed > 0 else 0.0
    report.write_csv(out_dir / "personas.csv")
    report.write_json(out_dir / "resumen.json", summary)
    if args.mot:
        report.write_mot(out_dir / "mot_resultados.txt")
    return summary


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = run(args)
    print("\nResumen")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nResultados guardados en: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
