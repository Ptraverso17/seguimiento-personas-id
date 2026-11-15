"""Conteo de cruces de una línea virtual por parte de objetos con ID.

Este módulo no depende de OpenCV ni de YOLO, así que se puede testear
de forma aislada. Recibe, frame a frame, el punto de referencia de cada
persona (por defecto el centro de los pies) y decide si ha cruzado la línea.
"""

from __future__ import annotations

from dataclasses import dataclass, field

Point = tuple[float, float]


def _cross(o: Point, a: Point, b: Point) -> float:
    """Producto vectorial (a - o) x (b - o). El signo indica el lado."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """True si el segmento p1-p2 corta al segmento q1-q2 (sin contar solo tocarse)."""
    d1 = _cross(q1, q2, p1)
    d2 = _cross(q1, q2, p2)
    d3 = _cross(p1, p2, q1)
    d4 = _cross(p1, p2, q2)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


@dataclass
class Crossing:
    track_id: int
    frame: int
    direction: str  # "entrada" o "salida"


@dataclass
class LineCounter:
    """Cuenta cuántos IDs cruzan el segmento ``start``-``end`` y en qué sentido.

    El sentido "entrada" es el que va del lado izquierdo al derecho de la
    línea mirando de ``start`` a ``end`` (en coordenadas de imagen, con el
    eje Y hacia abajo). Si te sale al revés, intercambia los dos puntos.

    Cada ID se cuenta como mucho una vez por sentido, para que una persona
    que se queda parada encima de la línea no sume decenas de cruces.
    """

    start: Point
    end: Point
    count_once_per_direction: bool = True
    entradas: int = 0
    salidas: int = 0
    crossings: list[Crossing] = field(default_factory=list)
    _last_point: dict[int, Point] = field(default_factory=dict, repr=False)
    _counted: dict[int, set[str]] = field(default_factory=dict, repr=False)

    @property
    def total(self) -> int:
        return self.entradas + self.salidas

    def side(self, point: Point) -> float:
        """>0 a un lado de la línea, <0 al otro, 0 encima."""
        return _cross(self.start, self.end, point)

    def update(self, track_id: int, point: Point, frame: int = 0) -> str | None:
        """Registra la nueva posición de ``track_id``.

        Devuelve "entrada" o "salida" si en este paso ha cruzado la línea,
        o ``None`` si no.
        """
        previous = self._last_point.get(track_id)
        self._last_point[track_id] = point
        if previous is None:
            return None
        if not segments_intersect(previous, point, self.start, self.end):
            return None

        # Antes estaba en el lado negativo y ahora en el positivo -> entrada.
        direction = "entrada" if self.side(point) > 0 else "salida"
        done = self._counted.setdefault(track_id, set())
        if self.count_once_per_direction and direction in done:
            return None
        done.add(direction)

        if direction == "entrada":
            self.entradas += 1
        else:
            self.salidas += 1
        self.crossings.append(Crossing(track_id, frame, direction))
        return direction

    def forget(self, track_id: int) -> None:
        """Olvida la última posición de un ID (por ejemplo, si se pierde)."""
        self._last_point.pop(track_id, None)


def parse_line(spec: str, width: int, height: int) -> tuple[Point, Point]:
    """Convierte "x1,y1,x2,y2" en dos puntos en píxeles.

    Si todos los valores están entre 0 y 1 se interpretan como fracciones
    del ancho y alto del vídeo, para que la misma línea sirva a cualquier
    resolución. Por ejemplo "0,0.5,1,0.5" es una línea horizontal en medio.
    """
    try:
        values = [float(v) for v in spec.split(",")]
    except ValueError as exc:
        raise ValueError(f"Línea no válida: {spec!r}") from exc
    if len(values) != 4:
        raise ValueError(f"La línea necesita 4 valores x1,y1,x2,y2 y tiene {len(values)}")
    if all(0.0 <= v <= 1.0 for v in values):
        values = [values[0] * width, values[1] * height, values[2] * width, values[3] * height]
    x1, y1, x2, y2 = values
    if (x1, y1) == (x2, y2):
        raise ValueError("Los dos puntos de la línea no pueden ser iguales")
    return (x1, y1), (x2, y2)
