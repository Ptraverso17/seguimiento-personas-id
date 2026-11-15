import csv
import json

from seguimiento.report import Report


def make_report():
    r = Report(fps=10)
    r.add_frame(0, [(1, (0, 0, 10, 20), 0.9)])
    r.add_frame(1, [(1, (1, 0, 11, 20), 0.7), (2, (50, 50, 60, 70), 0.5)])
    r.add_frame(2, [(1, (2, 0, 12, 20), 0.8)])
    r.add_crossing(1, "entrada")
    return r


def test_estadisticas_por_persona():
    r = make_report()
    p1 = r.people[1]
    assert (p1.first_frame, p1.last_frame, p1.frames_visible) == (0, 2, 3)
    assert abs(p1.mean_confidence - 0.8) < 1e-9
    assert p1.duration_seconds(10) == 0.3
    assert p1.crossings == ["entrada"]
    assert r.max_simultaneous == 2


def test_resumen_filtra_ids_cortos():
    r = make_report()
    s = r.summary(entradas=1, salidas=0, min_frames=2)
    assert s["personas_unicas"] == 1
    assert s["ids_totales_sin_filtrar"] == 2
    assert s["cruces_entrada"] == 1


def test_exportaciones(tmp_path):
    r = make_report()
    r.write_csv(tmp_path / "p.csv")
    r.write_json(tmp_path / "r.json", r.summary(1, 0))
    r.write_mot(tmp_path / "m.txt")

    rows = list(csv.DictReader((tmp_path / "p.csv").open(encoding="utf-8")))
    assert [row["id"] for row in rows] == ["1", "2"]
    data = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert data["resumen"]["personas_unicas"] == 2
    mot = (tmp_path / "m.txt").read_text().splitlines()
    assert len(mot) == 4
    assert mot[0].startswith("1,1,0.00,0.00,10.00,20.00")
