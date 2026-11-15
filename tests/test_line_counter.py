import pytest

from seguimiento.line_counter import LineCounter, parse_line, segments_intersect


def horizontal_line():
    # Línea horizontal en y=100, de x=0 a x=200. "Entrada" = moverse hacia abajo.
    return LineCounter((0, 100), (200, 100))


def test_segments_intersect():
    assert segments_intersect((50, 90), (50, 110), (0, 100), (200, 100))
    assert not segments_intersect((50, 80), (50, 95), (0, 100), (200, 100))
    # Cruza la recta pero fuera del segmento
    assert not segments_intersect((300, 90), (300, 110), (0, 100), (200, 100))


def test_count_entrada_y_salida():
    c = horizontal_line()
    assert c.update(1, (50, 90)) is None  # primera vez que lo vemos
    assert c.update(1, (50, 110)) == "entrada"
    assert c.update(2, (120, 130)) is None
    assert c.update(2, (120, 70)) == "salida"
    assert (c.entradas, c.salidas, c.total) == (1, 1, 2)


def test_no_cuenta_dos_veces_el_mismo_sentido():
    c = horizontal_line()
    c.update(1, (50, 90))
    c.update(1, (50, 110))  # entra
    c.update(1, (50, 90))   # sale
    c.update(1, (50, 110))  # vuelve a entrar: no suma
    assert (c.entradas, c.salidas) == (1, 1)


def test_cuenta_repetidos_si_se_desactiva():
    c = LineCounter((0, 100), (200, 100), count_once_per_direction=False)
    for y in (90, 110, 90, 110):
        c.update(1, (50, y))
    assert (c.entradas, c.salidas) == (2, 1)


def test_movimiento_sin_cruzar():
    c = horizontal_line()
    for y in (10, 30, 50, 70, 90):
        assert c.update(7, (50, y)) is None
    assert c.total == 0


def test_cruce_fuera_del_segmento_no_cuenta():
    c = horizontal_line()
    c.update(1, (500, 90))
    assert c.update(1, (500, 110)) is None


def test_forget_reinicia_posicion():
    c = horizontal_line()
    c.update(1, (50, 90))
    c.forget(1)
    assert c.update(1, (50, 110)) is None


def test_parse_line_relativa_y_absoluta():
    assert parse_line("0,0.5,1,0.5", 1920, 1080) == ((0, 540), (1920, 540))
    assert parse_line("10,20,300,20", 1920, 1080) == ((10, 20), (300, 20))


@pytest.mark.parametrize("spec", ["1,2,3", "a,b,c,d", "5,5,5,5"])
def test_parse_line_invalida(spec):
    with pytest.raises(ValueError):
        parse_line(spec, 100, 100)
