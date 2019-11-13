import os
import pytest
import random
import numpy

from edgemaze import analyze


def empty(height, width=None, dtype=None):
    """Empty maze of shape"""
    width = width or height
    dtype = dtype or numpy.uint8
    return numpy.zeros((height, width), dtype=dtype)


def run(maze):
    """Analyze a maze and print debug"""
    print(maze)
    analyzed = analyze(maze)
    print(getattr(analyzed, 'distances', 'NO distances'))
    print(getattr(analyzed, 'directions', 'NO directions'))
    print(getattr(analyzed, 'is_reachable', 'NO is_reachable'))
    return analyzed


def lot(iterable):
    """Returns a list of tuples from a 2D iterable"""
    return [tuple(c) for c in iterable]


def b(text):
    """Returns a list of single-byte ASCII encoded characters"""
    return [u.encode('ascii') for u in text]


def test_too_low_dimension():
    with pytest.raises(TypeError) as excinfo:
        analyze(numpy.zeros((1,), dtype=int))
    assert 'dimension' in str(excinfo.value)


def test_too_many_dimensions():
    with pytest.raises(TypeError) as excinfo:
        analyze(numpy.zeros((1, 1, 1), dtype=int))
    assert 'dimension' in str(excinfo.value)


def test_float_dtype_error():
    with pytest.raises(TypeError) as excinfo:
        analyze(numpy.zeros((1, 1), dtype=float))
    assert (
        'int' in str(excinfo.value) or
        'type' in str(excinfo.value) or
        'float' in str(excinfo.value)
    )


def test_various_ints():
    for dtype in 'uint8', 'uint32', 'int16', 'int64':
        m = empty(1, dtype=dtype)
        a = run(m)
        # smoke test only
        assert m.shape == a.distances.shape
        assert m.shape == a.directions.shape


def test_tiny_void():
    m = empty(1)
    a = run(m)
    assert numpy.array_equal(a.distances, [[-1]])
    assert numpy.array_equal(a.directions, [[b' ']])
    assert not a.is_reachable
    with pytest.raises(ValueError) as excinfo:
        a.path(0, 0)
    assert 'reachable' in str(excinfo.value)


def test_tiny_target():
    # we'll learn later how to create parametrized tests
    for target in 1, 5, 7, 15, -1, -3, 9:
        m = empty(1, dtype='int8')
        m[0, 0] = target
        a = run(m)
        assert numpy.array_equal(a.distances, [[0]])
        assert numpy.array_equal(a.directions, [[b'X']])
        assert a.is_reachable
        assert lot(a.path(0, 0)) == [(0, 0)]


def test_middle_target():
    m = empty(1, 3)
    m[0, 1] = 1
    a = run(m)
    assert numpy.array_equal(a.distances, [[1, 0, 1]])
    assert numpy.array_equal(a.directions, [b('>X<')])
    assert a.is_reachable
    assert lot(a.path(0, 0)) == [(0, 0), (0, 1)]
    assert lot(a.path(0, 1)) == [(0, 1)]
    assert lot(a.path(0, 2)) == [(0, 2), (0, 1)]


def test_middle_target_sideways():
    m = empty(3, 1)
    m[1, 0] = 1
    a = run(m)
    assert numpy.array_equal(a.distances, [[1], [0], [1]])
    assert numpy.array_equal(a.directions, [[b'v'], [b'X'], [b'^']])
    assert a.is_reachable
    assert lot(a.path(0, 0)) == [(0, 0), (1, 0)]
    assert lot(a.path(1, 0)) == [(1, 0)]
    assert lot(a.path(2, 0)) == [(2, 0), (1, 0)]


def test_middle_target_squared():
    m = empty(3)
    m[1, 1] = 1
    a = run(m)
    assert numpy.array_equal(a.distances, [[2, 1, 2], [1, 0, 1], [2, 1, 2]])
    assert a.directions[0, 1] == b'v'
    assert a.directions[1, 1] == b'X'
    assert a.directions[2, 1] == b'^'
    assert a.directions[1, 0] == b'>'
    assert a.directions[1, 2] == b'<'

    assert a.directions[0, 0] in b('v>')
    assert a.directions[0, 2] in b('v<')
    assert a.directions[2, 0] in b('^>')
    assert a.directions[2, 2] in b('^<')

    assert a.is_reachable
    assert lot(a.path(1, 1)) == [(1, 1)]
    assert lot(a.path(0, 0)) in ([(0, 0), (1, 0), (1, 1)],
                                 [(0, 0), (0, 1), (1, 1)])
    # rest of the paths are left as an exercise for the reader


def test_long_row_target():
    m = empty(1, 128)
    m[0, 16] = 1
    a = run(m)
    assert a.is_reachable

    assert a.directions[0, 16] == b'X'
    assert numpy.all(a.directions[:, :16] == b'>')
    assert numpy.all(a.directions[:, 17:] == b'<')

    dists = [list(reversed(range(17))) + list(range(1, 112))]
    assert numpy.array_equal(a.distances, dists)


def test_long_column_target():
    m = empty(128, 1)
    m[16, 0] = 1
    a = run(m)
    assert a.is_reachable

    assert a.directions[16, 0] == b'X'
    assert numpy.all(a.directions[:16, :] == b'v')
    assert numpy.all(a.directions[17:, :] == b'^')

    dists = [[i] for i in list(reversed(range(17))) + list(range(1, 112))]
    assert numpy.array_equal(a.distances, dists)


def test_two_targets():
    m = empty(1, 4)
    m[0, 0] = 1
    m[0, -1] = 1
    a = run(m)
    assert a.is_reachable

    assert numpy.array_equal(a.directions, [b('X<>X')])
    assert numpy.array_equal(a.distances, [[0, 1, 1, 0]])


def test_left_wall_blocks():
    m = empty(1, 2)
    m[0, 1] = 1 | 2  # target and left wall
    a = run(m)
    assert not a.is_reachable

    assert numpy.array_equal(a.directions, [b(' X')])
    assert numpy.array_equal(a.distances, [[-1, 0]])


def test_right_wall_blocks():
    m = empty(1, 2)
    m[0, 0] = 1
    m[0, 1] = 2
    a = run(m)
    assert not a.is_reachable

    assert numpy.array_equal(a.directions, [b('X ')])
    assert numpy.array_equal(a.distances, [[0, -1]])


def test_above_wall_blocks():
    m = empty(2, 1)
    m[1, 0] = 1 | 4
    a = run(m)
    assert not a.is_reachable

    assert numpy.array_equal(a.directions, [[b' '], [b'X']])
    assert numpy.array_equal(a.distances, [[-1], [0]])


def test_below_wall_blocks():
    m = empty(2, 1)
    m[0, 0] = 1
    m[1, 0] = 4
    a = run(m)
    assert not a.is_reachable

    assert numpy.array_equal(a.directions, [[b'X'], [b' ']])
    assert numpy.array_equal(a.distances, [[0], [-1]])


def test_stripes_shape():
    m = numpy.array([[4, 4, 4, 5]] * 4)
    a = run(m)
    assert a.is_reachable
    assert numpy.array_equal(a.distances, [[3, 2, 1, 0]] * 4)
    assert numpy.array_equal(a.directions, [b('>>>X')] * 4)
    for row in range(4):
        assert lot(a.path(row, 0)) == [(row, 0), (row, 1), (row, 2), (row, 3)]
        assert lot(a.path(row, 1)) == [(row, 1), (row, 2), (row, 3)]
        assert lot(a.path(row, 2)) == [(row, 2), (row, 3)]
        assert lot(a.path(row, 3)) == [(row, 3)]


def test_unreachable_area():
    m = empty(4)
    m[0, 0] = 1
    m[2, :] = 4
    a = run(m)
    assert not a.is_reachable

    assert numpy.array_equal(a.distances[0, :], range(4))
    assert numpy.array_equal(a.directions[0, :], b('X<<<'))

    assert numpy.array_equal(a.distances[1, :], range(1, 5))
    assert a.directions[1, 0] == b'^'
    assert numpy.all((a.directions[1, 1:] == b'^') |
                     (a.directions[1, 1:] == b'<'))

    for row in 2, 3:
        assert numpy.array_equal(a.distances[row, :], [-1] * 4)
        assert numpy.array_equal(a.directions[row, :], [b' '] * 4)


def test_up_down():
    m = numpy.array([
        [1, 2],
        [8, 2],
        [8, 3],
    ])
    a = run(m)
    assert a.is_reachable

    dists = numpy.array([
        [0, 2],
        [1, 1],
        [2, 0],
    ])
    assert numpy.array_equal(a.distances, dists)

    dirs = numpy.array([
        b('Xv'),
        b('^v'),
        b('^X'),
    ])
    assert numpy.array_equal(a.directions, dirs)

    assert lot(a.path(2, 0)) == [(2, 0), (1, 0), (0, 0)]
    assert lot(a.path(0, 1)) == [(0, 1), (1, 1), (2, 1)]


def test_swirl_shape():
    m = numpy.array([
        [6, 4, 4, 4, 4, 4, 4],
        [2, 6, 4, 4, 4, 4, 2],
        [2, 2, 6, 4, 4, 2, 2],
        [2, 2, 2, 5, 2, 2, 2],
        [2, 2, 4, 4, 0, 2, 2],
        [2, 4, 4, 4, 4, 0, 2],
        [4, 4, 4, 4, 4, 4, 0],
    ])
    a = run(m)
    assert a.is_reachable

    dists = numpy.array([
        [30, 31, 32, 33, 34, 35, 36],
        [29, 12, 13, 14, 15, 16, 37],
        [28, 11,  2,  3,  4, 17, 38],
        [27, 10,  1,  0,  5, 18, 39],
        [26,  9,  8,  7,  6, 19, 40],
        [25, 24, 23, 22, 21, 20, 41],
        [48, 47, 46, 45, 44, 43, 42],
    ])
    assert numpy.array_equal(a.distances, dists)

    dirs = numpy.array([
        b('?<<<<<<'),
        b('v?<<<<^'),
        b('vv?<<^^'),
        b('vv>X^^^'),
        b('v>>>?^^'),
        b('>>>>>?^'),
        b('>>>>>>?'),
    ])
    assert numpy.all((dirs == b'?') | (a.directions == dirs))
    for i in range(3):
        assert a.directions[i, i] in b('v<')
    for i in range(4, 7):
        assert a.directions[i, i] in b('^>')

    path = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (5, 1), (5, 2),
            (5, 3), (5, 4), (5, 5), (4, 5), (3, 5), (2, 5), (1, 5), (1, 4),
            (1, 3), (1, 2), (1, 1), (2, 1), (3, 1), (4, 1), (4, 2), (4, 3),
            (4, 4), (3, 4), (2, 4), (2, 3), (2, 2), (3, 2), (3, 3)]
    while path:
        assert lot(a.path(*path[0])) == path
        del path[0]


@pytest.mark.timeout(20)
def test_large_maze_fast():
    big = int(os.getenv('BIG_ENOUGH_NUMBER', '2049'))
    for i in range(20):
        m = empty(big)
        m[0, 0] = 1
        m[-1, -1] = 1
        a = run(m)
        assert a.distances[-1, -1] == 0
        assert a.distances[0, -1] == big - 1
        assert a.distances[-1, 0] == big - 1
        assert len(lot(a.path(big - 1, 0))) == big
        print(i)


@pytest.mark.timeout(2)
def test_large_maze_fast_path():
    big = int(os.getenv('BIG_ENOUGH_NUMBER', '2049'))
    m = empty(big)
    m[0, 0] = 1
    a = run(m)
    assert a.distances[-1, -1] == big * 2 - 2
    assert a.distances[0, -1] == big - 1
    assert a.distances[-1, 0] == big - 1
    assert len(lot(a.path(big - 1, big - 1))) == big * 2 - 1
    for i in range(50):
        lot(a.path(random.randrange(big), random.randrange(big)))
        print(i)
