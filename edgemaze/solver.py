import numpy

from . import speedup

DIRS = {
    b'<': lambda l: (l[0], l[1] - 1),
    b'>': lambda l: (l[0], l[1] + 1),
    b'^': lambda l: (l[0] - 1, l[1]),
    b'v': lambda l: (l[0] + 1, l[1]),
}

UNREACHABLE = b' '
TARGET = b'X'


def arrows_to_path(arrows, loc):
    if arrows[loc] == UNREACHABLE:
        raise ValueError('Cannot construct path for unreachable cell')
    path = [loc]

    nloc = loc
    while arrows[nloc] != TARGET:
        nloc = DIRS[arrows[nloc]](nloc)
        path.append(nloc)

    return path


def flood(maze):
    if maze.ndim != 2 or not numpy.issubdtype(maze.dtype, numpy.integer):
        raise TypeError('maze must be a 2-dimensional array of integers')
    return speedup.flood(maze.astype(numpy.int8, copy=False))


def is_reachable(directions):
    return UNREACHABLE not in directions


class AnalyzedMaze:
    def __init__(self, maze):
        self.maze = maze
        self.distances, self.directions = flood(maze)
        self.is_reachable = is_reachable(self.directions)

    def path(self, column, row):
        return arrows_to_path(self.directions, (column, row))


def analyze(maze):
    return AnalyzedMaze(maze)
