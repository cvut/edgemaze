import queue

import numpy


class HitError(ValueError):
    pass


class WallHitError(HitError):
    pass


class BorderHitError(HitError):
    pass


IS_TARGET = 1
WALL_LEFT = 2
WALL_UP = 4


def up(maze, loc, check_walls=True):
    if loc[0] == 0:
        raise BorderHitError
    if check_walls and maze[loc] & WALL_UP:
        raise WallHitError
    return loc[0] - 1, loc[1]


def down(maze, loc, check_walls=True):
    if loc[0] == (maze.shape[0] - 1):
        raise BorderHitError
    newloc = loc[0] + 1, loc[1]
    if check_walls and maze[newloc] & WALL_UP:
        raise WallHitError
    return newloc


def left(maze, loc, check_walls=True):
    if loc[1] == 0:
        raise BorderHitError
    if check_walls and maze[loc] & WALL_LEFT:
        raise WallHitError
    return loc[0], loc[1] - 1


def right(maze, loc, check_walls=True):
    if loc[1] == (maze.shape[1] - 1):
        raise BorderHitError
    newloc = loc[0], loc[1] + 1
    if check_walls and maze[newloc] & WALL_LEFT:
        raise WallHitError
    return newloc


def ends(maze):
    return numpy.asarray(numpy.where(maze & IS_TARGET)).T


DIRS = {
    b'^': up,
    b'<': left,
    b'>': right,
    b'v': down,
}

ANTIDIRS = {
    down: b'^',
    right: b'<',
    left: b'>',
    up: b'v'
}

UNREACHABLE = b' '
TARGET = b'X'


def arrows_to_path(arrows, loc):
    if arrows[loc] == UNREACHABLE:
        raise ValueError('Cannot construct path for unreachable cell')
    path = [loc]

    nloc = loc
    while arrows[nloc] != TARGET:
        nloc = DIRS[arrows[nloc]](arrows, nloc, check_walls=False)
        path.append(nloc)

    return path


def smallest_dtype(value):
    for dtype in numpy.int8, numpy.int16, numpy.int32, numpy.int64:
        if dtype(value) == value:
            return dtype
    raise ValueError(f'Maze of size {value} is too big for NumPy to handle')


def flood(maze):
    if maze.ndim != 2 or not numpy.issubdtype(maze.dtype, numpy.integer):
        raise TypeError('maze must be a 2-dimensional array of integers')

    # Initialize everything as unreachable
    dtype = smallest_dtype(maze.size)
    distances = numpy.full(maze.shape, -1, dtype=dtype)
    directions = numpy.full(maze.shape, UNREACHABLE, dtype=('a', 1))

    jobs = queue.Queue()
    for end in ends(maze):
        end = tuple(end)
        directions[end] = TARGET
        distances[end] = 0
        jobs.put((end, 1))

    while not jobs.empty():
        loc, dist = jobs.get()
        for walk in [up, left, right, down]:
            try:
                newloc = walk(maze, loc)
            except HitError:
                pass
            else:
                # Been there better
                if 0 <= distances[newloc] <= dist:
                    continue
                distances[newloc] = dist
                directions[newloc] = ANTIDIRS[walk]
                jobs.put((newloc, dist+1))

    return distances, directions


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
