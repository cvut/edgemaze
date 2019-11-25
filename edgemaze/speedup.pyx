import numpy

cimport cython
cimport numpy
from cpython.mem cimport PyMem_Malloc, PyMem_Free

cdef numpy.int8_t IS_TARGET = 1
cdef numpy.int8_t WALL_LEFT = 2
cdef numpy.int8_t WALL_UP = 4


cdef struct job:
    int r
    int c
    numpy.int64_t dist


# or use a deque from C++
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef class JobQueue:
    cdef job * jobs
    cdef int top, bottom, size

    def __cinit__(self, int size):
        self.jobs = <job *>PyMem_Malloc(size*sizeof(job))
        if self.jobs == NULL:
            raise MemoryError()
        self.top = 0
        self.bottom = 0
        self.size = size

    def __dealloc__(self):
        if self.jobs != NULL:
            PyMem_Free(self.jobs)

    cdef void put(self, job j):
        self.jobs[self.top % self.size] = j
        self.top += 1

    cdef job get(self):
        self.bottom += 1
        return self.jobs[(self.bottom-1) % self.size]

    cdef bint empty(self):
        return self.bottom == self.top


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def flood(numpy.ndarray[numpy.int8_t, ndim=2] maze):
    cdef numpy.ndarray[numpy.int64_t, ndim=2] distances
    cdef numpy.ndarray[char, ndim=2] directions
    cdef job j
    cdef int r, c

    # Initialize everything as unreachable
    distances = numpy.full((maze.shape[0], maze.shape[1]), -1, dtype=numpy.int64)
    directions = numpy.full((maze.shape[0], maze.shape[1]), b' ', dtype=('a', 1))

    cdef JobQueue jobs = JobQueue(maze.shape[0] * maze.shape[1])
    for end in numpy.asarray(numpy.where(maze & IS_TARGET)).T:
        directions[end[0], end[1]] = b'X'
        distances[end[0], end[1]] = 0
        j = job(end[0], end[1], 1)
        jobs.put(j)

    while not jobs.empty():
        j = jobs.get()

        # LEFT
        if j.c > 0 and not (maze[j.r, j.c] & WALL_LEFT):
            r = j.r
            c = j.c - 1
            if not (0 <= distances[r, c] <= j.dist):
                distances[r, c] = j.dist
                directions[r, c] = b'>'
                jobs.put(job(r, c, j.dist+1))
        # RIGHT
        if j.c+1 < maze.shape[1] and not (maze[j.r, j.c+1] & WALL_LEFT):
            r = j.r
            c = j.c + 1
            if not (0 <= distances[r, c] <= j.dist):
                distances[r, c] = j.dist
                directions[r, c] = b'<'
                jobs.put(job(r, c, j.dist+1))
        # UP
        if j.r > 0 and not (maze[j.r, j.c] & WALL_UP):
            r = j.r - 1
            c = j.c
            if not (0 <= distances[r, c] <= j.dist):
                distances[r, c] = j.dist
                directions[r, c] = b'v'
                jobs.put(job(r, c, j.dist+1))
        # DOWN
        if j.r+1 < maze.shape[0] and not (maze[j.r+1, j.c] & WALL_UP):
            r = j.r + 1
            c = j.c
            if not (0 <= distances[r, c] <= j.dist):
                distances[r, c] = j.dist
                directions[r, c] = b'^'
                jobs.put(job(r, c, j.dist+1))

    return distances, directions
