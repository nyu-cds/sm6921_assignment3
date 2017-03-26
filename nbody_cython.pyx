"""
    Author: Saurabh Mahajan
    Netid: sm6921
    Execution Time: 33.57s
    Relative speedup = (113.92/31.159) = 3.65

    For this assignment, all the local variables and parameters were assigned the appropriate data type. 
    Run time = 6.78s
"""
import cython
from itertools import combinations
cpdef advance(dict BODIES, list combination, int iterations, float dt, list values):
    cdef str body1, body2, v1, v2, r
    cdef float x1, y1, z1, m1, x2, y2, z2, m2
    cdef float dx, dy, dz, vx, vy, vz, m, tmp, m2_tmp, m1_tmp
    for _ in range(iterations):
        '''
            advance by one time step
        '''
        for body1, body2 in combination:
            ([x1, y1, z1], v1, m1) = BODIES[body1]
            ([x2, y2, z2], v2, m2) = BODIES[body2]
            dx = x1-x2
            dy = y1-y2
            dz = z1-z2
            tmp = dt * ((dx * dx + dy * dy + dz * dz) ** (-1.5))
            m2_tmp = m2 * tmp
            m1_tmp = m1 * tmp
            v1[0] -= dx * m2_tmp
            v1[1] -= dy * m2_tmp
            v1[2] -= dz * m2_tmp
            v2[0] += dx * m1_tmp
            v2[1] += dy * m1_tmp
            v2[2] += dz * m1_tmp
            
        for body1 in values:
            (r, [vx, vy, vz], m) = body1
            r[0] += dt * vx
            r[1] += dt * vy
            r[2] += dt * vz
    
cpdef float report_energy(dict BODIES, list combination, list values, float e=0.0):
    '''
        compute the energy and return it so that it can be printed
    '''
    cdef str body1, body2, v1, v2, r
    cdef float x1, y1, z1, m1, x2, y2, z2, m2, dx, dy, dz, vx, vy, vz
    for body1, body2 in combination:
        ((x1, y1, z1), v1, m1) = BODIES[body1]
        ((x2, y2, z2), v2, m2) = BODIES[body2]
        dx = x1-x2
        dy = y1-y2
        dz = z1-z2
        e -= (m1 * m2) / ((dx * dx + dy * dy + dz * dz) ** 0.5)
        
    for body1 in values:
        (r, [vx, vy, vz], m) = body1
        e += m * (vx * vx + vy * vy + vz * vz) / 2.
        
    return e

cpdef offset_momentum(str ref, list values, float px=0.0, float py=0.0, float pz=0.0):
    '''
        ref is the body in the center of the system
        offset values from this reference
    '''
    cdef list body, r, v
    cdef float vx, vy, vz, m
    for body in values:
        (r, [vx, vy, vz], m) = body
        px -= vx * m
        py -= vy * m
        pz -= vz * m
        
    (r, v, m) = ref
    v[0] = px / m
    v[1] = py / m
    v[2] = pz / m


cpdef nbody(int loops, str reference,int iterations):
    '''
        nbody simulation
        loops - number of loops to run
        reference - body at center of system
        iterations - number of timesteps to advance
    '''
    cdef float PI = 3.14159265358979323
    cdef float SOLAR_MASS = 4 * PI * PI
    cdef float DAYS_PER_YEAR = 365.24

    cdef dict BODIES = {
        'sun': ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], SOLAR_MASS),

        'jupiter': ([4.84143144246472090e+00,
                     -1.16032004402742839e+00,
                     -1.03622044471123109e-01],
                    [1.66007664274403694e-03 * DAYS_PER_YEAR,
                     7.69901118419740425e-03 * DAYS_PER_YEAR,
                     -6.90460016972063023e-05 * DAYS_PER_YEAR],
                    9.54791938424326609e-04 * SOLAR_MASS),

        'saturn': ([8.34336671824457987e+00,
                    4.12479856412430479e+00,
                    -4.03523417114321381e-01],
                   [-2.76742510726862411e-03 * DAYS_PER_YEAR,
                    4.99852801234917238e-03 * DAYS_PER_YEAR,
                    2.30417297573763929e-05 * DAYS_PER_YEAR],
                   2.85885980666130812e-04 * SOLAR_MASS),

        'uranus': ([1.28943695621391310e+01,
                    -1.51111514016986312e+01,
                    -2.23307578892655734e-01],
                   [2.96460137564761618e-03 * DAYS_PER_YEAR,
                    2.37847173959480950e-03 * DAYS_PER_YEAR,
                    -2.96589568540237556e-05 * DAYS_PER_YEAR],
                   4.36624404335156298e-05 * SOLAR_MASS),

        'neptune': ([1.53796971148509165e+01,
                     -2.59193146099879641e+01,
                     1.79258772950371181e-01],
                    [2.68067772490389322e-03 * DAYS_PER_YEAR,
                     1.62824170038242295e-03 * DAYS_PER_YEAR,
                     -9.51592254519715870e-05 * DAYS_PER_YEAR],
                    5.15138902046611451e-05 * SOLAR_MASS)}

    # Set up global state
    cdef list values = list(BODIES.values())
    offset_momentum(BODIES[reference], values)
    cdef list combination = list(combinations(BODIES, 2))
    for _ in range(loops):
        report_energy(BODIES, combination[:], values)
        advance(BODIES, combination[:], iterations, 0.01, values)
        print(report_energy(BODIES, combination[:], values))

if __name__ == '__main__':
    import timeit
    print(timeit.timeit("nbody(100, 'sun', 20000)", setup="from __main__ import nbody", number=1))