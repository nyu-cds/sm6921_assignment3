"""
    Added jit decorators and used numba function signatures.
    Added vectorized function vec_deltas to compute deltas in advance and report_energy
"""

from itertools import combinations
from numba import jit, int64, float32, float64, vectorize
import numpy as np
@vectorize([float64(float64, float64)])
def vec_deltas(x, y):
    return x - y

@jit('void(float64[:,:,:], char[:,:], int32, float64, float64[:,:,:])')
def advance(BODIES, combination, iterations, dt, values):
    for _ in range(iterations):
        '''
            advance by one time step
        '''
        for body1, body2 in combination:
            (u1, v1, m1) = BODIES[body1]
            (u2, v2, m2) = BODIES[body2]
            (dx, dy , dz) = vec_deltas(u1, u2)
            tmp = dt * ((dx * dx + dy * dy + dz * dz) ** (-1.5))
            m2_tmp = m2 * tmp
            m1_tmp = m1 * tmp
            v1[0] -= dx * m2_tmp
            v1[1] -= dy * m2_tmp
            v1[2] -= dz * m2_tmp
            v2[0] += dx * m1_tmp
            v2[1] += dy * m1_tmp
            v2[2] += dz * m1_tmp
            
        for body in values:
            (r, [vx, vy, vz], m) = body
            r[0] += dt * vx
            r[1] += dt * vy
            r[2] += dt * vz
    
@jit('void(float64[:,:,:], char[:,:], float64[:,:,:], float64)')
def report_energy(BODIES, combination, values, e=0.0):
    '''
        compute the energy and return it so that it can be printed
    '''
    for body1, body2 in combination:
        (u1, v1, m1) = BODIES[body1]
        (u2, v2, m2) = BODIES[body2]
        (dx, dy , dz) = vec_deltas(u1, u2)
        e -= (m1 * m2) / ((dx * dx + dy * dy + dz * dz) ** 0.5)
        
    for body in values:
        (r, [vx, vy, vz], m) = body
        e += m * (vx * vx + vy * vy + vz * vz) / 2.
        
    return e

@jit('void(int32, float64[:,:,:], float64, float64, float64)')
def offset_momentum(ref, values, px=0.0, py=0.0, pz=0.0):
    '''
        ref is the body in the center of the system
        offset values from this reference
    '''
    for body in values:
        (r, [vx, vy, vz], m) = body
        px -= vx * m
        py -= vy * m
        pz -= vz * m
        
    (r, v, m) = ref
    v[0] = px / m
    v[1] = py / m
    v[2] = pz / m


@jit('void(int32, char, int32)')
def nbody(loops, reference, iterations):
    '''
        nbody simulation
        loops - number of loops to run
        reference - body at center of system
        iterations - number of timesteps to advance
    '''
    PI = 3.14159265358979323
    SOLAR_MASS = 4 * PI * PI
    DAYS_PER_YEAR = 365.24

    BODIES = {
        'sun': (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), SOLAR_MASS),

        'jupiter': (np.array([4.84143144246472090e+00,
                     -1.16032004402742839e+00,
                     -1.03622044471123109e-01]),
                    np.array([1.66007664274403694e-03 * DAYS_PER_YEAR,
                     7.69901118419740425e-03 * DAYS_PER_YEAR,
                     -6.90460016972063023e-05 * DAYS_PER_YEAR]),
                    9.54791938424326609e-04 * SOLAR_MASS),

        'saturn': (np.array([8.34336671824457987e+00,
                    4.12479856412430479e+00,
                    -4.03523417114321381e-01]),
                   np.array([-2.76742510726862411e-03 * DAYS_PER_YEAR,
                    4.99852801234917238e-03 * DAYS_PER_YEAR,
                    2.30417297573763929e-05 * DAYS_PER_YEAR]),
                   2.85885980666130812e-04 * SOLAR_MASS),

        'uranus': (np.array([1.28943695621391310e+01,
                    -1.51111514016986312e+01,
                    -2.23307578892655734e-01]),
                   np.array([2.96460137564761618e-03 * DAYS_PER_YEAR,
                    2.37847173959480950e-03 * DAYS_PER_YEAR,
                    -2.96589568540237556e-05 * DAYS_PER_YEAR]),
                   4.36624404335156298e-05 * SOLAR_MASS),

        'neptune': (np.array([1.53796971148509165e+01,
                     -2.59193146099879641e+01,
                     1.79258772950371181e-01]),
                    np.array([2.68067772490389322e-03 * DAYS_PER_YEAR,
                     1.62824170038242295e-03 * DAYS_PER_YEAR,
                     -9.51592254519715870e-05 * DAYS_PER_YEAR]),
                    5.15138902046611451e-05 * SOLAR_MASS)}

    # Set up global state
    values = BODIES.values()
    offset_momentum(BODIES[reference], values)
    combination = list(combinations(BODIES, 2))
    for _ in range(loops):
        report_energy(BODIES, combination[:], values)
        advance(BODIES, combination[:], iterations, 0.01, values)
        print(report_energy(BODIES, combination[:], values))

if __name__ == '__main__':
    import time
    t1 = time.time()
    nbody(100, 'sun', 20000)
    t2= time.time()
    print('Time:', (t2 - t1))