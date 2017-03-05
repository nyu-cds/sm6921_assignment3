# -----------------------------------------------------------------------------
# calculator.py
# ----------------------------------------------------------------------------- 
# relative speedup = 2.45/0.038 = 64.47
import numpy as np

def hypotenuse(x,y):
    """
    Return sqrt(x**2 + y**2) for two arrays, a and b.
    x and y must be two-dimensional arrays of the same shape.
    """
    xx = np.multiply(x,x)
    yy = np.multiply(y,y)
    zz = np.add(xx, yy)
    return np.sqrt(zz)