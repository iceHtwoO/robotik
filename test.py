import numpy as np


def rad_to_degre(rad):
    return (rad / np.pi) * 180

angle = np.arctan2([0,-1], [0,1])
print(angle)
print(rad_to_degre(angle[1]))