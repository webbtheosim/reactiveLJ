"""Independent NumPy energies for force checks and static energy landscapes."""
import numpy as np

RM = 2**(1/6)
RC = 1.5


def weight(r, inner=1.3, outer=1.5):
    t = np.clip((np.asarray(r)-inner)/(outer-inner), 0, 1)
    return (1+np.cos(np.pi*t))/2


def weakening(raw, p=4.0):
    excess = (raw+np.sqrt(raw*raw+1.e-12))/2
    return (1+excess)**(-p)


def shifted_lj(r, epsilon=1.0):
    r = np.asarray(r)
    return 4*epsilon*(r**(-12)-r**(-6)-RC**(-12)+RC**(-6))


def lj_derivative(r, epsilon=1.0):
    r = np.asarray(r)
    return 24*epsilon*(-2*r**(-13)+r**(-7))


def pair_energy(r, W, model, epsilon=1.0):
    r, W = np.broadcast_arrays(np.asarray(r, float), np.asarray(W, float))
    u = shifted_lj(r, epsilon)
    if model == 'ReactiveAH':
        depth = -shifted_lj(RM, epsilon)
        energy = np.where(r <= RM, u+(1-W)*depth, W*u)
    elif model == 'ReactiveLJ':
        energy = np.maximum(u, 0)+W*np.minimum(u, 0)
        shift = 4*(RC**(-12)-RC**(-6))
        elbow = ((1+np.sqrt(1+shift))/2)**(-1/6)
        delta = .05*(1-W)
        lo, hi = elbow-delta, elbow+delta
        inside = (delta > 1.e-6) & (r > lo) & (r < hi)
        # Clip a vanishing width only for evaluation of the unused branch.
        width = np.maximum(2*delta, 1.e-20)
        t = np.clip((r-lo)/width, 0, 1)
        h00, h10 = 2*t**3-3*t*t+1, t**3-2*t*t+t
        h01, h11 = -2*t**3+3*t*t, t**3-t*t
        hermite = (h00*shifted_lj(lo, epsilon)+h10*width*lj_derivative(lo, epsilon)
                   +h01*W*shifted_lj(hi, epsilon)+h11*width*W*lj_derivative(hi, epsilon))
        energy = np.where(inside, hermite, energy)
    else:
        raise ValueError(model)
    return np.where(r < RC, energy, 0.)


def total_energy(positions, model='ReactiveAH', epsilon=1., p=4., box=12., types=None):
    positions = np.asarray(positions, float)
    if types is not None:
        positions = positions[np.asarray(types) == 0]
    delta = positions[:, None]-positions[None, :]
    if box is not None:
        delta -= np.rint(delta/np.asarray(box))*np.asarray(box)
    distances = np.linalg.norm(delta, axis=-1)
    np.fill_diagonal(distances, np.inf)
    w = weight(distances)
    c = w.sum(axis=1)
    i, j = np.triu_indices(len(positions), 1)
    r = distances[i, j]
    W = weakening(c[i]+c[j]-2*w[i, j], p)
    return float(pair_energy(r, W, model, epsilon).sum())
