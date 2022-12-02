import numpy as np
import matplotlib.pyplot as plt
from numerical_solvers_and_models import *


def table_of_rbf_funs(shandle):
    dict_of_funs = {'multiquadric': lambda r, ep: np.sqrt((r / ep) ** 2. + 1.),
                    'inverse': lambda r, ep: 1.0 / np.sqrt((r / ep) ** 2. + 1.),
                    'gaussian': lambda r, ep: np.exp(-(r / ep) ** 2.),
                    'linear': lambda r, ep: r,
                    'cubic': lambda r, ep: r ** 3.,
                    'quintic': lambda r, ep: r ** 5.,
                    'thin_plate': lambda r, ep: r ** 2. * np.ma.log(r)}
    return dict_of_funs[shandle]


def table_of_rbf_jac_funs(shandle):
    dict_of_funs = {'multiquadric': lambda r, ep: (r / ep ** 2.) / np.sqrt((r / ep) ** 2. + 1.),
                    'inverse': lambda r, ep: -(r / ep ** 2.) / ((r / ep) ** 2. + 1.) ** (1.5),
                    'gaussian': lambda r, ep: -2. * r / ep ** 2. * np.exp(-(r / ep) ** 2.),
                    'linear': lambda r, ep: 1.,
                    'cubic': lambda r, ep: 3. * r ** 2.,
                    'quintic': lambda r, ep: 5. * r ** 4.,
                    'thin_plate': lambda r, ep: r * (2. * np.ma.log(r) + 1.)}
    return dict_of_funs[shandle]


def distance_matrix(ipts):
    nfvls = np.shape(ipts)[0]
    xcoords = np.zeros((nfvls, 1), dtype=np.float64)
    ycoords = np.zeros((nfvls, 1), dtype=np.float64)
    zcoords = np.zeros((nfvls, 1), dtype=np.float64)
    xcoords[:, 0] = ipts[:, 0]
    ycoords[:, 0] = ipts[:, 1]
    zcoords[:, 0] = ipts[:, 2]
    difx = np.tile(xcoords.T, (nfvls, 1)) - np.tile(xcoords, (1, nfvls))
    dify = np.tile(ycoords.T, (nfvls, 1)) - np.tile(ycoords, (1, nfvls))
    difz = np.tile(zcoords.T, (nfvls, 1)) - np.tile(zcoords, (1, nfvls))
    dist = np.sqrt(difx ** 2. + dify ** 2. + difz ** 2.)
    return dist


def fill_distance_compute(ipts, mpts):
    nivls = np.shape(ipts)[0]
    nmvls = np.shape(mpts)[0]

    ixcoords = np.zeros((nivls, 1), dtype=np.float64)
    iycoords = np.zeros((nivls, 1), dtype=np.float64)
    izcoords = np.zeros((nivls, 1), dtype=np.float64)
    ixcoords[:, 0] = ipts[:, 0]
    iycoords[:, 0] = ipts[:, 1]
    izcoords[:, 0] = ipts[:, 2]

    mxcoords = np.zeros((nmvls, 1), dtype=np.float64)
    mycoords = np.zeros((nmvls, 1), dtype=np.float64)
    mzcoords = np.zeros((nmvls, 1), dtype=np.float64)
    mxcoords[:, 0] = mpts[:, 0]
    mycoords[:, 0] = mpts[:, 1]
    mzcoords[:, 0] = mpts[:, 2]

    difx = np.tile(mxcoords.T, (nivls, 1)) - np.tile(ixcoords, (1, nmvls))
    dify = np.tile(mycoords.T, (nivls, 1)) - np.tile(iycoords, (1, nmvls))
    difz = np.tile(mzcoords.T, (nivls, 1)) - np.tile(izcoords, (1, nmvls))
    dist = np.sqrt(difx ** 2. + dify ** 2. + difz ** 2.)
    return np.amax(np.amin(dist, axis=0))


def shape_parameter_tuner(ipts, shandle):
    emin = 10.
    emax = 100.
    epvals = np.linspace(emin, emax, 101)
    myrbf = table_of_rbf_funs(shandle)
    dist = distance_matrix(ipts)
    freq = 1.
    testfunc = lambda x, y, z: np.sinc(freq*x) * np.sinc(freq*y) * np.sinc(freq*z)
    rhs = testfunc(ipts[:, 0], ipts[:, 1], ipts[:, 2])
    max_error = np.zeros(epvals.size)
    for i in range(epvals.size):
        phimat = myrbf(dist, epvals[i])
        invphimat = np.linalg.pinv(phimat)
        error_func = (np.matmul(invphimat, rhs)) / np.diag(invphimat)
        max_error[i] = np.linalg.norm(error_func, np.inf)
    opt_ind = np.argmin(max_error)
    opt_eps = epvals[opt_ind]
    #plt.plot(epvals, np.ma.log10(max_error))
    return opt_eps


def my_rbf_interpolator(ipts, fipts, qpt, shandle, epsilon):
    nfvls = np.shape(ipts)[0]
    myrbf = table_of_rbf_funs(shandle)
    jacmyrbf = table_of_rbf_jac_funs(shandle)
    dist = distance_matrix(ipts)
    phimat = myrbf(dist, epsilon)

    u, s, vh = np.linalg.svd(phimat, full_matrices='False')
    cvals = np.real(np.conj(vh.T) @ np.diag(1./s) @ np.conj(u.T) @ fipts)
    #cvals = np.linalg.solve(phimat, fipts)

    jacmat = np.zeros((3, 3), dtype=np.float64)
    dqdxj = np.tile(np.reshape(qpt, (1, 3)), (nfvls, 1)) - ipts
    qdists = np.linalg.norm(dqdxj, axis=1)
    jacdists = jacmyrbf(qdists, epsilon) / qdists

    vecxdir = np.zeros((1, nfvls), dtype=np.float64)
    vecydir = np.zeros((1, nfvls), dtype=np.float64)
    veczdir = np.zeros((1, nfvls), dtype=np.float64)
    vecxdir[0, :] = cvals[:, 0] * jacdists
    vecydir[0, :] = cvals[:, 1] * jacdists
    veczdir[0, :] = cvals[:, 2] * jacdists
    jacmat[0, :] = vecxdir @ dqdxj
    jacmat[1, :] = vecydir @ dqdxj
    jacmat[2, :] = veczdir @ dqdxj
    return jacmat, np.linalg.cond(phimat)


def condition_number_compute(ipts, shandle, epsilon):
    myrbf = table_of_rbf_funs(shandle)
    dist = distance_matrix(ipts)
    phimat = myrbf(dist, epsilon)
    return np.linalg.cond(phimat)


def jacobian_maker(reddata, shandle, indices, dt, epval, jj):
    dstp = 1e-6
    NTT = np.shape(reddata)[1]
    yloc = reddata[:, jj]
    nnindices = indices[jj]
    indsupper = nnindices + np.ones(np.size(nnindices), dtype=int) < NTT
    indslower = nnindices - np.ones(np.size(nnindices), dtype=int) >= 0
    indskp = indsupper * indslower
    nnindices_clip = nnindices[indskp]
    ipts = reddata[:, nnindices_clip]
    fwdvls = reddata[:, nnindices_clip + np.ones(np.size(nnindices_clip), dtype=int)]
    bwdvls = reddata[:, nnindices_clip - np.ones(np.size(nnindices_clip), dtype=int)]
    fipts = (fwdvls - bwdvls) / (2. * dt)
    jacmat, cval = my_rbf_interpolator(ipts.T, fipts.T, yloc + dstp * np.ones(3, dtype=np.float64), shandle, epval)
    return jacmat, cval


def centered_diff_test(reddata, indices, dt, jj, sys_handle, params):
    NTT = np.shape(reddata)[1]
    nnindices = indices[jj]
    indsupper = nnindices + np.ones(np.size(nnindices), dtype=int) < NTT
    indslower = nnindices - np.ones(np.size(nnindices), dtype=int) >= 0
    indskp = indsupper * indslower
    nnindices_clip = nnindices[indskp]
    ipts = reddata[:, nnindices_clip]
    fwdvls = reddata[:, nnindices_clip + np.ones(np.size(nnindices_clip), dtype=int)]
    bwdvls = reddata[:, nnindices_clip - np.ones(np.size(nnindices_clip), dtype=int)]
    fipts = (fwdvls - bwdvls) / (2. * dt)
    fiptsexact = np.zeros((np.shape(ipts)[0], np.shape(ipts)[1]), dtype=np.float64)

    if sys_handle == 'Lorenz':
        rval = params[0]
        sigma = params[1]
        bval = params[2]
        for ll in range(np.shape(ipts)[1]):
            fiptsexact[:, ll] = lorenz(ipts[:, ll], sigma, rval, bval)
    elif sys_handle == 'Rossler':
        aval = params[0]
        bval = params[1]
        cval = params[2]
        for ll in range(np.shape(ipts)[1]):
            fiptsexact[:, ll] = rossler(ipts[:, ll], aval, bval, cval)

    return [np.linalg.norm(fipts - fiptsexact)/np.linalg.norm(fiptsexact), np.linalg.norm(fiptsexact)]
