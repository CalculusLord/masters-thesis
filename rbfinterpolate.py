"""
Written by
    Christopher W. Curtis
    Nathanael J. Reynolds
        SDSU, 2022
            Version 1.0
"""

import numpy as np
import matplotlib.pyplot as plt
import fdscheme as fds
from sklearn.neighbors import NearestNeighbors
from save import *

class RBF_Interpolate:
    def __init__(self,
                 rbf_shape,
                 interpolant_points,
                 time_step):
        self.shandle = rbf_shape
        self.ipts = interpolant_points
        self.dt = time_step

    def table_of_rbf_funs(self):
        dict_of_funs = {'multiquadric': lambda r, ep: np.sqrt((r / ep) ** 2. + 1.),
                        'inverse': lambda r, ep: 1.0 / np.sqrt((r / ep) ** 2. + 1.),
                        'gaussian': lambda r, ep: np.exp(-(r / ep) ** 2.),
                        'linear': lambda r, ep: r,
                        'cubic': lambda r, ep: r ** 3.,
                        'quintic': lambda r, ep: r ** 5.,
                        'thin_plate': lambda r, ep: r ** 2. * np.ma.log(r)}
        return dict_of_funs[self.shandle]

    def table_of_rbf_jac_funs(self):
        dict_of_funs = {'multiquadric': lambda r, ep: (r / ep ** 2.) / np.sqrt((r / ep) ** 2. + 1.),
                        'inverse': lambda r, ep: -(r / ep ** 2.) / ((r / ep) ** 2. + 1.) ** (1.5),
                        'gaussian': lambda r, ep: -2. * r / ep ** 2. * np.exp(-(r / ep) ** 2.),
                        'linear': lambda r, ep: 1.,
                        'cubic': lambda r, ep: 3. * r ** 2.,
                        'quintic': lambda r, ep: 5. * r ** 4.,
                        'thin_plate': lambda r, ep: r * (2. * np.ma.log(r) + 1.)}
        return dict_of_funs[self.shandle]

    def distance_matrix(self):
        nfvls = np.shape(self.ipts)[0]
        xcoords = np.zeros((nfvls, 1), dtype=np.float64)
        ycoords = np.zeros((nfvls, 1), dtype=np.float64)
        zcoords = np.zeros((nfvls, 1), dtype=np.float64)
        xcoords[:, 0] = self.ipts[:, 0]
        ycoords[:, 0] = self.ipts[:, 1]
        zcoords[:, 0] = self.ipts[:, 2]
        difx = np.tile(xcoords.T, (nfvls, 1)) - np.tile(xcoords, (1, nfvls))
        dify = np.tile(ycoords.T, (nfvls, 1)) - np.tile(ycoords, (1, nfvls))
        difz = np.tile(zcoords.T, (nfvls, 1)) - np.tile(zcoords, (1, nfvls))
        dist = np.sqrt(difx ** 2. + dify ** 2. + difz ** 2.)
        return dist

    def fill_distance_compute(self, mpts):
        nivls = np.shape(self.ipts)[0]
        nmvls = np.shape(mpts)[0]

        ixcoords = np.zeros((nivls, 1), dtype=np.float64)
        iycoords = np.zeros((nivls, 1), dtype=np.float64)
        izcoords = np.zeros((nivls, 1), dtype=np.float64)
        ixcoords[:, 0] = self.ipts[:, 0]
        iycoords[:, 0] = self.ipts[:, 1]
        izcoords[:, 0] = self.ipts[:, 2]

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

    def shape_parameter_tuner(self):
        emin = 100.
        emax = 1000.
        epvals = np.linspace(emin, emax, 1001)
        myrbf = RBF_Interpolate.table_of_rbf_funs(self)
        dist = RBF_Interpolate.distance_matrix(self)

        testfunc = lambda x, y, z: np.sinc(x) * np.sinc(y) * np.sinc(z)
        rhs = testfunc(self.ipts[:, 0], self.ipts[:, 1], self.ipts[:, 2])
        max_error = np.zeros(epvals.size)
        for i in range(epvals.size):
            phimat = myrbf(dist, epvals[i])
            invphimat = np.linalg.pinv(phimat)
            error_func = (np.matmul(invphimat, rhs)) / np.diag(invphimat)
            max_error[i] = np.linalg.norm(error_func, np.inf)
        opt_ind = np.argmin(max_error)
        opt_eps = epvals[opt_ind]
#         print('Plotting epsilons...')
#         save = create_directory('Plots')
#         save = create_subdirectory(save, 'Epsilon')
#         plt.figure(figsize=(10,7))
#         plt.scatter(epvals, np.ma.log10(max_error), s=1)
#         plt.xlabel(r'$\epsilon$ values')
#         plt.ylabel(r'$log_{10}(E_k)$')
#         plt.title('Condition numbers')
#         plot_name = input('Save file as: ')
#         if str(type(plot_name)) != "<class 'str'>":
#             plot_name = str(plot_name)
#         plot_name = name_checker(save, plot_name, '.png')
#         filepath = save + plot_name
#         plt.savefig(filepath)
#         plt.show()
        return opt_eps

    def my_rbf_interpolator(self, fipts, qpt, epsilon):
        nfvls = np.shape(self.ipts)[0]
        myrbf = RBF_Interpolate.table_of_rbf_funs(self)
        jacmyrbf = RBF_Interpolate.table_of_rbf_jac_funs(self)
        dist = RBF_Interpolate.distance_matrix(self)
        phimat = myrbf(dist, epsilon)
        cvals = np.linalg.solve(phimat, fipts)

        jacmat = np.zeros((3, 3), dtype=np.float64)
        dqdxj = np.tile(np.reshape(qpt, (1, 3)), (nfvls, 1)) - self.ipts
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

    def condition_number_compute(self, epsilon):
        myrbf = RBF_Interpolate.table_of_rbf_funs(self)
        jacmyrbf = RBF_Interpolate.table_of_rbf_jac_funs(self)
        dist = RBF_Interpolate.distance_matrix(self)
        phimat = myrbf(dist, epsilon)
        return np.linalg.cond(phimat)

    def jacobian_maker(self, reddata, indices, epval, jj,NTT):
        dstp = 1e-6

        yloc = reddata[:, jj]
        nnindices = indices[jj]
        self.ipts = reddata[:, nnindices].T
        fwdvls = reddata[:, np.mod(nnindices + np.ones(np.size(nnindices), dtype=int), NTT)]
        bwdvls = reddata[:, np.mod(nnindices - np.ones(np.size(nnindices), dtype=int), NTT)]
        fipts = (fwdvls - bwdvls) / (2. * self.dt)
        jacmat, cval = RBF_Interpolate.my_rbf_interpolator(self, fipts.T, yloc + dstp * np.ones(3, dtype=np.float64), epval)
        return jacmat, cval

if __name__=="__main__":
    start_time = 0
    stop_time = 120
    time_step = 0.01

    time = fds.Grid(start_time, stop_time, time_step).create_grid()
    init_cond = np.array([0, 1, 0])
    system = lambda t, X: fds.Systems(X).lorenz(sigma=16, r=40, b=4)
    solver = fds.Scheme(system, init_cond, start_time, stop_time, time_step)
    rawdata, name = solver.runge_kutta()

    transient_skp = 400
    reddata = rawdata[:, transient_skp:]
    nbrs = NearestNeighbors(n_neighbors=100, algorithm='kd_tree').fit(reddata.T)
    distances, indices = nbrs.kneighbors(reddata.T)

    rbf_shape = 'gaussian'
    current_ind = 10
    yloc = reddata[:, current_ind]
    nnindices = indices[current_ind]
    ipts = reddata[:, nnindices]
    interpolator = RBF_Interpolate(rbf_shape, ipts, time_step)
    optimal_shape_parameter = interpolator.shape_parameter_tuner()

    NTT = np.shape(reddata)[1]
    for jj in range(10, 11):
        yloc = reddata[:, jj]
        nnindices = indices[jj]
        ipts = reddata[:, nnindices]

        #     pjac_mat_exact[1, 0] = rval-yloc[2]
        #     pjac_mat_exact[1, 2] = -yloc[0]
        #     pjac_mat_exact[2, 0] = yloc[1]
        #     pjac_mat_exact[2, 1] = yloc[0]

        jacmat, cval = interpolator.jacobian_maker(reddata, indices, optimal_shape_parameter, jj, NTT)
        print(jacmat)
        #     print(pjac_mat_exact)
        print("Condition number is: %1.2e" % cval)

    epvals = np.linspace(.1, 20, 1001)
    ptindx = 10
    dstp = 1e-6
    yloc = reddata[:, ptindx]
    nnindices = indices[ptindx]
    ipts = reddata[:, nnindices]
    interpolator = RBF_Interpolate(rbf_shape, ipts, time_step)
    fwdvls = reddata[:, np.mod(nnindices + np.ones(np.size(nnindices), dtype=int), NTT)]
    bwdvls = reddata[:, np.mod(nnindices - np.ones(np.size(nnindices), dtype=int), NTT)]
    fipts = (fwdvls - bwdvls) / (2. * time_step)
    condnumbers = [RBF_Interpolate(rbf_shape, ipts.T, time_step).condition_number_compute(epval) for epval in epvals]
    plt.figure()
    plt.plot(epvals, np.ma.log10(condnumbers))
    plt.show()

    NTT = np.shape(reddata)[1]
    qprior = np.eye(3, dtype=np.float64)
    imat = np.eye(3, dtype=np.float64)
    rvals = np.ones((3, NTT), dtype=np.float64)

    for jj in range(NTT - 1):
        jacmatn, cvaln = interpolator.jacobian_maker(reddata, indices, optimal_shape_parameter, jj, NTT)
        jacmatp, cvalp = interpolator.jacobian_maker(reddata, indices, optimal_shape_parameter, jj + 1, NTT)

        qprior, rnext = np.linalg.qr(np.linalg.solve((imat - time_step / 2 * jacmatp), (imat + time_step / 2 * jacmatn) @ qprior))
        rvals[:, jj] = np.diag(rnext)

    lvals = np.sum(np.log(np.abs(rvals)), 1) / stop_time

    print(lvals)
