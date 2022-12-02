import numpy as np
from numerical_solvers_and_models import *
from scipy.interpolate import Rbf
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import os
parent_directory = str(os.getcwd())
directory = 'Plots'
filepath = os.path.join(parent_directory, directory)
if os.path.exists(filepath) != True:
    os.mkdir(filepath)

def table_of_rbf_funs(shandle):
    dict_of_funs = {'multiquadric': lambda r,ep: np.sqrt((r/ep)**2. + 1.),
                    'inverse': lambda r,ep: 1.0/np.sqrt((r/ep)**2. + 1.),
                    'gaussian': lambda r,ep: np.exp(-(r/ep)**2.),
                    'linear': lambda r,ep: r,
                    'cubic': lambda r,ep: r**3.,
                    'quintic': lambda r,ep: r**5.,
                    'thin_plate': lambda r,ep: r**2. * np.ma.log(r)}
    return dict_of_funs[shandle]

def table_of_rbf_jac_funs(shandle):
    dict_of_funs = {'multiquadric': lambda r,ep: (r/ep**2.)/np.sqrt((r/ep)**2. + 1.),
                    'inverse': lambda r,ep: -(r/ep**2.)/((r/ep)**2. + 1.)**(1.5),
                    'gaussian': lambda r,ep: -2.*r/ep**2. * np.exp(-(r/ep)**2.),
                    'linear': lambda r,ep: 1.,
                    'cubic': lambda r,ep: 3.*r**2.,
                    'quintic': lambda r,ep: 5.*r**4.,
                    'thin_plate': lambda r,ep: r* (2.*np.ma.log(r)+1.)}
    return dict_of_funs[shandle]


def condition_no(ipts, fipts, shandle, epsilon):
    nfvls = np.shape(fipts)[0]
    xcoords = np.zeros((nfvls, 1), dtype=np.float64)
    ycoords = np.zeros((nfvls, 1), dtype=np.float64)
    zcoords = np.zeros((nfvls, 1), dtype=np.float64)

    myrbf = table_of_rbf_funs(shandle)

    xcoords[:, 0] = ipts[:, 0]
    ycoords[:, 0] = ipts[:, 1]
    zcoords[:, 0] = ipts[:, 2]
    difx = np.tile(xcoords.T, (nfvls, 1)) - np.tile(xcoords, (1, nfvls))
    dify = np.tile(ycoords.T, (nfvls, 1)) - np.tile(ycoords, (1, nfvls))
    difz = np.tile(zcoords.T, (nfvls, 1)) - np.tile(zcoords, (1, nfvls))
    dist = np.sqrt(difx ** 2. + dify ** 2. + difz ** 2.)

    phimat = myrbf(dist, epsilon)
    condi = np.linalg.cond(phimat)  # every digit of 10 is one less accuracy on condition number
    return condi

sigma = 16.
bval = 4.
rval = 40.
dt = .01
t0 = 0.
tf = 120.
tvals = np.linspace(t0,tf,int((tf-t0)/dt)+1)
x0 = np.array([0., 1., 0.])
fhandle = lambda x: lorentz(x,sigma,rval,bval)
rawdata = timestepper(x0, t0, tf, dt, fhandle)

shandle = 'gaussian'

elist = []
conds=[]

transient_skp = 400
reddata = rawdata[:,transient_skp:]
nbrs = NearestNeighbors(n_neighbors=100, algorithm='kd_tree').fit(reddata.T)
distances, indices = nbrs.kneighbors(reddata.T)

NTT = np.shape(reddata)[1]
qprior = np.eye(3,dtype=np.float64)
imat = np.eye(3,dtype=np.float64)
rvals = np.ones((3,NTT),dtype=np.float64)

transient_skp = 400
reddata = rawdata[:,transient_skp:]
nbrs = NearestNeighbors(n_neighbors=100, algorithm='kd_tree').fit(reddata.T)
distances, indices = nbrs.kneighbors(reddata.T)

jj = 10
dstp = 1e-6
yloc = reddata[:,jj]
nnindices = indices[jj]
ipts = reddata[:,nnindices]
fwdvls = reddata[:,np.mod(nnindices+np.ones(np.size(nnindices), dtype=int),NTT)]
bwdvls = reddata[:,np.mod(nnindices-np.ones(np.size(nnindices), dtype=int),NTT)]
fipts = (fwdvls - bwdvls)/(2.*dt)

for i in range (0,100000):
    epval = i
    condi = condition_no(ipts.T, fipts.T, shandle, epval)
    elist.append(epval)
    conds.append(condi)

plt.figure(figsize=(10,10))
plt.scatter(np.log10(elist),np.log10(conds))
plt.title(r'$\epsilon$ vs condition number of phimat log-plot')
plt.xlabel(r'log($\epsilon$)')
plt.ylabel('log(condition number)')
filename = filepath + '/Condition Log Plot (Gaussian)'
plt.savefig(filename)
plt.show()
print('Plot saved to ' + filename + '.png')
