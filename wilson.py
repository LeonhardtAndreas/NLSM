# This is wilson.py
#
# Purpose:  Calculate and plot the eigenvalues of the Wilson loop
#           matrix along a straight or circular path for various
#           starting points
import numpy as np
import matplotlib.pyplot as plt

import lattice
from lattice import K_a, K_b, K_c
import hamiltonian2 as hamiltonian



##################################################################
##################################################################
# WILSON LOOP and Berry curvature ################################

# number of bands:
nbands = hamiltonian.sizeH()
# number of occupied bands, assuming it is always half of them
nocc = nbands//2



def plot_wilson_path():
    # DEFINE PATH
    points =['K','$\Gamma$','M']
    points_per_segment = 100
    kpath = lattice.generate_path(points,points_per_segment)
    pathlength = kpath.shape[0]
 
    # wilson loop
    wilsoneigs = np.ones((pathlength,nocc),dtype=complex)   
    for x in range(pathlength):
        wilsoneigs[x] = wilson(kpath[x])
    np.savetxt('./wilson_current.csv',
            np.array([kpath[:,0],kpath[:,1],
                    np.real(wilsoneigs[:,0]),np.real(wilsoneigs[:,1])]).T)
    fig, ax = plt.subplots()
    # x-axis
    # label path according to segments defined above
    ax.set_xticks( np.arange(len(points) )*points_per_segment )
    ax.set_xticklabels(points)
    ax.set_xlabel('$k_\parallel$')
    ax.set_xlim(0,pathlength)
    # y-axis
    ax.set_ylabel('$\mathcal{P}_\mathrm{i}$')
    ax.set_ylim((-1.1*np.pi,1.1*np.pi))
    ax.set_yticks([-np.pi,0,np.pi])
    ax.set_yticklabels([r'$-\pi$','0',r'$\pi$'])
    for i in range(nocc):
        #ax.plot(np.imag(wilsoneigs[:,i]),'ro',markersize=1)
        ax.plot(np.real(wilsoneigs[:,i]),'o--',markersize=2+2*(nocc-i))
    # gray out the areas which are not defined (i.e. have NaN as value)
    cover_x = np.arange(pathlength)
    cover_y = np.ones(pathlength)*10
    cover_x = cover_x[np.isnan(wilsoneigs[:,0])] + 0.5
    cover_y = cover_y[np.isnan(wilsoneigs[:,0])]
    cover_x = np.concatenate((cover_x,cover_x[::-1]))
    cover_y = np.concatenate((cover_y,-cover_y))
    ax.fill(cover_x,cover_y,'tab:gray',zorder=10,hatch='//') 
    plt.show()



def wilson(k):
     # steps per loop
    m = 100
    # calculate all the eigen vectors along the path
    vecs = np.zeros((m+1,nbands,nocc),dtype=complex) 
    
    for z in range(0,m+1):
        # path in z-direction
        k_step = k + (z/m-1/2)*K_c
        
        # For a circle around the starting k,
        # uncomment these lines instead of the above
        #phi = 2*np.pi*z/m
        #R = np.array([  [ 1, 0          , 0         ], 
        #                [ 0, np.cos(phi),-np.sin(phi)],
        #                [ 0, np.sin(phi),np.cos(phi)]])
        #r = np.array([0,0,np.pi*0.2]) 
        #k_step = k +  np.dot(R,r)

        # numerical diagonalization
        evals, evecs = np.linalg.eigh(hamiltonian.H(k_step))
        ind = np.argsort(np.real(evals))
        # check for well defined gap along the path
        # with the hack of looking only for band crossings out
        # of the k_z=0 plane
        if ( np.abs(k_step[2]) > 0.1 and 
                ( evals[ind[2]] - evals[ind[1]] ) < 1e+1/m ):
            # if the gap closes along the path, the berry
            # phase is ill defined
            return np.nan

        vecs[z] = evecs[:,ind[0:2]]

    # calculate the Berry link variable etc.
    wilsonmatrix = np.identity(nocc,dtype=complex)
    for z in range(m):
        linkmatrix = vecs[z].T.conj() @ vecs[z+1]
        # ensure that the link matrix is unitary
        # by dividing by the nocc'th root of the determinant (multilinearity)
        det_correction_factor = np.abs(np.linalg.det(linkmatrix))**(1/nocc)
        wilsonmatrix = wilsonmatrix @ linkmatrix/( det_correction_factor )
        #correction += -1j*1/2*vecs[z,1]*vecs[z,1].conj()*2*np.pi/(m)
    
    # close the loop with the initial eigen vector
    wilsonmatrix = wilsonmatrix @ ( vecs[m].T.conj() @ vecs[0] )
    wilson_eigenvalues, wilson_eigenvectors = np.linalg.eig(wilsonmatrix)
    # add convergence factor to avoid jumps at the branch cut
    wilson_eigenvalues += -1e-14j
    return +1j*np.log(wilson_eigenvalues)

