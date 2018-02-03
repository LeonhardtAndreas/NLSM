# This is wilson.py
#
# Purpose:  Calculate and plot the eigenvalues of the Wilson loop
#           matrix along a straight or circular path for various
#           starting points
import numpy as np
import matplotlib.pyplot as plt

import lattice
from lattice import K_a, K_b, K_c
import hamiltonian as hamiltonian



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

    np.savetxt('current.csv', np.stack(kpath[:,0:2],wilsoneigs))

   
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
        ind = np.argsort(np.real(evals))[0:2]
        vecs[z] = evecs[:,ind]

    # calculate the Berry link variable etc.
    wilsonmatrix = np.identity(nocc,dtype=complex)
    for z in range(m):
        linkmatrix = vecs[z].T.conj() @ vecs[z+1]
        wilsonmatrix = wilsonmatrix @ linkmatrix/( np.abs(np.linalg.det(linkmatrix)) )
        #correction += -1j*1/2*vecs[z,1]*vecs[z,1].conj()*2*np.pi/(m)
    
    # close the loop with the initial eigen vector
    wilsonmatrix = wilsonmatrix @ ( vecs[m].T.conj() @ vecs[0] )
    wilson_eigenvalues, wilson_eigenvectors = np.linalg.eig(wilsonmatrix)
    # add convergence factor to avoid jumps at the branch cut
    wilson_eigenvalues += -1e-14j
    return +1j*np.log(wilson_eigenvalues)

