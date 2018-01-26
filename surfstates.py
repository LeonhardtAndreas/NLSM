# This is surfstates.py
# 
# Purpose:  One or several of the following:
#           - Calculate the surface states of a system
#             and their dispersion
#           - along a path or does a search for different energies
# 
#
##################################################################
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from multiprocessing import Pool,cpu_count
from functools import partial
import lattice
import hamiltonian

##################################################################
# surface LDOS states ############################################
##################################################################

def plot_ldos():
    ### parameters for the plot ##################################
    # slab thicknes 
    # (each layer consisting of 4 orbitals incl. spin)
    n_z = 200
    # number of slabs to take into consideration for the surface
    # DOS
    n_surf = 2

    # energy range for DOS
    nomega = 200
    omega = np.linspace(-9,9,nomega)

    # generate path
    points = ['K','$\Gamma$','M']
    n_segment = 100
    kpath = lattice.generate_path(points,points_per_segment=n_segment)
    pathlength = kpath.shape[0]
    
    # calculate the LDOS at each k_parallel
    dos = np.zeros((pathlength,nomega))
    # calculate at each k, parallelized 
    with Pool(cpu_count()) as p:
        dos = p.map(  
                    partial(ldos_k,
                            n_z,n_surf,omega
                            ),
                    kpath
                    )
    # convert back to numpy array
    dos = np.array(dos)

    # plot #######################################################
    fig, ax = plt.subplots()
    ax.imshow(dos.T,cmap='hot', interpolation='none',
                        origin='lower',aspect='auto',
                        extent=[0,pathlength-1,omega[0],omega[-1]])
    ax.set_ylabel(r'$\omega$')
    ax.set_xticks( np.arange(len(points))*n_segment )
    ax.set_xticklabels(points)
    ax.set_xlim(0,pathlength-1)
    ax.set_xlabel(r'$k_\parallel$')
    plt.show()


##################################################################
# calculate the LDOS for a given parallel momentum k and all omega
def ldos_k(n_z=100,n_surf=2, omega=np.linspace(-1,1,10), k=np.array([0,0,0])):
    # tags: local density of states, LDOS, ldos,Local Density Of States
    # broadening of the Greens function
    eps = 3e-2j

    # get number of orbitals and size of slab dynamically   
    norb = hamiltonian.sizeH()
    nomega = len(omega)
   
    greens = np.zeros(nomega,dtype=complex)
        
    evals, evecs = np.linalg.eigh(hamiltonian.H_z(k,n_z))
    
    for n in range(len(evals)):
        inner = np.vdot(
                        evecs[0:n_surf*norb,n],
                        evecs[0:n_surf*norb,n]
                        )
        for i, om in enumerate(omega):
            greens[i] += inner/( om +eps- evals[n])
    
    return -1/np.pi*np.imag(greens)



##################################################################
# get surface state properties ###################################


def plot_localization(k_par=np.array([0,0,0]),n_z=100):
    E, Psi = get_surfstates(n_z,n_z//3,k_par)
    fig, axes = plt.subplots()
    for j in range(len(E)): 
        plt.plot(np.sqrt(np.abs(Psi[j,0::4])**2 
                        +np.abs(Psi[j,1::4])**2
                        +np.abs(Psi[j,2::4])**2
                        +np.abs(Psi[j,3::4])**2))
    plt.show()


##################################################################
# real space calculations ########################################

# diagonalize the Hamiltonian and return
# (E, Psi) in pairs each
def get_surfstates(n_z=100,z0=10,k_par=[0,0,0]):
    # get number of orbitals
    norb = hamiltonian.sizeH()
    
    if hamiltonian.is_inside(k_par):
        vals, vecs = np.linalg.eigh(hamiltonian.H_z(k_par,n_z))
        # select the right ones based on localization
        # choose the ingap ones, take a slice of thicknes z0
        # and transpose
        Psi = vecs[:,norb*(n_z//2)-2:norb*(n_z//2)+2].T
        E =   vals[  norb*(n_z//2)-2:norb*(n_z//2)+2]
        # sort by norm and get the indices of those
        # with the biggest norm
        # since they are localized at the right edge
        ind = np.argsort(np.linalg.norm(Psi[:,0:norb*(n_z//2)],axis=1))[-2:]
        
        return ( np.array(E[ind]) , np.array(Psi[ind,0:norb*z0]) )
    else:
        print('No surface state outside the nodal line')
        return ( np.array([np.nan,np.nan]) , np.zeros((2,norb*z0)) )





##################################################################
## IGNORE EVERYTHING BELOW THIS LINE FOR NOW #####################
##################################################################
##################################################################






# return the spin expectation values
# for surface states
def spin_polarization(k_par=np.array([0,0,0])):
    S = np.zeros((2,3),dtype=complex)
    z_total = 200
    z0 = 200
    # calculate surface state eigenvectors and -values
    E, surfstates = get_surfstates(z_total,z0,k_par)
    # sort by energy to get consistent array
    surfstates = surfstates[np.argsort(E)]
    for s in [0,1]:
        for mu in [1,2,3]:
            Spin_matrix = hamiltonian.S(mu)
            for z in range(z0):
                S[s,mu-1] += np.vdot( surfstates[s,4*z:4*(z+1)],
                                        Spin_matrix.dot(
                                            surfstates[s,4*z:4*(z+1)]
                                        )
                                    )
    return S

def plot_spin_polarization(result_path='./',ncores=4):
    
    # generate k lattice #########################################
    # same spacing in all parallel directions   
    Nk1d = 51
    kf = hamiltonian.kfmax
    k1dx = np.linspace(-kf,kf,Nk1d)
    k1dy = np.linspace(-kf,kf,Nk1d)
    k_arr = np.zeros((len(k1dx)*len(k1dy),3))
    for i,x in enumerate(k1dx):
        for j,y in enumerate(k1dy):
            k_arr[i*len(k1dy)+j]=[x,y,0]

#    KX, KY = np.meshgrid(k1d,k1dy)
#    k_arr = np.stack([KX,KY,np.zeros_like(KX)]).T.reshape(-1,3)
    #k1d = np.linspace(-np.pi,np.pi,3)
    #k_list = []
    #for kx in k1d:
    #    for ky in k1d:
    #        k_list.append(np.array([kx,ky,0]))
    dk = np.abs(k1dx[1]-k1dx[0])    
    V = (2*np.pi/dk)**2
    # calculate surface state eigenvectors and -values
    with Pool(ncores) as p:
        res = p.map(
                        spin_polarization,
                        k_arr
                        )
    S = np.zeros((len(res),2,3),dtype=complex)
    for i in range(len(res)):
        S[i]=res[i]
    # kx, ky, E index, S(vec)
#    S = S.reshape(Nk1d,Nk1d,2,3)
#    k_arr = k_arr.reshape(Nk1d,Nk1d,3)
    np.save(result_path+'S_arr',S)
    np.save(result_path+'k_arr',k_arr)




# EOF
