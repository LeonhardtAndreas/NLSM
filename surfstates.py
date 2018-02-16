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
import hamiltonian2 as hamiltonian

##################################################################
# surface LDOS states ############################################
##################################################################

def plot_ldos():
    ### parameters for the plot ##################################
    # slab thicknes 
    # (each layer consisting of 4 orbitals incl. spin)
    n_z = 1600
    # number of slabs to take into consideration for the surface
    # DOS
    n_surf = 2

    # energy range for DOS
    nomega = 100
    omega = np.linspace(-.02,.02,nomega)

    # generate path
    points = ['$\Gamma$','M']
    n_segment = 10
    kpath = lattice.generate_path(points,points_per_segment=n_segment)*.003
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
    eps = 1e-3j

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


def localization(k_par=np.array([0,0,0]),n_z=100):
    
    # get number of orbitals (including spin dof)
    norb = hamiltonian.sizeH()

    # surface states are searched for within the 
    # 2*n_E energy eigenstates around the Fermi energy
    # WHICH IS ASSUMED TO HAVE THE SAME NUMBER OF STATES
    # ABOVE AND BELOW
    n_E = 20
    n_Emin = norb*n_z//2-n_E
    n_Emax = norb*n_z//2+n_E

    # then, they are picked based on localization, measured by
    # the norm within the n_loc layers from the surface
    n_loc = norb*n_z//20 # e.g. 5 percent of total layers

    # get all eigenvalues
    E, Psi = np.linalg.eigh(hamiltonian.H_z(k_par,n_z))
    Psi = Psi.T
    
    # pick the ones most localized on the left side
    ind_left  = np.argsort(np.linalg.norm(
            Psi[n_Emin:n_Emax,0:n_loc],axis=1))[-2:]
    # and the ones most localized on the right
    ind_right = np.argsort(np.linalg.norm(
            Psi[n_Emin:n_Emax,-n_loc:],axis=1))[-2:]
    ind = np.concatenate((ind_left,ind_right))+n_Emin
    
    # calculate the norm at each n,
    # i.e. 
    return np.sqrt(  np.abs(Psi[ind,0::4])**2 
                    +np.abs(Psi[ind,1::4])**2
                    +np.abs(Psi[ind,2::4])**2
                    +np.abs(Psi[ind,3::4])**2 )


def plot_localization():
    # parameters #################################################
    # number of layers
    n_z = 400
    # number of k-points
    # this is done manually, not through lattice.generate_path()
    n_k = 10
    
    # create plot
    fig, ax = plt.subplots()
    
    # loop over momenta
    for x in range(1,n_k+1):
        # start from Gamma, go in the direction of M
        k = (x/n_k)*1/4*lattice.named_points['M']
        # get surface states
        norm_Psi = localization(k,n_z)
        # create label only, when it hasn't been done for that k
        # otherwise skip by keeping label empty
        labeled = False 
        
        # loop over surface states
        for i in range(norm_Psi.shape[0]):
            if labeled:
                labelstring = ''
            else:
                labelstring = '$k_x={:.2f}\pi$'.format(k[0]/np.pi) 
                labeled = True
            # plot and with rgb colour manually calculated 
            ax.plot(norm_Psi[i],label=labelstring,
                color=(np.cos(np.pi/2*x/n_k),0,np.sin(np.pi/2*x/n_k),1) )
    
    # labeling the plot
    ax.set_xlabel('$n_z$')
    ax.set_xlim([-0.2,n_z-.8])
    ax.set_ylabel('$|\Psi|$')
    ax.legend() 
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






# EOF
