# This is hamiltonian2.py
#
# Purpose: Define an alternative Hamiltonian

import numpy as np
import lattice
from lattice import a, b, c, ab

##################################################################
# THE PHYSICAL SYSTEM
##################################################################

##################################################################
# HOPPINGS AND COUPLINGS #########################################
##################################################################

# intra orbital hopping
# orbital 1
t1_nu  = +2. 
t1_par = +0.5
t1_bot = +0.5
# orbital 2
t2_nu  = -t1_nu 
t2_par = -t1_par
t2_bot = -t1_bot

# inter orbital hopping
# with dispersion sin(kc) instead of sin(kc/2)
# see below
t_12= +0.50

# SOC induced mixing
L_Rashba   = 0.2 
L_Dresselh = 0.0

c1 = t1_bot
c2 = t2_bot

##################################################################
# HAMILTONIAN in k-space #########################################
##################################################################
# single entries, *h marking the parallel component

# diagonal term
def M1h(k):
    return  t1_nu - 2*t1_par*(      
                    np.cos(np.dot(k,a))
                    +np.cos(np.dot(k,b))
                    +np.cos(np.dot(k,ab)) ) 

# t1_bot*np.cos(np.dot(k,c))
def M2h(k):
    return  t2_nu - 2*t2_par*(  
                    np.cos(np.dot(k,a))
                    +np.cos(np.dot(k,b))
                    +np.cos(np.dot(k,ab)) ) 

def Ah(k):
    A = -t_12
    return A


# spinless Hamiltonian
def h(k):
    h = np.zeros((2,2),dtype=complex)
    h[0,0] = M1h(k) -2*t1_bot*np.cos(np.dot(k,c))
    h[1,1] = M2h(k) - 2*t2_bot*np.cos(np.dot(k,c))
    h[0,1] = -2j*Ah(k)*np.sin(np.dot(k,c))    
    h[1,0] = +2j*Ah(k)*np.sin(np.dot(k,c))
    return h

# SOC terms ######################################################
def Rh(k_par):
     return L_Rashba*(
                 np.exp(+1j*np.pi/3)*np.sin( np.dot(k_par, a))
                +np.exp(-1j*np.pi/3)*np.sin( np.dot(k_par, b))
                                    +np.sin( np.dot(k_par,ab))
                )

def Dh(k_par):
     return L_Dresselh*( 
                 np.exp(+1j*np.pi/6)*np.sin( np.dot(k_par, a))
                -np.exp(-1j*np.pi/6)*np.sin( np.dot(k_par, b))
                                 +1j*np.sin( np.dot(k_par,ab))
                        )

# complete spinful HAMILTONIAN ###################################
def H(k):
    hdim =4
    H=np.zeros((hdim,hdim),dtype=complex)
    # 
    k_par = np.array(k)
    k_par[2] = 0
    k_z = k[2]

    # intra spin block
    H[0:2,0:2]=h(k)
    H[2:,2:]=h(k)
    ## inter spin hopping
    H[0,2] =  ( Rh(k_par)+Dh(k_par) )
    H[1,3] = -( Rh(k_par)+Dh(k_par) )
    H[2:4,0:2] = H[0:2,2:4].conjugate()
    return H


# useful parameters, easy accesible
# size n means H is nxn
def sizeH():
    return H([0,0,0]).shape[0]




##################################################################
# real space (z-direction) H #####################################
# and functions ##################################################
##################################################################

# spinless Hamiltonian
def h_z(k_par,n_z=100):
    # short names 
    # hamiltonian parameters for k_par
    m1 = M1h(k_par)
    m2 = M2h(k_par)
    a  = Ah(k_par)
    # building blocks h(i) and t(i)
    # of size blocksize
    M_n = np.array([
            [m1             , a         ],
            [a.conjugate()  , m2        ]
            ])
    T_n = np.array([
            [ -c1       , 0         ],
            [ -a.conjugate(), -c2   ]
            ]) 
    # blocksize bs
    bs = M_n.shape[0]

    # build big matrix
    h = np.zeros((bs*n_z,bs*n_z),dtype=complex)
    h[0:bs] = np.block([M_n,T_n,np.zeros((bs,(n_z-2)*bs))])
    for i in range(1,n_z-1):
        h[i*bs:(i+1)*bs] = np.block([
                                np.zeros((bs, (i-1)*bs)),
                                T_n.conj().T,
                                M_n,  
                                T_n,
                                np.zeros((bs, (n_z-2-i)*bs))
                            ])
    h[(n_z-1)*bs:n_z*bs] = np.block([
                                np.zeros((bs,(n_z-2)*bs)),T_n.conj().T,M_n
                            ])
    ## create periodic boundary conditions
    #h[0:bs,(n_z-1)*bs:n_z*bs] = T_n.conj().T
    #h[(n_z-1)*bs:n_z*bs,0:bs] = T_n
    return h




# spinful Hamiltonian
def H_z(k_par,n_z=100):
    # short names for parameters 
    m1 = M1h(k_par)
    m2 = M2h(k_par)
    a  = Ah(k_par)
    r  = Rh(k_par)
    d  = Dh(k_par)
    # building blocks h(i) and t(i)
    # of size blocksize

    M_n = np.array([
            [ m1                , a             , r+d           , 0         ],
            [ a.conjugate()     , m2            , 0             ,-r-d       ],
            [r.conjugate()+d.conjugate(), 0     , m1            , a         ],
            [ 0 ,-r.conjugate()-d.conjugate()   , a.conjugate() , m2        ]
            ])
    T_n = np.array([
            [ -c1           , 0     , 0             , 0   ],
            [ -a.conjugate(), -c2   , 0             , 0   ],
            [ 0             , 0     ,-c1            , 0   ],
            [ 0             , 0     ,-a.conjugate() , -c2 ]
            ])

    # blocksize bs
    bs = M_n.shape[0]

    # build big matrix
    H = np.zeros((bs*n_z,bs*n_z),dtype=complex)
    H[0:bs] = np.block([M_n,T_n,np.zeros((bs,(n_z-2)*bs))])
    for i in range(1,n_z-1):
        H[i*bs:(i+1)*bs] = np.block([
                                np.zeros((bs, (i-1)*bs)),
                                T_n.conj().T,
                                M_n,  
                                T_n,
                                np.zeros((bs, (n_z-2-i)*bs))
                            ])
    H[(n_z-1)*bs:n_z*bs] = np.block([
                                np.zeros((bs,(n_z-2)*bs)),T_n.conj().T,M_n
                            ])
    return H




##################################################################
# Miscellaneous functions ######################################
##################################################################



##################################################################
# Spin matrixes tau_0 x sigma_i ##################################
##################################################################
def S(i):
    if i==0:
        return 1/2*np.identity(4)
    elif i==1:
        return 1/2*np.array([
                    [ 0 , 0 , 1 , 0 ],
                    [ 0 , 0 , 0 , 1 ],
                    [ 1 , 0 , 0 , 0 ],
                    [ 0 , 1 , 0 , 0 ]])
    elif i==2:
        return 1/2*np.array([
                    [ 0 , 0 ,-1j, 0 ],
                    [ 0 , 0 , 0 ,-1j],
                    [ 1j, 0 , 0 , 0 ],
                    [ 0 , 1j, 0 , 0 ]])
    elif i==3:
        return 1/2*np.array([
                    [ 1 , 0 , 0 , 0 ],
                    [ 0 , 1 , 0 , 0 ],
                    [ 0 , 0 ,-1 , 0 ],
                    [ 0 , 0 , 0 ,-1 ]])
    else:
        raise ValueError('Illegal Spin matrix index')


##################################################################
# check if k_parallel is within the nodal line ###################
# when no SOC is present                       ##################

def is_inside(k_par):
    h_k = h(k_par)
    if h_k[0,0] <= h_k[1,1]:
        return True
    else:
        return False
        




# EOF
