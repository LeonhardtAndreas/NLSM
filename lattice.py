# This is lattice.py
#
# Purpose: define the lattice and reciprocal lattice

import numpy as np


##################################################################
# LATTICE ########################################################

# normalized lattice vectors and combinations
a = np.array([ np.sqrt(3)/2 , 1/2 ,0])
b = np.array([-np.sqrt(3)/2 , 1/2 ,0])
c = np.array([0,0,1.])
# sum and difference
ab = a+b


# reciprocal lattice vectors
K_a = 2*np.pi*np.array([ 1/np.sqrt(3) ,1 ,0 ])
K_b = 2*np.pi*np.array([-1/np.sqrt(3) ,1 ,0 ])
K_c = 2*np.pi*np.array([ 0            ,0 ,1 ])

# define BZ boundary points
boundary = 1/3*np.array([  K_a+K_b,
                        2*K_b-K_a,
                        K_b-2*K_a,
                        -K_a-K_b,
                        K_a-2*K_b,
                        2*K_a-K_b,
                        K_a+K_b])
    
named_points = {'Gamma':      0*K_a +   0*K_b +  0*K_c,
                '$\Gamma$':   0*K_a +   0*K_b +  0*K_c,
                'G$':         0*K_a +   0*K_b +  0*K_c,
                    'M':    1/2*K_a - 1/2*K_b +  0*K_c,
                    'M2':     0*K_a + 1/2*K_b +  0*K_c,
                    'M3':   1/2*K_a +   0*K_b +  0*K_c,
                    'K':    2/3*K_a - 1/3*K_b +  0*K_c,
                    'K2':   1/3*K_a - 2/3*K_b +  0*K_c,
                    'K3':   1/3*K_a + 1/3*K_b +  0*K_c,
                    'A':      0*K_a +   0*K_b +1/2*K_c,
                    'L':    2/3*K_a - 1/3*K_b +1/2*K_c,
                    'L2':   1/3*K_a - 2/3*K_b +1/2*K_c,
                    'L3':   1/3*K_a + 1/3*K_b +1/2*K_c,
                    'H':    1/2*K_a - 1/2*K_b +1/2*K_c,
                   '-M':    -(1/2*K_a - 1/2*K_b +  0*K_c),
                   '-M2':   -(  0*K_a + 1/2*K_b +  0*K_c),
                   '-M3':   -(1/2*K_a +   0*K_b +  0*K_c),
                   '-K':    -(2/3*K_a - 1/3*K_b +  0*K_c),
                   '-K2':   -(1/3*K_a - 2/3*K_b +  0*K_c),
                   '-K3':   -(1/3*K_a + 1/3*K_b +  0*K_c),
                   '-A':    -(  0*K_a +   0*K_b +1/2*K_c),
                   '-L':    -(2/3*K_a - 1/3*K_b +1/2*K_c),
                   '-H':    -(1/2*K_a - 1/2*K_b +1/2*K_c)}




# generate a  path from the above dictionary
def generate_path(points=['Gamma','M','K','Gamma'],points_per_segment = 100 ):
    # e.g. Gamma-M-K-Gamma
    # number of steps per path segment
    number_segments = len(points)-1 
    pathlength= number_segments * points_per_segment+1
    kpath=np.zeros((pathlength,3))
    
    for i in range(number_segments):
        for x in range(points_per_segment):
            #
            kpath[i*points_per_segment + x] = (
                            (1-x/points_per_segment)*named_points[points[i]]
                           +(  x/points_per_segment)*named_points[points[i+1]] )
    kpath[-1] = named_points[points[-1]]
    return kpath
 

# generate a 2D-mesh in reciprocal space, covering the first 
# projected Brillouin zone.
# This function is written for readability and portability, 
# not for speed, avoid using it in a loop.
# The limits are shifted to avoid over- or undersampling at the
# boundaries due to numerical roundoff errors and to give a clear
# edge
#
# Returns:  np-array of k-vectors, approximately the size
#           n squared. The actual size should be manually checked.
def generate_BZmesh(n=100):
    k_mesh = np.zeros((4*n*n,3))
    m = 0
    for i in range(2*n):
        for j in range(2*n):
            k = (i/n-1)*K_a+(j/n-1)*K_b
            eps = 1e-14
            LK_a = np.dot(K_a,K_a)
            LK_b = np.dot(K_b,K_b)
            LK_ab = np.dot(K_a-K_b,K_a-K_b)-eps
            if (    -1/2*LK_a-eps  <= np.dot(K_a,k) < 1/2*LK_a-eps
                and -1/2*LK_b-eps  <= np.dot(K_b,k) < 1/2*LK_b-eps
                and -1/2*LK_ab-eps <= np.dot(K_a-K_b,k) < 1/2*LK_ab-eps ):
                k_mesh[m] = k
                m += 1
    return k_mesh[0:m]

