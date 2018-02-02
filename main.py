#!/usr/bin/env python3
#
# This is main.py
#
# Purpose: organize and run the functions defined in the other scripts
# Tested on Anaconda Python3 installation
# https://www.anaconda.com/download/

import os
import sys
import numpy as np
import subprocess
from datetime import datetime
import matplotlib

import lattice
import hamiltonian
import surfstates as surf
import wilson


# define which functions to run by uncommenting the proper lines
# options of these functions within the definition of said fcts.

def main():

    ###############################################################
    # SURFACE STATES ##############################################
    
    # calculate and plot the LDOS at the surface
    #surf.plot_ldos()
    
    # check the localization at a given k_parallel
    #k = 0.2*lattice.K_a
    #surf.plot_localization(k_par=k)
    
    wilson.plot_wilson_path()


##################################################################
# CALL OF MAIN ###################################################
main()
##################################################################

