This is NLSM repository

All the code is meant to be run from main.py in Python3.
It is tested with the Anaconda stand alone python installation,
which contains all the dependencies needed here.

    https://www.anaconda.com/download/




lattice.py, 
hamiltonian.py, 
hamiltonian2.py:
    The modules lattice.py and hamiltonian.py
    are for defining the system, e.g. the space group in lattice.py
    and the system in hamiltonian.py and hamiltonian2.py.
    For the latter, include the module as

            import hamiltonian2 as hamiltonian

    in the other files. 
    Most of the time, the notation of the draft is followed.


surfstates.py:
    The module surfstates.py contains the functions calculating
    various observables. Run them by uncommenting the lines in
    main.py. Some of these functions take a couple of minutes to
    finish. They might run in parallel. Parameters are usually
    fixed in the beginning of the functions, sometimes passed as
    parameters.


wilson.py:
    The module wilson calculates the Berry Phase along closed paths,
    usually along k_z direction. It shows the individual berry phases
    for each occupied band, They need to be added for the whole system,
    only then they are quantized to 0 and 1.


I apologize for the poor documentation, more functions follow
once I managed to polish them up a bit.

