This is NLSM repository

All the code is meant to be run from main.py in Python3.
It is tested with the Anaconda stand alone python installation,
which contains all the dependencies needed here.

https://www.anaconda.com/download/


The modules lattice.py and hamiltonian.py
are for defining the system, e.g. the space group in lattice.py
and the system in hamiltonian.py.
Most of the time, the notation of the draft is followed.


The module surfstates.py contains the functions calculating
various observables. Run them by uncommenting the lines in
main.py. Some of these functions take a couple of minutes to
finish. They might run in parallel. Parameters are usually
fixed in the beginning of the functions, sometimes passed as
parameters.

I apologize for the poor documentation, more functions follow
once I managed to polish them up a bit.

The spin polarization functions at the end of surfstates.py are
not yet functional in the uploaded version.



