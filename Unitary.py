import numpy as np;
from qiskit import *;
from matplotlib import pyplot as plt
def unitary(circ,eta,phi):
    theta = np.arccos(-eta);
    circ.x(0);
    return;


backend = Aer.get_backend('statevector_simulator');
circ = QuantumCircuit(1);
# print(circ)
unitary(circ,1,1);
circ.draw('mpl');