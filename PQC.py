from qiskit import *
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
from scipy.stats import rv_continuous

class PQC:
    def __init__(self,name):
        self.backend = Aer.get_backend('statevector_simulator');
        self.circ = QuantumCircuit(1);
        self.name = name;
        self.seed =14256;
        np.random.seed(self.seed);
        if self.name=="rz":
            self.circ.h(0);
        if self.name=="rzx":
            self.circ.h(0);
    def add(self):
        if self.name == "rz":
            th = np.random.uniform(0,2*np.pi);
            self.circ.rz(th,0);
        if self.name == "rzx":
            th1 = np.random.uniform(0,2*np.pi);
            self.circ.rz(th1,0);
            th2 = np.random.uniform(0,2*np.pi);
            self.circ.rx(th2,0);   
    def remove(self):
        if self.name == "rz":
            self.circ.data.pop(1);
        if self.name == "rzx":
            self.circ.data.pop(1);
            self.circ.data.pop(1);

    def get(self):
        self.add();
        result = execute(self.circ,self.backend).result();
        out_state = result.get_statevector();
        self.remove(); # remove a random gate
        return np.asmatrix(out_state).T;

    def draw(self):
        th = np.random.uniform(0,2*np.pi);
        self.circ.rz(th,0);
        self.circ.draw('mpl'); # 왜 안 그려지는지 모르겠습니다
        self.circ.data.pop(1);