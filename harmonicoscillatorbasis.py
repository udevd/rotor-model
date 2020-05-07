import qutip
import numpy as np
class OneDHarmonicOperators():
    def __init__(self, dim):
        self.dim=dim
        self.a=qutip.destroy(dim)
        self.adag=qutip.create(dim)
        self.x=(self.a+self.adag)/np.sqrt(2)
        self.p=-1.j*(self.a-self.adag)/np.sqrt(2)
        self.H=qutip.num(dim)
        self.id=qutip.qeye(dim)
