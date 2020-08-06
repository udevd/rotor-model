from tenpy.models.lattice import Chain
from tenpy.networks.site import SpinSite
from tenpy.models.model import MultiCouplingModel, NearestNeighborModel, MPOModel
from tenpy.networks.mps import MPS
from tenpy.networks.terms import TermList
from itertools import product
import numpy as np
import tenpy.algorithms
class IsingWitnessChain(MultiCouplingModel,  MPOModel):
    def __init__(self, L=10, Jxx=1, Jyy=1, Y=0, hz=0, options={}):
        # use predefined local Hilbert space and onsite operators
        site = SpinSite(S=1/2, conserve=None)
        self.L=L
        lat = Chain(L, site, bc="open", bc_MPS="finite") # define geometry
        MultiCouplingModel.__init__(self, lat)
        # add terms of the Hamiltonian;
        # operators "Sx", "Sy", "Sz" are defined by the SpinSite
        self.add_coupling(Jxx, 0, "Sx", 0, "Sx", 1)
        self.add_coupling(Jyy, 0, "Sy", 0, "Sy", 1)
       
        self.add_multi_coupling(Y, [("Sx", -1, 0), ("Sz", 0, 0), ("Sy", 1, 0)] )
        self.add_multi_coupling(-Y, [("Sy", -1, 0), ("Sz", 0, 0), ("Sx", 1, 0)] )
        
        self.add_onsite(hz, 0, "Sz")
        
        # finish initialization
        MPOModel.__init__(self, lat, self.calc_H_MPO())
        self.psi= MPS.from_product_state(self.lat.mps_sites(), [0]*L, "finite")
        self.options=options
        self.dmrg_retval=None
    def calculate_ground_state(self):
        self.dmrg_retval=tenpy.algorithms.dmrg.run(self.psi,self,self.options)
    def calculate_expvals(self):
        if self.dmrg_retval is None:
            self.calculate_ground_state()
        ESxx=sum(self.psi.expectation_value_multi_sites(['Sx','Sx'],i) for i in range(self.L-1) )
        ESyy=sum(self.psi.expectation_value_multi_sites(['Sy','Sy'],i) for i in range(self.L-1) )
        
        EW=sum(self.psi.expectation_value_multi_sites(['Sx','Sz','Sy'],i) for i in range(self.L-2) )
        EW=EW-sum(self.psi.expectation_value_multi_sites(['Sy','Sz','Sx'],i) for i in range(self.L-2) )
           
        ESz=sum(self.psi.expectation_value_multi_sites(['Sz'],i) for i in range(self.L))
        return(ESxx,ESyy,EW)
