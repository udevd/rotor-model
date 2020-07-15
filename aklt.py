from tenpy.models.lattice import Chain
from tenpy.networks.site import SpinSite
from tenpy.models.model import MultiCouplingModel, NearestNeighborModel, MPOModel
from tenpy.networks.mps import MPS
from tenpy.networks.terms import TermList
from itertools import product
import numpy as np
import tenpy.algorithms
class AKLTChain(MultiCouplingModel,  MPOModel):
    def __init__(self, L=10, J=1, Y=0, hz=0, options={}):
        # use predefined local Hilbert space and onsite operators
        site = SpinSite(S=1, conserve=None)
        self.L=L
        lat = Chain(L, site, bc="open", bc_MPS="finite") # define geometry
        MultiCouplingModel.__init__(self, lat)
        # add terms of the Hamiltonian;
        # operators "Sx", "Sy", "Sz" are defined by the SpinSite
        self.add_coupling(J, 0, "Sx", 0, "Sx", 1)
        self.add_coupling(J, 0, "Sy", 0, "Sy", 1)
        self.add_coupling(J, 0, "Sz", 0, "Sz", 1) 
        
        for oper1 in ['Sx','Sy','Sz']:
            for oper2 in ['Sx','Sy','Sz']: # (\vec S(i) \cdot \vec S(i+1))^2 coupling
                self.add_multi_coupling(Y, [(oper1, 0, 0),(oper2, 0, 0),\
                                            (oper1, 1, 0),(oper2, 1, 0)] ) #note that there are two operators on one site
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
        ESn=sum(self.psi.expectation_value_multi_sites([oper,oper],i) for i,oper in product(range(self.L-1),['Sx','Sy','Sz']) )
        
        relweight=1
        sstermlist = [ [(oper1,i),(oper2,i),(oper1,i+1),(oper2,i+1)]\
            for oper1,oper2 in product(['Sx','Sy','Sz'],['Sx','Sy','Sz'])\
            for i in range(self.L-1) ] #[(term,weight)], where term=[(operator,index),…]
        sstermlistobject = TermList(sstermlist, np.ones(len(sstermlist)))
        
        ESSn,_=self.psi.expectation_value_terms_sum(sstermlistobject) #some intermediate calculations are also returned
        
        ESz=sum(self.psi.expectation_value_multi_sites(['Sz'],i) for i in range(self.L))
        return(ESn,ESSn,ESz)
