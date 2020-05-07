import qutip
import numpy as np

from harmonicoscillatorbasis import OneDHarmonicOperators

class SingleSiteOperators3D:
    def __init__(self, dim):
        self.dim=dim
        self.osc=OneDHarmonicOperators(dim)
        self.id=qutip.tensor(*[self.osc.id for _ in range(3)])
        self.H=sum(self.onsubsystem(k,self.osc.H) for k in range(3))
        
        self.x=self.onsubsystem(0,self.osc.x)
        self.y=self.onsubsystem(1,self.osc.x)
        self.z=self.onsubsystem(2,self.osc.x)
        
        self.px=self.onsubsystem(0,self.osc.p)
        self.py=self.onsubsystem(1,self.osc.p)
        self.pz=self.onsubsystem(2,self.osc.p)
        
        self.Jx=self.y*self.pz-self.z*self.py
        self.Jy=self.z*self.px-self.x*self.pz
        self.Jz=self.x*self.py-self.y*self.px
        
        self.JJ=self.Jx**2+self.Jy**2+self.Jz**2
        
    def onsubsystem(self,k,oper):
        return qutip.tensor([oper if i==k else self.osc.id for i in range(3)])
    
class EnergyRestrictedSingleSiteOperators3D(SingleSiteOperators3D):
    def __init__(self, en):
        super().__init__(en)
        self.indices_to_remove=np.argwhere(self.H.diag()>=en)
        
        self.id=self.id.eliminate_states(self.indices_to_remove)
        self.H=self.H.eliminate_states(self.indices_to_remove)

        self.x=self.x.eliminate_states(self.indices_to_remove)
        self.y=self.y.eliminate_states(self.indices_to_remove)
        self.z=self.z.eliminate_states(self.indices_to_remove)

        self.px=self.px.eliminate_states(self.indices_to_remove)
        self.py=self.py.eliminate_states(self.indices_to_remove)
        self.pz=self.pz.eliminate_states(self.indices_to_remove)
        
        self.Jx=self.Jx.eliminate_states(self.indices_to_remove)
        self.Jy=self.Jy.eliminate_states(self.indices_to_remove)
        self.Jz=self.Jz.eliminate_states(self.indices_to_remove)
        
        self.JJ=self.JJ.eliminate_states(self.indices_to_remove)
        
    def restrict_in_energy(self, oper):
        oper=oper.eliminate_states(self.indices_to_remove)
        
        
class SingleSiteOperators2D:
    def __init__(self, dim):
        self.dim=dim
        self.osc=OneDHarmonicOperators(dim)
        self.id=qutip.tensor(*[self.osc.id for _ in range(2)])
        self.H=sum(self.onsubsystem(k,self.osc.H) for k in range(2))
        
        self.x=self.onsubsystem(0,self.osc.x)
        self.y=self.onsubsystem(1,self.osc.x)
        
        self.px=self.onsubsystem(0,self.osc.p)
        self.py=self.onsubsystem(1,self.osc.p)
        
        self.J=self.x*self.py-self.y*self.px
        
        self.JJ=self.J**2
        
    def onsubsystem(self,k,oper):
        return qutip.tensor([oper if i==k else self.osc.id for i in range(2)])
    
class EnergyRestrictedSingleSiteOperators2D(SingleSiteOperators2D):
    def __init__(self, en):
        super().__init__(en)
        self.indices_to_remove=np.argwhere(self.H.diag()>=en)
        
        self.id=self.id.eliminate_states(self.indices_to_remove)
        self.H=self.H.eliminate_states(self.indices_to_remove)

        self.x=self.x.eliminate_states(self.indices_to_remove)
        self.y=self.y.eliminate_states(self.indices_to_remove)

        self.px=self.px.eliminate_states(self.indices_to_remove)
        self.py=self.py.eliminate_states(self.indices_to_remove)
        
        self.J=self.J.eliminate_states(self.indices_to_remove)
        
        self.JJ=self.JJ.eliminate_states(self.indices_to_remove)
        
    def restrict_in_energy(self, oper):
        oper=oper.eliminate_states(self.indices_to_remove)
