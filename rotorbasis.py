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
class SingleSiteOperators:
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
    
class EnergyRestrictedSingleSiteOperators(SingleSiteOperators):
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
class RotorChainOperators:
    def __init__(self, sites, en):
        self.rotor=EnergyRestrictedSingleSiteOperators(en)
        self.sites=sites
        self.JJx=sum( self.onsites({k:self.rotor.Jx, (k+1)%sites: self.rotor.Jx})  for k in range(self.sites) )
        self.JJy=sum( self.onsites({k:self.rotor.Jy, (k+1)%sites: self.rotor.Jy})  for k in range(self.sites) )
        self.JJz=sum( self.onsites({k:self.rotor.Jz, (k+1)%sites: self.rotor.Jz})  for k in range(self.sites) )
        
        self.xx=sum( self.onsites({k:self.rotor.x, (k+1)%sites: self.rotor.x})  for k in range(self.sites) )
        self.yy=sum( self.onsites({k:self.rotor.y, (k+1)%sites: self.rotor.y})  for k in range(self.sites) )
        self.zz=sum( self.onsites({k:self.rotor.z, (k+1)%sites: self.rotor.z})  for k in range(self.sites) )
        
        self.JJ=self.JJx+self.JJy+self.JJz 
        self.rr=self.xx+self.yy+self.zz
        self.opers=[self.JJx,self.JJy,self.JJz,self.xx,self.yy,self.zz, self.JJ, self.rr]
    def onsites(self, sites_opers):
        return qutip.tensor(*[(sites_opers[k] if k in sites_opers else self.rotor.id ) for k in range(self.sites)])

