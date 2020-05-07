import qutip
import numpy as np

from singlesitebasis import EnergyRestrictedSingleSiteOperators3D, EnergyRestrictedSingleSiteOperators2D


class RotorChainOperators3D:
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


class RotorChainOperators2D:
    def __init__(self, sites, en):
        self.rotor=EnergyRestrictedSingleSiteOperators(en)
        self.sites=sites
        self.JJ=sum( self.onsites({k:self.rotor.JJ, (k+1)%sites: self.rotor.JJ})  for k in range(self.sites) )
        
        self.xx=sum( self.onsites({k:self.rotor.x, (k+1)%sites: self.rotor.x})  for k in range(self.sites) )
        self.yy=sum( self.onsites({k:self.rotor.y, (k+1)%sites: self.rotor.y})  for k in range(self.sites) )

        self.rr=self.xx+self.yy
        self.opers=[self.JJ, self.xx,self.yy, self.rr]
    def onsites(self, sites_opers):
        return qutip.tensor(*[(sites_opers[k] if k in sites_opers else self.rotor.id ) for k in range(self.sites)])



