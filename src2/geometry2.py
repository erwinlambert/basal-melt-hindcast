import numpy as np
import xarray as xr
import pyproj

from constants import ModelConstants

class Geometry(ModelConstants):
    """Create geometry input"""
    def __init__(self,name):
        if name=='R0':
            x0,x1,y0,y1 = -2.15e6,-1.71e6,.82e6,.33e6
        elif name=='R1':
            x0,x1,y0,y1 = -1.98e6,-1.73e6,.22e6,-.46e6
        elif name=='R2':
            x0,x1,y0,y1 = -1.70e6,-1.17e6,-.24e6,-1.2e6
        elif name=='R3':
            x0,x1,y0,y1 = -1.173e6,-.6e6,-1.11e6,-1.33e6
        elif name=='R4':
            x0,x1,y0,y1 = -.6e6,.4e6,-.43e6,-1.36e6            
        elif name=='R5':
            x0,x1,y0,y1 = .31e6,.54e6,-1.42e6,-1.97e6       
        elif name=='R6':
            x0,x1,y0,y1 = .4e6,1.9e6,-1.81e6,-2.18e6  
        elif name=='R7':
            x0,x1,y0,y1 = 1.9e6,2.5e6,-.75e6,-1.75e6  
        elif name=='R8':
            x0,x1,y0,y1 = 2.4e6,2.75e6,.4e6,-.65e6  
        elif name=='R9':
            x0,x1,y0,y1 = 1.67e6,2.25e6,.85e6,.5e6  
        elif name=='R10':
            x0,x1,y0,y1 = 1.78e6,2.2e6,1.7e6,1.15e6  
        elif name=='R11':
            x0,x1,y0,y1 = 0.345e6,1.45e6,2.2e6,1.65e6  
        elif name=='R12':
            x0,x1,y0,y1 = -.42e6,.345e6,2.245e6,1.975e6  
        elif name=='R13':
            x0,x1,y0,y1 = -.775e6,-.365e6,1.975e6,1.32e6  
        elif name=='R14':
            x0,x1,y0,y1 = -1.555e6,-.52e6,1.05e6,0.13e6  
        elif name=='R15':
            x0,x1,y0,y1 = -2.45e6,-1.555e6,1.4e6,0.85e6 
        elif name=='PIG':
            x0,x1,y0,y1 = -1.69e6,-1.55e6,-.24e6,-.38e6
        elif name=='CD':
            x0,x1,y0,y1 = -1.62e6,-1.48e6,-.53e6,-.70e6
        elif name=='Thwaites':
            x0,x1,y0,y1 = -1.61e6,-1.51e6,-.38e6,-.49e6

        self.ds = xr.open_dataset('../data/BedMachineAntarctica_2020-07-15_v02.nc')
        self.ds = self.ds.sel(x=slice(x0,x1),y=slice(y0,y1))

        self.dx = self.ds.x[1]-self.ds.x[0]
        self.dy = self.ds.y[1]-self.ds.y[0]
        if self.dx<0:
            print('inverting x-coordinates')
            self.ds = self.ds.reindex(x=list(reversed(self.ds.x)))
            self.dx = -self.dx
        if self.dy<0:
            print('inverting y-coordinates')
            self.ds = self.ds.reindex(y=list(reversed(self.ds.y)))
            self.dy = -self.dy

        #self.mask = self.ds.mask
        self.ds.mask[:] = xr.where(self.ds.mask==1,2,self.ds.mask)
        self.ds['draft'] = (self.ds.surface-self.ds.thickness).astype('float64')
        self.ds['thickness'] = self.ds.thickness.astype('float64')
        self.ds['surface'] = self.ds.surface.astype('float64')
        self.ds['bed'] = self.ds.bed.astype('float64')
        self.name = name
        ModelConstants.__init__(self)
    
    def coarsen(self,N):
        """Coarsen grid resolution by a factor N"""
        self.ds['mask'] = xr.where(self.ds.mask==0,np.nan,self.ds.mask)
        self.ds['draft'] = xr.where(np.isnan(self.ds.mask),np.nan,self.ds.draft)
        self.ds = self.ds.coarsen(x=N,y=N,boundary='trim').mean()

        self.ds['mask'] = np.round(self.ds.mask)
        self.ds['mask'] = xr.where(np.isnan(self.ds.mask),0,self.ds.mask)
        self.ds['draft'] = xr.where(self.ds.mask==0,0,self.ds.draft)
        self.ds['draft'] = xr.where(np.isnan(self.ds.draft),0,self.ds.draft)
        self.res *= N
        print(f'Resolution set to {self.res} km')
        
    def smoothen(self,N):
        """Smoothen geometry"""
        for n in range(0,N):
            self.ds.draft = .5*self.ds.draft + .125*(np.roll(self.ds.draft,-1,axis=0)+np.roll(self.ds.draft,1,axis=0)+np.roll(self.ds.draft,-1,axis=1)+np.roll(self.ds.draft,1,axis=1))

    def create(self):
        """Create geometry"""
        geom = self.ds[['mask','draft','surface','thickness','bed']]
        geom['name_geo'] = f'{self.name}_{self.res:1.1f}'
        print('Geometry',geom.name_geo.values,'created')
        
        #Add lon lat
        project = pyproj.Proj("epsg:3031")
        xx, yy = np.meshgrid(geom.x, geom.y)
        lons, lats = project(xx, yy, inverse=True)
        dims = ['y','x']
        geom = geom.assign_coords({'lat':(dims,lats), 'lon':(dims,lons)})  
        return geom