import numpy as np
import xarray as xr

from constants import ModelConstants

class Forcing(ModelConstants):
    """ """

    def __init__(self, geom):
        """ 
        input:
        geom     (xr.Dataset)  geometry dataset from one of the Geometry classes
        output:
        self.ds  (xr.Dataset)  original `geom` with additional fields:
            mitgcm:
            Tz   (z,y,x)   [degC]  ambient temperature
            Sz   (z,y,x)   [psu]   ambient salinity
            tanh:
            Tz   (z)       [degC]  ambient temperature
            Sz   (z)       [psu]   ambient salinity            
        """
        assert 'draft' in geom
        self.ds = geom
        self.ds = self.ds.assign_coords({'z':np.arange(-5000.,0,1)})
        ModelConstants.__init__(self)
        return

    def sose(self,reg,corrcold = 0.):
        assert corrcold>=0
        assert corrcold<=1

        dsT = xr.open_dataset('../data/sose/bsose_i105_2008to2012_monthly_Theta.nc')
        dsS = xr.open_dataset('../data/sose/bsose_i105_2008to2012_monthly_Salt.nc')
        if reg=='R0':
            x,y = 860,100
        elif reg=='R1':
            x,y = 790,75
        elif reg=='R2':
            x,y = 760,50
        elif reg=='R3':
            x,y = 620,15
        elif reg=='R4':
            x,y = 530,5
        elif reg=='R5':
            x,y = 500,40
        elif reg=='R6':
            x,y = 480,100
        elif reg=='R7':
            x,y = 350,120
        elif reg=='R8':
            x,y = 280,120
        elif reg=='R9':
            x,y = 225,97
        elif reg=='R10':
            x,y = 180,115
        elif reg=='R11':
            x,y = 75,90
        elif reg=='R12':
            x,y = 10,90
        elif reg=='R13':
            x,y = 1040,80
        elif reg=='R14':
            x,y = 965,10
        elif reg=='R15':
            x,y = 905,90

        ds = xr.open_dataset('../data/sose/bsose_i105_2008to2012_monthly_Theta.nc')
        ds = ds.isel(XC=x,YC=y)
        Told = xr.where(ds.hFacC>0,ds.THETA,np.nan).mean(dim='time')
        idx = np.nanargmax(np.isnan(Told))
        Told[idx:] = Told[idx-1]
        self.ds['Tz'] = ('z',np.interp(self.ds.z,ds.Z[::-1],Told[::-1]))
        ds.close()
        self.ds['Tz_uncorr'] = self.ds.Tz.copy()
        self.ds['Tz'] += corrcold*self.l3 * self.ds.z

        ds = xr.open_dataset('../data/sose/bsose_i105_2008to2012_monthly_Salt.nc')
        ds = ds.isel(XC=x,YC=y)
        Sold = xr.where(ds.hFacC>0,ds.SALT,np.nan).mean(dim='time')
        idx = np.nanargmax(np.isnan(Sold))
        Sold[idx:] = Sold[idx-1]
        self.ds['Sz'] = ('z',np.interp(self.ds.z,ds.Z[::-1],Sold[::-1]))
        ds.close()
        self.ds.attrs['name_forcing'] = f'sose_{reg}'
        self.ds = self.calc_fields()
        return self.ds

    def tanh(self, ztcl, Tdeep, drhodz=.2/1000,z1=200):
        """ creates tanh thermocline forcing profile
        input:
        ztcl    ..  (float)  [m]       thermocline depth
        Tdeep   ..  (float)  [degC]    in situ temperature at depth
        drhodz  ..  (float)  [kg/m^4]  linear density stratification
        z1      ..  (float)  [m]       thermocline sharpness
        """
        if ztcl>0:
            print(f'z-coordinate is postive upwards; ztcl was {ztcl}, now set ztcl=-{ztcl}')
            ztcl = -ztcl
        S0 = 34.0                     # [psu]  reference surface salinity
        T0 = self.l1*S0+self.l2       # [degC] surface freezing temperature
        
        #drho = drhodz*np.abs(self.ds.z)
        drho = .01*np.abs(self.ds.z)**.5

        self.ds['Tz'] = Tdeep + (T0-Tdeep) * (1+np.tanh((self.ds.z-ztcl)/z1))/2
        self.ds['Sz'] = S0 + self.alpha*(self.ds.Tz-T0)/self.beta + drho/(self.beta*self.rho0)
        self.ds = self.calc_fields()
        self.ds.attrs['name_forcing'] = f'tanh_Tdeep{Tdeep:.1f}_ztcl{ztcl}'
        return self.ds
    
    def linear(self, S1,T1,z1=2000):
        """ creates linear forcing profile
        input:
        ztcl    ..  (float)  [m]       thermocline depth
        Tdeep   ..  (float)  [degC]    in situ temperature at depth
        drhodz  ..  (float)  [kg/m^4]  linear density stratification
        """
        if z1>0:
            print(f'z-coordinate is postive upwards; z1 was {z1}, now set z1=-{z1}')
            z1 = -z1
        S0 = 34.5                     # [psu]  reference surface salinity
        T0 = self.l1*S0+self.l2       # [degC] surface freezing temperature
        
        self.ds['Tz'] = T0 + self.ds.z*(T1-T0)/z1 
        self.ds['Sz'] = S0 + self.ds.z*(S1-S0)/z1
        self.ds = self.calc_fields()
        self.ds.attrs['name_forcing'] = f'linear_S1{S1:.1f}_T1{T1}'
        return self.ds
    
    def linear2(self, S1,T1,S0=33.8,T0=-1.9,z1=-720):
        """ creates 1D linear forcing profiles
        input:
        z1      ..  (float)  [m]       reference depth
        T0      ..  (float)  [degC]    temperature at the surface
        T1      ..  (float)  [degC]    temperature at depth
        S0      ..  (float)  [psu]     salinity at the surface
        S1      ..  (float)  [psu]     salinity at depth
        """
        if z1>0:
            print(f'z-coordinate is postive upwards; z1 was {z1}, now set z1=-{z1}')
            z1 = -z1
        
        if T1>T0:
            self.ds['Tz'] = np.minimum(T0 + self.ds.z*(T1-T0)/z1,T1)
        else:
            self.ds['Tz'] = np.maximum(T0 + self.ds.z*(T1-T0)/z1,T1)
        self.ds['Sz'] = np.minimum(S0 + self.ds.z*(S1-S0)/z1,S1)
        self.ds = self.calc_fields()
        self.ds.attrs['name_forcing'] = f'linear2_S1{S1:.1f}_T1{T1}'
        return self.ds

    def calc_fields(self):
        """ adds Ta/Sa fields to geometry dataset: forcing  = frac*COLD + (1-frac)*WARM """
        assert 'Tz' in self.ds
        assert 'Sz' in self.ds
        Sa = np.interp(self.ds.draft.values, self.ds.z.values, self.ds.Sz.values)
        Ta = np.interp(self.ds.draft.values, self.ds.z.values, self.ds.Tz.values)
        self.ds['Ta'] = (['y', 'x'], Ta)
        self.ds['Sa'] = (['y', 'x'], Sa)
        self.ds['Tf'] = self.l1*self.ds.Sa + self.l2 + self.l3*self.ds.draft  # l3 -> potential temperature
        self.ds.Ta.attrs = {'long_name':'ambient potential temperature' , 'units':'degC'}
        self.ds.Sa.attrs = {'long_name':'ambient salinity'              , 'units':'psu' }
        self.ds.Tf.attrs = {'long_name':'local potential freezing point', 'units':'degC'}  # from:Eq. 3 of Favier19
        return self.ds