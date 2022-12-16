import numpy as np
import xarray as xr
import preprocess as pp
from tools import div0,ip_,jm_


def interpolate_icevel(object):
    #Interpolate observed ice velocity onto laddie grid

    ds0 = xr.open_dataset('../data/velocity/ANT_G0240_0000.nc')
    ds0 = ds0.reindex(y=list(reversed(ds0.y)))
    ds0 = ds0.sel(x=slice(np.min(object.x.values),np.max(object.x.values)),y=slice(np.min(object.y.values),np.max(object.y.values)))

    xx = ds0.x
    yy = ds0.y

    object.Uiobs = 0.*object.mask.copy()
    object.Viobs = 0.*object.mask.copy()

    for i,ii in enumerate(object.x.values):
        i0v = np.argmin((ii-ds0.x.values)**2)
        i0u = np.argmin((ii+object.dx/2-ds0.x.values)**2)
        for j,jj in enumerate(object.y.values):
            j0u = np.argmin((jj-ds0.y.values)**2)
            j0v = np.argmin((jj+object.dy/2-ds0.y.values)**2)
            object.Uiobs[j,i] = ds0.vx[j0u,i0u]
            object.Viobs[j,i] = ds0.vy[j0v,i0v]

    object.Uiobs = np.where(np.isnan(object.Uiobs),0,object.Uiobs)
    object.Viobs = np.where(np.isnan(object.Viobs),0,object.Viobs)

    ds0.close()
    return

def apply_calving(object):
    #Routine to remove ice shelf front grid cells with a thickness below 'calvlim'
    #Only calves from the ice shelf front

    pp.create_mask(object)

    calvmask = np.logical_and(object.H<object.calvlim,object.tmask*(object.ocnxm1+object.ocnxp1+object.ocnym1+object.ocnyp1)>0)
    while sum(sum(calvmask))>0:
        object.H = np.where(calvmask,0,object.H)
        object.mask = np.where(calvmask,0,object.mask)
        pp.create_mask(object)
        calvmask = np.logical_and(object.H<object.calvlim,object.tmask*(object.ocnxm1+object.ocnxp1+object.ocnym1+object.ocnyp1)>0)

    #Update ice surface, not sure this is necessary here
    object.zs = np.where(object.tmask==1,(1-object.rhoi/object.rho0)*object.H,object.zs)

    return

def create_imask(object):
    #Boundary mask
    object.bmask = np.ones((len(object.y),len(object.x)))
    object.bmask[0,:] = 0
    object.bmask[-1,:] = 0
    object.bmask[:,0] = 0
    object.bmask[:,-1] = 0

    #Ice mask (floating + grounded)
    object.imask = np.where(object.mask>0,1,0)*object.bmask

    #Ice velocity masks (floating + grounded + ice shelf front)
    object.uimask = object.umask*object.bmask
    object.vimask = object.vmask*object.bmask

    return

def nearedge(var,mask,roll,axis):
    #Tool to compute average value near ISF
    VV = np.roll(var*mask,roll,axis=axis)
    VM = np.roll(mask,roll,axis=axis)
    return VV + (1-VM) \
            * div0(np.roll(VV,1,axis=0) + np.roll(VV,-1,axis=0) + np.roll(VV,1,axis=1) + np.roll(VV,-1,axis=1),\
                   np.roll(VM,1,axis=0) + np.roll(VM,-1,axis=0) + np.roll(VM,1,axis=1) + np.roll(VM,-1,axis=1) )


def prepare_SSA(object,dhlim = .01):
    #Compute some variable that remain constant until geometry is updated

    #Probably not necessry:
    #object.zs = object.zs*object.imask + np.roll(object.zs,-1,axis=1)*object.isfW + np.roll(object.zs,-1,axis=0)*object.isfS

    #d/dx and d/dy of ice surface at U- and V-grids
    object.dhdx_u = (np.roll(object.zs,-1,axis=1)-object.zs)/object.dx * np.roll(object.uimask,1,axis=1) * object.bmask *(1-np.roll(object.isfW,1,axis=1)) *(1-np.roll(object.isfE,-1,axis=1))
    object.dhdy_v = (np.roll(object.zs,-1,axis=0)-object.zs)/object.dy * np.roll(object.vimask,1,axis=0) * object.bmask *(1-np.roll(object.isfS,1,axis=0)) *(1-np.roll(object.isfN,-1,axis=0))
    
    #Optionally add constraints on max and min dhdx, dhdy
    object.dhdx_u = np.maximum(-dhlim,np.minimum(dhlim,object.dhdx_u))
    object.dhdy_v = np.maximum(-dhlim,np.minimum(dhlim,object.dhdy_v))


    object.div_u = 8/object.dx**2 + 2/object.dy**2
    object.div_v = 8/object.dy**2 + 2/object.dx**2
    
    return

def get_ussa(object,U,V):
    #SSA solver for U 

    A = -object.rhoi*object.g/object.nui * object.dhdx_u

    VV = nearedge(V,object.vimask,0,0)
    Vxp1 = nearedge(V,object.vimask,-1,1)
    Vym1 = nearedge(V,object.vimask,1,0)
    Vxp1ym1 = nearedge(np.roll(V,-1,axis=1),np.roll(object.vimask,-1,axis=1),1,0)
    B = 3 * (Vxp1 + Vym1 - VV - Vxp1ym1) / (object.dx*object.dy)

    Uxp1 = np.where(object.isfE,np.maximum(0,U),np.roll(U,-1,axis=1))
    Uxm1 = np.where(object.isfW,np.minimum(0,U),np.roll(U,1,axis=1))
    C = 4 * (Uxp1 + Uxm1) / object.dx**2

    Uyp1 = np.where(np.roll(object.isfN,-1,axis=0),U,np.roll(U,-1,axis=0))
    Uym1 = np.where(object.isfS,U,np.roll(U,1,axis=0))
    D = (Uyp1 + Uym1) / object.dy**2

    return (A+B+C+D)/object.div_u * object.uimask

def get_vssa(object,U,V):
    #SSA solver for V 

    A = -object.rhoi*object.g/object.nui * object.dhdy_v

    UU = nearedge(U,object.uimask,0,0)
    Uyp1 = nearedge(U,object.uimask,-1,0)
    Uxm1 = nearedge(U,object.uimask,1,1)
    Uxm1yp1 = nearedge(np.roll(U,-1,axis=0),np.roll(object.uimask,-1,axis=0),1,1)
    B = 3 * (Uyp1 + Uxm1 - UU - Uxm1yp1) / (object.dx*object.dy)

    Vyp1 = np.where(object.isfN,np.maximum(0,V),np.roll(V,-1,axis=0))
    Vym1 = np.where(object.isfS,np.minimum(0,V),np.roll(V,1,axis=0))
    C = 4 * (Vyp1 + Vym1) / object.dy**2

    Vxp1 = np.where(np.roll(object.isfE,-1,axis=1),V,np.roll(V,-1,axis=1))
    Vxm1 = np.where(object.isfW,V,np.roll(V,1,axis=1))
    D = (Vxp1 + Vxm1) / object.dx**2

    return (A+B+C+D)/object.div_v * object.vimask


def get_nui(object,U,V,eps=1e-8,minnui=1e3):
    #Compute ice viscosity

    dudx = (U-np.roll(U,1,axis=1))/object.dx# * (1-laddie.isfW)
    dvdy = (V-np.roll(V,1,axis=0))/object.dy# * (1-laddie.isfS)

    vx = (V-np.roll(V,1,axis=1))/object.dx# * (1-laddie.isfW)
    uy = (U-np.roll(U,1,axis=0))/object.dy# * (1-laddie.isfS)

    dvdx = ip_(jm_(vx,object.bmask),object.bmask)
    dudy = ip_(jm_(uy,object.bmask),object.bmask)

    return np.maximum(minnui,(1-object.dam)*.5*object.Aglen**(-1/object.nglen) * (dudx**2+dvdy**2+dudx*dvdy+.25*(dudy+dvdx)**2+eps**2)**((1-object.nglen)/(2*object.nglen))) #* imask*mask2

def extrap_boundary(object):
    #Extrapolate U and V into boundary mask
    object.Ussa[1,0,:] = object.Ussa[1,1,:]
    object.Vssa[1,0,:] = object.Vssa[1,1,:]
    object.Ussa[1,-1,:] = object.Ussa[1,-2,:]
    object.Vssa[1,-1,:] = object.Vssa[1,-2,:]
    object.Ussa[1,:,0] = object.Ussa[1,:,1]
    object.Vssa[1,:,0] = object.Vssa[1,:,1]
    object.Ussa[1,:,-1] = object.Ussa[1,:,-2]
    object.Vssa[1,:,-1] = object.Vssa[1,:,-2]    

    #Corner points
    object.Ussa[1,0,0] = (object.Ussa[1,1,0]+object.Ussa[1,0,1])/2
    object.Vssa[1,0,0] = (object.Vssa[1,1,0]+object.Vssa[1,0,1])/2

    object.Ussa[1,-1,0] = (object.Ussa[1,-2,0]+object.Ussa[1,-1,1])/2
    object.Vssa[1,-1,0] = (object.Vssa[1,-2,0]+object.Vssa[1,-1,1])/2

    object.Ussa[1,0,-1] = (object.Ussa[1,1,-1]+object.Ussa[1,0,-2])/2
    object.Vssa[1,0,-1] = (object.Vssa[1,1,-1]+object.Vssa[1,0,-2])/2

    object.Ussa[1,-1,-1] = (object.Ussa[1,-2,-1]+object.Ussa[1,-1,-2])/2
    object.Vssa[1,-1,-1] = (object.Vssa[1,-2,-1]+object.Vssa[1,-1,-2])/2
    return 

def cut_icevel(object,limit=4000):
    #Cut ice velocity above maximum limit. Can probably be removed for SSA_float
    object.Ussa[1,:,:] = np.minimum(limit,object.Ussa[1,:,:])
    object.Vssa[1,:,:] = np.minimum(limit,object.Vssa[1,:,:])
    object.Ussa[1,:,:] = np.maximum(-limit,object.Ussa[1,:,:])
    object.Vssa[1,:,:] = np.maximum(-limit,object.Vssa[1,:,:])
    return

def calc_damage(object,dfac=1,Href=300,dlim=.9):
    #Routine to calculate continuum damage. Needs heavy revision
    dhdx_t = np.gradient(object.H,object.dx,axis=1)
    dhdy_t = np.gradient(object.H,object.dy,axis=0)

    ds = np.maximum(np.abs(dhdx_t),np.abs(dhdy_t))
    #dhlim = .1
    #ds = np.minimum(dhlim,ds)

    d1 = dfac*object.rho0/(object.rho0-object.rhoi)*ds
    #d1 = d1 * Href/np.maximum(100,object.H)

    object.dam = np.maximum(0,np.minimum(dlim,d1)) * (1-np.roll(object.isfW,1,axis=1)) * (1-np.roll(object.isfS,1,axis=0))

    return

def calc_damage2(object,dfac=10,dmin=.2,dmax=.9):
    #Routine to calculate continuum damage. Needs heavy revision
    vx = (object.Vssa[0,:,:]-np.roll(object.Vssa[0,:,:],1,axis=1))/object.dx
    uy = (object.Ussa[0,:,:]-np.roll(object.Ussa[0,:,:],1,axis=0))/object.dy

    dvdx = ip_(jm_(vx,object.bmask),object.bmask)
    dudy = ip_(jm_(uy,object.bmask),object.bmask)

    object.dam = dmin+np.maximum(0,np.minimum(dmax-dmin,dfac * (np.abs(dudy)+np.abs(dvdx))))*object.tmask

    return

def integrate_ice(object,errlim0=4000,errlim1=400,omega2=1,omega1=.4,calcdam = True):
    #Main iteration to compute ice velocities
    
    if calcdam:
        calc_damage(object,dfac=1)

    err0 = 1.e10
    jj = 0
    while err0>errlim0:
        err1 = 1.e6
        ii = 0
        while err1>errlim1:
            ussa = get_ussa(object,object.Ussa[0,:,:],object.Vssa[0,:,:])
            vssa = get_vssa(object,object.Ussa[0,:,:],object.Vssa[0,:,:])
            unew = (1-omega1) * object.Ussa[0,:,:] + omega1 * ussa
            vnew = (1-omega1) * object.Vssa[0,:,:] + omega1 * vssa
            object.Ussa[1,:,:] = object.uimask * unew + (1-object.uimask) * object.Uiobs
            object.Vssa[1,:,:] = object.vimask * vnew + (1-object.vimask) * object.Viobs

            #ideally remove the following two commands:
            extrap_boundary(object) #Extrapolate U,V to domain boundary | should be redundant because of masks
            cut_icevel(object)

            #Adjust to relative error
            err1 = sum(sum(np.abs(object.Ussa[1,:,:]-object.Ussa[0,:,:])))+sum(sum(np.abs(object.Vssa[1,:,:]-object.Vssa[0,:,:])))
            #print(jj,int(err0),ii,int(err1))

            #Roll for integration step
            object.Ussa = np.roll(object.Ussa,-1,axis=0)
            object.Vssa = np.roll(object.Vssa,-1,axis=0)

            ii += 1
            if ii>1000:
                break

        nuinew = (1-omega2) * object.nui + omega2 * get_nui(object,object.Ussa[0,:,:],object.Vssa[0,:,:])
        err0 = sum(sum(np.abs(nuinew-object.nui)))
        object.nui = nuinew

        jj += 1
        if jj>1000:
            break