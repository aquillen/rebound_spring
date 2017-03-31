
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import median_filter
from numpy import linalg as LA
#from matplotlib import rc
#rc('text', usetex=True)
from numpy import polyfit
from kepcart import *
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from outils import * # useful short routines

angfac = 180.0/np.pi # for converting radians to degrees

# read in a pointmass file  format fileroot_pm0.txt
def readpmfile(fileroot,npi):
    junk = '.txt'
    filename = "%s_pm%d%s"%(fileroot,npi,junk)
    print(filename)
    tt,x,y,z,vx,vy,vz,mm =\
           np.loadtxt(filename, skiprows=1, unpack='true') 
    return tt,x,y,z,vx,vy,vz,mm

# read in an extended mass output  file  format fileroot_ext.txt
def readresfile(fileroot):
    filename = fileroot+'_ext.txt'
    print(filename)
    t,x,y,z,vx,vy,vz,omx,omy,omz,llx,lly,llz,Ixx,Iyy,Izz,Ixy,Iyz,Ixz=\
        np.loadtxt(filename, skiprows=1, unpack='true') 
    return t,x,y,z,vx,vy,vz,omx,omy,omz,llx,lly,llz,Ixx,Iyy,Izz,Ixy,Iyz,Ixz


# read in all the point mass files at once
# return a mass array
# return time array
# return tuple of position and velocity vectors?
def readallpmfiles(fileroot,numberpm):
    mvec = np.zeros(0)    
    tt,x,y,z,vx,vy,vz,mm=readpmfile(fileroot,0)
    nt = len(tt)  # length of arrays
    mvec = np.append(mvec,mm[0])
    xarr = np.zeros((numberpm,nt))
    yarr = np.zeros((numberpm,nt))
    zarr = np.zeros((numberpm,nt))
    vxarr = np.zeros((numberpm,nt))
    vyarr = np.zeros((numberpm,nt))
    vzarr = np.zeros((numberpm,nt))
    xarr[0] = x
    yarr[0] = y
    zarr[0] = z
    vxarr[0] = vx
    vyarr[0] = vy
    vzarr[0] = vz
    for i in range(1,numberpm):
        ttt,x,y,z,vx,vy,vz,mm=readpmfile(fileroot,i)
        mvec = np.append(mvec,mm[0])
        xarr[i] = x
        yarr[i] = y
        zarr[i] = z
        vxarr[i] = vx
        vyarr[i] = vy
        vzarr[i] = vz

    return tt, mvec, xarr,yarr,zarr,vxarr,vyarr,vzarr


# fill arrays with orbital elements all w.r.t to first point mass
# which is assumed to be the central object
# resolved body orbit is put in first index of arrays
def orbels_arr(fileroot,numberpm):
    t,x,y,z,vx,vy,vz,omx,omy,omz,llx,lly,llz,Ixx,Iyy,Izz,Ixy,Iyz,Ixz=\
       readresfile(fileroot)  # resolved body stuff
    tt, mvec, xarr,yarr,zarr,vxarr,vyarr,vzarr=\
       readallpmfiles(fileroot,numberpm)  # point mass stuff
    aaarr = xarr*0.0
    eearr = xarr*0.0
    iiarr = xarr*0.0
    lnarr = xarr*0.0
    ararr = xarr*0.0
    maarr = xarr*0.0
    nl = len(tt)
    imc = 0  # index of central mass
    GM = mvec[imc] 
    # coordinates with respect to first point mass that is assumed to be central object
    dxarr = x- xarr[imc];  dyarr= y- yarr[imc];  dzarr= z- zarr[imc]
    dvxarr=vx-vxarr[imc]; dvyarr=vy-vyarr[imc]; dvzarr=vz-vzarr[imc]
    for k in range(nl):      # for the resolved body
        aa,ee,ii,longnode,argperi,meananom=\
               keplerian(GM,dxarr[k],dyarr[k],dzarr[k],dvxarr[k],dvyarr[k],dvzarr[k])
        aaarr[imc][k] = aa
        eearr[imc][k] = ee
        iiarr[imc][k] = ii
        lnarr[imc][k] = longnode
        ararr[imc][k] = argperi 
        maarr[imc][k] = meananom 

    for i in range(1,numberpm):
        dxarr = xarr[i]- xarr[imc];  dyarr= yarr[i] -  yarr[imc];  dzarr= zarr[i]- zarr[imc]
        dvxarr=vxarr[i]-vxarr[imc]; dvyarr=vyarr[i] - vyarr[imc]; dvzarr=vzarr[i]-vzarr[imc]
        for k in range(nl):    # for the point masses  
            dx  =  xarr[i][k] -  xarr[imc][k]
            dy  =  yarr[i][k] -  yarr[imc][k]
            dz  =  zarr[i][k] -  zarr[imc][k]
            dvx = vxarr[i][k] - vxarr[imc][k]
            dvy = vyarr[i][k] - vyarr[imc][k]
            dvz = vzarr[i][k] - vzarr[imc][k]
            aa,ee,ii,longnode,argperi,meananom=\
               keplerian(GM,dxarr[k],dyarr[k],dzarr[k],dvxarr[k],dvyarr[k],dvzarr[k])
            aaarr[i][k] = aa
            eearr[i][k] = ee
            iiarr[i][k] = ii
            lnarr[i][k] = longnode
            ararr[i][k] = argperi 
            maarr[i][k] = meananom 
    return t,mvec,aaarr,eearr,iiarr,lnarr,ararr,maarr


# computes obliquity,spin,J so far
# probably we also at some point want to compute
#  precession angle 
def give_body_angs(fileroot):
    # angle between body angular momentum and orbit normal
    t,x,y,z,vx,vy,vz,omx,omy,omz,llx,lly,llz,Ixx,Iyy,Izz,Ixy,Iyz,Ixz=\
       readresfile(fileroot)  # resolved body stuff
    numberpm=1
    tt, mvec, xarr,yarr,zarr,vxarr,vyarr,vzarr=\
       readallpmfiles(fileroot,numberpm)  # point mass stuff
    # normal to orbit
    imc = 0  # central mass
    dx = x - xarr[imc]; dvx = vx - vxarr[imc]
    dy = y - yarr[imc]; dvy = vy - vyarr[imc]
    dz = z - zarr[imc]; dvz = vz - vzarr[imc]
    no_x,no_y,no_z=crossprod_unit(dx,dy,dz,dvx,dvy,dvz)  #orbit normal
    nlx,nly,nlz = normalize_vec(llx,lly,llz)  # body spin angular momentum unit vect
    ang_so = dotprod(nlx,nly,nlz,no_x,no_y,no_z)
    ang_so = np.arccos(ang_so)*angfac   # obliquity  in degrees
    obliquity_deg = ang_so
    spin = len_vec(omx,omy,omz)
    # see tilts() on what this computes
    kb=1  # every 
    tvec_b,svec_ma,lvec_ma,svec_mi,lvec_mi,svec_me,lvec_me,ang_ll,gdot,ldot,\
            lam1dot, spinvec=\
            vec_tilts(kb,t,omx,omy,omz,llx,lly,llz,Ixx,Iyy,Izz,Ixy,Iyz,Ixz)
    Jvec = lvec_ma  # angle between angular momentum and principal body axis
    return obliquity_deg,spin,Jvec


def plt_cols(fileroot,numberpm,saveit,tmax):
    tt,mvec,aaarr,eearr,iiarr,lnarr,ararr,maarr=\
       orbels_arr(fileroot,numberpm)
    obliq_deg,spin,Jvec = give_body_angs(fileroot)

    ###########set up figure
    #plt.rcParams.update({'font.size': 14})
    f,axarr =  plt.subplots(2,2, dpi=100, figsize=(10,8), sharex=True)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.subplots_adjust(left=0.09, right=0.99, top=0.99, bottom=0.10, \
        wspace=0.22, hspace=0.0)
    xmin = 0.0; xmax = np.max(tt)
    if (tmax > 0):
        xmax = np.min([xmax,tmax])

    colorstr = ['k', 'r', 'b', 'g', 'm', 'c']
    ncolors = len(colorstr)

    il = 0; ih=0  # top left
    axarr[il,ih].set_ylabel('obliquity (deg)')
    axarr[il,ih].plot(tt,obliq_deg,'c.', ms=2) # label='')
    axarr[il,ih].set_xlim([xmin,xmax])

    il = 1; ih=0  # second left
    axarr[il,ih].set_ylabel('a e')
    for ip in range(numberpm):
        colorl = colorstr[ip%ncolors]
        axarr[il,ih].scatter(tt,aaarr[ip],color=colorl, s=1) # label='')
        ytop = aaarr[ip]*eearr[ip]
        ybot = ytop             
        axarr[il,ih].errorbar(tt,aaarr[ip],yerr=[ybot,ytop],\
            linestyle="None", marker="None", color=colorl)

    il = 0; ih=1  # top right
    axarr[il,ih].set_ylabel('spin')
    axarr[il,ih].plot(tt,spin,'r.', ms=2) # label='')

    il = 1; ih=0; axarr[il,ih].set_xlabel('time')
    il = 1; ih=1; axarr[il,ih].set_xlabel('time')


# make an inertial coordinate system, numbers not vectors returned
# total angular momentum is nearly fixed
def ntvec(lx,ly,lz,lox,loy,loz):
    lt_x = lx[0] + lox[0];  #sum the angular momentum of spin and orbit
    lt_y = ly[0] + loy[0];
    lt_z = lz[0] + loz[0];
    lt = len_vec(lt_x,lt_y,lt_z)
    nt_x = lt_x/lt  # unit vector
    nt_y = lt_y/lt
    nt_z = lt_z/lt
    #print("nt=",nt_x,nt_y,nt_z)   # unit vector in total ang momentum direction
    return nt_x,nt_y,nt_z  

# returns two vectors perpendicular to total angular momentum (or given vector)
# the given vector give need not be normalized
# one of the vectors returned is near x axis
# the two vectors returned are normalized and perpendicular to each other
def exy(nt_x,nt_y,nt_z):
    ex_x,ex_y,ex_z = aperp(1.0,0.0,0.0,nt_x,nt_y,nt_z)  # a vector near x
    ex_x,ex_y,ex_z = normalize_vec(ex_x,ex_y,ex_z)  #normalize
    #print ("ex=",ex_x,ex_y,ex_z)
    ey_x, ey_y, ey_z = crossprod_unit(nt_x,nt_y,nt_z,ex_x,ex_y,ex_z)
    #print ("ey=",ey_x,ey_y,ey_z)
    return  ex_x,ex_y,ex_z,ey_x,ey_y,ey_z


# project spin angular momentum onto ex,ey, all vectors
def precess_ang(lx,ly,lz,ex_x,ex_y,ex_z,ey_x,ey_y,ey_z):
    xproj = dotprod(lx,ly,lz,ex_x,ex_y,ex_z)
    yproj = dotprod(lx,ly,lz,ey_x,ey_y,ey_z)
    prec_ang = np.arctan2(yproj,xproj)
    return prec_ang


# median filter  the precession angle, returing precession rate, cleaned
def prec_dphidt(tt,prec_ang,boxsize):
    dt = tt[1] - tt[0]
    nn = np.size(tt)
    dphidt =np.diff(prec_ang)/dt  # precession rate
    dphidt =np.append(dphidt,dphidt[nn-2])# so array the same size all others 
    mf = median_filter(dphidt,boxsize)
    return mf


    
# at index j from moments of inertia arrays
# return eigenvector of max eigen value 
#    and eigenvector of min eigen value
#    and eigenvector of middle eigen value
# should now work if some eigenvalues are same as others
# these are the principal body axes
def evec(j,Ixx,Iyy,Izz,Ixy,Iyz,Ixz):
    Imat = np.matrix([[Ixx[j],Ixy[j],Ixz[j]],\
         [Ixy[j],Iyy[j],Iyz[j]],[Ixz[j],Iyz[j],Izz[j]]])
    w, v = LA.eig(Imat)  # eigenvecs v are unit length 
    jsort = np.argsort(w) # arguments of a sorted array of eigenvalues
    jmax = jsort[2]  # index of maximum eigenvalue
    jmin = jsort[0]  # index of minimum eigenvalue
    jmed = jsort[1]  # index of middle  eigenvalue
    vmax = np.squeeze(np.asarray(v[:,jmax]))   # corresponding eigenvector
    vmin = np.squeeze(np.asarray(v[:,jmin]))   # corresponding eigenvector
    vmed = np.squeeze(np.asarray(v[:,jmed]))   # corresponding eigenvector
    return vmax,vmin,vmed


# return eigenvalues!
# at index j
# order max,med,min
# these are I3,I2,I1 in order moments of inertia in body frame
def I3I2I1(j,Ixx,Iyy,Izz,Ixy,Iyz,Ixz):
    Imat = np.matrix([[Ixx[j],Ixy[j],Ixz[j]],\
         [Ixy[j],Iyy[j],Iyz[j]],[Ixz[j],Iyz[j],Izz[j]]])
    w, v = LA.eig(Imat)
    jsort = np.argsort(w) # arguments of a sorted array of eigenvalues
    jmax = jsort[2]  # index of maximum eigenvalue
    jmin = jsort[0]  # index of minimum eigenvalue
    jmed = jsort[1]  # index of middle  eigenvalue
    return w[jmax],w[jmed],w[jmin]


# to help give angles between 0 and pi/2
def piminus(ang):
    x = ang
    if (ang > np.pi/2.0):   # if angle greater than pi/2 returns pi-angle
        x = np.pi - ang
    return x

# body tilt angles with respect to body spin angular momentum and spin vectors
#   at index j 
# return acos of dot prod of spin omega with max principal axis
# return acos of dot prod of spin angular momentum with max principal axis
# and also returns same acosines for min and medium principal axis directions
def tilts(j,omx,omy,omz,llx,lly,llz,Ixx,Iyy,Izz,Ixy,Iyz,Ixz):
    slen = len_vec(omx[j],omy[j],omz[j])
    nox = omx[j]/slen;   # direction of omega (spin)
    noy = omy[j]/slen;
    noz = omz[j]/slen;
    llen = len_vec(llx[j],lly[j],llz[j])
    nlx = llx[j]/llen  # direction of spin angular momentum
    nly = lly[j]/llen
    nlz = llz[j]/llen
    vmax,vmin,vmed = evec(j,Ixx,Iyy,Izz,Ixy,Iyz,Ixz)  
    # evec returns eigenvectors of max and min and med eigenvalue of I matrix
    ds_ma =  dotprod(vmax[0],vmax[1],vmax[2],nox,noy,noz);  # cos = omega dot vmax
    dl_ma =  dotprod(vmax[0],vmax[1],vmax[2],nlx,nly,nlz);  # cos = angmom dot vmax
    # note that dl_ma is equivalent to cos J, 
    # with J the so-called non-principal rotation angle
    # see page 86 in Celletti's book
    ds_mi =  dotprod(vmin[0],vmin[1],vmin[2],nox,noy,noz);  # same but for vmin
    dl_mi =  dotprod(vmin[0],vmin[1],vmin[2],nlx,nly,nlz);  # "
    ds_me =  dotprod(vmed[0],vmed[1],vmed[2],nox,noy,noz);  # same but for vmed
    dl_me =  dotprod(vmed[0],vmed[1],vmed[2],nlx,nly,nlz);  # "
    angs_ma = piminus(np.arccos(ds_ma))    # return angles in range [0,pi/2]
    angl_ma = piminus(np.arccos(dl_ma))
    angs_mi = piminus(np.arccos(ds_mi))
    angl_mi = piminus(np.arccos(dl_mi))
    angs_me = piminus(np.arccos(ds_me))
    angl_me = piminus(np.arccos(dl_me))
    return angs_ma,angl_ma,angs_mi,angl_mi,angs_me,angl_me

# this angle is relevant for precession when spinning about a non-principal axis
# return the angle l conjugate to L (see page 86 of Celletti's book)
#   at array index j
# see Figure 5.2 by Celletti
def ll_vec(j,llx,lly,llz,Ixx,Iyy,Izz,Ixy,Iyz,Ixz):
    vmax,vmin,vmed = evec(j,Ixx,Iyy,Izz,Ixy,Iyz,Ixz)  
    llen = len_vec(llx[j],lly[j],llz[j])
    nlx = llx[j]/llen  # direction of spin angular momentum
    nly = lly[j]/llen
    nlz = llz[j]/llen
    # n_doublebar is vmax cross spin angular momentum direction nlx,nly,nlz
    ndd_x,ndd_y,ndd_z=crossprod_unit(vmax[0],vmax[1],vmax[2],nlx,nly,nlz)
    # we want vmin dotted with n_doublebar
    cosll =  dotprod(vmin[0],vmin[1],vmin[2],ndd_x,ndd_y,ndd_z); 
    ang_ll = piminus(np.arccos(cosll))
    return ang_ll


# return averaged values for gdot and ldot
# these are Andoyer Deprit angles spin and precession rates
# using equations on page 88 of book by Celletti but averaging
# over possible values for l
# at index j
def body_precs(j,llx,lly,llz,Ixx,Iyy,Izz,Ixy,Iyz,Ixz):
    I3,I2,I1 = I3I2I1(j,Ixx,Iyy,Izz,Ixy,Iyz,Ixz) 
    llen = len_vec(llx[j],lly[j],llz[j])
    G= llen   # spin angular momentum, and Andoyer Deprit variable
    nlx = llx[j]/llen  # direction of spin angular momentum
    nly = lly[j]/llen
    nlz = llz[j]/llen
    vmax,vmin,vmed = evec(j,Ixx,Iyy,Izz,Ixy,Iyz,Ixz)  
    # evec returns eigenvectors of max and min and med eigenvalue of I matrix
    cosJ =  dotprod(vmax[0],vmax[1],vmax[2],nlx,nly,nlz);  # cos = angmom dot vmax
    # J is the so-called non-principal rotation angle
    # see page 86 in Celletti's book
    L = np.abs(G*cosJ) # Andoyer Deprit variable
    inv_I_med = 0.5*(1.0/I1 + 1.0/I2);
    gdot =G*inv_I_med  # averaged over l page 88 Celletti
    ldot = L/I3 - L*inv_I_med # averaged over l 
    lambda1dot = L/I3 + G*inv_I_med - L*inv_I_med  # is gdot + ldot
    return gdot,ldot,lambda1dot


# vector of tilts
# do it for every k spacing in index, not every index (unless k=1)
# returns angles for largest eigendirection of I and minimum and medium
# returns angles for both omega and spin angular momentum 
# the angles are those between omega and eigendirections
#  or those between spin and eigendirections
# the eigendirections are the principal axes
def vec_tilts(k,tt,omx,omy,omz,llx,lly,llz,Ixx,Iyy,Izz,Ixy,Iyz,Ixz):
    nn = np.size(tt)
    nt = np.int(nn/k)
    svec_ma=[]
    lvec_ma=[]
    svec_mi=[]
    lvec_mi=[]
    svec_me=[]
    lvec_me=[]
    tvec=[]
    ang_ll_vec = []
    gdot_vec =[]
    ldot_vec =[]
    lam1dot_vec = []
    omvec = np.sqrt(omx*omx + omy*omy + omz*omz)
    Gvec = np.sqrt(llx*llx + lly*lly + llz*llz)
    spin_vec = []
    for i in range(nt):
        j = k*i
        angs_ma,angl_ma,angs_mi,angl_mi,angs_me,angl_me =\
              tilts(j,omx,omy,omz,llx,lly,llz,Ixx,Iyy,Izz,Ixy,Iyz,Ixz) 
        svec_ma = np.append(svec_ma,angs_ma)  #largest
        lvec_ma = np.append(lvec_ma,angl_ma)
        svec_mi = np.append(svec_mi,angs_mi)  #smallest
        lvec_mi = np.append(lvec_mi,angl_mi)
        svec_me = np.append(svec_me,angs_me)  #medium
        lvec_me = np.append(lvec_me,angl_me)
        spin_vec = np.append(spin_vec,omvec[j])
        tvec = np.append(tvec,tt[j])  #time
        ang_ll = ll_vec(j,llx,lly,llz,Ixx,Iyy,Izz,Ixy,Iyz,Ixz) 
        ang_ll_vec = np.append(ang_ll_vec,ang_ll)
        gdot,ldot,lam1dot = body_precs(j,llx,lly,llz,Ixx,Iyy,Izz,Ixy,Iyz,Ixz) 
        gdot_vec=np.append(gdot_vec,gdot)
        ldot_vec=np.append(ldot_vec,ldot)
        lam1dot_vec=np.append(lam1dot_vec,lam1dot)

    return tvec,svec_ma,lvec_ma,svec_mi,lvec_mi,svec_me,lvec_me,ang_ll_vec,gdot_vec,ldot_vec,lam1dot_vec,spin_vec




