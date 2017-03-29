
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

angfac = 180.0/np.pi

# some useful subroutines

# get an angle between [0,2pi]
def mod_two_pi(x):
    twopi = 2.0*np.pi
    y=x
    while (y > twopi):
        y -= twopi;
    while (y < 0.0):
        y += twopi;
    return y

def mod_two_pi_arr(x):
    nvec = np.size(x)
    mvec = x*0.0
    for i in range(0,nvec):
        mvec[i] = mod_two_pi(x[i])
    return mvec

# length of a vector
def len_vec(x,y,z):
    r= np.sqrt(x*x + y*y + z*z)
    return r

# normalize a vector
def normalize_vec(x,y,z):
    r = len_vec(x,y,z)
    return x/r, y/r, z/r

def dotprod(ax,ay,az,bx,by,bz):
    z = ax*bx + ay*by + az*bz  # dot product
    return z

# return cross product of two vectors
def crossprod(ax,ay,az,bx,by,bz):
    cx = ay*bz-az*by;
    cy = az*bx-ax*bz;
    cz = ax*by-ay*bx;
    return cx,cy,cz

# return normalized cross product of two vectors
def crossprod_unit(ax,ay,az,bx,by,bz):
    cx,cy,cz = crossprod(ax,ay,az,bx,by,bz)
    cc = len_vec(cx,cy,cz)
    return cx/cc, cy/cc, cz/cc

# return the vector part of a that is perpendicular to b direction
def aperp(ax,ay,az,bx,by,bz):
    z = ax*bx + ay*by + az*bz  # dot product = ab cos theta
    bmag = len_vec(bx,by,bz)
    #theta = np.acos(z/(amag*bmag))
    cx = ax - z*bx/bmag
    cy = ay - z*by/bmag
    cz = az - z*bz/bmag
    return cx,cy,cz

# return the vector part of a that is parallel to b direction
def apar(ax,ay,az,bx,by,bz):
    z = ax*bx + ay*by + az*bz  # dot product = ab cos theta
    bmag = len_vec(bx,by,bz)
    #theta = np.acos(z/(amag*bmag))
    cx = z*bx/bmag
    cy = z*by/bmag
    cz = z*bz/bmag
    return cx,cy,cz
    

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


# read in filename, these are _tab.txt files
# ii,ee orb elements w.r.t to cartesian coord system
def readfile(filename):
    tt,aa,nn,ee,ii,omx,omy,omz,A,B,C,E,lx,ly,lz,ang,lox,loy,loz\
        ,Ixx,Iyy,Izz,Ixy,Iyz,Ixz= \
        np.loadtxt(filename, skiprows=1, unpack='true')
    return tt,aa,nn,ee,ii,omx,omy,omz,A,B,C,E,lx,ly,lz,ang,lox,loy,loz


# read in _bin.txt file
def readbinfile2(filename):
    tt,xc,yc,zc,vxc,vyc,vzc, xs,ys,zs,vxs,vys,vzs,xpc,ypc,zpc,vxpc,vypc,vzpc,\
        llx,lly,llz,Ixx,Iyy,Izz,Ixy,Iyz,Ixz=\
        np.loadtxt(filename, skiprows=1, unpack='true')
    return tt,xc,yc,zc,vxc,vyc,vzc,xs,ys,zs,vxs,vys,vzs,xpc,ypc,zpc,vxpc,vypc,vzpc,\
        llx,lly,llz,Ixx,Iyy,Izz,Ixy,Iyz,Ixz



# return some useful vectors
# mc is central mass (sum of binary)
def useful_vecs(mc,aa,omx,omy,omz,ang):
    meanmotion = np.sqrt(mc+1.0)*aa**-1.5  # not corrected for bin quad
    spin = len_vec(omx,omy,omz)
    obldeg = angfac*ang  #obliquity vector degrees
    return meanmotion,spin,obldeg   #vectors


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

def compute_semi(GM,x,y,z,vx,vy,vz):
    r = np.sqrt(x*x + y*y + z*z)
    ke = 0.5*(vx*vx + vy*vy + vz*vz)
    pe = -GM/r
    E = ke+pe
    a = -0.5*GM/E
    lx = y*vz - z*vy;
    ly = z*vx - x*vz;
    lz = x*vy - y*vx;
    ltot2 = lx*lx + ly*ly + lz*lz
    e2 = 1.0 - ltot2/(a*GM)
    e = np.sqrt(np.abs(e2) + 1e-20)
    return a,e


# try to get an angular rotation rate from a single particle
# I am not sure that this will work for vectors, it did! joy!
def om_s(rx,ry,rz,vx,vy,vz):
    rpx,rpy,rpz=aperp(rx,ry,rz,vx,vy,vz)   # part of r perp to v
    # compute rp = r - (r dot hat v) r
    cx,cy,cz = crossprod(rpx,rpy,rpz,vx,vy,vz)  # rp x v
    rp = len_vec(rpx,rpy,rpz)
    rp2 = rp*rp
    # compute rp x v/|rp|^2
    omx = cx/rp2  # is this omega?
    omy = cy/rp2
    omz = cz/rp2
    return omx,omy,omz

# compute semi-major axis from position and velocity, given mass M 
# and assuming G=1
def semi(M,rx,ry,rz,vx,vy,vz):
    ke=0.5*(vx*vx + vy*vy + vz*vz)
    r = np.sqrt(rx*rx+ry*ry+rz*rz)
    pe = -M/r
    energy = ke + pe
    a = -0.5*M/energy
    return a 

# return some useful frequencies, based on means
# returns numbers not arrays
# correction here for mm
def useful_freqs(mB,qratio,aa,xpc,ypc,zpc,vxpc,vypc,vzpc):
    muBratio = qratio/(1.0 + qratio)**2  # ratio of reduced mass to binary mass
    a_mean = np.mean(aa)
    a_b = semi(mB,xpc,ypc,zpc,vxpc,vypc,vzpc)
    a_b_mean = np.mean(a_b)
    n_b = np.sqrt(mB)*a_b_mean**-1.5
    mm = np.sqrt(mB+1.0)*a_mean**-1.5 
    mufac = muBratio*(a_b_mean/a_mean)**2
    fac = 1.0 + (3.0/8.0)*mufac
    mm = mm*fac
    return mm,n_b,mufac


restorenb=0  #make 1 if want tight lims on spin plot
dowobble=1 # I_perp and I_para will be defined here!
I_perp = 0.51; I_para=0.77
dowobbleprime=1 


def plotspin(mc,qratio,froot,saveit,tmax,Ipara,Iperp):
    tfile=froot + '_tab.txt'
    bfile=froot + '_bin.txt'
    tt,aa,nn,ee,ii,omx,omy,omz,A,B,C,E,lx,ly,lz,ang,lox,loy,loz=readfile(tfile)
    meanmotion,spin,obldeg = useful_vecs(mc,aa,omx,omy,omz,ang)
    tt,xc,yc,zc,vxc,vyc,vzc,xs,ys,zs,vxs,vys,vzs,xpc,ypc,zpc,vxpc,vypc,vzpc,\
        llx,lly,llz,Ixx,Iyy,Izz,Ixy,Iyz,Ixz=readbinfile2(bfile)
    n_o,n_b,mufac=useful_freqs(mc,qratio,aa,xpc,ypc,zpc,vxpc,vypc,vzpc) # numbers
    correct_no = 1.0 + (3.0/8.0)*mufac  # for mm, multiply n_o_filt by this
    aB,eB = compute_semi(mc,xpc,ypc,zpc,vxpc,vypc,vzpc) # binary, vectors
    nB = np.sqrt(mc)/aB**1.5  #vector 
    boxs =30
    n_o_filt = median_filter(meanmotion,boxs)
    nB_filt = median_filter(nB,boxs)
    #body tilts
    kb=5
    tvec_b,svec_ma,lvec_ma,svec_mi,lvec_mi,svec_me,lvec_me,ang_ll,gdot,ldot,\
            lam1dot, spinvec=\
            vec_tilts(kb,tt,omx,omy,omz,llx,lly,llz,Ixx,Iyy,Izz,Ixy,Iyz,Ixz)
    line1 = Line2D(range(10), range(10), linestyle='-', color='black', label='spin')
    maxspin = np.max(spinvec); minspin = np.min(spinvec)
    myhandles=[line1]
    nbmark=0
    plt.figure(figsize=(5,3))
    plt.rcParams.update({'font.size': 14})
    plt.ylabel('spin')
    plt.xlabel('time')
    plt.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
    plt.plot(tvec_b,spinvec*np.cos(svec_ma),'.',color='orange',ms=1) 
    plt.plot(tt,spin,'k.',ms=1, label='spin')
    # for wobble res
    if (dowobbleprime==1):
        print("no=", np.mean(n_o_filt))
        print("nB=", np.mean(nB))
        for kindex in range(20,0,-1):
            div2 = 1.0
            nbres = I_perp/I_para*median_filter(nB*kindex/div2 + n_o_filt*correct_no*(1.0 - kindex/div2),20)
            nbresmax = np.max(nbres)
            nbresmin = np.min(nbres)
            nbresmean= np.mean(nbres)
            print(kindex,nbresmean)
            if (nbresmax < 1.1*maxspin) and (nbresmin > 0.9*minspin):
                plt.plot(tt[::20],nbres[::20],'-',color='green',ms=1)
                iBlabel = "JB=%d" % (kindex)
                line3 = Line2D(range(10),range(10),linestyle='-',color='green',label=iBlabel)
                myhandles = myhandles + [line3]
                nbmark=1
    if (nbmark==1):
        plt.legend(handles= myhandles, loc='best', \
            numpoints = 1, handlelength=0.5,prop={'size':12})
    if (saveit==1):
        ofile = froot+"_spin.png"
        print(ofile)
        plt.savefig(ofile)
    


def plotcols(mc,qratio,froot,saveit,tmax):
    tfile=froot + '_tab.txt'
    bfile=froot + '_bin.txt'
    tt,aa,nn,ee,ii,omx,omy,omz,A,B,C,E,lx,ly,lz,ang,lox,loy,loz=readfile(tfile)
    meanmotion,spin,obldeg = useful_vecs(mc,aa,omx,omy,omz,ang)
    # meanmotion here is not corrected for binary quad
    tt,xc,yc,zc,vxc,vyc,vzc,xs,ys,zs,vxs,vys,vzs,xpc,ypc,zpc,vxpc,vypc,vzpc,\
        llx,lly,llz,Ixx,Iyy,Izz,Ixy,Iyz,Ixz=readbinfile2(bfile)
    #coord system, inertial, numbers not vecs
    nt_x,nt_y,nt_z=ntvec(lx,ly,lz,lox,loy,loz)   #using sum of spin and orbit ang mom
    ex_x,ex_y,ex_z,ey_x,ey_y,ey_z=exy(nt_x,nt_y,nt_z)

    n_o,n_b,mufac=useful_freqs(mc,qratio,aa,xpc,ypc,zpc,vxpc,vypc,vzpc) # numbers
    # n_o is corrected for binary quad
    # mufac is mu_B/m_B*(a_B/a)^2 if you want to correct things here
    correct_no = 1.0 + (3.0/8.0)*mufac  # for mm, multiply n_o_filt by this
    aB,eB = compute_semi(mc,xpc,ypc,zpc,vxpc,vypc,vzpc) # binary, vectors
    nB = np.sqrt(mc)/aB**1.5  #vector 
    boxs =30
    n_o_filt = median_filter(meanmotion,boxs)
    # meanmotion here is not corrected for binary quad
    nB_filt = median_filter(nB,boxs)
    nnvec = len(n_o_filt)
    # get mean motion at beginning of simulation 
    if (nnvec > 100):
       no_start = np.median(n_o_filt[0:nnvec/10])
    else:
       no_start = np.median(n_o_filt)

    # normal to orbit
    no_x,no_y,no_z=crossprod_unit(xc,yc,zc,vxc,vyc,vzc)
    # normal to binary orbit
    noB_x,noB_y,noB_z=crossprod_unit(xpc,ypc,zpc,vxpc,vypc,vzpc)
    # angle between these two normals
    inclination = dotprod(no_x,no_y,no_z,noB_x,noB_y,noB_z)
    inclination = np.arccos(inclination)*angfac

    # angle between body angular momentum and orbit normal
    nlx,nly,nlz = normalize_vec(lx,ly,lz)  # body ang mom unit vect
    ang_so = dotprod(nlx,nly,nlz,no_x,no_y,no_z)
    ang_so = np.arccos(ang_so)*angfac   # obliquity  in degrees
    # angle between body angular momentum and binary orbit normal
    ang_sb = dotprod(nlx,nly,nlz,noB_x,noB_y,noB_z)
    ang_sb = np.arccos(ang_sb)*angfac   # obliquity_B  in degrees

    # get orbital elements, but with respect to xyz coordinate system
    k=5   # every 5 outputs
    # for orbit
    tvec,avec,evec,ivec,lnvec,arvec,mavec=\
         orbels_vec(k,mc,tt,xc,yc,zc,vxc,vyc,vzc)
    # for binary 
    tvecb,avecb,evecb,ivecb,lnvecb,arvecb,mavecb=\
         orbels_vec(k,mc,tt,xpc,ypc,zpc,vxpc,vypc,vzpc)
    # but these are not yet modulo 2 pi
    mean_long = lnvec+arvec+mavec
    mean_longb = lnvecb+arvecb+mavecb
    #no_vec = sqrt(mc)*avec**-1.5
    # some more precession rates
    dlndt = prec_dphidt(tvec,lnvec,5)  # cleaned dlongnode/dt
    dardt = prec_dphidt(tvec,arvec,5)  # cleaned dargperi/dt
    #dmadt = prec_dphidt(tvec,mavec,10)  # cleaned dM/dt 

    # precession angle of spin angular momentum vs our inertial coordinate system
    #prec_ang=precess_ang(lx,ly,lz,ex_x,ex_y,ex_z,ey_x,ey_y,ey_z)
    prec_ang=precess_ang(lx,ly,lz,1.0,0.0,0.0,0.0,1.0,0.0) # xyz coord system
    # spin precession rate, cleaned
    dphidt=prec_dphidt(tt,prec_ang,10)

    #body tilts
    kb=5
    tvec_b,svec_ma,lvec_ma,svec_mi,lvec_mi,svec_me,lvec_me,ang_ll,gdot,ldot,\
            lam1dot, spinvec=\
            vec_tilts(kb,tt,omx,omy,omz,llx,lly,llz,Ixx,Iyy,Izz,Ixy,Iyz,Ixz)

    ########################
    #set up figure 
    #plt.rc('text', usetex=True)
    plt.rcParams.update({'font.size': 14})
    f,axarr =  plt.subplots(4,2, dpi=100, figsize=(10,8), sharex=True)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.subplots_adjust(left=0.09, right=0.99, top=0.99, bottom=0.10, \
        wspace=0.22, hspace=0.0)
    #f.subplots_adjust(hspace=0)
    #plt.setp([a.get_xticklabels() for a in f.axes[-1:0]], visible=False)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
    xmin = np.min(tt)
    xmax = np.max(tt)
    if (tmax > 0):
        xmax = np.min([xmax,tmax])
    axarr[0,0].set_xlim([xmin,xmax]) 

    ########################
    il = 0; ih=0
    axarr[il,ih].set_ylabel('obliquity (deg)')
    axarr[il,ih].plot(tt,ang_sb,'c.', ms=1, label='w orbit') 
    axarr[il,ih].plot(tt[::5],ang_so[::5],'k.', ms=1, label='w binary') 
    obmax1 = np.max(ang_sb) ; obmax2 = np.max(ang_so)
    obmax = np.max([obmax1,obmax2])
    axarr[il,ih].set_ylim(-1.0,obmax+5)
    #axarr[il,ih].legend(loc='upper left', numpoints = 1 )
    line1 = Line2D(range(10), range(10), linestyle='-', color='black', label='orbit')
    line2 = Line2D(range(10), range(10), linestyle='-', color='cyan', label='binary')
    axarr[il,ih].legend(handles= [line1,line2], loc='best', \
        numpoints = 1,  handlelength=0.5, prop={'size':12})
    axarr[il,ih].yaxis.set_major_locator(MaxNLocator(prune='lower'))
    axarr[il,ih].yaxis.set_major_locator(MaxNLocator(prune='upper'))
    axarr[il,ih].yaxis.set_major_locator(MaxNLocator(nbins=6))

    il = 1; ih=0
    axarr[il,ih].set_ylabel('spin/meanmotion')
    tarr = np.array([0,np.max(tt)])
    imax = np.int(np.max(2*spin/n_o_filt)) 
    imin = np.int(np.min(2*spin/n_o_filt))
    for i in range(imin,imax+2):
        rarr = tarr*0.0 + 0.5*i
        axarr[il,ih].plot(tarr,rarr,'k-')
    ymin = np.min(spin/n_o_filt)-0.5
    ymax = np.max(spin/n_o_filt)+0.5
    axarr[il,ih].set_ylim(ymin,ymax)
    axarr[il,ih].plot(tt,spin/(n_o_filt*correct_no), 'g.',ms=1) 
    #axarr[il,ih].yaxis.set_major_locator(MaxNLocator(prune='lower'))
    #axarr[il,ih].yaxis.set_major_locator(MaxNLocator(prune='upper'))
    #axarr[il,ih].yaxis.set_major_locator(MaxNLocator(nbins=6))
    axarr[il,ih].yaxis.set_ticks(np.arange(imin/2, imax/2+1, 0.5))

    il = 2; ih=0
    axarr[il,ih].set_ylabel('inclination (deg)')
    #axarr[il,ih].plot(tt,ii*angfac, 'b.',ms=1)
    axarr[il,ih].plot(tt,inclination, 'b.',ms=1)
    axarr[il,ih].yaxis.set_major_locator(MaxNLocator(prune='lower'))
    axarr[il,ih].yaxis.set_major_locator(MaxNLocator(prune='upper'))
    axarr[il,ih].yaxis.set_major_locator(MaxNLocator(nbins=6))

    il = 3; ih=0
    # precession rates
    #axarr[il,ih].set_ylabel(r'$\dot \Omega, \dot \Omega_s$')
    axarr[il,ih].set_ylabel('precession rates')
    axarr[il,ih].plot(tt,dphidt/no_start, 'r.',ms=1)
    axarr[il,ih].plot(tvec,dlndt/no_start, 'b.',ms=1)
    line1 = Line2D(range(10), range(10), linestyle='-', color='red', label='body')
    line2 = Line2D(range(10), range(10), linestyle='-', color='blue', label='longnode')
    axarr[il,ih].legend(handles= [line1,line2], loc='best', \
        numpoints = 1, handlelength=0.5, prop={'size':8})
    axarr[il,ih].yaxis.set_major_locator(MaxNLocator(prune='lower'))
    axarr[il,ih].yaxis.set_major_locator(MaxNLocator(prune='upper'))
    axarr[il,ih].yaxis.set_major_locator(MaxNLocator(nbins=6))

    il = 0; ih=1  # top right
    #axarr[il,ih].set_ylabel(r'$n_B/n_o$')
    axarr[il,ih].set_ylabel('Period ratio (O/B)')
    axarr[il,ih].plot(tt,nB_filt/(n_o_filt*correct_no),'.',color='dodgerblue',ms=1)
    # n_o_filt is not corrected mm by bin quad
    axarr[il,ih].yaxis.set_major_locator(MaxNLocator(prune='lower'))
    axarr[il,ih].yaxis.set_major_locator(MaxNLocator(prune='upper'))
    axarr[il,ih].yaxis.set_major_locator(MaxNLocator(nbins=6))

    il = 1; ih=1
    axarr[il,ih].set_ylabel('orb eccentricity')
    #axarr[il,ih].plot(tt,eB, '.',color='sienna',ms=1)
    axarr[il,ih].plot(tt,ee, '.',color='blueviolet',ms=1)
    axarr[il,ih].yaxis.set_major_locator(MaxNLocator(prune='lower'))
    axarr[il,ih].yaxis.set_major_locator(MaxNLocator(prune='upper'))
    axarr[il,ih].yaxis.set_major_locator(MaxNLocator(nbins=6))

    il = 2; ih=1
    #axarr[il,ih].set_ylabel('orb semi major')
    #axarr[il,ih].plot(tvec,avec,'k.',ms=1)
    axarr[il,ih].set_ylabel('spin')
    maxspin = np.max(spin); minspin = np.min(spin)
    #print(maxspin,minspin)
    mms = 0.5  # size of something in plot line?
    # trying to make lines with Correia's resonances
    line1 = Line2D(range(10), range(10), linestyle='-', color='black', label='spin')
    myhandles=[line1]
    nbmark=0
    for kindex in range(40,1,-1):
        nbres = median_filter(nB*kindex/2.0 + n_o_filt*correct_no*(1.0 - kindex/2.0),20)
        nbresmax = np.max(nbres)
        nbresmin = np.min(nbres)
        if (nbresmax < maxspin) and (nbresmin > minspin):
            axarr[il,ih].plot(tt[::20],nbres[::20],'-',color='rosybrown',ms=mms)
            iBlabel = "kB=%d" % (kindex)
            line2 = Line2D(range(10),range(10),linestyle='-',color='rosybrown',label=iBlabel)
            myhandles = myhandles + [line2]
            nbmark=1
    for kindex in range(-10,-1,1):
        nbres = median_filter(nB*kindex/2.0 + n_o_filt*correct_no*(1.0 - kindex/2.0),20)
        nbresmax = np.max(nbres)
        nbresmin = np.min(nbres)
        if (nbresmax < maxspin) and (nbresmin > minspin):
            axarr[il,ih].plot(tt[::20],nbres[::20],'-',color='rosybrown',ms=mms)
            iBlabel = "kB=%d" % (kindex)
            line2 = Line2D(range(10),range(10),linestyle='-',color='rosybrown',label=iBlabel)
            myhandles = myhandles + [line2]
            nbmark=1

    # for wobble res
    if (dowobble==1):
        print("no=", np.mean(n_o_filt))
        print("nB=", np.mean(nB))
        for kindex in range(40,1,-1):
            div2 = 1.0
            nbres = I_perp/I_para*median_filter(nB*kindex/div2 + n_o_filt*correct_no*(1.0 - kindex/div2),20)
            nbresmax = np.max(nbres)
            nbresmin = np.min(nbres)
            if (nbresmax < maxspin) and (nbresmin > 0.8*minspin):
                axarr[il,ih].plot(tt[::20],nbres[::20],'-',color='green',ms=mms)
                iBlabel = "J=%d" % (kindex)
                line3 = Line2D(range(10),range(10),linestyle='-',color='green',label=iBlabel)
                myhandles = myhandles + [line3]
                nbmark=1
       
    if (restorenb==1):
        axarr[il,ih].set_ylim(0.9*minspin,1.0*maxspin)  # restore!!!!!!
    axarr[il,ih].plot(tvec_b,spinvec*np.cos(svec_ma),'.',color='orange',ms=1) 
    axarr[il,ih].plot(tt,spin,'k.',ms=1, label='spin')
    if (nbmark==1):
        axarr[il,ih].legend(handles= myhandles, loc='best', \
            numpoints = 1, handlelength=0.5,prop={'size':12})

    axarr[il,ih].yaxis.set_major_locator(MaxNLocator(prune='lower'))
    axarr[il,ih].yaxis.set_major_locator(MaxNLocator(prune='upper'))
    axarr[il,ih].yaxis.set_major_locator(MaxNLocator(nbins=6))


    il = 3; ih=1
    axarr[il,ih].set_ylabel('non-principal angle J')
    axarr[il,ih].plot(tvec_b,lvec_ma*angfac,'r.',ms=1)
    axarr[il,ih].plot(tvec_b,lvec_mi*angfac,'g.',ms=1)
    axarr[il,ih].set_ylim(-2.0,92.0)
    axarr[il,ih].yaxis.set_major_locator(MaxNLocator(prune='lower'))
    axarr[il,ih].yaxis.set_major_locator(MaxNLocator(prune='upper'))
    axarr[il,ih].yaxis.set_major_locator(MaxNLocator(nbins=6))

    il = 3; ih=0; axarr[il,ih].set_xlabel('time')
    il = 3; ih=1; axarr[il,ih].set_xlabel('time')

    if (saveit==1):
        ofile = froot+".png"
        print(ofile)
        plt.savefig(ofile)
    

def plot_resangs(mc,qratio,froot,saveit,tmax):
    tfile=froot + '_tab.txt'
    bfile=froot + '_bin.txt'
    tt,aa,nn,ee,ii,omx,omy,omz,A,B,C,E,lx,ly,lz,ang,lox,loy,loz=readfile(tfile)
    meanmotion,spin,obldeg = useful_vecs(mc,aa,omx,omy,omz,ang)
    # meanmotion here is not corrected for binary quad
    tt,xc,yc,zc,vxc,vyc,vzc,xs,ys,zs,vxs,vys,vzs,xpc,ypc,zpc,vxpc,vypc,vzpc,\
        llx,lly,llz,Ixx,Iyy,Izz,Ixy,Iyz,Ixz=readbinfile2(bfile)
    #coord system, inertial, numbers not vecs
    nt_x,nt_y,nt_z=ntvec(lx,ly,lz,lox,loy,loz)   #using sum of spin and orbit ang mom
    ex_x,ex_y,ex_z,ey_x,ey_y,ey_z=exy(nt_x,nt_y,nt_z)

    n_o,n_b,mufac=useful_freqs(mc,qratio,aa,xpc,ypc,zpc,vxpc,vypc,vzpc) # numbers
    # n_o is corrected for binary quad
    # mufac is mu_B/m_B*(a_B/a)^2 if you want to correct things here
    correct_no = 1.0 + (3.0/8.0)*mufac  # for mm, multiply n_o_filt by this
    aB,eB = compute_semi(mc,xpc,ypc,zpc,vxpc,vypc,vzpc) # binary, vectors
    nB = np.sqrt(mc)/aB**1.5  #vector 
    boxs =30
    n_o_filt = median_filter(meanmotion,boxs)
    # meanmotion here is not corrected for binary quad
    nB_filt = median_filter(nB,boxs)
    nnvec = len(n_o_filt)
    # get mean motion at beginning of simulation 
    if (nnvec > 100):
       no_start = np.median(n_o_filt[0:nnvec/10])
       nb_start = np.median(nB_filt[0:nnvec/10])
    else:
       no_start = np.median(n_o_filt)
       nb_start = np.median(nB_filt)

    jres = np.int(nb_start/no_start  + 0.5)
    print(jres)
    # normal to orbit
    no_x,no_y,no_z=crossprod_unit(xc,yc,zc,vxc,vyc,vzc)
    # normal to binary orbit
    noB_x,noB_y,noB_z=crossprod_unit(xpc,ypc,zpc,vxpc,vypc,vzpc)
    # angle between these two normals
    inclination = dotprod(no_x,no_y,no_z,noB_x,noB_y,noB_z)
    inclination = np.arccos(inclination)*angfac

    # angle between body angular momentum and orbit normal
    nlx,nly,nlz = normalize_vec(lx,ly,lz)  # body ang mom unit vect
    ang_so = dotprod(nlx,nly,nlz,no_x,no_y,no_z)
    ang_so = np.arccos(ang_so)*angfac   # obliquity  in degrees
    # angle between body angular momentum and binary orbit normal
    ang_sb = dotprod(nlx,nly,nlz,noB_x,noB_y,noB_z)
    ang_sb = np.arccos(ang_sb)*angfac   # obliquity_B  in degrees

    # get orbital elements, but with respect to xyz coordinate system
    k=1   # every k outputs
    # for orbit
    tvec,avec,evec,ivec,lnvec,arvec,mavec=\
         orbels_vec(k,mc,tt,xc,yc,zc,vxc,vyc,vzc)
    # for binary 
    tvecb,avecb,evecb,ivecb,lnvecb,arvecb,mavecb=\
         orbels_vec(k,mc,tt,xpc,ypc,zpc,vxpc,vypc,vzpc)
    # but these are not yet modulo 2 pi
    mean_long = lnvec+arvec+mavec
    mean_longb = lnvecb+arvecb+mavecb
    res_ang = jres*mean_long - mean_longb - (jres-1.0)*lnvec
    res_ang = mod_two_pi_arr(res_ang)
    #no_vec = sqrt(mc)*avec**-1.5
    # some more precession rates
    dlndt = prec_dphidt(tvec,lnvec,5)  # cleaned dlongnode/dt
    dardt = prec_dphidt(tvec,arvec,5)  # cleaned dargperi/dt
    #dmadt = prec_dphidt(tvec,mavec,10)  # cleaned dM/dt 

    # precession angle of spin angular momentum vs our inertial coordinate system
    #prec_ang=precess_ang(lx,ly,lz,ex_x,ex_y,ex_z,ey_x,ey_y,ey_z)
    prec_ang=precess_ang(lx,ly,lz,1.0,0.0,0.0,0.0,1.0,0.0) # xyz coord system
    # spin precession rate, cleaned
    dphidt=prec_dphidt(tt,prec_ang,10)
    spin_ang2 = jres*mean_long - mean_longb - (jres-1.0)*prec_ang
    spin_ang2 = mod_two_pi_arr(spin_ang2)
    spin_ang1 = jres*mean_long - mean_longb - (jres-2.0)*prec_ang - lnvec
    spin_ang1 = mod_two_pi_arr(spin_ang1)


    ########################
    #set up figure 
    plt.rcParams.update({'font.size': 14})
    f,axarr =  plt.subplots(5,1, dpi=200, figsize=(5.5,7), sharex=True)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.subplots_adjust(left=0.14, right=0.97, top=0.99, bottom=0.10, \
        wspace=0.22, hspace=0.0)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
    xmin = np.min(tt)
    xmax = np.max(tt)
    if (tmax > 0):
        xmax = np.min([xmax,tmax])
    axarr[0].set_xlim([xmin,xmax]) 

    ########################
    il = 0; 
    axarr[il].set_ylabel('obliquity (deg)')
    axarr[il].plot(tt,ang_sb,'c.', ms=1, label='w orbit') 
    axarr[il].plot(tt[::5],ang_so[::5],'k.', ms=1, label='w binary') 
    obmax1 = np.max(ang_sb) ; obmax2 = np.max(ang_so)
    obmax = np.max([obmax1,obmax2])
    axarr[il].set_ylim(-1.0,obmax+5)
    line1 = Line2D(range(10), range(10), linestyle='-', color='black', label='orbit')
    line2 = Line2D(range(10), range(10), linestyle='-', color='cyan', label='binary')
    axarr[il].legend(handles= [line1,line2], loc='best', \
        numpoints = 1,  handlelength=0.5, prop={'size':12})
    axarr[il].yaxis.set_major_locator(MaxNLocator(prune='lower'))
    axarr[il].yaxis.set_major_locator(MaxNLocator(prune='upper'))
    axarr[il].yaxis.set_major_locator(MaxNLocator(nbins=6))

    il = 1; 
    axarr[il].set_ylabel('spin 1')
    axarr[il].plot(tvec,spin_ang1, '.',color="darkgreen", ms=1)
    axarr[il].yaxis.set_major_locator(MaxNLocator(prune='lower'))
    axarr[il].yaxis.set_major_locator(MaxNLocator(prune='upper'))
    axarr[il].yaxis.set_major_locator(MaxNLocator(nbins=6))
    axarr[il].set_ylim([0.0,2.0*np.pi]) 
    axarr[il].set_yticks([0.0, np.pi])
    axarr[il].set_yticklabels(['$0$', r'$\pi$'])

    il = 2; 
    axarr[il].set_ylabel('spin 2')
    axarr[il].plot(tvec,spin_ang2, '.',color="maroon", ms=1)
    axarr[il].yaxis.set_major_locator(MaxNLocator(prune='lower'))
    axarr[il].yaxis.set_major_locator(MaxNLocator(prune='upper'))
    axarr[il].yaxis.set_major_locator(MaxNLocator(nbins=6))
    axarr[il].set_ylim([0.0,2.0*np.pi]) 
    axarr[il].set_yticks([0.0, np.pi])
    axarr[il].set_yticklabels(['$0$', r'$\pi$'])

    il = 3; 
    axarr[il].set_ylabel('inclination (deg)')
    axarr[il].plot(tt,inclination, 'b.',ms=1)
    axarr[il].yaxis.set_major_locator(MaxNLocator(prune='lower'))
    axarr[il].yaxis.set_major_locator(MaxNLocator(prune='upper'))
    axarr[il].yaxis.set_major_locator(MaxNLocator(nbins=6))

    il = 4; 
    axarr[il].set_ylabel('mm angle')
    axarr[il].plot(tvec,res_ang, '.',color="royalblue", ms=1)
    axarr[il].yaxis.set_major_locator(MaxNLocator(prune='lower'))
    axarr[il].yaxis.set_major_locator(MaxNLocator(prune='upper'))
    axarr[il].yaxis.set_major_locator(MaxNLocator(nbins=6))
    axarr[il].set_ylim([0.0,2.0*np.pi]) 
    axarr[il].set_yticks([0.0, np.pi])
    axarr[il].set_yticklabels(['$0$', r'$\pi$'])
    axarr[il].set_xlabel('time')


    if (saveit==1):
        ofile = froot+"_ang.png"
        print(ofile)
        plt.savefig(ofile)
    

def plot_resangs41(mc,qratio,froot,saveit,tmax):
    tfile=froot + '_tab.txt'
    bfile=froot + '_bin.txt'
    tt,aa,nn,ee,ii,omx,omy,omz,A,B,C,E,lx,ly,lz,ang,lox,loy,loz=readfile(tfile)
    meanmotion,spin,obldeg = useful_vecs(mc,aa,omx,omy,omz,ang)
    # meanmotion here is not corrected for binary quad
    tt,xc,yc,zc,vxc,vyc,vzc,xs,ys,zs,vxs,vys,vzs,xpc,ypc,zpc,vxpc,vypc,vzpc,\
        llx,lly,llz,Ixx,Iyy,Izz,Ixy,Iyz,Ixz=readbinfile2(bfile)
    #coord system, inertial, numbers not vecs
    nt_x,nt_y,nt_z=ntvec(lx,ly,lz,lox,loy,loz)   #using sum of spin and orbit ang mom
    ex_x,ex_y,ex_z,ey_x,ey_y,ey_z=exy(nt_x,nt_y,nt_z)

    n_o,n_b,mufac=useful_freqs(mc,qratio,aa,xpc,ypc,zpc,vxpc,vypc,vzpc) # numbers
    # n_o is corrected for binary quad
    # mufac is mu_B/m_B*(a_B/a)^2 if you want to correct things here
    correct_no = 1.0 + (3.0/8.0)*mufac  # for mm, multiply n_o_filt by this
    aB,eB = compute_semi(mc,xpc,ypc,zpc,vxpc,vypc,vzpc) # binary, vectors
    nB = np.sqrt(mc)/aB**1.5  #vector 
    boxs =30
    n_o_filt = median_filter(meanmotion,boxs)
    # meanmotion here is not corrected for binary quad
    nB_filt = median_filter(nB,boxs)
    nnvec = len(n_o_filt)
    # get mean motion at beginning of simulation 
    if (nnvec > 100):
       no_start = np.median(n_o_filt[0:nnvec/10])
       nb_start = np.median(nB_filt[0:nnvec/10])
    else:
       no_start = np.median(n_o_filt)
       nb_start = np.median(nB_filt)

    jres = np.int(nb_start/no_start  + 0.5)
    print(jres)
    # normal to orbit
    no_x,no_y,no_z=crossprod_unit(xc,yc,zc,vxc,vyc,vzc)
    # normal to binary orbit
    noB_x,noB_y,noB_z=crossprod_unit(xpc,ypc,zpc,vxpc,vypc,vzpc)
    # angle between these two normals
    inclination = dotprod(no_x,no_y,no_z,noB_x,noB_y,noB_z)
    inclination = np.arccos(inclination)*angfac

    # angle between body angular momentum and orbit normal
    nlx,nly,nlz = normalize_vec(lx,ly,lz)  # body ang mom unit vect
    ang_so = dotprod(nlx,nly,nlz,no_x,no_y,no_z)
    ang_so = np.arccos(ang_so)*angfac   # obliquity  in degrees
    # angle between body angular momentum and binary orbit normal
    ang_sb = dotprod(nlx,nly,nlz,noB_x,noB_y,noB_z)
    ang_sb = np.arccos(ang_sb)*angfac   # obliquity_B  in degrees

    # get orbital elements, but with respect to xyz coordinate system
    k=1   # every k outputs
    # for orbit
    tvec,avec,evec,ivec,lnvec,arvec,mavec=\
         orbels_vec(k,mc,tt,xc,yc,zc,vxc,vyc,vzc)
    # for binary 
    tvecb,avecb,evecb,ivecb,lnvecb,arvecb,mavecb=\
         orbels_vec(k,mc,tt,xpc,ypc,zpc,vxpc,vypc,vzpc)
    # but these are not yet modulo 2 pi
    mean_long = lnvec+arvec+mavec
    mean_longb = lnvecb+arvecb+mavecb
    res_ang = jres*mean_long - mean_longb - (jres-1.0)*lnvec
    res_ang = mod_two_pi_arr(res_ang)
    res_ang2 = jres*mean_long - mean_longb - (jres-2.0)*lnvec
    res_ang2 = mod_two_pi_arr(res_ang2)
    #no_vec = sqrt(mc)*avec**-1.5
    # some more precession rates
    dlndt = prec_dphidt(tvec,lnvec,5)  # cleaned dlongnode/dt
    dardt = prec_dphidt(tvec,arvec,5)  # cleaned dargperi/dt
    #dmadt = prec_dphidt(tvec,mavec,10)  # cleaned dM/dt 

    # precession angle of spin angular momentum vs our inertial coordinate system
    #prec_ang=precess_ang(lx,ly,lz,ex_x,ex_y,ex_z,ey_x,ey_y,ey_z)
    prec_ang=precess_ang(lx,ly,lz,1.0,0.0,0.0,0.0,1.0,0.0) # xyz coord system
    # spin precession rate, cleaned
    dphidt=prec_dphidt(tt,prec_ang,10)
    spin_ang2 = jres*mean_long - mean_longb - (jres-1.0)*prec_ang
    spin_ang2 = mod_two_pi_arr(spin_ang2)
    spin_ang1 = jres*mean_long - mean_longb - (jres-2.0)*prec_ang - lnvec
    spin_ang1 = mod_two_pi_arr(spin_ang1)
    spin_ang0 = jres*mean_long - mean_longb - (jres-3.0)*prec_ang - 2*lnvec
    spin_ang0 = mod_two_pi_arr(spin_ang0)


    ########################
    #set up figure 
    plt.rcParams.update({'font.size': 14})
    f,axarr =  plt.subplots(7,1, dpi=200, figsize=(5.5,8), sharex=True)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.subplots_adjust(left=0.14, right=0.97, top=0.99, bottom=0.10, \
        wspace=0.22, hspace=0.0)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
    xmin = np.min(tt)
    xmax = np.max(tt)
    if (tmax > 0):
        xmax = np.min([xmax,tmax])
    axarr[0].set_xlim([xmin,xmax]) 

    ########################
    il = 0; 
    axarr[il].set_ylabel('obliquity (deg)')
    axarr[il].plot(tt,ang_sb,'c.', ms=1, label='w orbit') 
    axarr[il].plot(tt[::5],ang_so[::5],'k.', ms=1, label='w binary') 
    obmax1 = np.max(ang_sb) ; obmax2 = np.max(ang_so)
    obmax = np.max([obmax1,obmax2])
    axarr[il].set_ylim(-1.0,obmax+5)
    line1 = Line2D(range(10), range(10), linestyle='-', color='black', label='orbit')
    line2 = Line2D(range(10), range(10), linestyle='-', color='cyan', label='binary')
    axarr[il].legend(handles= [line1,line2], loc='best', \
        numpoints = 1,  handlelength=0.5, prop={'size':12})
    axarr[il].yaxis.set_major_locator(MaxNLocator(prune='lower'))
    axarr[il].yaxis.set_major_locator(MaxNLocator(prune='upper'))
    axarr[il].yaxis.set_major_locator(MaxNLocator(nbins=6))

    il = 1; 
    axarr[il].set_ylabel('spin 1')
    axarr[il].plot(tvec,spin_ang1, '.',color="darkgreen", ms=1)
    axarr[il].yaxis.set_major_locator(MaxNLocator(prune='lower'))
    axarr[il].yaxis.set_major_locator(MaxNLocator(prune='upper'))
    axarr[il].yaxis.set_major_locator(MaxNLocator(nbins=6))
    axarr[il].set_ylim([0.0,2.0*np.pi]) 
    axarr[il].set_yticks([0.0, np.pi])
    axarr[il].set_yticklabels(['$0$', r'$\pi$'])

    il = 2; 
    axarr[il].set_ylabel('spin 2')
    axarr[il].plot(tvec,spin_ang2, '.',color="maroon", ms=1)
    axarr[il].yaxis.set_major_locator(MaxNLocator(prune='lower'))
    axarr[il].yaxis.set_major_locator(MaxNLocator(prune='upper'))
    axarr[il].yaxis.set_major_locator(MaxNLocator(nbins=6))
    axarr[il].set_ylim([0.0,2.0*np.pi]) 
    axarr[il].set_yticks([0.0, np.pi])
    axarr[il].set_yticklabels(['$0$', r'$\pi$'])

    il = 3; 
    axarr[il].set_ylabel('spin 3')
    axarr[il].plot(tvec,spin_ang0, '.',color="maroon", ms=1)
    axarr[il].yaxis.set_major_locator(MaxNLocator(prune='lower'))
    axarr[il].yaxis.set_major_locator(MaxNLocator(prune='upper'))
    axarr[il].yaxis.set_major_locator(MaxNLocator(nbins=6))
    axarr[il].set_ylim([0.0,2.0*np.pi]) 
    axarr[il].set_yticks([0.0, np.pi])
    axarr[il].set_yticklabels(['$0$', r'$\pi$'])

    il = 4; 
    axarr[il].set_ylabel('inclination (deg)')
    axarr[il].plot(tt,inclination, 'b.',ms=1)
    axarr[il].yaxis.set_major_locator(MaxNLocator(prune='lower'))
    axarr[il].yaxis.set_major_locator(MaxNLocator(prune='upper'))
    axarr[il].yaxis.set_major_locator(MaxNLocator(nbins=6))

    il = 5; 
    axarr[il].set_ylabel('mm angle')
    axarr[il].plot(tvec,res_ang, '.',color="royalblue", ms=1)
    axarr[il].yaxis.set_major_locator(MaxNLocator(prune='lower'))
    axarr[il].yaxis.set_major_locator(MaxNLocator(prune='upper'))
    axarr[il].yaxis.set_major_locator(MaxNLocator(nbins=6))
    axarr[il].set_ylim([0.0,2.0*np.pi]) 
    axarr[il].set_yticks([0.0, np.pi])
    axarr[il].set_yticklabels(['$0$', r'$\pi$'])
    axarr[il].set_xlabel('time')

    il = 6; 
    axarr[il].set_ylabel('mm angle 2')
    axarr[il].plot(tvec,res_ang2, '.',color="royalblue", ms=1)
    axarr[il].yaxis.set_major_locator(MaxNLocator(prune='lower'))
    axarr[il].yaxis.set_major_locator(MaxNLocator(prune='upper'))
    axarr[il].yaxis.set_major_locator(MaxNLocator(nbins=6))
    axarr[il].set_ylim([0.0,2.0*np.pi]) 
    axarr[il].set_yticks([0.0, np.pi])
    axarr[il].set_yticklabels(['$0$', r'$\pi$'])
    axarr[il].set_xlabel('time')


    if (saveit==1):
        ofile = froot+"_ang.png"
        print(ofile)
        plt.savefig(ofile)
    

def nice_freqs(mc,qratio,froot):
    tfile=froot + '_tab.txt'
    bfile=froot + '_bin.txt'
    tt,aa,nn,ee,ii,omx,omy,omz,A,B,C,E,lx,ly,lz,ang,lox,loy,loz=readfile(tfile)
    meanmotion,spin,obldeg = useful_vecs(mc,aa,omx,omy,omz,ang)
    tt,xc,yc,zc,vxc,vyc,vzc,xs,ys,zs,vxs,vys,vzs,xpc,ypc,zpc,vxpc,vypc,vzpc=readbinfile(bfile)
    n_o,n_b,mufac=useful_freqs(mc,qratio,aa,xpc,ypc,zpc,vxpc,vypc,vzpc)
    return tt,n_o,n_b,spin,obldeg


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


#  Rv is volumetric radius in km
def egk2(Rv):
    eg = 1.2*(Rv/1e3)**2    # in GPa
    E_ice = 4.0   # GPa mu of ice
    fac = 38.0*np.pi/3.0
    mutilde = fac*E_ice/eg
    k2 = 1.5/(mutilde + 1.0)
    return eg,k2


# GM_PC is G*MB in units of km^3/s^2
# Porb is in days is orbital period
# Rv is volumetric radius in km
# mratio is mass ratio with PC binary, unitless
# k2 unitless, Q unitless
def despin(GM_PC,Porb,Rv,mratio,k2,Q):
    day = 24.0*60.0*60.0
    om  = 2.0*np.pi/(Porb*day)
    a3 = GM_PC/om**2
    a = a3**0.333333333 # units km now
    print("a= ",a, "km")
    aratio  = a/Rv
    print("a/Rv= ", aratio);
    fac = 1.0/(15.0*np.pi)
    tdes =  fac*Porb*(aratio**4.5)*(mratio**1.5)*(Q/k2) #in days
    tdes = tdes/365.25 #in years
    return a,np.log10(tdes)


# compute asphericity, oblatenemss parm
# oblateness parm is q_eff
def aspher(ba,ca):
    asph = np.sqrt(3.0*(1.0 - ba**2)/(1.0 + ba**2))
    z = 1.0 + ba**2
    ob = (0.5*z - ca**2)/z 
    return asph,ob

def slope_spin(tt,spin):
    p,v=polyfit(tt,spin,1,full=False, cov=True)
    slope = p[0]
    slope_err = np.sqrt(v[0,0])/np.abs(slope) # percent error in slope measurement
    print("slope=", slope, "+- ",slope_err, "%")
    #print(np.sqrt(v[0,0])/slope)  # percent error in slope measurement
    return slope,slope_err

# vector of orbital elements
# do it for every k, not every index
def orbels_vec(k,mB,tt,xc,yc,zc,vxc,vyc,vzc):
    nn = np.size(tt)
    nt = np.int(nn/k)
    avec=[]
    evec=[]
    ivec=[]
    lnvec=[]
    arvec=[]
    mavec=[]
    tvec=[]
    GM = mB+1.0
    for i in range(nt):
        j = k*i
        x = xc[j]; y = yc[j]; z = zc[j]
        vx = vxc[j]; vy = vyc[j]; vz = vzc[j]
        aa,ee,ii,longnode,argperi,meananom=keplerian(GM,x,y,z,vx,vy,vz)
        avec = np.append(avec,aa)
        evec = np.append(evec,ee)
        ivec = np.append(ivec,ii)
        lnvec = np.append(lnvec,longnode)
        arvec = np.append(arvec,argperi)
        mavec = np.append(mavec,meananom)
        tvec = np.append(tvec,tt[j])

    return tvec,avec,evec,ivec,lnvec,arvec,mavec

def plotangs(mc,qratio,froot,saveit):
    tfile=froot + '_tab.txt'
    bfile=froot + '_bin.txt'
    tt,aa,nn,ee,ii,omx,omy,omz,A,B,C,E,lx,ly,lz,ang,lox,loy,loz=readfile(tfile)
    meanmotion,spin,obldeg = useful_vecs(mc,aa,omx,omy,omz,ang)
    tt,xc,yc,zc,vxc,vyc,vzc,xs,ys,zs,vxs,vys,vzs,xpc,ypc,zpc,vxpc,vypc,vzpc,\
        llx,lly,llz,Ixx,Iyy,Izz,Ixy,Iyz,Ixz=readbinfile2(bfile)

    # normal to binary orbit
    nb_x,nb_y,nb_z=crossprod_unit(xpc,ypc,zpc,vxpc,vypc,vzpc)

    # angle between body angular momentum and binary orbit normal
    nlx,nly,nlz = normalize_vec(lx,ly,lz)
    ang_sb = dotprod(nlx,nly,nlz,nb_x,nb_y,nb_z)
    ang_sb = np.arccos(ang_sb)*angfac   #in degrees

    # angle between binary orbit and body orbit 
    nlox,nloy,nloz = normalize_vec(lox,loy,loz)
    ang_bo = dotprod(nlox,nloy,nloz,nb_x,nb_y,nb_z)
    ang_bo = np.arccos(ang_bo)*angfac    #in degrees

    # angle between binary orbit and x axis
    ang_bx = dotprod(1.0,0,0,nb_x,nb_y,nb_z)
    ang_bx = np.arccos(ang_bx)*angfac    #in degrees

    # angle between body angular momentum and x axis
    ang_sx = dotprod(1.0,0,0,nlx,nly,nlz)
    ang_sx = np.arccos(ang_sx)*angfac      

    plt.figure(figsize=(10, 6), dpi=100)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.subplots_adjust(left=0.08, right=0.99, top=0.99, bottom=0.10, wspace=0.20, hspace=0.22)
 
    plt.subplot(221)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
    plt.plot(tt,ang_sb,'k.', ms=1)
    plt.ylabel("ang sb (deg)")
    plt.xlabel("time")
    plt.ylim(-1.0,181.0)

    plt.subplot(222)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
    plt.plot(tt,ang_bo,'b.', ms=1)
    plt.plot(tt,ang_bx,'g.', ms=1)
    plt.ylabel("ang bo, ang bx (deg)")
    plt.xlabel("time")
    plt.ylim(-1.0,181.0)

    plt.subplot(223)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
    plt.plot(tt,ang_sx,'g.', ms=1)
    plt.ylabel("ang sx (deg)")
    plt.xlabel("time")
    plt.ylim(-1.0,181.0)

    k=4  
    tvec,avec,evec,ivec,lnvec,arvec,mavec=orbels_vec(k,mc,tt,xc,yc,zc,vxc,vyc,vzc)
    dlndt = prec_dphidt(tvec,lnvec,5)  # cleaned dlongnode/dt
    dardt = prec_dphidt(tvec,arvec,5)  # cleaned dargperi/dt

    plt.subplot(224)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
    #plt.plot(tvec,lnvec,'m.', ms=1)
    #plt.plot(tvec,arvec,'g.', ms=1)
    plt.plot(tvec,dlndt,'c.', ms=1)
    plt.plot(tvec,dardt,'r.', ms=1)
    #plt.ylabel("ang bx (deg)")
    plt.xlabel("time")
    #plt.ylim(-1.0,181.0)


