
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "rebound.h"
#include "spring.h"
#include "tools.h"
#include "output.h"
#include "kepcart.h"


extern int NS; // number of springs
int NSmax=0; // springs
extern int NPERT;  // number of external point masses 

// delete a spring
// here i is the index of the spring
void del_spring(struct reb_simulation* const r, int i){
   if (NS >0){
      springs[i] = springs[NS-1];
      NS--;
   }
}

// return spring length
double spring_length(struct reb_simulation* const r, struct spring spr){
   struct reb_particle* particles = r->particles;
   int ii = spr.i;
   int jj = spr.j;
   double xi = particles[ii].x; double yi = particles[ii].y; double zi = particles[ii].z;
   double xj = particles[jj].x; double yj = particles[jj].y; double zj = particles[jj].z;
   double dr = sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));
   return dr;
}

// compute spring midpoint, radius and angles from center position
void spr_ang_mid(struct reb_simulation* const r, struct spring spr, 
   double xc, double yc, double zc, 
   double *rmid, double *thetamid, double *phimid)
{
   struct reb_particle* particles = r->particles;
   int ii = spr.i;
   int jj = spr.j;
   double xi = particles[ii].x; double yi = particles[ii].y; double zi = particles[ii].z;
   double xj = particles[jj].x; double yj = particles[jj].y; double zj = particles[jj].z;
   double xmid = 0.5*(xi+xj) - xc;  // midpoint of spring from center
   double ymid = 0.5*(yi+yj) - yc;
   double zmid = 0.5*(zi+zj) - zc;
   double rc = sqrt(xmid*xmid + ymid*ymid + zmid*zmid);
   double theta = acos(zmid/rc);
   double phi = atan2(ymid,xmid);
   *rmid = rc;
   *thetamid = theta;
   *phimid = phi;
}

// compute spring midpoint, xyz from center position
void spr_xyz_mid(struct reb_simulation* const r, struct spring spr, 
   double xc, double yc, double zc, 
   double *xmid, double *ymid, double *zmid)
{
   struct reb_particle* particles = r->particles;
   int ii = spr.i;
   int jj = spr.j;
   double xi = particles[ii].x; double yi = particles[ii].y; double zi = particles[ii].z;
   double xj = particles[jj].x; double yj = particles[jj].y; double zj = particles[jj].z;
   double xm = 0.5*(xi+xj) - xc;  // midpoint of spring from center
   double ym = 0.5*(yi+yj) - yc;
   double zm = 0.5*(zi+zj) - zc;
   *xmid = xm; *ymid = ym; *zmid = zm;
}


//  normalize vector (coordinates of particle)
void normalize(struct reb_simulation* const r, struct reb_particle *pt){
  double rad = sqrt(pt->x*pt->x + pt->y*pt->y + pt->z*pt->z);
  pt->x /= rad; pt->y /= rad; pt->z /= rad;
}


// compute the closest distance between any pair of particles
// with index in range [imin,imax-1]
double mindist(struct reb_simulation* const r,int imin, int imax){
  double dist  = 1e10;
  struct reb_particle* particles = r->particles;
  for (int i=imin;i<imax-1;i++){
    for (int j=i+1;j<imax;j++){
       double dx = particles[i].x - particles[j].x;
       double dy = particles[i].y - particles[j].y;
       double dz = particles[i].z - particles[j].z;
       double dr = sqrt(dx*dx + dy*dy + dz*dz);
       if (dr < dist) dist=dr;
    }
  }
  return dist;
}

// add a spring! bare routine no checking
void springs_add(struct reb_simulation* const r,struct spring spr){
   while (NSmax<=NS){
            NSmax += 128;
            springs = realloc(springs,sizeof(struct spring)*NSmax);
   }
   springs[NS] = spr;
   NS++;
    
}

// add a spring given two indices of particles 
// checking that a spring doesn't already exist between these two particles
// set the natural distance of the spring to the current inter particle distance
// spring constant is not scaled by anything 
// return index of spring if added
// return index of spring if already existing
// return -1 if not made because both indices are the same, =bad
// takes as argument spring_vals to set spring constant and stuff
// i1,i2 are indexes of 2 particles, spring connects these
int add_spring_i(struct reb_simulation* const r,int i1, int i2,  struct spring spring_vals)
{
   if (i1==i2) return -1; // don't add  spring, vertices same
   // make sure order of indices is correct
   int il = i1;
   int ih = i2;
   if (i2<i1) { 
     il = i2; ih = i1; // order of indices
   }
   // check if these two particles are already connected
   for(int ii=0;ii<NS;ii++){ // there is another spring already connecting 
                             // these two indices
      if ((springs[ii].i == il) && (springs[ii].j == ih)) return ii;
   }
   // there is not another spring connecting the two indices!
   struct spring spr = spring_vals;
   spr.i = il;
   spr.j = ih;
   double dr = spring_length(r,spr); // spring length
   spr.rs0 = dr;       // rest spring length  
   spr.ks = spring_vals.ks;  // spring constant 
   springs_add(r,spr);
   return NS-1; // index of new spring!
   // printf("add_spring_i: NS=%d\n",NS);
}


// compute center of mass coordinates in particle range [il,ih-1]
// values returned in xc, yc, zc
void compute_com(struct reb_simulation* const r,int il, int ih, double *xc, double *yc, double *zc){
   struct reb_particle* particles = r->particles;
   double xsum = 0.0; double ysum = 0.0; double zsum = 0.0;
   double msum = 0.0;
   for(int i=il;i<ih;i++){
      xsum += particles[i].x*particles[i].m;
      ysum += particles[i].y*particles[i].m;
      zsum += particles[i].z*particles[i].m;
      msum += particles[i].m;
   }
   *xc = xsum/msum;
   *yc = ysum/msum;
   *zc = zsum/msum;
}

// compute total mass of particles in particle range [il,ih)
double sum_mass(struct reb_simulation* const r,int il, int ih) 
{
   struct reb_particle* particles = r->particles;
   double msum = 0.0;
   for(int i=il;i<ih;i++){
      msum += particles[i].m;
   }
   return msum;
}

// compute center of velocity particles in particle range [il,ih-1]
// values returned in vxc, vyc, vzc
void compute_cov(struct reb_simulation* const r,int il, int ih, 
   double *vxc, double *vyc, double *vzc){
   double vxsum = 0.0; double vysum = 0.0; double vzsum = 0.0;
   double msum = 0.0;
   struct reb_particle* particles = r->particles;
   for(int i=il;i<ih;i++){
      vxsum += particles[i].vx*particles[i].m;
      vysum += particles[i].vy*particles[i].m;
      vzsum += particles[i].vz*particles[i].m;
      msum += particles[i].m;
   }
   *vxc = vxsum/msum;
   *vyc = vysum/msum;
   *vzc = vzsum/msum;
}


// go to coordinate frame of body defined by vertices/particles [il,ih-1]
// mass weighted center of mass
// only coordinates changed,  particle velocities not changed
// all particles are shifted, not just the extended body
void centerbody(struct reb_simulation* const r,int il, int ih){
   double xc =0.0; double yc =0.0; double zc =0.0;
   compute_com(r,il, ih, &xc, &yc, &zc);
   for(int i=0;i<(r->N);i++){ // all particles shifted
      r->particles[i].x -= xc; 
      r->particles[i].y -= yc;
      r->particles[i].z -= zc; 
   } 
}

// subtract center of velocity from the resolved body
// only changing particles in the resolved body [il,ih)
void subtractcov(struct reb_simulation* const r,int il, int ih){
   double vxc =0.0; double vyc =0.0; double vzc =0.0;
   compute_cov(r,il, ih, &vxc, &vyc, &vzc); // center of velocity of resolved body
   move_resolved(r,0.0,0.0,0.0,-vxc,-vyc,-vzc, il,ih);
}

// subtract center of mass position from the resolved body
// only changing particles in the resolved body [il,ih)
void subtractcom(struct reb_simulation* const r,int il, int ih){
   double xc =0.0; double yc =0.0; double zc =0.0;
   compute_com(r,il, ih, &xc, &yc, &zc); // center of velocity of resolved body
   move_resolved(r,-xc,-yc,-zc,0.0,0.0,0.0,il,ih);
}


// return spring strain
double strain(struct reb_simulation* const r,struct spring spr){
  double dr = spring_length(r,spr); // spring length
  return (dr-spr.rs0)/spr.rs0; // is positive under extension  
}


// calculate spring forces
// viscoelastic model, Clavet et al.  05
// "Particle-based Viscoelastic Fluid Simulation"
// Eurographics/ACM SIGGRAPH Symposium on Computer Animation (2005)
// K. Anjyo, P. Faloutsos (Editors)
// by Simon Clavet, Philippe Beaudoin, and Pierre Poulin 
#define L_EPS 1e-6; // softening for spring length
void spring_forces(struct reb_simulation* const r){
     for (int i=0;i<NS;i++){  // spring forces  
	    double L = spring_length(r,springs[i]) + L_EPS; // spring length
	    double rs0 = springs[i].rs0; // rest length
	    int ii = springs[i].i; int jj = springs[i].j;
            double dx = r->particles[ii].x - r->particles[jj].x;
            double dy = r->particles[ii].y - r->particles[jj].y;
            double dz = r->particles[ii].z - r->particles[jj].z;
	    double mii = r->particles[ii].m;
	    double mjj = r->particles[jj].m;
	    double ks = springs[i].ks;
	    double fac = -ks*(L-rs0)/L; // L here to normalize direction
            // apply elastic forces
            // accelerations are force divided by mass
	    r->particles[ii].ax +=  fac*dx/mii; r->particles[jj].ax -= fac*dx/mjj;
	    r->particles[ii].ay +=  fac*dy/mii; r->particles[jj].ay -= fac*dy/mjj;
	    r->particles[ii].az +=  fac*dz/mii; r->particles[jj].az -= fac*dz/mjj;
        
            // apply damping, depends on strain rate
            double gamma = springs[i].gamma;
            if (gamma>0.0) {
		  double dvx = r->particles[ii].vx - r->particles[jj].vx;
		  double dvy = r->particles[ii].vy - r->particles[jj].vy;
		  double dvz = r->particles[ii].vz - r->particles[jj].vz;
                  double dLdt = (dx*dvx + dy*dvy + dz*dvz)/L;  
                     // divide dL/dt by L to get strain rate
                  double mbar = mii*mjj/(mii+mjj); // reduced mass
                  double dampfac  = gamma*mbar*dLdt/L;   
                      // factor L here to normalize dx,dy,dz
		  r->particles[ii].ax -=  dampfac*dx/mii; r->particles[jj].ax += dampfac*dx/mjj;
		  r->particles[ii].ay -=  dampfac*dy/mii; r->particles[jj].ay += dampfac*dy/mjj;
		  r->particles[ii].az -=  dampfac*dz/mii; r->particles[jj].az += dampfac*dz/mjj;
                  // gamma is in units of 1/time
                  // force is gamma*dL/dt*mbar * dx/L = gamma*mbar*deps/dt*L*dx/L
             }
          
    }
}

//////////////////////////////////
// compute power lost in damping from a specific spring
// due to damping (viscoelasticity)
double dEdt(struct reb_simulation* const r,struct spring spr){
         double gamma = spr.gamma;
         if (gamma==0.0) return 0.0;

	 double L = spring_length(r,spr) + L_EPS; // spring length
	 int ii = spr.i; int jj = spr.j;
         double dx = r->particles[ii].x - r->particles[jj].x;
         double dy = r->particles[ii].y - r->particles[jj].y;
         double dz = r->particles[ii].z - r->particles[jj].z;
         // double dr = sqrt(dx*dx + dy*dy + dz*dz);
	 double mii = r->particles[ii].m;
	 double mjj = r->particles[jj].m;
         double mbar = mii*mjj/(mii+mjj); // reduced mass
	 double dvx = r->particles[ii].vx - r->particles[jj].vx;
	 double dvy = r->particles[ii].vy - r->particles[jj].vy;
	 double dvz = r->particles[ii].vz - r->particles[jj].vz;
         double dLdt = (dx*dvx + dy*dvy + dz*dvz)/L;  
                     // divide dL/dt by L to get strain rate
         double de = gamma*mbar*dLdt*dLdt; 
	 return de;  // units power de/dt as expected
                     // we do not need to multiply by timestep to get de/dt
}



// zero all particle accelerations
void zero_accel(struct reb_simulation* const r){
  for (int i=0;i<(r->N);i++){
     r->particles[i].ax = 0.0;
     r->particles[i].ay = 0.0;
     r->particles[i].az = 0.0;
  }
}


// connect springs to all particles with interparticle
// distances less than h_dist apart
// for particle index range [i0,imax-1]
// spring added with rest length at current length  
void connect_springs_dist(struct reb_simulation* const r, double h_dist, int i0, int imax,
      struct spring spring_vals)
{
   if (imax <= i0) return;
// find all the springs for near neighbors
   for(int ii=i0;ii<imax-1;ii++){
	double xi =  r->particles[ii].x;
	double yi =  r->particles[ii].y;
	double zi =  r->particles[ii].z;
   	for(int jj=ii+1;jj<imax;jj++){ // all pairs
	   double xj =  r->particles[jj].x;
	   double yj =  r->particles[jj].y;
	   double zj =  r->particles[jj].z;
	   double dr = sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));
           if (dr < h_dist){ 	   // try to add the spring 
              add_spring_i(r,ii, jj, spring_vals);
              // spring added at rest distance
           }
       }
   }
   // printf("NS=%d\n",NS);
}

// create a ~uniform random particle distribution with total mass  = total_mass
// fill particles within a football shape given by semi- axes ax, by, cz
// spacing set by parameter dist: no closer particles allowed
// to make this uniform first create a sphere that is large
// enough to hold entire football
// this should reduce non-uniformity from surface effects
void rand_football_from_sphere(struct reb_simulation* r, double dist, 
       double ax, double by, double cz, double total_mass)
{
   double rhold = ax + 2*dist;  // size of sphere that we need to hold everything
   struct reb_particle pt;
   int npart = (int)(40.0*pow(2.0*rhold/dist,3.0)); 
       // guess for number of random particles we need to generate
   pt.ax = 0.0; pt.ay = 0.0; pt.az = 0.0;
   pt.vx = 0.0; pt.vy = 0.0; pt.vz = 0.0;
   pt.m = 1.0;
   double particle_radius = dist/2.0;
   pt.r = particle_radius; 
   int N = r->N;
   int i0 = N;
   for(int i=0;i<npart;i++){
       pt.x = reb_random_uniform(-rhold,rhold);
       pt.y = reb_random_uniform(-rhold,rhold);
       pt.z = reb_random_uniform(-rhold,rhold);
       double x2 = pow(pt.x,2.0);
       double y2 = pow(pt.y,2.0);
       double z2 = pow(pt.z,2.0);
       double rval  =  sqrt(x2 + y2 + z2);
       if (rval < rhold){ // within rhold 
	  int toonear = 0;  // is there a particle too nearby?
          int j=i0;  
          N = r->N;
          while ((toonear==0) && (j < N)){
             double dx =  pt.x - r->particles[j].x;
             double dy =  pt.y - r->particles[j].y;
             double dz =  pt.z - r->particles[j].z;
             double dr = sqrt(dx*dx + dy*dy + dz*dz);
	     if (dr < dist) toonear=1;
	     j++;
          }
	  if (toonear==0) reb_add(r,pt); 
          // only add particle if not near any other
       }
   }
   N = r->N;
// now remove all particles outside our football
   int imax = N;
   for(int i = i0;i<imax;i++){
      double x =  r->particles[i].x;
      double y =  r->particles[i].y;
      double z =  r->particles[i].z;
      double xa2 = pow(x/ax,2.0);
      double ya2 = pow(y/by,2.0);
      double za2 = pow(z/cz,2.0);
      double rval  =  sqrt(xa2 + ya2 + za2);
      if (rval > 1.0){ // outside ellipsoid
          reb_remove(r,i,0);  // remove particle 
          i--; // we copy in a particle and we need to look at it
          imax = r->N;
      }
   }
   N = r->N;

// adjust mass of each particle so that sums to desired total mass
   double particle_mass = total_mass/(N-i0);	
// fix masses 
   for(int ii=i0;ii< N;ii++) { // all particles!
      r->particles[ii].m = particle_mass;
   }
   double md = mindist(r,i0, N);
   printf("rand_football_from_sphere: Nparticles=%d min_d=%.2f\n",N -i0,md);	
}


// create a ~uniform random particle distribution with total mass  = total_mass
// fill particles within a football shape given by semi- axes ax, by, cz
// spacing set by parameter dist: no closer particles allowed
void rand_football(struct reb_simulation* const r, double dist, 
       double ax, double by, double cz, double total_mass)
{
   struct reb_particle pt;
   int npart = 40*pow(2.0*ax/dist,3.0); 
       // guess for number of random particles we need to generate
   pt.ax = 0.0; pt.ay = 0.0; pt.az = 0.0;
   pt.vx = 0.0; pt.vy = 0.0; pt.vz = 0.0;
   pt.m = 1.0;
   double particle_radius = dist/2.0;
   pt.r = particle_radius/2.0;  // XXXXxxxxx temp
   int N = r->N;
   int i0 = N;
   for(int i=0;i<npart;i++){
       pt.x = reb_random_uniform(-ax,ax);
       pt.y = reb_random_uniform(-by,by);
       pt.z = reb_random_uniform(-cz,cz);
       double xa2 = pow(pt.x/ax,2.0);
       double yb2 = pow(pt.y/by,2.0);
       double zc2 = pow(pt.z/cz,2.0);
       double rval  =  sqrt(xa2 + yb2 + zc2);
       if (rval < 1.0){ // within football
	  int toonear = 0;  // is there a particle too nearby?
          int j=i0;  
          N = r->N;
          while ((toonear==0) && (j < N)){
             double dx =  pt.x - r->particles[j].x;
             double dy =  pt.y - r->particles[j].y;
             double dz =  pt.z - r->particles[j].z;
             double dr = sqrt(dx*dx + dy*dy + dz*dz);
	     if (dr < dist) toonear=1;
	     j++;
          }
	  if (toonear==0) reb_add(r,pt); 
          // only add particle if not near any other
       }
   }
   N = r->N;

// adjust mass of each particle so that sums to desired total mass
   double particle_mass = total_mass/(N-i0);	
// fix masses 
   for(int ii=i0;ii< N;ii++) { // all particles!
      r->particles[ii].m = particle_mass;
   }
   double md = mindist(r,i0, N);
   printf("rand_football: Nparticles=%d min_d=%.2f\n",N -i0,md);	
}


// return mean spring constant of all springs
double mean_ks(struct reb_simulation* const r,int type)
{
   double sum=0.0;
   int nn=0;
   for(int i=0;i<NS;i++){
         sum+= springs[i].ks;
         nn++;
   }
   return sum/nn;
}


// compute distance of particle from coordinates (xc,yc,zc)
double rad_com(struct reb_simulation* const r, int ii, double xc, double yc, double zc){
   double dx = r->particles[ii].x - xc;
   double dy = r->particles[ii].y - yc;
   double dz = r->particles[ii].z - zc;
   double rad = sqrt(dx*dx + dy*dy + dz*dz);
   return rad;
}


// compute Young's modulus of springs 
// using midpoints in radial range 
//    from center of mass [rmin,rmax]
// using equation 20 by Kot et al. 2014  sum_i k_iL_i^2/(6V)
// uses rest lengths
// only computes center of mass using particles index range [il,ih)
double Young_mush(struct reb_simulation* const r, int il, int ih, double rmin, double rmax){
  double sum=0.0;
  double xc,yc,zc;
  compute_com(r,il, ih, &xc, &yc, &zc); // center of mass coords for particles in range 
  double rmid,thetamid,phimid;
  for (int i=0;i<NS;i++){
       spr_ang_mid(r, springs[i],xc,yc,zc, &rmid, &thetamid, &phimid);

       double rc = rmid; // center of spring
       if ((rc<rmax) && (rc > rmin)){
         double ks = springs[i].ks;
         double Li = springs[i].rs0;
         sum += ks*Li*Li;
       }
       // printf("rc %.2e sum %.2e\n",rc,sum);
  }
  double volume = (4.0*M_PI/3.0)*(pow(rmax,3.0) - pow(rmin,3.0)); // in shell
  double E = sum/(6.0*volume); // equation 20 by Kot et al. 2014
  return E; // return Young's modulus
}

// alternate routine using every spring
double Young_mush_big(struct reb_simulation* const r, int il, int ih){
  double sum = 0.0;
  for (int i=0;i<NS;i++){
     double ks = springs[i].ks;
     double Li = springs[i].rs0;
     sum += ks*Li*Li;
  }
  double volume = 4.0*M_PI/3.0; // sphere of radius 1 assumed?
  double E = sum/(6.0*volume); // equation 20 by Kot et al. 2014
  return E; // return Young's modulus
}


// compute mean rest length of springs 
double mean_L(struct reb_simulation* const r){
  double sum=0.0;
  for(int i=0;i<NS;i++){
      sum += springs[i].rs0;
  }
  return sum/NS;
}


// reset the spring damping coefficient for all springs
void set_gamma(struct reb_simulation* const r,double new_gamma){
   for(int i=0;i<NS;i++){
      springs[i].gamma = new_gamma;
   }
   printf("\n gamma set to %.2f\n",new_gamma);
}

// adjust dampings of all springs by factor gamma_fac 
void set_gamma_fac(struct reb_simulation* const r,double gamma_fac){
   for(int i=0;i<NS;i++){
      springs[i].gamma /= gamma_fac;
   }
   printf("\n gamma divided by %.2f\n",gamma_fac);
}


// start the body spinning particles indexes in [il,ih-1]
// with spin value omegax,omegay,omegaz,  about center of mass
// center of mass assumed to have zero velocity (velocities set, not added to)
// spin vector is omegax,omegay,omegaz
void spin(struct reb_simulation* const r,int il, int ih, 
      double omegax, double omegay, double omegaz){
   double xc =0.0; double yc =0.0; double zc =0.0;
   compute_com(r,il, ih, &xc, &yc, &zc); // compute center of mass
   double omega = sqrt(omegax*omegax + omegay*omegay + omegaz*omegaz); 
   // size limit
   if (omega > 1e-5){
     for(int i=il;i<ih;i++){ 
       double dx =  r->particles[i].x - xc;
       double dy =  r->particles[i].y - yc;
       double dz =  r->particles[i].z - zc;
       double rcrosso_x = -dy*omegaz + dz*omegay;   // r cross omega
       double rcrosso_y = -dz*omegax + dx*omegaz;
       double rcrosso_z = -dx*omegay + dy*omegax;
       r->particles[i].vx = rcrosso_x; // set it spinning with respect to center of mass
       r->particles[i].vy = rcrosso_y; // note center of mass assumed to have zero velocity!
       r->particles[i].vz = rcrosso_z;
     }
   }

}



// make a binary with two masses m1,m2 spinning with vector omega
// masses are separated by distance sep
// connect two masses with spring with values given by spring_vals
// center of mass is set to origin
void make_binary_spring(struct reb_simulation* const r,double m1, double m2, double sep, 
      double omegax, double omegay, double omegaz,
      struct spring spring_vals)
{
   const int il= r->N; 
   struct reb_particle pt;
   pt.ax = 0.0; pt.ay = 0.0; pt.az = 0.0;
   pt.vx = 0.0; pt.vy = 0.0; pt.vz = 0.0;
   pt.y = 0.0; pt.z = 0.0;
   pt.m = m1; pt.x = sep*m2/(m1+m2); 
   pt.r = sep*0.3;
   reb_add(r,pt); 
   pt.m = m2; pt.x =-sep*m1/(m1+m2); 
   pt.r *= pow(m1/m2,0.33333);
   reb_add(r,pt); 
   int ih = il+2;
   spin(r,il, ih, omegax, omegay, omegaz); // spin it
   connect_springs_dist(r,sep*1.1, il, ih, spring_vals); // add spring

}

// return angle between body i and body j in xy plane
double get_angle(struct reb_simulation* const r,int i, int j)
{
   double dx = r->particles[i].x - r->particles[j].x; 
   double dy = r->particles[i].y - r->particles[j].y; 
   double theta = atan2(dy,dx);
   return theta;
}

// compute angular momentum vector of a body with respect to its center of mass position
// and velocity for particles in range [il,ih)
// can measure angular momentum of the entire system if il=0 and ih=N
void  measure_L(struct reb_simulation* const r,int il, int ih, double *llx, double *lly, double *llz){
   struct reb_particle* particles = r->particles;
   double xc =0.0; double yc =0.0; double zc =0.0;
   compute_com(r,il, ih, &xc, &yc, &zc);
   double vxc =0.0; double vyc =0.0; double vzc =0.0;
   compute_cov(r,il, ih, &vxc, &vyc, &vzc);

   double lx = 0.0; double ly = 0.0; double lz = 0.0;
   for(int i=il;i<ih;i++){ 
       double dx =  (particles[i].x - xc);
       double dy =  (particles[i].y - yc);
       double dz =  (particles[i].z - zc);
       double dvx =  (particles[i].vx - vxc);
       double dvy =  (particles[i].vy - vyc);
       double dvz =  (particles[i].vz - vzc);
       lx += particles[i].m*(dy*dvz - dz*dvy); // angular momentum vector
       ly += particles[i].m*(dz*dvx - dx*dvz);
       lz += particles[i].m*(dx*dvy - dy*dvx);
   }
   *llx = lx;
   *lly = ly;
   *llz = lz;

}

// compute the moment of inertia tensor of a body with particle indices [il,ih)
// with respect to center of mass
void mom_inertia(struct reb_simulation* const r,int il, int ih, 
 double *Ixx, double *Iyy, double *Izz, 
 double *Ixy, double *Iyz, double *Ixz) 
{
   double xc =0.0; double yc =0.0; double zc =0.0;
   compute_com(r,il, ih, &xc, &yc, &zc);

   double Axx=0.0; double Ayy=0.0; double Azz=0.0;
   double Axy=0.0; double Ayz=0.0; double Axz=0.0;
   for(int i=il;i<ih;i++){ 
     double dx = r->particles[i].x - xc;
     double dy = r->particles[i].y - yc;
     double dz = r->particles[i].z - zc;
     Axx += r->particles[i].m*(dy*dy + dz*dz);
     Ayy += r->particles[i].m*(dx*dx + dz*dz);
     Azz += r->particles[i].m*(dx*dx + dy*dy);
     Axy -= r->particles[i].m*(dx*dy);
     Ayz -= r->particles[i].m*(dy*dz);
     Axz -= r->particles[i].m*(dx*dz);
   }
   *Ixx = Axx; *Iyy = Ayy; *Izz = Azz;
   *Ixy = Axy; *Iyz = Ayz; *Ixz = Axz;

}

// compute orbital properties of body
// resolved body indexes [il,ih)
// primary mass is at index im1  
// and compute: 
//   mean-motion:nn, semi-major axis:aa 
//   eccentricity:ee and inclination:ii
//   LL orbital angular momentum per unit mass
void compute_semi(struct reb_simulation* const r, int il,int ih, int im1,
   double *aa, double *meanmo, 
   double *ee, double *ii, double *LL){
   struct reb_particle* particles = r->particles;
   static int first =0;
   static double tm = 0.0;  // its mass
   if (first==0){ // only calculate once
     for (int i = il;i<ih;i++) tm += r->particles[i].m; // mass of resolved body
     first=1;
   }
   double xc =0.0; double yc =0.0; double zc =0.0;
   compute_com(r,il, ih, &xc, &yc, &zc); // center of mass of resolved body
   double vxc =0.0; double vyc =0.0; double vzc =0.0;
   compute_cov(r,il, ih, &vxc, &vyc, &vzc); // center of velocity of resolved body

   // int im1 = r->N -1; // index for primary perturber
   double x0 = particles[im1].x; double vx0 = particles[im1].vx;
   double y0 = particles[im1].y; double vy0 = particles[im1].vy;
   double z0 = particles[im1].z; double vz0 = particles[im1].vz;
   double dv2 =  pow(vx0 - vxc, 2.0) + // square of velocity difference 
                 pow(vy0 - vyc, 2.0) +
                 pow(vz0 - vzc, 2.0);
   double m1 = particles[im1].m;
   double MM = m1+ tm; // total mass
   // double mu = tm*m1/MM;  // reduced mass
   double GMM = r->G*MM;
   double ke = 0.5*dv2; // kinetic energy /mu  (per unit mass)
   double dr = sqrt(pow(x0 - xc,2.0) +  // distance between
                    pow(y0 - yc,2.0) +
                    pow(z0 - zc,2.0));
   double pe = -GMM/dr; // potential energy/mu, interaction term  tm*m1 = GM*mu
   double E = ke + pe;  // total energy per unit mass
   double a = -0.5*GMM/E; // semi-major axis
   *aa = a;
   *meanmo = sqrt(GMM/(a*a*a)); // mean motion
   // printf("dr=%.2f dv2=%.2f\n",dr,dv2);
   // compute orbital angular momentum
   double dx  = x0 - xc;   double dy  = y0 - yc;   double dz  = z0 - zc;
   double dvx = vx0 - vxc; double dvy = vy0 - vyc; double dvz = vz0 - vzc;
   double lx = dy*dvz - dz*dvy;
   double ly = dz*dvx - dx*dvz;
   double lz = dx*dvy - dy*dvx;
   double ltot = sqrt(lx*lx + ly*ly + lz*lz);
   *LL = ltot; // orbital angular momentum per unit mass
   double e2 = 1.0 -ltot*ltot/(a*GMM);
   *ee = 0.0;
   if (e2 > 0.0) *ee = sqrt(e2); // eccentricity
   *ii =  acos(lz/ltot); // inclination ? if lz==ltot then is zero
   
}

// compute the orbital angular momentum vector
// resolved body indexes [il,ih)
// primary mass is at index [N-1] but includes npert if more than one perturbers
// returns orbital angular momentum vector
void compute_Lorb(struct reb_simulation* const r, int il,int ih, int npert,
 double *llx, double *lly, double *llz)
{
   struct reb_particle* particles = r->particles;
   static int first =0;
   static double tm = 0.0;  // its mass
   if (first==0){ // only calculate once
     for (int i = il;i<ih;i++) tm += r->particles[i].m; // mass of resolved body
     first=1;
   }
   double xc =0.0; double yc =0.0; double zc =0.0;
   compute_com(r,il, ih, &xc, &yc, &zc); // center of mass of resolved body
   double vxc =0.0; double vyc =0.0; double vzc =0.0;
   compute_cov(r,il, ih, &vxc, &vyc, &vzc); // center of velocity of resolved body

   double x0=0.0; double y0=0.0; double z0=0.0;
   double vx0=0.0; double vy0=0.0; double vz0=0.0;
   if (npert==1){
      int im1 = r->N -1; // index for primary perturber
      // double m1 = particles[im1].m;
      // double MM = m1+ tm; // total mass
      // double GMM = r->G*MM;
      x0 = particles[im1].x; vx0 = particles[im1].vx;
      y0 = particles[im1].y; vy0 = particles[im1].vy;
      z0 = particles[im1].z; vz0 = particles[im1].vz;
   }
   else {
      int iml = r->N -npert; // index range for perturbing masses 
      int imh = r->N; 
      compute_com(r,iml, imh, &x0, &y0, &z0); // center of mass of perturbing bodies
      compute_cov(r,iml, imh, &vx0, &vy0, &vz0); // center of velocity of perturbing body
      // for(int i=iml;i<imh;i++) m1 += particles[i].m; // total perturbing mass
   }

   double dx  = x0 - xc;   double dy  = y0 - yc;   double dz  = z0 - zc;
   double dvx = vx0 - vxc; double dvy = vy0 - vyc; double dvz = vz0 - vzc;
   double lx = dy*dvz - dz*dvy;
   double ly = dz*dvx - dx*dvz;
   double lz = dx*dvy - dy*dvx;
   *llx = lx;
   *lly = ly;
   *llz = lz;
}

// compute orbital properties of body, with respect
// to center of mass of a binary with two masses at index [N-1] and N-2 (or up to npert)
// resolved body indexes [il,ih)
// and compute: 
//   mean-motion:nn, semi-major axis:aa 
//   eccentricity:ee and inclination:ii
//   LL orbital angular momentum per unit mass
void compute_semi_bin(struct reb_simulation* const r, int il, int ih, 
       int npert, double *aa, double *meanmo, 
   double *ee, double *ii, double *LL){
   struct reb_particle* particles = r->particles;
   static int first =0;
   static double tm = 0.0;  // its mass
   if (first==0){ // only calculate once
     for (int i = il;i<ih;i++) tm += r->particles[i].m; // mass of resolved body
     first=1;
   }
   double xc =0.0; double yc =0.0; double zc =0.0;
   compute_com(r,il, ih, &xc, &yc, &zc); // center of mass of resolved body
   double vxc =0.0; double vyc =0.0; double vzc =0.0;
   compute_cov(r,il, ih, &vxc, &vyc, &vzc); // center of velocity of resolved body

   double xc0 =0.0; double yc0 =0.0; double zc0 =0.0;
   double vxc0 =0.0; double vyc0 =0.0; double vzc0 =0.0;
   double m1 = 0.0;
   if (npert ==1)   {
      int im1 = r->N -1; // index for primary perturber
      xc0 = particles[im1].x; vxc0 = particles[im1].vx;
      yc0 = particles[im1].y; vyc0 = particles[im1].vy;
      zc0 = particles[im1].z; vzc0 = particles[im1].vz;
      m1 = particles[im1].m;
   }
   else {
      int iml = r->N -npert; // index range for perturbing masses 
      int imh = r->N; 
      compute_com(r,iml, imh, &xc0, &yc0, &zc0); // center of mass of perturbing bodies
      compute_cov(r,iml, imh, &vxc0, &vyc0, &vzc0); // center of velocity of perturbing body
      for(int i=iml;i<imh;i++) m1 += particles[i].m; // total perturbing mass
   }
   
   double dv2 =  pow(vxc0 - vxc, 2.0) + // square of velocity difference 
                 pow(vyc0 - vyc, 2.0) +
                 pow(vzc0 - vzc, 2.0);
   double MM = m1+ tm; // total mass
   // double mu = tm*m1/MM;  // reduced mass
   double GMM = r->G*MM;
   double ke = 0.5*dv2; // kinetic energy /mu  (per unit mass)
   double dr = sqrt(pow(xc0 - xc,2.0) +  // distance between
                    pow(yc0 - yc,2.0) +
                    pow(zc0 - zc,2.0));
   double pe = -GMM/dr; // potential energy/mu, interaction term  tm*m1 = GM*mu
   double E = ke + pe;  // total energy
   double a = -0.5*GMM/E; // semi-major axis
   *aa = a;
   *meanmo = sqrt(GMM/(a*a*a)); // mean motion
   // printf("dr=%.2f dv2=%.2f\n",dr,dv2);
   // compute orbital angular momentum
   double dx = xc0 - xc;    double dy  = yc0 - yc;   double dz  = zc0 - zc;
   double dvx = vxc0 - vxc; double dvy = vyc0 - vyc; double dvz = vzc0 - vzc;
   double lx = dy*dvz - dz*dvy;
   double ly = dz*dvx - dx*dvz;
   double lz = dx*dvy - dy*dvx;
   double ltot = sqrt(lx*lx + ly*ly + lz*lz);
   *LL = ltot; // orbital angular momentum per unit mass
   double e2 = fabs(1.0 -ltot*ltot/(a*GMM));
   *ee = sqrt(e2+1e-16); // eccentricity
   *ii =  acos(lz/ltot); // inclination ? if lz==ltot then is zero
   
}


// compute obital angular momentum with respect to center of mass of whole system 
// single massive object is particle at i=N-1
// rest of particles assumed to be in a resolved body
void compute_lorb(struct reb_simulation* const r, double *lx, double *ly, double *lz)
{
   struct reb_particle* particles = r->particles;
   const int il=0; 
   const int ih= r->N-1; // index range for resolved body

   static int first =0;
   static double tm = 0.0;  // its mass
   if (first==0){ // only calculate once
     for (int i = il;i<ih;i++) tm += r->particles[i].m; // mass of resolved body
     first=1;
   }

   double xc =0.0; double yc =0.0; double zc =0.0;
   compute_com(r,il, ih, &xc, &yc, &zc); // center of mass of resolved body
   double vxc =0.0; double vyc =0.0; double vzc =0.0;
   compute_cov(r,il, ih, &vxc, &vyc, &vzc); // center of velocity of resolved body
   const int im1 = r->N -1; // index for primary 
   double m1 = particles[im1].m;
   double x1  = particles[im1].x;   double y1 = particles[im1].y;  double z1  = particles[im1].z;
   double vx1 = particles[im1].vx; double vy1 = particles[im1].vy; double vz1 = particles[im1].vz;
   double M=m1+tm;
   double xt = (tm*xc + m1*x1)/M; // center of mass of whole system
   double yt = (tm*yc + m1*y1)/M;
   double zt = (tm*zc + m1*z1)/M;
   double vxt = (tm*vxc + m1*vx1)/M; // center of vel of whole system
   double vyt = (tm*vyc + m1*vy1)/M;
   double vzt = (tm*vzc + m1*vz1)/M;

   double dx  = xc - xt;   double dy  = yc - yt;  double dz  = zc - zt;
   double dvx = vxc - vxt; double dvy = vyc - yt; double dvz = vzc - zt;
   double dx1  = x1 - xt;   double dy1  = y1 - yt;  double dz1  = z1 - zt;
   double dvx1 = vx1 - vxt; double dvy1 = vy1 - vyt; double dvz1 = vz1 - vzt;
   *lx = m1*(dy1*dvz1 - dz1*dvy1) + tm*(dy*dvz - dz*dvy);
   *ly = m1*(dz1*dvx1 - dx1*dvz1) + tm*(dz*dvx - dx*dvz); 
   *lz = m1*(dx1*dvy1 - dy1*dvx1) + tm*(dx*dvy - dy*dvx);
}

// sum total momentum of particles
void total_mom(struct reb_simulation* const r,int il, int ih, double *ppx, double *ppy, double *ppz){
   double px=0.0; double py=0.0; double pz=0.0;
   for(int i=il;i<ih;i++){ 
     px += r->particles[i].m * r->particles[i].vx;
     py += r->particles[i].m * r->particles[i].vy;
     pz += r->particles[i].m * r->particles[i].vz;
   }
   *ppx = px;
   *ppy = py;
   *ppz = pz;
   
}


// using Euler angles rotate a body with particle indices [il,ih)
// about center of mass
// rotate both position and velocities
void rotate_body(struct reb_simulation* const r, int il, int ih, 
     double alpha, double beta, double gamma)
{
   struct reb_particle* particles = r->particles;
   double xc,yc,zc;
   compute_com(r,il, ih, &xc, &yc, &zc);
   double vxc,vyc,vzc;
   compute_cov(r,il, ih, &vxc, &vyc, &vzc);
   for(int i=il;i<ih;i++){
     double x0 = particles[i].x - xc;
     double y0 = particles[i].y - yc;
     double z0 = particles[i].z - zc;
     double x1 = x0*cos(alpha) - y0*sin(alpha);  // rotate about z axis in xy plane
     double y1 = x0*sin(alpha) + y0*cos(alpha);
     double z1 = z0;
     double x2 = x1;                             // rotate about x' axis in yz plane
     double y2 = y1*cos(beta)  - z1*sin(beta);
     double z2 = y1*sin(beta)  + z1*cos(beta);
     double x3 = x2*cos(gamma) - y2*sin(gamma);  // rotate about z'' axis in xy plane
     double y3 = x2*sin(gamma) + y2*cos(gamma);
     double z3 = z2;
     particles[i].x = x3 + xc;
     particles[i].y = y3 + yc;
     particles[i].z = z3 + zc;

     double vx0 = particles[i].vx - vxc;
     double vy0 = particles[i].vy - vyc;
     double vz0 = particles[i].vz - vzc;
     double vx1 = vx0*cos(alpha) - vy0*sin(alpha);  // rotate about z axis in xy plane
     double vy1 = vx0*sin(alpha) + vy0*cos(alpha);
     double vz1 = vz0;
     double vx2 = vx1;                              // rotate about x' axis in yz plane
     double vy2 = vy1*cos(beta)  - vz1*sin(beta);
     double vz2 = vy1*sin(beta)  + vz1*cos(beta);
     double vx3 = vx2*cos(gamma) - vy2*sin(gamma);  // rotate about z'' axis in xy plane
     double vy3 = vx2*sin(gamma) + vy2*cos(gamma);
     double vz3 = vz2;
     particles[i].vx = vx3 + vxc;
     particles[i].vy = vy3 + vyc;
     particles[i].vz = vz3 + vzc;
   } 
}


// compute determinent of 3x3 symmetric matrix
double detI(double Ixx,double Iyy,double Izz,double Ixy,double Iyz,double Ixz){
     double r = Ixx*(Iyy*Izz -Iyz*Iyz) + Ixy*(Iyz*Ixz - Ixy*Izz) 
               + Ixz*(Ixy*Iyz - Iyy*Ixz);
     return r;
}

// compute inverse of 3x3 symmetric matrix
void invI(double Ixx,double Iyy,double Izz,double Ixy,double Iyz,double Ixz,
            double *Axx,double *Ayy,double *Azz,double *Axy,double *Ayz,double *Axz){
   double invd = 1.0/detI(Ixx,Iyy,Izz,Ixy,Iyz,Ixz);
   *Axx = invd*(Iyy*Izz-Iyz*Iyz);
   *Ayy = invd*(Ixx*Izz-Ixz*Ixz);
   *Azz = invd*(Ixx*Iyy-Ixy*Ixy);
   *Axy = invd*(Ixz*Iyz-Ixy*Izz);
   *Axz = invd*(Ixy*Iyz-Ixz*Iyy);
   *Ayz = invd*(Ixz*Ixy-Iyz*Ixx);
}

// return eigenvalues of symmetric matrix
// https://en.wikipedia.org/wiki/Eigenvalue_algorithm
// order returned: eig1 is largest eig>=eig2>=eig3
void eigenvalues(double Ixx, double Iyy,  double Izz, double Ixy, double Iyz, double Ixz,
   double *eig1, double *eig2, double *eig3){
// recipe from the wiki for eigenvalues of a symmetric matrix
   double e1,e2,e3;
   double p1 = Ixy*Ixy + Iyz*Iyz + Ixz*Ixz;
   if (p1 == 0) {
      // I is diagonal.
      e1 = Ixx; e2 = Iyy; e3 = Izz; // make sure in order
      if ((Ixx >= Iyy) && (Iyy >= Izz)) e1 = Ixx; e2 = Iyy; e3 = Izz;
      if ((Ixx >= Izz) && (Izz >= Iyy)) e1 = Ixx; e2 = Izz; e3 = Iyy;
      if ((Iyy >= Ixx) && (Ixx >= Izz)) e1 = Iyy; e2 = Ixx; e3 = Izz;
      if ((Iyy >= Izz) && (Izz >= Ixx)) e1 = Iyy; e2 = Izz; e3 = Ixx;
      if ((Izz >= Ixx) && (Ixx >= Iyy)) e1 = Izz; e2 = Ixx; e3 = Iyy;
      if ((Izz >= Iyy) && (Iyy >= Ixx)) e1 = Izz; e2 = Iyy; e3 = Ixx;
     
   }
   else{
     double q = (Ixx + Iyy + Izz)/3.0; // trace divided by 3
     double p2 = pow(Ixx - q,2.0) + pow(Iyy- q,2.0) + pow(Izz - q,2.0) + 2.0 * p1;
     double p = sqrt(p2 / 6.0);
     double Bxx, Byy, Bzz, Bxz, Byz, Bxy;
     Bxx = (1.0/p)*(Ixx - q);
     Byy = (1.0/p)*(Iyy - q);
     Bzz = (1.0/p)*(Izz - q);
     Bxz = (1.0/p)*Ixz; Byz = (1.0/p)*Iyz; Bxy = (1.0/p)*Ixy;
     // double B = (1.0/ p) * (A - q * I)       % I is the identity matrix
     double r = 0.5*detI(Bxx,Byy,Bzz,Bxy,Byz,Bxz);
     // double r = det(B) / 2

   // Is exact arithmetic for a symmetric matrix  -1 <= r <= 1
   // but computation error can leave it slightly outside this range.
     double phi = 0.0;
     if (r <= -1.0) phi = M_PI/3.0;
     if (r >= 1.0) phi = 0;
     if ((r <1.0) &&(r>-1.0)) phi = acos(r)/3.0;

   // the eigenvalues satisfy eig3 <= eig2 <= eig1
     e1 = q + 2.0*p*cos(phi);
     e3 = q + 2.0*p*cos(phi + (2.0*M_PI/3.0));
     e2 = 3.0*q - e1 - e3;  //  since trace(A) = eig1 + eig2 + eig3
   }
   *eig1 = e1; // largest to smallest
   *eig2 = e2;
   *eig3 = e3;
}

// compute spin vector of a body with indices [il,ih)
//  spin vector  is omx, omy, omz 
//    computed using inverse of moment of inertia matrix
// also return eigenvalues of moment of inertia  matrix
// order big>=middle>=small (
void body_spin(struct reb_simulation* const r, int il, int ih, 
   double *omx, double *omy, double *omz, double *big, double *middle, double *small)
{
   // struct reb_particle* particles = r->particles;

   //compute moment of inertia matrix
   double Ixx,Iyy,Izz,Ixy,Iyz,Ixz;
   mom_inertia(r,il,ih, &Ixx, &Iyy, &Izz,&Ixy, &Iyz, &Ixz);

   // compute inverse of moment of inertia matrix
   double Axx,Ayy,Azz,Axy,Ayz,Axz;
   invI(Ixx,Iyy,Izz,Ixy,Iyz,Ixz,&Axx,&Ayy,&Azz,&Axy,&Ayz,&Axz);

   // compute angular momentum vector body with respect to its center 
   // of mass position and velocity 
   double llx,lly,llz;
   measure_L(r,il, ih, &llx, &lly, &llz);

   // compute spin vector
   double ox = Axx*llx + Axy*lly + Axz*llz;
   double oy = Axy*llx + Ayy*lly + Ayz*llz;
   double oz = Axz*llx + Ayz*lly + Azz*llz;
   *omx = ox; *omy = oy; *omz = oz;
   double eig1,eig2,eig3;
   eigenvalues(Ixx,Iyy,Izz,Ixy,Iyz,Ixz, &eig1, &eig2, &eig3);
   *big    = eig1;  
   *middle = eig2;
   *small  = eig3;

}


// adjust ks and gamma 
// for springs with midpoints between rmin and rmax of center of mass
void adjust_ks(struct reb_simulation* const r, int npert, 
    double ksnew, double gammanew, double rmin, double rmax)
{
   // struct reb_particle* particles = r->particles;
   int il =0;
   int ih =r->N - npert; 
   double xc,yc,zc;
   compute_com(r,il, ih, &xc, &yc, &zc);
   double xmid,ymid,zmid;
   for(int i=0;i<NS;i++){
      spr_xyz_mid(r, springs[i], xc, yc, zc, &xmid, &ymid, &zmid);
      double rmid = sqrt(xmid*xmid + ymid*ymid + zmid*zmid); 
      // double thetamid = asin(zmid/rmid); // latitude angle
      // if you want to make an oval inside region you can use this angle
      if ((rmid >= rmin) && (rmid <= rmax)){
           springs[i].ks=ksnew;
           springs[i].gamma=gammanew;
      }
   }
   
}

// adjust ks and gamma 
// for springs with midpoints within ellipsoid set by x^2/a^2 + y^2/b^2 + z^2/c^2 = 1
void adjust_ks_abc(struct reb_simulation* const r, int npert, 
    double ksnew, double gammanew, double a, double b, double c)
{
   // struct reb_particle* particles = r->particles;
   int il =0;
   int ih =r->N - npert; 
   double xc,yc,zc;
   compute_com(r,il, ih, &xc, &yc, &zc);
   double xmid,ymid,zmid;
   for(int i=0;i<NS;i++){
      spr_xyz_mid(r, springs[i], xc, yc, zc, &xmid, &ymid, &zmid);
      double rmid2 = xmid*xmid/(a*a) + ymid*ymid/(b*b) + zmid*zmid/(c*c); 
      if (rmid2 <= 1.0){
           springs[i].ks=ksnew;
           springs[i].gamma=gammanew;
      }
   }
}


// change all masses with x>xmin by factor mfac, then rescale so sum is still 1
// done with respect to origin
void adjust_mass_side(struct reb_simulation* const r, int npert, double mfac, double xmin)
{
   struct reb_particle* particles = r->particles;
   int il =0;
   int ih =r->N- npert;  
   double tm=0.0;
   int npart=0;
   for(int i=il;i<ih;i++){
      if (particles[i].x > xmin) { 
         particles[i].m*=mfac;
         npart++;
      }
   }
   for(int i=il;i<ih;i++)
      tm += particles[i].m;
   for(int i=il;i<ih;i++)
     particles[i].m /=tm;
   printf("npart=%d \n",npart);
   
}

// returns 0 if even 1 if odd
int mym(int k){
   int z = floor(k/2);
   return  abs(k - 2*z); // returns 0 or 1
}


// make a football of particles hcp lattice
// and total mass = total_mass
// hexagonal close packed setup
// dd is minimum distance between particles
// ax,by,cz are semi-axes
double fill_hcp(struct reb_simulation* r, double dd, 
       double ax, double by, double cz, double total_mass)
{
   // struct reb_particle* particles = r->particles;
   struct reb_particle pt;
   int i0 = r->N; // store initial particle number
   double dx = 1.08*dd;
   int nx = (int)(1.2*ax/dx);
   // double dx = 2.0*ax/nx;
   double zfac = dx*sqrt(2.0/3.0); // factors for hcp lattice
   double yfac = dx*sin(M_PI/3.0); 
   double xfac_s = dx/2.0; // shifting
   // printf("dx =%.3f zfac=%.3f yfac=%.3f\n",dx,zfac,yfac);
// fill a cube
   double yy = 1.2*by/yfac; int ny = (int)yy;
   double zz = 1.2*cz/zfac; int nz = (int)zz;
   double midvalx = 0.0*nx*dx; // center of ball
   double midvaly = 0.0*ny*yfac;
   double midvalz = 0.0*nz*zfac;
   double particle_radius = dx/2.0; // temporary set
// make an hcp grid
   for(int k=-nz;k<=nz;k++){
     double z = zfac*k;
     for(int j=-ny;j<=ny;j++){
       double y = yfac*(j+0.5*mym(k));
       for(int i=-nx;i<=nx;i++){
          double x = dx*i +  xfac_s*mym(j)+ xfac_s*mym(k);
              // printf("%.2f %.2f %.2f\n",x-midval,y-midval,z-midval);
		pt.m 		= 1.0;
		pt.x 		= x - midvalx;
		pt.y 		= y - midvaly;
		pt.z 		= z - midvalz;
		pt.vx 	= 0; pt.ax = 0;
		pt.vy 	= 0; pt.ay = 0;
		pt.vz 	= 0; pt.az = 0;
		pt.r 		= particle_radius;
                double xa2 = pow(pt.x/ax,2.0);
                double yb2 = pow(pt.y/by,2.0);
                double zc2 = pow(pt.z/cz,2.0);
                double rval  =  sqrt(xa2 + yb2 + zc2);
		if (rval <= 1.0) {
                     reb_add(r,pt);
                }
       }
     }
   }
   int N= r->N;
   // printf("i0=%d N=%d\n",i0,N);
   double particle_mass = total_mass/(N-i0);
   double min_d = mindist(r,i0,N);
   // correct the radii and mass of the particles

   for(int i=i0;i<N;i++){
	r->particles[i].m = particle_mass;
        r->particles[i].r = min_d/4.0;
   }

   printf("fill_hcp: Nparticles =%d dx=%.2e dr=%.2e rad=%.2e m=%.2e\n"
        ,N-i0,dx,min_d,min_d/2.0, particle_mass);	
  return min_d;
}

// make a football of particles cubic lattice
// and total mass = total_mass
// dd is minimum distance between particles (cube side)
// ax,by,cz are semi-axes
double fill_cubic(struct reb_simulation* r, double dd, 
       double ax, double by, double cz, double total_mass)
{
   struct reb_particle pt;
   int i0 = r->N; // store initial particle number
   int nz = (int)(1.2*ax/dd);
   int ny = nz; int nx = nz;
   double zfac = dd;
   for(int k=-nz;k<=nz;k++){
     double z = zfac*k;
     for(int j=-ny;j<=ny;j++){
       double y = zfac*j;
       for(int i=-nx;i<=nx;i++){
          double x = zfac*i;
		pt.m = 1.0;
		pt.x = x;
		pt.y = y;
		pt.z = z;
		pt.vx 	= 0; pt.ax = 0;
		pt.vy 	= 0; pt.ay = 0;
		pt.vz 	= 0; pt.az = 0;
		pt.r = 1.0;
                double xa2 = pow(pt.x/ax,2.0);
                double yb2 = pow(pt.y/by,2.0);
                double zc2 = pow(pt.z/cz,2.0);
                double rval  =  sqrt(xa2 + yb2 + zc2);
		if (rval <= 1.0) {
                     reb_add(r,pt);
                }
       }
     }
   }
   int N= r->N;
   double particle_mass = total_mass/(N-i0);
   double min_d = mindist(r,i0,N);
   // correct the radii and mass of the particles
   for(int i=i0;i<N;i++){
	r->particles[i].m = particle_mass;
        r->particles[i].r = min_d/4.0;
   }
   printf("fill_cubic: Nparticles =%d dx=%.2e rad=%.2e m=%.2e\n"
        ,N-i0,dd,min_d/2.0, particle_mass);	
   return min_d;
}


// shift the position of a resolved body by dx,dy,dz,dvx,dvy,dvz
// shift all particles positions and velocities in the body [il,ih)
void move_resolved(struct reb_simulation* r, 
                    double dx,double dy,double dz,
                    double dvx,double dvy,double dvz,
                    int il,int ih)
{
   // struct reb_particle* particles = r->particles;
   // struct reb_particle pt;
   for(int i=il;i<ih;i++){
        r->particles[i].x += dx;
        r->particles[i].y += dy;
        r->particles[i].z += dz;
        r->particles[i].vx += dvx;
        r->particles[i].vy += dvy;
        r->particles[i].vz += dvz;
   }
}


