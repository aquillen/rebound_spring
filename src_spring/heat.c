
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
extern int NPERT;  // number of external perturbing bodies
double *heatvec; // global so can be reached by all routines here


// print out heat file
// npert is number of point mass perturbers
// ndt is number of times heatvec is added to 
//   (need to normalize so units of power)
// powerfac is how much of heat to distribute at center of spring 
//    rather than at nodes
void print_heat(struct reb_simulation* const r, int npert, 
      char* filename, int ndt, double powerfac)
{
   struct reb_particle* particles = r->particles;

   norm_heatvec(ndt);  // normalize power so is average power

   FILE *fpo;
   fpo = fopen(filename, "w");
   fprintf(fpo,"#%.2e\n",r->t);
   int il =0;
   int ih =r->N-npert; // NPERT?
   double xc,yc,zc;
   compute_com(r,il, ih, &xc, &yc, &zc);
   int im1 = r->N -1; // index for primary perturber
   // double m1 = particles[im1].m;
   double x1 = particles[im1].x; double y1 = particles[im1].y; double z1 = particles[im1].z;
   double theta = atan2(y1-yc,x1-xc);
   double ct = cos(theta); double st = sin(-theta);  // note sign!
   fprintf(fpo,"#%.3f %.3f %.3f %.3f %.3f %.3f %.3f \n",x1,y1,z1,xc,yc,zc,theta);
// theta =0 when m1 is at +x direction compared to resolved body
   double xmid,ymid,zmid;

   for(int i=0;i<NS;i++){
      spr_xyz_mid(r, springs[i], xc, yc, zc, &xmid, &ymid, &zmid);
       // rotate around center of body in xy plane
   // after rotation:
   // +x is toward perturber
   // +y is direction of rotation of perturber w.r.t to body
   // so -y is headwind direction for body in body
   // and +y is tailwind surface on body?????
      double xrot = xmid*ct - ymid*st;  
      double yrot = xmid*st + ymid*ct;
      // double power = dEdt(r,springs[i]);
      
      double power = heatvec[i];
      // double powerfac = 0.5; // fraction of heat goes to center of spring
      // heat on center of spring
      fprintf(fpo,"%d %.3f %.3f %.3f %.5e %.3f %.3f\n"
          ,i,xmid,ymid,zmid,power*powerfac,xrot,yrot);
      // put some heat on nodes as well as center of spring
      if (powerfac < 1.0){ 
         int ii = springs[i].i; 
         int jj = springs[i].j;
         double xpi = particles[ii].x - xc;
         double ypi = particles[ii].y - yc;
         double zpi = particles[ii].z - zc;
         double xpj = particles[jj].x - xc;
         double ypj = particles[jj].y - yc;
         double zpj = particles[jj].z - zc;
         fprintf(fpo,"%d %.3f %.3f %.3f %.5e %.3f %.3f\n"
           ,i,xpi,ypi,zpi,power*(1.0-powerfac)*0.5,xrot,yrot);
         fprintf(fpo,"%d %.3f %.3f %.3f %.5e %.3f %.3f\n"
           ,i,xpj,ypj,zpj,power*(1.0-powerfac)*0.5,xrot,yrot);
      }
   }
   fclose(fpo);
   for (int i=0;i<NS;i++) heatvec[i] = 0;
}

// add to heat store vector 
// stores power for each spring
// uses a heatvec array to store this
// adds dE/dt to array -- this is power
// so power is added together
// actually later on we should normalize this
// by the number of timesteps we did it
void addto_heatvec(struct reb_simulation* const r)
{
   static int first = 0;
   if (first==0){
      first=1;
      heatvec = malloc(NS*sizeof(double));
      for(int i=0;i<NS;i++) heatvec[i]=0.0;
   }
   for(int i=0;i<NS;i++){
      double power = dEdt(r,springs[i]);
      heatvec[i] += power;
   }
}

// normalize heatvec
// ndt is the number of timesteps used to store power!
void norm_heatvec(int ndt)
{
   for(int i=0;i<NS;i++){
      heatvec[i] /= ndt;
   }
}


// make a heat file name depending on numbers of tp
void hfilename(struct reb_simulation* const r,char *root, double tp, char *fname){
   int xd = (int)(r->t/tp);
   char junks[20];
   sprintf(junks,"%d",xd);
   sprintf(fname,"%s_",root);
   if (xd < 100000) strcat(fname,"0");
   if (xd < 10000)  strcat(fname,"0");
   if (xd < 1000)   strcat(fname,"0");
   if (xd < 100)    strcat(fname,"0");
   if (xd < 10)     strcat(fname,"0");
   strcat(fname,junks);
   strcat(fname,"_heat.txt");
}

