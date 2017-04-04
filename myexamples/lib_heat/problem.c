/**
 * resolved mass spring model
 * using the leap frog integrator. 
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "rebound.h"
#include "tools.h"
#include "output.h"
#include "spring.h"


int NS; 
struct spring* springs;
void reb_springs();// to pass springs to display
#define NPMAX 10  // maximum number of point masses
double itaua[NPMAX],itaue[NPMAX];

double gamma_fac; // for adjustment of gamma of all springs
double t_damp;    // end faster damping, relaxation
double t_print;   // for table printout 
double t_heat;    // for heat  printout 
char froot[30];   // output files
int npert; // number of point mass perturbers
double powerfac; // fraction of heat to center of spring rather than nodes

int icentral=-1; // central mass location

void heartbeat(struct reb_simulation* const r);

void additional_forces(struct reb_simulation* r){
   spring_forces(r); // spring forces
}


int main(int argc, char* argv[]){
	struct reb_simulation* const r = reb_create_simulation();
        struct spring spring_mush; // spring parameters for mush
	// Setup constants
	r->integrator	= REB_INTEGRATOR_LEAPFROG;
	r->gravity	= REB_GRAVITY_BASIC;
	r->boundary	= REB_BOUNDARY_NONE;
	r->G 		= 1;		
        r->additional_forces = additional_forces;  // setup callback function for additional forces
        double mball = 1.0;          // total mass of ball
        double rball = 1.0;          // radius of a ball
        double tmax = 0.0;  // if 0 integrate forever

// things to set! ////////////////////// could be read in with parameter file
        double dt, b_distance,omegaz,ks,mush_fac,gamma_all,extra;
        double ratio1,ratio2,ks_I,r_I,gamma_I,obliq_deg;
        double eps_c=0.0;
        int lattice_type;
        double rad[NPMAX],mp[NPMAX];
        double aa[NPMAX],ee[NPMAX],ii[NPMAX];
        double longnode[NPMAX],argperi[NPMAX],meananom[NPMAX];
        int npointmass=0;
        npert=0;

    if (argc ==1){
        strcpy(froot,"t1");   // to make output files
	dt	   = 1e-3;    // Timestep
	lattice_type       =0;        // 0=rand 1=hcp
        b_distance = 0.15;    // for creating random sphere, min separation between particles
        mush_fac    = 2.3;    // ratio of smallest spring distance to minimum interparticle dist
        omegaz     = 0.2;     // initial spin
        // spring damping
        gamma_fac   = 1.0;    // initial factor for initial damping value for springs
        gamma_all   = 1.0;    // final damping coeff
        t_damp      = 1.0;    // gamma to final values for all springs at this time
        ks          = 8e-2;   // spring constant

        ratio1 =0.7; // shape of resolved body  y/x b/a
        ratio2 =0.5; // z/x
        t_print =  1.0;  // printouts for table
        t_heat =  10000.0;  // heat printouts 
        r_I = 0.0;   // radius where to change ks  for interior
        ks_I = 0.008;  // interior ks value 
        gamma_I = gamma_all;  // interior gamma value 
        powerfac = 1.0; // fraction of heat to center of springs
        obliq_deg=0.0;
     }
     else{
        FILE *fpi;
        fpi = fopen(argv[1],"r");
        char line[300];
        fgets(line,300,fpi);  sscanf(line,"%s",froot);
        fgets(line,300,fpi);  sscanf(line,"%lf",&dt);
        fgets(line,300,fpi);  sscanf(line,"%lf",&tmax);
        fgets(line,300,fpi);  sscanf(line,"%lf",&t_print);
        fgets(line,300,fpi);  sscanf(line,"%lf",&t_heat);
        fgets(line,300,fpi);  sscanf(line,"%lf",&powerfac);
        fgets(line,300,fpi);  sscanf(line,"%d" ,&lattice_type);
        fgets(line,300,fpi);  sscanf(line,"%lf",&b_distance);
        fgets(line,300,fpi);  sscanf(line,"%lf",&mush_fac);
        fgets(line,300,fpi);  sscanf(line,"%lf",&ratio1);
        fgets(line,300,fpi);  sscanf(line,"%lf",&ratio2);
        fgets(line,300,fpi);  sscanf(line,"%lf",&omegaz);
        fgets(line,300,fpi);  sscanf(line,"%lf",&obliq_deg);
        fgets(line,300,fpi);  sscanf(line,"%lf",&gamma_fac);
        fgets(line,300,fpi);  sscanf(line,"%lf",&gamma_all);
        fgets(line,300,fpi);  sscanf(line,"%lf",&t_damp);
        fgets(line,300,fpi);  sscanf(line,"%lf",&ks);
        fgets(line,300,fpi);  sscanf(line,"%lf",&r_I);
        fgets(line,300,fpi);  sscanf(line,"%lf",&ks_I);
        fgets(line,300,fpi);  sscanf(line,"%lf",&gamma_I);
        fgets(line,300,fpi);  sscanf(line,"%lf",&eps_c);
        fgets(line,300,fpi);  sscanf(line,"%lf",&extra);
        fgets(line,300,fpi);  sscanf(line,"%d",&npointmass);
        for (int ip=0;ip<npointmass;ip++){
           fgets(line,300,fpi);  sscanf(line,"%lf %lf %lf %lf",
             mp+ip,rad+ip,itaua+ip,itaue+ip);
           fgets(line,300,fpi);  sscanf(line,"%lf %lf %lf %lf %lf %lf",
             aa+ip,ee+ip,ii+ip,longnode+ip,argperi+ip,meananom+ip);
        }


     }
     double obliquity = obliq_deg*M_PI/180.0;
     if (powerfac >1.0) powerfac = 1.0;
     if (powerfac <0.0) powerfac = 1.0;

/// end of things to set /////////////////////////

        r->dt=dt; // set integration timestep
	const double boxsize = 3.2*rball;    // display
	reb_configure_box(r,boxsize,1,1,1);
	r->softening      = b_distance/100.0;	// Gravitational softening length


   // properties of springs
   spring_mush.gamma     = gamma_fac*gamma_all; // initial damping coefficient
   spring_mush.ks        = ks; // spring constant
   spring_mush.smax      = 1e6; // not used currently
   double mush_distance=b_distance*mush_fac; 
       // distance for connecting and reconnecting springs

   FILE *fpr;
   char fname[200];
   sprintf(fname,"%s_run.txt",froot);
   fpr = fopen(fname,"w");

   NS=0; // start with no springs 

// do you want volume to be the same? adjusting here!!!!
   double volume_ratio = pow(rball,3.0)*ratio1*ratio2;  // neglecting 4pi/3 factor
   double vol_radius = pow(volume_ratio,1.0/3.0);

   rball /= vol_radius; // volume radius used to compute semi-major axis
// assuming that body semi-major axis is rball
   fprintf(fpr,"a %.3f\n",rball); 
   fprintf(fpr,"b %.3f\n",rball*ratio1); 
   fprintf(fpr,"c %.3f\n",rball*ratio2); 
   volume_ratio = pow(rball,3.0)*ratio1*ratio2;  // neglecting 4pi/3 factor
   fprintf(fpr,"vol_ratio %.6f\n",volume_ratio); // with respect to 4pi/3 
   // so I can check that it is set to 1

   // create particle distribution
   if (lattice_type==0){
      // rand_football_from_sphere(r,b_distance,rball,rball*ratio1, rball*ratio2,mball );
      rand_football(r,b_distance,rball,rball*ratio1, rball*ratio2,mball );
   }
   if (lattice_type ==1){
      fill_hcp(r, b_distance, rball , rball*ratio1, rball*ratio2, mball);
   }
   if (lattice_type ==2){
      fill_cubic(r, b_distance, rball , rball*ratio1, rball*ratio2, mball);
   }

   int il=0;
   int ih=r->N;
   centerbody(r,il,ih);  // move reference frame to resolved body 
   subtractcov(r,il,ih);  // subtract center of velocity
   // spin it
   spin(r,il, ih, 0.0, 0.0, omegaz);  // you can change one of these to tilt!
   subtractcov(r,il,ih);  // subtract center of velocity
   double speriod  = fabs(2.0*M_PI/omegaz);
   printf("spin period %.6f\n",speriod);
   fprintf(fpr,"spin period %.6f\n",speriod);
   fprintf(fpr,"omegaz %.6f\n",omegaz);
   if (obliquity != 0.0)
     rotate_body(r, il, ih, 0.0, obliquity, 0.0); // tilt by obliquity in radians

   // make springs, all pairs connected within interparticle distance mush_distance
   connect_springs_dist(r,mush_distance, 0, r->N, spring_mush);

   // assume minor semi is rball*ratio2
   double ddr = rball*ratio2 - 0.5*mush_distance;
   ddr = 0.4;
   double Emush = Young_mush(r,il,ih, 0.0, ddr);
   double Emush_big = Young_mush_big(r,il,ih);
   printf("Young's modulus %.6f\n",Emush);
   printf("Young's modulus big %.6f\n",Emush_big);
   fprintf(fpr,"Young's_modulus %.6f\n",Emush);
   fprintf(fpr,"Young's_modulus big %.6f\n",Emush_big);
   printf("ddr = %.3f mush_distance =%.3f \n",ddr,mush_distance);
   fprintf(fpr,"mush_distance %.4f\n",mush_distance);
   double LL = mean_L(r); 
   printf("mean L = %.4f\n",LL);
   fprintf(fpr,"mean_L  %.4f\n",LL);
   if (r_I > 0.0){  // change core strength and gamma !!!
      // adjust_ks(r, npert, ks_I, gamma_I*gamma_fac, 0.0, r_I);
      adjust_ks_abc(r, npert, ks_I, gamma_I*gamma_fac, r_I, r_I, r_I*(1.0 + eps_c));
      double Emush = Young_mush(r,il,ih, 0.0, r_I);
      printf("Young's modulus Interior different %.6f\n",Emush);
      fprintf(fpr,"Young's_modulus %.6f\n",Emush);
   }

   double om = 0.0; // set up the perturbing central mass
   if (npointmass >0){
      // set up central star
      int ip=0;
      om = add_pt_mass_kep(r, il, ih, -1, mp[ip], rad[ip],
           aa[ip],ee[ip], ii[ip], longnode[ip],argperi[ip],meananom[ip]);
      fprintf(fpr,"resbody mm=%.3f period=%.2f\n",om,2.0*M_PI/om);
      printf("resbody mm=%.3f period=%.2f\n",om,2.0*M_PI/om);
      icentral = ih;
      double na = om*aa[ip];
      double adot = 3.0*mp[ip]*na/pow(aa[ip],5.0); // should approximately be adot
      fprintf(fpr,"adot %.3e\n",adot);
      // set up rest of point masses
      for(int ipp = 1;ipp<npointmass;ipp++){
          double omp; // central mass assumed to be first one ipp=ih = icentral
          omp = add_pt_mass_kep(r, il, ih, icentral, mp[ipp], rad[ipp],
             aa[ipp],ee[ipp], ii[ipp], longnode[ipp],argperi[ipp],meananom[ipp]);
          fprintf(fpr,"pointm %d mm=%.3f period=%.2f\n",ipp,omp,2.0*M_PI/omp);
          printf("pointm %d mm=%.3f period=%.2f\n",ipp,omp,2.0*M_PI/omp);
      }
      npert = npointmass;
   }

   // factor of 0.5 is due to reduced mass being used in calculation
   double tau_relax = 1.0*gamma_all*0.5*(mball/(r->N -1))/spring_mush.ks; // Kelvin Voigt relaxation time
   printf("relaxation time %.3e\n",tau_relax);
   fprintf(fpr,"relaxation_time  %.3e\n",tau_relax);

   double barchi = 2.0*fabs(om - omegaz)*tau_relax;  // initial value of barchi
   fprintf(fpr,"barchi  %.4f\n",barchi);
   printf("barchi %.4f\n",barchi);


   double Nratio = (double)NS/(double)r->N;
   printf("N=%d  NS=%d NS/N=%.1f\n", r->N, NS, Nratio);
   fprintf(fpr,"N=%d  NS=%d NS/N=%.1f\n", r->N, NS,Nratio);
   fclose(fpr);

   reb_springs(r); // pass spring index list to display
   r->heartbeat = heartbeat;
#ifdef LIBPNG
// system("mkdir png");
#endif // LIBPNG

   if (tmax ==0.0)
      reb_integrate(r, INFINITY);
   else
      reb_integrate(r, tmax);
}


#define NSPACE 50
void heartbeat(struct reb_simulation* const r){
        char hfile[100];
        static int first=0;
        static char extendedfile[50];
        static char pointmassfile[NPMAX*NSPACE];
        if (first==0){
           first=1;
           sprintf(extendedfile,"%s_ext.txt",froot);
           for(int i=0;i<npert;i++){
             sprintf(pointmassfile+i*NSPACE,"%s_pm%d.txt",froot,i);
           }
        }

	if (reb_output_check(r,10.0*r->dt)){
		reb_output_timing(r,0);
	}
        if (fabs(r->t - t_damp) < 0.9*r->dt) set_gamma_fac(gamma_fac); 
            // damp initial bounce only 
            // reset gamma only at t near t_damp
	
         // stuff to do every timestep
         centerbody(r,0,r->N-npert);  // move reference frame, position only
         addto_heatvec(r); // store heat accumulated each timestep

         if (reb_output_check(r,t_print)) {
            print_extended(r,0,r->N-npert,extendedfile); // orbital info and stuff
            if (npert>0)
               for(int i=0;i<npert;i++){
                  int ip = icentral+i;
                  print_pm(r,ip,i,pointmassfile+i*NSPACE);
               }
         }


	 if (reb_output_check(r,t_heat)) { // heat files
            int ndt = (int)(t_heat/r->dt);
            hfilename(r,froot, t_heat, hfile);
            print_heat(r,npert,hfile,ndt,powerfac); // heat info printed out!
         }

}

// make a spring index list
void reb_springs(struct reb_simulation* const r){
   r->NS = NS;
   r->springs_ii = malloc(NS*sizeof(int));
   r->springs_jj = malloc(NS*sizeof(int));
   for(int i=0;i<NS;i++){
     r->springs_ii[i] = springs[i].i;
     r->springs_jj[i] = springs[i].j;
   }
}


