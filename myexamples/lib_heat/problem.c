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

double gamma_fac; // for adjustment of gamma of all springs
double t_damp;    // end faster damping, relaxation
double t_print;   // for table printout 
double t_heat;    // for heat  printout 
char froot[30];   // output files
int npert; // number of point mass perturbers
double powerfac; // fraction of heat to center of spring rather than nodes

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
        double dt, b_distance,omegaz,aa,ks,m1,mush_fac,gamma_all,ee;
        double ratio1,ratio2,ks_I,r_I,dmfac,xside,gamma_I,obliq_deg;
        int type,nporb;

    if (argc ==1){
        strcpy(froot,"t1");   // to make output files
	dt	   = 1e-3;    // Timestep
	type       =0;        // 0=rand 1=hcp
        b_distance = 0.15;    // for creating random sphere, min separation between particles
        mush_fac    = 2.3;    // ratio of smallest spring distance to minimum interparticle dist
        omegaz     = 0.2;     // initial spin
        // spring damping
        gamma_fac   = 1.0;    // initial factor for initial damping value for springs
        gamma_all   = 1.0;    // final damping coeff
        t_damp      = 1.0;    // gamma to final values for all springs at this time
        ks          = 8e-2;   // spring constant
	// orbit
        m1          = 0.1;   // mass of central body
        aa          = 7.0;   // semi-major axis of m1 from resolved body
	ee           =0.0;   // initial eccentricity

        ratio1 =0.7; // shape of resolved body  y/x b/a
        ratio2 =0.5; // z/x
        t_print =  1.0;  // printouts for table
        t_heat =  10000.0;  // heat printouts 
        r_I = 0.0;   // radius where to change ks  for interior
        ks_I = 0.008;  // interior ks value 
        gamma_I = gamma_all;  // interior gamma value 
	dmfac = 0.0;   // to make lopsided body
	xside = 1.0; //
        nporb = 0; // number of orbital periods to integrate
        powerfac = 1.0; // fraction of heat to center of springs
        obliq_deg=0.0;
     }
     else{
        FILE *fpi;
        fpi = fopen(argv[1],"r");
        char line[300];
        fgets(line,300,fpi);  sscanf(line,"%s",froot);
        fgets(line,300,fpi);  sscanf(line,"%lf",&dt);
        fgets(line,300,fpi);  sscanf(line,"%d",&type);
        fgets(line,300,fpi);  sscanf(line,"%lf",&b_distance);
        fgets(line,300,fpi);  sscanf(line,"%lf",&mush_fac);
        fgets(line,300,fpi);  sscanf(line,"%lf",&omegaz);
        fgets(line,300,fpi);  sscanf(line,"%lf",&gamma_fac);
        fgets(line,300,fpi);  sscanf(line,"%lf",&gamma_all);
        fgets(line,300,fpi);  sscanf(line,"%lf",&t_damp);
        fgets(line,300,fpi);  sscanf(line,"%lf",&ks);
        fgets(line,300,fpi);  sscanf(line,"%lf",&m1);
        fgets(line,300,fpi);  sscanf(line,"%lf",&aa);
        fgets(line,300,fpi);  sscanf(line,"%lf",&ee);
        fgets(line,300,fpi);  sscanf(line,"%lf",&ratio1);
        fgets(line,300,fpi);  sscanf(line,"%lf",&ratio2);
        fgets(line,300,fpi);  sscanf(line,"%lf",&t_print);
        fgets(line,300,fpi);  sscanf(line,"%lf",&t_heat);
        fgets(line,300,fpi);  sscanf(line,"%lf",&r_I);
        fgets(line,300,fpi);  sscanf(line,"%lf",&ks_I);
        fgets(line,300,fpi);  sscanf(line,"%lf",&gamma_I);
        fgets(line,300,fpi);  sscanf(line,"%lf",&dmfac);
        fgets(line,300,fpi);  sscanf(line,"%lf",&xside);
        fgets(line,300,fpi);  sscanf(line,"%d",&nporb);
        fgets(line,300,fpi);  sscanf(line,"%lf",&powerfac);
        fgets(line,300,fpi);  sscanf(line,"%lf",&obliq_deg);

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

// binary in circular or eccentric orbit
   npert = 1;  // number of perturbing masses
   double r1 = rball*pow(m1,1.0/3.0);  // radius as expected from density
   double r1min = 0.5;
   if (r1 < r1min) r1 = r1min;
   if (r1 > aa/2) r1 = aa/2;
   r1 = 1.0; // hard set here!!!!
   // double theta1 =0.0*M_PI/2.0;  // rotate x-> y of initial condition

   
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
   if (type==0){
      // rand_football_from_sphere(r,b_distance,rball,rball*ratio1, rball*ratio2,mball );
      rand_football(r,b_distance,rball,rball*ratio1, rball*ratio2,mball );
   }
   if (type ==1){
      fill_hcp(r, b_distance, rball , rball*ratio1, rball*ratio2, mball);
   }
   if (type ==2){
      fill_cubic(r, b_distance, rball , rball*ratio1, rball*ratio2, mball);
   }

   int il=0;
   int ih=r->N;
   centerbody(r,il,ih);  // move reference frame to resolved body 
   // spin it
   spin(r,il, ih, 0.0, 0.0, omegaz);  // you can change one of these to tilt!
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
   printf("ddr = %.3f mush_distance =%.3f \n",ddr,mush_distance);
   printf("Young's modulus %.6f\n",Emush);
   fprintf(fpr,"Young's_modulus %.6f\n",Emush);
   fprintf(fpr,"mush_distance %.4f\n",mush_distance);
   double LL = mean_L(r); 
   printf("mean L = %.4f\n",LL);
   fprintf(fpr,"mean_L  %.4f\n",LL);
   if (r_I > 0.0){  // change core strength and gamma !!!
      adjust_ks(r, npert, ks_I, gamma_I*gamma_fac, 0.0, r_I);
      double Emush = Young_mush(r,il,ih, 0.0, r_I);
      printf("Young's modulus Interior different %.6f\n",Emush);
      fprintf(fpr,"Young's_modulus %.6f\n",Emush);
   }
   if (fabs(dmfac) > 0.01){  // make lopsided by increases masses on one side!!!
      adjust_mass_side(r, npert, 1.0+dmfac, xside);
      // heavy side toward perturber with theta1=0 and dmfac >0
   }

   double om = 0.0; // set up the perturbing central mass
   if (m1>0.0){
      double ii      = 0.0;
      double argperi = 0.0;
      double meananom= 0.0;
      double longnode= 0.0;
      om = add_pt_mass_kep(r, il, ih, -1, m1, r1,
        aa,ee, ii, longnode,argperi,meananom);

      // om=add_one_mass_kep(r, m1, aa, ee, ii,longnode,argperi,meananom,mball,r1);
      npert = 1;
   }
   double period = 2.0*M_PI/om;  // initial orbital rotation period
   printf("rot_period %.3f \n", period);
   printf("mm  %.3f\n",om);
   fprintf(fpr,"rot_period  %.3f \n",period);
   fprintf(fpr,"mm  %.6f\n",om);

   // double eps = epsilon_mom(r, il, ih);
   // printf("eps= %.3e\n",eps);
   // fprintf(fpr,"eps  %.3e\n",eps);

   // note no 2.5 here!
   // factor of 0.5 is due to reduced mass being used in calculation
   double tau_relax = 1.0*gamma_all*0.5*(mball/(r->N -1))/spring_mush.ks; // Kelvin Voigt relaxation time
   printf("relaxation time %.3e\n",tau_relax);
   fprintf(fpr,"relaxation_time  %.3e\n",tau_relax);

   double barchi = 2.0*fabs(om - omegaz)*tau_relax;  // initial value of barchi
   fprintf(fpr,"barchi  %.4f\n",barchi);
   printf("barchi %.4f\n",barchi);
   // fprintf(fpr,"posc %.6f\n",posc);
   // t_damp = posc;   // for initial damping
   tmax = period*nporb; // integration time

   double na = om*aa;
   double adot = 3.0*m1*na/pow(aa,5.0); // should approximately be adot
   fprintf(fpr,"adot %.3e\n",adot);

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



void heartbeat(struct reb_simulation* const r){
        static int first=0;
        static char tabfile[50];
        static char binfile[50];
        char hfile[100];
        if (first==0){
           first=1;
           sprintf(tabfile,"%s_tab.txt",froot);
           sprintf(binfile,"%s_bin.txt",froot);
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
            print_tab(r,npert,tabfile); // orbital info and stuff
            print_bin(r,npert,binfile);
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


