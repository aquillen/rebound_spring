/**
 * @file        spring.h
 * @brief       Particle pair structure 
 * @author      Alice Quilen 
 */


#ifndef _SPRING_H
#define _SPRING_H


#include"particle.h"

// spring structure 
struct spring  {
	double ks;   // spring constant
	double rs0;  // distance of no force
	double gamma; // damping coefficient
	double smax;  // 
	int    i;    // vertex 1 referring to a particle
	int    j;    // vertex 2 referring to a particle
};


extern struct spring* springs;
extern int NS; // numbers of springs
extern int NPERT; // number of point masses
extern double b_distance; // mush formation
extern double mush_distance; // mush spring connection 
extern double t_reform; // formation springs timescale
extern double gamma_all; // for gamma  of all springs

void spring_forces();

void adjust_ks_abc();
void subtractcov();
void subtractcom();
void print_extended();
void print_pm();
void hfilename();
double add_pt_mass_kep();
void move_resolved();
double sum_mass();
double dEdt();
void spr_xyz_mid();
void compute_com();
void compute_cov();
void compute_Lorb();
double spring_length();
double strain();
void springs_add();
void normalize ();
double mindist();
void centerbody();
void connect_springs_dist();
void rand_football();
void rand_football_from_sphere();
double Young_mush();
double Young_mush_big();
void set_gamma();
void set_gamma_fac();
void spin();
void make_binary_spring();
void mom_inertia();
void measure_L();
void compute_semi();
void compute_semi_bin();
void total_mom();
double mean_L();
void spring_init(struct reb_simulation* r);
void output_png();
void output_png_single();
void spr_ang_mid();
void body_spin();
void print_tab();
void print_bin();
void print_heat();
void invI();
double detI();
void eigenvalues();
void adjust_ks();
void adjust_mass_side();
void rotate_body();
double fill_hcp();
double fill_cubic();
double add_pluto_charon();
double add_pluto_charon_kep();
double add_one_mass();
double add_one_mass_kep();
void dodrift_bin();
void dodrift_res();
void addto_heatvec();
void norm_heatvec();


#endif // _SPRING_H


