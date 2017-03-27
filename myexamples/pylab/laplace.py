import numpy as np


# Laplace coefficients via summation
#   equation 6.68 of Murray and Dermott, page 237
#   b_s^j (alpha)
# This program can be checked against expressions and values
# on page 262-263 in Murray and Dermott
ACC=1e-10  # accuracy required for coefficient, size of last term 
# this routine could be improved to have an alpha near 1 special case
def b_laplace(alpha, s, j):
    diff = 100.0;
    a2 = alpha*alpha;
    j_a = np.int(np.abs(j));

    if (alpha >1.0):
        print("b_laplace: alpha > 1.0, error");
        return -1.0;
   
   # outside the sum */
    osum= 1.0;
    for k in range(0,j_a):
        osum = osum*(s+k)/(k+1.0);
        
    osum = osum*alpha**j_a
    
    i = 0;
    # this is the function F(alpha2) in problem 6.2
    fac = 1.0;
    ssum = 1.0;
    while(diff>ACC):
        fac = fac*(s+i)*(s+j_a+i)/((i+1.0)*(j_a+i+1.0))*a2;
        ssum = ssum + fac;
        diff = osum*fac;
        i=i+1
   
    return ssum*osum*2.0;


# Derivative of the laplace coeficient
#   Db_s^j = d b_s^j/d alpha
#   From equation 6.70 from Murray and Dermott page 237 
def D_b_laplace(alpha, s, j):
    ja = np.fabs(j);
    return     s*(b_laplace(alpha, s+1.0, ja-1.0)\
       -2.0*alpha*b_laplace(alpha, s+1.0, ja   )\
             +    b_laplace(alpha, s+1.0, ja+1.0));


# Second Derivative of the laplace coeficient
#   D^2 b_s^j = d^2 b_s^j/d alpha^2
#   From equation 6.71 from Murray and Dermott page 237  
def D2_b_laplace(alpha, s, j):
    ja = np.abs(j);
    return   s*(D_b_laplace(alpha, s+1.0, ja-1.0)\
     -2.0*alpha*D_b_laplace(alpha, s+1.0, ja    )\
           +    D_b_laplace(alpha, s+1.0, ja+1.0)\
         -2.0 * b_laplace(alpha, s+1.0, ja));


# Third  Derivative of the laplace coeficient
#   D^3 b_s^j = d^3 b_s^j/d alpha^3
#   From equation 6.71 from Murray and Dermott page 237  
def D3_b_laplace(alpha, s,  j):
    ja = np.abs(j);
    return   s*(D2_b_laplace(alpha, s+1.0, ja-1.0)\
     -2.0*alpha*D2_b_laplace(alpha, s+1.0, ja    )\
           +    D2_b_laplace(alpha, s+1.0, ja+1.0)\
        -2.0*2.0*D_b_laplace(alpha, s+1.0, ja));



# Fourth Derivative of the laplace coeficient
#   D^4 b_s^j = d^4 b_s^j/d alpha^4
#   From equation 6.71 from Murray and Dermott page 237  */
def D4_b_laplace(alpha,  s, j):
    ja = np.abs(j);
    return   s*(D3_b_laplace(alpha, s+1.0, ja-1.0)\
      -2.0*alpha*D3_b_laplace(alpha, s+1.0, ja    )\
            +    D3_b_laplace(alpha, s+1.0, ja+1.0)\
        -2.0*3.0*D2_b_laplace(alpha, s+1.0, ja));


# testing!
#j = 3.0; s = 0.5; alpha = 0.480597
#A5 = (21.0*b_laplace(alpha,s,j)\
#           +10.0*alpha*D_b_laplace(alpha,s,j)\
#           +alpha*alpha*D2_b_laplace(alpha,s,j) )/8.0;
#print(A5);
#
# check routines  with A5 on page 262 (M+D)*/
# which is A=0.598100 for alpha3:1=0.480597


#from laplace import *
#j = 3.0; s = 0.5; alpha = 0.480597
#A5 = (21.0*b_laplace(alpha,s,j)\
#           +10.0*alpha*D_b_laplace(alpha,s,j)\
#           +alpha*alpha*D2_b_laplace(alpha,s,j) )/8.0;
#print(A5);
