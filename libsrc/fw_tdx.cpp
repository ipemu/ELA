#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <cstdlib>  // for std::abs

#include "gather.h"


    // Returns |a| with the sign of b.
    inline double sign(double a, double b) {
        return (b >= 0 ? std::fabs(a) : -std::fabs(a));
    }

    // transform from bounded parameter space Xp to optimization procedure space Xo
    std::array<double,4> Xo(const double* x, double zmin) {
        double o;
        // reflective transformation zp -> zo
        o = std::sqrt(std::pow((x[2] - zmin + 1.0), 2.0) - 1.0);
        return { x[0] - 1100.0, x[1] - 700.0, o, x[3] };
    }

    // transform from optimization procedure space Xo to bounded parameter space Xp
    std::array<double,4> Xp(const double* x, double zmin) {
        double p;
        // reflective transformation zo -> zp
        p = x[2];
        p = zmin - 1.0 + std::sqrt(p * p + 1.0);
        return { x[0] + 1100.0, x[1] + 700.0, p, x[3] };
    }
/* alternative with static array
    // transform from optimization procedure space Xo to bounded parameter space Xp
    double (&Xp(const double (& x)[4], double zmin))[4] {
        double p;
        // reflective transformation zo -> zp
        p = x[2];
        p = zmin - 1.0 + std::sqrt(p * p + 1.0);
        static double xp[4] = { x[0] + 1100.0, x[1] + 700.0, p, x[3] };
        return xp;
    }
*/
    // function dpdo(z_o)
    // the derivative of the z-coordinate transformation 
    // df(y)/dx = df(y)/dy * dy(x)/dx
    // x = z_o, y = z_p(z_o) = sqrt(z_o^2+1)-1
    double dpdo(double z_o) {
        double r;
        r = z_o / std::sqrt(1.0 + z_o * z_o);
        // prevention of zero
        r = r + sign(0.001, z_o);
        return r;
    }


extern "C" {
void tdxder(int* p_m, int* p_n, double* x, double* fvec, double* fjac, int* p_ldjac, int* p_iflag) {
    // Local variables
    int m = *p_m;
    int n = *p_n;
    int ldjac = *p_ldjac;
    int iflag = *p_iflag;
    
    double w[m];
    int mres;
    // transform from optimization x to bounded parametric xp space
    auto xp_=Xp(x, 0.0);  // relief = 0.0
    if (iflag == 1) {
	// solution of the forward problem for given xp_
	mres=gather.collect(xp_.data());
	// residuals between observed and calculated data and their weights
        mres=gather.get_res_w(xp_.data(), m, fvec, w);
        // fvec = fvec * w (elementwise multiplication)
        for (int j = 0; j < m; j++) {
            fvec[j] = fvec[j] * w[j];
        }
    } else if (iflag == 2) {
	// we do not need to solve the forward problem again
	// because it was already solved when iflag == 1
	// just get the Jacobian matrix and weights
        mres=gather.get_G_w(m, fjac, w);
        // transform dt/dz from bounded xp -> optimization xo space
        // dpdo = dz_p(z_o)/dz_o 
        for (int j = 0; j < m; j++) {
            fjac[j+2*m] *= dpdo(x[2]);
        }

        // Multiply each column elementwise by w
        for (int j = 0; j < m; j++) {
            float wt = w[j];
            fjac[j+0*m] *= wt;
	    fjac[j+1*m] *= wt;
	    fjac[j+2*m] *= wt;
	    fjac[j+3*m] *= wt;
	}
    } else {
        // print xp_
        std::cout << "xp_ = ";
        for (int i = 0; i < n; i++) {
            std::cout << xp_[i] << " ";
        }
        std::cout << std::endl;
    }
}

// subroutine tdx3 will be called by the optimization procedure lmder
// of standard fortran library MINPACK
// tdx3 is a function that computes the residuals and the Jacobian
// of the function f(x) = 0
// x = (x[0], x[1], x[2], x[3]), where x[2] is the depth fixed
// ldfjac = leading dimension of fjac
// iflag = 1: compute fvec
// iflag = 2: compute fjac
// m = number of residuals
// n = number of parameters

void tdx3(int* p_m, int* p_n, double* x, double* fvec, double* fjac, int* p_ldjac, int* p_iflag) {
    // Local variables
    int m = *p_m;
    int n = *p_n;
    int ldjac = *p_ldjac;
    int iflag = *p_iflag;

    double w[m];
    int mres;

/*    
    // print iflag and x for debugging
    std::cout << "iflag = " << iflag << std::endl;
    std::cout << "x = ";
    for (int i = 0; i < n; i++) {
	std::cout << x[i] << " ";
    }
    std::cout << std::endl;
*/
/*
 * Example output for debugging: (fixed depth x[2] = 1)
iflag = 1
x = 1099.58 766.639 1 1.411 
iflag = 2
x = 1099.58 766.639 1 1.411 
iflag = 1
x = 1099.92 766.573 1 1.68392 
iflag = 2
x = 1099.92 766.573 1 1.68392 
iflag = 1
x = 1099.93 766.571 1 1.68452 
iflag = 2
x = 1099.93 766.571 1 1.68452 
iflag = 1
x = 1099.92 766.574 1 1.6836 
     EXIT PARAMETER                         2
     FINAL APPROXIMATE SOLUTION   1099.9252796    766.5714316      1.0000000      1.6845235
Analyzing the debug output, we see that the optimization procedure
calls tdx3 alternately with iflag = 1 and iflag = 2 for each iteration of x.
For each iteration of x, first fvec is computed (iflag = 1),
then fjac is computed (iflag = 2).

The last call is with iflag = 1 for an unsuccessful iteration of x.
Therefore, after completing the optimization calculation, it is necessary
to recollect the gather data for the resulting solution of vector x.
*/   
    if (iflag == 1) {
	// solution of the forward problem for given x
	mres=gather.collect(x);
	//gather.print();
	// residuals between observed and calculated data and their weights
        mres=gather.get_res_w(x, m, fvec, w);
	// residuals with weights
	// fvec = fvec * w (elementwise multiplication)
        for (int j = 0; j < m; j++) {
            fvec[j] = fvec[j] * w[j];
        }
    } else if (iflag == 2) {
	// we do not need to solve the forward problem again
	// because it was already solved when iflag == 1
	// just get the Jacobian matrix and weights
        mres=gather.get_G_w(m, fjac, w);
	// Jacobian with weights
        for (int j = 0; j < m; j++) {
            float wt = w[j];
            fjac[j+0*m] *= wt;
	    fjac[j+1*m] *= wt;
	    fjac[j+2*m]  = 0.0;   //fix depth
	    fjac[j+3*m] *= wt;
        }
    } else {
        // print x
        std::cout << "x = ";
        for (int i = 0; i < n; i++) {
            std::cout << x[i] << " ";
        }
        std::cout << std::endl;
    }
}
} // extern "C"
