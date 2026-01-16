#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <cmath>

#include "param.h"
#include "calc_covar.h"
#include "arrivals.h"
#include "fw_tdx.h"
#include "gather.h"

void loc_hypo_lm(std::array<double, 4>& hypo, int m, double* fvec, 
                 double* fjac, bool fix_depth, int& info);

//---------------------------------------------------------------------
// Implementation of the pure function enorm
// Calculates and returns the Euclidean (L2) norm of the vector x.
//---------------------------------------------------------------------
double enorm(int n, double* x) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += x[i] * x[i];
    }
    return std::sqrt(sum);
}

    // allocatable array fjac of double 2D array
    // where data are contiguous in memory
    // fjac = new double[marr][4];
 
// https://stackoverflow.com/questions/5901476/sending-and-receiving-2d-array-over-mpi/5901671#5901671
/*
    int **alloc_2d_int(int rows, int cols) {
    int *data = (int *)malloc(rows*cols*sizeof(int));
    int **array= (int **)malloc(rows*sizeof(int*));
    for (int i=0; i<rows; i++)
        array[i] = &(data[cols*i]);

    return array;
}
int **A;
A = alloc_2d_init(N,M);
// ...
free(A[0]); // free the data
free(A); // free the array of pointers
*/

double **alloc_2d_double(int rows, int cols) {
	double *data = new double[rows * cols];
	double **array = new double*[rows];
	for (int i = 0; i < rows; i++)
		array[i] = &(data[cols * i]);
	return array;
}

double **alloc_2d_double_F(int rows, int cols) {
	double *data = new double[rows * cols];
	double **array = new double*[cols];
	for (int i = 0; i < cols; i++)
		array[i] = &(data[rows * i]);
	return array;
}

int get_hypo(const CArrivalFile & arrs, const TParams &param,
	       std::array<double,4> & hypo, double (& covM)[4][4]) {

    int marr, narr;
    marr = arrs.n_arr;
    double* fvec = new double[marr];
    double* w = new double[marr];
    double** fjac=alloc_2d_double_F(marr, 4);
    
    int info=0;
    //std::array<std::array<double,4>,4> covM;

    if (param.fix[0] || param.fix[1]) {
        hypo = param.startpt;
	// origin time to hypo[3]
	gather.collect(hypo.data());
        gather.get_topt(hypo.data());
    } else {
        // not xyz fix
        narr = 0;
        for (int i = 0; i < marr; i++) {
            if (arrs.arr[i].wt > 0) {
                narr = narr + 1;
            }
        }
        if (narr < 3) {
            std::cout << " # of arrivals in hyp_file  <  3\n";
            return 1;
        }
        //  search the nearest station
	//  it is the one with the smallest arrival time
        int n0 = 0; 
        double t0 = 1e20;
        for (int i = 0; i < marr; i++) {
            if (arrs.arr[i].phase.empty()) continue; 
            if (arrs.arr[i].trec > t0) continue; 
            t0 = arrs.arr[i].trec;
            n0 = i;
        }
        //  coord of the nearest station
        double x0 = arrs.arr[n0].X;
        double y0 = arrs.arr[n0].Y;
        //double z0 = arrs.arr[n0].Z;
   
        // initial hypocenter guess
	// slight offset from the nearest station
        hypo = {x0+0.1, y0+0.1, 0.0, t0-0.5};
        hypo[2] = param.startpt[2]; // start from the depth
	// if depth is zero and is not fixed,
	// use the startpt depth 0.01 km
	// to avoid singularity
	if (hypo[2] <= 0.0 && !param.fix[2]) {
		hypo[2] = 0.01;
	}

	// LM iterative solution (Levenberg-Marquardt)
        loc_hypo_lm(hypo, marr, fvec, &fjac[0][0], param.fix[2], info);

	// We call gather.collect again here, because the last call inside
	// the iterative solution may not have been made with the resulting
	// value of hypo
	gather.collect(hypo.data());
        /*
	std::cout << "     FINAL L2 NORM OF THE RESIDUALS"
         << std::setw(15) << std::fixed << std::setprecision(7) << enorm(marr, fvec) << std::endl;
        */
    } // end if not xyz fix

/*
    // print hypo values for debug
    std::cout << "Hypocenter solution:\n";
    std::cout << "  X = " << hypo[0] << " km\n";
    std::cout << "  Y = " << hypo[1] << " km\n";
    std::cout << "  Z = " << hypo[2] << " km\n";
    std::cout << "  Origin Time = " << hypo[3] << " sec\n";

    gather.print();
   
    gather.get_res_w(hypo.data(), marr, fvec, w);

    // print fvec values for debug
    for (int i = 0; i < marr; i++) {
	std::cout << "fvec[" << i << "] = " << fvec[i] << std::endl;
    }
    
*/    

    // covariance
    gather.get_res_w(hypo.data(), marr, fvec, w);
    gather.get_G_w(marr, &(fjac[0][0]), w);
    for (int i = 0; i < marr; i++) {
        fvec[i] = fvec[i] * w[i];
    }
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < marr; i++) {
            fjac[j][i] = fjac[j][i] * w[i];  // column-wise
        }
    }

    cov_matrix2(&(covM[0][0]), marr, 4, fvec, &(fjac[0][0]), w, param.reading_err, param.model_err);
    // print covariance matrix
    /*
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << std::fixed << std::setw(16) << std::setprecision(6) 
                      << covM[i][j] << " ";
        }
        std::cout << "\n";
    }
    */
    // free memory
    delete[] fvec; 
    delete[] w;
    //for (int j = 0; j < 4; j++) delete[] fjac[j];
    delete[] fjac[0]; // free the data in flat array
    delete[] fjac; // free the array of pointers

    return info;
} // end get_hypo
 
