#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

#include "arrivals.h"
#include "twopoint.h"

/*
// Gather class
// This class is used to collect the travel time information from the arrivals
// and calculate the residuals and weights for the inversion process.

class Gather {
private:
   int refresh_token = 0;
public:
   int ngather = 0; // number of gathers
   std::vector<double> ttcal;  // travel time calculated
   std::vector<double> dtdx;   // dt/dx
   std::vector<double> dtdy;   // dt/dy
   std::vector<double> dtdz;   // dt/dz
   std::vector<double> toa;    // take-off angle
   std::vector<double> trec;   // time recorded 
   std::vector<double> wt;     // weight

   void resize(int n) {
       ttcal.resize(n);
       dtdx.resize(n);
       dtdy.resize(n);
       dtdz.resize(n);
       toa.resize(n);
       trec.resize(n);
       wt.resize(n);
       ngather = n;
   }  

  // ~Gather() {
  // }  

   void clear() {
	ttcal.clear();
	dtdx.clear();
	dtdy.clear();
	dtdz.clear();
	toa.clear();
	trec.clear();
	wt.clear();
        ngather = 0;
   }

};
*/
#include "gather.h"

//   int Gather::collect(const double (& x)[4]) {

   int Gather::collect(const double* x) {
       std::string ph;
       double td[4];
       double takeoffangle;
       double obsxyz[3];
       double srcxyz[3] = {x[0], x[1], x[2]};

       int narr = arrs.arr.size();
       if (narr == 0) {
	   std::cerr << "No arrivals in the list" << std::endl;
	   return 0;
       }
       if (narr != ngather) {
	   if (ngather != 0) {
	       clear();
	       std::cerr << "Clearing the gather" << std::endl;
	   }
	   resize(narr);
       }
       auto a = arrs.arr;
       int i;
       for (i = 0; i < narr; ++i) {
	   obsxyz[0] = a[i].X;
	   obsxyz[1] = a[i].Y;
	   obsxyz[2] = a[i].Z;
	   ph = a[i].phase;
	   two_point_td(srcxyz, obsxyz, ph, td, takeoffangle);
	   
	   ttcal[i] = td[0]; // travel time
	   dtdx[i]  = td[1]; // dt/dx
	   dtdy[i]  = td[2]; // dt/dy
	   dtdz[i]  = td[3]; // dt/dz
	   toa[i]   = takeoffangle;   // take-off angle
	   
	   trec[i] = a[i].trec;  // time recorded
	   
	   if (td[0] > 0.0) {
	   // weight
	       wt[i] = a[i].wt;
	   } else {
	       wt[i] = 0.0;
	   }
       }
       return i;
   }


//   int Gather::get_topt(double(& x)[4]) {

   int Gather::get_topt(double* x) {

        //collect({x[0], x[1], x[2]});

        double numerator = 0.0;
        double denominator = 0.0;
	int j;
        for (j = 0; j < ngather; j++) {
            numerator += (trec[j] - ttcal[j]) * wt[j];
            denominator += wt[j];
        }
        if (denominator != 0.0) {
            x[3] = numerator / denominator;
        } else {
            x[3] = 0.0;
        }
	return j;
    }

//    void Gather::get_res_w(const double(& x)[4], double* fvec, double* w) {
//        collect({x[0], x[1], x[2]});
      
//    int Gather::get_res_w(const double(& x)[4], int ldjac, double* fvec, double* w) {

    int Gather::get_res_w(const double* x, int ldjac, double* fvec, double* w) {
	int m = std::min(ldjac, ngather);
	int j;
        for (j = 0; j < m; j++) {
            fvec[j] = trec[j] - x[3] - ttcal[j];
            w[j] = wt[j];
        }
	return j;
    }

//    void Gather::get_G_w(const double (& x)[4], double fjac[][100], double* w) {
//        collect({x[0], x[1], x[2]});
        
    int Gather::get_G_w(int ldjac, double* fjac, double* w) {
	int m = std::min(ldjac, ngather);
	int j;
        for (j = 0; j < m; j++) {
            fjac[j+0*ldjac] = -dtdx[j];
            fjac[j+1*ldjac] = -dtdy[j];
            fjac[j+2*ldjac] = -dtdz[j];
            fjac[j+3*ldjac] = -1.0;
            w[j] = wt[j];
        }
	return j;
    }

Gather gather;

// method to write gather data for debugging on console
void Gather::print() {
    std::cout << "# ObsX ObsY ObsZ Phase Trec Tcal dtdx dtdy dtdz Toa Wt" << std::endl;
    auto a = arrs.arr;
    for (int i = 0; i < ngather; ++i) {
	std::cout << a[i].X << " " << a[i].Y << " " << a[i].Z << " "
	    << a[i].phase << " "
	    << trec[i] << " "
	    << ttcal[i] << " "
	    << dtdx[i] << " "
	    << dtdy[i] << " "
	    << dtdz[i] << " "
	    << toa[i] << " "
	    << wt[i] << std::endl;
    }
}

