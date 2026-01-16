#ifndef GATHER_H
#define GATHER_H

#include <vector>

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

   int collect(const double* x);
   int get_topt(double* x);
   int get_res_w(const double* x, int ldjac, double* fvec, double* w);
   int get_G_w(int ldjac, double* fjac, double* w);
   void print();
};  // class Gather

extern Gather gather;

#endif // GATHER_H
