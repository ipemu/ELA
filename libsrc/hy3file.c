#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "hy3file.h"

#include "version.h"

struct date_time
  {
    int year;
    int month;
    int day;
    int hour;
    int minute;
    int second;
  };

void hy3save(struct hy3_file * hy3, FILE *fout) {

   if (!fout) fout = stdout;
   fprintf(fout,  "program       : %s", getBuildInfo());
   fprintf(fout, "\nmodel         : %s", hy3->model);
   fprintf(fout, "\nmodel error   :%6.3f s", hy3->model_error);
   fprintf(fout, "\nreading error :%6.3f s", hy3->reading_error);
   
   time_t create_time= time(NULL);
   struct tm *tm = localtime(&create_time);
   fprintf(fout, "\ncreate time   : %02d-%02d-%02d %02d:%02d:%02d",
		   tm->tm_year%100, tm->tm_mon+1, tm->tm_mday,
		   tm->tm_hour, tm->tm_min, tm->tm_sec);
   fprintf(fout, "\nevent         : %s", hy3->event);
   fprintf(fout, "\nstart(x,y,z,t): (%6.2f,%6.2f,%6.2f,%6.2f)", hy3->start_x, hy3->start_y,
		   hy3->start_z, hy3->start_t);
   // print flags for fixed coordinates
   fprintf(fout, "\nfixed         : ("); 
   if (hy3->fix[0] == 1) fprintf(fout, "fix X,"); else fprintf(fout, "      ,");
   if (hy3->fix[1] == 1) fprintf(fout, "fix Y,"); else fprintf(fout, "      ,");
   if (hy3->fix[2] == 1) fprintf(fout, "fix Z,"); else fprintf(fout, "      ,");
   if (hy3->fix[3] == 1) fprintf(fout, "fix T)"); else fprintf(fout, "      )");
   fprintf(fout, "\n"); 
   fprintf(fout, "\nreference time: %02d-%02d-%02d %02d:%02d",
		   hy3->ref_time.tm_year%100, hy3->ref_time.tm_mon+1, hy3->ref_time.tm_mday,
		   hy3->ref_time.tm_hour, hy3->ref_time.tm_min);

   // header for station data
   fprintf(fout, "\n-------------------------------------------------------------------------------");
   fprintf(fout, "\n sta     |obs. t.|cal. t.|res.  |amplitude|freq| w| epi |hypo | azm | ain |xmag");
   fprintf(fout, "\n         |  [s]  |  [s]  | [s]  |  [m/s]  |[Hz]|  |[km] |[km] | [o] | [o] |    ");
   fprintf(fout, "\n-------------------------------------------------------------------------------");
   int irec;
   for (irec=0;irec<hy3->nrec;irec++) {
	   fprintf(fout, "\n%-5s %-3s|% 7.3f|% 7.3f|%6.3f|%9.2e|%4.1f|%2.0f|%5.4g|%5.4g|%5.1f|%5.1f|", 
		   hy3->rec[irec].sta, hy3->rec[irec].ph,
		   hy3->rec[irec].obs_t, hy3->rec[irec].cal_t, hy3->rec[irec].res,
		   hy3->rec[irec].amp, hy3->rec[irec].freq,
		   hy3->rec[irec].w,
		   floor(hy3->rec[irec].epi*1000.0f)/1000.0f,   // truncate to 3 decimal places
		   floor(hy3->rec[irec].hypo*1000.0f)/1000.0f,
		   hy3->rec[irec].azm, hy3->rec[irec].ain);
	   if (hy3->rec[irec].xmag > -9.0) fprintf(fout, "%5.2f", hy3->rec[irec].xmag);
   }

   fprintf(fout, "\n"); 
   fprintf(fout, "\n\nhypocenter data:");
   fprintf(fout, "\n--------------- ");
   time_t origin_et = timelocal(&hy3->ref_time)+ (int) hy3->origin_time[0];
   tm = localtime(&origin_et);
   fprintf(fout, "\norigin time          t:  %02d-%02d-%02d %02d:%02d:%06.3f +-% 7.3f s",
		   tm->tm_year%100, tm->tm_mon+1, tm->tm_mday,
		   tm->tm_hour, tm->tm_min, hy3->origin_time[0], hy3->origin_time[1]);
   fprintf(fout, "\nx-coordinate         x: %8.2f +- %6.2f km     (fi:     %10.6f deg)",
		   hy3->x[0], hy3->x[1], hy3->lat);
   fprintf(fout, "\ny-coordinate         y: %8.2f +- %6.2f km     (lambda: %10.6f deg)",
		   hy3->y[0], hy3->y[1], hy3->lon);
   fprintf(fout, "\ndepth                z:  %5.2f +-%5.2f km", hy3->z[0], hy3->z[1]);
   fprintf(fout, "\nmagnitude           ml:  %5.2f +-%5.2f", hy3->ml[0], hy3->ml[1]);
   fprintf(fout, "\nrms of time residuals :  %5.2f s", hy3->rms);
   fprintf(fout, "\nangular gap           : %6.1f deg", hy3->gap);
   fprintf(fout, "\ninfo                  :  %d", hy3->info);
   fprintf(fout, "\nerror ellipse axis l1 : %5.2f km", hy3->eel1);
   fprintf(fout, "\n              axis l2 : %5.2f km", hy3->eel2);
   fprintf(fout, "\n              theta   : %6.1f deg (to grid)", hy3->theta);
   fprintf(fout, "       (azimuth: %6.1f deg)", hy3->azim);
   fprintf(fout, "\n"); 
}


int hy3load (struct hy3_file *hy3, FILE *fin) {

  char line[85];
  char *variable;
  char *value;
  int irec = 0;

  while (fgets (line, 84, fin) != NULL)
    {				//line

      if ((value = strchr (line, ':')))
	{			// find colon
	  *value++ = '\0';	// replace colon with a null and step value past it
	  variable = line;
// ---------------------------------------------------------       
	  if (strstr (variable, "program"))
	    {
//program       :hypo3d, rev. 2017-04 - v.10.64
	    }
// ---------------------------------------------------------       
	  else if (strstr (variable, "model") && !strstr (variable, "error"))
	    {
//model         :testdata/ete_3d_a.mod
	      char model[40];
	      sscanf (value, "%s ", model);
              strcpy(hy3->model,model);
	    }
// ---------------------------------------------------------       
	  else if (strstr (variable, "model") && strstr (variable, "error"))
	    {
//model error   :0.000 s
	      float model_error = 0.0;
	      sscanf (value, "%f", &model_error);
              hy3->model_error=model_error;
	    }
// ---------------------------------------------------------       
	  else if (strstr (variable, "reading") && strstr (variable, "error"))
	    {
//reading error :0.016 s
	      float reading_error = 0.0;
	      sscanf (value, "%f", &reading_error);
              hy3->reading_error=reading_error;
	    }
// ---------------------------------------------------------
/* 
	  else if (strstr (variable, "create") && strstr (variable, "time"))
	    {
//create time   :17-04-25 20:06:31
	      struct date_time create_dt;
	      sscanf (value, "%d-%d-%d %d:%d:%d", &create_dt.year,
		      &create_dt.month, &create_dt.day, &create_dt.hour,
		      &create_dt.minute, &create_dt.second);
              
	    }
*/
// ---------------------------------------------------------       
	  else if (strstr (variable, "event"))
	    {
//event         :testdata/romatestP.hyp
	      char event[40];
	      sscanf (value, "%s ", event);
              strcpy(hy3->event,event);
	    }
// ---------------------------------------------------------       
	  else if (strstr (variable, "start"))
	    {
//start(x,y,z,t):(   0.00,  15.00, 999.00,  21.00)
	      float start_x, start_y, start_z, start_t;
	      sscanf (value, " (%f,%f,%f,%f)", &start_x, &start_y, &start_z,
		      &start_t);
              hy3->start_x=start_x;
              hy3->start_y=start_y;
              hy3->start_z=start_z;
              hy3->start_t=start_t;
	    }
// ---------------------------------------------------------       
          else if (strstr (variable, "fixed"))
            {
//fixed coordinates:(       ,       , fix Z ,       )
              int fix[4];
              char* sfix;
              int i;
              sfix=strsep(&value,"(");
              for (i=0;i<4;i++) {
                 fix[i]=0;
                 sfix=strsep(&value,",)");
                 if (strspn(sfix," ") < strlen(sfix)) fix[i] = 1;
                 }
              for (i=0;i<4;i++) {
                 hy3->fix[i]=fix[i];
                 }
            }
// ---------------------------------------------------------       
	  else if (strstr (variable, "reference")
		   && strstr (variable, "time"))
	    {
//reference time:17-04-25 08:59
	      struct date_time ref_dt;
	      sscanf (value, "%d-%d-%d %d:%d", &ref_dt.year,
		      &ref_dt.month, &ref_dt.day, &ref_dt.hour,
		      &ref_dt.minute);
	      if (ref_dt.year > 1900) hy3->ref_time.tm_year=ref_dt.year-1900;
	      else if (ref_dt.year > 80) hy3->ref_time.tm_year=ref_dt.year;
	      else hy3->ref_time.tm_year=ref_dt.year+100;
	      hy3->ref_time.tm_mon=ref_dt.month-1;
	      hy3->ref_time.tm_mday=ref_dt.day;
	      hy3->ref_time.tm_hour=ref_dt.hour;
	      hy3->ref_time.tm_min=ref_dt.minute;
	      hy3->ref_time.tm_sec=0;
	      hy3->ref_time.tm_isdst=-1;
	      hy3->ref_time.tm_zone = NULL;
	      hy3->ref_time.tm_gmtoff = 0;
	    }
// ---------------------------------------------------------       
	  else if (strstr (variable, "hypocenter")
		   && strstr (variable, "data"))
	    {
//hypocenter data:
	    }
// ---------------------------------------------------------       
	  else if (strstr (variable, "origin") && strstr (variable, "time"))
	    {
//origin time          t:  17-04-25  08:59:09.305 +- 23.270
	      struct date_time origin_dt;
	      float origin_sec[2];
	      struct tm tm;
	      sscanf (value, "%d-%d-%d %d:%d:%f +-%f", &origin_dt.year,
		      &origin_dt.month, &origin_dt.day, &origin_dt.hour,
		      &origin_dt.minute, &origin_sec[0], &origin_sec[1]);
	      hy3->origin_time[0]=origin_sec[0];
	      hy3->origin_time[1]=origin_sec[1];
	      
	      if (origin_dt.year > 1900) tm.tm_year=origin_dt.year-1900;
	      else if (origin_dt.year > 80) tm.tm_year=origin_dt.year;
	      else tm.tm_year=origin_dt.year+100;
	      tm.tm_mon=origin_dt.month-1;
	      tm.tm_mday=origin_dt.day;
	      tm.tm_hour=origin_dt.hour;
	      tm.tm_min=origin_dt.minute;
	      tm.tm_sec= (int) origin_sec[0];
	      tm.tm_isdst=-1;
	      tm.tm_zone = NULL;
	      tm.tm_gmtoff = 0;
	      time_t origin_et = timelocal(&tm);
	      if ((origin_et - timelocal(&hy3->ref_time)) != (int) hy3->origin_time[0]) {
		fprintf(stderr,"Error: Inconsistency between reference time and origin time.\n");
                fprintf(stderr,"       Reference time: %d-%d-%d %d:%d:%d\n",
			hy3->ref_time.tm_year%100, hy3->ref_time.tm_mon+1,
			hy3->ref_time.tm_mday, hy3->ref_time.tm_hour,
			hy3->ref_time.tm_min, hy3->ref_time.tm_sec);
		fprintf(stderr,"Reference time in seconds: %d\n",
			(int) timelocal(&hy3->ref_time));
		fprintf(stderr,"       Origin time   : %d-%d-%d %d:%d:%f\n",
			origin_dt.year%100, origin_dt.month, origin_dt.day,
			origin_dt.hour, origin_dt.minute, origin_sec[0]);
		fprintf(stderr,"Origin time in seconds: %d\n",
			(int) origin_et);
	      }
	    }
// ---------------------------------------------------------       
	  else if (strstr (variable, "x-coordinate"))
	    {
	      float x, d_x;
	      float lat;
	      if (strstr (value, "fi"))
		{
//x-coordinate         x:  1309.90 +- 155.58    km     (fi: 47.664119 deg)
		  sscanf (value, "%f +-%f%*[^:]:%fdeg", &x, &d_x, &lat);
		}
	      else
		{
//x-coordinate         x:  1309.90 +- 155.58    km
                  lat = NAN;
		  sscanf (value, "%f +-%f", &x, &d_x);
		}
              hy3->x[0]=x;
              hy3->x[1]=d_x;
              hy3->lat=lat;
	    }
// ---------------------------------------------------------       
	  else if (strstr (variable, "y-coordinate"))
	    {
	      float y, d_y;
	      float lon;
	      if (strstr (value, "lambda"))
		{
//y-coordinate         y:   780.35 +- 138.51    km (lambda: 14.410148 deg)
		  sscanf (value, "%f +-%f%*[^:]:%fdeg", &y, &d_y, &lon);
		}
	      else
		{
//y-coordinate         y:   780.35 +- 138.51    km
                  lon = NAN;
		  sscanf (value, "%f +-%f", &y, &d_y);
		}
              hy3->y[0]=y;
              hy3->y[1]=d_y;
              hy3->lon=lon;
	    }
// ---------------------------------------------------------       
	  else if (strstr (variable, "depth"))
	    {
//depth                z:    -1.26 +-   0.00    km
	      float z, d_z;
	      sscanf (value, "%f +-%f", &z, &d_z);
              hy3->z[0]=z;
              hy3->z[1]=d_z;
	    }
// ---------------------------------------------------------       
	  else if (strstr (variable, "magnitude"))
	    {
//magnitude           ml:    -9.90 +-   0.00
	      float ml, d_ml;
	      sscanf (value, "%f +-%f", &ml, &d_ml);
              hy3->ml[0]=ml;
              hy3->ml[1]=d_ml;
	    }
// ---------------------------------------------------------       
	  else if (strstr (variable, "rms"))
	    {
//rms of time residuals :          2.54         s
	      float rms;
	      sscanf (value, "%f", &rms);
              hy3->rms=rms;
	    }
// ---------------------------------------------------------       
	  else if (strstr (variable, "gap"))
	    {
//angular gap           :           320         deg
	      float gap;
	      sscanf (value, "%f", &gap);
              hy3->gap=gap;
	    }
// ---------------------------------------------------------       
	  else if (strstr (variable, "iter"))
	    {
//number of iterations  :            11
	      int iter;
	      sscanf (value, "%d", &iter);
              hy3->info=iter;
	    }
// ---------------------------------------------------------       
	  else if (strstr (variable, "info"))
	    {
//info of iterations  :            11
	      int info;
	      sscanf (value, "%d", &info);
              hy3->info=info;
	    }
// ---------------------------------------------------------       
	  else if (strstr (variable, "axis") && strstr (variable, "l1"))
	    {
//error ellipse axis l1 :        202.51         km
	      float eel1;
	      sscanf (value, "%f", &eel1);
              hy3->eel1=eel1;
	    }
// ---------------------------------------------------------       
	  else if (strstr (variable, "axis") && strstr (variable, "l2"))
	    {
//              axis l2 :         48.78         km
	      float eel2;
	      sscanf (value, "%f", &eel2);
              hy3->eel2=eel2;
	    }
// ---------------------------------------------------------       
	  else if (strstr (variable, "theta"))
	    {
	      float theta=NAN;
	      float azim;
	      if (strstr (value, "azim"))
		{
//              theta   :     41.3 deg (to grid)   (azimuth: 213.4 deg)
		  sscanf (value, "%f%*[^:]:%f", &theta, &azim);
		}
	      else
		{
//              theta   :     213.4 deg
		  sscanf (value, "%f", &azim);
		}
              hy3->theta=theta;
              hy3->azim=azim;
	    }
// ---------------------------------------------------------       


	}			// no colon
      else if ((value = strchr (line, '|')))
	{			// find vertical bar, virgule
	  *value++ = '\0';	// replace colon with a null and step value past it
	  variable = line;
	  if (strstr (value, "obs.") || strstr (value, "amp"))
	    {
// sta  |obs. t.|cal. t.|res. |amplitude|freq|w| epi |hypo |azm|ain|xmag
	      irec = 0;
	    }
	  else if (strstr (value, "[s]"))
	    {
//      |  [s]  |  [s]  | [s] |  [m/s]  |[Hz]| |[km] |[km] |[o]|[o]|    
	      irec = 0;
	    }
	  else
	    {
//KLAU P|  34.57|  34.76|-.195| 0.00E+00|99.9|0|150.5|150.5|185| 41|
//VRCH P|  35.21|  38.81|*****| 0.00E+00|99.9|0|183.1|183.1|186| 33|
              char sta[10];
              char ph[10];

              sscanf(variable,"%s %s",sta,ph);

	      float obs_t;
	      float cal_t;
	      float res;
	      float amp;
	      float freq;
	      float w;
	      float epi;
	      float hypo;
	      float azm;
	      float ain;
	      float xmag;
	      char *sval;
	      sval = strtok (value, "|");
	      obs_t = atof (sval);
	      sval = strtok (NULL, "|");
	      cal_t = atof (sval);
	      sval = strtok (NULL, "|");
	      res = atof (sval);
	      if (strchr (sval, '*'))
		{
		  res = -1;
		  res = NAN;
		}
	      sval = strtok (NULL, "|");
	      amp = atof (sval);
	      sval = strtok (NULL, "|");
	      freq = atof (sval);
	      sval = strtok (NULL, "|");
	      w = atof (sval);
	      sval = strtok (NULL, "|");
	      epi = atof (sval);
	      sval = strtok (NULL, "|");
	      hypo = atof (sval);
	      sval = strtok (NULL, "|");
	      azm = atof (sval);
	      sval = strtok (NULL, "|");
	      ain = atof (sval);
	      sval = strtok (NULL, "\n");
	      if (sval)
		{
		  xmag = atof (sval);
		}
	      else
		{
		  xmag = -9.99;
		};

	      strcpy(hy3->rec[irec].sta,sta);
              strcpy(hy3->rec[irec].ph,ph);
              hy3->rec[irec].obs_t=obs_t;
              hy3->rec[irec].cal_t=cal_t;
              hy3->rec[irec].res=res;
              hy3->rec[irec].amp=amp;
              hy3->rec[irec].freq=freq;
              hy3->rec[irec].w=w;
              hy3->rec[irec].epi=epi;
              hy3->rec[irec].hypo=hypo;
              hy3->rec[irec].azm=azm;
              hy3->rec[irec].ain=ain;
              hy3->rec[irec].xmag=xmag;

	      irec++;
	    }  
	}			// vertical bar

    }				//line
    hy3->nrec=irec;

  return (0);
}
