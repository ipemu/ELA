// Version information
#define VERS "v0.10.6"

#include <stdio.h>
//#include <string.h>
//#include <stdlib.h>
//#include <time.h>

char* getVersion() {
    return (VERS);
}
char* getBuildDate() {
    return __DATE__;
}
char* getBuildTime() {
    return __TIME__;
}
char* getBuildInfo() {
    static char buildInfo[256];
    snprintf(buildInfo, sizeof(buildInfo), "ELA %s %s %s", getVersion(), getBuildDate(), getBuildTime());
    return buildInfo;
}

