# ELA - Earthquake Location Algorithm
A study of earthquake location procedures.

## Overview

ELA (Earthquake Location Algorithm) is a seismic event location program specifically designed for local and near-regional scales. It utilizes a flat-layered earth model to determine the location of seismic events.

## Features

- **Coordinate System**: ELA uses the S-JTSK Křovák coordinate system (EPSG:5513) for both input and output data, with units measured in kilometers.
- **Optimization Method**: The program solves the location problem using a local optimization method based on the classic Geiger method for earthquake location.
- **Ray Tracing and Travel Time Calculation**: ELA employs a modified TRVDRV subroutine, originally written by \[[Eaton 1969](#eaton1969)\], for ray tracing and travel time computations.

- **Weighting**: The weighted least squares algorithm considers weight codes
for phase arrivals measurements.
These weight codes range from full weight (0) to no weight (4),
following the conventions of HYPO71 and HYPOINVERSE.

## Velocity Models

ELA supports independent 1D velocity models for different seismic phases, including: Pg, Pn, Sg, Sn, Lg
It works only with those seismic phases that can be computed in the corresponding velocity model by
the modified TRVDRV subroutine (direct wave, refracted wave, head wave).

## Assumptions and Considerations

- **Flat-Earth Model**: The velocity model assumes that seismic stations are at the earth's surface. Although station elevations are not directly used, the delaying effect of elevation can be mostly accounted for with station delays.
- **Relative Earthquake Depths**: Due to the flat-earth assumption, earthquake depths are calculated relative to the average local surface, as defined by the nearby seismic stations.


## Install prerequisites
The ELA program runs on Linux.


1. BLAS & LAPACK libraries need to be installed.
2. Static library libdatetime.a.
   Download from [[https://github.com/wavebitscientific/datetime-fortran.git]]

## Installation
To install the program, follow these steps:

1. Clone the ELA repository from GitHub:
   ```bash
   git clone https://github.com/ipemu/ela.git
   ```

2. Navigate to the cloned repository directory and compile the program:
   ```bash
   cd ela/libsrc
   make
   cd ../src
   make
   ```

ELA is adapted for the Linux environment and the gfortran compiler.
Compilation is facilitated through the supplied Makefile,
resulting in the binary ''ela'', which can be manually copied to the binary path.

## Usage

The ELA program is executed with the following command-line arguments:
```
> ela --help

Usage: ela [options] input 
Allows options::

Generic options:
  -v [ --version ]                print version string
  -h [ --help ]                   produce help message
  -c [ --config ] arg (=ela.cfg)  name of a file of a configuration.

Command line I/O options:
  -i [ --input ] arg              input hyp file
  -o [ --output ] arg             output hy3 file

Configuration:
  -m [ --model ] arg              input model file
  -s [ --stations ] arg           input list of stations
  --read_err arg (=0)             set error of arrival time observationd [s]
  --model_err arg (=0)            set error of propagation times predictions 
                                  [s]
```

Example command to run ELA:
```bash
ela -i event000.hyp -o event000.hy3 -m modelA.mod
```

Ensure that all file paths are specified correctly either with the full path or relative to the current directory.


## Contributing

If you'd like to contribute to the ELA project, feel free to submit pull requests or open issues on the GitHub repository.


## Literature
<a name="eaton1969"></a>Eaton, J. P. (1969). HYPOLAYR, a computer program for determining hypocenters of local earthquakes in an earth consisting of uniform flat layers over a half space, Open File Report, U.S. Geological Survey, 155 pp.

