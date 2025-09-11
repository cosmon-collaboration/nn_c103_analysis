This readme file was generated on 2025-06-23 by Andre Walker-Loud


GENERAL INFORMATION

1. Title of Dataset: LQCD_NN_C103_S-wave

2. Author Information
	A. Principal Investigator Contact Information
		Name: Andre Walker-Loud
		ORCID: 0000-0002-4686-3667
		Institution: Lawrence Berkeley National Laboratory
		Address: 1 Cyclotron Rd, MS 70R319, Berkeley, CA, 94720, USA
		Email: walkloud@lbl.gov

	B. Associate or Co-investigator Contact Information
		Name: Andrew Hanlon
		ORCID: 0000-0001-8786-8053
		Institution: Kent State University
		Address: Department of Physics, Kent State University, Kent, OH 44242, USA
		Email: ahanlon7@kent.edu

	C. Associate or Co-investigator Contact Information
		Name: Colin Morningstar
		ORCID: 0000-0002-0607-9923
		Institution: Carnegie Mellon University
		Address: Department of Physics, Carnegie Mellon University, Pittsburgh, Pennsylvania 15213, USA
		Email: cmorning@andrew.cmu.edu

	D. Associate or Co-investigator Contact Information
		Name: Amy Nicholson
		ORCID: 0000-0001-7002-0945
		Institution: University of North Carolina, Chapel Hill
		Address: Department of Physics and Astronomy, University of North Carolina, Chapel Hill, NC 27516-3255, USA
		Email: annichol@email.unc.edu
	
3. Date of data collection: 2020-01-01 through 2022-12-31

4. Geographic location of data collection: 
	- Austin, TX, USA (TACC)
	- Berkeley, CA, USA (NERSC)
	- Livermore, CA, USA (LLNL)
	- Oak Ridge, TN, USA (OLCF)

5. Information about funding sources that supported the collection of the data:
	- DOE Office of Science, Office of Nuclear Physics
	- NSF Directorate for Mathematical and Physical Sciences


SHARING/ACCESS INFORMATION

1. Reuse restrictions placed on the data: 
	None.  This data is provided with our publication to support reproducibility and FAIR Principles for scientific research.

2. Links to publications that cite or use the data: 
	https://doi.org/10.48550/arXiv.2505.05547

3. Links to other publicly accessible locations of the data: 
	https://portal.nersc.gov/cfs/m2986/cosmon/nn_c103_2505.05547/

4. Links/relationships to ancillary data sets: 
	N/A

5. Was data derived from another source? If yes, list source(s): 
	N/A

6. Recommended citation for this dataset: 
	@article{BaSc:2025yhy,
    author = "Bulava, John and others",
    collaboration = "BaSc",
    title = "{Di-nucleons do not form bound states at heavy pion mass}",
    eprint = "2505.05547",
    archivePrefix = "arXiv",
    primaryClass = "hep-lat",
    reportNumber = "LLNL-JRNL-2005660",
    month = "5",
    year = "2025"
}

DATA & FILE OVERVIEW

1. File List: 
Two-point correlation function data:
- cosmon_c103_r005-8_nucleon.hdf5
- cosmon_c103_r005-8_dineutron_Swave.hdf5
- cosmon_c103_r005-8_deuteron_Swave.hdf5

The first file is for extracting the single nucleon spectrum and dispersion relation.  The 2nd and third file contain the two-nucleon correlation functions in different cubic irreps in the deuteron and di-neutron channels respectively.

HAL QCD potential data
- c103_n_nblock_15.h5
- c103_nn_pn_TRIP_NEG_PAR_hal_nblock_15.h5
- c103_nn_pn_TRIP_hal_nblock_15.h5
- c103_nn_pp_SING_NEG_PAR_hal_nblock_15.h5
- c103_nn_pp_SING_hal_nblock_15.h5

The first file is for extracting the single nucleon spectrum and dispersion relation with matching creation operator to the HAL QCD NN data.  The 2nd and 3rd file contain momentum-space NN "potential" data for the deuteron channel.  Time-reversal allows for a direct averaging of these.  The 4th and 5th file are the same but for the di-neutron channel (labeled "pp" which is the same in the isospin limit).

Results from analyzing the two-nucleon correlation functions with the sLapH + GEVP methods are saved in the folder
luscher/result

These files can be read with the gvar library, for example, within a python environment
gvar.load('luscher/result/NN_deuteron_tnorm3_t0-td_5-10_N_n3_t_4-20_NN_conspire_e0_t_4-15_ratio_False_block8.pickle')

This file contains results of the posteriors of each irrep fit, stored in a python dictionary with an admittedly painful key structure.  A similar file to the above, ending with "bsPrior-all.pickle_bs" contains the bootstrap resamplings of the fit, which can be used to perform the phase shift analysis.


2. Relationship between files, if important:
The cosmon_c103_r005-8_*hdf5 files contain correlation functions evaluated on each configuration.  The data are stored in an array with the first index being configuration number, such that correlations between the data sets can be determined.  The same holds for the HAL QCD potential data files.

3. Additional related data collected that was not included in the current data package: 
N/A

4. Are there multiple versions of the dataset?
NO

METHODOLOGICAL INFORMATION

1. Description of methods used for collection/generation of data: 
These data files were generated with the methods describe in, or referenced, the arXiv preprint
https://arxiv.org/abs/2505.05547

2. Methods for processing the data: 
These H5 files contain the raw lattice QCD data files generated on various supercomputers, including:
- Summit @ OLCF
- Lassen @ LLNL
- Frontera @ TACC
- Cori @ NERSC

The data files in the folder "luscher/result" were generated with python analysis code available at
https://github.com/cosmon-collaboration/nn_c103_analysis

3. Instrument- or software-specific information needed to interpret the data: 
The two-nucleon correlation functions were generated with the "chroma_laph" software library, available from Colin Morningstar upon request (cmorning@andrew.cmu.edu).

The HAL QCD potential data files were generated with the lalibe software library, in the "feature/mp_nn" branch
https://github.com/callat-qcd/lalibe


4. Standards and calibration information, if appropriate: 
N/A

5. Environmental/experimental conditions: 
N/A

6. Describe any quality-assurance procedures performed on the data: 
Prior to production of the data, the software was run and compared against known results to verify correctness.

7. People involved with sample collection, processing, analysis and/or submission: 
Colin Morningstar
André Walker-Loud
Amy Nicholson
Andrew Hanlon
Sarah Skinner
Ken McElvain
Fernando Romero-López
Joseph Moscoso
Ben Hörz
Henry Monge-Camacho
Christopher Körber
Aaron Meyer
Ermal Rrapaj
Andrea Shindler
Bálint Joó
John Bulava

DATA-SPECIFIC INFORMATION FOR:
The input parameters needed to compute the calculation are all contained in the accompanying publication
https://arxiv.org/abs/2505.05547

and the codes used to generate them (listed above) are also pointed to in the paper.
