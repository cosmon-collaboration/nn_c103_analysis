import logging
import h5py as h5
import numpy as np

# This sets the root logger to write to stdout (your console).
# Your script/app needs to call this somewhere at least once.
logging.basicConfig(level = logging.INFO)
verbosity = False

def initialize_h5_reader(h5_file_path):
    try:
        h5_file = h5.File(h5_file_path, 'r')
        if verbosity:
            logging.info("H5 file opened successfully")
        return h5_file
    except Exception as e:
        logging.error("Error opening H5 file: {}".format(e))
        return None
    
def print_h5_keys(h5_file_path):
    h5_file = initialize_h5_reader(h5_file_path)
    try:
        for key in h5_file.keys():
            logging.info("Key: {}".format(key))
    except Exception as e:
        logging.error("Error reading keys from H5 file: {}".format(e))

def single_hadron_list(h5_file_path):
        h5_file = initialize_h5_reader(h5_file_path)
        hadrons = []
        try:
            single_hadrons = h5_file['single_hadrons']
            for key in single_hadrons.keys():
                hadrons.append(key)
            return hadrons
        except Exception as e:
            logging.error("Error reading single hadron data: {}".format(e))

def single_hadron_data(h5_file_path, hadron_name):
    h5_file = initialize_h5_reader(h5_file_path)
    try:
        data = h5_file['single_hadrons'][hadron_name][:]
        return data
    except Exception as e:
        logging.error("Error reading single hadron data: {}".format(e))

def read_h5_dataset(h5_file_path, dataset_name):
    h5_file = initialize_h5_reader(h5_file_path)
    try:
        dataset = h5_file[dataset_name]
        logging.info("Dataset {} read successfully".format(dataset_name))
        return dataset
    except Exception as e:
        logging.error("Error reading dataset {}: {}".format(dataset_name, e))
        return None

def h5_dataset_attributes(h5_file_path):
    h5_file = initialize_h5_reader(h5_file_path)
    try:
        keys = []
        scat_channels = []
        for key in h5_file.keys():
            keys.append(key)
        for key in keys:
            if key != 'single_hadrons':
                iso_part = key.split('_')[0]
                if verbosity:
                    logging.info(iso_part)
                strangeness = key.split('_')[1]
                if verbosity:
                    logging.info("Strangeness={}".format(strangeness[-1]))  
                scat_channels.append(key) 
        return scat_channels   
    except Exception as e:
        logging.error("Error reading attributes from H5 dataset: {}".format(e))

def h5_dataset_psq_list(h5_file_path):
    h5_data = initialize_h5_reader(h5_file_path)
    scat_channel = h5_dataset_attributes(h5_data)
    try:
        if len(scat_channel) == 1:
            psq_list = []
            for key in h5_data[scat_channel[0]].keys():
                psq_list.append(key)
            return psq_list
    except Exception as e:
        logging.error("Error reading psq list from H5 dataset: {}".format(e))

def h5_dataset_data(h5_file_path, dataset_name):
    h5_data = initialize_h5_reader(h5_file_path)
    psq_list = h5_dataset_psq_list(h5_file_path)
    try:
        data = []
        for psq in psq_list:
            data.append(h5_data[dataset_name][psq])
        return data
    except Exception as e:
        logging.error("Error reading data from H5 dataset: {}".format(e))

def h5_full_label_printer(h5_file_path):
    h5_file = initialize_h5_reader(h5_file_path)
    scat_channel = h5_dataset_attributes(h5_file_path)
    h5_data = h5_file[scat_channel[0]]
    try:
        for key in h5_data.keys():
            for subkey in h5_data[key].keys():
                #for subsubkey in h5_data[key][subkey].keys():
                logging.info("Key: {}, Subkey: {}".format(key, subkey))
    except Exception as e:
        logging.error("Error reading full label from H5 dataset: {}".format(e))

def energy_data_loader(h5_file_path,level_dict):
    h5_file = initialize_h5_reader(h5_file_path)
    scat_channel = h5_dataset_attributes(h5_file_path)
    h5_data = h5_file[scat_channel[0]]
    #psq_list = h5_dataset_psq_list(h5_file_path)
    energy_dict = {}
    for psq in level_dict.keys():
        energy_dict[psq] = {}
        for irrep in level_dict[psq].keys():
            energy_dict[psq][irrep] = {}
            for level in level_dict[psq][irrep]:
                ecm_label = f'ecm_{level}'
                energy_dict[psq][irrep][ecm_label] = h5_data[psq][irrep][ecm_label][:]
    return energy_dict



#print_h5_keys("Data/fit_spectrum_levels-6Ntmin-Nbin4-SP-6tN-6t0-12tD_B-samplings.hdf5")
#h5_file_path = "Data/fit_spectrum_levels-6Ntmin-Nbin4-SP-6tN-6t0-12tD_B-samplings.hdf5"
#h5_data = read_h5_dataset(h5_file_path, 'isosinglet_S0')
#psq_list = h5_dataset_psq_list(h5_file_path)
#print(h5_data[psq_list[0]]['T1g']['ecm_0'][0])
#h5 = h5.File(h5_file_path, 'r')
#print(h5['single_hadrons']['N(0)'][:])
#print(h5_dataset_attributes(h5_file_path))
#h5_full_label_printer(h5_file_path)
#print(energy_data_loader(h5_file_path, level_dict))
