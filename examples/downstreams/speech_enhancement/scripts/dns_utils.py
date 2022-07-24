# coding: utf-8
# Author：WangTianRui
# Date ：2021/11/23 19:55
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 10:28:41 2019
@author: rocheng
"""
import os
import csv
from shutil import copyfile
import glob
from tqdm import tqdm


def get_dir(cfg, param_name, new_dir_name):
    '''Helper function to retrieve directory name if it exists,
       create it if it doesn't exist'''

    if param_name in cfg:
        dir_name = cfg[param_name]
    else:
        dir_name = os.path.join(os.path.dirname(__file__), new_dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name


def write_log_file(log_dir, log_filename, data):
    '''Helper function to write log file'''
    data = zip(*data)
    with open(os.path.join(log_dir, log_filename), mode='w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=' ',
                               quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in data:
            csvwriter.writerow([row])


def str2bool(string):
    return string.lower() in ("yes", "true", "t", "1")


def rename_copyfile(src_path, dest_dir, prefix='', ext='*.wav'):
    srcfiles = glob.glob(f"{src_path}/" + ext)
    for i in range(len(srcfiles)):
        dest_path = os.path.join(dest_dir, prefix + '_' + os.path.basename(srcfiles[i]))
        copyfile(srcfiles[i], dest_path)


def get_filename(file):
    if str(file).find('clean/') != -1:
        return "clean/" + str(file).split('clean/')[-1]
    if str(file).find('noisy/') != -1:
        return "noisy/" + str(file).split('noisy/')[-1]

    if str(file).find('clean\\') != -1:
        return "clean/" + str(file).split('clean\\')[-1]
    if str(file).find('noisy\\') != -1:
        return "noisy/" + str(file).split('noisy\\')[-1]
    print("file error:", file)
    return file


def preprocess_dns(in_dir, valid_percent=0.2):
    """Create json file from dataset folder.

    Args:
        in_dir (str): Location of the DNS data
        out_dir (str): Where to save the json files.
    """
    # Get all file ids
    clean_wavs = glob.glob(os.path.join(in_dir, "clean/*.wav"))
    clean_dic = make_wav_id_dict(clean_wavs)

    mix_wavs = glob.glob(os.path.join(in_dir, "noisy/*.wav"))
    mix_dic = make_wav_id_dict(mix_wavs)

    assert clean_dic.keys() == mix_dic.keys()
    with open(os.path.join(in_dir, "train.tsv"), "w") as train_f:
        with open(os.path.join(in_dir, "valid.tsv"), "w") as valid_f:
            for index, k in enumerate(tqdm(clean_dic.keys())):
                line_str = get_filename(mix_dic[k])+"\t"+get_filename(clean_dic[k]+"\n")
                if index % (int(1 / valid_percent)) == 0:
                    valid_f.write(line_str)
                else:
                    train_f.write(line_str)


def make_wav_id_dict(file_list):
    """
    Args:
        file_list(List[str]): List of DNS challenge filenames.

    Returns:
        dict: Look like {file_id: filename, ...}
    """
    return {get_file_id(fp): fp for fp in file_list}


def get_file_id(fp):
    """ Split string to get wave id in DNS challenge dataset."""
    if os.path.basename(fp)[0] == "_":
        return fp.split("_fileid_")[-1].split("_")[0]
    else:
        return fp.split("_")[-1].split(".")[0]
