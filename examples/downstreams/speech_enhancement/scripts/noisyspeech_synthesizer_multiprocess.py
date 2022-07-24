import copy
import glob
from scipy.io import wavfile
from pathlib import Path
import pandas as pd
import dns_utils as utils
from .audiolib import audioread, audiowrite, segmental_snr_mixer, activitydetector, is_clipped
from scipy import signal
import numpy as np
import librosa
import random
import time
from random import shuffle
import configparser as CP
import argparse
import os
import sys
import threading

# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../", "../")))


MAXTRIES = 50
MAXFILELEN = 100

np.random.seed(5)
random.seed(55)


def build_audio(is_clean, params, index, audio_samples_length=-1):
    '''Construct an audio signal from source files'''

    fs_output = params['fs']
    silence_length = params['silence_length']
    if audio_samples_length == -1:
        audio_samples_length = int(params['audio_length'] * params['fs'])

    output_audio = np.zeros(0)
    remaining_length = audio_samples_length
    files_used = []
    clipped_files = []

    if is_clean:
        source_files = params['cleanfilenames']
        idx = index
    else:
        if 'noisefilenames' in params.keys():
            source_files = params['noisefilenames']
            idx = index
        # if noise files are organized into individual subdirectories, pick a directory randomly
        else:
            noisedirs = params['noisedirs']
            # pick a noise category randomly
            idx_n_dir = np.random.randint(0, np.size(noisedirs))
            source_files = glob.glob(os.path.join(noisedirs[idx_n_dir],
                                                  params['audioformat']))
            shuffle(source_files)
            # pick a noise source file index randomly
            idx = np.random.randint(0, np.size(source_files))

    # initialize silence
    silence = np.zeros(int(fs_output * silence_length))

    # iterate through multiple clips until we have a long enough signal
    tries_left = MAXTRIES
    while remaining_length > 0 and tries_left > 0:

        # read next audio file and resample if necessary

        idx = (idx + 1) % np.size(source_files)
        input_audio, fs_input = audioread(source_files[idx])
        if input_audio is None:
            sys.stderr.write("WARNING: Cannot read file: %s\n" %
                             source_files[idx])
            continue
        if fs_input != fs_output:
            input_audio = librosa.resample(input_audio, fs_input, fs_output)

        # if current file is longer than remaining desired length, and this is
        # noise generation or this is training set, subsample it randomly
        if len(input_audio) > remaining_length:
            idx_seg = np.random.randint(0, len(input_audio) - remaining_length)
            input_audio = input_audio[idx_seg:idx_seg + remaining_length]

        # check for clipping, and if found move onto next file
        if is_clipped(input_audio):
            clipped_files.append(source_files[idx])
            tries_left -= 1
            continue

        # concatenate current input audio to output audio stream
        files_used.append(source_files[idx])
        output_audio = np.append(output_audio, input_audio)
        remaining_length -= len(input_audio)

        # add some silence if we have not reached desired audio length
        if remaining_length > 0:
            silence_len = min(remaining_length, len(silence))
            output_audio = np.append(output_audio, silence[:silence_len])
            remaining_length -= silence_len

    if tries_left == 0 and not is_clean and 'noisedirs' in params.keys():
        print("There are not enough non-clipped files in the " + noisedirs[idx_n_dir] +
              " directory to complete the audio build")
        return [], [], clipped_files, idx

    return output_audio, files_used, clipped_files, idx


def gen_audio(is_clean, params, index, audio_samples_length=-1):
    '''Calls build_audio() to get an audio signal, and verify that it meets the
       activity threshold'''

    clipped_files = []
    low_activity_files = []
    if audio_samples_length == -1:
        audio_samples_length = int(params['audio_length'] * params['fs'])
    if is_clean:
        activity_threshold = params['clean_activity_threshold']
    else:
        activity_threshold = params['noise_activity_threshold']

    while True:
        audio, source_files, new_clipped_files, index = \
            build_audio(is_clean, params, index, audio_samples_length)

        clipped_files += new_clipped_files
        if len(audio) < audio_samples_length:
            continue

        if activity_threshold == 0.0:
            break

        percactive = activitydetector(audio=audio)
        if percactive > activity_threshold:
            break
        else:
            low_activity_files += source_files

    return audio, source_files, clipped_files, low_activity_files, index


class main_body:
    def __init__(self, thread_num, cfg, cleanfilenames, noisefilenames):
        '''Main body of this file'''
        self.thread_num = thread_num
        params = dict()
        params['cfg'] = cfg

        params['fs'] = int(cfg['sampling_rate'])
        params['audioformat'] = cfg['audioformat']
        params['audio_length'] = float(cfg['audio_length'])
        params['silence_length'] = float(cfg['silence_length'])
        params['total_hours'] = float(cfg['total_hours'])

        params['clean_activity_threshold'] = float(cfg['clean_activity_threshold'])
        params['noise_activity_threshold'] = float(cfg['noise_activity_threshold'])
        params['snr_lower'] = int(cfg['snr_lower'])
        params['snr_upper'] = int(cfg['snr_upper'])

        params['continue_snr'] = utils.str2bool(cfg['continue_snr'])
        params['snr_step'] = int(cfg['snr_step'])
        params['target_level_lower'] = int(cfg['target_level_lower'])
        params['target_level_upper'] = int(cfg['target_level_upper'])

        params['noisyspeech_dir'] = utils.get_dir(cfg, 'noisy_destination', 'noisy')
        params['clean_proc_dir'] = utils.get_dir(cfg, 'clean_destination', 'clean')
        params['noise_proc_dir'] = utils.get_dir(cfg, 'noise_destination', 'noise')

        params['num_files'] = int((params['total_hours'] * 60 * 60) / params['audio_length'])
        params['fileindex_start'] = 0
        params['fileindex_end'] = params['num_files']

        # params["valid_percent"] = cfg['valid_percent']

        # cleanfilenames = []
        # # 获取了Clean_dir下面的所有wav，前面根据设置会默认是read数据
        # for path in Path(clean_dir).rglob('*.wav'):
        #     cleanfilenames.append(str(path.resolve()))
        # shuffle(cleanfilenames)  # 打乱
        params['cleanfilenames'] = cleanfilenames
        params['num_cleanfiles'] = len(params['cleanfilenames'])

        # noisefilenames = []
        # for path in Path(noise_dir).rglob('*.wav'):
        #     noisefilenames.append(str(path.resolve()))
        # shuffle(noisefilenames)
        params['noisefilenames'] = noisefilenames
        self.params = params
        self.thread_flags = np.zeros(self.thread_num)

    def main_gen(self, flag):
        params = copy.copy(self.params)
        all_num = params["fileindex_end"]
        section = all_num // self.thread_num
        params["fileindex_start"] = section * flag
        params["fileindex_end"] = section * (flag + 1)
        self.thread_flags[flag] = 0
        file_num = params['fileindex_start']

        while file_num < params['fileindex_end']:
            clean_index = random.randint(0, len(params['cleanfilenames']) - flag) + flag
            clean, clean_sf, clean_cf, clean_laf, clean_index = gen_audio(True, params, clean_index)

            # generate noise
            noise_index = random.randint(0, len(params['noisefilenames']) - flag) + flag
            noise, noise_sf, noise_cf, noise_laf, noise_index = gen_audio(False, params, noise_index, len(clean))

            if params["continue_snr"]:
                snr = np.random.randint(params['snr_lower'], params['snr_upper'])
            else:
                snrs = np.arange(start=params['snr_lower'], stop=params['snr_upper']+1, step=params["snr_step"])
                snr = snrs[np.random.randint(0, len(snrs))]

            clean_snr, noise_snr, noisy_snr, target_level = segmental_snr_mixer(params=params,
                                                                                clean=clean,
                                                                                noise=noise,
                                                                                snr=snr)
            # unexpected clipping
            if is_clipped(clean_snr) or is_clipped(noise_snr) or is_clipped(noisy_snr):
                print("Warning: File #" + str(file_num) + " has unexpected clipping, " +
                      "returning without writing audio to disk")
                continue

            # write resultant audio streams to files
            hyphen = '-'
            clean_source_filenamesonly = [i[:-4].split(os.path.sep)[-1] for i in clean_sf]
            clean_files_joined = hyphen.join(clean_source_filenamesonly)[:MAXFILELEN]
            noise_source_filenamesonly = [i[:-4].split(os.path.sep)[-1] for i in noise_sf]
            noise_files_joined = hyphen.join(noise_source_filenamesonly)[:MAXFILELEN]

            noisyfilename = clean_files_joined + '_' + noise_files_joined + '_snr' + \
                str(snr) + '_tl' + str(target_level) + '_fileid_' + str(file_num) + '.wav'
            cleanfilename = 'clean_fileid_' + str(file_num) + '.wav'
            noisefilename = 'noise_fileid_' + str(file_num) + '.wav'

            noisypath = os.path.join(params['noisyspeech_dir'], noisyfilename)
            cleanpath = os.path.join(params['clean_proc_dir'], cleanfilename)
            noisepath = os.path.join(params['noise_proc_dir'], noisefilename)

            audio_signals = [noisy_snr, clean_snr, noise_snr]
            file_paths = [noisypath, cleanpath, noisepath]

            file_num += 1
            for i in range(len(audio_signals)):
                try:
                    audiowrite(file_paths[i], audio_signals[i], params['fs'])
                except Exception as e:
                    print(str(e))
        self.thread_flags[flag] = 1

    def monitor(self, flag):
        if np.sum(self.thread_flags) != self.thread_num:
            time.sleep(5)
        else:
            print("data make ok")
            utils.preprocess_dns(in_dir=os.path.join(self.params['noisyspeech_dir'], "../"))
            print("tsv is saved at %s" % os.path.join(self.params['noisyspeech_dir'], "../"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='noisyspeech_synthesizer_singleprocess.cfg',
                        help='Read noisyspeech_synthesizer.cfg for all the details')
    args = parser.parse_args()
    cfgpath = os.path.join(os.path.dirname(__file__), args.cfg)
    cfg = CP.ConfigParser()
    cfg._interpolation = CP.ExtendedInterpolation()
    cfg.read(cfgpath)

    clean_home = cfg._sections["common"]["speech_dir"]
    noise_home = cfg._sections["common"]["noise_dir"]
    train_cfg = cfg._sections["train_speech"]
    valid_cfg = cfg._sections["valid_speech"]
    test_cfg = cfg._sections["test_speech"]

    cleanfilenames = []
    for path in Path(clean_home).rglob('*.wav'):
        cleanfilenames.append(str(path.resolve()))
    shuffle(cleanfilenames)  # 打乱
    valid_percent = valid_cfg["total_hours"] / (train_cfg["total_hours"] + valid_cfg["total_hours"] + test_cfg["total_hours"])
    test_percent = test_cfg["total_hours"] / (train_cfg["total_hours"] + valid_cfg["total_hours"] + test_cfg["total_hours"])
    train_percent = train_cfg["total_hours"] / (train_cfg["total_hours"] + valid_cfg["total_hours"] + test_cfg["total_hours"])
    valid_cleans = cleanfilenames[:int(valid_percent * len(cleanfilenames))]
    test_cleans = cleanfilenames[
        int(valid_percent * len(cleanfilenames)):int(valid_percent * len(cleanfilenames))+int(test_percent * len(cleanfilenames))]
    train_cleans = cleanfilenames[-int(train_percent * len(cleanfilenames)):]

    train_noises = []
    for path in Path(os.path.join(noise_home, "tr")).rglob('*.wav'):
        train_noises.append(str(path.resolve()))
    shuffle(train_noises)  # 打乱
    valid_noises = []
    for path in Path(os.path.join(noise_home, "cv")).rglob('*.wav'):
        valid_noises.append(str(path.resolve()))
    shuffle(valid_noises)  # 打乱
    test_noises = []
    for path in Path(os.path.join(noise_home, "tt")).rglob('*.wav'):
        test_noises.append(str(path.resolve()))
    shuffle(test_noises)  # 打乱

    threads = []
    # train 300h set
    train_cfg_300h = copy.copy(train_cfg)
    train_cfg_300h["total_hours"] = 300
    train_cfg_300h["noisy_destination"] = train_cfg_300h["noisy_destination"].replace("train_h", "train_300h")
    train_cfg_300h["clean_destination"] = train_cfg_300h["clean_destination"].replace("train_h", "train_300h")
    train_cfg_300h["noise_destination"] = train_cfg_300h["noise_destination"].replace("train_h", "train_300h")
    train_cfg_300h["log_dir"] = train_cfg_300h["log_dir"].replace("train_h", "train_300h")
    runner_train300 = main_body(thread_num=5, cfg=train_cfg_300h, cleanfilenames=train_cleans, noisefilenames=train_noises)
    threads.append(threading.Thread(target=runner_train300.monitor, args=(0,)))
    for thread_index in range(runner_train300.thread_num):
        threads.append(threading.Thread(target=runner_train300.main_gen, args=(thread_index,)))
    # train 100h set
    train_cfg_100h = copy.copy(train_cfg)
    train_cfg_100h["total_hours"] = 100
    train_cfg_100h["noisy_destination"] = train_cfg_100h["noisy_destination"].replace("train_h", "train_100h")
    train_cfg_100h["clean_destination"] = train_cfg_100h["clean_destination"].replace("train_h", "train_100h")
    train_cfg_100h["noise_destination"] = train_cfg_100h["noise_destination"].replace("train_h", "train_100h")
    train_cfg_100h["log_dir"] = train_cfg_100h["log_dir"].replace("train_h", "train_100h")
    runner_train100 = main_body(thread_num=2, cfg=train_cfg_100h, cleanfilenames=train_cleans, noisefilenames=train_noises)
    threads.append(threading.Thread(target=runner_train100.monitor, args=(0,)))
    for thread_index in range(runner_train100.thread_num):
        threads.append(threading.Thread(target=runner_train100.main_gen, args=(thread_index,)))
    # train 10h set
    train_cfg_10h = copy.copy(train_cfg)
    train_cfg_10h["total_hours"] = 10
    train_cfg_10h["noisy_destination"] = train_cfg_10h["noisy_destination"].replace("train_h", "train_10h")
    train_cfg_10h["clean_destination"] = train_cfg_10h["clean_destination"].replace("train_h", "train_10h")
    train_cfg_10h["noise_destination"] = train_cfg_10h["noise_destination"].replace("train_h", "train_10h")
    train_cfg_10h["log_dir"] = train_cfg_10h["log_dir"].replace("train_h", "train_10h")
    runner_train10 = main_body(thread_num=1, cfg=train_cfg_10h, cleanfilenames=train_cleans, noisefilenames=train_noises)
    threads.append(threading.Thread(target=runner_train10.monitor, args=(0,)))
    for thread_index in range(runner_train10.thread_num):
        threads.append(threading.Thread(target=runner_train10.main_gen, args=(thread_index,)))
    # train 1h set
    train_cfg_1h = copy.copy(train_cfg)
    train_cfg_1h["total_hours"] = 1
    train_cfg_1h["noisy_destination"] = train_cfg_1h["noisy_destination"].replace("train_h", "train_1h")
    train_cfg_1h["clean_destination"] = train_cfg_1h["clean_destination"].replace("train_h", "train_1h")
    train_cfg_1h["noise_destination"] = train_cfg_1h["noise_destination"].replace("train_h", "train_1h")
    train_cfg_1h["log_dir"] = train_cfg_1h["log_dir"].replace("train_h", "train_1h")
    runner_train1 = main_body(thread_num=1, cfg=train_cfg_1h, cleanfilenames=train_cleans, noisefilenames=train_noises)
    threads.append(threading.Thread(target=runner_train1.monitor, args=(0,)))
    for thread_index in range(runner_train1.thread_num):
        threads.append(threading.Thread(target=runner_train1.main_gen, args=(thread_index,)))
    # train 10m set
    train_cfg_10m = copy.copy(train_cfg)
    train_cfg_10m["total_hours"] = 1/6
    train_cfg_10m["noisy_destination"] = train_cfg_10m["noisy_destination"].replace("train_h", "train_10m")
    train_cfg_10m["clean_destination"] = train_cfg_10m["clean_destination"].replace("train_h", "train_10m")
    train_cfg_10m["noise_destination"] = train_cfg_10m["noise_destination"].replace("train_h", "train_10m")
    train_cfg_10m["log_dir"] = train_cfg_10m["log_dir"].replace("train_h", "train_10m")
    runner_train10m = main_body(thread_num=1, cfg=train_cfg_10m, cleanfilenames=train_cleans, noisefilenames=train_noises)
    threads.append(threading.Thread(target=runner_train10m.monitor, args=(0,)))
    for thread_index in range(runner_train10m.thread_num):
        threads.append(threading.Thread(target=runner_train10m.main_gen, args=(thread_index,)))
    # valid set
    runner_valid = main_body(thread_num=1, cfg=valid_cfg, cleanfilenames=valid_cleans, noisefilenames=valid_noises)
    threads.append(threading.Thread(target=runner_valid.monitor, args=(0,)))
    for thread_index in range(runner_valid.thread_num):
        threads.append(threading.Thread(target=runner_valid.main_gen, args=(thread_index,)))
    # test set
    runner_test = main_body(thread_num=1, cfg=test_cfg, cleanfilenames=test_cleans, noisefilenames=test_noises)
    threads.append(threading.Thread(target=runner_test.monitor, args=(0,)))
    for thread_index in range(runner_test.thread_num):
        threads.append(threading.Thread(target=runner_test.main_gen, args=(thread_index,)))

    for thread in threads:
        thread.setDaemon(True)
        thread.start()
    for thread in threads:
        thread.join()
