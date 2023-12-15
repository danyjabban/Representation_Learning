import subprocess
import concurrent 
from concurrent.futures import wait, as_completed
import os
import argparse
from tqdm import tqdm
import re

import numpy as np


def gen_inputs():
    ret_dict = {}
    labelled_perc = np.array([1, 10]).astype(int)
    # labelled_perc = np.array([10]).astype(int)
    prune_perc = np.array([70, 90, 95, 99]).astype(int)
    # prune_perc = np.array([70]).astype(int)
    quantise_bits = np.array([6]).astype(int)
    count = 0
    for i, labelled_p in enumerate(labelled_perc):
        for j, prune_p in enumerate(prune_perc):
            for k, quant in enumerate(quantise_bits):
                ret_dict[count] = [labelled_p, prune_p, quant]
                count += 1
    ret_dict[count] = [1, -1, -1] # 1-percent train data w/o prune and w/o quantise
    ret_dict[count + 1] = [10, -1, -1]  # 10-percent train data w/o prune and w/o quantise
    print(ret_dict)
    return ret_dict


if __name__ == '__main__':
    futures = []

    parser = argparse.ArgumentParser()
    # base_path == ./saved_models/PyTorchResNet_woDatNormalise/ or 
    # base_path == ./saved_models/wo_data_normalise_simple_resnet/
    parser.add_argument('-w', '--wide', type=int, required=True)
    args = parser.parse_args()
    path_to_models_dir_dict = {0: './saved_models/PyTorchResNet_woDatNormalise/', 
                               1: './saved_models/PyTorchResNet_woDatNormalise_wide/'}
    files_in_finetune_dir = os.listdir(path_to_models_dir_dict[args.wide] + 'finetune/')
    if 'tune_idx_p=1.txt' not in files_in_finetune_dir or 'tune_idx_p=10.txt' not in files_in_finetune_dir:
        assert 0 == 1, "tune_idx_p=1.txt or tune_idx_p=10.txt or both not in os.listdir(path_to_models_dir_dict + 'finetune/')"
    labelled_p_to_epoch_dict = {1: 60, 10: 30}
    path_to_models_dir = path_to_models_dir_dict[args.wide]
    if args.wide == 0:
        num_proc = 2
    else:
        num_proc = 1

    args_dict = gen_inputs()

    pbar = tqdm(total=len(args_dict.keys()), desc='finetune_multiproc')
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_proc) as executor:
        for idx, arguments in args_dict.items():
            labelled_p, prune_p, quant_bits = arguments[0], arguments[1], arguments[2]
            args_list = [
                        "python",
                        "finetune_model.py",
                        "--epoch_resnet", str(1000),
                        "--batch_size_resnet", str(2048),
                        "--lr_resnet", str(2.4),
                        "--embed_dim", str(128),
                        "--wide", str(args.wide),
                        "--batch_size_finetune", str(4096),
                        "--labelled_perc", str(labelled_p),
                        "--finetune_ep", str(labelled_p_to_epoch_dict[labelled_p]),
                        "--prune_perc", str(prune_p),
                        # "--quantise_bits", str(quant_bits),
                        "--base_path", str(path_to_models_dir)
                        ]
            if quant_bits != -1:
                args_list.append("--quantise_bits")
                args_list.append(str(quant_bits))
            futures.append(executor.submit(subprocess.run, args_list,
                                           capture_output=True, text=True))
            print(' '.join(args_list))
        for idx, future in enumerate(as_completed(futures)):
            pbar.update(n=1) 
            try:
                data = future.result().stderr
                print(str(idx) + '\n' + data + '\n')
                print(future.result().stdout)
            except Exception as e:
                print(e)
    wait(futures)