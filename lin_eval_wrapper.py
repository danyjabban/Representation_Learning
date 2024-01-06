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
    val_epoch = np.linspace(100, 1000, num=10).astype(int)
    val_bs = np.array([(1 << i) for i in range(8, 13)]).astype(int)
    val_lr = 0.3 * val_bs / 256
    count = 0
    for j in range(len(val_epoch)):
        for i, bs in enumerate(val_bs):
            ret_dict[count] = [val_epoch[j], bs, val_lr[i]]
            count += 1
    return ret_dict

if __name__ == '__main__':
    f_name = 'lin_eval_res.txt'
    num_proc = 4
    futures = []

    parser = argparse.ArgumentParser()
    # base_path == ./saved_models/PyTorchResNet_woDatNormalise/ or 
    # base_path == ./saved_models/wo_data_normalise_simple_resnet/
    parser.add_argument('-n', '--base_path', type=str, required=True)
    args = parser.parse_args()
    path_to_models_dir = args.base_path

    lin_eval_base_path = path_to_models_dir + 'lin_eval/'
    os.makedirs(lin_eval_base_path, exist_ok=True)
    if 'lin_eval_res.txt' not in os.listdir(lin_eval_base_path):
        print('yes')
        f_ptr = open(lin_eval_base_path + f_name, 'w')
        f_ptr.write("batch_size_resnet,epoch_resnet,lr_resnet,embed_dim,score\n")
    else:
        pass  # file exists, the children processes will open the file

    args_dict = gen_inputs()

    pbar = tqdm(total=len(args_dict.keys()), desc='lin_eval_multiproc')
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_proc) as executor:
        for idx, args in args_dict.items():
            futures.append(executor.submit(subprocess.run, [
                                           "python",
                                           "lin_eval.py",
                                           "--fname", str(lin_eval_base_path + f_name),
                                           "--epoch_resnet", str(args[0]),
                                           "--batch_size_resnet", str(args[1]),
                                           "--lr_resnet", str(args[2]),
                                           "--embed_dim", str(128),
                                           "--base_path", str(path_to_models_dir)
                                           ],
                                           capture_output=True, text=True))
        for idx, future in enumerate(as_completed(futures)):
            pbar.update(n=1) 
            try:
                data = future.result().stderr
                print(str(idx) + '\n' + data + '\n')
            except Exception as e:
                print(e)
    wait(futures)