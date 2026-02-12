import os
import os.path
import sys
import logging
import copy
import time
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import numpy as np


def train(args):
    seed_list = copy.deepcopy(args['seed'])
    device = copy.deepcopy(args['device'])
    # device = device.split(',')

    for seed in seed_list:
        args['seed'] = seed
        args['device'] = device
        _train(args)

    myseed = 42069  # set a random seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)


def _train(args):
    logdir = 'logs/{}/{}_{}_{}/{}/{}'.format(args['dataset'], args['init_cls'], args['increment'], args['net_type'], args['model_name'], args['optim'])

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logfilename = os.path.join(logdir, '{}'.format(args['seed']))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=logfilename + '.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    if not os.path.exists(logfilename):
        os.makedirs(logfilename)
    print(logfilename)
    _set_random(args)
    _set_device(args)
    print_args(args)
    data_manager = DataManager(args['dataset'], args['shuffle'], args['seed'], args['init_cls'], args['increment'], args)
    args['class_order'] = data_manager._class_order
    model = factory.get_model(args['model_name'], args)

    cnn_curve =  {'top1': []}
    cnn_matrix = []
    for task in range(data_manager.nb_tasks):
        logging.info('All params: {}'.format(count_parameters(model._network)))
        logging.info('Trainable params: {}'.format(count_parameters(model._network, True)))
        time_start = time.time()
        if args['eval']:
            checkpoint_path = os.path.join(logfilename, 'task_{}.pth'.format(int(task)))
            if os.path.exists(checkpoint_path):
                print('Loading checkpoint from:', checkpoint_path)
                checkpoint = torch.load(checkpoint_path, map_location=args['device'][0])
                model._network.load_state_dict(checkpoint)
            else:
                print('No checkpoint found at:', checkpoint_path)
                return

        model.incremental_train(data_manager)
        time_end = time.time()
        logging.info('Time:{}'.format(time_end - time_start))
        time_start = time.time()
        cnn_accy = model.eval_task()
        time_end = time.time()
        logging.info('Time:{}'.format(time_end - time_start))
        model.after_task()

        cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
        cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
        cnn_matrix.append(cnn_values)

        cnn_curve["top1"].append(cnn_accy["top1"])

        logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))

        # print('Average Accuracy (CNN):', sum(cnn_curve["top1"]) / len(cnn_curve["top1"]))
        logging.info("Average Accuracy (CNN): {} \n".format(round(sum(cnn_curve["top1"]) / len(cnn_curve["top1"]), 2)))

        # logging.info("ACC: {}".format(cnn_accy["grouped"]))
        # cnn_curve["top1"].append(cnn_accy["top1"])
        # logging.info("ACC top1 curve: {}".format(cnn_curve["top1"]))
        # print('Average Accuracy:', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
        # logging.info("Average Accuracy: {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))

        # if not args['eval']:
        #     torch.save(model._network.state_dict(), os.path.join(logfilename, "task_{}.pth".format(int(task))))
    
    print(f"\n{'=' * 80}")
    print(
        "Finished {}_init{}_inc{}  ".format(
            args["dataset"],
            args["init_cls"],
            args["increment"],
        )
    )
    print("Average Accuracy (Top1): {}".format(round(sum(cnn_curve["top1"]) / len(cnn_curve["top1"]), 2)))
    print("PD: {:.2f}".format(cnn_curve["top1"][0] - cnn_curve["top1"][-1]))

    # 将 cnn_matrix 填充为矩形，空缺位置补0
    max_len = max(len(row) for row in cnn_matrix) if cnn_matrix else 0
    padded_cnn_matrix = [row + [0] * (max_len - len(row)) for row in cnn_matrix]
    print("Backward Transfer (BWT):", backward_transfer(np.array(padded_cnn_matrix)))
    # print("Forward Transfer (FWT):", forward_transfer(args["dataset"], np.array(padded_cnn_matrix)))
    # print("Total Train Time:", round(total_train_time, 2), "s")
    # print("Total Test Time:", round(total_test_time, 2), "s")
    if len(cnn_matrix) > 0:
        np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(cnn_matrix):
            idxy = len(line)
            np_acctable[idxx, :idxy] = np.array(line)
        print("Accuracy Matrix (CNN):")
        print(np_acctable)

    print(f"{'=' * 80}\n")


def _set_device(args):
    device_type = args['device']
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(device))

        gpus.append(device)

    args['device'] = gpus


def _set_random(args):
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])
    torch.cuda.manual_seed_all(args['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info('{}: {}'.format(key, value))

def backward_transfer(matrix):
    """
    Calculate the backward transfer from the accuracy matrix.
    """

    # --- Backward Transfer (BWT) ---
    # 根据公式: BWT = (1/(T-1)) * Σ(i=1 to T-1) (R_T,i - R_i,i)
    # R_T,i 是训练完最后一个任务 T 后，在任务 i 上的准确率 -> 对应矩阵的最后一行
    # R_i,i 是刚训练完任务 i 后的准确率 -> 对应矩阵的对角线

    # 使用 0-based 索引:
    # R_T,i (i=1..T-1) -> matrix[-1, 0:T-1]
    # R_i,i (i=1..T-1) -> np.diag(matrix)[0:T-1]

    last_row_accs = matrix[-1, :-1]  # 任务 1..T-1 在训练完任务 T 后的准确率
    diagonal_accs = np.diag(matrix)[:-1]  # 任务 1..T-1 在刚训练完自己时的准确率

    bwt_diffs = last_row_accs - diagonal_accs
    backward_transfer = np.mean(bwt_diffs)

    # print("--- Backward Transfer (BWT) ---")
    # print(f"各任务在最终的准确率 (R_T,i): {np.round(last_row_accs, 2)}")
    # print(f"各任务在当时的准确率 (R_i,i): {np.round(diagonal_accs, 2)}")
    # print(f"BWT 差值 (R_T,i - R_i,i): {np.round(bwt_diffs, 2)}")
    # print(f"Backward Transfer 平均分: {backward_transfer:.2f}")

    return np.round(backward_transfer, 2)

def forward_transfer(dataset, matrix):
    """
    Calculate the forward transfer from the accuracy matrix.
    """
    if dataset == "cub":
        rand_init_acc = np.array([86.51, 52.53, 65.04, 67.86, 74.62, 57.89, 71.82, 87.97, 68.6, 83.61, 85.37])
    elif dataset == "cifar224":
        rand_init_acc = np.array([77.5, 44.4, 51.8, 30.4, 56.8, 68.4, 44.0, 36.4, 41.4])
    elif dataset == "mini_imagenet":
        rand_init_acc = np.array([94.97, 90.6, 69.2, 81.2, 82.8, 73.4, 70.2, 90.6, 89.8])
    else:
        rand_init_acc = np.array([62.95, 28.08, 51.15, 38.64, 56.76, 50.93, 41.38, 61.25, 49.1, 43.37, 44.44]) # INR

    fwt_accs = np.diag(matrix, k=1)  # 得到 R_0,1, R_1,2, ... (0-based)
    rand_init_acc_for_fwt = rand_init_acc[1:]  # 需要 R_0,1, R_0,2, ... (0-based)

    fwt_diffs = fwt_accs - rand_init_acc_for_fwt
    forward_transfer = np.mean(fwt_diffs)

    # print("--- Forward Transfer (FWT) ---")
    # print(f"新任务在前一任务训练后的准确率 (R_i-1,i): {np.round(fwt_accs, 2)}")
    # print(f"新任务在随机初始化时的准确率 (R_0,i): {np.round(rand_init_acc_for_fwt, 2)}")
    # print(f"FWT 差值 (R_i-1,i - R_0,i): {np.round(fwt_diffs, 2)}")
    # print(f"Forward Transfer 平均分: {forward_transfer:.2f}")

    return np.round(forward_transfer, 2)
