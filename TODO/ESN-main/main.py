import os
import copy
import numpy as np
import time
import argparse
import wandb

import torch
from torch.utils.data import DataLoader

from networks import incremental_vitood
from trainer import training, eval
from utils.data_manager import DataManager
from pytorch_lightning.loggers import WandbLogger


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--method', default='ESN', type=str, help='str for comment')
    parser.add_argument('--model_name', default='vit', type=str, help='str for comment')
    parser.add_argument('--dataset', default='imageneta', type=str, help='cifar100_vit, 5datasets_vit, core50')
    parser.add_argument('--init_cls', default=10, type=int, help='str for comment')
    parser.add_argument('--inc_cls', default=10, type=int, help='str for comment')
    parser.add_argument('--shuffle', action='store_false', help='false is l2p, which is not shuffle')
    parser.add_argument('--random_seed', default=1993, type=int, help='str for comment')
    parser.add_argument('--training_device', default="1", type=str, help='str for comment')
    parser.add_argument('--max_epochs', default=30, type=int, help='str for comment')
    parser.add_argument('--lr', default=0.01, type=float, help='Set learning rate')

    parser.add_argument('--using_prompt', default=True, type=bool, help='str for comment')
    parser.add_argument('--anchor_energy', default=-10, type=float, help='str for comment')
    parser.add_argument('--energy_beta', default=1, type=float, help='str for comment')
    parser.add_argument('--lamda', default=0.1, type=float, help='0 means do not use energy alignment')
    parser.add_argument('--temptures', default=20, type=int, help='max temperature')
    parser.add_argument('--voting', default=True, type=bool, help='wither or not to voting')

    parser.add_argument('--dil', default=False, type=bool, help='For domain incremental learning evaluation')
    parser.add_argument('--max_cls', default=2, type=int, help='For domain incremental learning evaluation')
    parser.add_argument('--notes', default='', type=str, help='str for comment')

    return parser

batch_size = 48
num_workers = 16
def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    args = setup_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.training_device
    _set_random()
    args.localtime = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())

    if args.dataset == "5datasets_vit" or args.dataset == "core50":
        args.shuffle = False
    data_manager = DataManager(args.dataset, args.shuffle, args.random_seed, args.init_cls, args.inc_cls,
                               args=vars(args))
    args.class_order = data_manager._class_order

    # 初始化 WandbLogger，并传递所有参数
    wandb_logger = WandbLogger(
        project="ESN",
        name=f"{args.method}_{args.model_name}_{args.dataset}_{args.init_cls}_{args.inc_cls}_{args.localtime}",
        group=f"{args.dataset}_{args.model_name}",
        save_code=True,
        config=args  # 将 args 作为配置记录到 wandb.config
    )

    # wandb.init(project="ESN",
    #     name='{}_{}_{}_{}_{}_'.format(args.method, args.model_name, args.dataset, args.init_cls, args.inc_cls) + args.localtime,
    #     save_code=True, group='{}_{}'.format(args.dataset, args.model_name), notes=args.notes, config=args)

    all_tabs, all_classifiers, all_tokens, accuracy_log, vitpromptlist = [], [], [], [], []
    vitprompt = None
    _known_classes = 0
    cls2task = dict()  # 类别与任务对应

    for taskid in range(data_manager.nb_tasks):
        print("current task: {}".format(taskid))
        _total_classes = _known_classes + data_manager.get_task_size(taskid)
        current_data = np.arange(_known_classes, _total_classes)
        for i in range(_known_classes, _total_classes):
            cls2task[i] = taskid
        train_dataset = data_manager.get_dataset(current_data, source='train', mode='train')

        if args.dataset == "core50":
            test_dataset = data_manager.get_dataset(np.arange(0, data_manager.get_task_size(0)), source='test',
                                                    mode='test')
        else:
            test_dataset = data_manager.get_dataset(current_data, source='test', mode='test')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True,
                                  pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        model, eval_result = training(args, incremental_vitood, taskid, train_loader, test_loader,
                                      known_classes=_known_classes, vitprompt=vitprompt,
                                      numclass=data_manager.get_task_size(taskid), total_classes=_total_classes,logger = wandb_logger)

        all_tabs.append(copy.deepcopy(model.tabs).cpu())
        all_classifiers.append(copy.deepcopy(model.classifiers).cpu())
        all_tokens.append(copy.deepcopy(model.task_tokens).cpu())
        vitprompt = copy.deepcopy(model.vitprompt).cpu()
        vitpromptlist.append(copy.deepcopy(model.vitprompt).cpu())

        del model

        _known_classes = _total_classes

    assembles = {'all_tabs': all_tabs, 'all_classifiers': all_classifiers, 'all_tokens': all_tokens,
                 'vitpromptlist': vitpromptlist}
    torch.save(assembles, './checkpoints/' + wandb.run.name+'.pth')

    eval(args, './checkpoints/' + wandb.run.name+'.pth', data_manager, cls2task)



if __name__ == '__main__':
    main()
