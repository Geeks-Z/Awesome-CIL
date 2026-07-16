import logging
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base import BaseLearner
from utils.inc_net import OurNet
from utils.toolkit import tensor2numpy


num_workers = 8


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]


def compute_orthogonality_loss(previous_weights_list, current_weights, epsilon=1e-8):
    total_ortho_loss = 0.0
    current_norm = torch.norm(current_weights.flatten())
    current_normalized = current_weights.flatten() / (current_norm + epsilon)

    for prev_weights in previous_weights_list:
        prev_norm = torch.norm(prev_weights.flatten())
        prev_normalized = prev_weights.flatten() / (prev_norm + epsilon)
        dot_product = torch.abs(torch.sum(prev_normalized * current_normalized))
        total_ortho_loss += dot_product

    if len(previous_weights_list) > 0:
        total_ortho_loss /= len(previous_weights_list)

    return total_ortho_loss


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = OurNet(args, True)

        self.args = args
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.init_cls = args["init_cls"]
        self.inc = args["increment"]

        self.use_exemplars = args["use_old_data"]
        self.use_init_ptm = args["use_init_ptm"]
        self.use_diagonal = args["use_diagonal"]

        self.recalc_sim = args["recalc_sim"]
        self.alpha = args["alpha"]
        self.beta = args["beta"]

        self.moni_adam = args["moni_adam"]
        self.adapter_num = args["adapter_num"]

        if self.moni_adam:
            self.use_init_ptm = True
            self.alpha = 1
            self.beta = 1

    def after_task(self):
        self._known_classes = self._total_classes
        self._network.freeze()
        self._network.backbone.add_adapter_to_list()

    def get_cls_range(self, task_id):
        if task_id == 0:
            start_cls = 0
            end_cls = self.init_cls
        else:
            start_cls = self.init_cls + (task_id - 1) * self.inc
            end_cls = start_cls + self.inc

        return start_cls, end_cls

    def replace_fc(self, train_loader):
        model = self._network.eval()

        with torch.no_grad():
            start_idx = -1 if self.use_init_ptm else 0

            for index in range(start_idx, self._cur_task + 1):
                if self.moni_adam and index > self.adapter_num - 1:
                    break
                if self.use_diagonal and index != -1 and index != self._cur_task:
                    continue

                embedding_list, label_list = [], []
                for _, batch in enumerate(train_loader):
                    (_, data, label) = batch
                    data = data.to(self._device)
                    label = label.to(self._device)
                    embedding = model.backbone.forward_proto(data, adapt_index=index)
                    embedding_list.append(embedding.cpu())
                    label_list.append(label.cpu())

                embedding_list = torch.cat(embedding_list, dim=0)
                label_list = torch.cat(label_list, dim=0)

                class_list = np.unique(self.train_dataset_for_protonet.labels)
                for class_index in class_list:
                    data_index = (label_list == class_index).nonzero().squeeze(-1)
                    embedding = embedding_list[data_index]
                    proto = embedding.mean(0)
                    if self.use_init_ptm:
                        model.fc.weight.data[class_index, (index + 1) * self._network.out_dim:(index + 2) * self._network.out_dim] = proto
                    else:
                        model.fc.weight.data[class_index, index * self._network.out_dim:(index + 1) * self._network.out_dim] = proto

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)

        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        self.data_manager = data_manager
        self.train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="train"
        )
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        self.test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        self.train_dataset_for_protonet = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="test"
        )
        self.train_loader_for_protonet = DataLoader(
            self.train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self._network.add_fc()
        self.replace_fc(self.train_loader_for_protonet)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        if self._cur_task == 0 or self.init_cls == self.inc:
            optimizer = self.get_optimizer(lr=self.args["init_lr"])
            scheduler = self.get_scheduler(optimizer, self.args["init_epochs"])
        else:
            if "later_lr" not in self.args or self.args["later_lr"] == 0:
                self.args["later_lr"] = self.args["init_lr"]
            if "later_epochs" not in self.args or self.args["later_epochs"] == 0:
                self.args["later_epochs"] = self.args["init_epochs"]

            optimizer = self.get_optimizer(lr=self.args["later_lr"])
            scheduler = self.get_scheduler(optimizer, self.args["later_epochs"])

        self._init_train(train_loader, test_loader, optimizer, scheduler)

    def get_optimizer(self, lr):
        params = filter(lambda p: p.requires_grad, self._network.parameters())
        if self.args["optimizer"] == "sgd":
            return optim.SGD(params, momentum=0.9, lr=lr, weight_decay=self.weight_decay)
        if self.args["optimizer"] == "adam":
            return optim.Adam(params, lr=lr, weight_decay=self.weight_decay)
        if self.args["optimizer"] == "adamw":
            return optim.AdamW(params, lr=lr, weight_decay=self.weight_decay)
        raise ValueError("Unsupported optimizer: {}".format(self.args["optimizer"]))

    def get_scheduler(self, optimizer, epoch):
        if self.args["scheduler"] == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epoch, eta_min=self.min_lr)
        if self.args["scheduler"] == "steplr":
            return optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=self.args["init_milestones"],
                gamma=self.args["init_lr_decay"],
            )
        if self.args["scheduler"] == "constant":
            return None
        raise ValueError("Unsupported scheduler: {}".format(self.args["scheduler"]))

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        if self.moni_adam and self._cur_task > self.adapter_num - 1:
            return

        epochs = self.args["init_epochs"] if self._cur_task == 0 or self.init_cls == self.inc else self.args["later_epochs"]
        prog_bar = tqdm(range(epochs))

        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0

            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                aux_targets = torch.where(targets - self._known_classes >= 0, targets - self._known_classes, -1)
                output = self._network(inputs, test=False)
                logits = output["logits"]

                loss = F.cross_entropy(logits, aux_targets)
                optimizer.zero_grad()

                if self._cur_task > 0:
                    kd_ratio = 5.0
                    temperature = 2

                    out_new, out_teacher = self._network.forward_kd(inputs, self._cur_task)
                    out_new_logits = out_new["logits"]
                    out_teacher_logits = out_teacher["logits"]
                    loss_kd = kd_ratio * _KD_loss(out_new_logits, out_teacher_logits, T=temperature)

                    loss_kd.backward()

                    for j in range(len(self._network.backbone.general_pos)):
                        pos = self._network.backbone.adapt_pos.index(self._network.backbone.general_pos[j])
                        for jj in range(len(self._network.backbone.msa)):
                            if self._network.backbone.msa[jj] == 1:
                                temp_weights = 1.0 * torch.norm(
                                    self._network.backbone.old_adapter_list[self._cur_task - 1][pos][jj].lora_A.weight,
                                    dim=1,
                                )
                                temp_weights = 1.0 * len(temp_weights) * temp_weights / torch.sum(temp_weights)
                                self._network.backbone.cur_adapter[pos][jj].lora_A.weight.grad = (
                                    temp_weights.unsqueeze(1) * self._network.backbone.cur_adapter[pos][jj].lora_A.weight.grad
                                )

                if self._cur_task > 0:
                    orth_loss_specific = compute_orthogonality_loss(
                        self._network.backbone.block_weight_list,
                        self._network.backbone.block_weight,
                    )
                    loss += 0.0001 * orth_loss_specific

                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(aux_targets.expand_as(preds)).cpu().sum()
                total += len(aux_targets)

            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                epochs,
                losses / len(train_loader),
                train_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model.forward(inputs, test=True)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        task_correct, task_acc, total = 0, 0, 0

        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)

            with torch.no_grad():
                outputs = self._network.forward(inputs, test=True)["logits"]
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

            task_ids = (targets - self.init_cls) // self.inc + 1
            task_logits = torch.zeros(outputs.shape).to(self._device)
            for i, task_id in enumerate(task_ids):
                if task_id == 0:
                    start_cls = 0
                    end_cls = self.init_cls
                else:
                    start_cls = self.init_cls + (task_id - 1) * self.inc
                    end_cls = self.init_cls + task_id * self.inc
                task_logits[i, start_cls:end_cls] += outputs[i, start_cls:end_cls]

            pred_task_ids = (torch.max(outputs, dim=1)[1] - self.init_cls) // self.inc + 1
            task_correct += (pred_task_ids.cpu() == task_ids).sum()

            pred_task_y = torch.max(task_logits, dim=1)[1]
            task_acc += (pred_task_y.cpu() == targets).sum()
            total += len(targets)

        logging.info("Task correct: {}".format(tensor2numpy(task_correct) * 100 / total))
        logging.info("Task acc: {}".format(tensor2numpy(task_acc) * 100 / total))

        return np.concatenate(y_pred), np.concatenate(y_true)