import logging
import os

import numpy as np
import timm
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from backbone.linears import SimpleLinear
from backbone.lora import LoRA_ViT_timm
from models.base import BaseLearner
from utils.toolkit import tensor2numpy


num_workers = 8


def _normalize_backbone_name(name):
    name = name.lower()
    aliases = {
        "pretrained_vit_b16_224": "vit_base_patch16_224",
        "pretrained_vit_b16_224_in21k": "vit_base_patch16_224_in21k",
        "pretrained_vit_large_patch16_224": "vit_large_patch16_224",
        "pretrained_vit_base_patch16_224_dino": "vit_base_patch16_224_dino",
        "pretrained_vit_base_patch16_224_sam": "vit_base_patch16_224_sam",
    }
    name = aliases.get(name, name)
    if name.endswith("_sdlora"):
        name = name[: -len("_sdlora")]
    return aliases.get(name, name)


def _checkpoint_prefix(filepath):
    filepath = filepath or "./sdlora_ckpts/"
    os.makedirs(filepath, exist_ok=True)
    if not filepath.endswith(("/", "\\")):
        filepath = filepath + os.sep
    return filepath


def build_sdlora_backbone(args, index=True, cur_task_index=None):
    backbone_name = _normalize_backbone_name(args["backbone_type"])
    vit_model = timm.create_model(
        backbone_name,
        pretrained=args.get("pretrained", True),
        num_classes=0,
    )
    rank = args.get("rank", args.get("lora_rank", 10))
    model = LoRA_ViT_timm(
        vit_model=vit_model.eval(),
        r=rank,
        num_classes=args.get("lora_num_classes", args["increment"]),
        index=index,
        increment=args["increment"],
        filepath=_checkpoint_prefix(args.get("filepath", "./sdlora_ckpts/")),
        cur_task_index=cur_task_index,
    )
    model.out_dim = getattr(vit_model, "num_features", getattr(vit_model, "embed_dim", args.get("embd_dim", 768)))
    return model


class SDLoraNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = build_sdlora_backbone(args, index=True)
        self.fc = None
        self._device = args["device"][0]
        self.model_type = "vit"

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def extract_vector(self, x):
        return self.backbone(x)

    def save_fc(self, filename, task_id):
        filename = _checkpoint_prefix(filename)
        torch.save(self.fc.weight.detach(), filename + "CLs_weight" + str(task_id) + ".pt")
        torch.save(self.fc.bias.detach(), filename + "CLs_bias" + str(task_id) + ".pt")

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = self.fc.weight.data.clone()
            bias = self.fc.bias.data.clone()
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        return SimpleLinear(in_dim, out_dim)

    def forward(self, x, ortho_loss=False, eval=False):
        if eval:
            out = self.backbone(x, eval=True)
            return out
        if ortho_loss:
            features, loss = self.backbone(x, loss=True)
            out = self.fc(features)
            out.update({"features": features})
            return out, loss

        features = self.backbone(x)
        out = self.fc(features)
        out.update({"features": features})
        return out


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args["filepath"] = _checkpoint_prefix(self.args.get("filepath", "./sdlora_ckpts/"))
        self._network = SDLoraNet(args)

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def update_network(self, index=True):
        return build_sdlora_backbone(
            self.args,
            index=index,
            cur_task_index=self._cur_task,
        )

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=self.args["init_lr"],
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=self.args["init_milestones"],
                gamma=self.args["init_lr_decay"],
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            if len(self._multiple_gpus) > 1:
                self._network = self._network.module
            self._network.backbone = self.update_network(index=False)
            if len(self._multiple_gpus) > 1:
                self._network = nn.DataParallel(self._network, self._multiple_gpus)
            self._network.to(self._device)

            optimizer = optim.SGD(
                self._network.parameters(),
                lr=self.args["lrate"],
                momentum=0.9,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=self.args["milestones"],
                gamma=self.args["lrate_decay"],
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

        save_lora_name = self.args["filepath"]
        if len(self._multiple_gpus) > 1:
            self._network.module.backbone.save_lora_parameters(save_lora_name, self._cur_task)
            self._network.module.save_fc(save_lora_name, self._cur_task)
        else:
            self._network.backbone.save_lora_parameters(save_lora_name, self._cur_task)
            self._network.save_fc(save_lora_name, self._cur_task)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["init_epoch"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits, ortho_loss = self._network(inputs, ortho_loss=True)
                logits = logits["logits"]

                fake_targets = targets - self._known_classes
                loss_clf = F.cross_entropy(logits[:, self._known_classes :], fake_targets)
                loss = loss_clf

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)
