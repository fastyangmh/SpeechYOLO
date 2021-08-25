# import
import timm
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from src.project_parameters import ProjectParameters
from pytorch_lightning import LightningModule
import torch.nn as nn
import pandas as pd
import numpy as np
from src.utils import load_checkpoint, load_yaml
import torch.optim as optim
from os.path import basename, dirname
import torch.nn.functional as F
import random

# def


def _get_backbone_model_from_file(filepath):
    import sys
    sys.path.append('{}'.format(dirname(filepath)))
    class_name = basename(filepath).split('.')[0]
    exec('from {} import {}'.format(*[class_name]*2))
    return eval('{}()'.format(class_name))


def _get_backbone_model(project_parameters):
    if project_parameters.backbone_model in timm.list_models():
        backbone_model = timm.create_model(model_name=project_parameters.backbone_model,
                                           pretrained=True, num_classes=project_parameters.output_features, in_chans=1)
    elif '.py' in project_parameters.backbone_model:
        backbone_model = _get_backbone_model_from_file(
            filepath=project_parameters.backbone_model)
    else:
        assert False, 'please check the backbone model. the backbone model: {}'.format(
            project_parameters.backbone_model)
    return backbone_model


def extract_data(out_tesor, C, B, K):

    out_coords = out_tesor[:, :, :3 * B].contiguous().view(-1, C, B, 3)
    out_xs = out_coords[:, :, :, 0].view(-1, C, B) / float(C)
    out_ws = torch.pow(out_coords[:, :, :, 1].view(-1, C, B), 2)
    out_start = (out_xs - (out_ws * 0.5))
    out_end = (out_xs + (out_ws * 0.5))
    pred_class_prob = out_tesor[:, :, 3 * B:].contiguous().view(-1, C, K)
    pred_class_prob = pred_class_prob.unsqueeze(
        2).repeat(1, 1, B, 1).view(-1, C, B, K)
    pred_conf = out_coords[:, :, :, 2].view(-1, C, B)
    return out_ws, out_start, out_end, pred_conf, pred_class_prob


def eval_actual(yolo_output, target, threshold, cells, boxes, num_classes, gap):

    C = cells
    B = boxes
    K = num_classes

    actual_lens = np.zeros(2)  # num_position_correct, len(target_labels

    pred_labels = convert_yolo_tags(yolo_output, C, B, K, threshold, gap)
    target_labels = convert_yolo_tags(
        target[:, :, :-1], C, B, K, threshold, gap)

    # pdb.set_trace()

    num_position_correct = counter_for_actual_accuracy(
        pred_labels, target_labels)  # find position for eval_actual

    num_classes = num_classes
    acc_per_term = np.zeros((num_classes, 3))  # tp, fp, fn
    f1_per_term = np.zeros(num_classes)
    for pred_key, pred_list in pred_labels.items():  # dict of keys "batch_wordIdx"
        pred_word = int(pred_key.split('_')[1])
        if pred_key in target_labels:

            target_list = target_labels[pred_key]
            len_target = len(target_list)
            len_pred = len(pred_list)
            if len_target == len_pred:
                acc_per_term[pred_word][0] += len_target  # true positive
            if len_target < len_pred:
                acc_per_term[pred_word][1] += len_pred - \
                    len_target  # false positive
                acc_per_term[pred_word][0] += len_target  # true positive
            if len_target > len_pred:
                # not calculating "miss" here
                acc_per_term[pred_word][0] += len_pred

        else:
            acc_per_term[pred_word][1] += 1  # false positive

    count_existance = np.zeros(K)
    exists_counter = 0
    for target_key, target_list in target_labels.items():
        target_word = int(target_key.split('_')[1])
        count_existance[target_word] += len(target_list)
        exists_counter += len(target_list)

    for item in range(len(acc_per_term)):  # false negative == miss
        acc_per_term[item][2] = count_existance[item] - acc_per_term[item][0]

    actual_lens[0], actual_lens[1] = num_position_correct, exists_counter

    return acc_per_term, actual_lens


def convert_yolo_tags(pred, c, b, k, threshold, gap):
    pred_ws, pred_start, pred_end, pred_conf, pred_class_prob = extract_data(
        pred, c, b, k)
    class_max, class_indices = torch.max(pred_class_prob, 3)
    conf_max, box_indices = torch.max((pred_conf * class_max), 2)

    pass_conf = (conf_max >= threshold).float()
    labels = []
    for batch in range(0, pred.size(0)):
        for cell_i in range(0, pred.size(1)):
            if pass_conf[batch, cell_i].item() <= 0:
                continue
            selected_box_index = box_indices[batch, cell_i].item()
            selected_class_index = class_indices[batch, cell_i, 0].item()
            label_start = pred_start[batch, cell_i, selected_box_index].item()
            label_end = pred_end[batch, cell_i, selected_box_index].item()
            x = (label_end + label_start)/2
            w = pred_ws[batch, cell_i, selected_box_index].item()
            labels.append([cell_i, x, w, selected_class_index, batch])

    width_cell = 1. / c  # width per cell
    final_pred_labels = {}

    for label in labels:
        # label[1] was already multiple with width cell
        real_x = (label[0] * width_cell + label[1])
        real_w = label[2]
        cur_start = (real_x - float(real_w) / 2.0)
        cur_end = (real_x + float(real_w) / 2.0)
        cur_class = str(label[4]) + "_" + str(label[3])  # batch_class

        if cur_class not in final_pred_labels:
            final_pred_labels[cur_class] = []

        else:
            prev_start = final_pred_labels[cur_class][-1][0]
            prev_end = final_pred_labels[cur_class][-1][1]
            if cur_start >= prev_end and cur_end >= prev_start:
                # --------
                #          -------
                if cur_end - prev_end <= gap:
                    final_pred_labels[cur_class].pop()  # remove last item
                    cur_start = prev_start
            elif cur_start <= prev_end and prev_start <= cur_end:
                # --------
                #      -------
                final_pred_labels[cur_class].pop()  # remove last item
                cur_start = prev_start
            elif cur_start >= prev_start and cur_end <= prev_end:
                # -----------
                #    ----
                final_pred_labels[cur_class].pop()  # remove last item
                cur_start = prev_start
                cur_end = pred_end
            elif cur_start >= prev_start and cur_end >= pred_end:
                #     -----
                #   ---------
                final_pred_labels[cur_class].pop()  # remove last item

        final_pred_labels[cur_class].append([cur_start, cur_end])
        # print "objet- start:{}, end:{}, class:{}".format(pred_start,pred_end, pred_class)

    return final_pred_labels


def calc_iou(pred, target):

    pred_start, pred_end = pred[0], pred[1]
    target_start, target_end = target[0], target[1]

    intersect_start = max(pred_start, target_start)
    intersect_end = min(pred_end, target_end)
    intersect_w = intersect_end - intersect_start

    if intersect_w < 0:  # no intersection
        intersect_w = 0.0

    pred_len = pred_end - pred_start
    target_len = target_end - target_start

    union = pred_len + target_len - intersect_w
    iou = float(intersect_w) / union
    return iou

# find position for eval_actual


def counter_for_actual_accuracy(pred_labels, target_labels):

    # given list of targets and predictions, find which prediction corresponds to which target.
    iou_choice_counter = 0
    mega_iou_choice = []
    for key, pred_label_list in pred_labels.items():
        if key in target_labels:

            iou_list = []
            target_label_list = target_labels[key]
            for target_idx, target_label in enumerate(target_label_list):

                for pred_idx, pred_label in enumerate(pred_label_list):

                    iou_val = calc_iou(pred_label, target_label)
                    iou_list.append(
                        [iou_val, pred_idx, target_idx, pred_label, target_label])

            list_len = min(len(target_label_list), len(pred_label_list))
            iou_list = sorted(iou_list, key=lambda k: (
                k[0], random.random()), reverse=True)
            iou_choice = []
            while len(iou_list) != 0 and len(iou_choice) < list_len:
                if len(iou_choice) == 0:
                    iou_choice.append(iou_list.pop(0))
                else:
                    # pdb.set_trace()
                    cur_item = iou_list.pop(0)
                    flag = True
                    for item in iou_choice:
                        if cur_item[1] == item[1]:
                            flag = False
                            break
                        if cur_item[2] == item[2]:
                            flag = False
                            break
                    if flag:
                        iou_choice.append(cur_item)

            mega_iou_choice.extend(iou_choice)

    # ============================================================================================

    # for actual accuracy: check if center of prediction is within (start, end) boundaries of target
    for item in mega_iou_choice:
        iou_val, pred_idx, target_idx, pred_label, target_label = item
        pred_start, pred_end = pred_label
        target_start, target_end = target_label

        center_pred = float(pred_end + pred_start) / 2

        if round(center_pred, 2) >= round(target_start, 2) and round(center_pred, 2) <= round(target_end, 2):
            iou_choice_counter += 1

    return iou_choice_counter


def _get_optimizer(model_parameters, project_parameters):
    optimizer_config = load_yaml(
        filepath=project_parameters.optimizer_config_path)
    optimizer_name = list(optimizer_config.keys())[0]
    if optimizer_name in dir(optim):
        for name, value in optimizer_config.items():
            if value is None:
                optimizer = eval('optim.{}(params=model_parameters, lr={})'.format(
                    optimizer_name, project_parameters.lr))
            elif type(value) is dict:
                value = ('{},'*len(value)).format(*['{}={}'.format(a, b)
                                                    for a, b in value.items()])
                optimizer = eval('optim.{}(params=model_parameters, lr={}, {})'.format(
                    optimizer_name, project_parameters.lr, value))
            else:
                assert False, '{}: {}'.format(name, value)
        return optimizer
    else:
        assert False, 'please check the optimizer. the optimizer config: {}'.format(
            optimizer_config)


def _get_lr_scheduler(project_parameters, optimizer):
    if project_parameters.lr_scheduler == 'StepLR':
        lr_scheduler = StepLR(optimizer=optimizer,
                              step_size=project_parameters.step_size, gamma=project_parameters.gamma)
    elif project_parameters.lr_scheduler == 'CosineAnnealingLR':
        lr_scheduler = CosineAnnealingLR(
            optimizer=optimizer, T_max=project_parameters.step_size)
    else:
        assert False, 'please check the lr scheduler. the lr scheduler: {}'.format(
            project_parameters.lr_scheduler)
    return lr_scheduler


def create_model(project_parameters):
    model = Net(project_parameters=project_parameters)
    if project_parameters.checkpoint_path is not None:
        model = load_checkpoint(model=model, num_classes=project_parameters.num_classes,
                                use_cuda=project_parameters.use_cuda, checkpoint_path=project_parameters.checkpoint_path)
    return model
# class


class YOLOSpeechLoss:
    def __init__(self, project_parameters, noobject_conf=0.5, obj_conf=1, coordinate=10, class_conf=1, loss_type="mse"):
        self.project_parameters = project_parameters
        self.noobject_conf = noobject_conf
        self.obj_conf = obj_conf
        self.coordinate = coordinate
        self.class_conf = class_conf
        self.loss_type = loss_type

    def _make_flatt(self, table):
        return table.view(table.size(0), -1)

    def __call__(self, y_pred, y_true):
        # target
        target_coords = y_true[:, :, :3 * self.project_parameters.boxes].contiguous(
        ).view(-1, self.project_parameters.cells, self.project_parameters.boxes, 3)
        target_xs = target_coords[:, :, :, 0].view(
            -1, self.project_parameters.cells, self.project_parameters.boxes, 1)
        target_xs_no_norm = target_coords[:, :, :, 0].view(
            -1, self.project_parameters.cells, self.project_parameters.boxes, 1) / float(self.project_parameters.cells)
        target_ws = torch.pow(target_coords[:, :, :, 1].view(-1, self.project_parameters.cells,
                                                             self.project_parameters.boxes, 1), 2)  # assuming the prediction is for sqrt(w)
        target_conf = target_coords[:, :, :, 2].view(
            -1, self.project_parameters.cells, self.project_parameters.boxes, 1)
        target_start = target_xs_no_norm - (target_ws * 0.5)
        target_end = target_xs_no_norm + (target_ws * 0.5)
        target_class_prob = y_true[:, :, 3 * self.project_parameters.boxes:-1].contiguous(
        ).view(-1, self.project_parameters.cells, self.project_parameters.num_classes, 1)

        # pred
        # get all the x,w values for all the boxes
        pred_coords = y_pred[:, :, :3 * self.project_parameters.boxes].contiguous(
        ).view(-1, self.project_parameters.cells, self.project_parameters.boxes, 3)
        pred_xs = pred_coords[:, :, :, 0].view(
            -1, self.project_parameters.cells, self.project_parameters.boxes, 1)
        pred_xs_no_norm = pred_coords[:, :, :, 0].view(
            -1, self.project_parameters.cells, self.project_parameters.boxes, 1) / float(self.project_parameters.cells)
        # assuming the prediction is for sqrt(w)
        pred_ws = torch.pow(pred_coords[:, :, :, 1].view(
            -1, self.project_parameters.cells, self.project_parameters.boxes, 1), 2)
        pred_conf = pred_coords[:, :, :, 2].view(
            -1, self.project_parameters.cells, self.project_parameters.boxes, 1)
        pred_start = pred_xs_no_norm - (pred_ws * 0.5)
        pred_end = pred_xs_no_norm + (pred_ws * 0.5)
        pred_class_prob = y_pred[:, :, 3 * self.project_parameters.boxes:].contiguous(
        ).view(-1, self.project_parameters.cells, self.project_parameters.num_classes, 1)
        # Calculate the intersection areas
        intersect_start = torch.max(pred_start, target_start)
        intersect_end = torch.min(pred_end, target_end)
        intersect_w = intersect_end - intersect_start

        # Calculate the best IOU, set 0.0 confidence for worse boxes
        iou = intersect_w / (pred_ws + target_ws - intersect_w)
        iou_max_value, iou_max_indices = torch.max(iou, 2)
        best_box = torch.eq(iou, iou_max_value.unsqueeze(2))
        one_confs_per_cell = best_box.float() * target_conf

        # the last place in y_true determines if the object exists
        real_exist = (y_true[:, :, -1]).unsqueeze(2)
        obj_exists_classes = real_exist.repeat(
            (1, 1, self.project_parameters.num_classes)).view(-1, self.project_parameters.cells, self.project_parameters.num_classes, 1)
        obj_exists = one_confs_per_cell
        noobj_exists = torch.zeros([obj_exists.size(
            0), self.project_parameters.cells, self.project_parameters.boxes, 1], dtype=torch.float32)
        noobj_exists = torch.eq(one_confs_per_cell, noobj_exists).float()

        if self.loss_type == "abs":
            first_part = torch.sum(self._make_flatt(
                self.coordinate * obj_exists * torch.abs((pred_xs - target_xs))), 1)
            second_part = torch.sum(self._make_flatt(
                5 * self.coordinate * obj_exists * torch.abs((pred_ws - target_ws))), 1)
        else:
            first_part = torch.sum(self._make_flatt(
                self.coordinate * obj_exists * torch.pow((pred_xs - target_xs), 2)), 1)
            second_part = torch.sum(self._make_flatt(
                self.coordinate * obj_exists * torch.pow((pred_ws - target_ws), 2)), 1)

        third_part = torch.sum(self._make_flatt(
            self.obj_conf * obj_exists * torch.pow((pred_conf - one_confs_per_cell), 2)), 1)
        fourth_part = torch.sum(
            self._make_flatt(self.noobject_conf * noobj_exists * torch.pow((pred_conf - one_confs_per_cell), 2)), 1)
        fifth_part = torch.sum(
            self._make_flatt(self.class_conf * obj_exists_classes * torch.pow((target_class_prob - pred_class_prob), 2)), 1)

        total_loss = first_part + second_part + third_part + fourth_part + fifth_part

        return torch.mean(total_loss), torch.mean(first_part), torch.mean(second_part), torch.mean(third_part), \
            torch.mean(fourth_part), torch.mean(fifth_part)


class YOLOSpeechAccuracy:
    def __init__(self, project_parameters) -> None:
        self.project_parameters = project_parameters

    def __call__(self, y_pred, y_true):
        total_actual_lens = np.zeros(2)
        acc_per_term, actual_lens = eval_actual(yolo_output=y_pred, target=y_true, threshold=self.project_parameters.confidence,
                                                cells=self.project_parameters.cells, boxes=self.project_parameters.boxes, num_classes=self.project_parameters.num_classes, gap=self.project_parameters.gap)
        total_actual_lens += actual_lens
        return float(total_actual_lens[0])/total_actual_lens[1]


class Net(LightningModule):
    def __init__(self, project_parameters):
        super().__init__()
        self.project_parameters = project_parameters
        self.backbone_model = _get_backbone_model(
            project_parameters=project_parameters)
        self.loss_function = YOLOSpeechLoss(
            project_parameters=project_parameters)
        self.accuracy = YOLOSpeechAccuracy(
            project_parameters=project_parameters)

    def training_forward(self, x):
        return self.backbone_model(x)

    def forward(self, x):
        pass

    def get_progress_bar_dict(self):
        # don't show the loss value
        items = super().get_progress_bar_dict()
        items.pop('loss', None)
        return items

    def _parse_outputs(self, outputs):
        epoch_loss = []
        epoch_accuracy = []
        for step in outputs:
            epoch_loss.append(step['loss'].item())
            epoch_accuracy.append(step['accuracy'].item())
        return epoch_loss, epoch_accuracy

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.training_forward(x)
        loss = self.loss_function(y_hat, y)
        train_step_accuracy = self.accuracy(y_hat, y)
        return {'loss': loss, 'accuracy': train_step_accuracy}

    def training_epoch_end(self, outputs) -> None:
        epoch_loss, epoch_accuracy = self._parse_outputs(
            outputs=outputs, calculate_confusion_matrix=False)
        self.log('training loss', np.mean(epoch_loss),
                 on_epoch=True, prog_bar=True)
        self.log('training accuracy', np.mean(epoch_accuracy))

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.training_forward(x)
        loss = self.loss_function(y_hat, y)
        val_step_accuracy = self.accuracy(y_hat, y)
        return {'loss': loss, 'accuracy': val_step_accuracy}

    def validation_epoch_end(self, outputs) -> None:
        epoch_loss, epoch_accuracy = self._parse_outputs(
            outputs=outputs, calculate_confusion_matrix=False)
        self.log('validation loss', np.mean(epoch_loss),
                 on_epoch=True, prog_bar=True)
        self.log('validation accuracy', np.mean(epoch_accuracy))

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.training_forward(x)
        loss = self.loss_function(y_hat, y)
        test_step_accuracy = self.accuracy(y_hat, y)
        return {'loss': loss, 'accuracy': test_step_accuracy}

    def test_epoch_end(self, outputs) -> None:
        epoch_loss, epoch_accuracy, confmat = self._parse_outputs(
            outputs=outputs, calculate_confusion_matrix=True)
        self.log('test loss', np.mean(epoch_loss))
        self.log('test accuracy', np.mean(epoch_accuracy))

    def configure_optimizers(self):
        optimizer = _get_optimizer(model_parameters=self.parameters(
        ), project_parameters=self.project_parameters)
        if self.project_parameters.step_size > 0:
            lr_scheduler = _get_lr_scheduler(
                project_parameters=self.project_parameters, optimizer=optimizer)
            return [optimizer], [lr_scheduler]
        else:
            return optimizer


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # create model
    model = create_model(project_parameters=project_parameters)

    # display model information
    model.summarize()

    # create input data
    x = torch.ones(project_parameters.batch_size, 1, 224, 224)

    # get model output
    y = model.forward(x)

    # display the dimension of input and output
    print(x.shape)
    print(y.shape)
