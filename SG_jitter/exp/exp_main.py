#coding=utf-8
from data_provider.data_factory import data_provider
from .exp_basic import Exp_Basic
from models import Transformer, Informer, Autoformer,iTransformer,PatchTST,TimeMixer,TimesNet,DLinear
from ns_models import ns_Transformer, ns_Informer, ns_Autoformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import csv


import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')


def write_csv(filename,data):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # 遍历数组的每个 (5, 1) 形状的切片，并将其展平后写入 CSV
        for slice_5x1 in data:
            # 展平 (5, 1) 形状的切片为一维数组
            flattened_slice = slice_5x1.flatten()

            # 写入 CSV 文件
            writer.writerow(flattened_slice)

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'TimesNet': TimesNet,
            'DLinear': DLinear,
            'PatchTST': PatchTST,
            'iTransformer': iTransformer,
            'TimeMixer': TimeMixer,
            'Transformer': Transformer,
            'Informer': Informer,
            'Autoformer': Autoformer,
            'ns_Transformer': ns_Transformer,
            'ns_Informer': ns_Informer,
            'ns_Autoformer': ns_Autoformer,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        ### ysn：通过flag {'train': 0, 'val': 1, 'test': 2}获得对应的数据集
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    ### ysn：验证部分
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val') ### ysn：val
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        '''### ysn：
        早停法是一种被广泛使用解决过拟合的方法，在很多案例上都比正则化的方法要好。
        基本含义是在训练中计算模型在验证集上的表现，当模型在验证集上的表现开始下降的时候，停止训练，这样就能避免继续训练导致过拟合的问题。
        其主要步骤如下：
            1. 将原始的训练数据集划分成训练集和验证集
            2. 只在训练集上进行训练，并每个一个周期计算模型在验证集上的误差，例如，每15次epoch（mini batch训练中的一个周期）
            3. 当模型在验证集上的误差比上一次训练结果差的时候停止训练
            4. 使用上一次迭代结果中的参数作为模型的最终参数
        '''
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        ### ysn：use automatic mixed precision training 默认False
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        ### ysn：训练
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                ### ysn：训练构建解码器的输入
                # 创建了一个与batch_y的最后一个self.args.pred_len维度相同的新张量，但是所有的值都是0。这个张量将用作解码器预测部分的占位符
                '''假设batch_y是一个三维数组（或称为张量），其形状为(a, b, c)(batch  label_len+pred_len, data_dim(4))
                 batch_y[:, -self.args.pred_len:, :]
                第一个维度（形状为a）取所有元素。
                第二个维度（形状为b）取最后pred_len的切片。
                第三个维度（形状为c）取所有元素。  （batch,1,4）
                '''
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                ### ysn：从`batch_y`中取出前`self.args.label_len`个时间步的标签
                # 将上述取出的标签与先前创建的零张量`dec_inp`在维度1（通常是序列长度维度）上进行拼接。
                # 这样得到了一个张量，其前半部分是真实的标签，后半部分是零（表示解码器需要预测的部分）(batch  label_len(真实值)+pred_len（全是0）, data_dim(4))
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                ### ysn：默认False
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    ### ysn：whether to output attention in encoder
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)  ### ysn：dec_out[:, -self.pred_len:, :]  # [B, L, D]

                    f_dim = -1 if self.args.features == 'MS' else 0  ### ysn：!!!!MS 和M 任务重要区分部分 由于data_x和data_y都是features+target ，所以是-1获得target
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)  #### ysn： 验证集
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)  #### ysn：早停法
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)  #### ysn： 动态学习率

        best_model_path = path + '/' + 'best_model.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'best_model.pth')))

        preds = []
        trues = []
        all_pred_data,all_true_data=[],[]  # 用于储存所有预测的features数据，便于进行反归一化
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                ### ysn:
                temp = outputs[:, -self.args.pred_len:, 0:]

                y = batch_y[:, -self.args.pred_len:, 0:].to(self.device)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                ### ysn:
                all_pred_data.append(temp.detach().cpu().numpy())
                all_true_data.append(y.detach().cpu().numpy())

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        all_pred_data=np.array(all_pred_data)
        all_true_data=np.array(all_true_data)

        print('test shape:', preds.shape, trues.shape,all_pred_data.shape,all_true_data.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        all_pred_data = all_pred_data.reshape(-1, all_pred_data.shape[-2], all_pred_data.shape[-1])
        all_true_data = all_true_data.reshape(-1, all_true_data.shape[-2], all_true_data.shape[-1])
        print('test shape:', preds.shape, trues.shape, all_pred_data.shape, all_true_data.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('rmse:{}, mse:{}, mae:{}, mape:{}'.format(rmse,mse, mae,mape))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('rmse:{}, mse:{}, mae:{}, mape:{}'.format(rmse,mse, mae,mape))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        ### ysn；将未反归一化jitter的数据写入文件
        true_file = "./CSV/"+self.args.model+"_Scale_jitter_true_lookback" + str(self.args.seq_len) + "_horizon" + str(
            self.args.pred_len) + ".csv"
        pred_file = "./CSV/"+self.args.model+"_Scale_jitter_pred_lookback" + str(self.args.seq_len) + "_horizon" + str(
            self.args.pred_len) + ".csv"
        write_csv(true_file,trues)
        write_csv(pred_file, preds)

        print("--------------------------反归一化------------------------------")
        data_reshaped = all_true_data.reshape(-1, all_true_data.shape[-1])
        new_trues = test_data.inverse_transform(data_reshaped)
        new_trues = new_trues.reshape(-1, all_true_data.shape[-2], all_true_data.shape[-1])
        print("new_trues.shape", new_trues.shape)
        jitter_trues = new_trues[:,:,-1:]
        print("jitter_trues.shape", jitter_trues.shape)

        if self.args.model=='Transformer' or self.args.model=='Informer' or self.args.model=='Autoformer':
            ##填充，由于无ns机制的模型产生的输出是[,,,1]  需要变成[,,,4] all_true_data.shape
            all_pred_data = all_pred_data * np.ones((1, 1, 1, all_true_data.shape[-1]))
            print('expanded array',all_pred_data)
        data_reshaped = all_pred_data.reshape(-1, all_pred_data.shape[-1])
        new_preds = test_data.inverse_transform(data_reshaped)
        new_preds = new_preds.reshape(-1, all_pred_data.shape[-2], all_pred_data.shape[-1])
        print("new_preds.shape", new_preds.shape)
        new_preds = new_preds[:,:,-1:]
        print("new_preds.shape", new_preds.shape)


        ### ysn；将反归一化的数据写入文件
        new_true_file = "./CSV/"+self.args.model+"_Inverse_jitter_true_lookback" + str(self.args.seq_len) + "_horizon" + str(
            self.args.pred_len) + ".csv"
        new_pred_file = "./CSV/"+self.args.model+"_Inverse_jitter_pred_lookback" + str(self.args.seq_len) + "_horizon" + str(
            self.args.pred_len) + ".csv"
        write_csv(new_true_file, jitter_trues)
        write_csv(new_pred_file, new_preds)

        print("-----------------------------------------反归一化后的指标---------------------------------------------")
        mae, mse, rmse, mape, mspe = metric(new_preds, jitter_trues)
        print('rmse:{}, mse:{}, mae:{}, mape:{}'.format(rmse, mse, mae, mape))
        f = open("result_inverse.txt", 'a')
        f.write(setting + "  \n")
        f.write('rmse:{}, mse:{}, mae:{}, mape:{}'.format(rmse, mse, mae, mape))
        f.write('\n')
        f.write('\n')
        f.close()


        print("--------------------------反归一化以及F10.7（含单位）------------------------------")
        smooth_trues = new_trues[:, :, 0:1]
        print("smooth_trues.shape", smooth_trues.shape)
        ### Smooth
        new_trues=jitter_trues+smooth_trues
        new_preds=new_preds+smooth_trues
        print("smooth_trues",smooth_trues)

        ### ysn；将反归一化的数据写入文件
        new_true_file = "./CSV/"+self.args.model+"_Inverse_F107_true_lookback" + str(self.args.seq_len) + "_horizon" + str(
            self.args.pred_len) + ".csv"
        new_pred_file = "./CSV/"+self.args.model+"_Inverse_F107_pred_lookback" + str(self.args.seq_len) + "_horizon" + str(
            self.args.pred_len) + ".csv"
        write_csv(new_true_file, new_trues)
        write_csv(new_pred_file, new_preds)

        print("-----------------------------------------反归一化后的指标（含单位）---------------------------------------------")
        mae, mse, rmse, mape, mspe = metric(new_preds, new_trues)
        print('rmse:{}, mse:{}, mae:{}, mape:{}'.format(rmse, mse, mae, mape))
        f = open("result_inverse.txt", 'a')
        f.write(setting + "  \n")
        f.write('rmse:{}, mse:{}, mae:{}, mape:{}'.format(rmse, mse, mae, mape))
        f.write('\n')
        f.write('\n')
        f.close()



        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'best_model.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return