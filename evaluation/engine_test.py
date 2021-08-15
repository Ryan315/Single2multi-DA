import os
from pickle import encode_long
import shutil
import time
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchnet as tnt
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
from util import *
import torch.nn.functional as F

from paired_loader import CVDataLoader
from tensorboardX import SummaryWriter

writer_y = SummaryWriter()

import logging

logging.basicConfig(level=logging.INFO,
                    filename='./log/log.txt',
                    filemode='w',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
class Engine(object):
    def __init__(self, state=None):
        if state is None:
            state = {}
        self.state = state
        if self._state('use_gpu') is None:
            self.state['use_gpu'] = torch.cuda.is_available()

        if self._state('image_size') is None:
            self.state['image_size'] = 224

        if self._state('batch_size') is None:
            self.state['batch_size'] = 64

        if self._state('workers') is None:
            self.state['workers'] = 25

        if self._state('device_ids') is None:
            self.state['device_ids'] = None

        if self._state('evaluate') is None:
            self.state['evaluate'] = False

        if self._state('start_epoch') is None:
            self.state['start_epoch'] = 0

        if self._state('max_epochs') is None:
            self.state['max_epochs'] = 90

        if self._state('epoch_step') is None:
            self.state['epoch_step'] = []

        # meters
        self.state['meter_loss1'] = tnt.meter.AverageValueMeter()
        self.state['meter_loss2'] = tnt.meter.AverageValueMeter()
        # time measure
        self.state['batch_time'] = tnt.meter.AverageValueMeter()
        self.state['data_time'] = tnt.meter.AverageValueMeter()
        # display parameters
        if self._state('use_pb') is None:
            self.state['use_pb'] = True
        if self._state('print_freq') is None:
            self.state['print_freq'] = 0

    def _state(self, name):
        if name in self.state:
            return self.state[name]

    def on_start_epoch(self):
        self.state['meter_loss1'].reset()
        self.state['meter_loss2'].reset()
        self.state['batch_time'].reset()
        self.state['data_time'].reset()

    def on_end_epoch(self):
        loss1 = self.state['meter_loss1'].value()[0]
        loss2 = self.state['meter_loss2'].value()[0]

    def on_start_batch(self,):
        pass

    def on_end_batch(self):
        self.state['loss_batch1'] = self.state['loss1'].item()
        self.state['loss_batch2'] = self.state['loss2'].item()
        self.state['meter_loss1'].add(self.state['loss_batch1'])
        self.state['meter_loss2'].add(self.state['loss_batch2'])

    def on_forward_test(self):
        print('call forward function of class Engine')

    def init_learning(self):
        if self._state('train_transform') is None:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            self.state['train_transform'] = transforms.Compose([
                MultiScaleCrop(self.state['image_size'], scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        if self._state('val_transform') is None:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            self.state['val_transform'] = transforms.Compose([
                Warp(self.state['image_size']),
                transforms.ToTensor(),
                normalize,
            ])

        self.state['best_score'] = 0

    def learning(self, nnGCN, train_dataset, val_dataset):

        self.init_learning()

        # define train and val transform
        train_dataset.transform = self.state['train_transform']
        train_dataset.target_transform = self._state('train_target_transform')
        val_dataset.transform = self.state['val_transform']
        val_dataset.target_transform = self._state('val_target_transform')

        train_loader = CVDataLoader()
        train_loader.initialize(train_dataset, val_dataset, self.state['batch_size'], shuffle=True)
        data_loaded = train_loader.load_data()

        test_loader = CVDataLoader()
        test_loader.initialize(train_dataset, val_dataset, self.state['batch_size'], shuffle=True)
        data_loaded_test = test_loader.load_data()

        # optionally resume from a checkpoint
        if self._state('resume') is not None:
            if os.path.isfile(self.state['resume']):
                print("=> loading checkpoint '{}'".format(self.state['resume']))
                checkpoint = torch.load(self.state['resume'])
                self.state['start_epoch'] = checkpoint['epoch']
                # self.state['best_score'] = checkpoint['best_score']
                # G.load_state_dict(checkpoint['state_dict_G'])
                # F1.load_state_dict(checkpoint['state_dict_F1'])
                # F2.load_state_dict(checkpoint['state_dict_F2'])
                nnGCN.load_state_dict(checkpoint['state_dict_GCN'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.state['evaluate'], checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.state['resume']))

        if self.state['use_gpu']:
            # train_loader.pin_memory = True
            # test_loader.pin_memory = True
            cudnn.benchmark = True

            # G = torch.nn.DataParallel(G, device_ids=self.state['device_ids']).cuda()
            # F1 = F1.cuda()
            # F2 = F2.cuda()
            nnGCN = nnGCN.cuda()

        if self.state['evaluate']:
            self.validate(data_loaded_test, nnGCN)
            print("test finished! results saved to 'log/log.txt'")
            return

        # TODO define optimizer

        for epoch in range(self.state['start_epoch'], self.state['max_epochs']):
            self.state['epoch'] = epoch
            # self.train(data_loaded, G, F1, F2, criterion, optimizer_g, optimizer_f1, optimizer_f2, epoch)
            prec1, prec2 = self.validate(data_loaded_test, nnGCN)
            self.state['pred'] = max(prec1, prec2)
            save_every_pth = True

            # if epoch%2==0:
            #     self.save_checkpoint({
            #         'epoch': epoch + 1,
            #         'arch': self._state('arch'),
            #         'state_dict_G': G.module.state_dict(),
            #         'state_dict_F1': F1.state_dict(),
            #         'state_dict_F2': F2.state_dict(),
            #         'state_dict_GCN':nnGCN.state_dict(),
            #         'best_score': self.state['best_score'],
            #     }, save_every_pth)
        return self.state['best_score']

    def validate(self, data_loader, nnGCN):
        self.on_start_epoch()
        end = time.time()
        for i, input in enumerate(data_loader):
            # measure data loading time
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

            self.state['input_source'] = input['S']
            self.state['label_source'] = input['S_label']
            self.state['input_target'] = input['T']
            self.state['label_target'] = input['T_label']

            if i * self.state['batch_size'] > self.state['num_val']:
                break

            self.on_start_batch()

            if self.state['use_gpu']:
                self.state['label_target'] = self.state['label_target'].cuda()
            
            self.on_forward_test(nnGCN, input)

            # writer_y.add_scalar('gcn/loss_gcn', self.state['loss_gcn'],
            #                     global_step=self.state['epoch'] * self.state['num_val'] // self.state['batch_size'] + i)

            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            # measure accuracy
            self.on_end_batch()
           
        score1, score2 = self.on_end_epoch()
        return score1, score2

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if self._state('save_model_path') is not None:
            filename = os.path.join(self.state['save_model_path'], 'epoch_{epoch}_pred_{score:.4f}.pth.tar'.format(epoch=self.state['epoch'], score=self.state['pred']))
            if not os.path.exists(self.state['save_model_path']):
                os.makedirs(self.state['save_model_path'])
        print('save model {filename}'.format(filename=filename))
        torch.save(state, filename)
        if is_best:
            filename_best = 'model_best.pth.tar'
            if self._state('save_model_path') is not None:
                filename_best = os.path.join(self.state['save_model_path'], filename_best)
            shutil.copyfile(filename, filename_best)
            if self._state('save_model_path') is not None:
                if self._state('filename_previous_best') is not None:
                    os.remove(self._state('filename_previous_best'))
                filename_best = os.path.join(self.state['save_model_path'], 'model_best_{score:.4f}.pth.tar'.format(score=state['best_score']))
                shutil.copyfile(filename, filename_best)
                self.state['filename_previous_best'] = filename_best

    def adjust_learning_rate(self, optimizer):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr_list = []
        decay = 0.1 if sum(self.state['epoch'] == np.array(self.state['epoch_step'])) > 0 else 1.0
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay
            lr_list.append(param_group['lr'])
        return np.unique(lr_list)


class MultiLabelMAPEngine(Engine):
    def __init__(self, state):
        Engine.__init__(self, state)
        if self._state('difficult_examples') is None:
            self.state['difficult_examples'] = False
        self.state['ap_meter_s1'] = AveragePrecisionMeter(self.state['difficult_examples'])
        self.state['ap_meter_s2'] = AveragePrecisionMeter(self.state['difficult_examples'])
        self.state['ap_meter_t1'] = AveragePrecisionMeter(self.state['difficult_examples'])
        self.state['ap_meter_t2'] = AveragePrecisionMeter(self.state['difficult_examples'])
        self.state['ap_meter_gcn'] = AveragePrecisionMeter(self.state['difficult_examples'])
        # print('initial ap_meter value is {}{}{}{}'.format(self.state['ap_meter_s1'].value(),self.state['ap_meter_s2'].value(),self.state['ap_meter_t1'].value(),self.state['ap_meter_t2'].value()))

    def on_start_epoch(self):
        Engine.on_start_epoch(self)
        self.state['ap_meter_s1'].reset()
        self.state['ap_meter_s2'].reset()
        self.state['ap_meter_t1'].reset()
        self.state['ap_meter_t2'].reset()
        self.state['ap_meter_gcn'].reset()

    def on_end_epoch(self):
        map_gcn = 100 * self.state['ap_meter_gcn'].value().mean()

        OP_g, OR_g, OF1_g, OF2_g = self.state['ap_meter_gcn'].overall()
        OP_k_g, OR_k_g, OF1_k_g, OF2_k_g = self.state['ap_meter_gcn'].overall_topk(3)
        
        logging.info('***-------testing----------------------------------------------------------------------------')
        logging.info('Epoch: [{0}]\t'.format('testing'))
        logging.info('OP_g: {OP_g:.4f}\t'
                'OR_g: {OR_g:.4f}\t'
                'OF1_g: {OF1_g:.4f}\t'
                'OF2_g: {OF1_g:.4f}\t'.format(OP_g=OP_g, OR_g=OR_g, OF1_g=OF1_g, OF2_g=OF2_g))
        logging.info('OP_3_g: {OP_g:.4f}\t'
                'OR_3_g: {OR_g:.4f}\t'
                'OF1_3_g: {OF1_g:.4f}\t'
                'OF2_3_g: {OF1_g:.4f}\t'.format(OP_g=OP_k_g, OR_g=OR_k_g, OF1_g=OF1_k_g, OF2_g=OF2_k_g))
        print('OP_g: {OP_g:.4f}\t'
                'OR_g: {OR_g:.4f}\t'
                'OF1_g: {OF1_g:.4f}\t'
                'OF2_g: {OF2_g:.4f}\t'.format(OP_g=OP_g, OR_g=OR_g, OF1_g=OF1_g, OF2_g=OF2_g))
        print('OP_3_g: {OP_g:.4f}\t'
                'OR_3_g: {OR_g:.4f}\t'
                'OF1_3_g: {OF1_g:.4f}\t'
                'OF2_3_g: {OF2_g:.4f}\t'.format(OP_g=OP_k_g, OR_g=OR_k_g, OF1_g=OF1_k_g, OF2_g=OF2_k_g))
        print('---------------------------------------------------------------')
        return 1,1

    def on_start_batch(self):
        pass

    def on_end_batch(self):
        self.state['ap_meter_gcn'].add(self.state['output_gcn'].data, self.state['label_target_gt'])

class GCNMultiLabelMAPEngine(MultiLabelMAPEngine):
    def discrepancy(self, out1, out2):
        return torch.mean(torch.abs(torch.sigmoid(out1) - torch.sigmoid(out2)))

    def on_forward_test(self, nnGCN, data_iter):
        batch_size = self.state['batch_size']

        data1 = data_iter['S'][0]
        target1 = data_iter['S_label']
        data2 = data_iter['T'][0]
        target2 = data_iter['T_label']
        inp_var = Variable(data_iter['S'][2]).float().detach().cuda() # for transfer source and target are the same

        # logging.info('target images:{}\t'.format(data_iter['T'][1]))

        data1, target1 = data1.cuda(), target1.cuda()
        data2, target2 = data2.cuda(), target2.cuda()

        # data1, target1 = Variable(data2, requires_grad=False), Variable(target2)

        target1 = Variable(target1)
        target2 = Variable(target2)

        data = Variable(torch.cat((data1, data2), 0), requires_grad=False)
        data2 = Variable(data2)

        self.state['output_gcn'] = nnGCN(data2, inp_var)

    def on_start_batch(self):
        # Todo: optimizing this step
        
        self.state['label_source_gt'] = self.state['label_source'].clone()
        self.state['label_source'][self.state['label_source'] == 0] = 1
        self.state['label_source'][self.state['label_source'] == -1] = 0

        self.state['label_target_gt'] = self.state['label_target'].clone()
        self.state['label_target'][self.state['label_target'] == 0] = 1
        self.state['label_target'][self.state['label_target'] == -1] = 0

        # input = self.state['input']
        # self.state['feature'] = input[0]
        # self.state['out'] = input[1]
        # self.state['input'] = input[2]
