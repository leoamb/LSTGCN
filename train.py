from __future__ import division
from __future__ import print_function
import datetime
import json
import logging
import os        
import pickle
import time
import sys, shutil
import numpy as np
import torch
import configparser
from config import parser
from models.base_models import NCModel, LPModel, NRModel
from utils.data_utils import *
from utils.train_utils import get_dir_name, format_metrics
from utils.pre_utils import *
import warnings

warnings.filterwarnings('ignore')

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs((preds - labels)/labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)
    
def compute_val_loss_mstgcn(net, val_loader, criterion,  masked_flag,missing_value, epoch, hgnn_adj, hgnn_weight, adj_mx, limit=None):
    '''
    for rnn, compute mean loss on validation set
    :param net: model
    :param val_loader: torch.utils.data.utils.DataLoader
    :param criterion: torch.nn.MSELoss
    :param sw: tensorboardX.SummaryWriter
    :param global_step: int, current global_step
    :param limit: int,
    :return: val_loss
    '''
    net.train(False)  # ensure dropout layers are in evaluation mode
    with torch.no_grad():
        val_loader_length = len(val_loader)  # nb of batch
        tmp = []  # batch loss
        for batch_index, batch_data in enumerate(val_loader):
            encoder_inputs, labels = batch_data
          #  outputs = net(encoder_inputs, edge_index_data)
            outputs = net.encode(encoder_inputs, hgnn_adj, hgnn_weight, adj_mx)
            if masked_flag:
                loss = criterion(outputs, labels, missing_value)
            else:
                loss = criterion(outputs, labels)
            tmp.append(loss.item())
            if batch_index % 100 == 0:
                print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_loader_length, loss.item()))
            if (limit is not None) and batch_index >= limit:
                break

        validation_loss = sum(tmp) / len(tmp)
    return validation_loss

def train(args):
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    args.patience = args.epochs if not args.patience else  int(args.patience)
    logging.getLogger().setLevel(logging.INFO)
    
    logging.info(f'Using: {args.device}')
    logging.info("Using seed {}.".format(args.seed))

    dataset_name = args.dataset_name

    if dataset_name in ['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08']:
        #sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm

        adj_filename = os.path.join('./data', dataset_name, dataset_name+'.csv')
        graph_signal_matrix_filename = os.path.join('./data', dataset_name, dataset_name+'.npz')

        if dataset_name == 'PEMS03':
            id_filename = "./data/PEMS03/PEMS03.txt"
        else:
            id_filename = None

        num_of_vertices = args.num_of_vertices
        model_name = args.model

        batch_size = args.batch_size
        num_of_weeks = args.num_of_weeks
        num_of_days = args.num_of_days
        num_of_hours = args.num_of_hours
        
        start_epoch = args.start_epoch
        epochs = args.epochs
        
        _, train_loader, train_target_tensor, _, val_loader, val_target_tensor, _, test_loader, test_target_tensor, _mean, _std = load_graphdata_channel1(
            graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, args.device, batch_size)
        
        args.feat_dim = args.in_channels
        args.n_nodes = train_target_tensor.shape[1]
        
        adj_mx = get_adjacency_matrix2(adj_filename, num_of_vertices, id_filename=id_filename)
        hgnn_adj, hgnn_weight = convert_hgnn_adj(adj_mx)

        adj_mx = adj_mx + np.eye(adj_mx.shape[-1], dtype = adj_mx.dtype)
        adj_mx = torch.from_numpy(adj_mx).cuda()


        folder_dir = '%s_h%dd%dw%d_channel%d_%e' % (model_name, args.dim[0], args.num_layers, num_of_weeks, args.feat_dim, args.lr)
        print('folder_dir:', folder_dir)
        params_path = os.path.join('myexperiments', dataset_name, folder_dir)
        print('params_path:', params_path)
        if (start_epoch == 0) and (not os.path.exists(params_path)):
            os.makedirs(params_path)
            print('create params directory %s' % (params_path))
        elif (start_epoch == 0) and (os.path.exists(params_path)):
            shutil.rmtree(params_path)
            os.makedirs(params_path)
            print('delete the old one and create params directory %s' % (params_path))
        elif (start_epoch > 0) and (os.path.exists(params_path)):
            print('train from params directory %s' % (params_path))
        else:
            raise SystemExit('Wrong type of model!')
        
    else:
        print("Error! This dataset is not supported!")
        return 0
    
    if args.task == 'nc':
        Model = NCModel
        args.n_classes = int(data['labels'].max() + 1)
        logging.info(f'Num classes: {args.n_classes}')
    elif args.task == 'nr':
        Model = NRModel
    elif args.task == 'lp':
        args.nb_false_edges = len(data['train_edges_false'])
        args.nb_edges = len(data['train_edges'])
        if args.task == 'lp':
            Model = LPModel
        else:
            Model = RECModel
            args.eval_freq = args.epochs + 1

    model = Model(args)
    optimizer, lr_scheduler = set_up_optimizer_scheduler(args, model, args.lr)
    
    if args.cuda is not None and int(args.cuda) >= 0 :
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        model = model.to(args.device)

    total_param = 0

    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    print('Net\'s total params:', tot_params)

    t_total = time.time()
    counter = 0
    best_val_loss = model.init_metric_dict()
    best_test_metrics = None
    best_emb = None
    
    criterion = nn.SmoothL1Loss().to(args.device)
    
    global_step = 0
    best_epoch = start_epoch

    if start_epoch > 0:

        params_filename = os.path.join(params_path, 'epoch_%s.params' % start_epoch)

        model.load_state_dict(torch.load(params_filename))

        print('start epoch:', start_epoch)

        print('load weight from: ', params_filename)

    start_time = time.time()
    for epoch in range(start_epoch, epochs):       
        t = time.time()
        
        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)
        
        
        optimizer.zero_grad()
        for batch_index, batch_data in enumerate(train_loader):

            encoder_inputs, labels = batch_data

            optimizer.zero_grad()
            
            embeddings = model.encode(encoder_inputs, hgnn_adj, hgnn_weight, adj_mx)

            loss = criterion(embeddings, labels)
            loss.backward()

            if args.grad_clip is not None:
                max_norm = float(args.grad_clip)
                all_params = list(model.parameters())
                for param in all_params:
                    torch.nn.utils.clip_grad_norm_(param, max_norm)

            optimizer.step()

            training_loss = loss.item()

            global_step += 1
 
            if global_step % 1000 == 0:

                print('global step: %s, training loss: %.2f, time: %.2fs' % (global_step, training_loss, time.time() - start_time))
        lr_scheduler.step()
        val_loss = model.compute_metrics(model, val_loader, criterion, epoch, hgnn_adj, hgnn_weight, adj_mx)
    
        print('val loss', val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            
            print('best val loss: ', best_val_loss)
            torch.save(model.state_dict(), params_filename)
            print('save parameters to file: %s' % params_filename)
        print('Epoch: ', epoch)
        print('best epoch: ', best_epoch)
        model.train()
        print('LR: ',lr_scheduler.get_lr()[0])
        
    predict_main(model, params_path, best_epoch, test_loader, test_target_tensor, _mean, _std, hgnn_adj, hgnn_weight, adj_mx, 'test')

        
    logging.info("Optimization Finished!")
    logging.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    
    del model


def predict_main(model, params_path, global_step, data_loader, data_target_tensor, _mean, _std, hgnn_adj, hgnn_weight, adj_mx, type):
    '''
    :param global_step: int
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param mean: (1, 1, 3, 1)
    :param std: (1, 1, 3, 1)
    :param type: string
    :return:
    '''

    params_filename = os.path.join(params_path, 'epoch_%s.params' % global_step)
    print('load weight from:', params_filename)

    model.load_state_dict(torch.load(params_filename))

    predict_and_save_results_mstgcn(model, data_loader, data_target_tensor, global_step, _mean, _std, params_path, hgnn_adj, hgnn_weight, adj_mx, type)


def cal_std(acc):
    if acc[0] < 1:
        for i in range(len(acc)):
            acc[i] = acc[i] * 100
    mean = np.mean(acc)
    var = np.var(acc)
    std = np.std(acc)
    return mean, std

if __name__ == '__main__':


    args = parser.parse_args()
    set_seed(args.seed)
    args.dim = list(map(int, args.dim.strip('[]').split(',')))
    args.models_dim = list(map(int, args.models_dim.strip('[]').split(',')))     
    assert(args.dim[0] == sum(args.models_dim))
    if len(args.dim) == 1:
        args.dim = [args.dim[0]] * args.num_layers
                                            
    args.variables = []
                                                
    result = train(args)
    
    print(result)