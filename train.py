import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
from torch import nn, optim
import argparse
from torch.utils.data import DataLoader
from data.noisedata import NoiseData
from model.nonlinear import NonLinear
from utils.transform import Normalizer
from torch.autograd import Variable
from loss.MyLoss import FocalLoss
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Noise estimation')
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
          default=500, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=4, type=int)
    parser.add_argument('--lr', dest='lr', help='Base learning rate.',
          default=0.01, type=float)
    parser.add_argument('--lr_decay', type = list, default = [100,200,300,400], help = 'learning rate decay')
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default='../data', type=str)
    parser.add_argument('--filename', dest='filename', help='data filename.',
          default='data_final_train.xlsx', type=str)
    parser.add_argument('--output_string', dest='output_string', help='String appended to output snapshots.', default = '', type=str)
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='NoiseData', type=str)
    parser.add_argument('--log_dir', dest='log_dir', type = str, default = 'logs/train')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    transformations = Normalizer(mean=[354.16, 32.17, 2649.37], std=[187.5, 647.17, 2045.62])#哪里来的数据？
#数据加载器
    if args.dataset == 'NoiseData':
        dataset = NoiseData(dir=args.data_dir, filename=args.filename, transform=transformations)

    train_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=2)
#建立模型  
    model = NonLinear(nc=400, out_nc=50)
#损失函数
    reg_criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    criterion1=FocalLoss(alpha=0.25, gamma=2.0)

#优化器，随机梯度
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    milestones = args.lr_decay
#学习率调度器
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    
    # tensorboard visualization
    Loss_writer = SummaryWriter(log_dir = args.log_dir)#日志记录
    softmax = nn.Softmax(dim=1)
    idx_tensor = [idx for idx in range(50)]
    
    idx_tensor = Variable(torch.FloatTensor(idx_tensor))
    total_correct=0

    for epoch in range(args.num_epochs):
        for i, (inputs, outputs, binned_outputs) in tqdm(enumerate(train_loader)):
            inputs = Variable(inputs)
            labels = Variable(outputs)
            binned_outputs = Variable(binned_outputs)
            optimizer.zero_grad()#梯度清零
            preds = model(inputs)
            #MSE loss
            _predicted = softmax(preds)
            predicted = torch.sum(_predicted * idx_tensor, 1)+20
            mse_loss= reg_criterion(predicted, outputs)

            #Cross entropy loss
            loss = criterion(preds, binned_outputs)
            total_loss=2*loss+mse_loss

            total_loss.backward()#执行反向传播的关键方法。这个调用会计算损失函数相对于模型参数的梯度
            optimizer.step()#根据梯度更新参数

            Loss_writer.add_scalar('train_loss', total_loss, epoch)
            if (i+1) % 100 == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] Losses: %.4f'
                       %(epoch+1, num_epochs, i+1, len(dataset)//batch_size, total_loss))
            # Save models at numbered epochs.

        scheduler.step()
        if (epoch+1) % 100 == 0 and (epoch+1) < num_epochs:
            print('Taking snapshot...')
            if not os.path.exists('snapshots/'):
                os.makedirs('snapshots/')
            torch.save(model.state_dict(),
            'snapshots/' + args.output_string + '_epoch_'+ str(epoch+1) + '.pth')