import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.noisedata import NoiseData
from utils.transform import Normalizer
from model.nonlinear import NonLinear
import torch
from torch.autograd import Variable
from torch import nn

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Noise estimation')
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=4, type=int)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default='../data', type=str)
    parser.add_argument('--filename', dest='filename', help='data filename.',
          default='data_final_test.xlsx', type=str)
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='NoiseData', type=str)
    parser.add_argument('--snapshot', dest='snapshot', help='Name of model snapshot.',
          default='', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    batch_size = args.batch_size
    snapshot_path = args.snapshot
    transformations = Normalizer(mean=[354.16, 32.17, 2649.37], std=[187.5, 647.17, 2045.62])

    if args.dataset == 'NoiseData':
        dataset = NoiseData(dir=args.data_dir, filename=args.filename, transform=transformations)

    print ('Loading snapshot.')
    # Load snapshot
    model = NonLinear(nc=400, out_nc=50)
    saved_state_dict = torch.load(snapshot_path, weights_only=True)
    model.load_state_dict(saved_state_dict)

    test_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=2)
    
    criterion = nn.MSELoss()
    test_error = .0
    total_correct=0
    idx_tensor = [idx for idx in range(50)]
    softmax = nn.Softmax(dim=1)
    idx_tensor = Variable(torch.FloatTensor(idx_tensor))
    total = 0
    for i, (inputs, outputs, binned_outputs) in tqdm(enumerate(test_loader)):
        total += outputs.size(0)
        inputs = Variable(inputs)
        labels = Variable(outputs)
        binned_outputs = Variable(binned_outputs)
        preds = model(inputs)
        _predicted = softmax(preds)
        predicted = torch.sum(_predicted * idx_tensor, 1)+20
        # Mean absolute error
        test_error += torch.sum(torch.abs(predicted - labels))
        _, predicted = torch.max(preds.data, 1)
        correct1 = (predicted == binned_outputs).sum().item()
        correct2 = (predicted==binned_outputs-1).sum().item()
        correct3 = (predicted==binned_outputs+1).sum().item()
        correct4 = (predicted==binned_outputs-2).sum().item()
        correct5 = (predicted==binned_outputs+2).sum().item()
        correct6 = (predicted==binned_outputs-3).sum().item()
        correct7 = (predicted==binned_outputs+3).sum().item()
        total_correct=total_correct+correct1+correct2+correct3+correct4+correct5+correct6+correct7

    accuracy = total_correct / len(dataset)
      #   print(inputs, preds, labels, binned_outputs)
        
      #   _, cls = torch.max(preds, 1)
      #   print(cls)
      #   exit()
      #   test_loss = criterion(preds, labels)
      #   test_error += torch.sum(test_loss)
        # print(preds, labels, test_loss, torch.sum(test_loss))
    print('Test error on the ' + str(total) +' test samples. MSE: %.4f' % (test_error / total))
    print('test accuracy : '+str(accuracy))
        #计算准确率
