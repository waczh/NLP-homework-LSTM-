import torch
# from gluonts.nursery.tsbench.src.cli import datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from data_cnews import CnewsDataset
from vocab_cnews import Vocab
from model import MyLSTM, read_pretrained_wordvec
from tqdm import tqdm
import numpy as np
import random
import argparse
import os
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description='Text Classification Training Script')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and evaluation')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for optimizer')
    parser.add_argument('--vocab_size', type=int, default=6000, help='Vocabulary size')
    parser.add_argument('--padding_len', type=int, default=500, help='Padding length for sequences')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], default='mps',
                        help='Device for training and evaluation (cpu, cuda, or mps)')

    return parser.parse_args()


def train(epoch, net, datasetname):

    print(f"training for {datasetname}")
    shutil.rmtree(datasetname+'_log')
    tensorboard_writer = SummaryWriter(datasetname+'_log')

    def evaluate(curr_ep,datasetname):
        if datasetname == 'Cnews':
            net.eval()
            correct = 0
            all = 0
            with torch.no_grad():
                for (x, y) in tqdm(val_dataset):
                    x, y = x.to(device), y.to(device)
                    logits = net(x)
                    logits = torch.argmax(logits, dim=-1)
                    correct += torch.sum(logits.eq(y)).float().item()
                    all += y.size()[0]
                    tensorboard_writer.add_scalar('Validation/Acc', correct / all, curr_ep)
            print(f'evaluate done! acc {correct / all:.5f}')
        elif datasetname == 'MSP2018':
            net.eval()
            correct = 0
            all = 0
            with torch.no_grad():
                for (x, y) in tqdm(train_dataset):
                    x, y = x.to(device), y.to(device)
                    logits = net(x)
                    logits = torch.argmax(logits, dim=-1)
                    correct += torch.sum(logits.eq(y)).float().item()
                    all += y.size()[0]
                    tensorboard_writer.add_scalar('Train/Acc', correct / all, curr_ep)
            print(f'train evaluate done! acc {correct / all:.5f}')

    is_model_stored = False
    for ep in range(epoch):
        print(f'epoch {ep} start')
        net.train()
        for (x, y) in tqdm(train_dataset):
            x, y = x.to(device), y.to(device)

            if not is_model_stored:
                tensorboard_writer.add_graph(net, x)
                is_model_stored = True

            logits = net(x)
            loss = criterion(logits, y)
            tensorboard_writer.add_scalar('Train/Loss', loss.item(), ep)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        evaluate(ep,datasetname)

    tensorboard_writer.close()


def test(net):
        net.eval()
        correct = 0
        all = 0

        with torch.no_grad():
            for (x, y) in tqdm(test_dataset):
                x, y = x.to(device), y.to(device)
                logits = net(x)
                logits = torch.argmax(logits, dim=-1)
                correct += torch.sum(logits.eq(y)).float().item()
                all += y.size()[0]
        print(f'test done! acc {correct / all:.5f}')


if __name__ == '__main__':
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    dataset_names = ['Cnews','MSP2018']

    for dataset_name in dataset_names:

        dataset_folder = 'dataset'
        dataset_folder = os.path.join(dataset_folder,dataset_name)
        print('*'*20)
        print(dataset_folder)
        os.makedirs(dataset_folder, exist_ok=True)

        train_dataset_path = os.path.join(dataset_folder, dataset_name+'.train.pth')
        val_dataset_path = os.path.join(dataset_folder, dataset_name+'.val.pth')
        test_dataset_path = os.path.join(dataset_folder, dataset_name+'.test.pth')


        if os.path.exists(train_dataset_path) and os.path.exists(val_dataset_path) \
                and os.path.exists(test_dataset_path):

            train_dataset = torch.load(train_dataset_path)
            val_dataset = torch.load(val_dataset_path)
            test_dataset = torch.load(test_dataset_path)
            print("preprocess already")
        else:
            if dataset_name == 'Cnews':

                train_dataset = CnewsDataset(r'cnews/cnews.train.txt')
                val_dataset = CnewsDataset(r'cnews/cnews.val.txt')
                test_dataset = CnewsDataset(r'cnews/cnews.test.txt')
                torch.save(train_dataset, train_dataset_path)
                torch.save(val_dataset, val_dataset_path)
                torch.save(test_dataset, test_dataset_path)

            elif dataset_name =='MSP2018':

                train_dataset = CnewsDataset(r'MSP2018/train data.txt')
                test_dataset = CnewsDataset(r'MSP2018/test data.txt')
                torch.save(train_dataset, train_dataset_path)
                # torch.save(val_dataset, val_dataset_path)
                torch.save(test_dataset, test_dataset_path)

        if dataset_name =='MSP2018': args.vocab_size = 1000; args.padding_len = 6
        vocab = Vocab(train_dataset.inputs, args.vocab_size)
        train_dataset.token2seq(vocab, args.padding_len)
        if dataset_name == 'Cnews':
            val_dataset.token2seq(vocab, args.padding_len)
        test_dataset.token2seq(vocab, args.padding_len)
        train_dataset = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        if dataset_name == 'Cnews':
            val_dataset = DataLoader(val_dataset, batch_size=args.batch_size)
        test_dataset = DataLoader(test_dataset, batch_size=args.batch_size)

        seed = 2024
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if dataset_name == 'Cnews':
            label_num = 10
        else:
            label_num = 31
            args.epochs = 50


        net = MyLSTM(read_pretrained_wordvec(r'pretrained model/tencent-ailab-embedding-zh-d100-v0.2.0-s.txt', vocab, 100),
                     len(vocab), 100, 2, 128, label_num=label_num)

        net = net.to(device)
        optimizer = optim.Adam(net.parameters(), lr = args.lr)
        criterion = nn.CrossEntropyLoss().to(device)

        train(args.epochs, net, dataset_name)
        test(net)

        print('*'*20)



