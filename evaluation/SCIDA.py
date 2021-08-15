import argparse
from engine_test import *
from models import *
from paired_loader import *
from basenet import *
import gdown
# from loss import wFocalLoss

parser = argparse.ArgumentParser(description='s2multi Training')
parser.add_argument('--source',default='data/source', type=str, metavar='PATH', 
                    help='path to dataset (e.g. data/')
parser.add_argument('--target', default='data/target', type=str, metavar='PATH', 
                    help='path to dataset (e.g. data/')
parser.add_argument('--image-size', '-i', default=448, type=int,
                    metavar='N', help='image size (default: 448)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=[50], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='checkpoint/checkpoint_test.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--num-layer', type=int, default=2, metavar='K',
                    help='how many layers for classifier')
parser.add_argument('--optimizer', type=str, default='momentum', metavar='OP',
                    help='the name of optimizer')
parser.add_argument('--num_k', type=int, default=4, metavar='K',
                    help='how many steps to repeat the generator update')

args = parser.parse_args()
num_layer = args.num_k

def download_checkpoint():
    url = "https://drive.google.com/uc?id=1GevNHacgYkqpXtlH9-exWHwHkclPObHe"
    output = "checkpoint/checkpoint_test.pth.tar"
    print("downloading checkpoint")
    gdown.download(url, output, quiet=False)


def partial2multi():
    global args, use_gpu
    use_gpu = torch.cuda.is_available()

    # define dataset
    source_dataset = Data_single2multi(args.source, 'AID', inp_name='data/source/labels/AID_word2vector.pkl')
    target_dataset = Data_single2multi(args.target, 'AID', inp_name='data/target/labels/AID_word2vector.pkl')
    num_classes = 20
 
    ## for testing only-------------------------------------------------------------------------------
    
    # G = ResBase('resnet101')
    # C1 = ResClassifier(num_layer=num_layer)
    # C2 = ResClassifier(num_layer=num_layer)
    nnGCN = gcn_resnet101(num_classes=num_classes, t=0.4)
    
    # define loss function (criterion)
    # criterion = wFocalLoss(alpha=0.25, gamma=2, logits=True, size_average=True)
    # criterion = nn.BCEWithLogitsLoss()
    # criterionGCN = nn.MultiLabelSoftMarginLoss()

    # if args.optimizer == 'momentum':
    #     optimizer_g = torch.optim.SGD(list(G.features.parameters()),  lr=args.lr, weight_decay=0.0005)
    #     optimizer_f1 = torch.optim.SGD(list(C1.parameters()), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    #     optimizer_f2 = torch.optim.SGD(list(C2.parameters()), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    #     optimizer_gcn = torch.optim.SGD(nnGCN.get_config_optim(0.1, 0.1), lr=0.1, momentum=args.momentum, weight_decay=args.weight_decay)
    # elif args.optimizer == 'adam':
    #     optimizer_g = torch.optim.Adam(G.get_config_optim(args.lr, args.lrp), lr=args.lr, weight_decay=0.0005)
    #     optimizer_f1 = torch.optim.Adam(C1.parameters(), lr=args.lr, weight_decay=0.0005)
    #     optimizer_f2 = torch.optim.Adam(C2.parameters(), lr=args.lr, weight_decay=0.0005)
    # else:
    #     optimizer_g = torch.optim.Adadelta(G.features.parameters(), lr=args.lr, weight_decay=0.0005)
    #     optimizer_f1 = torch.optim.Adadelta(C1.parameters(), lr=args.lr, weight_decay=0.0005)
    #     optimizer_f2 = torch.optim.Adadelta(C2.parameters(), lr=args.lr, weight_decay=0.0005)

    state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs, 'evaluate': args.evaluate,
             'resume': args.resume, 'num_classes': num_classes, 'difficult_examples': True, 'save_model_path': 'checkpoint/GCN_branch_top4',
             'workers': args.workers, 'epoch_step': args.epoch_step, 'lr': args.lr, 'num_k': 4, 'num_train': 6000, 'num_val': 50, 'train_G': False}

    if args.evaluate:
        state['evaluate'] = True

    engine = GCNMultiLabelMAPEngine(state)
    # for training G, C1, C2, nnGCN 
    # engine.learning(G, C1, C2, nnGCN, criterion, criterionGCN, source_dataset, target_dataset, optimizer_g, optimizer_f1, optimizer_f2, optimizer_gcn)
    # for testing only
    engine.learning(nnGCN, source_dataset, target_dataset)


if __name__ == '__main__':
    if args.evaluate:
        download_checkpoint()
    partial2multi()
