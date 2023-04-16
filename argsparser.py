import argparse


def arg_parser():
    parser = argparse.ArgumentParser('train entrance')
    # ===== Data related parameters =================
    parser.add_argument('--main_path', type=str, default='/ceph/11329/chain/DHIN_Debias/datasets')
    parser.add_argument('--dataset', type=str, default='clean', help='')
    parser.add_argument('--n_run', type=int, default=1, help='')
    parser.add_argument('--n_epoch', type=int, default=3, help='')
    parser.add_argument('--bs', type=int, default=2, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='')
    parser.add_argument('--training', type=bool, default=False, help='')
    parser.add_argument('--task', type=str, default="class",
                        help='regression problem or classification problem')
    parser.add_argument('--wd', type=float, default=0.00001,
                        help='weight decay')
    # ===== Model parameters =====
    parser.add_argument('--model', type=str, default='resnet18', help='resnet18, resnet50 model')
    parser.add_argument('--pretrained', type=bool, default=False, help='pretrained?')
    parser.add_argument('--continue_train', type=bool, default=False, help='pretrained?')
    args = parser.parse_args()

    return args
