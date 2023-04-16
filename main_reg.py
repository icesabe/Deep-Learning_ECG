# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from argsparser import arg_parser
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet18, ResNet18_Weights
import torchvision as tv
from data_loader import Dataset_classification, Dataset_regression
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch
import os
import torch.nn as nn
import logging
import time
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import pickle
import gc


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model_t, logger_t):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model_t, logger_t)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            logger_t.write(f'EarlyStopping counter: {self.counter} out of {self.patience} \n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model_t, logger_t)
            self.counter = 0

    def save_checkpoint(self, val_loss, model_s, logger_t):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            logger_t.write(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ... \n')
        if torch.cuda.device_count() > 1:  # multi-GPU
            torch.save(model_s.module.state_dict(), self.path)
        else:
            torch.save(model_s.state_dict(), self.path)
        self.val_loss_min = val_loss


def get_loader(args_t, mode, is_training=False, seed=2023):
    pos_dir = os.path.join(args_t.main_path, 'data/processed_pos/fig')
    neg_dir = os.path.join(args_t.main_path, 'data/processed_neg/part2')
    trainTransform = tv.transforms.Compose([tv.transforms.ToTensor(),
                                            tv.transforms.Normalize((0.4914, 0.4822, 0.4466), (0.247, 0.243, 0.261))])
    if args_t.task == "class":
        data_l = Dataset_classification(pos_dir, neg_dir, mode, transform=trainTransform, seed=seed)
    else:
        csv_file = os.path.join(args_t.main_path, 'data/processed_pos/label_3.csv')
        data_l = Dataset_regression(pos_dir, csv_file, mode, seed=seed, transform=trainTransform)

    return DataLoader(data_l, batch_size=args_t.bs, num_workers=6, pin_memory=False, shuffle=is_training)


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def get_logger(path_dir, mode="train"):
    logging.basicConfig(level=logging.INFO)
    logger_t = logging.getLogger('PIL')
    # logger_t = logging.getLogger()
    logger_t.setLevel(logging.WARNING)
    # logger_t.setLevel(logging.DEBUG)
    fh = logging.FileHandler(f'{path_dir}/{mode}_info_{time.time()}.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger_t.addHandler(fh)
    logger_t.addHandler(ch)
    logger_t.info(args)
    return logger_t


def metrics(pred_v, target_v):
    mae_l = mean_absolute_error(pred_v, target_v)
    mse_l = mean_squared_error(pred_v, target_v)
    return mse_l, mae_l


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = arg_parser()
    # setup seeds for reproducibility
    set_seed(2023 + int(args.n_run))
    model_path = f'{args.main_path}/model_reg/{args.model}_{args.pretrained}_{args.n_run}_{args.task}'
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    logger_file = f'{model_path}/train_info_{time.time()}.log'
    logger = open(logger_file, 'w')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    # building new model
    if '50' in args.model:
        if args.pretrained:
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            # No weights - random initialization
            model = resnet50(weights=None)
        model.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(2048, 1)
    else:  # resnet 18
        if args.pretrained:
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            # No weights - random initialization
            model = resnet18(weights=None)
        # output model
        model.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(512, 1)

    # model path where the final model is saved
    save_model_path = os.path.join(model_path, f'model_weights')
    # continue train
    if args.continue_train:
        model = model.load_state_dict(save_model_path)

    # put model on GPU device
    model = model.to(args.device)
    if args.training:
        if torch.cuda.device_count() > 1:
            logger.write("Train on multiple GPUs ... \n")
            model = nn.DataParallel(model)

        # train and valid loader
        early_stop = EarlyStopping(patience=5, verbose=True, path=save_model_path)
        # criterion = nn.BCEWithLogitsLoss()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        train_loader = get_loader(args, mode='train', is_training=True, seed=2023 + int(args.n_run))
        valid_loader = get_loader(args, mode='valid', is_training=False, seed=2023 + int(args.n_run))
        train_loss = []
        valid_loss = []
        train_metric = {"mse": [], "mae": []}
        valid_metric = {"mse": [], "mae": []}
        for epoch in range(args.n_epoch):
            # train epoch
            train_epoch_l = []
            model = model.train()
            train_label_t = []
            train_label_p = []
            pbar = tqdm(enumerate(train_loader))
            pbar.set_description(f"Training Epoch {epoch}")
            for ind, data_batch in pbar:
                for key in data_batch:
                    data_batch[key] = data_batch[key].to(device)
                # zero grad
                for param in model.parameters():
                    param.grad = None
                # forward
                outputs = torch.squeeze(model(data_batch['images']))
                # print(outputs.size(), data_batch['labels'].size())
                # sys.exit()
                loss = criterion(outputs, data_batch['labels'].to(torch.float32))
                # backward
                loss.backward()
                optimizer.step()
                pbar.set_postfix(Train_loss=loss.detach().cpu().item())
                train_epoch_l.append(loss.detach().cpu().item())
                train_label_t.extend(list(data_batch['labels'].cpu().numpy()))
                train_label_p.extend(list(outputs.detach().cpu().numpy()))
            # save training info for this epoch
            mse, mae = metrics(train_label_p, train_label_t)
            train_metric['mse'].append(mse)
            train_metric['mae'].append(mae)
            logger.write(f"train {epoch} mae: {mse}, mae: {mae}\n")
            train_loss.append(np.mean(train_epoch_l))
            # ======== validation epoch
            valid_epoch_l = []
            valid_label_t = []
            valid_label_p = []
            model = model.eval()
            pbar = tqdm(enumerate(valid_loader))
            pbar.set_description(f"Validation Epoch {epoch}")
            with torch.no_grad():
                for ind, data_batch in pbar:
                    for key in data_batch:
                        data_batch[key] = data_batch[key].to(device)
                    # forward
                    outputs = torch.squeeze(model(data_batch['images']))
                    loss = criterion(outputs, data_batch['labels'].to(torch.float32))

                    pbar.set_postfix(Valid_loss=loss.detach().cpu().item())
                    valid_epoch_l.append(loss.detach().cpu().item())

                    valid_label_t.extend(list(data_batch['labels'].cpu().numpy()))
                    valid_label_p.extend(list(outputs.detach().cpu().numpy()))
            valid_loss.append(np.mean(valid_epoch_l))
            # logger epoch information
            mse, mae = metrics(train_label_p, train_label_t)
            logger.write(f"Epoch {epoch}: train loss: {train_loss[-1]}, valid loss: {valid_loss[-1]} \n")
            logger.write(f"Valid {epoch} mse: {mse}, mae: {mae}\n")

            valid_metric['mse'].append(mse)
            valid_metric['mae'].append(mae)
            # early stop
            early_stop(valid_loss[-1], model, logger)
            if early_stop.early_stop:
                logger.write(f'Early stopping at {epoch} epoch ... \n')
                break
        # save train valid loss
        result_info = os.path.join(model_path, f'train_loss.pickle')
        res = {"train_loss": train_loss, "valid_loss": valid_loss}
        with open(result_info, "wb") as fid:
            pickle.dump(res, fid, -1)

        # save train valid metrics
        metric_info = os.path.join(model_path, f'train_metric.pickle')
        met_res = {"train": train_metric, "valid": valid_metric}
        with open(metric_info, "wb") as fid:
            pickle.dump(met_res, fid, -1)

        del train_loader
        del valid_loader
        gc.collect()
        logger.close()

        # ====== TEST
    # reload the best model and test
    torch.cuda.empty_cache()
    logger_file = f'{model_path}/test_info_{time.time()}.log'
    logger = open(logger_file, 'w')
    if '50' in args.model:
        if args.pretrained:
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            # No weights - random initialization
            model = resnet50(weights=None)
        model.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(2048, 1)
    else:  # resnet 18
        if args.pretrained:
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            # No weights - random initialization
            model = resnet18(weights=None)
        # output model
        model.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(512, 1)

    model.load_state_dict(torch.load(save_model_path, map_location=args.device))
    model = model.to(args.device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.eval()
    # test
    test_loader = get_loader(args, mode='test', is_training=False, seed=2023 + int(args.n_run))
    prob = []
    true_l = []
    pbar = tqdm(enumerate(test_loader))
    pbar.set_description(f"Testing ... ")
    with torch.no_grad():
        for ind, data_batch in pbar:
            for key in data_batch:
                data_batch[key] = data_batch[key].to(device)
            # zero grad
            for param in model.parameters():
                param.grad = None
            # forward
            outputs = torch.squeeze(model(data_batch['images']))
            # print(torch.square(outputs))
            prob.extend(list(outputs.detach().cpu().numpy()))
            true_l.extend(list(data_batch['labels'].cpu().numpy()))
    print(f"number of testing samples; {len(prob)}")
    mse, mae = metrics(prob, true_l)
    test_res = {"mse": mse, "mae": mae, }
    metric_info = os.path.join(model_path, f'test_metric.pickle')
    with open(metric_info, "wb") as fid:
        pickle.dump(test_res, fid, -1)

    print("test results: \n")
    print(f'acc: {mse} \n')
    print(f'precision: {mae} \n')

    # can remove the following.
    logger.write("test results: \n")
    logger.write(f'MSE: {mse} \n')
    logger.write(f'MAE: {mae} \n')
    logger.close()

'''
run: 
python main.py --main_path "/Users/chain/git/AI-medical/ECG/" --n_epoch 50 --lr 0.001 --task "class" --model 'resnet18' 
--training True --bs 64  --pretrained True --continue_train True --n_run 1

/home/chain/gpu
'''
