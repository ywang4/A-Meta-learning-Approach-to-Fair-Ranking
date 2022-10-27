import sys
import pandas as pd
import os
import torch
import numpy as np
import tqdm
from torch.utils.data import DataLoader

from utils.deltr_torch import DeltrDataset
from utils.deltr_torch import compute_deltr_loss
from utils.deltr_torch import LinearModel
from evaluate_deltr_optuna import DELTR_Evaluator


from utils.mwn import MLP, MetaSGD

from torch.utils.tensorboard import SummaryWriter
import argparse

torch.manual_seed(42)

USE_CUDA = False
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")


def prepare_meta_data(train_csv, num_meta_per_query=10, inclusive=True):
    train_csv['row_id'] = range(len(train_csv))
    try:
        train_meta_csv = train_csv.groupby(['query_id', 'gender']).apply(lambda x: x.sample(num_meta_per_query,
                                                                                            random_state=42)).reset_index(drop=True)
    except:
        train_meta_csv = train_csv.groupby(['query_id', 'gender']).apply(lambda x: x.sample(8,
                                                                                            random_state=42)).reset_index(drop=True)
    if inclusive:
        train_new_csv = train_csv
    else:
        train_new_csv = train_csv[~train_csv.row_id.isin(train_meta_csv.row_id)]

    train_new_csv = train_new_csv.drop(columns=['row_id'])
    train_meta_csv = train_meta_csv.drop(columns=['row_id'])

    return train_new_csv, train_meta_csv


def train_mfr(train_csv, train_meta_csv, test_csv, dataset='TREC', fold=0, prot_idx=0, epoch=100,
                num_feature=6):
    '''

    :param prot_idx: 0 for gender attr in TREC dataset
    :return:
    '''
    writer = SummaryWriter(comment='mwn_{}_gamma_{}_fold_{}'.format(dataset, gamma, fold))

    train_dataset = DeltrDataset(raw_csv=train_csv, prot_idx=prot_idx, standardize=True)
    train_meta_dataset = DeltrDataset(raw_csv=train_meta_csv, prot_idx=prot_idx, standardize=True)
    # test_dataset = DeltrDataset(raw_csv=test_csv)

    train_loader = DataLoader(train_dataset, shuffle=False, batch_size=1)
    train_meta_loader = DataLoader(train_meta_dataset, shuffle=False, batch_size=1)
    # test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)

    model = LinearModel(hidden_size=num_feature)
    vnet = MLP(args.vnet_hidden_dim, args.vnet_hidden_layers)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    vnet_optimizer = torch.optim.Adam(vnet.parameters(), lr=args.vnet_lr)

    model = model.to(DEVICE)
    vnet = vnet.to(DEVICE)

    results = []
    weights = []
    prot_attr_saved = []
    q = []
    # train
    for e in tqdm.tqdm(range(epoch)):
        loss_values = []

        # summed_loss = 0.0
        train_loss = 0.0
        meta_loss = 0.0
        meta_dataloader_iter = iter(train_meta_loader)
        weights_epoch = []
        for i, (query_id, feature_mat, ys) in enumerate(train_loader):
            model.train()
            feature_mat = feature_mat.to(DEVICE)
            feature_mat = torch.squeeze(feature_mat, 0)
            ys = ys.to(DEVICE)
            ys = ys.T
            if e % args.frequency == 0:

                meta_model = LinearModel(hidden_size=num_feature)
                meta_model = meta_model.to(DEVICE)
                meta_model.load_state_dict(model.state_dict())
                meta_model.train()

                outputs = meta_model(feature_mat)
                cost = compute_deltr_loss(ys, outputs, feature_mat[:, prot_idx], gamma, reduce=False)
                cost_v = cost * feature_mat.shape[0]

                v_lambda = vnet(cost_v.data)
                l_f_meta = torch.sum(cost_v * v_lambda)/len(cost_v)
                meta_model.zero_grad()
                grads = torch.autograd.grad(l_f_meta, meta_model.parameters(), create_graph=True)

                # meta_lr = 0.01 * ((0.1 ** int(epoch >= 50)) * (0.1 ** int(epoch >= 80)))
                meta_optimizer = MetaSGD(meta_model, meta_model.parameters(), momentum=args.meta_momentum,
                                         lr=args.meta_lr)

                meta_optimizer.load_state_dict(optimizer.state_dict())
                meta_optimizer.meta_step(grads)

                del grads
                try:
                    query_id, feature_val, ys_val = next(meta_dataloader_iter)
                except StopIteration:
                    meta_dataloader_iter = iter(train_meta_loader)
                    query_id, feature_val, ys_val = next(meta_dataloader_iter)

                feature_val, ys_val = feature_val.to(DEVICE), ys_val.to(DEVICE)
                feature_val = torch.squeeze(feature_val, 0)
                ys_val = ys_val.T

                y_g_hat = meta_model(feature_val)
                l_g_meta = compute_deltr_loss(ys_val, y_g_hat, feature_val[:, prot_idx], gamma, reduce=True)
                # l_g_meta = compute_deltr_loss(ys_val, y_g_hat, feature_val[:, prot_idx], 0, reduce=True)


                vnet_optimizer.zero_grad()
                l_g_meta.backward()
                vnet_optimizer.step()
                meta_loss += l_g_meta.item()

            outputs = model(feature_mat)
            cost = compute_deltr_loss(ys, outputs, feature_mat[:, prot_idx], gamma, reduce=False)
            cost_v = cost * feature_mat.shape[0]

            with torch.no_grad():
                w_new = vnet(cost_v)
                weights_epoch.append(w_new.detach().cpu().numpy().squeeze())
                # prot_attr_saved.append(feature_mat[:, prot_idx].detach().cpu().numpy())

            loss = torch.sum(cost_v * w_new) / len(cost_v)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            train_loss += loss.item()
            if e + 1 == epoch:
                prot_attr_saved.append(feature_mat[:, prot_idx].detach().cpu().numpy().squeeze())
                results.append(outputs.detach().cpu().numpy().squeeze())
                q.append(query_id.tile(ys.shape[0]).detach().cpu().numpy().squeeze())

        weights.append(np.concatenate(weights_epoch))

        writer.add_scalar('Loss/train', train_loss/len(train_loader), e)
        writer.add_scalar('Loss/meta', meta_loss/len(train_loader), e)

    # prot_attr_saved = np.concatenate(prot_attr_saved)
    # results = np.concatenate(results)
    # q = np.concatenate(q)
    # prot_attr_saved = prot_attr_saved.squeeze()
    # if dataset == 'LAW_GENDER' or 'LAW_RACE':
    #     df = pd.DataFrame(data={'prot_attr': prot_attr_saved, 'score': results, 'query': q})
    #     df['max_rank'] = df['score'].rank(method='min', ascending=False)
    #     df[df['prot_attr'] == 1].to_csv('./results/deltr_torch/weights/{}_{}_rank_1.csv'.format(
    #              dataset_name, fold), index=False)

    np.save('./results/mfr/weights/results_{}_{}'.format(dataset, fold), results)
    np.save('./results/mfr/weights/weights_{}_{}'.format(dataset, fold), weights)
    np.save('./results/mfr/weights/prot_attr_{}_{}'.format(dataset, fold), prot_attr_saved)


    # inference
    model.eval()
    test_data = test_csv.values
    test_feature_mat = test_data[:, 1:-1]
    test_query_ids = test_data[:, 0].reshape(-1,1)

    test_feature_mat = torch.FloatTensor(test_feature_mat).to(DEVICE)
    test_ys_pred = np.squeeze(model(test_feature_mat).detach().cpu().numpy()).reshape(-1,1)
    test_ys = test_data[:, -1].reshape(-1,1)

    if args.saved.strip() == 'small':
        gamma_str = 'SMALL'
    elif args.saved.strip() == 'large':
        gamma_str = 'LARGE'
    else:
        raise ValueError

    # if dataset in ['TREC', 'LAW_GENDER']:
    #     if gamma < 20001:
    #         gamma_str = 'SMALL'
    #     else:
    #         gamma_str = 'LARGE'
    # elif dataset in ['LAW_RACE']:
    #     if gamma < 1000001:
    #         gamma_str = 'SMALL'
    #     else:
    #         gamma_str = 'LARGE'
    # elif dataset in ['ENG_HIGHSCHOOL']:
    #     if gamma < 100001:
    #         gamma_str = 'SMALL'
    #     else:
    #         gamma_str = 'LARGE'
    # elif dataset in ['ENG_GENDER']:
    #     if gamma < 3001:
    #         gamma_str = 'SMALL'
    #     else:
    #         gamma_str = 'LARGE'

    torch.save(model.state_dict(), './results/mfr/{}/fold_{}/GAMMA={}/model'.format(dataset,
                                                                                                    fold,
                                                                                                    gamma_str))

    f_pred_path = './results/mfr/{}/fold_{}/GAMMA={}/predictions.pred'.format(dataset,
                                                                                                    fold,
                                                                                                    gamma_str)
    f_true_path = './results/mfr/{}/fold_{}/GAMMA={}/trainingScores_ORIG.pred'.format(dataset,
                                                                                                    fold,
                                                                                                    gamma_str)
    # with open(f_pred_path, 'w+') as f_pred:
    #     with open(f_true_path, 'w+') as f_true:
    #         for j in range(test_ys_pred.shape[0]):
    #             q_id = test_query_ids[j]
    #             f_pred.write('{},{},{},{}\n'.format(int(q_id),
    #                                            j,
    #                                            test_ys_pred[j],
    #                                            test_feature_mat[j, prot_idx]))
    #
    #             f_true.write('{},{},{},{}\n'.format(int(q_id),
    #                                            j,
    #                                            test_ys[j],
    #                                            test_feature_mat[j, prot_idx]))

    prot_data = test_feature_mat[:, prot_idx].detach().cpu().numpy().reshape(-1,1)
    doc_ids = np.asarray(list(range(test_ys_pred.shape[0]))).reshape(-1,1)
    pred_data = np.concatenate((test_query_ids, doc_ids, test_ys_pred, prot_data), 1)
    true_data = np.concatenate((test_query_ids, doc_ids, test_ys, prot_data), 1)

    if np.isnan(pred_data).any():
        raise ValueError('Nan in prediction array!!!')

    pred_df = pd.DataFrame(data=pred_data, columns=["query_id", "doc_id", "prediction", "prot_attr"])
    if dataset == 'LAW_GENDER' or 'LAW_RACE':
        temp_df = pred_df.copy()
        temp_df['rank'] = temp_df['prediction'].rank(method='min', ascending=False)
        temp_df[temp_df['prot_attr'] == 1].to_csv('./results/mfr/weights/{}_{}_rank_1.csv'.format(
                 dataset_name, fold), index=False)

    agg_rows = []
    for g, rows in pred_df.groupby(['query_id']):
        rows = rows.sort_values(['prediction'], ascending=False)
        agg_rows.append(rows)
    pred_df_sorted = pd.concat(agg_rows)
    pred_df_sorted.to_csv(f_pred_path, header=False, index=False)

    true_df = pd.DataFrame(data=true_data, columns=["query_id", "doc_id", "prediction", "prot_attr"])
    true_df.to_csv(f_true_path, header=False, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataset', type=str, default='LAW_RACE')
    parser.add_argument('-g', '--gamma', type=int, default=200000)
    parser.add_argument('-e', '--epoch', type=int, default=100)
    parser.add_argument('-f', '--frequency', type=int, default=1)
    parser.add_argument('-s', '--saved', type=str, default='large')
    parser.add_argument('--vnet_hidden_dim', type=int, default=50)
    parser.add_argument('--vnet_hidden_layers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--vnet_lr', type=float, default=1e-3)
    parser.add_argument('--meta_lr', type=float, default=1e-3)
    parser.add_argument('--meta_momentum', type=float, default=0.9)
    parser.add_argument('--skewed', type=int, default=0)
    parser.add_argument('--down_sample', type=float, default=1.0)
    args = parser.parse_args()

    print(args)

    # dataset_name = 'LAW_RACE'
    # gamma = 'large'
    dataset_name = args.dataset
    gamma = args.gamma
    sampled = args.down_sample

    print(dataset_name)
    print(gamma)

    if dataset_name == 'TREC':
        ROOT_PAHT = '/home/ywang/fairsearch/DELTR-Experiments/data/TREC/'
        TRAIN_FILE = 'features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv'
        TEST_FILE = 'features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv'

        for fold in range(1,7):
            print('----------------- FOLD {} -----------------'.format(fold))
            train_file = os.path.join(ROOT_PAHT, 'fold_{}'.format(fold), TRAIN_FILE)
            test_file = os.path.join(ROOT_PAHT, 'fold_{}'.format(fold), TEST_FILE)
            train_csv = pd.read_csv(train_file,
                                    sep=',',
                                    names=['query_id', 'gender', '1', '2', '3', '4', '5', 'score'])
            test_csv = pd.read_csv(test_file,
                                   sep=',',
                                   names=['query_id', 'gender', '1', '2', '3', '4', '5', 'score'])
            train_csv, train_meta_csv = prepare_meta_data(train_csv.sample(frac=sampled), num_meta_per_query=int(15*sampled))

            # 20K, 200K
            train_mfr(train_csv, train_meta_csv, test_csv, dataset=dataset_name, prot_idx=0,
                            fold=fold, epoch=args.epoch)
    elif dataset_name == 'LAW_GENDER':
        ROOT_PAHT = '/home/ywang/fairsearch/DELTR-Experiments/data/LawStudents/gender/'
        TRAIN_FILE = 'LawStudents_Gender_train.txt'
        TEST_FILE = 'LawStudents_Gender_test.txt'

        train_file = os.path.join(ROOT_PAHT, TRAIN_FILE)
        test_file = os.path.join(ROOT_PAHT, TEST_FILE)

        if not args.skewed:
            train_csv = pd.read_csv(train_file,
                                    sep=',',
                                    names=['query_id', 'gender', '1', '2', 'score'])
        else:
            train_csv = pd.read_csv(train_file,
                                    sep=',',
                                    names=['query_id', 'gender', '1', '2', 'score'])
            train_csv = pd.concat([train_csv[train_csv.gender == 0],
                                   train_csv[train_csv.gender == 1].sample(frac=0.15)]).reset_index(drop=True)
        print('skewness:', len(train_csv[train_csv.gender == 0]) / len(train_csv[train_csv.gender == 1]))

        test_csv = pd.read_csv(test_file,
                               sep=',',
                               names=['query_id', 'gender', '1', '2', 'score'])

        train_csv, train_meta_csv = prepare_meta_data(train_csv.sample(frac=sampled), num_meta_per_query=int(100*sampled))
        # 3K, 50K
        train_mfr(train_csv, train_meta_csv, test_csv, dataset=dataset_name, prot_idx=0, fold=1,
                        epoch=args.epoch, num_feature=3)




    elif dataset_name == 'LAW_RACE':
        ROOT_PAHT = '/home/ywang/fairsearch/DELTR-Experiments/data/LawStudents/race_black/'
        TRAIN_FILE = 'LawStudents_Race_train.txt'
        TEST_FILE = 'LawStudents_Race_test.txt'

        train_file = os.path.join(ROOT_PAHT, TRAIN_FILE)
        test_file = os.path.join(ROOT_PAHT, TEST_FILE)
        train_csv = pd.read_csv(train_file,
                                sep=',',
                                names=['query_id', 'gender', '1', '2','score'])
        test_csv = pd.read_csv(test_file,
                               sep=',',
                               names=['query_id', 'gender', '1', '2', 'score'])

        train_csv, train_meta_csv = prepare_meta_data(train_csv.sample(frac=sampled), num_meta_per_query=int(100*sampled))
        # 1M, 50M
        train_mfr(train_csv, train_meta_csv, test_csv, dataset=dataset_name, prot_idx=0, fold=1,
                        epoch=args.epoch, num_feature=3,)
    elif dataset_name == 'ENG_HIGHSCHOOL':
        ROOT_PAHT = '/home/ywang/fairsearch/DELTR-Experiments/data/EngineeringStudents/NoSemiPrivate/highschool/'
        for fold in range(1,6):
            print('----------------- FOLD {} -----------------'.format(fold))
            TRAIN_FILE = 'chileDataL2R_highschool_nosemi_fold{}_train.txt'.format(fold)
            TEST_FILE = 'chileDataL2R_highschool_nosemi_fold{}_test.txt'.format(fold)

            train_file = os.path.join(ROOT_PAHT, 'fold_{}'.format(fold), TRAIN_FILE)
            test_file = os.path.join(ROOT_PAHT, 'fold_{}'.format(fold), TEST_FILE)
            train_csv = pd.read_csv(train_file,
                                    sep=',',
                                    names=['query_id', 'gender', '1', '2', '3', '4', 'score'])
            if args.skewed:
                train_csv = pd.concat([train_csv[train_csv.gender == 0],
                                       train_csv[train_csv.gender == 1].groupby('query_id').apply(
                                           lambda x: x.sample(frac=0.5))]).reset_index(drop=True)
            test_csv = pd.read_csv(test_file,
                                   sep=',',
                                   names=['query_id', 'gender', '1', '2', '3', '4', 'score'])
            train_csv, train_meta_csv = prepare_meta_data(train_csv.sample(frac=sampled), num_meta_per_query=int(50*sampled))
            # 100k, 5M
            train_mfr(train_csv, train_meta_csv, test_csv, dataset=dataset_name, prot_idx=0,
                            fold=fold, epoch=args.epoch, num_feature=5)
    elif dataset_name == 'ENG_GENDER':
        ROOT_PAHT = '/home/ywang/fairsearch/DELTR-Experiments/data/EngineeringStudents/NoSemiPrivate/gender/'
        for fold in range(1,6):
            print('----------------- FOLD {} -----------------'.format(fold))
            TRAIN_FILE = 'chileDataL2R_gender_nosemi_fold{}_train.txt'.format(fold)
            TEST_FILE = 'chileDataL2R_gender_nosemi_fold{}_test.txt'.format(fold)

            train_file = os.path.join(ROOT_PAHT, 'fold_{}'.format(fold), TRAIN_FILE)
            test_file = os.path.join(ROOT_PAHT, 'fold_{}'.format(fold), TEST_FILE)
            train_csv = pd.read_csv(train_file,
                                    sep=',',
                                    names=['query_id', 'gender', '1', '2', '3', '4', 'score'])
            num_meta_per_query = int(50*sampled)
            if args.skewed:
                train_csv = pd.concat([train_csv[train_csv.gender == 0],
                                       train_csv[train_csv.gender == 1].groupby('query_id').apply(
                                           lambda x: x.sample(frac=0.3))]).reset_index(drop=True)
                num_meta_per_query = 20
            test_csv = pd.read_csv(test_file,
                                   sep=',',
                                   names=['query_id', 'gender', '1', '2', '3', '4', 'score'])

            train_csv, train_meta_csv = prepare_meta_data(train_csv.sample(frac=sampled), num_meta_per_query=num_meta_per_query)
            # 3K, 50K
            train_mfr(train_csv, train_meta_csv, test_csv,dataset=dataset_name, prot_idx=0,
                            fold=fold, epoch=args.epoch, num_feature=5)
    elif dataset_name == 'GERMAN_AGE25':
        ROOT_PAHT = '/home/ywang/fairsearch/german_credits/age25/'
        TRAIN_FILE = 'train.txt'
        TEST_FILE = 'test.txt'

        train_file = os.path.join(ROOT_PAHT, TRAIN_FILE)
        test_file = os.path.join(ROOT_PAHT, TEST_FILE)
        train_csv = pd.read_csv(train_file,
                                sep=',',
                                names=['query_id', 'gender', '1', '2','score'])
        test_csv = pd.read_csv(test_file,
                               sep=',',
                               names=['query_id', 'gender', '1', '2', 'score'])

        train_csv, train_meta_csv = prepare_meta_data(train_csv, num_meta_per_query=100)
        # 3K, 50K
        train_mfr(train_csv, train_meta_csv, test_csv, dataset=dataset_name, prot_idx=0, fold=1,
                        epoch=args.epoch, num_feature=3)
    elif dataset_name == 'GERMAN_AGE35':

        ROOT_PAHT = '/home/ywang/fairsearch/german_credits/age35/'
        TRAIN_FILE = 'train.txt'
        TEST_FILE = 'test.txt'

        train_file = os.path.join(ROOT_PAHT, TRAIN_FILE)
        test_file = os.path.join(ROOT_PAHT, TEST_FILE)
        train_csv = pd.read_csv(train_file,
                                sep=',',
                                names=['query_id', 'gender', '1', '2','score'])
        test_csv = pd.read_csv(test_file,
                               sep=',',
                               names=['query_id', 'gender', '1', '2', 'score'])
        train_csv, train_meta_csv = prepare_meta_data(train_csv, num_meta_per_query=100)

        # 3K, 50K
        train_mfr(train_csv, train_meta_csv, test_csv, dataset=dataset_name, prot_idx=0, fold=1,
                        epoch=args.epoch, num_feature=3)
    elif dataset_name == 'GERMAN_SEX':

        ROOT_PAHT = '/home/ywang/fairsearch/german_credits/sex/'
        TRAIN_FILE = 'train.txt'
        TEST_FILE = 'test.txt'

        train_file = os.path.join(ROOT_PAHT, TRAIN_FILE)
        test_file = os.path.join(ROOT_PAHT, TEST_FILE)
        train_csv = pd.read_csv(train_file,
                                sep=',',
                                names=['query_id', 'gender', '1', '2','score'])
        test_csv = pd.read_csv(test_file,
                               sep=',',
                               names=['query_id', 'gender', '1', '2', 'score'])
        train_csv, train_meta_csv = prepare_meta_data(train_csv, num_meta_per_query=100)

        # 3K, 50K
        train_mfr(train_csv, train_meta_csv, test_csv, dataset=dataset_name, prot_idx=0, fold=1,
                        epoch=args.epoch, num_feature=3)

    elif dataset_name == 'COMPASS_RACE':

        ROOT_PAHT = '/home/ywang/fairsearch/compass/'
        TRAIN_FILE = 'compass_race_train.txt'
        TEST_FILE = 'compass_race_test.txt'

        train_file = os.path.join(ROOT_PAHT, TRAIN_FILE)
        test_file = os.path.join(ROOT_PAHT, TEST_FILE)
        train_csv = pd.read_csv(train_file,
                                sep=',',
                                names=['query_id', 'gender', '1', 'score'])
        test_csv = pd.read_csv(test_file,
                               sep=',',
                               names=['query_id', 'gender', '1', 'score'])
        train_csv, train_meta_csv = prepare_meta_data(train_csv.sample(frac=sampled), num_meta_per_query=int(500*sampled))

        # 3K, 50K
        train_mfr(train_csv, train_meta_csv, test_csv, dataset=dataset_name, prot_idx=0, fold=1,
                        epoch=args.epoch, num_feature=2)

    if dataset_name == 'ENG_HIGHSCHOOL':
        deltr_dataset = 'engineering-highschool-withoutSemiPrivate'
        binSize = 20
    elif dataset_name == 'ENG_GENDER':
        deltr_dataset = 'engineering-gender-withoutSemiPrivate'
        binSize = 20
    elif dataset_name == 'LAW_RACE':
        deltr_dataset = 'law-black'
        binSize = 200
    elif dataset_name == 'LAW_GENDER':
        deltr_dataset = 'law-gender'
        binSize = 200
    elif dataset_name == 'TREC':
        deltr_dataset = 'trec'
        binSize = 10
    elif dataset_name == 'GERMAN_AGE25':
        deltr_dataset = 'german_age25'
        binSize = 10
    elif dataset_name == 'GERMAN_AGE35':
        deltr_dataset = 'german_age35'
        binSize = 10
    elif dataset_name == 'GERMAN_SEX':
        deltr_dataset = 'german_sex'
        binSize = 10
    elif dataset_name == 'COMPASS_SEX':
        deltr_dataset = 'compass_sex'
        binSize = 10
    elif dataset_name == 'COMPASS_RACE':
        deltr_dataset = 'compass_race'
        binSize = 10
    else:
        deltr_dataset = None
        binSize = None
        raise ValueError

    resultDir = './results/mfr/{}/'.format(dataset_name)
    trainingDir = './results/mfr/'
    protAttr = 1
    evaluator = DELTR_Evaluator(deltr_dataset,
                                resultDir,
                                trainingDir,
                                binSize,
                                protAttr)
    result = evaluator.evaluate()

    # if result.values[9] == 1 or result.values[21]==1:
    #     return 0.0

    if dataset_name == 'TREC':
        # p@10
        print('p@10:', result.values[9])
    else:
        # kt
        print('kt:', result.values[21])

    print('fairness', result.values[23])
