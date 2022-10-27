import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import random

import numpy as np

class DeltrDataset(Dataset):
    def __init__(self, raw_csv, prot_idx=0, standardize=False):
        # self.query_ids = query_ids
        # self.features_matrix = features_matrix
        # self.y = y
        """
            self._mus = feature_matrix.mean()
            self._sigmas = feature_matrix.std()
            protected_feature = feature_matrix[:,self._protected_feature]
            feature_matrix = (feature_matrix - self._mus) / self._sigmas
            feature_matrix[:, self._protected_feature] = protected_feature
        :param raw_csv:
        """
        if standardize:
            feature_mat = raw_csv.values[:, 1:-1]
            mus = feature_mat.mean()
            sigmas = feature_mat.std()
            protected_feature = feature_mat[:, prot_idx]
            feature_mat = (feature_mat - mus) / sigmas
            feature_mat[:, prot_idx] = protected_feature
            raw_csv[list(raw_csv.columns)[1:-1]] = feature_mat

        self.grouped_csv = raw_csv.groupby(['query_id'])
        self.groups = list(self.grouped_csv.groups.keys())

    def __getitem__(self, idx):
        query_id = self.groups[idx]
        data = self.grouped_csv.get_group(query_id).values
        feature_vec = torch.FloatTensor(data[:, 1:-1])
        y = torch.FloatTensor(data[:, -1])
        return query_id, feature_vec, y

    def __len__(self):
        return len(self.groups)


class LinearModel(nn.Module):
    def __init__(self, hidden_size=6):
        super(LinearModel, self).__init__()
        torch.manual_seed(100)
        self.fc1 = nn.Linear(hidden_size, 1)
        # if hidden_size == 6:
        #     self.fc1.weight.data = nn.Parameter(torch.FloatTensor([[-0.2043, -0.0414,  0.2660,  0.2034,  0.0674,  0.3530]]))
        # self.fc1.weight.data =  self.fc1.weight.data * 0.01

    def forward(self, x):
        # return nn.Sigmoid()(self.fc1(x))
        return self.fc1(x)


class LinearModel3Layers(nn.Module):
    def __init__(self, hidden_size=6):
        super(LinearModel3Layers, self).__init__()
        torch.manual_seed(100)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        # if hidden_size == 6:
        #     self.fc1.weight.data = nn.Parameter(torch.FloatTensor([[-0.2043, -0.0414,  0.2660,  0.2034,  0.0674,  0.3530]]))
        # self.fc1.weight.data =  self.fc1.weight.data * 0.01

    def forward(self, x):
        # return nn.Sigmoid()(self.fc1(x))
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = nn.ReLU()(self.fc3(x))
        return x

    def forward_meta_mix(self, x, l, t_index, f_index):
        n = np.random.randint(1, 4)
        x = nn.ReLU()(self.fc1(x))

        if n == 1:
            x = l * x[t_index] + (1 - l) * x[f_index]
            x = nn.ReLU()(self.fc2(x))
            x = nn.ReLU()(self.fc3(x))
        elif n == 2:
            x = nn.ReLU()(self.fc2(x))
            x = l * x[t_index] + (1 - l) * x[f_index]
            x = nn.ReLU()(self.fc3(x))
        elif n == 3:
            x = nn.ReLU()(self.fc2(x))
            x = nn.ReLU()(self.fc3(x))
            x = l * x[t_index] + (1 - l) * x[f_index]

        return x

    def forward_meta_mix_v2(self, x, l, mixed_idx_1, mixed_idx_2):
        n = np.random.randint(1, 4)
        x = nn.ReLU()(self.fc1(x))

        if n == 1:
            x_mixed = l * x[mixed_idx_1] + (1-l) * x[mixed_idx_2]
            x_mixed = torch.squeeze(x_mixed, dim=1)
            x = torch.cat([x, x_mixed], dim=0)
            x = nn.ReLU()(self.fc2(x))
            x = nn.ReLU()(self.fc3(x))
        elif n == 2:
            x = nn.ReLU()(self.fc2(x))
            x_mixed = l * x[mixed_idx_1] + (1 - l) * x[mixed_idx_2]
            x_mixed = torch.squeeze(x_mixed, dim=1)
            x = torch.cat([x, x_mixed], dim=0)
            x = nn.ReLU()(self.fc3(x))
        elif n == 3:
            x = nn.ReLU()(self.fc2(x))
            x = nn.ReLU()(self.fc3(x))
            x_mixed = l * x[mixed_idx_1] + (1 - l) * x[mixed_idx_2]
            x_mixed = torch.squeeze(x_mixed, dim=1)
            x = torch.cat([x, x_mixed], dim=0)

        return x

    def forward_meta_mix_v3(self, x1, x2, l):
        n = np.random.randint(1, 4)
        x1 = nn.ReLU()(self.fc1(x1))
        x2 = nn.ReLU()(self.fc1(x2))

        if n == 1:
            x = l * x1 + (1-l) * x2
            x = nn.ReLU()(self.fc2(x))
            x = nn.ReLU()(self.fc3(x))
        elif n == 2:
            x1 = nn.ReLU()(self.fc2(x1))
            x2 = nn.ReLU()(self.fc2(x2))
            x = l * x1 + (1-l) * x2
            x = nn.ReLU()(self.fc3(x))
        elif n == 3:
            x1 = nn.ReLU()(self.fc2(x1))
            x2 = nn.ReLU()(self.fc2(x2))

            x1 = nn.ReLU()(self.fc3(x1))
            x2 = nn.ReLU()(self.fc3(x2))

            x = l * x1 + (1-l) * x2
        return x


def compute_deltr_loss_nohinge(training_judgments, predictions, prot_attr, gamma, no_exposure=False, reduce=True):
    # predictions = torch.squeeze(predictions)
    # predictions = torch.nn.Sigmoid()(predictions)
    # training_judgments = torch.nn.Sigmoid()(training_judgments)
    a = topp(training_judgments)
    b = torch.log(topp(predictions))

    if reduce:
        # results = -torch.sum(a*b)/torch.log(torch.FloatTensor([predictions.shape[0]]).to(predictions.device))
        results = -torch.sum(a*b)
        results = torch.unsqueeze(results, dim=0)
    else:
        results = -(a*b)/torch.log(torch.FloatTensor([predictions.shape[0]]).to(predictions.device))
        # results = -(a*b)

    if not no_exposure:
        results += gamma * (exposure_diff(predictions, prot_attr) ** 2)

    return results


def compute_deltr_loss_rankmse(training_judgments, predictions, prot_attr, gamma, no_exposure=False, reduce=True):
    # predictions = torch.squeeze(predictions)
    # predictions = torch.nn.Sigmoid()(predictions)
    # training_judgments = torch.nn.Sigmoid()(training_judgments)
    # a = topp(training_judgments)
    # b = torch.log(topp(predictions))

    if reduce:
        predictions_a = torch.unsqueeze(predictions, dim=0) # 1xn
        training_judgments_a = torch.unsqueeze(training_judgments, dim=0) # 1xn
        results = F.mse_loss(predictions_a, training_judgments_a)
        # results = -torch.sum(a*b)/torch.log(torch.FloatTensor([predictions.shape[0]]).to(predictions.device))
        # results = -torch.sum(a*b)
        results = torch.unsqueeze(results, dim=0)
    else:
        results = F.mse_loss(predictions, training_judgments, reduction='none')

        # results = -(a*b)/torch.log(torch.FloatTensor([predictions.shape[0]]).to(predictions.device))
        # results = -(a*b)

    if not no_exposure:
        results += gamma * (exposure_diff_hinge(predictions, prot_attr) ** 2)

    return results


def compute_deltr_loss_rankmse_nohinge(training_judgments, predictions, prot_attr, gamma, no_exposure=False, reduce=True):
    # predictions = torch.squeeze(predictions)
    # predictions = torch.nn.Sigmoid()(predictions)
    # training_judgments = torch.nn.Sigmoid()(training_judgments)
    # a = topp(training_judgments)
    # b = torch.log(topp(predictions))

    if reduce:
        predictions_a = torch.unsqueeze(predictions, dim=0) # 1xn
        training_judgments_a = torch.unsqueeze(training_judgments, dim=0) # 1xn
        results = F.mse_loss(predictions_a, training_judgments_a)
        # results = -torch.sum(a*b)/torch.log(torch.FloatTensor([predictions.shape[0]]).to(predictions.device))
        # results = -torch.sum(a*b)
        results = torch.unsqueeze(results, dim=0)
    else:
        results = F.mse_loss(predictions, training_judgments, reduction='none')

        # results = -(a*b)/torch.log(torch.FloatTensor([predictions.shape[0]]).to(predictions.device))
        # results = -(a*b)

    if not no_exposure:
        results += gamma * (exposure_diff(predictions, prot_attr) ** 2)

    return results


def get_pairwise_comp_probs(batch_preds, batch_std_labels, sigma=None):
    '''
    Get the predicted and standard probabilities p_ij which denotes d_i beats d_j
    @param batch_preds:
    @param batch_std_labels:
    @param sigma:
    @return:
    '''
    # computing pairwise differences w.r.t. predictions, i.e., s_i - s_j
    batch_s_ij = torch.unsqueeze(batch_preds, dim=2) - torch.unsqueeze(batch_preds, dim=1)
    batch_p_ij = torch.sigmoid(sigma * batch_s_ij)

    # computing pairwise differences w.r.t. standard labels, i.e., S_{ij}
    batch_std_diffs = torch.unsqueeze(batch_std_labels, dim=2) - torch.unsqueeze(batch_std_labels, dim=1)
    # ensuring S_{ij} \in {-1, 0, 1}
    batch_Sij = torch.clamp(batch_std_diffs, min=-1.0, max=1.0)
    batch_std_p_ij = 0.5 * (1.0 + batch_Sij)

    return batch_p_ij, batch_std_p_ij


def compute_deltr_loss_ranknet_nohinge(training_judgments, predictions, prot_attr, gamma, no_exposure=False, reduce=True):
    # predictions = torch.squeeze(predictions)
    # predictions = torch.nn.Sigmoid()(predictions)
    # training_judgments = torch.nn.Sigmoid()(training_judgments)
    # a = topp(training_judgments)
    # b = torch.log(topp(predictions))

    if reduce:
        # predictions_a = torch.unsqueeze(predictions, dim=0) # 1xn
        # training_judgments_a = torch.unsqueeze(training_judgments, dim=0) # 1xn
        predictions_a = predictions
        training_judgments_a = training_judgments
        batch_p_ij, batch_std_p_ij = get_pairwise_comp_probs(batch_preds=predictions_a, batch_std_labels=training_judgments_a,
                                                             sigma=1.0)
        results = F.binary_cross_entropy(input=torch.triu(batch_p_ij, diagonal=1),
                                             target=torch.triu(batch_std_p_ij, diagonal=1))

        results = torch.unsqueeze(results, dim=0)
    else:
        # predictions_a = torch.unsqueeze(predictions, dim=0) # 1xn
        # training_judgments_a = torch.unsqueeze(training_judgments, dim=0) # 1xn
        predictions_a = predictions
        training_judgments_a = training_judgments
        batch_p_ij, batch_std_p_ij = get_pairwise_comp_probs(batch_preds=predictions_a, batch_std_labels=training_judgments_a,
                                                             sigma=1.0)
        results = F.binary_cross_entropy(input=torch.triu(batch_p_ij, diagonal=1),
                                         target=torch.triu(batch_std_p_ij, diagonal=1), reduction='none')
        # results = -(a*b)/torch.log(torch.FloatTensor([predictions.shape[0]]).to(predictions.device))
        # results = -(a*b)

    if not no_exposure:
        results += gamma * (exposure_diff(predictions, prot_attr) ** 2)

    return results


def compute_deltr_loss_ranknet(training_judgments, predictions, prot_attr, gamma, no_exposure=False, reduce=True):
    # predictions = torch.squeeze(predictions)
    # predictions = torch.nn.Sigmoid()(predictions)
    # training_judgments = torch.nn.Sigmoid()(training_judgments)
    # a = topp(training_judgments)
    # b = torch.log(topp(predictions))

    if reduce:
        # predictions_a = torch.unsqueeze(predictions, dim=0) # 1xn
        # training_judgments_a = torch.unsqueeze(training_judgments, dim=0) # 1xn
        predictions_a = predictions
        training_judgments_a = training_judgments
        batch_p_ij, batch_std_p_ij = get_pairwise_comp_probs(batch_preds=predictions_a, batch_std_labels=training_judgments_a,
                                                             sigma=0.5)
        results = F.binary_cross_entropy(input=torch.triu(batch_p_ij, diagonal=1),
                                             target=torch.triu(batch_std_p_ij, diagonal=1))

        results = torch.unsqueeze(results, dim=0)
    else:
        # predictions_a = torch.unsqueeze(predictions, dim=0) # 1xn
        # training_judgments_a = torch.unsqueeze(training_judgments, dim=0) # 1xn
        predictions_a = predictions
        training_judgments_a = training_judgments
        batch_p_ij, batch_std_p_ij = get_pairwise_comp_probs(batch_preds=predictions_a, batch_std_labels=training_judgments_a,
                                                             sigma=0.5)
        results = F.binary_cross_entropy(input=torch.triu(batch_p_ij, diagonal=1),
                                         target=torch.triu(batch_std_p_ij, diagonal=1), reduction='none')
        # results = -(a*b)/torch.log(torch.FloatTensor([predictions.shape[0]]).to(predictions.device))
        # results = -(a*b)


    if not no_exposure:
        results += gamma * (exposure_diff_hinge(predictions, prot_attr) ** 2)

    return results


def compute_deltr_loss(training_judgments, predictions, prot_attr, gamma, no_exposure=False, reduce=True):
    # predictions = torch.squeeze(predictions)
    # predictions = torch.nn.Sigmoid()(predictions)
    # training_judgments = torch.nn.Sigmoid()(training_judgments)
    a = topp(training_judgments)
    b = torch.log(topp(predictions))

    if reduce:
        # results = -torch.sum(a*b)/torch.log(torch.FloatTensor([predictions.shape[0]]).to(predictions.device))
        results = -torch.sum(a*b)
        results = torch.unsqueeze(results, dim=0)
    else:
        results = -(a*b)/torch.log(torch.FloatTensor([predictions.shape[0]]).to(predictions.device))
        # results = -(a*b)

    if not no_exposure:
        results += gamma * (exposure_diff_hinge(predictions, prot_attr) ** 2)

    return results

def compute_deltr_loss_spl_v0(training_judgments, predictions, prot_attr, gamma, threshold,
                              no_exposure=False, reduce=True):
    # predictions = torch.squeeze(predictions)
    # predictions = torch.nn.Sigmoid()(predictions)
    # training_judgments = torch.nn.Sigmoid()(training_judgments)
    a = topp(training_judgments)
    b = torch.log(topp(predictions))

    if reduce:
        # results = -torch.sum(a*b)/torch.log(torch.FloatTensor([predictions.shape[0]]).to(predictions.device))
        r = a*b
        r = torch.where(r <= threshold, r, torch.tensor(0, dtype=r.dtype))

        results = -torch.sum(r)
        results = torch.unsqueeze(results, dim=0)
    else:
        results = -(a*b)/torch.log(torch.FloatTensor([predictions.shape[0]]).to(predictions.device))
        results = torch.where(results <= threshold, results, torch.tensor(0, dtype=results.dtype))

        # results = -(a*b)

    if not no_exposure:
        results += gamma * (exposure_diff(predictions, prot_attr) ** 2)


    # v_index = results < threshold
    # v = torch.unsqueeze(torch.zeros(results.shape).int().to(results.device)[v_index], 1)
    # results = results * v
    return results


def topp(v):
    return torch.exp(v)/torch.sum(torch.exp(v))


def exposure_diff_hinge(data, prot_attr):
    exposure_prot = normalized_exposure(data[prot_attr==True], data)
    exposure_nprot = normalized_exposure(data[prot_attr==False], data)
    exposure_diff = torch.max(torch.FloatTensor([0.0]).to(data.device), (exposure_nprot - exposure_prot))

    return exposure_diff

def exposure_diff(data, prot_attr):
    exposure_prot = normalized_exposure(data[prot_attr==True], data)
    exposure_nprot = normalized_exposure(data[prot_attr==False], data)
    exposure_diff = exposure_nprot - exposure_prot

    return exposure_diff


def normalized_exposure(group_data, all_data):
    """
    calculates the exposure of a group in the entire ranking
    implementation of equation 4 in DELTR paper

    :param group_data: predictions of relevance scores for one group
    :param all_data: all predictions

    :return: float value that is normalized exposure in a ranking for one group
             nan if group size is 0
    """
    return (torch.sum(topp_prot(group_data, all_data) / np.log(2))) / group_data.shape[0]


def topp_prot(group_items, all_items):
    """
    given a dataset of features what is the probability of being at the top position
    for one group (group_items) out of all items
    example: what is the probability of each female (or male respectively) item (group_items) to be in position 1
    implementation of equation 7 in paper DELTR

    :param group_items: vector of predicted scores of one group (protected or non-protected)
    :param all_items: vector of predicted scores of all items

    :return: numpy array of float values
    """
    # return torch.exp(group_items) / torch.sum(torch.exp(all_items))
    exp_1 = torch.exp(group_items)
    exp_2 = torch.exp(all_items)

    if torch.isnan(exp_1).any() or torch.isnan(exp_2).any():
        raise ValueError

    return exp_1/torch.sum(exp_2)