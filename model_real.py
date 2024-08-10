import functools

print = functools.partial(print, flush=True)
import torch, os,sys
from torch import Tensor
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
import time, math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from Focal_loss import FocalLoss
from torchensemble import FusionClassifier, VotingClassifier, BaggingClassifier, GradientBoostingClassifier, \
    AdversarialTrainingClassifier, FastGeometricClassifier, SnapshotEnsembleClassifier, SoftGradientBoostingClassifier
from copy import deepcopy
import datetime
# from linformer import Linformer
from sklearn.metrics import confusion_matrix

from Hyperparameters import args

from typing import Any, List, Optional, Tuple

Cnames = ['Health', 'NAFLD']
from matplotlib import font_manager

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier


class deepmodel(nn.Module):
    def __init__(self,classnum, device):
        super(deepmodel, self).__init__()
        print("Model creation...")
        self.classnum = classnum
        self.NLLloss = torch.nn.NLLLoss(reduction='none')
        self.CEloss = torch.nn.CrossEntropyLoss(reduction='none')
        self.device = 'cuda:' + args['device']

        # self.embedding = nn.Embedding(args['vocabularySize'], args['embeddingSize']).to(self.device)
        d_model = args['embeddingSize']
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.leaky_relu = nn.LeakyReLU()
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=args['dropout'])
        self.z22z3 = nn.Linear(args['hiddenSize'], args['hiddenSize']).to(self.device)
        self.z32z4 = nn.Linear(args['hiddenSize'], args['hiddenSize']).to(self.device)

        self.x2z = nn.Linear(args['maxLength'], args['hiddenSize']).to(self.device)
        self.z2z = nn.Linear(args['hiddenSize'], args['hiddenSize']).to(self.device)
        self.z2z2 = nn.Linear(args['hiddenSize'], args['hiddenSize']).to(self.device)
        self.z2z3 = nn.Linear(args['hiddenSize'], args['hiddenSize']).to(self.device)

        # print(self.x2z.weight.get_device())
        self.age2z = nn.Linear(d_model, 50).to(self.device)
        self.gender2z = nn.Linear(d_model, 50).to(self.device)
        # self.DiseaseClassifier = nn.Sequential(
        #     nn.Linear(args['hiddenSize'],2),
        #     nn.LogSoftmax(dim=-1)
        # ).to(self.device)
        self.DiseaseClassifier = nn.Linear(args['hiddenSize'], self.classnum).to(self.device)
        self.gender_embeddings = nn.Embedding(2, args['embeddingSize'])

        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(100)])  # age <=100
        # 偶数列使用sin，奇数列使用cos
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        position_encoding = torch.from_numpy(position_encoding)
        self.age_embeddings = nn.Embedding(100, d_model)
        self.age_embeddings.weight = nn.Parameter(position_encoding.float(), requires_grad=False)
        self.layer_norm = nn.LayerNorm(args['maxLength'])
        self.layer_norm2 = nn.LayerNorm(args['hiddenSize'])
        # self.linformer = Linformer(args['maxLength']).to(self.device)
        self.softmask = nn.Parameter(torch.randn(args['maxLength'], 2)).to(self.device)
        self.output_mid_value = [self.softmask]

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape).to(self.device)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature=1.0):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y_hard = (y_hard - y).detach() + y
        return y_hard, y

    def build(self, x):
        '''
        :return:
        '''

        # print(x['enc_input'])
        self.encoderInputs = x.to(self.device)
        batch_size = len(x)
        # select_probs = torch.stack([1-self.sigmoid(self.softmask), self.sigmoid(self.softmask)])
        # select_probs = select_probs.transpose(0,1)
        # cumulate = torch.zeros_like(self.softmask)
        # for i in range(20):
        #     selected,_ = self.gumbel_softmax(self.softmask) # maxlen
        #     cumulate += selected
        # cumulate = torch.clamp(cumulate, min=0.0,max=1.0)
        # print(selected)
        if False:
            # if self.training:
            batch_softmask = self.softmask[None, :, :].repeat(batch_size, 1, 1)
            self.selected, _ = self.gumbel_softmax(batch_softmask)  # batch maxlen 2
            select_probs = self.softmax(self.softmask)

            # else:
            #     select_probs = self.softmax(self.softmask)
            #     self.selected = (select_probs >0.5).int()

            self.encoderInputs = self.encoderInputs * self.selected[:, :, 1]
            # print('enc input suze: ', self.encoderInputs.size())
            #    self.encoderInputs = x['gene_input'].to(self.device) / 100
            # self.age = x['age'].to(self.device)
            # self.gender = x['gender'].to(self.device)
            # self.age_emb = self.age_embeddings(self.age)
            # self.gender_emb = self.gender_embeddings(self.gender)

            I_x_z = torch.mean(-torch.log(select_probs[:, 0] + 1e-6))
        else:
            I_x_z = 0
        self.batch_size = self.encoderInputs.size()[0]

        # xx = self.linformer(self.encoderInputs)
        # self.encoderInputs = xx
        # print(self.encoderInputs.size(), args['maxLength'])
        s_w_feature = self.x2z(self.layer_norm(self.encoderInputs))
        z = self.z2z(self.relu(s_w_feature))
        z = self.dropout(z)
        z1 = self.z2z2(self.layer_norm2(z + s_w_feature))
        z2 = z1 + self.z2z3(self.relu(z1))
        z2 = self.dropout(z2)
        z3 = self.z22z3(z2)
        z4 = z3 + self.z32z4(self.relu(z3))
        # age_f = self.age2z(self.age_emb.float())
        # gender_f = self.gender2z(self.gender_emb)
        # print(s_w_feature.get_device(), self.z2z.weight.get_device())
        #
        output = self.DiseaseClassifier(self.relu(z4)).to(self.device)  # batch chargenum
        # output = self.DiseaseClassifier(self.relu(z)).to(self.device)  # batch chargenum
        outprob = self.softmax(output)

        return outprob, (output, 10 * I_x_z)

    def forward(self, x):
        _, output = self.build(x)
        # output = self.softmax(output/10)
        return output

    def GetSelectRatio(self):
        return self.selected[:, 1].sum() / self.selected.size(0)

    def predict(self, x):

        output = self.build(x)
        # print(output)
        choose = torch.argmax(output, dim=1)
        # print(output, xmax, maxclass, choose, x['labels'])
        return choose

    def predict_proba(self, x):
        output = self.build(x)

        return output.detach().cpu()


def train(X_train, y_train, X_valid, y_valid, nclass, feature_names, human_diseasename_list=None):
    X_train_gene = X_train
    X_valid_gene = X_valid
    args['maxLength'] = X_train.size(1)
    print(X_train_gene)
    print(y_train)
    print(X_valid_gene)
    print(y_valid)
    assert len(feature_names) == args['maxLength']

    train_data_tuple = [(train_x, train_y) for
                        train_x, train_y in zip(X_train_gene, y_train)]
    test_data_tuple = [(test_x, test_y) for
                       test_x, test_y in zip(X_valid_gene, y_valid)]
    print(train_data_tuple[0][0].size(), len(test_data_tuple))
    train_loader = torch.utils.data.DataLoader(train_data_tuple, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data_tuple, batch_size=32, shuffle=True)

    regr = make_pipeline(
        # StandardScaler(),
        # SGDClassifier(loss='log_loss', penalty='elasticnet', alpha=0.0001, l1_ratio=0.15, random_state=42)
        # LogisticRegression(verbose=1)
        GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,     max_depth=1, random_state=0)
        # SGDClassifier(loss='log_loss',  alpha=0.0001, l1_ratio=0.15, random_state=42)
    # PCA(n_components=60),
    #     RandomForestClassifier(n_estimators=100)
    )
    regr.fit(X_train_gene, y_train)

    # model = AdversarialTrainingClassifier(
    #     estimator=deepmodel,
    #     n_estimators=10,
    #     estimator_args={'classnum': nclass},
    #     cuda=[args['device']],
    # )
    # # Set the criterion
    # criterion = nn.CrossEntropyLoss()
    # # crit = lambda org_output, target: criterion(org_output[0], target) + 0.01* org_output[1]
    # focal_loss = FocalLoss(nclass, gamma=2)
    # model.set_criterion(focal_loss)
    #
    # # Set the optimizer
    # model.set_optimizer('Adam', lr=1e-3, weight_decay=5e-4)
    #
    # # Train and Evaluate
    # model.fit(
    #     train_loader,
    #     epochs=200,
    #     test_loader=test_loader,
    #     save_dir=args['rootDir'],
    # )

    importances = regr[-1].feature_importances_
    feature_importance_pairs = list(zip(feature_names, importances))

    # 根据重要性降序排序
    sorted_feature_importances = sorted(feature_importance_pairs, key=lambda x: x[1], reverse=True)

    for feature, importance in sorted_feature_importances[:100]:
        print(f"{feature}: {importance:.4f}")


    all_probs = regr.predict_proba(X_valid_gene)
    # all_probs = model(X_valid_gene)
    all_probs = torch.Tensor(all_probs).detach().cpu()
    all_auc = []
    rows, cols = 2, 3
    fig, ax = plt.subplots(rows, cols)
    thres = [0 for _ in range(nclass)]
    for i in range(1, nclass):
        print('====================' + human_diseasename_list[i] + ' start ===================')
        prob_y = [(prob, y) for prob, y in zip(all_probs, y_valid) if y == 0 or y == i]
        # print(prob_y)
        C_y_valid = [(0 if b == 0 else 1) for a, b in prob_y]
        C_y_probs = torch.stack([a for a, b in prob_y])[:, i]

        auc, best_thres = subdraw_roc_by_proba(ax[int(i / cols)][i % cols], C_y_valid, C_y_probs,
                                   name=human_diseasename_list[i])
        if not best_thres:
            best_thres = 0.5
        print(human_diseasename_list[i] + ':', auc)

        thres[i] = best_thres
        C_y_probs_cali = Calibrate(best_thres, C_y_probs)
        print('0: ,', [p.item() for p, l in zip(C_y_probs_cali, C_y_valid) if l == 0])
        print('1: ,', [p.item() for p, l in zip(C_y_probs_cali, C_y_valid) if l == 1])
        all_auc.append(auc)
        print('######################' + human_diseasename_list[i] + ' end #####################')
    # plt.subplot(2,2,nclass)
    pan_y_probs = 1 - all_probs[:, 0]
    pan_y_valid = [(0 if b == 0 else 1) for b in y_valid]

    auc, best_thres = subdraw_roc_by_proba(ax[0][0], pan_y_valid, pan_y_probs, name='pan cancer')
    if not best_thres:
        best_thres = 0.5
    thres[0] = best_thres
    pan_y_probs_cali = Calibrate(best_thres, pan_y_probs)
    print('0: ,', [p.item() for p, l in zip(pan_y_probs_cali, pan_y_valid) if l == 0])
    print('1: ,', [p.item() for p, l in zip(pan_y_probs_cali, pan_y_valid) if l == 1])
    print('pan cancer:', auc)
    all_auc.append(auc)
    fig.tight_layout()
    fig.savefig(args['rootDir'] + '/multi_total_roc.png', bbox_inches='tight', dpi=150)
    plt.show()
    for a in all_auc:
        print(a)

    for i,b_thres in enumerate(thres):
        all_probs[:,i] = Calibrate(b_thres, all_probs[:,i])
    all_probs[:, 0] = 1 - all_probs[:,i]
    y_pred = torch.argmax(all_probs, dim=-1)
    cm = confusion_matrix(np.asarray(y_valid), np.asarray(y_pred))
    cm = cm / cm.sum(1)[:, None]
    print(cm)
    confusion_matrix_plot(cm, human_diseasename_list, filename='real_multi')

    print(human_diseasename_list)
    return regr


def draw_roc_by_proba(y_valid, gbm_y_proba, name=''):
    fig = plt.figure(figsize=(5, 5))
    gbm_auc = roc_auc_score(y_valid, gbm_y_proba)  # 计算auc
    gbm_fpr, gbm_tpr, gbm_threasholds = roc_curve(y_valid, gbm_y_proba)  # 计算ROC的值
    plt.title("roc_curve of %s(AUC=%.4f)" % (name, gbm_auc))
    plt.xlabel('1- Specificity(False Positive)')  # specificity = 1 - np.array(gbm_fpr))
    plt.ylabel('Sensitivity(True Positive)')  # sensitivity = gbm_tpr
    plt.plot(list(np.array(gbm_fpr)), gbm_tpr)
    # plt.gca().invert_xaxis()  # 将X轴反转
    fig.savefig(args['rootDir'] + name + '_roc.png', bbox_inches='tight', dpi=150)
    plt.show()
    return gbm_auc

def Calibrate(b_thres, probs):
    less_mask = probs < b_thres
    less_value = probs / b_thres * 0.5
    more_value = 1 - (1 - probs) / (1 - b_thres) * 0.5
    res = less_value * less_mask.float() + more_value * (1 - less_mask.float())
    return res

def subdraw_roc_by_proba(ax, y_valid, gbm_y_proba, name='', c='b'):
    # fig = plt.figure(figsize=(5, 5))
    gbm_auc = roc_auc_score(y_valid, gbm_y_proba)  # 计算auc
    gbm_fpr, gbm_tpr, gbm_threasholds = roc_curve(y_valid, gbm_y_proba)  # 计算ROC的值
    ax.set_title("roc_curve of %s(AUC=%.4f)" % (name, gbm_auc), fontsize=5)
    ax.set_xlabel('1- Specificity(False Positive)', fontsize=5)  # specificity = 1 - np.array(gbm_fpr))
    ax.set_ylabel('Sensitivity(True Positive)', fontsize=5)  # sensitivity = gbm_tpr
    # ax.xticks(fontsize=5)
    # ax.yticks(fontsize=5)
    print('x: ', gbm_fpr)
    print('y: ', gbm_tpr)
    ax.plot(list(np.array(gbm_fpr)), gbm_tpr, c)
    ax.fill_between(list(np.array(gbm_fpr)), y1=gbm_tpr, color=c, alpha=0.5)
    # plt.gca().invert_xaxis()  # 将X轴反转
    # fig.savefig(args['rootDir']+name + '_roc.png', bbox_inches='tight', dpi=150)
    # plt.show()
    best_threshold = None
    best_j = 0
    for i in range(len(gbm_fpr)):
        j = gbm_tpr[i] - gbm_fpr[i]
        print(j,gbm_tpr[i], gbm_fpr[i], gbm_threasholds[i])
        if j > best_j:
            best_j = j
            best_threshold = gbm_threasholds[i]
    print("best thres:" , best_threshold)
    return gbm_auc,best_threshold


def confusion_matrix_plot(cfm, human_diseasename_list=None, save_dir=None, filename=None):
    import seaborn as sns
    fig = plt.figure(figsize=(20, 20))
    ax = sns.heatmap(cfm, annot=True, fmt='.2%', cmap='Blues')

    ax.set_title('Cancer Confusion Matrix\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')

    labels = human_diseasename_list if human_diseasename_list else []
    ax.xaxis.set_ticklabels(labels, rotation=45, ha='right')
    ax.yaxis.set_ticklabels(labels, rotation=0, ha='right')
    if save_dir:
        fig.savefig(save_dir, bbox_inches='tight', dpi=150)
    else:
        fig.savefig(args['rootDir'] + '/' + filename + '_cm.png', bbox_inches='tight', dpi=150)
    ## Display the visualization of the Confusion Matrix.
    plt.show()


def blind_test(model, test_rpkmt, ans_label, human_diseasename_list, separate=False):
    '''
    if separate == True then human_diseasename_list is focus_list
    else human_diseasename_list
    '''
    # print(test_rpkmt.values,type(test_rpkmt.values))
    test_data = torch.FloatTensor(test_rpkmt.values) / 100
    if separate:
        probs_on_each_model = []
        for m in model:
            # m=m.to(self.device)
            y_prob = m(test_data)
            probs_on_each_model.append(y_prob)
            m = m.to('cpu')

        probs_on_each_model = torch.stack(probs_on_each_model)[:, :, 1]
        ans = torch.argmax(probs_on_each_model, dim=0)
        # print(probs_on_each_model.size(),probs_on_each_model,ans)

        for idx, (a, al) in enumerate(zip(ans, ans_label)):
            # print(a)
            print(human_diseasename_list[a], 'Gold Label: ', al)
            for d_idx, dis in enumerate(human_diseasename_list):
                print(dis, ' : ', probs_on_each_model[d_idx, idx].item())
            print()

    else:
        y_prob = model(test_data)
        ans = torch.argmax(y_prob, dim=1)
        for a, la, yp in zip(ans, ans_label, y_prob):
            print(human_diseasename_list[a], 'Gold Label: ', la)
            for h, prob in zip(human_diseasename_list, yp):
                print(h, ' : ', prob.item())

def main():
    # 数据分割
    tsv_dir = '/home/siweideng/OxTium_cfDNA'
    dataset = ChromosomeDataset(tsv_dir)
    print(tsv_dir)
    X = dataset.processed_data
    y = dataset.labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    train(X_train, y_train, X_test, y_test, 2, human_diseasename_list=['healthy','disease'])