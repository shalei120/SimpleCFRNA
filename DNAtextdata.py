import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os,re,sys
from  tqdm import tqdm
import pickle
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

import model_real
from model_real import train
from sklearn.model_selection import train_test_split
from Hyperparameters import args
import argparse
from RNAData import receive_any_task,RNAseq_Record

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g')
cmdargs = parser.parse_args()
if cmdargs.gpu is None:
    args['device'] = 'cpu'
else:
    args['device'] = str(cmdargs.gpu)
chromosome_basepairs = {
    "1": 249_000_000,
    "2": 243_000_000,
    "3": 198_260_000,
    "4": 191_000_000,
    "5": 182_000_000,
    "6": 171_000_000,
    "7": 160_000_000,
    "8": 146_000_000,
    "9": 141_000_000,
    "10": 136_000_000,
    "11": 135_090_000,
    "12": 134_000_000,
    "13": 115_000_000,
    "14": 107_000_000,
    "15": 102_000_000,
    "16": 90_250_000,
    "17": 84_000_000,
    "18": 80_290_000,
    "19": 59_000_000,
    "20": 64_350_000,
    "21": 47_000_000,
    "22": 51_000_000,
    "X": 156_050_000,
    "Y": 57_000_000
}

print(args)
class ChromosomeDataset(Dataset):
    def __init__(self, tsv_dir, bin_size=100000):
        self.bin_size = bin_size
        self.pkl_path = args['rootDir'] + 'name_label_data'+str(bin_size)+'.pkl'
        self.processed_data = []
        self.ids = []
        self.labels = []
        self.tsv_dir = tsv_dir
        # info_filename = tsv_dir + '/cfDNA_LG_FJ_pheno.xlsx'
        info_filename = './cfDNA0624.xlsx'
        print('efef')

        print('here')
        if os.path.exists(self.pkl_path):
            with open(self.pkl_path, 'rb') as f:
                loaded_dict = pickle.load(f)
                self.processed_data = loaded_dict['feature']
                self.ids = loaded_dict['id']
                self.labels = loaded_dict['label']

            print("Data loaded from binary file.")
        else:
            # 读取Excel文件
            info_df = pd.read_excel(info_filename, sheet_name='Sheet1')
            info_df['样本编号'] = info_df['样本编号'].apply(self.process_sample_number)
            if info_df['样本编号'].is_unique:
                name_label_dict = dict(zip(info_df['样本编号'], info_df['样本类型']))
            else:
                print(
                    "Warning: 'name' column contains duplicate values. The dictionary will only keep the last occurrence of each name.")
                name_label_dict = dict(zip(info_df['样本编号'], info_df['样本类型']))


            datafile_list = os.listdir(tsv_dir)

            for tsv_file in tqdm(datafile_list):
                if 'xlsx' not in tsv_file:
                    checkkey = tsv_file.split('.')[0]
                    self.ids.append(checkkey)
                    self.labels.append(1 if name_label_dict[checkkey]=='疾病' else 0)

            self.ordered_parallel_process_files(datafile_list)

            self.processed_data = np.array(self.processed_data)
            self.labels = np.array(self.labels)
            with open(self.pkl_path, 'wb') as f:
                pickle.dump({'feature':self.processed_data,'id':self.ids, 'label': self.labels}, f)

            print("Data saved to binary file.")

        # total_gene_num = self.processed_data.shape[1]
        # generank = np.argsort(self.processed_data, axis=1)
        # generank = np.argsort(generank, axis=1)  # twice for rank index
        # generank = generank - total_gene_num + 200
        # generank[generank < 0] = 0
        self.processed_data = torch.tensor(self.processed_data, dtype=torch.float32)
    @staticmethod
    def process_sample_number(sample_number):
        # 使用正则表达式替换 '+数字-' 为空字符串
        sample_number = str(sample_number)
        if '（' in sample_number:
            sample_number = re.sub(r'（.*?）', '', sample_number)
            print(sample_number)
        # if sample_number[-2:] == '-T':
        #     sample_number = sample_number[:-2]
        sample_number = re.sub(r'\+\d+', '', sample_number)

        subname = sample_number.split('_')
        changename = sample_number.replace('_', '-')
        if len(subname) == 4:
            if subname[0].isdigit() and subname[1].isdigit() and subname[2].isdigit() and subname[3].isdigit():
                changename = '-'.join([subname[0], subname[1], subname[3]])
        return changename

    def process_file(self, tsv_file):
        if 'xlsx' not in tsv_file:
            data = pd.read_csv(os.path.join(self.tsv_dir, tsv_file),
                               sep='\t', header=None,
                               names=['chromosome', 'position', 'ignored'])
            return self.process_data(data)
        return None

    def ordered_parallel_process_files(self, datafile_list):
        # 确定使用的CPU核心数
        num_workers = min(32, os.cpu_count() or 1)  # 限制最大使用32个核心

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # 创建一个任务列表，但不立即开始执行
            futures = [executor.submit(self.process_file, tsv_file)
                       for tsv_file in datafile_list if 'xlsx' not in tsv_file]

            # 按照原始顺序处理结果
            for future in tqdm(futures, total=len(futures), desc="Processing files"):
                result = future.result()  # 这会阻塞直到任务完成
                if result is not None:
                    self.processed_data.append(result)

    def process_data(self, data):
        chromosome_vectors = []
        # print(data['chromosome'].unique())
        chrom2bin = {}
        for key,value in chromosome_basepairs.items():
            max_position = value
            # print(max_position)
            num_bins = max_position // self.bin_size + 1
            bins = np.zeros(num_bins, dtype=int)
            chrom2bin[key] = bins
        print(chrom2bin.keys())

        for _, row in data.iterrows():
            bin_index = row['position'] // self.bin_size
            try:
                chrom2bin[row['chromosome'][3:]][bin_index] += 1
            except:
                print(row['chromosome'][3:], bin_index, row['position'])
                chrom2bin[row['chromosome'][3:]][bin_index] += 1


        for key,value in chromosome_basepairs.items():
            chromosome_vectors.append(chrom2bin[key])

        return np.concatenate(chromosome_vectors)

    def __len__(self):
        return len(self.processed_data)  # We're treating the entire processed data as one sample

    def __getitem__(self, idx):
        return self.processed_data[idx]



def create_dataloader(tsv_dir, batch_size=1, bin_size=100000):
    dataset = ChromosomeDataset(tsv_dir, bin_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def process_RNA_sample_number(name):
    name = name.split('.')[0][5:]

    subname = name.split('_')
    changename = name.replace('_','-')
    if len(subname) == 4:
        if subname[0].isdigit() and subname[1].isdigit() and subname[2].isdigit() and subname[3].isdigit():
            changename = '-'.join([subname[0],subname[1],subname[3]])
    return changename

def GetDNANames(bin_size=1000):
    DNAnames = []
    for key, value in chromosome_basepairs.items():
        max_position = value
        # print(max_position)
        num_bins = max_position // bin_size + 1
        start = 1
        for i in range(num_bins):
            name = key + ':'+str(1+i*bin_size) + '-'+str(min(1+(i+1)*bin_size-1, max_position))
            # print(name)
            DNAnames.append(name)

    return DNAnames

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

def Calibrate(b_thres, probs):
    less_mask = probs < b_thres
    less_value = probs / b_thres * 0.5
    more_value = 1 - (1 - probs) / (1 - b_thres) * 0.5
    res = less_value * less_mask.astype(float) + more_value * (1 - less_mask.astype(float))
    return res

def drawEvery(testY, all_probs):

    fig, ax = plt.subplots(1, 1)
    y_probs = 1 - all_probs[:, 0]
    auc, best_thres = subdraw_roc_by_proba(ax, testY, y_probs, name='pan cancer')

    pan_y_probs_cali = Calibrate(best_thres, y_probs)
    print('0: ,', [p.item() for p, l in zip(pan_y_probs_cali, testY) if l == 0])
    print('1: ,', [p.item() for p, l in zip(pan_y_probs_cali, testY) if l == 1])

    print('AUC=', auc)


def test(DNA_dict, RNA_dict, model, ID_train):
    DNA_info_filename = './cfDNA_孕早期.xlsx'
    RNA_info_filename = './cfRNA_孕早期.xlsx'
    ID_train_set = set(ID_train)

    info_df = pd.read_excel(DNA_info_filename, sheet_name='Sheet1')
    # print(list(info_df['Sample']))
    info_df['Sample'] = info_df['Sample'].apply(ChromosomeDataset.process_sample_number)
    if info_df['Sample'].is_unique:
        name_label_dict = dict(zip(info_df['Sample'], info_df['Type']))
    else:
        print(
            "Warning: 'name' column contains duplicate values. The dictionary will only keep the last occurrence of each name.")
        name_label_dict = dict(zip(info_df['Sample'], info_df['Type']))

    # testX_dna,testY_dna = [],[]
    # for sample, type in name_label_dict.items():
    #     if sample in DNA_dict['DNA_id2ind'] and sample not in ID_train_set:
    #         ID = DNA_dict['DNA_id2ind'][sample]
    #         testX_dna.append(DNA_dict['X'][ID])
    #         testY_dna.append(DNA_dict['Y'][ID])
    #     else:
    #         print('Not exist', sample)
    # testX_dna = torch.stack(testX_dna)
    # testY_dna = torch.Tensor(testY_dna)
    # all_probs = model.predict_proba(testX_dna)
    # print(testX_dna.size(),testY_dna.size())
    #
    # drawEvery(testY_dna, all_probs)

    testX_rna, testY_rna = [], []
    info = pd.read_excel(RNA_info_filename, sheet_name='Sheet1')
    info = info.rename(columns= {'Type': 'disease', 'Sample': 'Run'})
    info['Run'] = info['Run'].apply(process_RNA_sample_number)
    # info['disease'] = info['disease'].replace(HealthyLabel, '健康' + args['amount'])
    # for sample in info['Run']:
    #     if sample in RNA_dict['RNA_id2ind'] and sample not in ID_train_set:
    #         ID = RNA_dict['RNA_id2ind'][sample]
    #         testX_rna.append(RNA_dict['X'][ID])
    #         # print(RNA_dict['Y'], len(RNA_dict['Y']))
    #         testY_rna.append(1 if RNA_dict['Y'][ID]=='CASE' else 0)
    #     else:
    #         print('Not exist', sample)
    # testX_rna = torch.stack(testX_rna)
    # testY_rna = torch.Tensor(testY_rna)
    # all_probs = model.predict_proba(testX_rna)
    # drawEvery(testY_rna, all_probs)
    #
    #
    testX_dr, testY_dr = [], []
    for sample in info['Run']:
        if sample in name_label_dict  and sample not in ID_train_set:
            ID_dna = DNA_dict['DNA_id2ind'][sample]
            ID_rna = RNA_dict['RNA_id2ind'][sample]
            testX_dr.append(torch.cat([DNA_dict['X'][ID_dna],RNA_dict['X'][ID_rna]], dim = 0))
            testY_dr.append(DNA_dict['Y'][ID_dna])
    testX_dr = torch.stack(testX_dr)
    testY_dr = torch.Tensor(testY_dr)
    print(testX_dr.size(),testY_dr.size())
    all_probs = model.predict_proba(testX_dr)
    drawEvery(testY_dr, all_probs)


if __name__ == "__main__":
    tsv_dir = '/home/siweideng/OxTium_cfDNA'
    # dataloader = create_dataloader(tsv_dir)
    DNA_names = GetDNANames(bin_size=1000)
    dataset = ChromosomeDataset(tsv_dir,bin_size=1000)
    receive_any_task(rpkm_filename = '../Huada00.data/LG_02.All_sample_reads_count.csv',
                     info_filename = '../Huada00.data/LG_03.All_sample_type.txt',
                     rpkmt_filename = '../Huada00.data/alldataLG.csv',
                     DictInfoColumnRename = {'Type': 'disease', 'Sample': 'Run'},
                     HealthyLabel ='CTRL', GeneColumn= 'gene_id',
                     T = RNAseq_Record.tranpose)
    receive_any_task(rpkm_filename = '../Huada00.data/FJ_02.All_sample_reads_count.csv',
                     info_filename = '../Huada00.data/FJ_03.All_sample_type.txt',
                     rpkmt_filename = '../Huada00.data/alldataFJ.csv',
                     DictInfoColumnRename = {'Type': 'disease', 'Sample': 'Run'},
                     HealthyLabel ='CTRL', GeneColumn= 'gene_id',
                     T = RNAseq_Record.tranpose)
    rpkmt_filename = '../Huada00.data/alldataLG.csv'
    RPKMt1 = pd.read_csv(rpkmt_filename, delimiter=',')
    RPKMt1['Run'] = RPKMt1['Run'].apply(process_RNA_sample_number)

    rpkmt_filename2 = '../Huada00.data/alldataLG.csv'
    RPKMt2 = pd.read_csv(rpkmt_filename2, delimiter=',')
    RPKMt2['Run'] = RPKMt2['Run'].apply(process_RNA_sample_number)
    RPKMt = pd.concat([RPKMt1, RPKMt2], axis=0)
    print(tsv_dir)
    print('RPKMt',RPKMt['Run'])
    ID = dataset.ids
    X = dataset.processed_data
    y = dataset.labels

    DNA_id2ind = {id:ind for ind, id in enumerate(ID)}

    # 存储未找到匹配的样本
    not_found = []
    matched = []
    search_list = set(ID)
    # 遍历DataFrame的sample列
    for ind, sample in enumerate(RPKMt['Run']):
        # print(ind,sample)
        if sample in search_list:
            matched.append(sample)
    matched = list(set(matched))
    X_dna = []
    y_dna = []
    for matchedID in matched:
        dna_index = DNA_id2ind[matchedID]
        X_dna.append(X[dna_index])
        y_dna.append(y[dna_index])

    X_dna = torch.stack(X_dna)
    y_dna = torch.Tensor(y_dna)
    print(X_dna.size(),y_dna.size(),len(matched))
    RPKMt_reordered = RPKMt.set_index('Run').loc[matched].reset_index()
    RPKMt_reordered = RPKMt_reordered.drop_duplicates(subset='Run', keep='first')
    # for i,id in enumerate(RPKMt_reordered['Run']):
    #     print('ji:',i, id)
    #
    # for i,m in enumerate(matched):
    #     print('m:',i,m)
    RNA_id2ind = {id:ind for ind, id in enumerate(RPKMt_reordered['Run'])}
    y_rna = list(RPKMt_reordered['disease'])
    RPKMt_reordered = RPKMt_reordered.drop(['Run','region','disease'], axis=1)
    print(RPKMt_reordered.columns)
    RNA_names = list(RPKMt_reordered.columns)
    X_rna = torch.Tensor(RPKMt_reordered.values)
    print(X_rna.size())
    Xdata = torch.cat([X_dna,X_rna],dim=1)
    # Xdata = X_rna
    print(len(DNA_names),X_dna.size(1),len(RNA_names) , X_rna.size(1))
    assert len(DNA_names) == X_dna.size(1)
    assert len(RNA_names) == X_rna.size(1)
    ID_train, ID_test, X_train, X_test, y_train, y_test = train_test_split(matched, Xdata, y_dna, test_size=0.4, random_state=42)
    model = train(X_train, y_train, X_test, y_test, 2, DNA_names,RNA_names, human_diseasename_list=['healthy', 'disease'])

    test({"DNA_id2ind":DNA_id2ind, 'X':X, 'Y':y}, {'RNA_id2ind':RNA_id2ind, 'X':X_rna, 'Y':y_rna}, model, ID_train)