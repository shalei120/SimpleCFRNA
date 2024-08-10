import os,sys

from sklearn.model_selection import train_test_split
from RNAdata import receive_any_task,RNAseq_Record

from Hyperparameters import args
from model_real import train
import torch
import numpy as np
import pandas as pd

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g')
cmdargs = parser.parse_args()

if cmdargs.gpu is None:
    args['device'] = 'cpu'
else:
    args['device'] = str(cmdargs.gpu)


if __name__ == '__main__':
    RPKMT_df = receive_any_task(rpkm_filename='/mnt/data/dms/raw_data/20240808_norm',
                                info_filename='./Sample_information_calibration.xlsx',
                                rpkmt_filename='./artifacts/alldata.csv',
                                DictInfoColumnRename={'disease': 'disease1', 'quantile': 'disease', 'sample_ID': 'Run'},
                                HealthyLabel='control',
                                rpkm_delimiter='\t',
                                GeneColumn='Geneid',
                                merge_rpkm=True,
                                T=RNAseq_Record.tranpose,
                                other_features = ['age'],
                                info_filename_sheetname='Sheet4')

    RPKMT_df = RPKMT_df[RPKMT_df['disease'].isin(['upper', 'lower'])]
    diseases_list = sorted(list(set(list(RPKMT_df.disease))))
    # health_index = -1
    # for i, _ in enumerate(diseases_list):
    #     if '健康' in diseases_list[i]:
    #         health_index = i
    #         break
    # assert health_index != -1
    # diseases_list[health_index], diseases_list[0] = diseases_list[0], diseases_list[health_index]
    # print(diseases_list)

    RPKMT_df['disease'] = pd.Categorical(RPKMT_df['disease'], categories=diseases_list).codes
    y_rna = list(RPKMT_df['disease'])
    RPKMT_df = RPKMT_df.drop(['Run','region','disease'], axis=1)
    RNA_names = list(RPKMT_df.columns)

    X_rna = torch.Tensor(RPKMT_df.values.astype(float))
    X_train, X_test, y_train, y_test = train_test_split(X_rna, y_rna, test_size=0.4, random_state=42)
    model = train(X_train, y_train, X_test, y_test, len(diseases_list),RNA_names, human_diseasename_list=diseases_list)