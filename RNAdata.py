import functools
print = functools.partial(print, flush=True)
import pandas as pd
from Hyperparameters import args
import argparse
import numpy as np
import datetime,time
import statsmodels.api as sm
from tqdm import tqdm
import json,torch

import sys
import os


class RNAseq_Record:
    def __init__(self):
        self.record_file = '../qc统计20240130.xlsx'
        self.cfRNA_dir = '/home/siweideng/OxTium_data/'
        self.focus_6G_list = ['健康6G', '结直肠癌6G', '胃癌6G', '肺癌6G', '肝癌6G']
        # self.focus_6G_list = ['健康6G', '胰腺癌6G']
        self.focus_1G_list = ['健康1G', '结直肠癌1G', '胃癌1G', '肺癌1G', '肝癌1G']
        self.pklfile = args['rootDir'] + '/' + args['id'] + '_real2_'+ args['amount'] +'_T.csv'
        self.Zh2En = {'健康': 'Healthy', '结直肠癌': 'Colorectal Cancer',
                      '胃癌': 'Stomach Cancer', '胰腺癌': 'Pancreatic Cancer',
                      '肺癌': 'Lung Cancer', '肝癌': 'Liver Cancer'}


    @classmethod
    def tranpose(cls, fpm):
        fpmt = fpm.T
        # fpmt.drop(index=0)

        fpmt = fpmt.reset_index()
        fpmt.columns = fpmt.iloc[0]
        fpmt = fpmt.drop(index=0)
        return fpmt

    @classmethod
    def process_RNA_sample_number(cls, name):
        name = name.split('_')[0]

        return name

    @classmethod
    def Preprocess_other_features(cls, feature_df):
        for c in feature_df.columns:
            if c == 'age':
                feature_df[c] = feature_df[c] / 100

        return feature_df

    @classmethod
    def To_Ranking_feature(cls, geneinfo):
        total_gene_num = len(geneinfo.columns)
        no_people = geneinfo.drop(labels=['Run'], axis=1)
        # generank = pd.DataFrame(
        #     np.argsort(no_people.values, axis=1),
        #     index=no_people.index,
        #     columns=no_people.columns
        # )
        # # generank = pd.DataFrame(
        # #     np.argsort(generank.values, axis=1),
        # #     index=generank.index,
        # #     columns=generank.columns
        # # ) # twice for rank index
        # generank = generank - total_gene_num + 200
        # generank[generank < 0] = 0
        generank = no_people
        generank = generank / 100

        generank['Run'] = geneinfo['Run']

        # catres = pd.merge(geneinfo[['Run', 'disease', 'region']],generank, on="Run")
        return generank

def receive_any_task(rpkm_filename, info_filename, rpkmt_filename, DictInfoColumnRename, HealthyLabel, GeneColumn,
                         T, rpkm_delimiter=',', other_features=[], merge_rpkm=False, Test = False, info_filename_sheetname = None):
    '''
    output columns: Run, disease, region, [other features], cfrna genes
    '''
    if merge_rpkm:
        assert os.path.isdir(rpkm_filename)
        files = os.listdir(rpkm_filename)
        fpms = []
        for f in files:
            if f[-3:] == 'tsv':
                fpm = pd.read_csv(rpkm_filename +'/'+ f, delimiter=rpkm_delimiter)
                fpms.append(fpm)
        fpm = fpms[0]
        for i in range(1, len(fpms)):
            fpm = pd.merge(fpm, fpms[i], on="Geneid")
        rpkm = fpm
        print('rpkm ', rpkm)
    else:
        print('read from rpkm file')
        rpkm = pd.read_csv(rpkm_filename, delimiter=rpkm_delimiter)

    rpkm = rpkm.rename(columns={GeneColumn: 'Run'})
    rpkmt = T(rpkm)
    rpkmt['Run'] = rpkmt['Run'].apply(RNAseq_Record.process_RNA_sample_number)
    # rpkmt = RNAseq_Record.To_Ranking_feature(rpkmt)
    merged_df = rpkmt
    print('rpkmt',rpkmt)
    if not Test:
        if info_filename[-4:] == 'xlsx':
            info = pd.read_excel(info_filename, sheet_name = info_filename_sheetname)
        else:
            info = pd.read_csv(info_filename, delimiter='\t')
        info = info.rename(columns=DictInfoColumnRename)
        info['disease'] = info['disease'].replace(HealthyLabel, '健康')
        print('info',info)
        merged_df = pd.merge(merged_df, info[['Run', 'disease']+other_features], on='Run')
        merged_df = RNAseq_Record.Preprocess_other_features(merged_df)
    merged_df['region'] = '津渡'
    merged_df.to_csv(rpkmt_filename, index=False)
    print('merged_df:',merged_df)
    return merged_df

