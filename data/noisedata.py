import os
import pandas as pd
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import LabelEncoder
import torch
import numpy as np
#数据集加载器
class NoiseData(Dataset):
    def __init__(self, dir='../data', filename='data_after.xlsx', use_type = None, transform = None):
        self.dir = dir
        self.filename = filename
        self.use_type = use_type
        self.transform = transform
        try:
            all_sheets = pd.read_excel(os.path.join(self.dir, self.filename), sheet_name=None)
        except Exception as e:
            raise e
        self.dataFrame = pd.DataFrame()
        self.sheet_names = []
        for sheet_name, df in all_sheets.items():
            self.sheet_names.append(sheet_name)
            self.dataFrame = pd.concat([self.dataFrame, df], ignore_index=True)
        
        self.le = []
        keys = self.dataFrame.keys()
        if self.use_type:
            le = LabelEncoder()
            self.dataFrame[keys[0]] = le.fit_transform(self.dataFrame[keys[0]])
            self.le = le.classes_

    def __len__(self):
        return self.dataFrame.__len__()

    def __getitem__(self, idx):
        keys = self.dataFrame.keys()
        if self.use_type:
            input = [self.dataFrame[keys[0]][idx], self.dataFrame[keys[1]][idx], self.dataFrame[keys[2]][idx], self.dataFrame[keys[3]][idx]]
            output = [self.dataFrame[keys[len(keys)-1]][idx]]
        else:
            input = [self.dataFrame[keys[1]][idx], self.dataFrame[keys[2]][idx], self.dataFrame[keys[3]][idx]]
            output = [self.dataFrame[keys[len(keys)-1]][idx]]
        if self.transform is not None:
            input = self.transform(input)
        # Bin values
        bins = np.array(range(20, 70, 1))
        binned_output = torch.LongTensor(np.digitize(output, bins) - 1)
        input = torch.tensor(input).to(torch.float32)
        output = torch.tensor(output).to(torch.float32)
        return input, output.squeeze(), binned_output.squeeze()
    
class NoiseDataFiltered(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass