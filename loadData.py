from torch.utils.data import Dataset
import torch
from sklearn import preprocessing
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    def __init__(self, input_mat, output_mat, is_scale=True, device='cpu'):
        # scale output to zeroscore
        if is_scale:
            self.input_scaler = preprocessing.StandardScaler().fit(input_mat)
            self.output_scaler = preprocessing.StandardScaler().fit(output_mat)
            input_mat = self.input_scaler.transform(input_mat)
            output_mat = self.output_scaler.transform(output_mat)

        # numpy to torch tensor
        self.x_data = torch.from_numpy(input_mat).to(device).float()
        self.y_data = torch.from_numpy(output_mat).to(device).float()

        # get length of pair CAD_sim_1e6
        self.len = self.x_data.shape[0]

        # get dimension of input and output CAD_sim_1e6
        self.input_dim = input_mat.shape[1]
        self.output_dim = output_mat.shape[1]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def createLoader(input_mat, output_mat,batch_size,valid_ratio,is_scale=True,device='cpu'):
    dataset = CustomDataset(input_mat, output_mat, is_scale=True, device=device)
    train_ratio = 1 - valid_ratio
    train_size = int(dataset.__len__() * train_ratio)
    test_size = dataset.__len__() - train_size
    train_dataset, validate_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              num_workers=0,
                              shuffle=True
                              )
    valid_loader = DataLoader(validate_dataset,
                              batch_size=batch_size,
                              num_workers=0,
                              shuffle=True)
    if is_scale:
        return train_loader, valid_loader, dataset.input_scaler, dataset.output_scaler
    else:
        return train_loader, valid_loader, None, None
