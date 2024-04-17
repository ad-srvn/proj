def dataaa(typee,e):    
    import os
    import numpy as np
    import torch
    from torch.utils.data import Dataset

    class NpyDataset(Dataset):
        def __init__(self, data_dir, transform=None):
            self.data_dir = data_dir
            self.transform = transform
            self.data_files = sorted(os.listdir(data_dir))
            self.labels = self._load_labels()

        def _load_labels(self):
            labels = []
            for filename in self.data_files:
                if "spo" in filename:
                    labels.append(1)
                elif "bon" in filename:
                    labels.append(0)
                else:
                    raise ValueError("Unknown label type")
            return labels

        def __len__(self):
            return len(self.data_files)

        def __getitem__(self, idx):
            data_path = os.path.join(self.data_dir, self.data_files[idx])
            data = np.load(data_path)

            if self.transform:
                data = self.transform(data)

            label = self.labels[idx]
            return data, label

    # Example usage:
    # Define transform (if any)
    transform = None  # You can define transforms here if needed
        
    # Path to your dataset directory containing .npy files
    data_dir = "/Volumes/Seagate/mese/"+typee+"/"+e+"/"

    # Create an instance of NpyDataset
    dataset = NpyDataset(data_dir, transform)
    return dataset