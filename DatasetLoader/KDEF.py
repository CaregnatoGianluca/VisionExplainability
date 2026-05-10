import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

class KDEFDataset(Dataset):
    def __init__(self, root_dir, subject_list, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # KDEF: HA = Happy, SA = Sad, NE = Neutral
        self.emotion_to_idx = {'HA': 0, 'SA': 1, 'NE': 2}

        for subj in subject_list:
            subj_path = os.path.join(root_dir, subj)
            if not os.path.isdir(subj_path):
                continue
            
            for filename in os.listdir(subj_path):
                if filename.endswith(".JPG"):
                    # Emotion code is located in the filename, typically at positions 4 and 5 (0-indexed)
                    # Esempio: BM30HAFR.JPG -> HA
                    emotion_code = filename[4:6]
                    
                    if emotion_code in self.emotion_to_idx:
                        self.image_paths.append(os.path.join(subj_path, filename))
                        self.labels.append(self.emotion_to_idx[emotion_code])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

'''

# 1. Ottieni la lista di tutte le cartelle dei soggetti
root_path = "path/alla/tua/cartella/KDEF" # Sostituisci con il tuo percorso
all_subjects = sorted([d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))])

# 2. Dividi i soggetti (es. 80% training, 20% test)
train_subjects, test_subjects = train_test_split(all_subjects, test_size=0.2, random_state=42)

# 3. Definisci le trasformazioni
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. Crea le istanze del Dataset
train_dataset = KDEFDataset(root_path, train_subjects, transform=data_transforms)
test_dataset = KDEFDataset(root_path, test_subjects, transform=data_transforms)

# 5. Crea i DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Immagini nel Training Set: {len(train_dataset)}")
print(f"Immagini nel Test Set: {len(test_dataset)}")

'''