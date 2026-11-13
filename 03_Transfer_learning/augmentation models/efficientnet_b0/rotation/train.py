import os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# MODEL: EfficientNet-B0 | AUGMENTATION: Rotation (±5 degrees)
PROJECT_ROOT = r"C:\Users\prave\Desktop\college projects\biometric-from-camera-capture-handwritten-documents"
CSV_PATH = os.path.join(PROJECT_ROOT, "patch_dataset", "metadata.csv")
BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, SEED = 16, 10, 0.0001, 42
HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_PATH = os.path.join(OUTPUT_DIR, "efficientnet_b0_rotation_writer_classifier.pth")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, "| Augmentation:", "Rotation (±5 degrees)")

df = pd.read_csv(CSV_PATH)
df = df[df["kept"] == 1].copy()
df["filepath"] = df["patch_path"].apply(lambda p: os.path.normpath(os.path.join(PROJECT_ROOT, p)))
df = df[df["filepath"].apply(os.path.isfile)].copy()
if df.empty: raise ValueError("No patch images found; check PROJECT_ROOT and metadata.csv.")

writers = sorted(df["writer"].unique())
writer_to_idx = {w:i for i,w in enumerate(writers)}
idx_to_writer = {i:w for w,i in writer_to_idx.items()}
num_classes = len(writers)

# Keep the original test pages; split each writer's original train pages into train/validation.
train_pool = df[df["split"]=="train"].copy()
test_df = df[df["split"]=="test"].copy()
train_parts, val_parts = [], []
for writer, group in train_pool.groupby("writer"):
    pages = group["page"].drop_duplicates().tolist()
    if len(pages) < 2: raise ValueError(f"Writer {writer} has fewer than 2 training pages.")
    train_pages, val_pages = train_test_split(pages, test_size=1, random_state=SEED)
    train_parts.append(group[group["page"].isin(train_pages)])
    val_parts.append(group[group["page"].isin(val_pages)])
train_df = pd.concat(train_parts, ignore_index=True)
val_df = pd.concat(val_parts, ignore_index=True)
if train_df.empty or val_df.empty or test_df.empty: raise ValueError("A data split is empty.")
print("Train:",len(train_df),"Validation:",len(val_df),"Test:",len(test_df),"Writers:",num_classes)

norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
# Augmentation is applied ONLY to training images.
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomRotation(degrees=5),,
    transforms.ToTensor(), norm
])
eval_transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),norm])

class WriterDataset(Dataset):
    def __init__(self, frame, transform):
        self.df=frame.reset_index(drop=True); self.transform=transform
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        row=self.df.iloc[i]
        with Image.open(row["filepath"]) as im: image=im.convert("RGB")
        return self.transform(image),torch.tensor(writer_to_idx[row["writer"]],dtype=torch.long)

train_loader=DataLoader(WriterDataset(train_df,train_transform),batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
val_loader=DataLoader(WriterDataset(val_df,eval_transform),batch_size=BATCH_SIZE,shuffle=False,num_workers=0)

# Same EfficientNet-B0 transfer-learning setup as the supplied baseline:
# freeze all pretrained parameters and train only classifier[1].
model=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
for p in model.parameters(): p.requires_grad=False
model.classifier[1]=nn.Linear(model.classifier[1].in_features,num_classes)
for p in model.classifier[1].parameters(): p.requires_grad=True
model=model.to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(filter(lambda p:p.requires_grad,model.parameters()),lr=LEARNING_RATE)

history={"Epoch":[],"Train Loss":[],"Validation Loss":[],"Train Accuracy":[],"Validation Accuracy":[]}
for epoch in range(NUM_EPOCHS):
    model.train(); loss_sum=correct=total=0
    for x,y in train_loader:
        x,y=x.to(device),y.to(device); optimizer.zero_grad()
        out=model(x); loss=criterion(out,y); loss.backward(); optimizer.step()
        loss_sum+=loss.item()*x.size(0); correct+=(out.argmax(1)==y).sum().item(); total+=y.size(0)
    tr_loss=loss_sum/total; tr_acc=100*correct/total

    model.eval(); loss_sum=correct=total=0
    with torch.no_grad():
        for x,y in val_loader:
            x,y=x.to(device),y.to(device); out=model(x); loss=criterion(out,y)
            loss_sum+=loss.item()*x.size(0); correct+=(out.argmax(1)==y).sum().item(); total+=y.size(0)
    va_loss=loss_sum/total; va_acc=100*correct/total
    for k,v in [("Epoch",epoch+1),("Train Loss",tr_loss),("Validation Loss",va_loss),("Train Accuracy",tr_acc),("Validation Accuracy",va_acc)]: history[k].append(v)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss {tr_loss:.4f} Acc {tr_acc:.2f}% | Val Loss {va_loss:.4f} Acc {va_acc:.2f}%")

hist=pd.DataFrame(history)
hist.to_csv(os.path.join(OUTPUT_DIR,"training_history.csv"),index=False)
for metric,ylabel,filename in [("Loss","Loss","loss_curve.png"),("Accuracy","Accuracy (%)","accuracy_curve.png")]:
    plt.figure(figsize=(8,5))
    plt.plot(hist["Epoch"],hist[f"Train {metric}"],marker="o",label="Training")
    plt.plot(hist["Epoch"],hist[f"Validation {metric}"],marker="o",label="Validation")
    plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.title(f"EfficientNet-B0 - Rotation (±5 degrees) - {metric}")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,filename),dpi=300); plt.close()

torch.save({"model_state_dict":model.state_dict(),"writer_to_idx":writer_to_idx,
            "idx_to_writer":idx_to_writer,"num_classes":num_classes,
            "model_name":"EfficientNet-B0","augmentation":"Rotation (±5 degrees)"},MODEL_PATH)
print("Training complete. Checkpoint:",MODEL_PATH)
