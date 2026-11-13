import os
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# AUGMENTATION TYPE: Brightness and Contrast (0.1)
PROJECT_ROOT = r"C:\Users\prave\Desktop\college projects\biometric-from-camera-capture-handwritten-documents"
CSV_PATH = os.path.join(PROJECT_ROOT,"patch_dataset","metadata.csv")
BATCH_SIZE=16
HERE=os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR=os.path.join(HERE,"output")
MODEL_PATH=os.path.join(OUTPUT_DIR,"alexnet_brightness_contrast_writer_classifier.pth")
os.makedirs(OUTPUT_DIR,exist_ok=True)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

df=pd.read_csv(CSV_PATH); df=df[df["kept"]==1].copy()
df["filepath"]=df["patch_path"].apply(lambda p:os.path.normpath(os.path.join(PROJECT_ROOT,p)))
df=df[df["filepath"].apply(os.path.isfile)].copy()
test_df=df[df["split"]=="test"].copy()
if test_df.empty: raise ValueError("No test images found.")

checkpoint=torch.load(MODEL_PATH,map_location=device)
writer_to_idx=checkpoint["writer_to_idx"]
idx_to_writer={int(k):v for k,v in checkpoint["idx_to_writer"].items()}
num_classes=checkpoint["num_classes"]
transform=transforms.Compose([
    transforms.Resize((224,224)),transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

class WriterDataset(Dataset):
    def __init__(self,frame): self.df=frame.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        row=self.df.iloc[i]
        with Image.open(row["filepath"]) as im: image=im.convert("RGB")
        return transform(image),torch.tensor(writer_to_idx[row["writer"]],dtype=torch.long)

loader=DataLoader(WriterDataset(test_df),batch_size=BATCH_SIZE,shuffle=False,num_workers=0)
model=models.alexnet(weights=None)
model.classifier[6]=nn.Linear(model.classifier[6].in_features,num_classes)
model.load_state_dict(checkpoint["model_state_dict"]); model=model.to(device); model.eval()
criterion=nn.CrossEntropyLoss(); preds=[]; labels=[]; loss_sum=0; total=0
with torch.no_grad():
    for x,y in loader:
        x,y=x.to(device),y.to(device); out=model(x); loss=criterion(out,y)
        loss_sum+=loss.item()*x.size(0); preds.extend(out.argmax(1).cpu().numpy()); labels.extend(y.cpu().numpy()); total+=y.size(0)

test_loss=loss_sum/total
metrics={
 "model":"AlexNet","experiment":"Brightness and Contrast (0.1)","test_images":total,"test_loss":test_loss,
 "accuracy":accuracy_score(labels,preds),
 "precision_macro":precision_score(labels,preds,average="macro",zero_division=0),
 "recall_macro":recall_score(labels,preds,average="macro",zero_division=0),
 "f1_macro":f1_score(labels,preds,average="macro",zero_division=0),
 "precision_weighted":precision_score(labels,preds,average="weighted",zero_division=0),
 "recall_weighted":recall_score(labels,preds,average="weighted",zero_division=0),
 "f1_weighted":f1_score(labels,preds,average="weighted",zero_division=0)
}
print(pd.Series(metrics))
pd.DataFrame([metrics]).to_csv(os.path.join(OUTPUT_DIR,"metrics_summary.csv"),index=False)

names=[idx_to_writer[i] for i in range(num_classes)]
report=classification_report(labels,preds,labels=list(range(num_classes)),target_names=names,output_dict=True,zero_division=0)
pd.DataFrame(report).transpose().to_csv(os.path.join(OUTPUT_DIR,"classification_report.csv"),index_label="class_or_average")
with open(os.path.join(OUTPUT_DIR,"classification_report.txt"),"w",encoding="utf-8") as f:
    f.write(classification_report(labels,preds,labels=list(range(num_classes)),target_names=names,zero_division=0))

cm=confusion_matrix(labels,preds,labels=list(range(num_classes)))
pd.DataFrame(cm,index=[f"Actual_{n}" for n in names],columns=[f"Predicted_{n}" for n in names]).to_csv(os.path.join(OUTPUT_DIR,"confusion_matrix.csv"))
plt.figure(figsize=(16,14))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=names,yticklabels=names)
plt.xlabel("Predicted Writer"); plt.ylabel("Actual Writer")
plt.title("AlexNet - Brightness and Contrast (0.1) - Confusion Matrix")
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR,"confusion_matrix.png"),dpi=300,bbox_inches="tight"); plt.close()
print("Evaluation outputs saved to:",OUTPUT_DIR)
