"""Contact sheet from training samples only; never opens validation/test images."""
from pathlib import Path
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision.transforms import functional as F
from augment import TrainingTransform, MEAN, STD, letterbox

ROOT=Path(__file__).resolve().parents[3]
random.seed(42);np.random.seed(42);torch.manual_seed(42)
df=pd.read_csv(ROOT/'artifacts/tid-v6/data/manifest.csv',encoding='utf-8-sig')
df=df[df.split=='train']
font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',18)
transform=TrainingTransform()
sheet=Image.new('RGB',(5*192,4*226),'#e8ecf0')
for row,label in enumerate(['A','B','L','Y']):
    source=Image.open(df[df.label==label].iloc[0].path).convert('RGB')
    for col in range(5):
        if col==0:im=letterbox(source)
        else:
            tensor=transform(source)
            im=F.to_pil_image((tensor*torch.tensor(STD)[:,None,None]+torch.tensor(MEAN)[:,None,None]).clamp(0,1))
        x,y=col*192,row*226
        sheet.paste(im.resize((192,192)),(x,y+32))
        ImageDraw.Draw(sheet).text((x+8,y+4),f'{label} · '+('Orijinal' if col==0 else f'Augmentation {col}'),font=font,fill='#172033')
sheet.save(ROOT/'artifacts/tid-v6/augmentation_preview.jpg',quality=92)

# Rebuild the 3D overview with a Turkish-capable font.
hands=pd.read_csv(ROOT/'artifacts/tid-v6/synthetic/manifest.csv',encoding='utf-8-sig').groupby('label',sort=False).first().reset_index()
sheet=Image.new('RGB',(8*128,4*152),'#e8ecf0')
for i,row in enumerate(hands.itertuples()):
    im=Image.open(row.path).convert('RGBA').resize((128,128))
    x,y=i%8*128,i//8*152
    sheet.paste(im,(x,y+24),im)
    ImageDraw.Draw(sheet).text((x+5,y),row.label,font=font,fill='#172033')
sheet.save(ROOT/'artifacts/tid-v6/synthetic/preview.jpg',quality=92)
print('Training-only augmentation preview and 3D overview saved.')
