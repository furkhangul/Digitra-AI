"""Close-up motion previews: an even sample across each letter's transition and hold."""
import json, numpy as np, sys
from PIL import Image, ImageDraw, ImageFont
import letters as LT, render_pair as RP
ROOT=r"C:/Users/Furkan/Desktop/Digitra AI"
M=json.load(open(ROOT+"/tmp/tid/motion-frames.json",encoding="utf-8"))
OUT=ROOT+"/output/hand-check"
PICK=[0,4,8,12,16,22,30,40,52]
W=260
try: font=ImageFont.truetype("arialbd.ttf",18)
except: font=ImageFont.load_default()
for L in M["letters"]:
    ch=L["ch"]; rows=[]
    for i in PICK:
        f=L["frames"][min(i,len(L["frames"])-1)]
        hands=(LT.Hand(f["frame"]["right"],"right"), LT.Hand(f["frame"]["left"],"left"))
        p=f'{OUT}/motion-{ch}-{i:02d}.png'
        RP.render(hands,p,W=W,H=W)
        im=Image.open(p); im.load(); im=im.convert('RGB')
        ImageDraw.Draw(im).text((6,4),f"{f['t']:.2f}s {f['phase']}",fill=(255,225,150),font=font)
        rows.append(im)
    s=Image.new('RGB',(W*len(rows),W+24),(18,15,30))
    d=ImageDraw.Draw(s)
    d.text((6,4),f"{ch}   (from {L['from']}, motion={L['motion']})",fill=(255,225,150),font=font)
    for i,im in enumerate(rows): s.paste(im,(i*W,24))
    s.save(f'{OUT}/motion-{ch}.png'); print('saved',ch)
