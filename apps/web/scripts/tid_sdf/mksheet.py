import sys, numpy as np
from PIL import Image, ImageDraw, ImageFont
import letters as LT, render_pair as RP
tag = sys.argv[1] if len(sys.argv)>1 else 'final'
args = set(sys.argv[2:])
compose_only = '--compose' in args
wanted = args - {'--compose'}
cols=6; W=300
tiles=[]
letters = list(LT.letters())
if not compose_only:
    for ch,L in letters:
        if wanted and ch not in wanted: continue
        hands=LT.pair(L)
        f=f'sheet/{tag}-{ch}.png'
        RP.render(hands,f,W=W,H=W)
        print('rendered', ch, flush=True)
if wanted and not compose_only:
    sys.exit(0)
for ch,_L in letters:
    f=f'sheet/{tag}-{ch}.png'
    im=Image.open(f); im.load(); tiles.append((ch,im.convert('RGB')))
rows=(len(tiles)+cols-1)//cols
sheet=Image.new('RGB',(cols*W,rows*(W+26)),(20,17,34))
try: font=ImageFont.truetype("arialbd.ttf",26)
except: font=ImageFont.load_default()
d=ImageDraw.Draw(sheet)
for i,(ch,im) in enumerate(tiles):
    r,c=divmod(i,cols)
    sheet.paste(im,(c*W,r*(W+26)+26))
    d.text((c*W+8,r*(W+26)+1),ch,fill=(255,225,150),font=font)
sheet.save(f'sheet/{tag}-alphabet.png')
print('saved',f'sheet/{tag}-alphabet.png',sheet.size)
