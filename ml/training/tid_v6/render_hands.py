"""Render the website's exact SDF geometry in an offscreen OpenGL context."""
import argparse
import csv
import json
from pathlib import Path
from gl_context import Renderer
import numpy as np
from PIL import Image, ImageDraw


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    data=json.loads(args.source.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True,exist_ok=True)
    fragment=data['fragment'].replace('precision highp float;','').replace('varying vec2 vUv;','in vec2 vUv;\nout vec4 fragColor;\nuniform vec3 skin;\nuniform vec3 lamp;')
    fragment=fragment.replace('vec3(-.55, .95, 1.35)','lamp').replace('mix(vec3(.96, .69, .37), vec3(.32, .12, .72), material)','skin')
    fragment=fragment.replace('gl_FragColor','fragColor').replace('#include <colorspace_fragment>','fragColor.rgb = pow(clamp(fragColor.rgb, 0.0, 1.0), vec3(1.0/2.2));')
    renderer=Renderer('#version 330\n'+fragment)
    skins=[(.65,.36,.20),(.35,.17,.08),(.85,.61,.43)]
    rows=[]
    previews=[]
    for i,frame in enumerate(data['frames']):
        for key,value in frame['uniforms'].items():
            renderer.uniform(key,value)
        for tone,skin in enumerate(skins):
            renderer.uniform('skin',skin)
            renderer.uniform('lamp',[(-.6,.9,1.3),(.65,.6,1.4),(-.2,1.2,1.0)][tone])
            image=Image.fromarray(renderer.render(),'RGBA')
            if np.asarray(image)[:,:,3].sum()==0: raise RuntimeError('Empty render')
            filename=f'{i:03d}-{tone}.png'
            image.save(args.output/filename)
            # Dynamic letters need temporal supervision: retain renders as an
            # auxiliary review set, never inject them as static training labels.
            rows.append({'path':str((args.output/filename).resolve()),'label':frame['label'],'source':'digitra_3d','split':'auxiliary' if frame['dynamic'] else 'train','group':f"authored-{frame['label']}",'view':str(frame['view']),'tone':tone})
            if i%5==0 and tone==0:
                tile=Image.new('RGB',(128,150),(220,225,228))
                tile.paste(image.resize((128,128)),(0,20),image.resize((128,128)))
                ImageDraw.Draw(tile).text((5,3),frame['label'],fill='black')
                previews.append(tile)
        if i%5==0: print(f"Rendered {frame['label']} ({i+1}/{len(data['frames'])})",flush=True)
    with (args.output/'manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)
    sheet=Image.new('RGB',(128*8,150*4),'white')
    for i,tile in enumerate(previews):sheet.paste(tile,(i%8*128,i//8*150))
    sheet.save(args.output/'preview.jpg')
    (args.output/'provenance.json').write_text(json.dumps({'renderer':renderer.name,'source_hashes':data['source_hashes'],'images':len(rows),'static_training_images':sum(r['split']=='train' for r in rows),'role':'train-only auxiliary; no real-world accuracy inferred from renders'},indent=2),encoding='utf-8')
    print(json.dumps({'images':len(rows),'static_training_images':sum(r['split']=='train' for r in rows)}),flush=True)


if __name__=='__main__':main()
