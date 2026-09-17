"""Bounded, train-only camera perturbations for fingerspelling."""
import io
import random
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms as T
from torchvision.transforms import functional as F, InterpolationMode

MEAN=(.485,.456,.406)
STD=(.229,.224,.225)


def letterbox(image,size=224):
    return ImageOps.pad(image.convert('RGB'),(size,size),method=Image.Resampling.BICUBIC,color=(114,114,114))


def background(image):
    """Composite rendered RGBA hands on a label-independent random backdrop."""
    if image.mode!='RGBA':return image.convert('RGB')
    w,h=image.size
    colors=np.random.uniform(35,225,(2,3))
    axis=np.linspace(0,1,h)[:,None,None]
    pixels=np.broadcast_to(colors[0]*(1-axis)+colors[1]*axis,(h,w,3)).copy()
    pixels+=np.random.normal(0,random.uniform(1,8),(h,w,1))
    bg=Image.fromarray(pixels.clip(0,255).astype('uint8'),'RGB')
    bg.paste(image,mask=image.getchannel('A'))
    return bg


def camera_effect(image,kind,strength=1.):
    pixels=np.asarray(image,dtype=np.float32)/255
    h,w=pixels.shape[:2]
    if kind=='gamma':
        pixels=np.power(pixels,random.uniform(1-.35*strength,1+.5*strength))
    elif kind=='exposure':
        # Auto-exposure can be wrong in either direction while the hand stays
        # visible. Keep the perturbation bounded instead of clipping the sign.
        pixels*=random.uniform(.55,1.45)**strength
    elif kind=='high_key':
        pixels=pixels*(1-.22*strength)+.22*strength
    elif kind=='color_temperature':
        gains=np.array([random.uniform(.86,1.14),1.,random.uniform(.86,1.14)],dtype=np.float32)
        pixels*=gains[None,None,:]**strength
    elif kind=='white_balance':
        pixels*=np.random.uniform(1-.14*strength,1+.14*strength,(1,1,3))
    elif kind=='noise':
        pixels+=np.random.normal(0,random.uniform(.004,.035)*strength,pixels.shape)
    elif kind=='shadow':
        x,y=np.meshgrid(np.linspace(-1,1,w),np.linspace(-1,1,h))
        a=random.uniform(-np.pi,np.pi)
        mask=1/(1+np.exp(-(x*np.cos(a)+y*np.sin(a)-random.uniform(-.4,.4))*5))
        pixels*=1-mask[:,:,None]*random.uniform(.15,.45)*strength
    elif kind=='backlight':
        # A bright window behind the hands creates a soft directional wash.
        x,y=np.meshgrid(np.linspace(-1,1,w),np.linspace(-1,1,h))
        mask=np.clip((x*.65+y*.35+.35),0,1)**2
        pixels*=1-mask[:,:,None]*.42*strength
    elif kind=='jpeg':
        buf=io.BytesIO();image.save(buf,format='JPEG',quality=random.randint(40,90));buf.seek(0)
        with Image.open(buf) as im:return im.convert('RGB').copy()
    elif kind=='low_resolution':
        s=random.uniform(.4,.8);return image.resize((max(8,int(w*s)),max(8,int(h*s))),Image.Resampling.BILINEAR).resize((w,h),Image.Resampling.BILINEAR)
    elif kind=='motion_blur':
        kernel=np.zeros((3,3),dtype=np.float32)
        if random.random()<.5:kernel[1,:]=1/3
        else:kernel[:,1]=1/3
        pixels=cv2.filter2D(pixels,-1,kernel)
    elif kind=='gaussian_blur':return image.filter(ImageFilter.GaussianBlur(random.uniform(.2,.9)*strength))
    return Image.fromarray((pixels.clip(0,1)*255).astype('uint8'))


CAMERA_EFFECTS=('gamma','exposure','high_key','color_temperature','white_balance','noise','shadow','backlight','jpeg','low_resolution','motion_blur','gaussian_blur')


class TrainingTransform:
    def __init__(self,size=224):self.size=size;self.strength=1.

    def __call__(self,image):
        image=letterbox(background(image),self.size)
        s=self.strength
        # Reflect the complete pair; never independently swap one hand.
        if random.random()<.5:image=F.hflip(image)
        image=F.affine(image,angle=random.uniform(-10,10)*s,
            translate=[int(random.uniform(-.10,.10)*self.size*s) for _ in range(2)],
            scale=random.uniform(1-.25*s,1+.08*s),shear=[random.uniform(-3,3)*s,random.uniform(-2,2)*s],
            interpolation=InterpolationMode.BILINEAR,fill=[114]*3)
        if random.random()<.15:
            image=T.RandomPerspective(distortion_scale=.08*s,p=1,fill=114)(image)
        image=T.ColorJitter(brightness=.20*s,contrast=.20*s,saturation=.12*s,hue=.015*s)(image)
        for kind in random.sample(CAMERA_EFFECTS,k=random.choices([0,1,2],[.25,.50,.25])[0]):
            image=camera_effect(image,kind,s)
        if random.random()<.03:image=ImageOps.grayscale(image).convert('RGB')
        tensor=F.to_tensor(image)
        if random.random()<.05:
            tensor=T.RandomErasing(p=1,scale=(.005,.015),ratio=(.5,2),value='random')(tensor)
        return F.normalize(tensor,MEAN,STD)


def evaluation_transform(image):
    return F.normalize(F.to_tensor(letterbox(image)),MEAN,STD)


def stress_transform(image,variant):
    image=letterbox(image)
    if variant=='mirror':image=F.hflip(image)
    elif variant=='far':image=F.affine(image,0,[0,0],.60,[0,0],fill=[114]*3)
    elif variant=='shift':image=F.affine(image,0,[20,-15],.8,[0,0],fill=[114]*3)
    elif variant=='rotation':image=F.affine(image,10,[0,0],.9,[0,0],fill=[114]*3)
    elif variant=='low_light':image=F.adjust_brightness(image,.55)
    elif variant=='high_key':image=camera_effect(image,'high_key',1.)
    elif variant=='backlight':image=camera_effect(image,'backlight',1.)
    elif variant=='color_temperature':
        pixels=np.asarray(image,dtype=np.float32)*np.array([1.10,1.,.90],dtype=np.float32)
        image=Image.fromarray(pixels.clip(0,255).astype('uint8'))
    elif variant=='blur':image=image.filter(ImageFilter.GaussianBlur(.8))
    elif variant=='jpeg':
        buf=io.BytesIO();image.save(buf,format='JPEG',quality=40);buf.seek(0)
        with Image.open(buf) as opened:image=opened.convert('RGB').copy()
    return F.normalize(F.to_tensor(image),MEAN,STD)
