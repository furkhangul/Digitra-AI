"""Minimal hidden Windows OpenGL renderer; uses the installed graphics driver."""
import ctypes as c
from ctypes import wintypes as w
import numpy as np


class PixelFormat(c.Structure):
    _fields_=[('size',w.WORD),('version',w.WORD),('flags',w.DWORD),('pixel',c.c_byte),('color',c.c_byte),('red',c.c_byte),('redshift',c.c_byte),('green',c.c_byte),('greenshift',c.c_byte),('blue',c.c_byte),('blueshift',c.c_byte),('alpha',c.c_byte),('alphashift',c.c_byte),('accum',c.c_byte),('accumred',c.c_byte),('accumgreen',c.c_byte),('accumblue',c.c_byte),('accumalpha',c.c_byte),('depth',c.c_byte),('stencil',c.c_byte),('aux',c.c_byte),('layer',c.c_byte),('reserved',c.c_byte),('layermask',w.DWORD),('visiblemask',w.DWORD),('damagemask',w.DWORD)]


class Renderer:
    def __init__(self,fragment,size=256):
        self.size=size
        user=c.WinDLL('user32'); gdi=c.WinDLL('gdi32'); self.gl=c.WinDLL('opengl32')
        user.CreateWindowExW.argtypes=[w.DWORD,w.LPCWSTR,w.LPCWSTR,w.DWORD,c.c_int,c.c_int,c.c_int,c.c_int,w.HWND,w.HMENU,w.HINSTANCE,c.c_void_p]
        user.CreateWindowExW.restype=w.HWND
        user.GetDC.argtypes=[w.HWND];user.GetDC.restype=w.HDC
        self.window=user.CreateWindowExW(0,'STATIC','Digitra offscreen training renderer',0x80000000,0,0,size,size,None,None,None,None)
        self.dc=user.GetDC(self.window)
        gdi.ChoosePixelFormat.argtypes=[w.HDC,c.POINTER(PixelFormat)];gdi.SetPixelFormat.argtypes=[w.HDC,c.c_int,c.POINTER(PixelFormat)]
        p=PixelFormat();p.size=c.sizeof(p);p.version=1;p.flags=4|32;p.color=32;p.alpha=8
        index=gdi.ChoosePixelFormat(self.dc,c.byref(p))
        if not index or not gdi.SetPixelFormat(self.dc,index,c.byref(p)):raise RuntimeError('Pixel format unavailable')
        self.gl.wglCreateContext.argtypes=[w.HDC];self.gl.wglCreateContext.restype=c.c_void_p
        self.context=self.gl.wglCreateContext(self.dc)
        self.gl.wglMakeCurrent.argtypes=[w.HDC,c.c_void_p]
        if not self.gl.wglMakeCurrent(self.dc,self.context):raise RuntimeError('OpenGL context unavailable')
        self.gl.wglGetProcAddress.argtypes=[c.c_char_p];self.gl.wglGetProcAddress.restype=c.c_void_p
        self.gl.glGetString.restype=c.c_char_p
        self.name=self.gl.glGetString(0x1F01).decode()
        self.fn={}
        self._load('glCreateShader',c.c_uint,c.c_uint)
        self._load('glShaderSource',None,c.c_uint,c.c_int,c.POINTER(c.c_char_p),c.POINTER(c.c_int))
        self._load('glCompileShader',None,c.c_uint)
        self._load('glGetShaderiv',None,c.c_uint,c.c_uint,c.POINTER(c.c_int))
        self._load('glGetShaderInfoLog',None,c.c_uint,c.c_int,c.POINTER(c.c_int),c.c_char_p)
        self._load('glCreateProgram',c.c_uint)
        self._load('glAttachShader',None,c.c_uint,c.c_uint)
        self._load('glLinkProgram',None,c.c_uint)
        self._load('glGetProgramiv',None,c.c_uint,c.c_uint,c.POINTER(c.c_int))
        self._load('glGetProgramInfoLog',None,c.c_uint,c.c_int,c.POINTER(c.c_int),c.c_char_p)
        self._load('glUseProgram',None,c.c_uint)
        self._load('glGetUniformLocation',c.c_int,c.c_uint,c.c_char_p)
        self._load('glUniform1f',None,c.c_int,c.c_float)
        for n in [2,3,4]:self._load(f'glUniform{n}fv',None,c.c_int,c.c_int,c.POINTER(c.c_float))
        self._load('glUniformMatrix4fv',None,c.c_int,c.c_int,c.c_ubyte,c.POINTER(c.c_float))
        vertex='#version 330\nout vec2 vUv; void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);vUv=p;gl_Position=vec4(p*2.0-1.0,0,1);}'
        self.program=self.fn['glCreateProgram']()
        for kind,code in [(0x8B31,vertex),(0x8B30,fragment)]:
            shader=self.fn['glCreateShader'](kind);src=c.c_char_p(code.encode())
            self.fn['glShaderSource'](shader,1,c.byref(src),None);self.fn['glCompileShader'](shader)
            ok=c.c_int();self.fn['glGetShaderiv'](shader,0x8B81,c.byref(ok))
            if not ok.value:
                log=c.create_string_buffer(10000);self.fn['glGetShaderInfoLog'](shader,10000,None,log);raise RuntimeError(log.value.decode())
            self.fn['glAttachShader'](self.program,shader)
        self.fn['glLinkProgram'](self.program)
        ok=c.c_int();self.fn['glGetProgramiv'](self.program,0x8B82,c.byref(ok))
        if not ok.value:
            log=c.create_string_buffer(10000);self.fn['glGetProgramInfoLog'](self.program,10000,None,log);raise RuntimeError(log.value.decode())
        self.fn['glUseProgram'](self.program)
        self.gl.glViewport(0,0,size,size)
        self.gl.glReadPixels.argtypes=[c.c_int,c.c_int,c.c_int,c.c_int,c.c_uint,c.c_uint,c.c_void_p]

    def _load(self,name,result,*args):
        addr=self.gl.wglGetProcAddress(name.encode())
        if not addr:raise RuntimeError(f'{name} unavailable')
        self.fn[name]=c.WINFUNCTYPE(result,*args)(addr)

    def uniform(self,name,value):
        loc=self.fn['glGetUniformLocation'](self.program,name.encode())
        if loc<0:return
        if isinstance(value,(float,int)):self.fn['glUniform1f'](loc,value);return
        data=np.asarray(value,dtype='f4');ptr=data.ctypes.data_as(c.POINTER(c.c_float))
        if name=='uInverse':self.fn['glUniformMatrix4fv'](loc,len(data)//16,0,ptr)
        else:
            n=2 if name in ['uPresence','uDepth'] else 3 if name in ['skin','lamp'] else 4
            self.fn[f'glUniform{n}fv'](loc,len(data)//n,ptr)

    def render(self):
        self.gl.glClear(0x4000)
        self.gl.glDrawArrays(4,0,3)
        pixels=np.zeros((self.size,self.size,4),dtype=np.uint8)
        self.gl.glReadPixels(0,0,self.size,self.size,0x1908,0x1401,pixels.ctypes.data)
        return pixels[::-1].copy()
