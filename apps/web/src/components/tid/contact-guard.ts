import * as THREE from "three";
import { MeshBVH } from "three-mesh-bvh";
import surface from "./collision-surface.json";
import { FINGERS, rigSpec } from "./poses";
import type { PairFrame, HandFrame } from "./motion";

const identity=new THREE.Matrix4();
const y=new THREE.Vector3(0,1,0), z=new THREE.Vector3(0,0,1);
const rad=THREE.MathUtils.degToRad;
const boneNames=["wrist",...FINGERS.flatMap(f=>[`${f}_0`,`${f}_1`,`${f}_2`])];
const weights=surface.weights.map(row=>{
  const total=row.reduce((n,p)=>n+Number(p[1]),0);
  return row.map(p=>[boneNames.indexOf(String(p[0])),Number(p[1])/total]);
});
const rest=surface.positions.map(p=>new THREE.Vector3(...p));

class CollisionHand {
  geometry=new THREE.BufferGeometry();
  bvh:MeshBVH;
  positions=new Float32Array(rest.length*3);
  heads=boneNames.map(()=>new THREE.Vector3());
  rotations=boneNames.map(()=>new THREE.Quaternion());
  restHeads=boneNames.map(name=>name==='wrist'?new THREE.Vector3():new THREE.Vector3(...rigSpec[name.split('_')[0] as typeof FINGERS[number]].points[Number(name.split('_')[1])]));
  constructor() {
    this.geometry.setAttribute('position',new THREE.BufferAttribute(this.positions,3));
    this.geometry.setIndex(surface.triangles.flat());
    rest.forEach((p,i)=>p.toArray(this.positions,i*3));
    this.bvh=new MeshBVH(this.geometry,{maxLeafTris:6});
    (this.geometry as THREE.BufferGeometry & {boundsTree:MeshBVH}).boundsTree=this.bvh;
  }
  update(frame:HandFrame,left:boolean) {
    const v=new THREE.Vector3(),q=new THREE.Quaternion();
    FINGERS.forEach((f,fi)=>{
      const spec=rigSpec[f], direction=new THREE.Vector3(...spec.direction);
      const axis=direction.clone().cross(z).normalize();
      const base=1+fi*3,offset=fi*5;
      this.heads[base].fromArray(spec.base);
      this.rotations[base].setFromAxisAngle(z,rad(frame.angles[offset+3])).multiply(q.setFromAxisAngle(y,rad(frame.angles[offset+4])));
      for(let j=0;j<3;j++) {
        const n=base+j;
        if(j>0) {
          this.heads[n].copy(this.heads[n-1]).add(v.copy(direction).multiplyScalar(spec.lengths[j-1]).applyQuaternion(this.rotations[n-1]));
          this.rotations[n].copy(this.rotations[n-1]);
        }
        this.rotations[n].multiply(q.setFromAxisAngle(axis,rad(frame.angles[offset+j])));
      }
    });
    const out=new THREE.Vector3();
    rest.forEach((p,i)=>{
      out.set(0,0,0);
      for(const [bone,w] of weights[i]) out.addScaledVector(v.copy(p).sub(this.restHeads[bone]).applyQuaternion(this.rotations[bone]).add(this.heads[bone]),w);
      if(left)out.x*=-1;
      out.multiplyScalar(frame.presence).applyQuaternion(frame.rotation).add(frame.position).toArray(this.positions,i*3);
    });
    this.bvh.refit();
  }
  dispose(){this.geometry.dispose();}
}

/** Lightweight skinned surfaces guard inter-hand contact during playback.
 * The proxy is a coarse mesh; static final poses also have a full-mesh audit.
 */
export class ContactGuard {
  right=new CollisionHand();left=new CollisionHand();
  lastRelease=0;
  intersects(frame:PairFrame) {
    if(frame.right.presence<.95||frame.left.presence<.95)return false;
    this.right.update(frame.right,false);this.left.update(frame.left,true);
    return this.right.bvh.intersectsGeometry(this.left.geometry,identity);
  }
  apply(frame:PairFrame) {
    this.lastRelease=0;
    if(!this.intersects(frame))return;
    const r=new THREE.Vector3(0,.5,0).applyQuaternion(frame.right.rotation).add(frame.right.position);
    const l=new THREE.Vector3(0,.5,0).applyQuaternion(frame.left.rotation).add(frame.left.position);
    const direction=r.sub(l).normalize();
    if(direction.lengthSq()<.1)direction.set(0,0,1);
    // A small gap rather than crossing surfaces. Root changes are captured by
    // HandPlayer when the user interrupts an animation.
    for(let i=0;i<60;i++) {
      frame.right.position.addScaledVector(direction,.01);
      frame.left.position.addScaledVector(direction,-.01);
      this.lastRelease+=.02;
      if(!this.intersects(frame))break;
    }
  }
  dispose(){this.right.dispose();this.left.dispose();}
}
