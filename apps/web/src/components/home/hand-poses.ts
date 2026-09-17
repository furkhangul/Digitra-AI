import * as THREE from "three";

// Rig-specific constants and handshapes for the hero hand, shared by the
// animated hero model and the /debug-hand diagnostic page so both stay in sync.
//
// Every axis below is ground truth taken from the rigger's own authored
// "Pose_OK" clip inside hand.glb — dumped with scripts/extract-pose.js, which
// reports each bone's rotation delta from the bind pose as axis+angle. Do not
// derive these geometrically: an earlier attempt computed the palm normal via
// cross(indexToPinky, wristToTip), which silently assumes a right hand. This
// rig is the other chirality, so that method returned the exact opposite axis
// and the fingers hyperextended backwards out of the palm.

// Fingers flex around local +Z (Pose_OK: index_prox +42.6°, midd +78.1°, dist +46.2°).
const FINGER_AXIS = new THREE.Vector3(0, 0, 1);

// The thumb does not share that axis — opposition tilts it (Pose_OK reports
// axis ≈ [-0.22, 0.57, 0.79] at thumb_prox), so each thumb bone carries its own.
const THUMB_TRAPEZ_AXIS = new THREE.Vector3(0.03, 0.13, 0.99).normalize();
const THUMB_META_AXIS = new THREE.Vector3(-0.16, 0.63, 0.76).normalize();
const THUMB_PROX_AXIS = new THREE.Vector3(-0.22, 0.57, 0.79).normalize();
const THUMB_DIST_AXIS = new THREE.Vector3(-0.01, 0.11, 0.99).normalize();

// Yaw that turns the palm toward the camera. At 0°/180° the hand is edge-on
// and reads as a sliver with no visible fingers; 90° is the full front view.
export const BASE_Y_ROTATION = Math.PI * 0.5;

// The rig's bounding box includes a forearm stub, which pulls the visual
// centre low; lift the model so the hand itself sits centred in the frame.
export const HAND_CENTER_Y = 0.35;

// An orbit camera always renders its target at the centre of the frame, so the
// only way to raise the hand on screen is to aim below it. The gap also means
// the hand pivots about the wrist rather than mid-palm, which is how a real
// hand turns.
export const ORBIT_TARGET_Y = HAND_CENTER_Y - 1.85;

// The mesh isn't symmetric about its bounding box once yawed to face the
// camera — the palm mass sits right of centre. Aiming here puts the hand,
// rather than the box, in the middle of frame.
export const ORBIT_TARGET_X = 0.2;

export type FingerName = "index" | "middle" | "ring" | "pinky" | "thumb";

export type Joint = { bone: string; axis: THREE.Vector3; maxBend: number };

// Per-joint flexion at full curl, in radians, following real range of motion:
// MCP ~90°, PIP ~100°, DIP ~70° for the fingers.
const finger = (prefix: string): Joint[] => [
  { bone: `${prefix}_prox`, axis: FINGER_AXIS, maxBend: 1.55 },
  { bone: `${prefix}_midd`, axis: FINGER_AXIS, maxBend: 1.75 },
  { bone: `${prefix}_dist`, axis: FINGER_AXIS, maxBend: 1.2 },
];

export const CHAINS: Record<FingerName, Joint[]> = {
  index: finger("index"),
  middle: finger("midd"),
  ring: finger("ring"),
  pinky: finger("pinky"),
  // Thumb angles scale up the partial fold the OK pose uses, so curl 1.0
  // reads as the thumb folded across the palm.
  thumb: [
    { bone: "thumb_trapez", axis: THUMB_TRAPEZ_AXIS, maxBend: 0.82 },
    { bone: "thumb_meta", axis: THUMB_META_AXIS, maxBend: 0.82 },
    { bone: "thumb_prox", axis: THUMB_PROX_AXIS, maxBend: 0.45 },
    { bone: "thumb_dist", axis: THUMB_DIST_AXIS, maxBend: 0.47 },
  ],
};

// Abduction axis for the base (prox) joint, letting a finger fan sideways
// before it curls — needed to tell V/K/P/R apart from U, which share the
// same flexion values. Local Y, guessed from the rig's convention (flexion
// consistently sits on local Z for every finger); verify visually if a new
// rig is swapped in.
export const SPREAD_AXIS = new THREE.Vector3(0, 1, 0);

export type Pose = Record<FingerName, number> & {
  // Lateral fan in radians per finger, applied at the base joint before
  // curling — negative toward the thumb side, positive toward the pinky.
  spread?: Partial<Record<FingerName, number>>;
}; // 0 = open/straight, 1 = fully curled

// Approximate ASL fingerspelling handshapes — simplified (thumb opposition
// isn't modelled, only flexion). Deliberately limited to open handshapes:
// closed-fist letters (A, S, E, M, N) curl every finger behind the palm, so
// the silhouette collapses into a lump and stops reading as a hand at all.
// Each of these keeps at least two fingers extended and visible.
// Open palm — the poster pose. Every finger at rest reads instantly as a hand
// from any angle, which a cycling fingerspell does not when it fills a screen.
export const OPEN_PALM: Pose = { index: 0, middle: 0, ring: 0, pinky: 0, thumb: 0 };

// Index extended, everything else folded away — the gesture the intro morphs
// into so the fingertip can come at the camera and become the transition.
export const POINT: Pose = { index: 0, middle: 1, ring: 1, pinky: 1, thumb: 0.85 };

export const LETTERS: { label: string; pose: Pose }[] = [
  { label: "B", pose: { index: 0, middle: 0, ring: 0, pinky: 0, thumb: 1 } },
  { label: "W", pose: { index: 0, middle: 0, ring: 0, pinky: 1, thumb: 0.9 } },
  { label: "V", pose: { index: 0, middle: 0, ring: 1, pinky: 1, thumb: 0.7 } },
  { label: "L", pose: { index: 0, middle: 1, ring: 1, pinky: 1, thumb: 0 } },
  { label: "Y", pose: { index: 1, middle: 1, ring: 1, pinky: 0, thumb: 0 } },
];
