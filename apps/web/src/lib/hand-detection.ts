import type { HandLandmarkerResult, NormalizedLandmark } from "@mediapipe/tasks-vision";

// Handedness.score describes left/right classification, not hand presence.
// Presence is checked by MediaPipe itself with these options:
// https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/web_js
export const HAND_DETECTION_OPTIONS = {
  minHandDetectionConfidence: 0.65,
  minHandPresenceConfidence: 0.65,
  minTrackingConfidence: 0.6,
};

type Palm = { x: number; y: number; size: number };
type Track = Palm & { firstSeen: number; lastSeen: number; frames: number };
const PALM_JOINTS = [0, 5, 9, 13, 17];

function plausiblePalm(points: NormalizedLandmark[], aspect: number): Palm | null {
  if (points.length !== 21 || points.some((p) => ![p.x, p.y, p.z].every(Number.isFinite))) {
    return null;
  }
  if (points.filter((p) => p.x >= -0.1 && p.x <= 1.1 && p.y >= -0.1 && p.y <= 1.1).length < 17) {
    return null;
  }
  // z is normalized on the same scale as x. Keeping depth preserves side-on
  // and folded fingers; this does not require an open, front-facing palm.
  const distance = (a: number, b: number) => Math.hypot(
    (points[a].x - points[b].x) * aspect,
    points[a].y - points[b].y,
    (points[a].z - points[b].z) * aspect,
  );
  const length = distance(0, 9);
  const width = distance(5, 17);
  if (length < 0.018 || length > 1.2 || width / length < 0.18 || width / length > 2.5) {
    return null;
  }
  // Broad bone-length checks reject collapsed/exploded landmark sets, while
  // tolerating one occluded finger and arbitrary finger bending.
  const plausibleFingers = [5, 9, 13, 17].filter((base) => {
    const bones = [0, 1, 2].map((offset) => distance(base + offset, base + offset + 1) / length);
    return bones.every((bone) => bone > 0.035 && bone < 1.3);
  });
  if (plausibleFingers.length < 3) return null;
  return {
    x: PALM_JOINTS.reduce((sum, i) => sum + points[i].x * aspect, 0) / PALM_JOINTS.length,
    y: PALM_JOINTS.reduce((sum, i) => sum + points[i].y, 0) / PALM_JOINTS.length,
    size: Math.max(length, width),
  };
}

/** Admit new hands only after 3+ consistent frames spanning at least 150 ms. */
export class StableHandDetections {
  private tracks: Track[] = [];

  reset() {
    this.tracks = [];
  }

  update(result: HandLandmarkerResult, timestamp: number, aspect: number): HandLandmarkerResult {
    const next: Track[] = [];
    const used = new Set<number>();
    const accepted: number[] = [];
    for (const [index, points] of result.landmarks.slice(0, 2).entries()) {
      const palm = plausiblePalm(points, aspect);
      if (!palm) continue;
      let match = -1;
      let closest = Infinity;
      this.tracks.forEach((track, trackIndex) => {
        if (used.has(trackIndex) || timestamp <= track.lastSeen || timestamp - track.lastSeen > 250) return;
        const ratio = palm.size / track.size;
        const displacement = Math.hypot(palm.x - track.x, palm.y - track.y);
        if (ratio >= 0.45 && ratio <= 2.2 && displacement < Math.max(0.07, 1.25 * Math.max(palm.size, track.size)) && displacement < closest) {
          closest = displacement;
          match = trackIndex;
        }
      });
      const previous = match >= 0 ? this.tracks[match] : null;
      if (previous) used.add(match);
      const track = {
        ...palm,
        firstSeen: previous?.firstSeen ?? timestamp,
        lastSeen: timestamp,
        frames: (previous?.frames ?? 0) + 1,
      };
      next.push(track);
      if (track.frames >= 3 && timestamp - track.firstSeen >= 150) accepted.push(index);
    }
    // Keep only track identity across a brief finger/hand overlap. Do not
    // draw or send old landmarks: every accepted point still comes from the
    // current frame. Otherwise a one-frame miss triggers 150 ms of extra
    // two-hand loss and repeatedly breaks Ğ's motion window.
    const missing = this.tracks.filter((track, index) =>
      !used.has(index) && timestamp > track.lastSeen && timestamp - track.lastSeen <= 250,
    );
    this.tracks = [...next, ...missing].slice(0, 4);
    const handedness = result.handedness ?? result.handednesses;
    return {
      ...result,
      landmarks: accepted.map((i) => result.landmarks[i]),
      worldLandmarks: accepted.map((i) => result.worldLandmarks[i] ?? []),
      handedness: accepted.map((i) => handedness[i] ?? []),
      handednesses: accepted.map((i) => handedness[i] ?? []),
    };
  }
}
