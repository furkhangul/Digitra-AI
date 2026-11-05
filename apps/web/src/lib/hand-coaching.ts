// Live position coaching from MediaPipe landmarks (normalized 0–1 image
// coordinates). It only judges framing — distance, centring, the number of
// hands and how far apart two hands are — never the handshape itself, which
// is the recognizer's job.

export type CoachPoint = { x: number; y: number };

type Box = { minX: number; maxX: number; minY: number; maxY: number; cx: number; cy: number; size: number };

const EDGE = 0.025;
const TOO_SMALL = 0.2;
const TOO_LARGE = 0.8;
const TWO_HAND_GAP = 2.4;

function box(points: CoachPoint[], aspect: number): Box {
  let minX = 1, maxX = 0, minY = 1, maxY = 0;
  for (const { x, y } of points) {
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  }
  // Compare sizes in frame-height units so wide cameras are not penalised.
  const size = Math.max(maxY - minY, (maxX - minX) * aspect);
  return { minX, maxX, minY, maxY, cx: (minX + maxX) / 2, cy: (minY + maxY) / 2, size };
}

/**
 * @param hands     landmark lists, one per detected hand
 * @param twoHanded whether the target letter needs both hands; null if unknown
 * @param aspect    video width / height
 */
export function coachHands(hands: CoachPoint[][], twoHanded: boolean | null, aspect = 16 / 9): string | null {
  if (!hands.length) return null;
  const boxes = hands.map((points) => box(points, aspect));

  if (boxes.some((b) => b.minX < EDGE || b.maxX > 1 - EDGE || b.minY < EDGE || b.maxY > 1 - EDGE)) {
    return "Elin kadrajın kenarından taşıyor; biraz ortaya getir.";
  }
  if (twoHanded && boxes.length === 1) {
    return "Bu harf iki elle yapılır; diğer elini de kadraja getir.";
  }
  if (twoHanded === false && boxes.length === 2) {
    return "Bu harf tek elle yapılır; diğer elini indir.";
  }
  const largest = Math.max(...boxes.map((b) => b.size));
  if (largest < TOO_SMALL) return "Elini kameraya biraz yaklaştır.";
  if (largest > TOO_LARGE) return "Elini kameradan biraz uzaklaştır.";
  if (boxes.length === 2) {
    const [a, b] = boxes;
    const gap = Math.hypot((a.cx - b.cx) * aspect, a.cy - b.cy) / ((a.size + b.size) / 2);
    if (gap > TWO_HAND_GAP) return "İki elini birbirine yaklaştır.";
  }
  return null;
}
