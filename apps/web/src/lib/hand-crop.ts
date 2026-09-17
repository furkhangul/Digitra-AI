type Point = { x: number; y: number };
type Bounds = { left: number; top: number; right: number; bottom: number };
export type HandCrop = { x: number; y: number; width: number; height: number };

function bounds(points: Point[]): Bounds {
  return {
    left: Math.min(...points.map((p) => p.x)),
    top: Math.min(...points.map((p) => p.y)),
    right: Math.max(...points.map((p) => p.x)),
    bottom: Math.max(...points.map((p) => p.y)),
  };
}
function center(box: Bounds) {
  return { x: (box.left + box.right) / 2, y: (box.top + box.bottom) / 2 };
}

/** Keep hands at a consistent input scale as their distance changes. */
export class HandCropTracker {
  private pair: { boxes: Bounds[]; timestamp: number } | null = null;

  reset() { this.pair = null; }

  update(hands: Point[][], width: number, height: number, timestamp: number): HandCrop | null {
    if (!hands.length || width <= 0 || height <= 0) { this.reset(); return null; }
    const boxes = hands.slice(0, 2).filter((points) => points.length && points.every((p) => Number.isFinite(p.x) && Number.isFinite(p.y)))
      .map((points) => bounds(points.map((p) => ({ x: p.x * width, y: p.y * height }))));
    if (!boxes.length) { this.reset(); return null; }
    let context = [...boxes];
    if (boxes.length === 2) {
      this.pair = { boxes, timestamp };
    } else if (this.pair && timestamp >= this.pair.timestamp && timestamp - this.pair.timestamp <= 600) {
      const current = center(boxes[0]);
      const distances = this.pair.boxes.map((box) => {
        const previous = center(box);
        return Math.hypot(current.x - previous.x, current.y - previous.y);
      });
      const nearest = distances[0] <= distances[1] ? 0 : 1;
      const prior = this.pair.boxes[nearest];
      if (distances[nearest] < Math.max(prior.right - prior.left, prior.bottom - prior.top) * 0.8) {
        // Preserve a briefly occluded partner, rather than expanding to the
        // entire frame whenever MediaPipe reports a single hand.
        context = [boxes[0], this.pair.boxes[1 - nearest]];
      }
    }
    const left = Math.min(...context.map((box) => box.left));
    const top = Math.min(...context.map((box) => box.top));
    const right = Math.max(...context.map((box) => box.right));
    const bottom = Math.max(...context.map((box) => box.bottom));
    // The training images are hand crops. A single hand needs the same close
    // framing; missing partners are handled by the short pair history above.
    const padding = 0.18;
    const side = Math.max(48, Math.max(right - left, bottom - top) * (1 + 2 * padding));
    const cropWidth = Math.min(side, width);
    const cropHeight = Math.min(side, height);
    return {
      x: Math.max(0, Math.min(width - cropWidth, (left + right - cropWidth) / 2)),
      y: Math.max(0, Math.min(height - cropHeight, (top + bottom - cropHeight) / 2)),
      width: cropWidth,
      height: cropHeight,
    };
  }
}
