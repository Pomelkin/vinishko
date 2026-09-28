import type { BottleDetection, Point } from "../api/contracts";
export function contain(iw: number, ih: number, cw: number, ch: number) {
  const scale = Math.min(cw / iw, ch / ih),
    width = iw * scale,
    height = ih * scale;
  return { scale, width, height, x: (cw - width) / 2, y: (ch - height) / 2 };
}
export function project(
  point: Point,
  frame: ReturnType<typeof contain>,
): Point {
  return [frame.x + point[0] * frame.width, frame.y + point[1] * frame.height];
}
export function polygonArea(p: Point[]) {
  return (
    Math.abs(
      p.reduce(
        (a, v, i) =>
          a + v[0] * p[(i + 1) % p.length][1] - p[(i + 1) % p.length][0] * v[1],
        0,
      ),
    ) / 2
  );
}
export function center(p: Point[]): Point {
  return [
    (Math.min(...p.map((v) => v[0])) + Math.max(...p.map((v) => v[0]))) / 2,
    (Math.min(...p.map((v) => v[1])) + Math.max(...p.map((v) => v[1]))) / 2,
  ];
}
export function includesPoint(p: Point[], [x, y]: Point) {
  let inside = false;
  for (let i = 0, j = p.length - 1; i < p.length; j = i++) {
    const [xi, yi] = p[i],
      [xj, yj] = p[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi)
      inside = !inside;
  }
  return inside;
}
/** Smallest containing silhouette wins; stable id breaks equal-area ties. */
export function hitTest(detections: BottleDetection[], point: Point) {
  return detections
    .filter((d) => includesPoint(d.polygon, point))
    .sort(
      (a, b) =>
        polygonArea(a.polygon) - polygonArea(b.polygon) ||
        a.id.localeCompare(b.id),
    )[0];
}
