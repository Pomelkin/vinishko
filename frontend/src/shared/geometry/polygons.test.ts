import { describe, it, expect } from "vitest";
import { contain, project, hitTest } from "./polygons";
import type { BottleDetection, Point } from "../api/contracts";
describe("image coordinate system", () => {
  it("centers portrait and landscape photos with contain letterboxing", () => {
    const portrait = contain(1000, 2000, 390, 400);
    expect(portrait).toEqual({
      scale: 0.2,
      width: 200,
      height: 400,
      x: 95,
      y: 0,
    });
    expect(project([0.5, 0.5], portrait)).toEqual([195, 200]);
    const landscape = contain(2000, 1000, 390, 400);
    expect(project([0, 0], landscape)).toEqual([0, 102.5]);
    expect(project([1, 1], landscape)).toEqual([390, 297.5]);
  });
  it("keeps normalized points aligned after resize and rotation", () => {
    for (const [w, h] of [
      [360, 844],
      [844, 360],
      [1280, 800],
    ]) {
      const f = contain(900, 1000, w, h);
      const p: Point = [0.23, 0.77],
        screen = project(p, f);
      expect((screen[0] - f.x) / f.width).toBeCloseTo(p[0]);
      expect((screen[1] - f.y) / f.height).toBeCloseTo(p[1]);
    }
  });
  it("chooses smallest overlapping silhouette deterministically", () => {
    const d = (id: string, p: Point[]): BottleDetection => ({
      id,
      polygon: p,
      detectionConfidence: null,
      status: "unmatched",
      match: null,
      similar: [],
    });
    const a = d("large", [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
      ]),
      b = d("small", [
        [0.2, 0.2],
        [0.8, 0.2],
        [0.8, 0.8],
        [0.2, 0.8],
      ]);
    expect(hitTest([a, b], [0.5, 0.5])?.id).toBe("small");
    expect(hitTest([b, a], [0.5, 0.5])?.id).toBe("small");
    expect(hitTest([a, b], [2, 2])).toBeUndefined();
  });
});
