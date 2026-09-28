import { it, expect } from "vitest";
import { RecognitionSchema, WineDetailsSchema } from "./contracts";
import { response } from "../../test/fixture";
import { wines } from "../../mocks/catalog";
it("accepts independent detection without a catalogue match and null F1", () => {
  expect(RecognitionSchema.safeParse(response).success).toBe(true);
});
it("rejects out-of-range coordinates, duplicate ids and degenerate polygons", () => {
  for (const detections of [
    [
      {
        ...response.detections[0],
        polygon: [
          [0, 0],
          [2, 0],
          [1, 1],
        ],
      },
    ],
    [response.detections[0], response.detections[0]],
    [
      {
        ...response.detections[0],
        polygon: [
          [0, 0],
          [0.5, 0.5],
          [1, 1],
        ],
      },
    ],
  ])
    expect(
      RecognitionSchema.safeParse({ ...response, detections }).success,
    ).toBe(false);
});
it("rejects 135 percent alcohol and ratings exceeding their scale", () => {
  expect(
    WineDetailsSchema.safeParse({ ...wines[0], alcoholPercent: 135 }).success,
  ).toBe(false);
  expect(
    WineDetailsSchema.safeParse({
      ...wines[0],
      ratings: { community: { value: 6, scaleMax: 5 }, roskachestvo: null },
    }).success,
  ).toBe(false);
});
