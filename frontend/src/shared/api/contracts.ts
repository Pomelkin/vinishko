import { z } from "zod";

const ratio = z.number().min(0).max(1).nullable();
const rating = z
  .object({ value: z.number().min(0), scaleMax: z.number().positive() })
  .refine((v) => v.value <= v.scaleMax)
  .nullable();
const slug = z
  .string()
  .min(1)
  .regex(/^[a-z0-9]+(?:[-_][a-z0-9]+)*$/);
const imageUrl = z
  .string()
  .refine(
    (s) => (s.startsWith("/") && !s.startsWith("//")) || /^https?:\/\//.test(s),
  )
  .nullable();
export const WineSummarySchema = z.object({
  slug,
  name: z.string().min(1),
  producer: z.string(),
  imageUrl,
  region: z.string().nullable(),
  grapeVarieties: z.array(z.string()),
  color: z.string().nullable(),
  category: z.string().nullable(),
  vintage: z.number().int().min(1900).max(2100).nullable(),
  shortDescription: z.string().nullable(),
  ratings: z.object({ community: rating, roskachestvo: rating }),
});
export const WineDetailsSchema = WineSummarySchema.extend({
  description: z.string().nullable(),
  alcoholPercent: z.number().min(0).max(100).nullable(),
  servingTemperature: z.string().nullable(),
  foodPairings: z.array(z.string()),
  similarWines: z.array(WineSummarySchema),
  catalog: z
    .record(
      z.string(),
      z.union([z.string(), z.number(), z.boolean(), z.null()]),
    )
    .optional(),
  candidateCatalogs: z
    .array(
      z.record(
        z.string(),
        z.union([z.string(), z.number(), z.boolean(), z.null()]),
      ),
    )
    .max(5)
    .optional(),
});
const CandidateSchema = z
  .object({
    slug,
    rank: z.number().int().positive(),
    similarityScore: ratio,
    wine: WineSummarySchema,
  })
  .refine((c) => c.slug === c.wine.slug);
const PointSchema = z.tuple([
  z.number().min(0).max(1),
  z.number().min(0).max(1),
]);
const PolygonSchema = z
  .array(PointSchema)
  .min(3)
  .max(4096)
  .refine(
    (p) =>
      Math.abs(
        p.reduce(
          (a, v, i) =>
            a +
            v[0] * p[(i + 1) % p.length][1] -
            p[(i + 1) % p.length][0] * v[1],
          0,
        ),
      ) > 1e-8,
  );
export const UnknownWineSchema = z.object({
  category: z.string(),
  brand: z.string(),
});
const base = {
  id: z.string().min(1),
  polygon: PolygonSchema,
  polygons: z.array(PolygonSchema).min(1).optional(),
  unknownWine: UnknownWineSchema.nullable().optional(),
  unknownWineError: z.string().nullable().optional(),
  rejection: z
    .object({
      stage: z.string(),
      reason: z.string(),
      label: z.string(),
      description: z.string(),
      detail: z.string(),
      message: z.string(),
    })
    .nullable()
    .optional(),
  detectionConfidence: ratio,
};
const match = z
  .object({ slug, matchConfidence: ratio, wine: WineSummarySchema })
  .refine((m) => m.slug === m.wine.slug);
export const DetectionSchema = z.discriminatedUnion("status", [
  z.object({
    ...base,
    status: z.literal("matched"),
    match,
    similar: z.array(CandidateSchema),
  }),
  z.object({
    ...base,
    status: z.literal("unmatched"),
    match: z.null(),
    similar: z.array(CandidateSchema),
  }),
]);
export const RecognitionSchema = z.object({
  schemaVersion: z.literal("1.0"),
  requestId: z.string().min(1),
  image: z.object({
    width: z.number().int().positive(),
    height: z.number().int().positive(),
    coordinateSpace: z.literal("normalized"),
    orientationApplied: z.literal(true),
  }),
  bestMatch: z.object({ slug }).nullable(),
  metrics: z.object({
    f1Top1: ratio,
    f1Top5: ratio,
    scope: z.literal("evaluation-dataset"),
    datasetId: z.string().nullable(),
    isMock: z.boolean(),
  }),
  detections: z
    .array(DetectionSchema)
    .max(100)
    .refine((d) => new Set(d.map((x) => x.id)).size === d.length),
  processingTimeMs: z.number().nonnegative().nullable(),
  ignored: z.number().int().nonnegative().optional(),
});
export type Point = [number, number];
export type WineSummary = z.infer<typeof WineSummarySchema>;
export type WineDetails = z.infer<typeof WineDetailsSchema>;
export type Candidate = z.infer<typeof CandidateSchema>;
export type BottleDetection = z.infer<typeof DetectionSchema>;
export type RecognitionResponse = z.infer<typeof RecognitionSchema>;
export interface RecognitionService {
  recognize(
    image: Blob,
    options?: { signal?: AbortSignal },
  ): Promise<RecognitionResponse>;
  getWine(
    slug: string,
    options?: { signal?: AbortSignal },
  ): Promise<WineDetails>;
  searchWines(
    query: string,
    options?: { signal?: AbortSignal },
  ): Promise<WineSummary[]>;
}
