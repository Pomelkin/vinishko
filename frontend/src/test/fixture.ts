import type {
  RecognitionResponse,
  RecognitionService,
} from "../shared/api/contracts";
import type { ScanSession } from "../shared/storage/sessions";
export const response: RecognitionResponse = {
  schemaVersion: "1.0",
  requestId: "request",
  image: {
    width: 900,
    height: 1000,
    coordinateSpace: "normalized",
    orientationApplied: true,
  },
  bestMatch: null,
  metrics: {
    f1Top1: null,
    f1Top5: null,
    scope: "evaluation-dataset",
    datasetId: null,
    isMock: true,
  },
  detections: [
    {
      id: "bottle",
      polygon: [
        [0.1, 0.1],
        [0.8, 0.1],
        [0.8, 0.9],
        [0.1, 0.9],
      ],
      detectionConfidence: null,
      status: "unmatched",
      match: null,
      similar: [],
    },
  ],
  processingTimeMs: null,
};
export const prepared = {
  blob: new Blob(["photo"], { type: "image/jpeg" }),
  width: 900,
  height: 1000,
};
export const session: ScanSession = {
  scanId: "session",
  imageBlob: prepared.blob,
  imageDimensions: { width: 900, height: 1000 },
  recognitionResult: response,
  selectedDetectionId: "bottle",
  sheetView: "similar",
  selectedSimilarWineSlug: null,
  relevantScrollPositions: { page: 10, similar: 180, sheet: 0 },
  createdAt: Date.now(),
  source: "demo",
};
export const stub = (
  recognize: RecognitionService["recognize"],
): RecognitionService => ({
  recognize,
  getWine: async () => {
    throw Error("unused");
  },
  searchWines: async () => [],
});
