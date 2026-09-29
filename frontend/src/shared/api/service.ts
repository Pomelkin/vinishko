import type { RecognitionService } from "./contracts";
import { HttpRecognitionService } from "./http-recognition-service";
import { MockRecognitionService } from "./mock-recognition-service";
export const isMockMode = import.meta.env.VITE_API_MODE === "mock";
export const service: RecognitionService = isMockMode
  ? new MockRecognitionService()
  : new HttpRecognitionService(import.meta.env.VITE_API_BASE_URL || "/api");
export { demoScenarios, type DemoScenario } from "./mock-recognition-service";
