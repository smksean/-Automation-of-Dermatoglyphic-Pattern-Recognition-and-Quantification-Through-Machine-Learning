export const broadLabels = {
  arch: "Arch",
  left_slant_loop: "Left-slant loop",
  right_slant_loop: "Right-slant loop",
  whorl: "Whorl",
} as const;

export const subtypeLabels = {
  plain_arch: "Plain arch",
  tented_arch: "Tented arch",
  plain_whorl: "Plain whorl",
  central_pocket_loop_whorl: "Central pocket loop whorl",
  double_loop_whorl: "Double loop whorl",
} as const;

export type BroadClass = keyof typeof broadLabels;
export type SubtypeClass = keyof typeof subtypeLabels;

export const patternIntensityContributions: Record<BroadClass, 0 | 1 | 2> = {
  arch: 0,
  left_slant_loop: 1,
  right_slant_loop: 1,
  whorl: 2,
};

export function patternFamily(value: BroadClass): "arch" | "loop" | "whorl" {
  if (value === "left_slant_loop" || value === "right_slant_loop") return "loop";
  return value;
}

export type SubtypePrediction = {
  predictedSubtype: SubtypeClass;
  score: number;
  probabilities: Partial<Record<SubtypeClass, number>>;
  needsReview: boolean;
  reviewReasons: string[];
  scope: string;
};

export type FeatureEndpointStatus = {
  status: "validation_required" | "not_reported";
  reason: string;
};

export type FeatureAnalysis = {
  quality: {
    score: number;
    grade: "good" | "review" | "insufficient";
    needsReview: boolean;
    reasons: string[];
  };
  measurements: {
    sourceWidth: number;
    sourceHeight: number;
    foregroundCoverage: number;
    ridgeContrast: number;
    sharpness: number;
    orientationCoherence: number;
    ridgeDensityProxy: number;
  };
  orientationBlocks: Array<{
    x: number;
    y: number;
    angleDegrees: number;
    coherence: number;
  }>;
  overlayDataUrl: string | null;
  minutiae: FeatureEndpointStatus & {
    ridgeEndings: number | null;
    bifurcations: number | null;
  };
  landmarks: FeatureEndpointStatus & {
    cores: number | null;
    deltas: number | null;
  };
  ridgeCount: FeatureEndpointStatus & { value: number | null };
  scope: string;
};

export type PredictionResponse = {
  predictedClass: BroadClass;
  score: number;
  probabilities: Record<BroadClass, number>;
  foldPredictions: BroadClass[];
  agreement: number;
  topTwoMargin: number;
  needsReview: boolean;
  reviewReasons: string[];
  subtype: SubtypePrediction | null;
  featureAnalysis: FeatureAnalysis;
  processingSeconds: number;
};

export type BatchPredictionResponse = {
  fingerResults: Array<{ fingerId: string; prediction: PredictionResponse }>;
  counts: { arch: number; loop: number; whorl: number };
  pii: number;
  reviewCount: number;
  featureReviewCount: number;
  processingSeconds: number;
  scope: string;
};

export type ApiError = {
  error: string;
  detail?: string;
};
