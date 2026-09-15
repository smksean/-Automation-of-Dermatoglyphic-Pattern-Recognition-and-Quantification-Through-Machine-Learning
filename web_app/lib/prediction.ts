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

export type SubtypePrediction = {
  predictedSubtype: SubtypeClass;
  score: number;
  probabilities: Partial<Record<SubtypeClass, number>>;
  needsReview: boolean;
  reviewReasons: string[];
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
  processingSeconds: number;
};

export type ApiError = {
  error: string;
  detail?: string;
};
