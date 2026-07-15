export type Detector = 'H1' | 'L1' | 'V1';
export type SyntheticStrategy = 'oracle' | 'model';

export interface DenoiseQuery {
  gpsTime: number;
  detector: Detector;
  duration: number;
  synthetic?: boolean;
  syntheticStrategy?: SyntheticStrategy;
}

export interface DetectionThresholds {
  excessPowerRaw: number;
  excessPowerResidual: number;
}

export interface DetectionStats {
  rawExcessPower: number;
  residualExcessPower: number;
  excessPowerRatio: number;
  rawDetected: boolean;
  residualDetected: boolean;
  falseAlarmRate: number;
  thresholds: DetectionThresholds;
  calibrationNote: string;
  checkpointLoaded: boolean;
}

export interface DenoisedStrainResult {
  detector: Detector;
  gpsTime: number;
  sampleRate: number;
  t0: number;
  rawStrain: number[];
  predictedNoise: number[];
  residual: number[];
  cached: boolean;
  synthetic: boolean;
  syntheticStrategy?: SyntheticStrategy;
  groundTruthSignal?: number[];
  groundTruthNoise?: number[];
  detection?: DetectionStats;
}

export interface CoincidenceDetectorResult {
  detector: Detector;
  available: boolean;
  rawExcessPower: number | null;
  residualExcessPower: number | null;
  rawDetected: boolean;
  residualDetected: boolean;
  error?: string;
}

export interface CoincidenceResult {
  gpsTime: number;
  duration: number;
  detectors: CoincidenceDetectorResult[];
  rawCoincident: boolean;
  residualCoincident: boolean;
  falseAlarmRate: number;
  calibrationNote: string;
  checkpointLoaded: boolean;
  cached?: boolean;
}

export interface SeriesPoint {
  time: number;
  value: number;
}
