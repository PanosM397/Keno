export type Detector = 'H1' | 'L1' | 'V1';
export type SyntheticStrategy = 'oracle' | 'model';

export interface DenoiseQuery {
  gpsTime: number;
  detector: Detector;
  duration: number;
  synthetic?: boolean;
  syntheticStrategy?: SyntheticStrategy;
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
}

export interface SeriesPoint {
  time: number;
  value: number;
}
