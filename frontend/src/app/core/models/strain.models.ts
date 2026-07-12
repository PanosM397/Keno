export type Detector = 'H1' | 'L1' | 'V1';

export interface DenoiseQuery {
  gpsTime: number;
  detector: Detector;
  duration: number;
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
}

export interface SeriesPoint {
  time: number;
  value: number;
}
