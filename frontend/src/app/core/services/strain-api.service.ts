import { HttpClient, HttpParams } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable, map } from 'rxjs';

import { environment } from '../../environments/environment';
import {
  CoincidenceResult,
  DenoiseQuery,
  DenoisedStrainResult,
  DetectionStats,
  Detector,
  SyntheticStrategy,
} from '../models/strain.models';

interface DenoisedStrainResponse {
  detector: string;
  gps_time: number;
  sample_rate: number;
  t0: number;
  raw_strain: number[];
  predicted_noise: number[];
  residual: number[];
  cached: boolean;
  synthetic?: boolean;
  synthetic_strategy?: SyntheticStrategy;
  ground_truth_signal?: number[];
  ground_truth_noise?: number[];
}

interface DetectStrainResponse extends DenoisedStrainResponse {
  raw_excess_power: number;
  residual_excess_power: number;
  excess_power_improvement: number;
  raw_detected: boolean;
  residual_detected: boolean;
  false_alarm_rate: number;
  thresholds: {
    excess_power_raw: number;
    excess_power_residual: number;
  };
  calibration_note: string;
  checkpoint_loaded: boolean;
}

interface CoincidenceResponse {
  gps_time: number;
  duration: number;
  detectors: Array<{
    detector: string;
    available: boolean;
    raw_excess_power: number | null;
    residual_excess_power: number | null;
    raw_detected: boolean;
    residual_detected: boolean;
    error?: string;
  }>;
  raw_coincident: boolean;
  independent_residual_coincident: boolean;
  residual_coincident: boolean;
  coherent?: {
    coherent_excess_power: number;
    best_lag_ms: number;
    best_polarity: number;
    peak_dt_ms: number;
    timing_ok: boolean;
    coherent_detected: boolean;
    max_lag_ms: number;
  } | null;
  false_alarm_rate: number;
  calibration_note: string;
  checkpoint_loaded: boolean;
  cached?: boolean;
}

export interface BackendHealth {
  status: string;
  service?: string;
  mlEngine?: {
    status?: string;
    device?: string;
    model_loaded?: boolean;
    checkpoint_loaded?: boolean;
    checkpoint_path?: string;
  };
}

@Injectable({ providedIn: 'root' })
export class StrainApiService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = environment.apiBaseUrl;

  getHealth(): Observable<BackendHealth> {
    return this.http.get<BackendHealth>(`${this.baseUrl}/health`);
  }

  getDenoisedStrain(query: DenoiseQuery): Observable<DenoisedStrainResult> {
    let params = new HttpParams()
      .set('gpsTime', query.gpsTime)
      .set('detector', query.detector)
      .set('duration', query.duration);

    if (query.synthetic) {
      params = params
        .set('synthetic', 'true')
        .set('syntheticStrategy', query.syntheticStrategy ?? 'oracle');
    }

    return this.http
      .get<DenoisedStrainResponse>(`${this.baseUrl}/strain/denoised`, { params })
      .pipe(map((response) => this.toResult(response)));
  }

  getStrainDetection(query: Omit<DenoiseQuery, 'synthetic' | 'syntheticStrategy'>): Observable<DenoisedStrainResult> {
    const params = new HttpParams()
      .set('gpsTime', query.gpsTime)
      .set('detector', query.detector)
      .set('duration', query.duration);

    return this.http
      .get<DetectStrainResponse>(`${this.baseUrl}/strain/detect`, { params })
      .pipe(map((response) => this.toDetectResult(response)));
  }

  getStrainCoincidence(
    query: Omit<DenoiseQuery, 'synthetic' | 'syntheticStrategy' | 'detector'> & {
      detectors?: Detector[];
    },
  ): Observable<CoincidenceResult> {
    const params = new HttpParams()
      .set('gpsTime', query.gpsTime)
      .set('duration', query.duration)
      .set('detectors', (query.detectors ?? ['H1', 'L1']).join(','));

    return this.http
      .get<CoincidenceResponse>(`${this.baseUrl}/strain/detect/coincidence`, { params })
      .pipe(map((response) => this.toCoincidenceResult(response)));
  }

  private toCoincidenceResult(response: CoincidenceResponse): CoincidenceResult {
    const coherent = response.coherent
      ? {
          coherentExcessPower: response.coherent.coherent_excess_power,
          bestLagMs: response.coherent.best_lag_ms,
          bestPolarity: response.coherent.best_polarity,
          peakDtMs: response.coherent.peak_dt_ms,
          timingOk: response.coherent.timing_ok,
          coherentDetected: response.coherent.coherent_detected,
          maxLagMs: response.coherent.max_lag_ms,
        }
      : null;

    return {
      gpsTime: response.gps_time,
      duration: response.duration,
      detectors: response.detectors.map((detector) => ({
        detector: detector.detector as Detector,
        available: detector.available,
        rawExcessPower: detector.raw_excess_power,
        residualExcessPower: detector.residual_excess_power,
        rawDetected: detector.raw_detected,
        residualDetected: detector.residual_detected,
        error: detector.error,
      })),
      rawCoincident: response.raw_coincident,
      independentResidualCoincident: response.independent_residual_coincident,
      residualCoincident: response.residual_coincident,
      coherent,
      falseAlarmRate: response.false_alarm_rate,
      calibrationNote: response.calibration_note,
      checkpointLoaded: response.checkpoint_loaded,
      cached: response.cached,
    };
  }

  private toDetectResult(response: DetectStrainResponse): DenoisedStrainResult {
    return {
      ...this.toResult(response),
      detection: this.toDetectionStats(response),
    };
  }

  private toDetectionStats(response: DetectStrainResponse): DetectionStats {
    return {
      rawExcessPower: response.raw_excess_power,
      residualExcessPower: response.residual_excess_power,
      excessPowerRatio: response.excess_power_improvement,
      rawDetected: response.raw_detected,
      residualDetected: response.residual_detected,
      falseAlarmRate: response.false_alarm_rate,
      thresholds: {
        excessPowerRaw: response.thresholds.excess_power_raw,
        excessPowerResidual: response.thresholds.excess_power_residual,
      },
      calibrationNote:
        response.calibration_note ??
        'Noise-only GWOSC background calibration; residual threshold uses artifact-trimmed calibration.',
      checkpointLoaded: response.checkpoint_loaded,
    };
  }

  private toResult(response: DenoisedStrainResponse): DenoisedStrainResult {
    return {
      detector: response.detector as DenoisedStrainResult['detector'],
      gpsTime: response.gps_time,
      sampleRate: response.sample_rate,
      t0: response.t0,
      rawStrain: response.raw_strain,
      predictedNoise: response.predicted_noise,
      residual: response.residual,
      cached: response.cached,
      synthetic: response.synthetic ?? false,
      syntheticStrategy: response.synthetic_strategy,
      groundTruthSignal: response.ground_truth_signal,
      groundTruthNoise: response.ground_truth_noise,
    };
  }
}
