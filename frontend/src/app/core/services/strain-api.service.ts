import { HttpClient, HttpParams } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable, map } from 'rxjs';

import { environment } from '../../environments/environment';
import { DenoiseQuery, DenoisedStrainResult, SyntheticStrategy } from '../models/strain.models';

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

@Injectable({ providedIn: 'root' })
export class StrainApiService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = environment.apiBaseUrl;

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
