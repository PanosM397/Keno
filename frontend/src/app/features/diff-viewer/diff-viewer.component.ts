import { Component, OnDestroy, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { forkJoin } from 'rxjs';

import { ChartPanelComponent } from './chart-panel/chart-panel.component';
import { StrainApiService } from '../../core/services/strain-api.service';
import {
  CoincidenceResult,
  Detector,
  DenoisedStrainResult,
  DetectionStats,
  SeriesPoint,
  SyntheticStrategy,
} from '../../core/models/strain.models';

interface LoadingPhase {
  afterSeconds: number;
  message: string;
  hint?: string;
  stepId: 'backend' | 'gwosc' | 'model';
}

const LOADING_PHASES: LoadingPhase[] = [
  {
    afterSeconds: 0,
    stepId: 'backend',
    message: 'Contacting backend…',
    hint: 'First run per event can take 1–3 minutes while strain is downloaded from GWOSC.',
  },
  {
    afterSeconds: 3,
    stepId: 'gwosc',
    message: 'Downloading strain from GWOSC…',
    hint: 'LIGO open data is fetched live — no local copy yet.',
  },
  {
    afterSeconds: 25,
    stepId: 'model',
    message: 'Whitening data and running noise model…',
    hint: 'The U-Net predicts instrumental noise to subtract from the raw signal.',
  },
  {
    afterSeconds: 75,
    stepId: 'model',
    message: 'Still working — large downloads take time…',
    hint: 'Repeat runs use the backend cache and usually finish in seconds.',
  },
];

const SYNTHETIC_LOADING_PHASES: LoadingPhase[] = [
  {
    afterSeconds: 0,
    stepId: 'backend',
    message: 'Generating synthetic strain…',
    hint: 'Built locally from known noise plus an injected burst — no GWOSC download.',
  },
  {
    afterSeconds: 1,
    stepId: 'model',
    message: 'Running validation subtraction…',
    hint: 'Oracle mode uses the true noise mirror; model mode uses the live U-Net.',
  },
];

const SYNTHETIC_LOADING_STEPS = [
  { id: 'backend' as const, label: 'Synthetic strain generated' },
  { id: 'model' as const, label: 'Noise subtraction applied' },
];

const LOADING_STEPS = [
  { id: 'backend' as const, label: 'Backend connected' },
  { id: 'gwosc' as const, label: 'GWOSC strain download' },
  { id: 'model' as const, label: 'AI noise subtraction' },
  { id: 'model' as const, label: 'Blind excess-power search' },
];

interface KnownEvent {
  id: string;
  title: string;
  subtitle: string;
  gpsTime: number;
  detector: Detector;
}

const KNOWN_EVENTS: KnownEvent[] = [
  {
    id: 'GW150914',
    title: 'First black hole collision',
    subtitle: 'Historic 2015 discovery',
    gpsTime: 1126259462.4,
    detector: 'H1',
  },
  {
    id: 'GW170817',
    title: 'Neutron stars colliding',
    subtitle: 'Also seen as light in the sky',
    gpsTime: 1187008882.4,
    detector: 'H1',
  },
  {
    id: 'GW190425',
    title: 'Another neutron star event',
    subtitle: 'Best viewed from Livingston',
    gpsTime: 1240215503.0,
    detector: 'L1',
  },
];

function toSeries(
  values: number[],
  sampleRate: number,
  t0: number,
  gpsTime: number,
): SeriesPoint[] {
  return values.map((value, index) => ({
    time: t0 + index / sampleRate - gpsTime,
    value,
  }));
}

interface SeriesStats {
  mean: number;
  std: number;
  min: number;
  max: number;
}

interface DetectionSummary {
  farLabel: string;
  verdict: string;
  signalInResidual: boolean;
  checkpointLoaded: boolean;
  calibrationNote: string;
  rawThresholdPercent: number;
  residualThresholdPercent: number;
}

interface ResultDiagnostics {
  timeStart: number;
  timeEnd: number;
  lengthsMatch: boolean;
  subtractionVerified: boolean;
  maxSubtractionError: number;
  raw: SeriesStats;
  noise: SeriesStats;
  residual: SeriesStats;
  noiseToRawStdRatio: number;
  modelLikelyUntrained: boolean;
  synthetic: boolean;
  syntheticStrategy?: SyntheticStrategy;
  residualMatchesGroundTruth: boolean | null;
  maxGroundTruthError: number | null;
  signalOverlap: number | null;
  recoveryGrade: 'excellent' | 'good' | 'partial' | 'poor' | null;
  verdict: string;
}

const SYNTHETIC_RECOVERY_EXCELLENT = 0.5;
const SYNTHETIC_RECOVERY_GOOD = 1.5;
const SYNTHETIC_OVERLAP_GOOD = 0.75;

function signalOverlap(residual: number[], groundTruth: number[]): number {
  if (!residual.length || residual.length !== groundTruth.length) return 0;

  const mean = (values: number[]) => values.reduce((sum, v) => sum + v, 0) / values.length;
  const a = groundTruth.map((v) => v - mean(groundTruth));
  const b = residual.map((v) => v - mean(residual));
  const normA = Math.sqrt(a.reduce((sum, v) => sum + v * v, 0));
  const normB = Math.sqrt(b.reduce((sum, v) => sum + v * v, 0));
  if (normA === 0 || normB === 0) return 0;
  return a.reduce((sum, v, i) => sum + v * b[i], 0) / (normA * normB);
}

function summarize(values: number[]): SeriesStats {
  if (!values.length) {
    return { mean: 0, std: 0, min: 0, max: 0 };
  }

  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance =
    values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;

  return {
    mean,
    std: Math.sqrt(variance),
    min: Math.min(...values),
    max: Math.max(...values),
  };
}

@Component({
  selector: 'app-diff-viewer',
  standalone: true,
  imports: [FormsModule, ChartPanelComponent],
  templateUrl: './diff-viewer.component.html',
  styleUrl: './diff-viewer.component.scss',
})
export class DiffViewerComponent implements OnDestroy {
  private readonly strainApi = inject(StrainApiService);
  private timerId: ReturnType<typeof setInterval> | null = null;

  readonly knownEvents = KNOWN_EVENTS;

  readonly gpsTime = signal(KNOWN_EVENTS[0].gpsTime);
  readonly detector = signal<Detector>('H1');
  readonly duration = signal(4);
  readonly selectedEventId = signal(KNOWN_EVENTS[0].id);
  readonly syntheticMode = signal(false);
  readonly syntheticStrategy = signal<SyntheticStrategy>('oracle');

  readonly loading = signal(false);
  readonly elapsedSeconds = signal(0);
  readonly error = signal<string | null>(null);
  readonly result = signal<DenoisedStrainResult | null>(null);
  readonly coincidence = signal<CoincidenceResult | null>(null);

  readonly rawSeries = computed<SeriesPoint[]>(() => this.buildSeries((r) => r.rawStrain));
  readonly noiseSeries = computed<SeriesPoint[]>(() => this.buildSeries((r) => r.predictedNoise));
  readonly residualSeries = computed<SeriesPoint[]>(() => this.buildSeries((r) => r.residual));
  readonly groundTruthSeries = computed<SeriesPoint[]>(() => {
    const result = this.result();
    if (!result?.groundTruthSignal?.length) return [];
    return toSeries(result.groundTruthSignal, result.sampleRate, result.t0, result.gpsTime);
  });

  readonly activeLoadingPhases = computed(() =>
    this.syntheticMode() ? SYNTHETIC_LOADING_PHASES : LOADING_PHASES,
  );

  readonly activeLoadingSteps = computed(() =>
    this.syntheticMode() ? SYNTHETIC_LOADING_STEPS : LOADING_STEPS,
  );

  readonly loadingPhase = computed(() => {
    const elapsed = this.elapsedSeconds();
    let phase = this.activeLoadingPhases()[0];
    for (const candidate of this.activeLoadingPhases()) {
      if (elapsed >= candidate.afterSeconds) {
        phase = candidate;
      }
    }
    return phase;
  });

  readonly formattedElapsed = computed(() => this.formatElapsed(this.elapsedSeconds()));

  readonly loadingProgress = computed(() => {
    const elapsed = this.elapsedSeconds();
    const max = 92;
    const tau = 45;
    return Math.min(max, max * (1 - Math.exp(-elapsed / tau)));
  });

  readonly loadingSteps = computed(() => {
    const activeStepId = this.loadingPhase().stepId;
    const steps = this.activeLoadingSteps();
    const stepOrder = steps.map((step) => step.id);
    const activeIndex = stepOrder.indexOf(activeStepId);

    return steps.map((step, index) => ({
      ...step,
      active: index === activeIndex,
      done: index < activeIndex,
    }));
  });

  readonly loadingStatusMessage = computed((): string => this.loadingPhase().message);

  readonly statusMessage = computed((): string => {
    if (this.loading()) {
      return this.loadingPhase().message;
    }
    if (this.error()) {
      return this.error() ?? 'Something went wrong.';
    }
    const result = this.result();
    if (!result) {
      return this.syntheticMode()
        ? 'Synthetic validation is on — press Run analysis to inject a known burst.'
        : 'Pick an event below, then press Run analysis.';
    }
    const elapsed = this.formattedElapsed();
    if (result.synthetic) {
      const strategy =
        result.syntheticStrategy === 'model'
          ? 'U-Net model mode'
          : 'oracle mode (perfect noise mirror)';
      return `Synthetic validation in ${elapsed} — ${strategy}.`;
    }
    return result.cached
      ? `Loaded from cache in ${elapsed} — this result was fetched earlier.`
      : `Fresh result in ${elapsed} — just downloaded and analyzed.`;
  });

  readonly statusHint = computed(() => {
    if (this.loading()) {
      return this.loadingPhase().hint ?? null;
    }
    return null;
  });

  readonly resultDiagnostics = computed((): ResultDiagnostics | null => {
    const result = this.result();
    if (!result) return null;

    const raw = summarize(result.rawStrain);
    const noise = summarize(result.predictedNoise);
    const residual = summarize(result.residual);
    const timeStart = result.t0 - result.gpsTime;
    const timeEnd = timeStart + result.rawStrain.length / result.sampleRate;

    let maxSubtractionError = 0;
    for (let index = 0; index < result.rawStrain.length; index += 1) {
      const expected = result.rawStrain[index] - result.predictedNoise[index];
      maxSubtractionError = Math.max(
        maxSubtractionError,
        Math.abs(expected - result.residual[index]),
      );
    }

    const lengthsMatch =
      result.rawStrain.length === result.predictedNoise.length &&
      result.rawStrain.length === result.residual.length;
    const subtractionVerified = maxSubtractionError < 1e-6;
    const noiseToRawStdRatio = raw.std > 0 ? noise.std / raw.std : 0;
    const modelLikelyUntrained = !result.synthetic && noiseToRawStdRatio < 0.05;

    let residualMatchesGroundTruth: boolean | null = null;
    let maxGroundTruthError: number | null = null;
    let overlap: number | null = null;
    let recoveryGrade: ResultDiagnostics['recoveryGrade'] = null;

    if (result.groundTruthSignal?.length) {
      maxGroundTruthError = 0;
      for (let index = 0; index < result.groundTruthSignal.length; index += 1) {
        maxGroundTruthError = Math.max(
          maxGroundTruthError,
          Math.abs(result.residual[index] - result.groundTruthSignal[index]),
        );
      }
      overlap = signalOverlap(result.residual, result.groundTruthSignal);

      if (maxGroundTruthError < SYNTHETIC_RECOVERY_EXCELLENT || (overlap ?? 0) >= 0.95) {
        recoveryGrade = 'excellent';
        residualMatchesGroundTruth = true;
      } else if (maxGroundTruthError < SYNTHETIC_RECOVERY_GOOD || (overlap ?? 0) >= SYNTHETIC_OVERLAP_GOOD) {
        recoveryGrade = 'good';
        residualMatchesGroundTruth = true;
      } else if ((overlap ?? 0) >= 0.5) {
        recoveryGrade = 'partial';
        residualMatchesGroundTruth = false;
      } else {
        recoveryGrade = 'poor';
        residualMatchesGroundTruth = false;
      }
    }

    let verdict = 'Pipeline math checks out: arrays align and subtraction is exact.';
    if (result.synthetic) {
      if (result.syntheticStrategy === 'oracle') {
        verdict = residualMatchesGroundTruth
          ? 'Synthetic validation passed: subtracting the known noise recovered the injected burst.'
          : 'Synthetic validation mismatch: residual does not match the injected signal.';
      } else if (recoveryGrade === 'excellent' || recoveryGrade === 'good') {
        verdict =
          'Model recovered the injected burst: noise was subtracted and the signal remains in the residual. Compare Step 3 to Step 4.';
      } else if (recoveryGrade === 'partial') {
        verdict =
          'Partial recovery: the burst shape is visible in the residual but some noise remains. The model is trained; further training can sharpen the match.';
      } else {
        verdict =
          'Weak recovery: the U-Net may not have loaded a trained checkpoint, or needs more training. Check ML engine /health for checkpoint_loaded.';
      }
    } else if (modelLikelyUntrained) {
      verdict +=
        ' The U-Net is still untrained, so predicted noise is nearly flat and the residual looks like a shifted copy of the input.';
    }

    return {
      timeStart,
      timeEnd,
      lengthsMatch,
      subtractionVerified,
      maxSubtractionError,
      raw,
      noise,
      residual,
      noiseToRawStdRatio,
      modelLikelyUntrained,
      synthetic: result.synthetic,
      syntheticStrategy: result.syntheticStrategy,
      residualMatchesGroundTruth,
      maxGroundTruthError,
      signalOverlap: overlap,
      recoveryGrade,
      verdict,
    };
  });

  readonly resultSummary = computed(() => {
    const result = this.result();
    if (!result) return null;

    if (result.synthetic) {
      const strategy =
        result.syntheticStrategy === 'model'
          ? 'Synthetic validation · U-Net on fake data'
          : 'Synthetic validation · oracle subtraction';
      return `${strategy} · Window: ${this.duration()} seconds · Samples: ${result.rawStrain.length.toLocaleString()}`;
    }

    const observatory =
      result.detector === 'H1'
        ? 'Hanford, Washington'
        : result.detector === 'L1'
          ? 'Livingston, Louisiana'
          : 'Virgo, Italy';

    return `Observatory: ${observatory} · Window: ${this.duration()} seconds · Samples: ${result.rawStrain.length.toLocaleString()}`;
  });

  readonly detectionStats = computed((): DetectionStats | null => this.result()?.detection ?? null);

  readonly coincidenceSummary = computed(() => {
    const coincidence = this.coincidence();
    if (!coincidence) return null;

    const available = coincidence.detectors.filter((detector) => detector.available);
    if (!available.length) {
      return 'Multi-detector search unavailable for this GPS time.';
    }

    if (coincidence.residualCoincident) {
      return `H1+L1 residual coincidence at GPS ${coincidence.gpsTime} — both observatories trigger on subtracted strain.`;
    }
    if (coincidence.rawCoincident) {
      return 'Raw excess-power coincidence in both detectors, but not on residuals after subtraction.';
    }

    const triggered = available.filter((detector) => detector.residualDetected).length;
    return `${triggered}/${available.length} detectors trigger on residual excess power at this GPS time.`;
  });

  readonly detectionSummary = computed((): DetectionSummary | null => {
    const detection = this.detectionStats();
    if (!detection) return null;

    const farLabel = `${(detection.falseAlarmRate * 100).toFixed(1)}% FAR`;
    const rawThresholdPercent = this.thresholdPercent(
      detection.rawExcessPower,
      detection.thresholds.excessPowerRaw,
    );
    const residualThresholdPercent = this.thresholdPercent(
      detection.residualExcessPower,
      detection.thresholds.excessPowerResidual,
    );
    const signalInResidual = detection.residualExcessPower > detection.rawExcessPower * 2;

    let verdict = `No trigger at noise-calibrated thresholds (residual ${this.formatPercentValue(residualThresholdPercent)} of threshold).`;
    if (detection.residualDetected && !detection.rawDetected) {
      verdict =
        'Residual-only trigger: subtraction surfaced excess power that raw strain search missed.';
    } else if (detection.residualDetected && detection.rawDetected) {
      verdict = 'Both raw and residual exceed the noise-calibrated excess-power threshold.';
    } else if (!detection.residualDetected && detection.rawDetected) {
      verdict = 'Raw strain triggered, but the residual is below threshold after subtraction.';
    } else if (signalInResidual && residualThresholdPercent >= 50) {
      verdict = `Structured energy in the residual (${this.formatPercentValue(residualThresholdPercent)} of threshold) — consistent with signal surviving subtraction.`;
    } else if (!signalInResidual && detection.residualExcessPower < detection.rawExcessPower) {
      verdict = 'Subtraction lowered excess power — noise-dominated segment at this threshold.';
    }

    if (!detection.checkpointLoaded) {
      verdict += ' Warning: U-Net checkpoint not loaded.';
    }

    return {
      farLabel,
      verdict,
      signalInResidual,
      checkpointLoaded: detection.checkpointLoaded,
      calibrationNote: detection.calibrationNote,
      rawThresholdPercent,
      residualThresholdPercent,
    };
  });

  toggleSyntheticMode(enabled: boolean): void {
    this.syntheticMode.set(enabled);
    this.result.set(null);
    this.coincidence.set(null);
    this.error.set(null);
  }

  formatStat(value: number): string {
    return value.toFixed(4);
  }

  formatExcessPower(value: number): string {
    if (!Number.isFinite(value)) return '—';
    if (value >= 1000) return value.toFixed(1);
    if (value >= 10) return value.toFixed(2);
    return value.toFixed(4);
  }

  formatPercent(value: number): string {
    return `${(value * 100).toFixed(1)}%`;
  }

  thresholdPercent(value: number, threshold: number): number {
    if (!Number.isFinite(value) || !Number.isFinite(threshold) || threshold <= 0) {
      return 0;
    }
    return (value / threshold) * 100;
  }

  formatThresholdPercent(value: number, threshold: number): string {
    const percent = this.thresholdPercent(value, threshold);
    if (percent >= 999) {
      return '>999% of threshold';
    }
    return `${percent.toFixed(0)}% of threshold`;
  }

  formatPercentValue(percent: number): string {
    if (percent >= 999) {
      return '>999%';
    }
    return `${percent.toFixed(0)}%`;
  }

  private buildSeries(select: (r: DenoisedStrainResult) => number[]): SeriesPoint[] {
    const r = this.result();
    if (!r) return [];
    return toSeries(select(r), r.sampleRate, r.t0, r.gpsTime);
  }

  selectEvent(event: KnownEvent): void {
    this.selectedEventId.set(event.id);
    this.gpsTime.set(event.gpsTime);
    this.detector.set(event.detector);
    this.fetch();
  }

  ngOnDestroy(): void {
    this.stopTimer();
  }

  fetch(): void {
    this.loading.set(true);
    this.error.set(null);
    this.coincidence.set(null);
    this.startTimer();

    const query = {
      gpsTime: this.gpsTime(),
      detector: this.detector(),
      duration: this.duration(),
    };

    if (this.syntheticMode()) {
      this.strainApi
        .getDenoisedStrain({
          ...query,
          synthetic: true,
          syntheticStrategy: this.syntheticStrategy(),
        })
        .subscribe({
          next: (result) => {
            this.result.set(result);
            this.finishLoading();
          },
          error: (err) => {
            this.error.set(this.toFriendlyError(err));
            this.finishLoading();
          },
        });
      return;
    }

    forkJoin({
      strain: this.strainApi.getStrainDetection(query),
      coincidence: this.strainApi.getStrainCoincidence({
        gpsTime: query.gpsTime,
        duration: query.duration,
        detectors: ['H1', 'L1'],
      }),
    }).subscribe({
      next: ({ strain, coincidence }) => {
        this.result.set(strain);
        this.coincidence.set(coincidence);
        this.finishLoading();
      },
      error: (err) => {
        this.error.set(this.toFriendlyError(err));
        this.finishLoading();
      },
    });
  }

  private startTimer(): void {
    this.stopTimer();
    this.elapsedSeconds.set(0);
    this.timerId = setInterval(() => {
      this.elapsedSeconds.update((seconds) => seconds + 1);
    }, 1000);
  }

  private finishLoading(): void {
    this.loading.set(false);
    this.stopTimer();
  }

  private stopTimer(): void {
    if (this.timerId) {
      clearInterval(this.timerId);
      this.timerId = null;
    }
  }

  private formatElapsed(totalSeconds: number): string {
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  }

  private toFriendlyError(err: { error?: { error?: string; details?: string } }): string {
    const message = err?.error?.error ?? '';
    const details = err?.error?.details ?? '';

    if (message.includes('ML inference engine') && details.includes('timeout')) {
      return 'The analysis took too long. The servers may be downloading data — try again in a moment.';
    }
    if (details.includes('offline') || details.includes('no valid data')) {
      return 'That observatory had no data for this moment. Try another observatory or pick a preset event.';
    }
    if (message) {
      return message;
    }
    return 'Could not reach the backend. Make sure the backend and ML engine are running.';
  }
}
