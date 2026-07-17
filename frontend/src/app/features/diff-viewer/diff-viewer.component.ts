import { Component, OnDestroy, OnInit, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Subscription } from 'rxjs';

import { ChartPanelComponent } from './chart-panel/chart-panel.component';
import { BackendHealth, StrainApiService } from '../../core/services/strain-api.service';
import {
  CatalogEventSummary,
  CoincidenceResult,
  Detector,
  DenoisedStrainResult,
  DetectionStats,
  SeriesPoint,
  SyntheticStrategy,
} from '../../core/models/strain.models';

const CATALOG_OPTIONS = [
  { id: 'GWTC', label: 'All GWTC' },
  { id: 'GWTC-3-confident', label: 'GWTC-3 confident' },
  { id: 'GWTC-2.1-confident', label: 'GWTC-2.1 confident' },
  { id: 'GWTC-1-confident', label: 'GWTC-1 confident' },
] as const;

function pickPreferredDetector(detectors: Detector[]): Detector {
  if (detectors.includes('H1')) return 'H1';
  if (detectors.includes('L1')) return 'L1';
  if (detectors.includes('V1')) return 'V1';
  return 'H1';
}

function coincidenceDetectorsFromPublicStrain(detectors: Detector[]): Detector[] {
  const pair = (['H1', 'L1'] as Detector[]).filter((detector) => detectors.includes(detector));
  return pair.length >= 2 ? pair : [];
}

interface LoadingPhase {
  afterSeconds: number;
  message: string;
  hint?: string;
  stepId: 'backend' | 'gwosc' | 'model' | 'search';
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
    stepId: 'search',
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
  { id: 'search' as const, label: 'Blind excess-power search' },
];

interface KnownEvent {
  id: string;
  title: string;
  subtitle: string;
  gpsTime: number;
  detector: Detector;
  /** Detectors for coherent coincidence; omit to default H1+L1. Empty = skip. */
  coincidenceDetectors?: Detector[];
  coincidenceNote?: string;
  /** Plain-language tip shown while this preset GPS is selected. */
  tip: string;
}

/** Curated demo presets (not loaded from GWOSC /events). */
const KNOWN_EVENTS: KnownEvent[] = [
  {
    id: 'GW150914',
    title: 'First black hole collision',
    subtitle: 'Historic 2015 discovery',
    gpsTime: 1126259462.4,
    detector: 'H1',
    tip:
      'Look for a residual spike near t=0. Production Yes with Independent No is normal: Hanford residual is loud; Livingston alone stays below the single-detector gate; the coherent scan still passes.',
  },
  {
    id: 'GW170817',
    title: 'Neutron stars colliding',
    subtitle: 'Also seen as light in the sky',
    gpsTime: 1187008882.4,
    detector: 'H1',
    tip:
      'Expect an envelope veto, not a dual-IFO detection. Livingston has a huge glitch; coherent EP can look enormous while peak timing fails the glitch gate.',
  },
  {
    id: 'GW190425',
    title: 'Another neutron star event',
    subtitle: 'Best viewed from Livingston',
    gpsTime: 1240215503.0,
    detector: 'L1',
    // H1 was offline; GWOSC can still return a frame and invent a dual-IFO trigger.
    coincidenceDetectors: [],
    coincidenceNote:
      'H1 was offline for GW190425 — dual-detector coincidence is not valid. Use L1 single-detector residual search only (matches the freeze).',
    tip:
      'Hanford was offline. Use Livingston only — dual-detector coincidence is skipped so the UI cannot invent a false H1+L1 trigger.',
  },
];

function knownEventForGps(gpsTime: number): KnownEvent | undefined {
  return KNOWN_EVENTS.find((event) => Math.abs(event.gpsTime - gpsTime) < 0.05);
}

/** Argmax of short boxcar energy on |residual|; returns time in seconds from event. */
function findResidualPeakTimeSeconds(points: SeriesPoint[], boxcar = 8): number | null {
  if (points.length < boxcar + 2) return null;
  let bestIndex = 0;
  let bestEnergy = -1;
  for (let i = 0; i <= points.length - boxcar; i += 1) {
    let energy = 0;
    for (let j = 0; j < boxcar; j += 1) {
      const value = points[i + j].value;
      energy += value * value;
    }
    if (energy > bestEnergy) {
      bestEnergy = energy;
      bestIndex = i + Math.floor(boxcar / 2);
    }
  }
  return points[bestIndex]?.time ?? null;
}

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
export class DiffViewerComponent implements OnInit, OnDestroy {
  private readonly strainApi = inject(StrainApiService);
  private timerId: ReturnType<typeof setInterval> | null = null;
  private inFlight: Subscription | null = null;

  readonly knownEvents = KNOWN_EVENTS;
  readonly catalogOptions = CATALOG_OPTIONS;

  readonly gpsTime = signal(KNOWN_EVENTS[0].gpsTime);
  readonly detector = signal<Detector>('H1');
  readonly duration = signal(4);
  readonly selectedEventId = signal(KNOWN_EVENTS[0].id);
  readonly syntheticMode = signal(false);
  readonly syntheticStrategy = signal<SyntheticStrategy>('oracle');

  readonly catalogId = signal<string>(CATALOG_OPTIONS[0].id);
  readonly catalogEvents = signal<CatalogEventSummary[]>([]);
  readonly catalogQuery = signal('');
  readonly catalogLoading = signal(false);
  readonly catalogSelecting = signal(false);
  readonly catalogError = signal<string | null>(null);
  readonly catalogSelectWarning = signal<string | null>(null);
  readonly selectedCatalogName = signal<string | null>(null);
  /** From GWOSC strain metadata; null = use preset/default H1+L1. */
  readonly coincidenceDetectorsOverride = signal<Detector[] | null>(null);

  readonly loading = signal(false);
  readonly coincidenceLoading = signal(false);
  readonly cacheClearing = signal(false);
  readonly cacheMessage = signal<string | null>(null);
  readonly elapsedSeconds = signal(0);
  readonly error = signal<string | null>(null);
  readonly coincidenceWarning = signal<string | null>(null);
  readonly result = signal<DenoisedStrainResult | null>(null);
  readonly coincidence = signal<CoincidenceResult | null>(null);
  readonly engineHealth = signal<BackendHealth | null>(null);
  readonly engineHealthError = signal<string | null>(null);

  readonly healthBadge = computed(() => {
    if (this.engineHealthError()) {
      return { kind: 'down' as const, label: 'Backend unreachable' };
    }
    const health = this.engineHealth();
    if (!health) {
      return { kind: 'pending' as const, label: 'Checking services…' };
    }
    const ml = health.mlEngine;
    if (!ml || ml.status === 'unreachable') {
      return { kind: 'warn' as const, label: 'ML engine unreachable' };
    }
    if (ml.checkpoint_loaded === false) {
      return { kind: 'warn' as const, label: 'Checkpoint missing' };
    }
    return { kind: 'ok' as const, label: 'ML ready · checkpoint loaded' };
  });

  readonly rawSeries = computed<SeriesPoint[]>(() => this.buildSeries((r) => r.rawStrain));
  readonly noiseSeries = computed<SeriesPoint[]>(() => this.buildSeries((r) => r.predictedNoise));
  readonly residualSeries = computed<SeriesPoint[]>(() => this.buildSeries((r) => r.residual));
  readonly groundTruthSeries = computed<SeriesPoint[]>(() => {
    const result = this.result();
    if (!result?.groundTruthSignal?.length) return [];
    return toSeries(result.groundTruthSignal, result.sampleRate, result.t0, result.gpsTime);
  });

  readonly markEventTime = computed(() => {
    const result = this.result();
    return !!result && !result.synthetic;
  });

  readonly residualPeakTimeSeconds = computed(() => {
    const result = this.result();
    if (!result || result.synthetic) return null;
    return findResidualPeakTimeSeconds(this.residualSeries());
  });

  readonly selectedEventTip = computed(() => {
    if (this.syntheticMode()) {
      return 'Synthetic mode buries a known burst in fake noise. Compare Step 3 (residual) to Step 4 (ground truth) — they should match in oracle mode.';
    }
    const known =
      knownEventForGps(this.gpsTime()) ?? this.knownEvents.find((e) => e.id === this.selectedEventId());
    if (known?.tip) return known.tip;
    const catalogName = this.selectedCatalogName();
    if (catalogName) {
      return (
        `Catalog event ${catalogName}. Charts use the observatory in Advanced; ` +
        `coincidence runs only when both H1 and L1 have public strain for this GPS.`
      );
    }
    return null;
  });

  readonly filteredCatalogEvents = computed(() => {
    const query = this.catalogQuery().trim().toLowerCase();
    const events = this.catalogEvents();
    const filtered = query
      ? events.filter(
          (event) =>
            event.name.toLowerCase().includes(query) || String(event.gpsTime).includes(query),
        )
      : events;
    return filtered.slice(0, 40);
  });

  readonly showRetryCoincidence = computed(
    () =>
      !!this.coincidenceWarning() &&
      !!this.result() &&
      !this.result()?.synthetic &&
      !this.coincidenceLoading() &&
      !this.loading(),
  );

  readonly coincidenceDetectorRows = computed(() => {
    const coincidence = this.coincidence();
    const detection = this.detectionStats();
    if (!coincidence) return [];
    const residualThreshold = detection?.thresholds.excessPowerResidual ?? null;
    return coincidence.detectors.map((row) => {
      const residualEp = row.residualExcessPower;
      const residualPct =
        residualThreshold !== null && residualEp !== null && residualEp !== undefined
          ? this.thresholdPercent(residualEp, residualThreshold)
          : null;
      return { ...row, residualThresholdPercent: residualPct };
    });
  });

  readonly residualAdvantageLine = computed(() => {
    const detection = this.detectionStats();
    if (!detection) return null;
    if (detection.residualDetected && !detection.rawDetected) {
      return (
        `Advantage: residual is ${this.formatThresholdPercent(detection.residualExcessPower, detection.thresholds.excessPowerResidual)} ` +
        `while raw is only ${this.formatThresholdPercent(detection.rawExcessPower, detection.thresholds.excessPowerRaw)} — ` +
        `subtraction revealed excess power that a raw-strain search missed.`
      );
    }
    return null;
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

    const known = knownEventForGps(result.gpsTime);
    const eventLabel = known ? `${known.id} · ` : '';

    return (
      `${eventLabel}GPS ${result.gpsTime} · Observatory: ${observatory} · ` +
      `Window: ${this.duration()} seconds · Samples: ${result.rawStrain.length.toLocaleString()}`
    );
  });

  readonly staleResultWarning = computed(() => {
    const result = this.result();
    if (!result || this.loading()) return null;
    if (Math.abs(result.gpsTime - this.gpsTime()) > 0.05) {
      return (
        `Stale view: charts are GPS ${result.gpsTime} but the form asks for ${this.gpsTime()}. ` +
        `Press Run analysis again.`
      );
    }
    const known = knownEventForGps(result.gpsTime);
    const selected = this.selectedEventId();
    if (known && selected && known.id !== selected) {
      return `Stale view: showing ${known.id} while ${selected} is selected. Press Run analysis again.`;
    }
    return null;
  });

  readonly detectionStats = computed((): DetectionStats | null => this.result()?.detection ?? null);

  readonly coincidencePanelTitle = computed(() => {
    const coincidence = this.coincidence();
    if (!coincidence?.detectors?.length) {
      return 'Multi-detector coincidence';
    }
    const labels = coincidence.detectors.map((detector) => detector.detector);
    return `${labels.join(' + ')} coincidence`;
  });

  readonly coincidenceSummary = computed(() => {
    const coincidence = this.coincidence();
    if (!coincidence) return null;

    const available = coincidence.detectors.filter((detector) => detector.available);
    if (!available.length) {
      return 'Multi-detector search unavailable for this GPS time.';
    }

    const coherent = coincidence.coherent;
    if (coincidence.residualCoincident && coherent) {
      const lag = coherent.bestLagMs >= 0 ? `+${coherent.bestLagMs.toFixed(1)}` : coherent.bestLagMs.toFixed(1);
      const asymmetric =
        !coincidence.independentResidualCoincident
          ? ' Independent=no is expected when only one IFO clears the single-detector gate — production uses the coherent scan.'
          : '';
      return (
        `Production coherent residual coincidence — EP ${this.formatExcessPower(coherent.coherentExcessPower)}, ` +
        `lag ${lag} ms (envelope peak dt ${coherent.peakDtMs.toFixed(1)} ms).${asymmetric}`
      );
    }
    if (coincidence.residualCoincident) {
      return `Residual coincidence at GPS ${coincidence.gpsTime}.`;
    }
    if (coherent && !coherent.envelopeOk) {
      return (
        `Envelope veto: coherent EP ${this.formatExcessPower(coherent.coherentExcessPower)} but peak dt ` +
        `${coherent.peakDtMs.toFixed(1)} ms exceeds ±${coherent.maxEnvelopeDtMs} ms (likely single-IFO glitch).`
      );
    }
    if (coherent && !coherent.timingOk) {
      return (
        `Coherent EP ${this.formatExcessPower(coherent.coherentExcessPower)} but lag outside ` +
        `±${coherent.maxLagMs} ms window (best lag ${coherent.bestLagMs.toFixed(1)} ms).`
      );
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
    this.coincidenceWarning.set(null);
    this.cacheMessage.set(null);
  }

  retryCoincidence(): void {
    const result = this.result();
    if (!result || result.synthetic || this.coincidenceLoading()) return;

    const { detectors: coincidenceDetectors, note } = this.resolveCoincidenceDetectors(result.gpsTime);
    if (coincidenceDetectors.length < 2) {
      this.coincidenceWarning.set(
        note ?? 'Dual-detector coincidence is not available for this GPS time.',
      );
      return;
    }

    const detectorLabel = coincidenceDetectors.join('+');
    this.inFlight?.unsubscribe();
    this.coincidenceWarning.set(null);
    this.coincidenceLoading.set(true);
    this.startTimer();

    this.inFlight = this.strainApi
      .getStrainCoincidence({
        gpsTime: result.gpsTime,
        duration: this.duration(),
        detectors: coincidenceDetectors,
      })
      .subscribe({
        next: (coincidence) => {
          this.coincidence.set(coincidence);
          this.coincidenceLoading.set(false);
          this.stopTimer();
          this.refreshHealth();
        },
        error: (err) => {
          this.coincidenceWarning.set(
            `${detectorLabel} coincidence unavailable — ${this.toFriendlyCoincidenceError(err, detectorLabel)}`,
          );
          this.coincidenceLoading.set(false);
          this.stopTimer();
          this.refreshHealth();
        },
      });
  }

  clearBackendCache(): void {
    if (this.cacheClearing()) return;
    this.cacheClearing.set(true);
    this.cacheMessage.set(null);
    this.strainApi.clearStrainCache().subscribe({
      next: () => {
        this.cacheClearing.set(false);
        this.cacheMessage.set('Backend strain cache cleared. Next Run analysis will re-download if needed.');
      },
      error: (err) => {
        this.cacheClearing.set(false);
        this.cacheMessage.set(this.toFriendlyError(err));
      },
    });
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
    this.selectedCatalogName.set(null);
    this.coincidenceDetectorsOverride.set(null);
    this.gpsTime.set(event.gpsTime);
    this.detector.set(event.detector);
    this.result.set(null);
    this.coincidence.set(null);
    this.coincidenceWarning.set(null);
    this.coincidenceLoading.set(false);
    this.cacheMessage.set(null);
    this.error.set(null);
    this.fetch();
  }

  onGpsTimeChange(value: number): void {
    this.gpsTime.set(Number(value));
    this.selectedCatalogName.set(null);
    this.coincidenceDetectorsOverride.set(null);
    const known = knownEventForGps(Number(value));
    this.selectedEventId.set(known?.id ?? '');
  }

  onCatalogIdChange(catalogId: string): void {
    this.catalogId.set(catalogId);
    this.loadCatalog();
  }

  loadCatalog(): void {
    this.catalogLoading.set(true);
    this.catalogError.set(null);
    this.catalogSelectWarning.set(null);
    this.strainApi.getEventCatalog(this.catalogId()).subscribe({
      next: (events) => {
        this.catalogEvents.set(events);
        this.catalogLoading.set(false);
      },
      error: (err) => {
        this.catalogEvents.set([]);
        this.catalogLoading.set(false);
        this.catalogError.set(this.toFriendlyError(err));
      },
    });
  }

  selectCatalogEvent(event: CatalogEventSummary): void {
    if (this.catalogSelecting() || this.loading() || this.coincidenceLoading()) return;

    this.catalogSelecting.set(true);
    this.catalogSelectWarning.set(null);
    this.selectedCatalogName.set(event.name);
    this.selectedEventId.set('');
    this.gpsTime.set(event.gpsTime);
    this.result.set(null);
    this.coincidence.set(null);
    this.coincidenceWarning.set(null);
    this.coincidenceLoading.set(false);
    this.cacheMessage.set(null);
    this.error.set(null);

    this.strainApi.getEventMetadata(event.name).subscribe({
      next: (detail) => {
        if (Number.isFinite(detail.gpsTime)) {
          this.gpsTime.set(detail.gpsTime);
        }
        this.detector.set(pickPreferredDetector(detail.detectors));
        this.coincidenceDetectorsOverride.set(coincidenceDetectorsFromPublicStrain(detail.detectors));
        const known = knownEventForGps(this.gpsTime());
        this.selectedEventId.set(known?.id ?? '');
        this.catalogSelecting.set(false);
        this.fetch();
      },
      error: (err) => {
        // GPS from the catalog list is enough to run; default to H1+L1 coincidence.
        this.coincidenceDetectorsOverride.set(['H1', 'L1']);
        this.catalogSelecting.set(false);
        this.catalogSelectWarning.set(
          `Could not load detectors for ${event.name} — using H1 and default coincidence. ${this.toFriendlyError(err)}`,
        );
        this.fetch();
      },
    });
  }

  ngOnInit(): void {
    this.refreshHealth();
    this.loadCatalog();
  }

  refreshHealth(): void {
    this.strainApi.getHealth().subscribe({
      next: (health) => {
        this.engineHealth.set(health);
        this.engineHealthError.set(null);
      },
      error: () => {
        this.engineHealth.set(null);
        this.engineHealthError.set('unreachable');
      },
    });
  }

  ngOnDestroy(): void {
    this.inFlight?.unsubscribe();
    this.stopTimer();
  }

  fetch(): void {
    this.inFlight?.unsubscribe();
    this.loading.set(true);
    this.coincidenceLoading.set(false);
    this.error.set(null);
    this.coincidenceWarning.set(null);
    this.result.set(null);
    this.coincidence.set(null);
    this.startTimer();

    const query = {
      gpsTime: this.gpsTime(),
      detector: this.detector(),
      duration: this.duration(),
    };

    // Keep the event pill in sync with the GPS actually being requested.
    const knownForQuery = knownEventForGps(query.gpsTime);
    this.selectedEventId.set(knownForQuery?.id ?? '');
    if (knownForQuery) {
      this.selectedCatalogName.set(null);
    }

    if (this.syntheticMode()) {
      this.inFlight = this.strainApi
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

    const { detectors: coincidenceDetectors, note: coincidenceNote } =
      this.resolveCoincidenceDetectors(query.gpsTime);

    if (coincidenceDetectors.length < 2) {
      if (coincidenceNote) {
        this.coincidenceWarning.set(coincidenceNote);
      }
      this.inFlight = this.strainApi.getStrainDetection(query).subscribe({
        next: (strain) => {
          this.result.set(strain);
          this.coincidence.set(null);
          this.finishLoading();
        },
        error: (err) => {
          this.error.set(this.toFriendlyError(err));
          this.finishLoading();
        },
      });
      return;
    }

    // Run single-detector detect first so charts appear as soon as H1/L1 is ready,
    // then coincidence (two GWOSC downloads) with its own longer timeout budget.
    const detectorLabel = coincidenceDetectors.join('+');
    this.inFlight = this.strainApi.getStrainDetection(query).subscribe({
      next: (strain) => {
        this.result.set(strain);
        this.loading.set(false);
        this.coincidenceLoading.set(true);
        this.coincidenceWarning.set(null);

        this.inFlight = this.strainApi
          .getStrainCoincidence({
            gpsTime: query.gpsTime,
            duration: query.duration,
            detectors: coincidenceDetectors,
          })
          .subscribe({
            next: (coincidence) => {
              this.coincidence.set(coincidence);
              this.coincidenceLoading.set(false);
              this.stopTimer();
              this.refreshHealth();
            },
            error: (err) => {
              this.coincidenceWarning.set(
                `${detectorLabel} coincidence unavailable — ${this.toFriendlyCoincidenceError(err, detectorLabel)}`,
              );
              this.coincidenceLoading.set(false);
              this.stopTimer();
              this.refreshHealth();
            },
          });
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
    this.refreshHealth();
  }

  private stopTimer(): void {
    if (this.timerId) {
      clearInterval(this.timerId);
      this.timerId = null;
    }
  }

  private resolveCoincidenceDetectors(gpsTime: number): {
    detectors: Detector[];
    note?: string;
  } {
    const known = knownEventForGps(gpsTime);
    if (known && known.coincidenceDetectors !== undefined) {
      return { detectors: known.coincidenceDetectors, note: known.coincidenceNote };
    }
    const override = this.coincidenceDetectorsOverride();
    if (override !== null) {
      return {
        detectors: override,
        note:
          override.length < 2
            ? 'Dual-detector coincidence skipped — fewer than two of H1/L1 have public strain for this event.'
            : undefined,
      };
    }
    return { detectors: ['H1', 'L1'] };
  }

  private formatElapsed(totalSeconds: number): string {
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  }

  private toFriendlyError(err: { error?: { error?: string; details?: string } }): string {
    const message = err?.error?.error ?? '';
    const details = err?.error?.details ?? '';

    if (message.includes('ML inference engine') && details.toLowerCase().includes('timeout')) {
      return 'The analysis took too long. The servers may be downloading data — try again in a moment.';
    }
    if (details.includes('offline') || details.includes('no valid data')) {
      return 'That observatory had no data for this moment. Try another observatory or pick a preset event.';
    }
    if (details && message.includes('ML inference engine')) {
      return details;
    }
    if (message && details && details !== message) {
      return `${message}: ${details}`;
    }
    if (message) {
      return message;
    }
    if (details) {
      return details;
    }
    return 'Could not reach the backend. Make sure the backend and ML engine are running.';
  }

  private toFriendlyCoincidenceError(
    err: { error?: { error?: string; details?: string } },
    detectorLabel: string,
  ): string {
    const details = (err?.error?.details ?? '').toLowerCase();
    const message = (err?.error?.error ?? '').toLowerCase();
    if (message.includes('timeout') || details.includes('timeout')) {
      return (
        `${detectorLabel} needs two GWOSC downloads and can take several minutes on a cold fetch. ` +
        `Press Run analysis again — the single-detector result is often cached and coincidence can finish alone.`
      );
    }
    return this.toFriendlyError(err);
  }
}
