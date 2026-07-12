import { Component, OnDestroy, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { ChartPanelComponent } from './chart-panel/chart-panel.component';
import { StrainApiService } from '../../core/services/strain-api.service';
import { Detector, DenoisedStrainResult, SeriesPoint } from '../../core/models/strain.models';

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

const LOADING_STEPS = [
  { id: 'backend' as const, label: 'Backend connected' },
  { id: 'gwosc' as const, label: 'GWOSC strain download' },
  { id: 'model' as const, label: 'AI noise subtraction' },
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

  readonly loading = signal(false);
  readonly elapsedSeconds = signal(0);
  readonly error = signal<string | null>(null);
  readonly result = signal<DenoisedStrainResult | null>(null);

  readonly rawSeries = computed<SeriesPoint[]>(() => this.buildSeries((r) => r.rawStrain));
  readonly noiseSeries = computed<SeriesPoint[]>(() => this.buildSeries((r) => r.predictedNoise));
  readonly residualSeries = computed<SeriesPoint[]>(() => this.buildSeries((r) => r.residual));

  readonly loadingPhase = computed(() => {
    const elapsed = this.elapsedSeconds();
    let phase = LOADING_PHASES[0];
    for (const candidate of LOADING_PHASES) {
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
    const stepOrder = LOADING_STEPS.map((step) => step.id);
    const activeIndex = stepOrder.indexOf(activeStepId);

    return LOADING_STEPS.map((step, index) => ({
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
      return 'Pick an event below, then press Run analysis.';
    }
    const elapsed = this.formattedElapsed();
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

  readonly resultSummary = computed(() => {
    const result = this.result();
    if (!result) return null;

    const observatory =
      result.detector === 'H1'
        ? 'Hanford, Washington'
        : result.detector === 'L1'
          ? 'Livingston, Louisiana'
          : 'Virgo, Italy';

    return `Observatory: ${observatory} · Window: ${this.duration()} seconds · Samples: ${result.rawStrain.length.toLocaleString()}`;
  });

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
    this.startTimer();

    this.strainApi
      .getDenoisedStrain({
        gpsTime: this.gpsTime(),
        detector: this.detector(),
        duration: this.duration(),
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
