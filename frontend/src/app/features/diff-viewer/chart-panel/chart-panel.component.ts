import {
  AfterViewInit,
  Component,
  ElementRef,
  Input,
  OnChanges,
  OnDestroy,
  SimpleChanges,
  ViewChild,
} from '@angular/core';
import * as echarts from 'echarts/core';
import { LineChart } from 'echarts/charts';
import {
  DataZoomComponent,
  GridComponent,
  MarkLineComponent,
  TooltipComponent,
} from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';

import { SeriesPoint } from '../../../core/models/strain.models';

echarts.use([
  LineChart,
  GridComponent,
  TooltipComponent,
  DataZoomComponent,
  MarkLineComponent,
  CanvasRenderer,
]);

const connectedGroups = new Set<string>();

@Component({
  selector: 'app-chart-panel',
  standalone: true,
  templateUrl: './chart-panel.component.html',
  styleUrl: './chart-panel.component.scss',
})
export class ChartPanelComponent implements AfterViewInit, OnChanges, OnDestroy {
  @Input({ required: true }) step = 1;
  @Input({ required: true }) title = '';
  @Input({ required: true }) description = '';
  @Input({ required: true }) accentColor = '#e8e8e8';
  @Input({ required: true }) group = '';
  @Input() points: SeriesPoint[] = [];
  @Input() loading = false;
  @Input() loadingMessage = 'Loading chart…';
  /** Vertical mark at t=0 (catalog GPS). Off for synthetic runs. */
  @Input() markEventTime = false;
  /** Optional residual/energy peak time in seconds from event. */
  @Input() peakTimeSeconds: number | null = null;
  @Input() peakLabel = 'residual peak';

  @ViewChild('chartHost', { static: true }) chartHost!: ElementRef<HTMLDivElement>;

  private chart?: echarts.ECharts;
  private resizeObserver?: ResizeObserver;

  ngAfterViewInit(): void {
    this.resizeObserver = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      if (width <= 0 || height <= 0) return;

      if (!this.chart) {
        this.chart = echarts.init(this.chartHost.nativeElement, undefined, {
          renderer: 'canvas',
          useDirtyRect: true,
          width,
          height,
        });
        this.chart.group = this.group;
        if (this.group && !connectedGroups.has(this.group)) {
          echarts.connect(this.group);
          connectedGroups.add(this.group);
        }
        this.applyOption();
      } else {
        this.chart.resize({ width, height });
      }
    });
    this.resizeObserver.observe(this.chartHost.nativeElement);
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (
      (changes['points'] ||
        changes['markEventTime'] ||
        changes['peakTimeSeconds'] ||
        changes['peakLabel'] ||
        changes['accentColor'] ||
        changes['title']) &&
      this.chart
    ) {
      this.applyOption();
      requestAnimationFrame(() => this.chart?.resize());
    }
  }

  ngOnDestroy(): void {
    this.resizeObserver?.disconnect();
    this.chart?.dispose();
  }

  private buildMarkLine() {
    const data: Array<{
      xAxis: number;
      name: string;
      lineStyle: { color: string; width: number; type: 'solid' | 'dashed' };
      label: { formatter: string; color: string; fontSize: number };
    }> = [];

    if (this.markEventTime) {
      data.push({
        xAxis: 0,
        name: 'event',
        lineStyle: { color: 'rgba(242,242,242,0.45)', width: 1, type: 'dashed' },
        label: { formatter: 'event', color: 'rgba(242,242,242,0.55)', fontSize: 10 },
      });
    }

    if (this.peakTimeSeconds !== null && Number.isFinite(this.peakTimeSeconds)) {
      data.push({
        xAxis: this.peakTimeSeconds,
        name: this.peakLabel,
        lineStyle: { color: this.accentColor, width: 1.25, type: 'solid' },
        label: {
          formatter: this.peakLabel,
          color: this.accentColor,
          fontSize: 10,
        },
      });
    }

    if (!data.length) {
      return undefined;
    }

    return {
      symbol: 'none',
      animation: false,
      data,
    };
  }

  private applyOption(): void {
    if (!this.chart) return;

    const data = this.points.map((p) => [p.time, p.value]);
    const seriesName = this.title || 'strain';

    this.chart.setOption(
      {
        backgroundColor: 'transparent',
        animation: false,
        grid: { left: 48, right: 12, top: 12, bottom: 28, containLabel: true },
        textStyle: { fontFamily: 'ui-monospace, SF Mono, Menlo, monospace', color: '#f2f2f2' },
        tooltip: {
          trigger: 'axis',
          confine: true,
          appendToBody: false,
          axisPointer: { type: 'cross', lineStyle: { color: 'rgba(242,242,242,0.35)' } },
          backgroundColor: '#0a0a0a',
          borderColor: 'rgba(255,255,255,0.2)',
          padding: [8, 12],
          textStyle: {
            color: '#f2f2f2',
            fontFamily: 'ui-monospace, SF Mono, Menlo, monospace',
            fontSize: 12,
          },
          formatter: (params: unknown) => {
            const items = Array.isArray(params) ? params : [params];
            const first = items[0] as {
              axisValue?: number | string;
              data?: [number, number];
              seriesName?: string;
            };
            const time = Number(first?.axisValue ?? first?.data?.[0] ?? 0);
            const value = Number(first?.data?.[1] ?? 0);
            const name = first?.seriesName || seriesName;
            return `${name}<br/>${time.toFixed(4)} s from event<br/>value ${value.toFixed(4)}`;
          },
        },
        xAxis: {
          type: 'value',
          name: 'seconds from event',
          nameLocation: 'middle',
          nameGap: 18,
          nameTextStyle: { color: 'rgba(242,242,242,0.45)', fontSize: 10 },
          axisLine: { lineStyle: { color: 'rgba(255,255,255,0.16)' } },
          axisLabel: { color: 'rgba(242,242,242,0.56)', fontSize: 10 },
          splitLine: { show: false },
        },
        yAxis: {
          type: 'value',
          scale: true,
          axisLine: { show: false },
          axisLabel: {
            color: 'rgba(242,242,242,0.56)',
            fontSize: 10,
            formatter: (value: number) => value.toFixed(2),
          },
          splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } },
        },
        dataZoom: [{ type: 'inside', xAxisIndex: 0 }],
        series: [
          {
            name: seriesName,
            type: 'line',
            data,
            showSymbol: false,
            sampling: 'lttb',
            lineStyle: { color: this.accentColor, width: 1.25 },
            areaStyle: { color: this.accentColor, opacity: 0.06 },
            emphasis: { disabled: true },
            markLine: this.buildMarkLine(),
          },
        ],
      },
      { notMerge: true },
    );
  }
}
