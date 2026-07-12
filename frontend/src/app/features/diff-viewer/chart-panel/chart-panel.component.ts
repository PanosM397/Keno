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
import { GridComponent, TooltipComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';

import { SeriesPoint } from '../../../core/models/strain.models';

echarts.use([LineChart, GridComponent, TooltipComponent, CanvasRenderer]);

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
        this.applyOption();
      } else {
        this.chart.resize({ width, height });
      }
    });
    this.resizeObserver.observe(this.chartHost.nativeElement);
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['points'] && this.chart) {
      this.applyOption();
      requestAnimationFrame(() => this.chart?.resize());
    }
  }

  ngOnDestroy(): void {
    this.resizeObserver?.disconnect();
    this.chart?.dispose();
  }

  private applyOption(): void {
    if (!this.chart) return;

    const data = this.points.map((p) => [p.time, p.value]);

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
          textStyle: { color: '#f2f2f2', fontFamily: 'ui-monospace, SF Mono, Menlo, monospace', fontSize: 12 },
          valueFormatter: (value: number | string) => Number(value).toFixed(4),
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
            type: 'line',
            data,
            showSymbol: false,
            sampling: 'lttb',
            lineStyle: { color: this.accentColor, width: 1.25 },
            areaStyle: { color: this.accentColor, opacity: 0.06 },
            emphasis: { disabled: true },
          },
        ],
      },
      { notMerge: false },
    );
  }
}
