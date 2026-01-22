<template>
  <div class="card chart-card">
    <div class="card-header">
      <h2>人流趋势与站点排名</h2>
      <button class="btn" @click="refresh">刷新数据</button>
    </div>
    <div class="chart-grid">
      <div ref="lineChart" class="chart"></div>
      <div ref="barChart" class="chart"></div>
    </div>
  </div>
</template>

<script setup>
import { onMounted, onBeforeUnmount, ref } from 'vue';
import * as echarts from 'echarts';
import { useMetricsStore } from '../store/metrics';

const lineChart = ref(null);
const barChart = ref(null);
const store = useMetricsStore();
let lineInstance;
let barInstance;

const resizeCharts = () => {
  lineInstance?.resize();
  barInstance?.resize();
};

const initCharts = () => {
  lineInstance = echarts.init(lineChart.value);
  barInstance = echarts.init(barChart.value);
  updateCharts();
  window.addEventListener('resize', resizeCharts);
};

const updateCharts = () => {
  const trend = store.hourlyTrend;
  const ranking = store.topStations;

  lineInstance.setOption({
    grid: { left: 32, right: 16, top: 32, bottom: 32 },
    xAxis: {
      type: 'category',
      data: trend.map((item) => item.hour),
      axisLabel: { color: '#94a3b8' }
    },
    yAxis: {
      type: 'value',
      axisLabel: { color: '#94a3b8' },
      splitLine: { lineStyle: { color: '#1e293b' } }
    },
    series: [
      {
        type: 'line',
        data: trend.map((item) => item.volume),
        smooth: true,
        areaStyle: { color: 'rgba(56,189,248,0.3)' },
        lineStyle: { color: '#38bdf8' },
        symbol: 'circle',
        symbolSize: 8
      }
    ]
  });

  barInstance.setOption({
    grid: { left: 32, right: 16, top: 32, bottom: 32 },
    xAxis: {
      type: 'value',
      axisLabel: { color: '#94a3b8' },
      splitLine: { lineStyle: { color: '#1e293b' } }
    },
    yAxis: {
      type: 'category',
      data: ranking.map((item) => item.name),
      axisLabel: { color: '#94a3b8' }
    },
    series: [
      {
        type: 'bar',
        data: ranking.map((item) => item.value),
        itemStyle: {
          color: '#2dd4bf'
        },
        barWidth: 18
      }
    ]
  });
};

const refresh = () => {
  store.shuffle();
  updateCharts();
};

onMounted(() => {
  initCharts();
});

onBeforeUnmount(() => {
  window.removeEventListener('resize', resizeCharts);
  lineInstance?.dispose();
  barInstance?.dispose();
});
</script>
