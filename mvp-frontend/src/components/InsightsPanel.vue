<template>
  <div class="card insights">
    <div class="card-header">
      <h2>运行洞察</h2>
      <span class="tag">算法建议</span>
    </div>
    <ul>
      <li v-for="item in insights" :key="item.title">
        <strong>{{ item.title }}</strong>
        <p>{{ item.detail }}</p>
      </li>
    </ul>
  </div>
</template>

<script setup>
import { computed } from 'vue';
import { useMetricsStore } from '../store/metrics';

const store = useMetricsStore();

const insights = computed(() => [
  {
    title: `高峰拥挤站点：${store.topStations[0].name}`,
    detail: '建议在 18:00-20:00 增加 2 条接驳班线。'
  },
  {
    title: '新区客流增幅 12%',
    detail: '地铁 14 号线开通后新增 3 个热点。'
  },
  {
    title: '异常密度预警 4 次',
    detail: '结合摄像头与气象数据分析后已自动过滤。'
  }
]);
</script>
