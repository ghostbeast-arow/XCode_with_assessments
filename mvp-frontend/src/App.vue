<template>
  <div class="app-shell">
    <header class="app-header">
      <div>
        <p class="eyebrow">GIS 数据洞察 MVP</p>
        <h1>城市运行与人群分布可视化</h1>
      </div>
      <div class="status">
        <span class="dot"></span>
        <span>实时数据 · 延迟 2 分钟</span>
      </div>
    </header>

    <main class="app-main">
      <section class="panel left">
        <MapPanel />
      </section>
      <section class="panel right">
        <ChartPanel />
        <InsightsPanel />
      </section>
    </main>
  </div>
</template>

<script setup>
import MapPanel from './components/MapPanel.vue';
import ChartPanel from './components/ChartPanel.vue';
import InsightsPanel from './components/InsightsPanel.vue';
</script>
