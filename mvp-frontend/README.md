# Geo Insight MVP

一个基于 **Vue 3 + Composition API** 的 GIS 数据可视化前端原型，集成了 **ECharts** 与 **MapLibre GL JS**，并提供了工程化与模块化的基础结构。

## 技术栈

- 核心框架：Vue 3 + Composition API
- 可视化：ECharts
- 地图引擎：MapLibre GL JS（可替换 Mapbox GL JS）
- 工程化：Vite + ESLint + Prettier
- 状态管理：Pinia
- 网络请求：Axios（预留接口）
- 样式：SCSS
- 测试：Vitest

## 本地启动

```bash
npm install
npm run dev
```

## 项目结构

```
src/
  components/  # 视图组件
  data/        # 业务模拟数据 & GeoJSON
  store/       # Pinia 状态
  styles/      # 样式入口
  utils/       # 通用工具
```

## MVP 展示点

- 地图热点图层（GeoJSON + MapLibre GL）
- 人流趋势/站点排名（ECharts）
- 数据刷新交互与洞察卡片

