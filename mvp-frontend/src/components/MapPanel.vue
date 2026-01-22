<template>
  <div class="card">
    <div class="card-header">
      <h2>城市热点地图</h2>
      <div class="legend">
        <span class="legend-item high">高密度</span>
        <span class="legend-item mid">中密度</span>
        <span class="legend-item low">低密度</span>
      </div>
    </div>
    <div ref="mapContainer" class="map-container"></div>
  </div>
</template>

<script setup>
import { onMounted, onBeforeUnmount, ref } from 'vue';
import maplibregl from 'maplibre-gl';
import { hotspotGeoJson } from '../data/hotspots';

const mapContainer = ref(null);
let mapInstance;

onMounted(() => {
  mapInstance = new maplibregl.Map({
    container: mapContainer.value,
    style: 'https://demotiles.maplibre.org/style.json',
    center: [121.4737, 31.2304],
    zoom: 9.8
  });

  mapInstance.on('load', () => {
    mapInstance.addSource('hotspots', {
      type: 'geojson',
      data: hotspotGeoJson
    });

    mapInstance.addLayer({
      id: 'hotspots-layer',
      type: 'circle',
      source: 'hotspots',
      paint: {
        'circle-color': [
          'interpolate',
          ['linear'],
          ['get', 'density'],
          20,
          '#2dd4bf',
          60,
          '#38bdf8',
          100,
          '#f97316'
        ],
        'circle-radius': [
          'interpolate',
          ['linear'],
          ['get', 'density'],
          20,
          6,
          100,
          18
        ],
        'circle-opacity': 0.75
      }
    });
  });
});

onBeforeUnmount(() => {
  if (mapInstance) {
    mapInstance.remove();
  }
});
</script>
