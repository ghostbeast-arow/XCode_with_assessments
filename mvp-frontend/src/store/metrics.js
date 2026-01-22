import { defineStore } from 'pinia';
import { hourlyTrend, topStations } from '../data/metrics';
import { shuffleArray } from '../utils/shuffle';

export const useMetricsStore = defineStore('metrics', {
  state: () => ({
    hourlyTrend: [...hourlyTrend],
    topStations: [...topStations]
  }),
  actions: {
    shuffle() {
      this.hourlyTrend = shuffleArray(this.hourlyTrend).slice(0, 8);
      this.topStations = shuffleArray(this.topStations).slice(0, 6);
    }
  }
});
