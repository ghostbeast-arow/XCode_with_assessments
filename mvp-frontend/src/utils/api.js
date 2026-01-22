import axios from 'axios';

export const apiClient = axios.create({
  baseURL: 'https://api.example.com',
  timeout: 8000
});

export const fetchSummary = async () => {
  const response = await apiClient.get('/summary');
  return response.data;
};
