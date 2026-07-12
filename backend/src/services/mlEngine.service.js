const axios = require('axios');
const config = require('../config');
const logger = require('../utils/logger');
const ApiError = require('../utils/ApiError');

const client = axios.create({
  baseURL: config.mlEngine.baseUrl,
});

// GWOSC frame downloads (via gwpy) can take several minutes on a cold
// fetch, so the inference call gets a much longer allowance than routine
// health checks.
const DENOISE_TIMEOUT_MS = 300000;
const HEALTH_TIMEOUT_MS = 5000;

async function requestDenoising({
  gpsTime,
  detector = 'H1',
  duration = 4,
  synthetic = false,
  syntheticStrategy = 'oracle',
}) {
  try {
    const { data } = await client.post(
      '/api/v1/denoise',
      {
        gps_time: gpsTime,
        detector,
        duration,
        synthetic,
        synthetic_strategy: syntheticStrategy,
      },
      { timeout: synthetic ? HEALTH_TIMEOUT_MS : DENOISE_TIMEOUT_MS },
    );
    return data;
  } catch (error) {
    logger.error('ML engine denoising request failed', error.message);
    throw new ApiError(502, 'Failed to reach the ML inference engine', error.message);
  }
}

async function checkHealth() {
  try {
    const { data } = await client.get('/health', { timeout: HEALTH_TIMEOUT_MS });
    return data;
  } catch (error) {
    logger.warn('ML engine health check failed', error.message);
    return { status: 'unreachable' };
  }
}

module.exports = { requestDenoising, checkHealth };
