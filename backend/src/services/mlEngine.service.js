const axios = require('axios');
const config = require('../config');
const logger = require('../utils/logger');
const ApiError = require('../utils/ApiError');

const client = axios.create({
  baseURL: config.mlEngine.baseUrl,
});

// GWOSC frame downloads (via gwpy) can take several minutes on a cold
// fetch, so the inference call gets a much longer allowance than routine
// health checks. Coincidence fetches two detectors, so it needs ~2× budget.
const DENOISE_TIMEOUT_MS = 300000;
const COINCIDENCE_TIMEOUT_MS = 600000;
const HEALTH_TIMEOUT_MS = 5000;

function mlEngineErrorDetail(error) {
  const detail = error?.response?.data?.detail;
  if (typeof detail === 'string' && detail.trim()) {
    return detail;
  }
  if (Array.isArray(detail)) {
    return detail
      .map((item) => (typeof item === 'string' ? item : item?.msg || JSON.stringify(item)))
      .filter(Boolean)
      .join('; ');
  }
  if (detail && typeof detail === 'object') {
    return detail.message || JSON.stringify(detail);
  }
  if (typeof error?.response?.data?.error === 'string') {
    return error.response.data.error;
  }
  return error?.message || 'Unknown ML engine error';
}

function throwMlEngineError(context, error) {
  const detail = mlEngineErrorDetail(error);
  const status = error?.response?.status;
  logger.error(`${context} failed`, detail);
  throw new ApiError(
    status && status >= 400 && status < 600 ? status : 502,
    'Failed to reach the ML inference engine',
    detail,
  );
}

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
    throwMlEngineError('ML engine denoising request', error);
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

async function requestDetection({ gpsTime, detector = 'H1', duration = 4 }) {
  try {
    const { data } = await client.post(
      '/api/v1/detect',
      {
        gps_time: gpsTime,
        detector,
        duration,
      },
      { timeout: DENOISE_TIMEOUT_MS },
    );
    return data;
  } catch (error) {
    throwMlEngineError('ML engine detection request', error);
  }
}

async function requestCoincidenceDetection({
  gpsTime,
  detectors = ['H1', 'L1'],
  duration = 4,
}) {
  try {
    const { data } = await client.post(
      '/api/v1/detect/coincidence',
      {
        gps_time: gpsTime,
        detectors,
        duration,
      },
      { timeout: COINCIDENCE_TIMEOUT_MS },
    );
    return data;
  } catch (error) {
    throwMlEngineError('ML engine coincidence request', error);
  }
}

module.exports = { requestDenoising, requestDetection, requestCoincidenceDetection, checkHealth };
