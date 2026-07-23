const gwoscService = require('../services/gwosc.service');
const mlEngineService = require('../services/mlEngine.service');
const cacheService = require('../services/cache.service');
const ApiError = require('../utils/ApiError');
const logger = require('../utils/logger');

async function getDenoisedStrain(req, res, next) {
  try {
    const { gpsTime, detector = 'H1', duration = 4, synthetic, syntheticStrategy = 'oracle' } = req.query;

    if (!gpsTime || Number.isNaN(Number(gpsTime))) {
      throw new ApiError(400, 'A valid numeric gpsTime query parameter is required');
    }

    const params = {
      gpsTime: Number(gpsTime),
      detector,
      duration: Number(duration),
      synthetic: synthetic === 'true',
      syntheticStrategy,
    };
    const cached = cacheService.get('denoised-strain', params);
    if (cached) {
      logger.info(`Cache hit for gpsTime=${params.gpsTime} detector=${detector}`);
      return res.json({ ...cached, cached: true });
    }

    const result = await mlEngineService.requestDenoising(params);
    cacheService.set('denoised-strain', params, result);

    return res.json({ ...result, cached: false });
  } catch (error) {
    return next(error);
  }
}

async function getEventCatalog(req, res, next) {
  try {
    const catalog = req.query.catalog || 'GWTC';
    const cached = cacheService.get('event-catalog', { catalog });
    if (cached) {
      return res.json({ ...cached, cached: true });
    }

    const data = await gwoscService.fetchEventCatalog(catalog);
    cacheService.set('event-catalog', { catalog }, data);

    return res.json({ ...data, cached: false });
  } catch (error) {
    return next(error);
  }
}

async function getEventMetadata(req, res, next) {
  try {
    const { eventName } = req.params;
    const cached = cacheService.get('event-metadata', { eventName });
    if (cached) {
      return res.json({ ...cached, cached: true });
    }

    const data = await gwoscService.fetchEventMetadata(eventName);
    cacheService.set('event-metadata', { eventName }, data);

    return res.json({ ...data, cached: false });
  } catch (error) {
    return next(error);
  }
}

async function getStrainDetection(req, res, next) {
  try {
    const { gpsTime, detector = 'H1', duration = 4 } = req.query;

    if (!gpsTime || Number.isNaN(Number(gpsTime))) {
      throw new ApiError(400, 'A valid numeric gpsTime query parameter is required');
    }

    const params = {
      gpsTime: Number(gpsTime),
      detector,
      duration: Number(duration),
    };
    const cached = cacheService.get('strain-detection', params);
    if (cached) {
      logger.info(`Detection cache hit for gpsTime=${params.gpsTime} detector=${detector}`);
      return res.json({ ...cached, cached: true });
    }

    const result = await mlEngineService.requestDetection(params);
    cacheService.set('strain-detection', params, result);

    return res.json({ ...result, cached: false });
  } catch (error) {
    return next(error);
  }
}

async function getStrainCoincidence(req, res, next) {
  try {
    const { gpsTime, duration = 4, detectors = 'H1,L1' } = req.query;

    if (!gpsTime || Number.isNaN(Number(gpsTime))) {
      throw new ApiError(400, 'A valid numeric gpsTime query parameter is required');
    }

    const detectorList = String(detectors)
      .split(',')
      .map((value) => value.trim())
      .filter(Boolean);

    const params = {
      gpsTime: Number(gpsTime),
      duration: Number(duration),
      detectors: detectorList,
    };
    const cached = cacheService.get('strain-coincidence', params);
    if (cached) {
      logger.info(`Coincidence cache hit for gpsTime=${params.gpsTime}`);
      return res.json({ ...cached, cached: true });
    }

    const result = await mlEngineService.requestCoincidenceDetection(params);
    cacheService.set('strain-coincidence', params, result);

    return res.json({ ...result, cached: false });
  } catch (error) {
    return next(error);
  }
}

async function clearStrainCache(req, res, next) {
  try {
    cacheService.clear();
    logger.info('Strain response cache cleared');
    return res.json({ cleared: true });
  } catch (error) {
    return next(error);
  }
}

module.exports = {
  getDenoisedStrain,
  getStrainDetection,
  getStrainCoincidence,
  getEventCatalog,
  getEventMetadata,
  clearStrainCache,
};
