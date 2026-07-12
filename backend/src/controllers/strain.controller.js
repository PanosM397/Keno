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
    const { catalog } = req.query;
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

module.exports = { getDenoisedStrain, getEventCatalog, getEventMetadata };
