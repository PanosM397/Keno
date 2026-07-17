const axios = require('axios');
const config = require('../config');
const logger = require('../utils/logger');
const ApiError = require('../utils/ApiError');

const client = axios.create({
  baseURL: config.gwosc.baseUrl,
  timeout: 15000,
});

async function fetchEventCatalog(catalog = 'GWTC') {
  try {
    // Trailing slash avoids a GWOSC 301 that some clients mishandle.
    const { data } = await client.get(`/eventapi/json/${catalog}/`);
    return data;
  } catch (error) {
    logger.error('GWOSC catalog fetch failed', error.message);
    throw new ApiError(502, 'Failed to reach GWOSC event catalog', error.message);
  }
}

async function fetchEventMetadata(eventName) {
  try {
    const { data } = await client.get(`/eventapi/json/event/${eventName}/`);
    return data;
  } catch (error) {
    logger.error('GWOSC event metadata fetch failed', error.message);
    throw new ApiError(502, `Failed to reach GWOSC for event ${eventName}`, error.message);
  }
}

module.exports = { fetchEventCatalog, fetchEventMetadata };
