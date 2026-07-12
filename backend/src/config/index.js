require('dotenv').config();

const config = {
  env: process.env.NODE_ENV || 'development',
  port: Number(process.env.PORT) || 4000,
  gwosc: {
    baseUrl: process.env.GWOSC_BASE_URL || 'https://gwosc.org',
  },
  mlEngine: {
    baseUrl: process.env.ML_ENGINE_BASE_URL || 'http://localhost:8000',
  },
  cache: {
    ttlSeconds: Number(process.env.CACHE_TTL_SECONDS) || 3600,
  },
  cors: {
    origin: process.env.CORS_ORIGIN || 'http://localhost:4200',
  },
};

module.exports = config;
