const logger = require('../utils/logger');
const ApiError = require('../utils/ApiError');

function notFoundHandler(req, res) {
  res.status(404).json({ error: `Route not found: ${req.method} ${req.originalUrl}` });
}

function errorHandler(err, req, res, next) {
  const statusCode = err instanceof ApiError ? err.statusCode : 500;
  const message = err.message || 'Internal server error';

  if (statusCode >= 500) {
    logger.error(message, err.stack);
  } else {
    logger.warn(message);
  }

  res.status(statusCode).json({
    error: message,
    details: err instanceof ApiError ? err.details : undefined,
  });
}

module.exports = { notFoundHandler, errorHandler };
