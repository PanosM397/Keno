const mlEngineService = require('../services/mlEngine.service');

async function getHealth(req, res) {
  const mlEngineStatus = await mlEngineService.checkHealth();
  res.json({
    status: 'ok',
    service: 'keno-backend',
    timestamp: new Date().toISOString(),
    mlEngine: mlEngineStatus,
  });
}

module.exports = { getHealth };
