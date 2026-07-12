const createApp = require('./app');
const config = require('./config');
const logger = require('./utils/logger');

const app = createApp();

app.listen(config.port, () => {
  logger.info(`gwburst-backend listening on port ${config.port} [${config.env}]`);
});
