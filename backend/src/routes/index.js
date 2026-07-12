const express = require('express');
const healthRoutes = require('./health.routes');
const strainRoutes = require('./strain.routes');

const router = express.Router();

router.use('/health', healthRoutes);
router.use('/strain', strainRoutes);

module.exports = router;
