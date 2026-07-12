const express = require('express');
const {
  getDenoisedStrain,
  getEventCatalog,
  getEventMetadata,
} = require('../controllers/strain.controller');

const router = express.Router();

router.get('/denoised', getDenoisedStrain);
router.get('/events', getEventCatalog);
router.get('/events/:eventName', getEventMetadata);

module.exports = router;
