const express = require('express');
const {
  getDenoisedStrain,
  getStrainDetection,
  getStrainCoincidence,
  getEventCatalog,
  getEventMetadata,
} = require('../controllers/strain.controller');

const router = express.Router();

router.get('/denoised', getDenoisedStrain);
router.get('/detect', getStrainDetection);
router.get('/detect/coincidence', getStrainCoincidence);
router.get('/events', getEventCatalog);
router.get('/events/:eventName', getEventMetadata);

module.exports = router;
