const { test, beforeEach } = require('node:test');
const assert = require('node:assert/strict');

const cache = require('./cache.service');

beforeEach(() => {
  cache.clear();
});

test('cache stores and returns values', () => {
  cache.set('denoise', { gps: 1 }, { ok: true }, 60);
  assert.deepEqual(cache.get('denoise', { gps: 1 }), { ok: true });
});

test('cache miss returns null', () => {
  assert.equal(cache.get('denoise', { gps: 2 }), null);
});

test('cache expires entries', async () => {
  cache.set('denoise', { gps: 3 }, { ok: true }, 0);
  await new Promise((resolve) => setTimeout(resolve, 5));
  assert.equal(cache.get('denoise', { gps: 3 }), null);
});
