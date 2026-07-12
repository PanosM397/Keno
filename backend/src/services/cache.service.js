const config = require('../config');

const store = new Map();

function buildKey(namespace, params) {
  return `${namespace}:${JSON.stringify(params)}`;
}

function get(namespace, params) {
  const key = buildKey(namespace, params);
  const entry = store.get(key);
  if (!entry) return null;
  if (Date.now() > entry.expiresAt) {
    store.delete(key);
    return null;
  }
  return entry.value;
}

function set(namespace, params, value, ttlSeconds = config.cache.ttlSeconds) {
  const key = buildKey(namespace, params);
  store.set(key, {
    value,
    expiresAt: Date.now() + ttlSeconds * 1000,
  });
}

function clear() {
  store.clear();
}

module.exports = { get, set, clear };
