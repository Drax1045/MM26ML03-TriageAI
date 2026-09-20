/* ==========================================================================
   TriageAI — live API layer
   --------------------------------------------------------------------------
   The ONLY place this app talks to the network. Every prediction shown in the
   UI comes from the FastAPI service and the trained XGBoost pipeline behind
   it. There is deliberately no client-side predictor and no cached fallback:
   if the backend is down, the UI reports it rather than inventing a result.

   Public surface consumed by index.html:
     window.TriageAPI.baseUrl                 -> string
     window.TriageAPI.predict(features)       -> { prediction, probabilities }
     window.TriageAPI.connect()               -> { live, detail, schema }
     window.TRIAGE_SCHEMA                     -> last loaded /schema payload

   Events dispatched on document:
     "triage-status"         detail: { live, detail, schema }
     "triage-schema-loaded"  detail: <schema payload>
   ========================================================================== */
(() => {
  "use strict";

  const API_URL = (window.TRIAGE_API_URL || "http://localhost:8000").replace(/\/$/, "");
  const CLASSES = ["RED", "YELLOW", "GREEN", "BLACK"];
  const TIMEOUT_MS = 15000;

  window.TRIAGE_API_URL = API_URL;
  window.TRIAGE_SCHEMA = null;

  /* ---------- fetch with a timeout so a dead port fails fast ------------- */
  async function request(path, options = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
    try {
      return await fetch(API_URL + path, { ...options, signal: controller.signal });
    } finally {
      clearTimeout(timer);
    }
  }

  function emit(name, detail) {
    document.dispatchEvent(new CustomEvent(name, { detail }));
  }

  /* ---------- GET /health ------------------------------------------------ */
  async function health() {
    const response = await request("/health");
    if (!response.ok) throw new Error(`Health check failed: HTTP ${response.status}`);
    return await response.json();
  }

  /* ---------- GET /schema ------------------------------------------------
     The fitted pipeline is the source of truth for which inputs exist and
     which categorical values are valid. The UI never invents either.        */
  async function loadSchema() {
    const response = await request("/schema");
    if (!response.ok) throw new Error(`Schema request failed: HTTP ${response.status}`);

    const schema = await response.json();
    if (!schema || !Array.isArray(schema.features))
      throw new Error("Schema response did not contain a feature list.");

    schema.numeric = schema.numeric || [];
    schema.categorical = schema.categorical || [];
    schema.categories = schema.categories || {};

    window.TRIAGE_SCHEMA = schema;
    emit("triage-schema-loaded", schema);
    return schema;
  }

  /* ---------- POST /predict ---------------------------------------------- */
  async function predict(features) {
    const response = await request("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(features || {})
    });

    if (!response.ok) {
      let message = `Prediction failed: HTTP ${response.status}`;
      try {
        const body = await response.json();
        if (body && body.detail) message = String(body.detail);
      } catch (_) { /* non-JSON error body */ }
      throw new Error(message);
    }

    const data = await response.json();

    if (!data || typeof data.prediction !== "string" || !data.probabilities)
      throw new Error("Malformed response: expected { prediction, probabilities }.");

    if (!CLASSES.includes(data.prediction))
      throw new Error(`Unexpected predicted class: ${data.prediction}`);

    const missing = CLASSES.filter(c => !Number.isFinite(Number(data.probabilities[c])));
    if (missing.length)
      throw new Error(`Missing probabilities for: ${missing.join(", ")}`);

    const sum = CLASSES.reduce((total, c) => total + Number(data.probabilities[c]), 0);
    if (!(sum > 0.99 && sum < 1.01))
      throw new Error(`Probabilities sum to ${sum.toFixed(6)}, not ~1.`);

    const probabilities = {};
    CLASSES.forEach(c => { probabilities[c] = Number(data.probabilities[c]); });

    return { prediction: data.prediction, probabilities };
  }

  /* ---------- connect: health + schema, then broadcast status ------------ */
  async function connect() {
    try {
      const info = await health();
      const schema = await loadSchema();

      const model = (info && info.model) || "XGBoost pipeline";
      const count = schema.features.length;
      const status = { live: true, detail: `${model} · ${count} inputs`, schema };

      console.info("TriageAI: LIVE MODEL connected at", API_URL);
      console.info("TriageAI: model inputs:", schema.features);

      emit("triage-status", status);
      return status;
    } catch (error) {
      const status = { live: false, detail: error.message, schema: window.TRIAGE_SCHEMA };
      console.error("TriageAI: backend unreachable at", API_URL, "—", error.message);
      emit("triage-status", status);
      return status;
    }
  }

  window.TriageAPI = {
    get baseUrl() { return API_URL; },
    health,
    loadSchema,
    predict,
    connect
  };
})();