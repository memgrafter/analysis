(() => {
  const yearSelect = document.getElementById("year-select");
  const metricSelect = document.getElementById("metric-select");
  const limitSelect = document.getElementById("limit-select");
  const statusEl = document.getElementById("status");
  const cloudEl = document.getElementById("cloud");

  const ALL_YEARS = [2023, 2024, 2025];
  const ALL_SCOPES = ["title", "core", "body"];

  let payload = null;

  const escapeHtml = (value) =>
    String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");

  function hashString(value) {
    let h = 2166136261;
    for (let i = 0; i < value.length; i += 1) {
      h ^= value.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return h >>> 0;
  }

  function sortForCloud(rows, year) {
    return [...rows].sort((a, b) => {
      const ah = hashString(`${year}:${a.term}`);
      const bh = hashString(`${year}:${b.term}`);
      return ah - bh;
    });
  }

  function buildSearchHref(term) {
    const url = new URL("/search/", window.location.origin);
    url.searchParams.set("q", term);
    url.searchParams.set("sort", "relevance");
    for (const year of ALL_YEARS) {
      url.searchParams.append("y", String(year));
    }
    for (const scope of ALL_SCOPES) {
      url.searchParams.append("f", scope);
    }
    return `${url.pathname}${url.search}`;
  }

  function renderCloud() {
    if (!payload) return;

    const year = yearSelect.value;
    const metric = metricSelect.value;
    const limit = Number.parseInt(limitSelect.value, 10) || 200;

    const yearData = payload.years?.[year];
    const rows = Array.isArray(yearData?.terms) ? yearData.terms.slice(0, limit) : [];

    if (rows.length === 0) {
      cloudEl.innerHTML = `<p class="muted">No cloud terms found for ${escapeHtml(year)}.</p>`;
      statusEl.textContent = `No terms available for ${year}.`;
      return;
    }

    const weights = rows.map((row) => {
      const v = Number(row?.[metric] ?? 0);
      return Number.isFinite(v) ? v : 0;
    });

    const min = Math.min(...weights);
    const max = Math.max(...weights);
    const spread = max - min || 1;

    const shuffledRows = sortForCloud(rows, year);

    const html = shuffledRows
      .map((row) => {
        const term = String(row.term || "").trim();
        const score = Number(row?.[metric] ?? 0);
        const normalized = (score - min) / spread;
        const fontSizePx = 13 + normalized * 34;
        const opacity = 0.62 + normalized * 0.38;
        const href = buildSearchHref(term);

        return `<a href="${href}" title="${escapeHtml(term)} (${metric}: ${score})" style="font-size:${fontSizePx.toFixed(1)}px;opacity:${opacity.toFixed(2)};">${escapeHtml(term)}<span class="word-score">${score}</span></a>`;
      })
      .join("\n");

    cloudEl.innerHTML = html;

    statusEl.textContent = `Showing ${rows.length} term(s) for ${year}. Click a term to run full-corpus full-text search.`;
  }

  async function init() {
    try {
      const res = await fetch("/search/cloud-terms.json", { cache: "no-store" });
      if (!res.ok) {
        throw new Error(`Could not load /search/cloud-terms.json (${res.status})`);
      }
      payload = await res.json();
      renderCloud();
    } catch (err) {
      statusEl.textContent = "Failed to load cloud terms.";
      cloudEl.innerHTML = `<p style="color:#cc3b3b;font-weight:600;">${escapeHtml(String(err))}</p>`;
    }
  }

  for (const el of [yearSelect, metricSelect, limitSelect]) {
    el.addEventListener("change", renderCloud);
  }

  init();
})();
