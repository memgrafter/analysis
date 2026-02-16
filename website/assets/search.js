(() => {
  const formEl = document.getElementById("search-form");
  const inputEl = document.getElementById("q");
  const statusEl = document.getElementById("status");
  const metaEl = document.getElementById("meta");
  const resultsEl = document.getElementById("results");

  let db = null;
  let manifest = null;
  let supportsFts = false;

  const escapeHtml = (value) =>
    String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");

  const qs = new URLSearchParams(window.location.search);
  const initialQ = (qs.get("q") || "").trim();
  inputEl.value = initialQ;

  function updateUrlQuery(q) {
    const url = new URL(window.location.href);
    if (q) {
      url.searchParams.set("q", q);
    } else {
      url.searchParams.delete("q");
    }
    window.history.replaceState({}, "", url.toString());
  }

  function renderEmpty(message) {
    resultsEl.innerHTML = `<p class=\"muted\">${escapeHtml(message)}</p>`;
  }

  function renderError(message) {
    resultsEl.innerHTML = `<p style=\"color:#cc3b3b;font-weight:600;\">${escapeHtml(message)}</p>`;
  }

  function renderResults(rows, q) {
    if (rows.length === 0) {
      renderEmpty(`No results for \"${q}\"`);
      return;
    }

    resultsEl.innerHTML = rows
      .map((row) => {
        const digestId = row.digest_id || "";
        const title = row.title || "(untitled)";
        const arxiv = row.arxiv_id || "n/a";
        return `
          <article class=\"result\">
            <div class=\"title\"><a href=\"/view/?id=${encodeURIComponent(digestId)}\">${escapeHtml(title)}</a></div>
            <div><code>${escapeHtml(digestId)}</code></div>
            <div class=\"muted\">arXiv: ${escapeHtml(arxiv)}</div>
            <div style=\"margin-top:6px;\">
              <a href=\"/view/?id=${encodeURIComponent(digestId)}\">view</a>
              &nbsp;·&nbsp;
              <a href=\"/view/${encodeURIComponent(digestId)}.md\">raw</a>
            </div>
          </article>
        `;
      })
      .join("\n");
  }

  function normalizeFtsQuery(input) {
    const tokens = input
      .toLowerCase()
      .split(/\s+/)
      .map((token) => token.replace(/[^a-z0-9.]/g, ""))
      .filter(Boolean);

    return tokens.map((token) => `${token}*`).join(" AND ");
  }

  function looksLikeArxivId(input) {
    return /^\d{4}\.\d{4,5}(v\d+)?$/i.test(input.trim());
  }

  function queryArxiv(input) {
    const stmt = db.prepare(`
      SELECT digest_id, title, arxiv_id
      FROM digests
      WHERE arxiv_id = ?
      ORDER BY timestamp_suffix DESC, digest_id DESC
      LIMIT 50
    `);
    stmt.bind([input.trim()]);

    const rows = [];
    while (stmt.step()) rows.push(stmt.getAsObject());
    stmt.free();
    return rows;
  }

  function queryFts(input) {
    const fts = normalizeFtsQuery(input);
    if (!fts) return [];

    const stmt = db.prepare(`
      SELECT d.digest_id AS digest_id, d.title AS title, d.arxiv_id AS arxiv_id, bm25(digests_fts) AS score
      FROM digests_fts
      JOIN digests d ON d.id = digests_fts.rowid
      WHERE digests_fts MATCH ?
      ORDER BY score
      LIMIT 50
    `);
    stmt.bind([fts]);

    const rows = [];
    while (stmt.step()) rows.push(stmt.getAsObject());
    stmt.free();
    return rows;
  }

  function queryLike(input) {
    const q = input.trim().toLowerCase();
    if (!q) return [];
    const pattern = `%${q}%`;

    const stmt = db.prepare(`
      SELECT digest_id, title, arxiv_id
      FROM digests
      WHERE lower(COALESCE(digest_id, '')) LIKE ?
         OR lower(COALESCE(arxiv_id, '')) LIKE ?
         OR lower(COALESCE(title, '')) LIKE ?
         OR lower(COALESCE(core_contribution, '')) LIKE ?
         OR lower(COALESCE(tags, '')) LIKE ?
         OR lower(COALESCE(body_preview, '')) LIKE ?
      ORDER BY
        CASE
          WHEN lower(COALESCE(arxiv_id, '')) = ? THEN 0
          WHEN lower(COALESCE(digest_id, '')) = ? THEN 1
          ELSE 2
        END,
        timestamp_suffix DESC,
        digest_id DESC
      LIMIT 50
    `);
    stmt.bind([pattern, pattern, pattern, pattern, pattern, pattern, q, q]);

    const rows = [];
    while (stmt.step()) rows.push(stmt.getAsObject());
    stmt.free();
    return rows;
  }

  function detectFtsSupport() {
    try {
      db.exec("SELECT rowid FROM digests_fts LIMIT 1");
      return true;
    } catch {
      return false;
    }
  }

  function runSearch() {
    const q = inputEl.value.trim();
    updateUrlQuery(q);

    if (!q) {
      renderEmpty("Enter a query to search digests.");
      return;
    }

    if (!db) {
      renderEmpty("Search DB is still loading…");
      return;
    }

    try {
      statusEl.textContent = `Searching for \"${q}\"…`;
      const started = performance.now();

      let rows;
      let mode = "like";
      if (looksLikeArxivId(q)) {
        mode = "arxiv";
        rows = queryArxiv(q);
      } else if (supportsFts) {
        mode = "fts";
        try {
          rows = queryFts(q);
        } catch (err) {
          const msg = String(err).toLowerCase();
          if (msg.includes("fts5") || msg.includes("digests_fts")) {
            supportsFts = false;
            mode = "like";
            rows = queryLike(q);
          } else {
            throw err;
          }
        }
      } else {
        rows = queryLike(q);
      }

      const elapsedMs = (performance.now() - started).toFixed(1);
      statusEl.textContent = `Found ${rows.length} result(s) for \"${q}\" in ${elapsedMs} ms (${mode}).`;
      renderResults(rows, q);
    } catch (err) {
      statusEl.textContent = "Search failed.";
      renderError(String(err));
    }
  }

  formEl.addEventListener("submit", (event) => {
    event.preventDefault();
    runSearch();
  });

  async function init() {
    try {
      statusEl.textContent = "Loading manifest…";
      const manifestRes = await fetch("/search/manifest.json", { cache: "no-store" });
      if (!manifestRes.ok) {
        throw new Error(`Could not load /search/manifest.json (${manifestRes.status})`);
      }
      manifest = await manifestRes.json();

      const dbFile = manifest.db_file;
      if (!dbFile) {
        throw new Error("manifest.json is missing db_file");
      }

      const sqlPromise = initSqlJs({
        locateFile: (file) => `/assets/sqljs/${file}`,
      });

      statusEl.textContent = `Downloading ${dbFile}…`;
      const dbRes = await fetch(`/search/${dbFile}`, { cache: "no-store" });
      if (!dbRes.ok) {
        throw new Error(`Could not load /search/${dbFile} (${dbRes.status})`);
      }
      const dbBytes = await dbRes.arrayBuffer();

      statusEl.textContent = "Opening SQLite in browser…";
      const SQL = await sqlPromise;
      db = new SQL.Database(new Uint8Array(dbBytes));
      supportsFts = detectFtsSupport();

      const sizeMB = ((dbBytes.byteLength || 0) / (1024 * 1024)).toFixed(1);
      const modeLabel = supportsFts ? "fts" : "like-fallback";
      metaEl.innerHTML =
        `<span class=\"pill\">digests: ${escapeHtml(manifest.digest_count ?? "?")}</span> ` +
        `<span class=\"pill\">arXiv IDs: ${escapeHtml(manifest.arxiv_count ?? "?")}</span> ` +
        `<span class=\"pill\">db: ${escapeHtml(dbFile)}</span> ` +
        `<span class=\"pill\">size: ${sizeMB} MB</span> ` +
        `<span class=\"pill\">mode: ${modeLabel}</span>`;

      statusEl.textContent = "Search index ready.";

      if (initialQ) {
        runSearch();
      } else {
        renderEmpty("Search index loaded. Enter a query above.");
      }
    } catch (err) {
      statusEl.textContent = "Failed to initialize search.";
      renderError(String(err));
    }
  }

  init();
})();
