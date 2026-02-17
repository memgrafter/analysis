(() => {
  const formEl = document.getElementById("search-form");
  const inputEl = document.getElementById("q");
  const statusEl = document.getElementById("status");
  const metaEl = document.getElementById("meta");
  const resultsEl = document.getElementById("results");

  let db = null;
  let workerDb = null;
  let backend = null; // "httpvfs" | "full"
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

  function isFtsError(error) {
    const text = String(error || "").toLowerCase();
    return text.includes("fts5") || text.includes("digests_fts") || text.includes("no such module");
  }

  async function execRaw(sql, params = []) {
    if (backend === "httpvfs" && workerDb?.db) {
      return workerDb.db.exec(sql, params);
    }
    if (backend === "full" && db) {
      return db.exec(sql, params);
    }
    throw new Error("Search backend is not initialized");
  }

  async function execRows(sql, params = []) {
    const out = await execRaw(sql, params);
    if (!Array.isArray(out) || out.length === 0) return [];

    const first = out[0];
    const cols = first.columns || [];
    const values = first.values || [];
    return values.map((rowValues) => {
      const row = {};
      cols.forEach((col, i) => {
        row[col] = rowValues[i];
      });
      return row;
    });
  }

  async function detectFtsSupport() {
    try {
      await execRaw("SELECT rowid FROM digests_fts LIMIT 1");
      return true;
    } catch {
      return false;
    }
  }

  async function queryArxiv(input) {
    return execRows(
      `
      SELECT digest_id, title, arxiv_id
      FROM digests
      WHERE arxiv_id = ?
      ORDER BY timestamp_suffix DESC, digest_id DESC
      LIMIT 50
      `,
      [input.trim()]
    );
  }

  async function queryFts(input) {
    const fts = normalizeFtsQuery(input);
    if (!fts) return [];

    return execRows(
      `
      SELECT d.digest_id AS digest_id, d.title AS title, d.arxiv_id AS arxiv_id, bm25(digests_fts) AS score
      FROM digests_fts
      JOIN digests d ON d.id = digests_fts.rowid
      WHERE digests_fts MATCH ?
      ORDER BY score
      LIMIT 50
      `,
      [fts]
    );
  }

  async function queryLike(input) {
    const q = input.trim().toLowerCase();
    if (!q) return [];
    const pattern = `%${q}%`;

    return execRows(
      `
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
      `,
      [pattern, pattern, pattern, pattern, pattern, pattern, q, q]
    );
  }

  async function runSearch() {
    const q = inputEl.value.trim();
    updateUrlQuery(q);

    if (!q) {
      renderEmpty("Enter a query to search digests.");
      return;
    }

    try {
      statusEl.textContent = `Searching for \"${q}\"…`;
      const started = performance.now();

      let mode = "like";
      let rows = [];

      if (looksLikeArxivId(q)) {
        mode = "arxiv_exact";
        rows = await queryArxiv(q);
      } else if (supportsFts) {
        mode = "fts";
        try {
          rows = await queryFts(q);
        } catch (err) {
          if (isFtsError(err)) {
            supportsFts = false;
            if (backend === "httpvfs") {
              throw new Error(
                "FTS unavailable for HTTP-range backend. Refusing broad LIKE fallback to avoid large transfer."
              );
            }
            mode = "like";
            rows = await queryLike(q);
          } else {
            throw err;
          }
        }
      } else {
        if (backend === "httpvfs") {
          throw new Error(
            "FTS unavailable for HTTP-range backend. Try an arXiv ID query or use full-download fallback."
          );
        }
        mode = "like";
        rows = await queryLike(q);
      }

      const elapsedMs = (performance.now() - started).toFixed(1);
      statusEl.textContent = `Found ${rows.length} result(s) for \"${q}\" in ${elapsedMs} ms (${backend}/${mode}).`;
      renderResults(rows, q);
    } catch (err) {
      statusEl.textContent = "Search failed.";
      renderError(String(err));
    }
  }

  formEl.addEventListener("submit", (event) => {
    event.preventDefault();
    runSearch().catch((err) => {
      statusEl.textContent = "Search failed.";
      renderError(String(err));
    });
  });

  async function initHttpRangeBackend(dbFile) {
    if (typeof createDbWorker !== "function") {
      throw new Error("createDbWorker is unavailable");
    }

    const config = {
      from: "inline",
      config: {
        serverMode: "full",
        requestChunkSize: 4096,
        url: `/search/${dbFile}`,
      },
    };

    workerDb = await createDbWorker(
      [config],
      "/assets/sqljs-httpvfs/sqlite.worker.js",
      "/assets/sqljs-httpvfs/sql-wasm.wasm"
    );

    backend = "httpvfs";
    supportsFts = await detectFtsSupport();
  }

  async function initFullDownloadBackend(dbFile) {
    if (typeof initSqlJs !== "function") {
      throw new Error("initSqlJs is unavailable");
    }

    const sqlPromise = initSqlJs({
      locateFile: (file) => `/assets/sqljs/${file}`,
    });

    const dbRes = await fetch(`/search/${dbFile}`, { cache: "no-store" });
    if (!dbRes.ok) {
      throw new Error(`Could not load /search/${dbFile} (${dbRes.status})`);
    }
    const dbBytes = await dbRes.arrayBuffer();

    const SQL = await sqlPromise;
    db = new SQL.Database(new Uint8Array(dbBytes));
    backend = "full";
    supportsFts = await detectFtsSupport();

    return ((dbBytes.byteLength || 0) / (1024 * 1024)).toFixed(1);
  }

  async function init() {
    try {
      statusEl.textContent = "Loading manifest…";
      const manifestRes = await fetch("/search/manifest.json", { cache: "no-store" });
      if (!manifestRes.ok) {
        throw new Error(`Could not load /search/manifest.json (${manifestRes.status})`);
      }
      const manifest = await manifestRes.json();

      const dbFile = manifest.db_file;
      if (!dbFile) {
        throw new Error("manifest.json is missing db_file");
      }

      let fetchLabel = "on-demand-range";
      try {
        statusEl.textContent = "Initializing HTTP range SQLite…";
        await initHttpRangeBackend(dbFile);
      } catch (rangeErr) {
        statusEl.textContent = `HTTP range init failed (${String(rangeErr)}). Falling back to full download…`;
        fetchLabel = `${await initFullDownloadBackend(dbFile)} MB`;
      }

      const modeLabel = supportsFts ? "fts" : "like-fallback";
      const backendLabel = backend === "httpvfs" ? "http-range" : "full-download";

      metaEl.innerHTML =
        `<span class=\"pill\">digests: ${escapeHtml(manifest.digest_count ?? "?")}</span> ` +
        `<span class=\"pill\">arXiv IDs: ${escapeHtml(manifest.arxiv_count ?? "?")}</span> ` +
        `<span class=\"pill\">db: ${escapeHtml(dbFile)}</span> ` +
        `<span class=\"pill\">backend: ${backendLabel}</span> ` +
        `<span class=\"pill\">search: ${modeLabel}</span> ` +
        `<span class=\"pill\">fetch: ${escapeHtml(fetchLabel)}</span>`;

      statusEl.textContent = "Search index ready.";

      if (initialQ) {
        await runSearch();
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
