(() => {
  const formEl = document.getElementById("search-form");
  const inputEl = document.getElementById("q");
  const statusEl = document.getElementById("status");
  const metaEl = document.getElementById("meta");
  const resultsEl = document.getElementById("results");

  const pagerTopEl = document.getElementById("pager-top");
  const pagerBottomEl = document.getElementById("pager-bottom");

  const controls = [
    {
      prev: document.getElementById("prev-top"),
      next: document.getElementById("next-top"),
      label: document.getElementById("page-label-top"),
      pageSize: document.getElementById("page-size-top"),
      sort: document.getElementById("sort-top"),
    },
    {
      prev: document.getElementById("prev-bottom"),
      next: document.getElementById("next-bottom"),
      label: document.getElementById("page-label-bottom"),
      pageSize: document.getElementById("page-size-bottom"),
      sort: document.getElementById("sort-bottom"),
    },
  ];

  const PAGE_SIZE_OPTIONS = new Set([10, 25, 50, 100]);
  const DEFAULT_PAGE_SIZE = 10;
  const SORT_OPTIONS = new Set(["newest", "relevance", "title_asc"]);
  const DEFAULT_SORT = "relevance";

  let db = null;
  let workerDb = null;
  let backend = null; // "httpvfs" | "full"
  let supportsFts = false;
  let isSearching = false;

  const state = {
    query: "",
    mode: null, // "fts" | "arxiv_exact"
    sort: DEFAULT_SORT,
    pageSize: DEFAULT_PAGE_SIZE,
    currentPage: 1,
    currentCursorToken: null,
    nextCursorToken: null,
    cursorStack: [],
    hasMore: false,
    rowCount: 0,
  };

  class CursorError extends Error {
    constructor(message) {
      super(message);
      this.name = "CursorError";
    }
  }

  const escapeHtml = (value) =>
    String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");

  function parsePageSize(value) {
    const parsed = Number.parseInt(String(value || ""), 10);
    return PAGE_SIZE_OPTIONS.has(parsed) ? parsed : DEFAULT_PAGE_SIZE;
  }

  function parseSort(value) {
    const sort = String(value || "").trim();
    return SORT_OPTIONS.has(sort) ? sort : DEFAULT_SORT;
  }

  function setPagerVisible(visible) {
    pagerTopEl.hidden = !visible;
    pagerBottomEl.hidden = !visible;
  }

  function syncPageSizeControls() {
    for (const control of controls) {
      control.pageSize.value = String(state.pageSize);
    }
  }

  function syncSortControls() {
    for (const control of controls) {
      control.sort.value = state.sort;
    }
  }

  function setControlsDisabled(disabled) {
    for (const control of controls) {
      control.prev.disabled = disabled || state.currentPage <= 1 || state.cursorStack.length === 0;
      control.next.disabled = disabled || !state.hasMore || !state.nextCursorToken;
      control.pageSize.disabled = disabled;
      control.sort.disabled = disabled;
    }
  }

  function updatePagerLabels() {
    const start = state.rowCount > 0 ? (state.currentPage - 1) * state.pageSize + 1 : 0;
    const end = state.rowCount > 0 ? start + state.rowCount - 1 : 0;
    const rangeText = state.rowCount > 0 ? ` · showing ${start}-${end}` : "";
    const label = `Page ${state.currentPage}${rangeText}`;

    for (const control of controls) {
      control.label.textContent = label;
    }
  }

  function updatePagerUi() {
    syncPageSizeControls();
    syncSortControls();
    updatePagerLabels();
    setControlsDisabled(isSearching);
  }

  function renderEmpty(message) {
    resultsEl.innerHTML = `<p class="muted">${escapeHtml(message)}</p>`;
  }

  function renderError(message) {
    resultsEl.innerHTML = `<p style="color:#cc3b3b;font-weight:600;">${escapeHtml(message)}</p>`;
  }

  function renderResults(rows, q) {
    if (rows.length === 0) {
      renderEmpty(`No results for "${q}"`);
      return;
    }

    resultsEl.innerHTML = rows
      .map((row) => {
        const digestId = row.digest_id || "";
        const title = row.title || "(untitled)";
        const takeaway = (row.core_contribution || "").trim();
        const arxivId = (row.arxiv_id || "").trim();
        const arxivLink = arxivId
          ? `<a href="https://arxiv.org/abs/${encodeURIComponent(arxivId)}" target="_blank" rel="noopener noreferrer">${escapeHtml(arxivId)}</a>`
          : "(no arXiv ID)";

        return `
          <article class="result">
            <div class="title"><a href="/view/?id=${encodeURIComponent(digestId)}">${escapeHtml(title)}</a></div>
            <div class="result-meta-links">${arxivLink} | <a href="/view/?id=${encodeURIComponent(digestId)}">view</a> · <a href="/view/${encodeURIComponent(digestId)}.md">raw</a></div>
            <div class="result-takeaway">${escapeHtml(takeaway || "(No key takeaway available)")}</div>
          </article>
        `;
      })
      .join("\n");
  }

  function normalizeFtsQuery(input) {
    const normalized = input
      .toLowerCase()
      // Treat common separators as token boundaries so queries like "test-time" become "test time".
      .replace(/[-_/]+/g, " ")
      .replace(/[^a-z0-9.\s]/g, " ");

    const tokens = normalized.split(/\s+/).filter(Boolean);
    return tokens.join(" AND ");
  }

  function looksLikeArxivId(input) {
    return /^\d{4}\.\d{4,5}(v\d+)?$/i.test(input.trim());
  }

  function isFtsError(error) {
    const text = String(error || "").toLowerCase();
    return text.includes("fts5") || text.includes("digests_fts") || text.includes("no such module");
  }

  function hashString(value) {
    let h = 2166136261;
    for (let i = 0; i < value.length; i += 1) {
      h ^= value.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return (h >>> 0).toString(16);
  }

  function queryHash(mode, sort, query) {
    return hashString(`${mode}\n${sort}\n${query.trim().toLowerCase()}`);
  }

  function encodeCursor(payload) {
    const json = JSON.stringify(payload);
    const bytes = new TextEncoder().encode(json);
    let binary = "";
    for (const byte of bytes) {
      binary += String.fromCharCode(byte);
    }

    return btoa(binary)
      .replace(/\+/g, "-")
      .replace(/\//g, "_")
      .replace(/=+$/g, "");
  }

  function decodeCursor(token) {
    try {
      const padded = token.replace(/-/g, "+").replace(/_/g, "/") + "=".repeat((4 - (token.length % 4)) % 4);
      const binary = atob(padded);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i += 1) {
        bytes[i] = binary.charCodeAt(i);
      }
      const json = new TextDecoder().decode(bytes);
      return JSON.parse(json);
    } catch {
      return null;
    }
  }

  function buildNextCursor(mode, sort, query, pageSize, nextPageNumber, lastRow) {
    const base = {
      v: 1,
      m: mode,
      srt: sort,
      qh: queryHash(mode, sort, query),
      ps: pageSize,
      p: nextPageNumber,
    };

    if (sort === "relevance") {
      return encodeCursor({ ...base, sc: Number(lastRow.score), id: String(lastRow.digest_id || "") });
    }

    if (sort === "title_asc") {
      return encodeCursor({ ...base, tk: String(lastRow.title_key || ""), id: String(lastRow.digest_id || "") });
    }

    // newest
    return encodeCursor({ ...base, a: String(lastRow.arxiv_key || ""), id: String(lastRow.digest_id || "") });
  }

  function validateCursor(token, mode, sort, query, pageSize) {
    if (!token) return null;
    const payload = decodeCursor(token);
    if (!payload || typeof payload !== "object") {
      throw new CursorError("Invalid cursor token");
    }

    const expectedQh = queryHash(mode, sort, query);
    if (
      payload.v !== 1 ||
      payload.m !== mode ||
      payload.srt !== sort ||
      payload.qh !== expectedQh ||
      Number(payload.ps) !== pageSize
    ) {
      throw new CursorError("Cursor does not match current query/page size/sort");
    }

    return payload;
  }

  function updateUrlState() {
    const url = new URL(window.location.href);

    if (state.query) {
      url.searchParams.set("q", state.query);
      url.searchParams.set("ps", String(state.pageSize));
      url.searchParams.set("sort", state.sort);
    } else {
      url.searchParams.delete("q");
      url.searchParams.delete("ps");
      url.searchParams.delete("sort");
    }

    if (state.currentCursorToken) {
      url.searchParams.set("cursor", state.currentCursorToken);
    } else {
      url.searchParams.delete("cursor");
    }

    window.history.replaceState({}, "", url.toString());
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

  async function queryArxivPage(input, pageSize, cursor, sort) {
    const limitPlusOne = pageSize + 1;
    const arxivId = input.trim();

    const base = `
      WITH scoped AS (
        SELECT
          digest_id,
          title,
          arxiv_id,
          core_contribution,
          COALESCE(arxiv_id, '') AS arxiv_key,
          lower(COALESCE(title, '')) AS title_key,
          0.0 AS score
        FROM digests
        WHERE arxiv_id = ?
      )
      SELECT *
      FROM scoped
    `;

    let rows;
    if (sort === "title_asc") {
      if (!cursor) {
        rows = await execRows(
          `${base}
          ORDER BY title_key ASC, digest_id ASC
          LIMIT ?
          `,
          [arxivId, limitPlusOne]
        );
      } else {
        rows = await execRows(
          `${base}
          WHERE
            title_key > ?
            OR (title_key = ? AND digest_id > ?)
          ORDER BY title_key ASC, digest_id ASC
          LIMIT ?
          `,
          [arxivId, cursor.tk, cursor.tk, cursor.id, limitPlusOne]
        );
      }
    } else if (sort === "relevance") {
      if (!cursor) {
        rows = await execRows(
          `${base}
          ORDER BY score ASC, digest_id ASC
          LIMIT ?
          `,
          [arxivId, limitPlusOne]
        );
      } else {
        rows = await execRows(
          `${base}
          WHERE
            score > ?
            OR (score = ? AND digest_id > ?)
          ORDER BY score ASC, digest_id ASC
          LIMIT ?
          `,
          [arxivId, cursor.sc, cursor.sc, cursor.id, limitPlusOne]
        );
      }
    } else {
      // newest
      if (!cursor) {
        rows = await execRows(
          `${base}
          ORDER BY arxiv_key DESC, digest_id DESC
          LIMIT ?
          `,
          [arxivId, limitPlusOne]
        );
      } else {
        rows = await execRows(
          `${base}
          WHERE
            arxiv_key < ?
            OR (arxiv_key = ? AND digest_id < ?)
          ORDER BY arxiv_key DESC, digest_id DESC
          LIMIT ?
          `,
          [arxivId, cursor.a, cursor.a, cursor.id, limitPlusOne]
        );
      }
    }

    const hasMore = rows.length > pageSize;
    const pageRows = hasMore ? rows.slice(0, pageSize) : rows;
    return { rows: pageRows, hasMore };
  }

  async function queryFtsPage(input, pageSize, cursor, sort) {
    const fts = normalizeFtsQuery(input);
    if (!fts) return { rows: [], hasMore: false };

    const limitPlusOne = pageSize + 1;
    const base = `
      SELECT *
      FROM (
        SELECT
          d.digest_id AS digest_id,
          d.title AS title,
          d.arxiv_id AS arxiv_id,
          d.core_contribution AS core_contribution,
          COALESCE(d.arxiv_id, '') AS arxiv_key,
          lower(COALESCE(d.title, '')) AS title_key,
          bm25(digests_fts) AS score
        FROM digests_fts
        JOIN digests d ON d.id = digests_fts.rowid
        WHERE digests_fts MATCH ?
      ) ranked
    `;

    let rows;

    if (sort === "newest") {
      if (!cursor) {
        rows = await execRows(
          `${base}
          ORDER BY arxiv_key DESC, digest_id DESC
          LIMIT ?
          `,
          [fts, limitPlusOne]
        );
      } else {
        rows = await execRows(
          `${base}
          WHERE
            arxiv_key < ?
            OR (arxiv_key = ? AND digest_id < ?)
          ORDER BY arxiv_key DESC, digest_id DESC
          LIMIT ?
          `,
          [fts, cursor.a, cursor.a, cursor.id, limitPlusOne]
        );
      }
    } else if (sort === "title_asc") {
      if (!cursor) {
        rows = await execRows(
          `${base}
          ORDER BY title_key ASC, digest_id ASC
          LIMIT ?
          `,
          [fts, limitPlusOne]
        );
      } else {
        rows = await execRows(
          `${base}
          WHERE
            title_key > ?
            OR (title_key = ? AND digest_id > ?)
          ORDER BY title_key ASC, digest_id ASC
          LIMIT ?
          `,
          [fts, cursor.tk, cursor.tk, cursor.id, limitPlusOne]
        );
      }
    } else {
      // relevance
      if (!cursor) {
        rows = await execRows(
          `${base}
          ORDER BY score ASC, digest_id ASC
          LIMIT ?
          `,
          [fts, limitPlusOne]
        );
      } else {
        rows = await execRows(
          `${base}
          WHERE
            score > ?
            OR (score = ? AND digest_id > ?)
          ORDER BY score ASC, digest_id ASC
          LIMIT ?
          `,
          [fts, cursor.sc, cursor.sc, cursor.id, limitPlusOne]
        );
      }
    }

    const hasMore = rows.length > pageSize;
    const pageRows = hasMore ? rows.slice(0, pageSize) : rows;
    return { rows: pageRows, hasMore };
  }

  function detectMode(input) {
    if (looksLikeArxivId(input)) return "arxiv_exact";
    if (supportsFts) return "fts";
    throw new Error("FTS is unavailable for this DB/runtime. Only exact arXiv ID search is available.");
  }

  async function fetchPage(input, sort, pageSize, cursorToken) {
    const mode = detectMode(input);
    const cursor = validateCursor(cursorToken, mode, sort, input, pageSize);

    if (mode === "arxiv_exact") {
      const result = await queryArxivPage(input, pageSize, cursor, sort);
      return { mode, cursor, ...result };
    }

    try {
      const result = await queryFtsPage(input, pageSize, cursor, sort);
      return { mode, cursor, ...result };
    } catch (err) {
      if (isFtsError(err)) {
        supportsFts = false;
        throw new Error("FTS query failed. Rebuild DB/runtime with FTS5 support.");
      }
      throw err;
    }
  }

  async function runSearch({ allowCursorReset = true } = {}) {
    const q = inputEl.value.trim();
    state.query = q;

    if (!q) {
      state.mode = null;
      state.currentPage = 1;
      state.currentCursorToken = null;
      state.nextCursorToken = null;
      state.cursorStack = [];
      state.hasMore = false;
      state.rowCount = 0;
      updateUrlState();
      setPagerVisible(false);
      updatePagerUi();
      renderEmpty("Enter a query to search digests.");
      statusEl.textContent = "Search index ready.";
      return;
    }

    let shouldRetryWithoutCursor = false;

    isSearching = true;
    updatePagerUi();

    try {
      statusEl.textContent = `Searching for \"${q}\"…`;
      const started = performance.now();

      const page = await fetchPage(q, state.sort, state.pageSize, state.currentCursorToken);
      state.mode = page.mode;

      if (page.cursor && Number.isInteger(page.cursor.p) && page.cursor.p > 1) {
        state.currentPage = page.cursor.p;
      } else if (!state.currentCursorToken) {
        state.currentPage = 1;
      }

      state.hasMore = page.hasMore;
      state.rowCount = page.rows.length;

      const nextPageNumber = state.currentPage + 1;
      state.nextCursorToken =
        page.hasMore && page.rows.length > 0
          ? buildNextCursor(state.mode, state.sort, q, state.pageSize, nextPageNumber, page.rows[page.rows.length - 1])
          : null;

      const elapsedMs = (performance.now() - started).toFixed(1);
      statusEl.textContent = `Showing ${page.rows.length} result(s) for \"${q}\" in ${elapsedMs} ms (${backend}/${state.mode}/${state.sort}).`;

      setPagerVisible(true);
      updateUrlState();
      updatePagerUi();
      renderResults(page.rows, q);
    } catch (err) {
      if (allowCursorReset && err instanceof CursorError && state.currentCursorToken) {
        state.currentCursorToken = null;
        state.cursorStack = [];
        state.currentPage = 1;
        shouldRetryWithoutCursor = true;
      } else {
        state.hasMore = false;
        state.nextCursorToken = null;
        state.rowCount = 0;
        updatePagerUi();
        statusEl.textContent = "Search failed.";
        renderError(String(err));
      }
    } finally {
      isSearching = false;
      updatePagerUi();
    }

    if (shouldRetryWithoutCursor) {
      await runSearch({ allowCursorReset: false });
    }
  }

  async function goToNextPage() {
    if (isSearching || !state.nextCursorToken) return;

    state.cursorStack.push(state.currentCursorToken);
    state.currentCursorToken = state.nextCursorToken;
    state.currentPage += 1;

    await runSearch();
  }

  async function goToPrevPage() {
    if (isSearching || state.cursorStack.length === 0) return;

    state.currentCursorToken = state.cursorStack.pop() || null;
    state.currentPage = Math.max(1, state.currentPage - 1);

    await runSearch();
  }

  async function changePageSize(nextPageSize) {
    const parsed = parsePageSize(nextPageSize);
    if (parsed === state.pageSize) {
      syncPageSizeControls();
      return;
    }

    state.pageSize = parsed;
    state.currentCursorToken = null;
    state.nextCursorToken = null;
    state.cursorStack = [];
    state.currentPage = 1;

    if (state.query) {
      await runSearch();
    } else {
      updateUrlState();
      updatePagerUi();
    }
  }

  async function changeSort(nextSort) {
    const parsed = parseSort(nextSort);
    if (parsed === state.sort) {
      syncSortControls();
      return;
    }

    state.sort = parsed;
    state.currentCursorToken = null;
    state.nextCursorToken = null;
    state.cursorStack = [];
    state.currentPage = 1;

    if (state.query) {
      await runSearch();
    } else {
      updateUrlState();
      updatePagerUi();
    }
  }

  formEl.addEventListener("submit", (event) => {
    event.preventDefault();

    state.currentCursorToken = null;
    state.nextCursorToken = null;
    state.cursorStack = [];
    state.currentPage = 1;

    runSearch().catch((err) => {
      statusEl.textContent = "Search failed.";
      renderError(String(err));
    });
  });

  for (const control of controls) {
    control.prev.addEventListener("click", () => {
      goToPrevPage().catch((err) => {
        statusEl.textContent = "Search failed.";
        renderError(String(err));
      });
    });

    control.next.addEventListener("click", () => {
      goToNextPage().catch((err) => {
        statusEl.textContent = "Search failed.";
        renderError(String(err));
      });
    });

    control.pageSize.addEventListener("change", (event) => {
      changePageSize(event.target.value).catch((err) => {
        statusEl.textContent = "Search failed.";
        renderError(String(err));
      });
    });

    control.sort.addEventListener("change", (event) => {
      changeSort(event.target.value).catch((err) => {
        statusEl.textContent = "Search failed.";
        renderError(String(err));
      });
    });
  }

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
      const qs = new URLSearchParams(window.location.search);
      const initialQ = (qs.get("q") || "").trim();
      const initialPs = parsePageSize(qs.get("ps"));
      const initialSort = parseSort(qs.get("sort"));
      const initialCursor = (qs.get("cursor") || "").trim() || null;

      state.pageSize = initialPs;
      state.sort = initialSort;
      state.currentCursorToken = initialCursor;
      inputEl.value = initialQ;
      state.query = initialQ;

      if (initialCursor) {
        const payload = decodeCursor(initialCursor);
        if (payload && Number.isInteger(payload.p) && payload.p > 1) {
          state.currentPage = payload.p;
        }
      }

      updatePagerUi();

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

      const modeLabel = supportsFts ? "fts" : "arxiv-only (fts unavailable)";
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
        setPagerVisible(true);
        await runSearch();
      } else {
        setPagerVisible(false);
        renderEmpty("Search index loaded. Enter a query above.");
      }
    } catch (err) {
      statusEl.textContent = "Failed to initialize search.";
      renderError(String(err));
    }
  }

  init();
})();
