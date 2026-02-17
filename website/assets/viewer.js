(() => {
  const contentEl = document.getElementById("content");
  const statusEl = document.getElementById("status");
  const backLinkEl = document.getElementById("back-link");
  const titleEl = document.getElementById("digest-title");
  const frontmatterDetailsEl = document.getElementById("frontmatter-details");
  const frontmatterCodeEl = document.getElementById("frontmatter-code");
  const filenameRowEl = document.getElementById("filename-row");

  if (!contentEl || !statusEl || !backLinkEl || !titleEl || !frontmatterDetailsEl || !frontmatterCodeEl || !filenameRowEl) {
    throw new Error("Viewer DOM is missing required elements.");
  }

  const params = new URLSearchParams(window.location.search);
  const rawId = (params.get("id") || "").trim();
  const digestId = rawId.endsWith(".md") ? rawId.slice(0, -3) : rawId;

  const escapeHtml = (value) =>
    String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");

  const parseFrontmatter = (markdown) => {
    if (!markdown.startsWith("---\n")) return { frontmatter: {}, rawFrontmatter: "", body: markdown };
    const end = markdown.indexOf("\n---\n", 4);
    if (end === -1) return { frontmatter: {}, rawFrontmatter: "", body: markdown };

    const fmRaw = markdown.slice(4, end);
    const body = markdown.slice(end + 5);
    const frontmatter = {};

    let currentKey = null;
    for (const line of fmRaw.split("\n")) {
      const keyMatch = line.match(/^([A-Za-z0-9_-]+)\s*:\s*(.*)$/);
      if (keyMatch) {
        currentKey = keyMatch[1];
        frontmatter[currentKey] = keyMatch[2].trim();
        continue;
      }

      if (currentKey && /^\s+\S/.test(line)) {
        frontmatter[currentKey] = `${frontmatter[currentKey]} ${line.trim()}`.trim();
      }
    }

    return { frontmatter, rawFrontmatter: fmRaw, body };
  };

  const highlightYaml = (yamlText) =>
    yamlText
      .split("\n")
      .map((line) => {
        if (!line.trim()) return "";
        if (/^\s*#/.test(line)) {
          return `<span class="yaml-comment">${escapeHtml(line)}</span>`;
        }

        const match = line.match(/^(\s*)([^:#][^:]*?)(\s*:\s*)(.*)$/);
        if (!match) return escapeHtml(line);

        const [, indent, key, punct, value] = match;
        return `${escapeHtml(indent)}<span class="yaml-key">${escapeHtml(key)}</span><span class="yaml-punct">${escapeHtml(punct)}</span><span class="yaml-value">${escapeHtml(value)}</span>`;
      })
      .join("\n");

  const extractTitle = (frontmatter, body, fallbackId) => {
    const fmTitle = String(frontmatter.title || "").trim();
    if (fmTitle) return fmTitle;

    const h1 = body.match(/^\s*#\s+(.+)$/m);
    if (h1 && h1[1]) return h1[1].trim().replace(/\s+#+\s*$/, "");

    return fallbackId;
  };

  const stripLeadingH1 = (body) => {
    const match = body.match(/^\s*#\s+[^\n]+\n*/);
    if (!match) return body;
    return body.slice(match[0].length).replace(/^\n+/, "");
  };

  const showFrontmatter = (rawFrontmatter, filename) => {
    frontmatterDetailsEl.hidden = false;
    frontmatterDetailsEl.open = false;

    if (!rawFrontmatter.trim()) {
      frontmatterCodeEl.textContent = "(no frontmatter)";
    } else {
      frontmatterCodeEl.innerHTML = highlightYaml(rawFrontmatter);
    }

    filenameRowEl.textContent = `filename: ${filename}.md`;
  };

  backLinkEl.addEventListener("click", (event) => {
    event.preventDefault();
    window.history.back();
  });

  if (!digestId) {
    titleEl.textContent = "Viewer";
    frontmatterDetailsEl.hidden = true;
    statusEl.textContent = "View home is not implemented yet.";
    contentEl.innerHTML =
      '<p>Please use <a href="/search/">search</a> to find a digest, then open it in the viewer.</p>' +
      "<p>Example: <code>/view/?id=2404.18923_example_20260212_090033</code></p>";
    return;
  }

  if (rawId !== digestId) {
    const canonical = new URL(window.location.href);
    canonical.searchParams.set("id", digestId);
    window.history.replaceState({}, "", canonical.toString());
  }

  statusEl.textContent = "";

  fetch(`/view/${encodeURIComponent(digestId)}.md`, { cache: "no-store" })
    .then((res) => {
      if (!res.ok) throw new Error(`Failed to fetch markdown (${res.status})`);
      return res.text();
    })
    .then((markdown) => {
      const { frontmatter, rawFrontmatter, body } = parseFrontmatter(markdown);
      const resolvedTitle = extractTitle(frontmatter, body, digestId);
      const bodyWithoutDuplicateTitle = stripLeadingH1(body);

      titleEl.textContent = resolvedTitle;
      showFrontmatter(rawFrontmatter, digestId);

      if (window.marked && typeof window.marked.parse === "function") {
        window.marked.setOptions({ gfm: true, breaks: false });
        contentEl.innerHTML = window.marked.parse(bodyWithoutDuplicateTitle);
      } else {
        contentEl.innerHTML = `<pre>${escapeHtml(bodyWithoutDuplicateTitle)}</pre>`;
      }
    })
    .catch((err) => {
      titleEl.textContent = digestId;
      showFrontmatter("", digestId);
      contentEl.innerHTML = `<p class="error">${escapeHtml(String(err))}</p>`;
    });
})();
