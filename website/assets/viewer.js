(() => {
  const contentEl = document.getElementById("content");
  const metaEl = document.getElementById("meta");
  const statusEl = document.getElementById("status");

  const params = new URLSearchParams(window.location.search);
  const rawId = (params.get("id") || "").trim();
  const digestId = rawId.endsWith(".md") ? rawId.slice(0, -3) : rawId;

  const escapeHtml = (value) =>
    value
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");

  const parseFrontmatter = (markdown) => {
    if (!markdown.startsWith("---\n")) return { frontmatter: {}, body: markdown };
    const end = markdown.indexOf("\n---\n", 4);
    if (end === -1) return { frontmatter: {}, body: markdown };

    const fmRaw = markdown.slice(4, end);
    const body = markdown.slice(end + 5);
    const frontmatter = {};

    for (const line of fmRaw.split("\n")) {
      const idx = line.indexOf(":");
      if (idx === -1) continue;
      const key = line.slice(0, idx).trim();
      const value = line.slice(idx + 1).trim();
      if (key) frontmatter[key] = value;
    }

    return { frontmatter, body };
  };

  if (!digestId) {
    statusEl.textContent = "Missing query parameter: ?id=<digest-id>";
    contentEl.innerHTML =
      "<p>Example: <code>/view/?id=2404.18923_example_20260212_090033</code></p>";
    return;
  }

  if (rawId !== digestId) {
    const canonical = new URL(window.location.href);
    canonical.searchParams.set("id", digestId);
    window.history.replaceState({}, "", canonical.toString());
  }

  statusEl.innerHTML = `Viewing <code>${escapeHtml(digestId)}</code> · <a href="/view/${encodeURIComponent(digestId)}.md">raw</a>`;

  fetch(`/view/${encodeURIComponent(digestId)}.md`, { cache: "no-store" })
    .then((res) => {
      if (!res.ok) throw new Error(`Failed to fetch markdown (${res.status})`);
      return res.text();
    })
    .then((markdown) => {
      const { frontmatter, body } = parseFrontmatter(markdown);
      const entries = Object.entries(frontmatter);

      if (entries.length === 0) {
        metaEl.innerHTML = "<p>No YAML frontmatter detected.</p>";
      } else {
        metaEl.innerHTML = entries
          .map(([k, v]) => `<p><strong>${escapeHtml(k)}:</strong> ${escapeHtml(v)}</p>`)
          .join("\n");
      }

      if (window.marked && typeof window.marked.parse === "function") {
        contentEl.innerHTML = window.marked.parse(body);
      } else {
        contentEl.innerHTML = `<pre>${escapeHtml(body)}</pre>`;
      }
    })
    .catch((err) => {
      contentEl.innerHTML = `<p class="error">${escapeHtml(String(err))}</p>`;
      metaEl.innerHTML = "";
    });
})();
