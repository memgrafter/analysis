(() => {
  const REPO_URL = "https://github.com/memgrafter/analysis";

  function inject() {
    if (!document.body) return;
    if (document.getElementById("shared-top-links")) return;

    if (!document.getElementById("shared-top-links-style")) {
      const style = document.createElement("style");
      style.id = "shared-top-links-style";
      style.textContent = `
        .shared-top-links-row {
          display: flex;
          justify-content: flex-end;
          margin: 0 0 12px;
        }

        .shared-top-links {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          padding: 4px 8px;
          border: 1px solid var(--border, #d0d6e8);
          border-radius: 999px;
          background: var(--panel, rgba(255,255,255,0.92));
          box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
          font-size: 12px;
          line-height: 1;
        }

        .shared-top-links a {
          color: var(--accent, #2f63d8);
          text-decoration: none;
        }

        .shared-top-links .sep {
          color: var(--muted, #6b7280);
          user-select: none;
        }
      `;
      document.head.appendChild(style);
    }

    const nav = document.createElement("nav");
    nav.id = "shared-top-links";
    nav.className = "shared-top-links";
    nav.setAttribute("aria-label", "Site links");
    nav.innerHTML = `
      <a href="/search/">search</a>
      <span class="sep">|</span>
      <a href="/cloud/">cloud</a>
      <span class="sep">|</span>
      <a href="/about/">AGENTS.md</a>
      <span class="sep">|</span>
      <a href="${REPO_URL}" target="_blank" rel="noopener noreferrer">github</a>
    `;

    const row = document.createElement("div");
    row.className = "shared-top-links-row";
    row.appendChild(nav);

    const host = document.querySelector("main") || document.body;
    host.insertBefore(row, host.firstChild);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", inject, { once: true });
  } else {
    inject();
  }
})();
