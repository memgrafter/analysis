(() => {
  const REPO_URL = "https://github.com/memgrafter/analysis";

  function updatePlacement() {
    const nav = document.getElementById("shared-top-links");
    if (!nav) return;

    let right = 12;
    const container = document.querySelector("main") || document.querySelector("header") || document.body;

    if (container && container !== document.body) {
      const rect = container.getBoundingClientRect();
      if (rect.width > 0) {
        right = Math.max(12, Math.round(window.innerWidth - rect.right));
      }
    }

    nav.style.right = `${right}px`;
  }

  function inject() {
    if (!document.body) return;
    if (document.getElementById("shared-top-links")) return;

    if (!document.getElementById("shared-top-links-style")) {
      const style = document.createElement("style");
      style.id = "shared-top-links-style";
      style.textContent = `
        .shared-top-links {
          position: fixed;
          top: 10px;
          right: 12px;
          z-index: 1000;
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
      <a href="/about/">about</a>
      <span class="sep">|</span>
      <a href="${REPO_URL}" target="_blank" rel="noopener noreferrer">github</a>
    `;

    document.body.appendChild(nav);
    updatePlacement();

    window.addEventListener("resize", updatePlacement, { passive: true });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", inject, { once: true });
  } else {
    inject();
  }
})();
