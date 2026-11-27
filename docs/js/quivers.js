(function () {
    const focusIcon = [
        '<svg viewBox="0 0 24 24" aria-hidden="true">',
        '<path d="M7 3H3v4h2V5h2V3m10 0v2h2v2h2V3h-6M5 17H3v4h4v-2H5v-2m16 0h-2v2h-2v2h6v-4Z"/>',
        "</svg>",
    ].join("");
    let sidebarResizeObserver;

    function storedFocusMode() {
        try {
            return window.localStorage.getItem("qv-focus") === "true";
        } catch (_) {
            return false;
        }
    }

    function setFocusMode(enabled) {
        document.body.classList.toggle("qv-focus", enabled);
        const button = document.querySelector(".qv-focus-toggle");
        if (button) {
            button.setAttribute("aria-pressed", String(enabled));
            button.setAttribute(
                "aria-label",
                enabled ? "Exit focus mode" : "Enter focus mode",
            );
            button.title = enabled ? "Exit focus mode" : "Enter focus mode";
        }
        try {
            window.localStorage.setItem("qv-focus", String(enabled));
        } catch (_) {
            // The mode still works when storage is unavailable.
        }
    }

    function installFocusToggle() {
        const header = document.querySelector(".md-header__inner");
        if (!header || header.querySelector(".qv-focus-toggle")) {
            return;
        }
        const button = document.createElement("button");
        button.type = "button";
        button.className = "qv-focus-toggle";
        button.innerHTML = focusIcon;
        button.addEventListener("click", function () {
            setFocusMode(!document.body.classList.contains("qv-focus"));
        });

        const repository = header.querySelector(".md-header__source");
        header.insertBefore(button, repository || null);
        setFocusMode(storedFocusMode());
    }

    function closeExpandedDiagram(diagram) {
        diagram.classList.remove("is-zoomed");
        diagram.setAttribute("aria-expanded", "false");
        diagram.setAttribute(
            "aria-label",
            "Expand " + (diagram.dataset.qvDiagramTitle || "system") + " diagram",
        );
        document.body.classList.remove("qv-diagram-open");
    }

    function installDiagramZoom() {
        document.querySelectorAll(".mermaid").forEach(function (diagram) {
            if (diagram.dataset.qvZoomBound === "true") {
                return;
            }
            diagram.dataset.qvZoomBound = "true";
            diagram.tabIndex = 0;
            diagram.setAttribute("role", "button");
            const title = diagram.dataset.qvDiagramTitle || "system";
            diagram.setAttribute("aria-label", "Expand " + title + " diagram");
            diagram.setAttribute("aria-expanded", "false");

            function toggle() {
                const expanded = diagram.classList.toggle("is-zoomed");
                diagram.setAttribute("aria-expanded", String(expanded));
                diagram.setAttribute(
                    "aria-label",
                    expanded
                        ? "Close expanded " + title + " diagram"
                        : "Expand " + title + " diagram",
                );
                document.body.classList.toggle("qv-diagram-open", expanded);
            }

            diagram.addEventListener("click", toggle);
            diagram.addEventListener("keydown", function (event) {
                if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    toggle();
                } else if (event.key === "Escape") {
                    closeExpandedDiagram(diagram);
                }
            });
        });
    }

    function installGlobalEscape() {
        if (document.documentElement.dataset.qvEscapeBound === "true") {
            return;
        }
        document.documentElement.dataset.qvEscapeBound = "true";
        document.addEventListener("keydown", function (event) {
            if (event.key !== "Escape") {
                return;
            }
            const diagram = document.querySelector(".mermaid.is-zoomed");
            if (diagram) {
                closeExpandedDiagram(diagram);
                diagram.focus();
            }
        });
    }

    function keepSidebarEndVisible() {
        sidebarResizeObserver?.disconnect();
        const panes = Array.from(document.querySelectorAll(".md-sidebar__scrollwrap"), (pane) => ({
            pane,
            height: pane.clientHeight,
            end: pane.scrollHeight - pane.clientHeight,
        }));
        sidebarResizeObserver = new ResizeObserver(() => {
            for (const state of panes) {
                const { pane } = state;
                const height = pane.clientHeight;
                const end = pane.scrollHeight - height;
                // Keep the last link visible if Material shortens the pane at the footer.
                if (height < state.height && state.end > 0 && pane.scrollTop >= state.end - 1) {
                    pane.scrollTop = end;
                }
                state.height = height;
                state.end = end;
            }
        });
        for (const { pane } of panes) {
            sidebarResizeObserver.observe(pane);
            if (pane.firstElementChild) {
                sidebarResizeObserver.observe(pane.firstElementChild);
            }
        }
    }

    function initializeQuiversUI() {
        installFocusToggle();
        installGlobalEscape();
        keepSidebarEndVisible();
        window.setTimeout(installDiagramZoom, 0);
    }

    if (typeof document$ !== "undefined") {
        document$.subscribe(initializeQuiversUI);
    } else {
        document.addEventListener("DOMContentLoaded", initializeQuiversUI);
    }
})();
