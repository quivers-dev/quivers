(function () {
    function diagramPalette() {
        const dark = document.body.getAttribute("data-md-color-scheme") === "slate";
        return dark
            ? {
                  background: "#202732",
                  primaryColor: "#283657",
                  primaryTextColor: "#edf1f6",
                  primaryBorderColor: "#8fa9ff",
                  secondaryColor: "#273c39",
                  secondaryTextColor: "#edf1f6",
                  secondaryBorderColor: "#70c6b2",
                  tertiaryColor: "#352e46",
                  tertiaryTextColor: "#edf1f6",
                  tertiaryBorderColor: "#bba5e7",
                  lineColor: "#a8b3c2",
                  textColor: "#edf1f6",
                  edgeLabelBackground: "#202732",
                  clusterBkg: "#1b212b",
                  clusterBorder: "#465365",
                  noteBkgColor: "#2c3441",
                  noteBorderColor: "#68768a",
                  noteTextColor: "#edf1f6",
              }
            : {
                  background: "#ffffff",
                  primaryColor: "#e7edff",
                  primaryTextColor: "#1b2430",
                  primaryBorderColor: "#3157c8",
                  secondaryColor: "#e8f5f1",
                  secondaryTextColor: "#1b2430",
                  secondaryBorderColor: "#15705f",
                  tertiaryColor: "#f1ebfb",
                  tertiaryTextColor: "#1b2430",
                  tertiaryBorderColor: "#7654a8",
                  lineColor: "#647184",
                  textColor: "#1b2430",
                  edgeLabelBackground: "#f8fafc",
                  clusterBkg: "#f8fafc",
                  clusterBorder: "#c7d1df",
                  noteBkgColor: "#f3f6fa",
                  noteBorderColor: "#aab6c6",
                  noteTextColor: "#1b2430",
              };
    }

    function nearestHeading(diagram) {
        const article = diagram.closest("article");
        let shortLead = "";
        let node = diagram;
        while (node && node !== article) {
            let previous = node.previousElementSibling;
            while (previous) {
                if (previous.matches("h2, h3, h4")) {
                    return previous.textContent.replace("¶", "").trim();
                }
                if (!shortLead && previous.matches("p")) {
                    const text = previous.textContent.trim();
                    if (text.length <= 80) {
                        shortLead = text.replace(/:\s*$/, "");
                    }
                }
                const headings = previous.querySelectorAll("h2, h3, h4");
                if (headings.length) {
                    return headings[headings.length - 1].textContent
                        .replace("¶", "")
                        .trim();
                }
                previous = previous.previousElementSibling;
            }
            node = node.parentElement;
        }
        if (shortLead) return shortLead;
        const pageTitle = article && article.querySelector("h1");
        return pageTitle
            ? pageTitle.textContent.replace("¶", "").trim()
            : "System diagram";
    }

    function diagramKind(source) {
        const firstLine = source.split("\n", 1)[0].trim();
        if (firstLine.startsWith("sequenceDiagram")) return "Sequence";
        if (firstLine.startsWith("classDiagram")) return "Type hierarchy";
        if (/^(flowchart|graph)\s/.test(firstLine)) return "Flow map";
        return "Diagram";
    }

    function diagramDirection(source) {
        const firstLine = source.split("\n", 1)[0].trim();
        const match = firstLine.match(/^(?:flowchart|graph)\s+(TB|TD|BT|LR|RL)/i);
        return match ? match[1].toLowerCase() : "";
    }

    function prepareDiagramSources() {
        document.querySelectorAll("pre.qv-mermaid-source").forEach(function (pre) {
            const code = pre.querySelector("code");
            const source = (code ? code.textContent : pre.textContent) || "";
            const div = document.createElement("div");
            div.className = "mermaid";
            div.dataset.mermaidSource = source.trim();
            div.textContent = source.trim();
            pre.parentNode.replaceChild(div, pre);
        });
    }

    function addDiagramFrame(diagram) {
        if (diagram.closest(".qv-diagram-frame")) return;

        const title = nearestHeading(diagram);
        const figure = document.createElement("figure");
        figure.className = "qv-diagram-frame";
        const caption = document.createElement("figcaption");
        caption.className = "qv-diagram-caption";
        caption.innerHTML = [
            '<span class="qv-diagram-caption__title"></span>',
            '<span class="qv-diagram-caption__kind"></span>',
            '<span class="qv-diagram-caption__action">Select to enlarge</span>',
        ].join("");
        caption.querySelector(".qv-diagram-caption__title").textContent = title;
        const kind = diagramKind(diagram.dataset.mermaidSource || "");
        caption.querySelector(".qv-diagram-caption__kind").textContent = kind;

        diagram.dataset.qvDiagramTitle = title;
        diagram.dataset.qvDiagramKind = kind.toLowerCase().replace(/\s+/g, "-");
        diagram.dataset.qvDiagramDirection = diagramDirection(
            diagram.dataset.mermaidSource || "",
        );
        diagram.parentNode.insertBefore(figure, diagram);
        figure.appendChild(caption);
        figure.appendChild(diagram);
    }

    async function renderQuiversMermaid(options) {
        if (typeof mermaid === "undefined") return;
        prepareDiagramSources();

        const rerender = options && options.rerender;
        const diagrams = Array.from(document.querySelectorAll("div.mermaid"));
        diagrams.forEach(addDiagramFrame);
        if (rerender) {
            diagrams.forEach(function (diagram) {
                if (!diagram.dataset.mermaidSource) return;
                diagram.removeAttribute("data-processed");
                diagram.textContent = diagram.dataset.mermaidSource;
            });
        }

        mermaid.initialize({
            startOnLoad: false,
            theme: "base",
            securityLevel: "loose",
            fontFamily: "IBM Plex Sans, system-ui, sans-serif",
            themeVariables: diagramPalette(),
            flowchart: {
                useMaxWidth: true,
                htmlLabels: true,
                curve: "basis",
                nodeSpacing: 36,
                rankSpacing: 58,
                padding: 14,
            },
            sequence: {
                useMaxWidth: true,
                actorMargin: 54,
                messageMargin: 34,
                boxMargin: 10,
                mirrorActors: false,
            },
        });

        const pending = diagrams.filter(function (diagram) {
            return !diagram.hasAttribute("data-processed");
        });
        if (pending.length) {
            await mermaid.run({ nodes: pending });
        }
    }

    function installPaletteRerender() {
        if (document.documentElement.dataset.qvMermaidPaletteBound === "true") {
            return;
        }
        document.documentElement.dataset.qvMermaidPaletteBound = "true";
        document.addEventListener("change", function (event) {
            if (!event.target.matches('[data-md-component="palette"] input')) {
                return;
            }
            window.setTimeout(function () {
                renderQuiversMermaid({ rerender: true });
            }, 0);
        });
    }

    installPaletteRerender();
    if (typeof document$ !== "undefined") {
        document$.subscribe(function () {
            renderQuiversMermaid();
        });
    } else {
        document.addEventListener("DOMContentLoaded", function () {
            renderQuiversMermaid();
        });
    }
})();
