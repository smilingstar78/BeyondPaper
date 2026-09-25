import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { spineSize, swatchFor } from "../state.js";

const GHOST_HEIGHTS = [126, 140, 118, 148, 132, 122, 144, 128, 136];
const GHOSTS = Array.from({ length: 40 }, (_, i) => GHOST_HEIGHTS[i % GHOST_HEIGHTS.length]);

function statusText(paper) {
  if (paper.state === "read") return "Read in full";
  if (paper.state === "reading") return "Reading now";
  return "Abstract only";
}

function Card({ paper, position, canAsk, onAsk, onClose }) {
  const link = paper.pdf || (paper.doi ? paper.doi : "");
  const authors = paper.authors && paper.authors.length ? paper.authors.join(", ") : "Authors not listed";

  return (
    <div
      className="card"
      role="dialog"
      aria-label={paper.title}
      style={{ left: position.left, top: position.top, width: position.width }}
    >
      <button type="button" className="card__close" onClick={onClose} aria-label="Close">
        <span aria-hidden="true">&times;</span>
      </button>

      <h3 className="card__title">{paper.title}</h3>
      <p className="card__authors">{authors}</p>

      <dl className="card__facts">
        <div>
          <dt>Found on</dt>
          <dd>{paper.source === "openalex" ? "OpenAlex" : "arXiv"}</dd>
        </div>
        {paper.year ? (
          <div>
            <dt>Year</dt>
            <dd>{paper.year}</dd>
          </div>
        ) : null}
        {paper.citations != null ? (
          <div>
            <dt>Cited</dt>
            <dd>{paper.citations.toLocaleString()} times</dd>
          </div>
        ) : null}
        {paper.pages ? (
          <div>
            <dt>Length</dt>
            <dd>{paper.pages} pages</dd>
          </div>
        ) : null}
        <div>
          <dt>Status</dt>
          <dd>{statusText(paper)}</dd>
        </div>
      </dl>

      {paper.abstract ? <p className="card__abstract">{paper.abstract}</p> : null}

      <div className="card__actions">
        <button type="button" className="btn btn--solid" disabled={!canAsk} onClick={() => onAsk(paper)}>
          Ask about this paper
        </button>
        {link ? (
          <a className="btn" href={link} target="_blank" rel="noreferrer noopener">
            {paper.pdf ? "Open PDF" : "Open DOI"}
          </a>
        ) : null}
      </div>
    </div>
  );
}

export default function Shelf({
  papers,
  busy,
  folded,
  onToggleFold,
  online,
  onNewTopic,
  canReset,
  hotId,
  openId,
  onOpen,
  onAsk,
  canAsk,
}) {
  const rootRef = useRef(null);
  const rowRef = useRef(null);
  const cardRef = useRef(null);
  const [position, setPosition] = useState(null);

  const readCount = papers.filter((p) => p.state === "read").length;
  const openPaper = papers.find((p) => p.id === openId) || null;

  // Bring newly found papers into view (but not when a saved session loads).
  const seenCount = useRef(papers.length);
  useEffect(() => {
    const row = rowRef.current;
    if (row && papers.length > seenCount.current) {
      row.scrollTo({ left: row.scrollWidth, behavior: "smooth" });
    }
    seenCount.current = papers.length;
  }, [papers.length]);

  // Place the catalogue card under the spine that opened it.
  useLayoutEffect(() => {
    if (!openId || folded) {
      setPosition(null);
      return;
    }

    const spine = document.getElementById(`spine-${openId}`);
    const root = rootRef.current;
    if (!spine || !root) {
      setPosition(null);
      return;
    }

    spine.scrollIntoView({ block: "nearest", inline: "center" });

    const s = spine.getBoundingClientRect();
    const r = root.getBoundingClientRect();
    const width = Math.min(360, window.innerWidth - 24);
    const left = Math.max(12, Math.min(s.left + s.width / 2 - width / 2, window.innerWidth - width - 12));

    setPosition({ left, top: r.bottom + 8, width });
  }, [openId, folded, papers.length]);

  // Close the card on Escape, on an outside click, or when the window changes size.
  useEffect(() => {
    if (!openId) return undefined;

    const onKey = (e) => {
      if (e.key === "Escape") onOpen(null);
    };
    const onPointer = (e) => {
      const inCard = cardRef.current && cardRef.current.contains(e.target);
      const inSpine = e.target.closest && e.target.closest(".spine, .cite");
      if (!inCard && !inSpine) onOpen(null);
    };
    const onResize = () => onOpen(null);

    document.addEventListener("keydown", onKey);
    document.addEventListener("pointerdown", onPointer);
    window.addEventListener("resize", onResize);

    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("pointerdown", onPointer);
      window.removeEventListener("resize", onResize);
    };
  }, [openId, onOpen]);

  const summary =
    papers.length === 0
      ? "Nothing on the shelf yet"
      : `${papers.length} ${papers.length === 1 ? "paper" : "papers"} on the shelf, ${readCount} read in full`;

  return (
    <header className={`shelf${folded ? " is-folded" : ""}`} data-busy={busy ? "true" : "false"} ref={rootRef}>
      <div className="shelf__bar">
        <h1 className="wordmark">Stacks</h1>

        <p className="shelf__summary" aria-live="polite">
          {summary}
        </p>

        <div className="shelf__tools">
          <span className={`status${online ? " is-online" : ""}`} title={online ? "Agent ready" : "Agent offline"}>
            <i aria-hidden="true" />
            <span className="status__text">{online ? "Agent ready" : "Agent offline"}</span>
          </span>
          <button type="button" className="tool" onClick={onToggleFold} aria-expanded={!folded}>
            {folded ? "Show shelf" : "Hide shelf"}
          </button>
          <button type="button" className="tool" onClick={onNewTopic} disabled={!canReset}>
            New topic
          </button>
        </div>
      </div>

      <div className="shelf__bay">
        <div className="shelf__bayInner">
          <div className="shelf__row" ref={rowRef}>
            {papers.map((p, i) => {
              const sw = swatchFor(p);
              const size = spineSize(p);
              const cls = [
                "spine",
                p.state === "reading" ? "is-reading" : "",
                p.state === "read" ? "is-read" : "",
                openId === p.id ? "is-open" : "",
                hotId === p.id ? "is-hot" : "",
              ]
                .filter(Boolean)
                .join(" ");

              return (
                <button
                  key={p.id}
                  id={`spine-${p.id}`}
                  type="button"
                  className={cls}
                  title={p.title}
                  style={{
                    "--bg": sw.bg,
                    "--fg": sw.fg,
                    "--h": `${size.height}px`,
                    "--w": `${size.width}px`,
                    animationDelay: `${Math.min(i, 12) * 55}ms`,
                  }}
                  aria-label={`${p.title}${p.year ? `, ${p.year}` : ""}. ${statusText(p)}.`}
                  aria-expanded={openId === p.id}
                  onClick={() => onOpen(openId === p.id ? null : p.id)}
                >
                  <span className="spine__title">{p.title}</span>
                  <span className="spine__year">{p.year || ""}</span>
                  {p.state === "read" ? <i className="spine__ribbon" aria-hidden="true" /> : null}
                </button>
              );
            })}

            <div className="ghosts" aria-hidden="true">
              {GHOSTS.map((h, i) => (
                <span className="ghost" key={i} style={{ "--h": `${h}px` }} />
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="plank" aria-hidden="true" />

      <p className="shelf__caption">
        {papers.length === 0
          ? "Papers the agent finds will stand here."
          : "Cool spines came from arXiv, warm ones from OpenAlex. A red ribbon means the whole paper was read."}
      </p>

      {openPaper && position ? (
        <div ref={cardRef}>
          <Card
            paper={openPaper}
            position={position}
            canAsk={canAsk}
            onClose={() => onOpen(null)}
            onAsk={(p) => {
              onOpen(null);
              onAsk(p);
            }}
          />
        </div>
      ) : null}
    </header>
  );
}
