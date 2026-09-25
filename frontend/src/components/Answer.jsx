import { memo, useMemo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { swatchFor } from "../state.js";

const escapeRegExp = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

// Turns every paper title mentioned in the answer into a link to its spine on the shelf.
function linkTitles(markdown, papers) {
  const usable = papers.filter((p) => p.title && p.title.length >= 14);
  if (!usable.length) return markdown;

  const byTitle = new Map(usable.map((p) => [p.title.toLowerCase(), p]));
  const pattern = usable
    .map((p) => p.title)
    .sort((a, b) => b.length - a.length)
    .map(escapeRegExp)
    .join("|");

  return markdown.replace(new RegExp(pattern, "gi"), (match) => {
    const paper = byTitle.get(match.toLowerCase());
    if (!paper) return match;
    return `[${match.replace(/[[\]]/g, "\\$&")}](#paper-${paper.id})`;
  });
}

function Answer({ text, papers, onHot, onOpenPaper }) {
  const source = useMemo(() => linkTitles(text, papers), [text, papers]);
  const byId = useMemo(() => new Map(papers.map((p) => [p.id, p])), [papers]);

  const components = {
    a({ href, children }) {
      if (href && href.startsWith("#paper-")) {
        const id = href.slice("#paper-".length);
        const paper = byId.get(id);

        if (paper) {
          return (
            <a
              href={href}
              className="cite"
              style={{ "--c": swatchFor(paper).bg }}
              onMouseEnter={() => onHot(id)}
              onMouseLeave={() => onHot(null)}
              onFocus={() => onHot(id)}
              onBlur={() => onHot(null)}
              onClick={(e) => {
                e.preventDefault();
                onOpenPaper(id);
              }}
              title="Show this paper on the shelf"
            >
              {children}
            </a>
          );
        }

        return <span>{children}</span>;
      }

      return (
        <a href={href} target="_blank" rel="noreferrer noopener">
          {children}
        </a>
      );
    },
    table({ children }) {
      return (
        <div className="table-wrap">
          <table>{children}</table>
        </div>
      );
    },
  };

  return (
    <div className="prose">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {source}
      </ReactMarkdown>
    </div>
  );
}

export default memo(Answer);
