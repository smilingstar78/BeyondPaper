// All app state lives here:
// the conversation, the shelf of papers, and helpers.

const STORAGE_KEY = "stacks.session.v1";


// ============================================================
// ID
// ============================================================

export function uid(prefix = "id") {
  return (
    prefix +
    Math.random()
      .toString(36)
      .slice(2, 9) +
    Date.now()
      .toString(36)
      .slice(-3)
  );
}


// ============================================================
// Small helpers
// ============================================================

function hash(text) {
  let h = 5381;
  const s = String(text || "");

  for (let i = 0; i < s.length; i++) {
    h =
      ((h << 5) +
        h +
        s.charCodeAt(i)) >>>
      0;
  }

  return h;
}


const normalizeTitle = (s) =>
  String(s || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();


/**
 * Two links to the same paper
 * get the same key.
 *
 * Handles:
 * - arxiv abstract URL
 * - arxiv PDF URL
 * - http / https
 * - trailing slash
 * - .pdf
 * - arxiv versions
 */
export function urlKey(url) {
  if (!url) return "";

  const text = String(url);

  const arxivId = text.match(
    /(\d{4}\.\d{4,5})(v\d+)?/
  );

  if (
    /arxiv\.org/i.test(text) &&
    arxivId
  ) {
    return (
      "arxiv:" +
      arxivId[1]
    );
  }

  return text
    .toLowerCase()
    .replace(
      /^https?:\/\/(www\.)?/,
      ""
    )
    .replace(
      /[#?].*$/,
      ""
    )
    .replace(
      /\/$/,
      ""
    )
    .replace(
      /\.pdf$/,
      ""
    );
}


// ============================================================
// Shelf appearance
// ============================================================

// Cool spines = arXiv
// Warm spines = OpenAlex

const COOL = [
  {
    bg: "#4a76a6",
    fg: "#f4f7fb"
  },
  {
    bg: "#3e7f80",
    fg: "#f2f8f8"
  },
  {
    bg: "#93b6c9",
    fg: "#14262d"
  },
  {
    bg: "#34507c",
    fg: "#eef2f8"
  },
  {
    bg: "#7fa383",
    fg: "#12261a"
  },
  {
    bg: "#b7c9d9",
    fg: "#16232c"
  }
];

const WARM = [
  {
    bg: "#dba53f",
    fg: "#2a1d05"
  },
  {
    bg: "#b8483d",
    fg: "#fbefec"
  },
  {
    bg: "#e58f5d",
    fg: "#2c1408"
  },
  {
    bg: "#cf7d8f",
    fg: "#2b0f16"
  },
  {
    bg: "#8c3a4a",
    fg: "#fbecef"
  },
  {
    bg: "#ead27f",
    fg: "#2a2409"
  }
];


export function swatchFor(paper) {
  const list =
    paper.source === "openalex"
      ? WARM
      : COOL;

  return list[
    hash(paper.key) %
      list.length
  ];
}


export function spineSize(paper) {
  let height =
    118 +
    (hash(paper.key) % 26);

  if (paper.pages) {
    height = Math.max(
      122,
      Math.min(
        160,
        108 +
          paper.pages * 3
      )
    );
  }

  const c = paper.citations;

  const width =
    c == null
      ? 38
      : Math.round(
          34 +
            Math.min(
              32,
              Math.log10(c + 1) *
                9.5
            )
        );

  return {
    height,
    width
  };
}


// ============================================================
// Papers
// ============================================================

export function addPapers(
  papers,
  incoming
) {
  const out = papers.slice();

  for (const raw of incoming || []) {
    if (!raw || !raw.title) {
      continue;
    }

    const key =
      normalizeTitle(
        raw.title
      );

    if (!key) {
      continue;
    }

    const at =
      out.findIndex(
        (p) =>
          p.key === key ||
          (
            raw.pdf &&
            p.pdf &&
            urlKey(p.pdf) ===
              urlKey(raw.pdf)
          )
      );

    if (at >= 0) {
      const p = out[at];

      out[at] = {
        ...p,

        authors:
          p.authors &&
          p.authors.length
            ? p.authors
            : raw.authors || [],

        year:
          p.year ||
          raw.year ||
          null,

        citations:
          p.citations ??
          raw.citations ??
          null,

        abstract:
          p.abstract ||
          raw.abstract ||
          "",

        pdf:
          p.pdf ||
          raw.pdf ||
          "",

        doi:
          p.doi ||
          raw.doi ||
          "",

        source:
          p.source ||
          (
            raw.source ===
            "openalex"
              ? "openalex"
              : "arxiv"
          )
      };
    } else {
      out.push({
        id:
          "p" +
          (out.length + 1),

        key,

        title:
          raw.title,

        authors:
          raw.authors || [],

        year:
          raw.year || null,

        citations:
          raw.citations ??
          null,

        abstract:
          raw.abstract || "",

        pdf:
          raw.pdf || "",

        doi:
          raw.doi || "",

        source:
          raw.source ===
          "openalex"
            ? "openalex"
            : "arxiv",

        // seen -> reading -> read
        state: "seen",

        pages: null
      });
    }
  }

  return out;
}


function updateByUrl(
  papers,
  url,
  patch
) {
  const key = urlKey(url);

  if (!key) {
    return papers;
  }

  let list = papers;

  let at =
    list.findIndex(
      (p) =>
        p.pdf &&
        urlKey(p.pdf) === key
    );

  if (at < 0) {
    // The agent read a paper
    // that never came from a search.
    const guess =
      String(url)
        .split("/")
        .filter(Boolean)
        .pop() ||
      String(url);

    list = addPapers(
      list,
      [
        {
          title: guess,
          pdf: String(url),
          source: "arxiv"
        }
      ]
    );

    at =
      list.findIndex(
        (p) =>
          p.pdf &&
          urlKey(p.pdf) === key
      );

    if (at < 0) {
      return papers;
    }
  }

  return list.map(
    (p, i) =>
      i === at
        ? {
            ...p,
            ...patch
          }
        : p
  );
}


// ============================================================
// Conversation events
// ============================================================

function applyEvent(
  ex,
  event
) {
  switch (event.type) {

    // --------------------------------------------------------
    // NEW:
    // Backend sends explicit "step" events.
    // --------------------------------------------------------

    case "step":
      return {
        ...ex,

        steps: [
          ...ex.steps,

          {
            id:
              event.id ||
              uid("step"),

            text:
              event.text ||
              "",

            status:
              event.status ||
              "running",

            name:
              event.name ||
              "agent"
          }
        ]
      };


    // --------------------------------------------------------
    // Tool started
    // --------------------------------------------------------

    case "tool_start":
      return {
        ...ex,

        steps: [
          ...ex.steps,

          {
            id: event.id,

            name:
              event.name,

            args:
              event.args || {},

            status:
              "running"
          }
        ]
      };


    // --------------------------------------------------------
    // Tool finished
    // --------------------------------------------------------

    case "tool_end":
      return {
        ...ex,

        steps:
          ex.steps.map(
            (s) =>
              s.id === event.id
                ? {
                    ...s,

                    status:
                      event.ok
                        ? "done"
                        : "error",

                    count:
                      event.papers
                        ? event.papers.length
                        : undefined,

                    error:
                      event.error ||
                      ""
                  }
                : s
          )
      };


    // --------------------------------------------------------
    // Answer
    //
    // Backend sends:
    // { text: "..." }
    //
    // Older code expected:
    // { content: "..." }
    //
    // Support both.
    // --------------------------------------------------------

    case "answer":
      return {
        ...ex,

        answer:
          event.text ||
          event.content ||
          "",

        status:
          "done",

        fresh:
          true
      };


    // --------------------------------------------------------
    // Error
    // --------------------------------------------------------

    case "error":
      return {
        ...ex,

        error:
          event.message ||
          "Something went wrong.",

        status:
          "error",

        steps:
          ex.steps.map(
            (s) =>
              s.status ===
              "running"
                ? {
                    ...s,
                    status:
                      "error"
                  }
                : s
          )
      };


    // --------------------------------------------------------
    // Done
    // --------------------------------------------------------

    case "done":
      return ex.status ===
        "working"
        ? {
            ...ex,

            status:
              "error",

            error:
              "The agent stopped without an answer. Ask again.",

            steps:
              ex.steps.map(
                (s) =>
                  s.status ===
                  "running"
                    ? {
                        ...s,
                        status:
                          "error"
                      }
                    : s
              )
          }
        : ex;


    // --------------------------------------------------------
    // Unknown event
    // --------------------------------------------------------

    default:
      return ex;
  }
}


// ============================================================
// Paper events
// ============================================================

function applyEventToPapers(
  papers,
  ex,
  event
) {

  // ----------------------------------------------------------
  // Backend sends a "paper" event when a paper is selected/read.
  // ----------------------------------------------------------

  if (
    event.type === "paper"
  ) {
    const paper = {
      title:
        event.title ||
        "Untitled Paper",

      pdf:
        event.pdf ||
        event.pdf_url ||
        "",

      year:
        event.year ||
        null,

      source:
        event.source ||
        "arxiv",

      state:
        event.state ||
        "seen"
    };

    return addPapers(
      papers,
      [paper]
    );
  }


  // ----------------------------------------------------------
  // Tool started reading a paper.
  // ----------------------------------------------------------

  if (
    event.type ===
      "tool_start" &&
    event.name ===
      "read_paper"
  ) {
    return updateByUrl(
      papers,

      event.args &&
        event.args.pdf_url,

      {
        state:
          "reading"
      }
    );
  }


  // ----------------------------------------------------------
  // Tool finished.
  // ----------------------------------------------------------

  if (
    event.type ===
    "tool_end"
  ) {
    let next = papers;

    if (
      event.papers &&
      event.papers.length
    ) {
      next =
        addPapers(
          next,
          event.papers
        );
    }

    if (
      event.name ===
      "read_paper"
    ) {
      const step =
        ex.steps.find(
          (s) =>
            s.id ===
            event.id
        );

      const asked =
        step &&
        step.args
          ? step.args.pdf_url
          : "";

      if (
        event.ok &&
        event.read
      ) {
        next =
          updateByUrl(
            next,

            event.read
              .pdf_url ||
              asked,

            {
              state:
                "read",

              pages:
                event.read.pages ||
                null
            }
          );
      } else if (asked) {
        next =
          updateByUrl(
            next,
            asked,
            {
              state:
                "seen"
            }
          );
      }
    }

    return next;
  }


  // ----------------------------------------------------------
  // When the run ends, nothing should remain "reading".
  // ----------------------------------------------------------

  if (
    event.type ===
      "error" ||
    event.type ===
      "done"
  ) {
    return papers.some(
      (p) =>
        p.state ===
        "reading"
    )
      ? papers.map(
          (p) =>
            p.state ===
            "reading"
              ? {
                  ...p,
                  state:
                    "seen"
                }
              : p
        )
      : papers;
  }


  return papers;
}


// ============================================================
// Session
// ============================================================

export function freshSession() {
  return {
    bootId: null,

    threadId:
      uid("t"),

    exchanges: [],

    papers: []
  };
}


export function initState() {
  let saved = null;

  try {
    saved =
      JSON.parse(
        localStorage.getItem(
          STORAGE_KEY
        ) || "null"
      );
  } catch {
    saved = null;
  }

  const base =
    freshSession();

  if (
    !saved ||
    !Array.isArray(
      saved.exchanges
    ) ||
    !Array.isArray(
      saved.papers
    )
  ) {
    return {
      ...base,
      busy: false
    };
  }


  // Anything still running when
  // the page closed did not finish.

  const exchanges =
    saved.exchanges.map(
      (ex) =>
        ex.status ===
        "working"
          ? {
              ...ex,

              status:
                "error",

              fresh:
                false,

              error:
                "This question was interrupted when the page closed.",

              steps:
                (
                  ex.steps ||
                  []
                ).map(
                  (s) =>
                    s.status ===
                    "running"
                      ? {
                          ...s,
                          status:
                            "error"
                        }
                      : s
                )
            }
          : {
              ...ex,
              fresh: false
            }
    );


  const papers =
    saved.papers.map(
      (p) =>
        p.state ===
        "reading"
          ? {
              ...p,
              state:
                "seen"
            }
          : p
    );


  return {
    bootId:
      saved.bootId ||
      null,

    threadId:
      saved.threadId ||
      base.threadId,

    exchanges,

    papers,

    busy: false
  };
}


export function saveSession(
  state
) {
  try {
    localStorage.setItem(
      STORAGE_KEY,

      JSON.stringify({
        bootId:
          state.bootId,

        threadId:
          state.threadId,

        exchanges:
          state.exchanges,

        papers:
          state.papers
      })
    );
  } catch {
    // Storage full or blocked.
    // The app still works without it.
  }
}


// ============================================================
// Reducer
// ============================================================

export function reducer(
  state,
  action
) {
  switch (action.type) {

    // --------------------------------------------------------
    // New question
    // --------------------------------------------------------

    case "ask":
      return {
        ...state,

        busy: true,

        exchanges: [
          ...state.exchanges,

          {
            id:
              action.id,

            question:
              action.question,

            steps: [],

            answer: "",

            error: "",

            status:
              "working",

            startedAt:
              Date.now(),

            fresh:
              false
          }
        ]
      };


    // --------------------------------------------------------
    // Incoming backend event
    // --------------------------------------------------------

    case "event": {
      const target =
        state.exchanges.find(
          (e) =>
            e.id ===
            action.exId
        );

      if (!target) {
        return state;
      }


      const exchanges =
        state.exchanges.map(
          (e) =>
            e.id ===
            action.exId
              ? applyEvent(
                  e,
                  action.event
                )
              : e
        );


      return {
        ...state,

        exchanges,

        papers:
          applyEventToPapers(
            state.papers,
            target,
            action.event
          ),

        busy:
          action.event.type ===
          "done"
            ? false
            : state.busy
      };
    }


    // --------------------------------------------------------
    // Remove exchange
    // --------------------------------------------------------

    case "drop":
      return {
        ...state,

        exchanges:
          state.exchanges.filter(
            (e) =>
              e.id !==
              action.id
          )
      };


    // --------------------------------------------------------
    // Server boot
    // --------------------------------------------------------

    case "boot":

      // The server restarted,
      // so its memory of this
      // conversation is gone.

      if (
        state.bootId &&
        state.bootId !==
          action.bootId
      ) {
        return {
          ...freshSession(),

          bootId:
            action.bootId,

          busy: false
        };
      }

      return state.bootId ===
        action.bootId
        ? state
        : {
            ...state,
            bootId:
              action.bootId
          };


    // --------------------------------------------------------
    // Reset everything
    // --------------------------------------------------------

    case "reset":
      return {
        ...freshSession(),

        bootId:
          state.bootId,

        busy: false
      };


    default:
      return state;
  }
}


// ============================================================
// Words
// ============================================================

const plural = (
  n,
  one,
  many
) =>
  `${n} ${
    n === 1
      ? one
      : many
  }`;


// ============================================================
// Step text
// ============================================================

export function stepText(
  step,
  papers
) {

  // ----------------------------------------------------------
  // NEW:
  // Explicit backend step events already contain human text.
  // Use that directly.
  // ----------------------------------------------------------

  if (
    step.text &&
    (
      step.name ===
        "agent" ||
      !step.name ||
      step.name ===
        "backend"
    )
  ) {
    return step.text;
  }


  const q =
    step.args &&
    step.args.query
      ? `“${step.args.query}”`
      : "";


  const running =
    step.status ===
    "running";

  const failed =
    step.status ===
    "error";


  const found =
    typeof step.count ===
    "number"
      ? `, found ${plural(
          step.count,
          "paper",
          "papers"
        )}`
      : "";


  // ----------------------------------------------------------
  // OpenAlex / arXiv tool events
  // ----------------------------------------------------------

  if (
    step.name ===
      "search_arxiv" ||
    step.name ===
      "search_open_alex"
  ) {

    const where =
      step.name ===
      "search_arxiv"
        ? "arXiv"
        : "OpenAlex";


    if (running) {
      return `Searching ${where} for ${q}`;
    }


    if (failed) {
      return `The ${where} search for ${q} failed`;
    }


    return `Searched ${where} for ${q}${found}`;
  }


  // ----------------------------------------------------------
  // Paper reading
  // ----------------------------------------------------------

  if (
    step.name ===
    "read_paper"
  ) {

    const key =
      urlKey(
        step.args &&
          step.args.pdf_url
      );


    const paper =
      papers.find(
        (p) =>
          p.pdf &&
          urlKey(p.pdf) ===
            key
      );


    const name =
      paper
        ? `“${paper.title}”`
        : "a paper";


    if (running) {
      return `Reading ${name}`;
    }


    if (failed) {
      return `Could not read ${name}`;
    }


    return `Read ${name}`;
  }


  return running
    ? `Running ${step.name}`
    : `Ran ${step.name}`;
}


// ============================================================
// Trail summary
// ============================================================

export function trailSummary(
  steps
) {

  const searches =
    steps.filter(
      (s) =>
        s.name !==
        "read_paper"
    ).length;


  const reads =
    steps.filter(
      (s) =>
        s.name ===
          "read_paper" &&
        s.status ===
          "done"
    ).length;


  if (
    !searches &&
    !reads
  ) {
    return "Answered from what was already on the shelf";
  }


  const parts = [];


  if (searches) {
    parts.push(
      `ran ${plural(
        searches,
        "search",
        "searches"
      )}`
    );
  }


  if (reads) {
    parts.push(
      `read ${plural(
        reads,
        "paper",
        "papers"
      )}`
    );
  }


  const text =
    parts.join(
      " and "
    );


  return (
    text.charAt(0)
      .toUpperCase() +
    text.slice(1)
  );
}