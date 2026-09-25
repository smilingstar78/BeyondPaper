import { useEffect, useState } from "react";

import Answer from "./Answer.jsx";

import {
  stepText,
  trailSummary
} from "../state.js";


function useElapsed(since, active) {
  const [now, setNow] = useState(
    () => Date.now()
  );

  useEffect(() => {
    if (!active) {
      return undefined;
    }

    const timer = setInterval(
      () => setNow(Date.now()),
      1000
    );

    return () => clearInterval(timer);
  }, [active]);

  return Math.max(
    0,
    Math.floor(
      (now - since) / 1000
    )
  );
}


function Ribbon({ swaying }) {
  return (
    <svg
      className={`ribbon${
        swaying ? " is-swaying" : ""
      }`}
      viewBox="0 0 12 24"
      aria-hidden="true"
    >
      <path d="M0 0h12v24l-6-6-6 6z" />
    </svg>
  );
}


function StepList({
  steps,
  papers
}) {
  return (
    <ul className="steps">
      {steps.map((s) => (
        <li
          key={s.id}
          className={`step is-${s.status}`}
        >
          <span
            className="step__mark"
            aria-hidden="true"
          />

          <span>
            {stepText(
              s,
              papers
            )}

            {s.status === "error" &&
            s.error ? (
              <span className="step__error">
                {s.error}
              </span>
            ) : null}
          </span>
        </li>
      ))}
    </ul>
  );
}


function Working({
  ex,
  papers
}) {
  const seconds = useElapsed(
    ex.startedAt,
    true
  );

  /*
   * The backend sends real step events such as:
   *
   * Searching literature...
   * Found papers on OpenAlex. Searching arXiv...
   * Found papers on arXiv...
   * Reading: Paper Title
   * Extracting research gaps...
   *
   * The latest event is displayed in the header.
   */

  const latestStep =
    ex.steps &&
    ex.steps.length > 0
      ? ex.steps[
          ex.steps.length - 1
        ].text
      : "Starting research agent...";

  return (
    <div
      className="working"
      role="status"
    >
      <div className="working__head">
        <Ribbon swaying />

        <span>
          {latestStep}
        </span>

        <span className="working__time">
          {seconds}s
        </span>
      </div>

      {ex.steps &&
      ex.steps.length ? (
        <StepList
          steps={ex.steps}
          papers={papers}
        />
      ) : null}
    </div>
  );
}


export default function Exchange({
  ex,
  papers,
  isLast,
  busy,
  onHot,
  onOpenPaper,
  onRetry,
  onAsk,
  followUps
}) {
  const working =
    ex.status === "working";

  return (
    <article className="exchange">

      <h2 className="ask">
        {ex.question}
      </h2>

      {working ? (
        <Working
          ex={ex}
          papers={papers}
        />
      ) : null}


      {!working &&
      !ex.error &&
      ex.steps &&
      ex.steps.length ? (
        <details className="trail">
          <summary>
            {trailSummary(
              ex.steps
            )}
          </summary>

          <StepList
            steps={ex.steps}
            papers={papers}
          />
        </details>
      ) : null}


      {!working &&
      !ex.error &&
      (!ex.steps ||
        !ex.steps.length) &&
      ex.answer ? (
        <p className="trail trail--plain">
          {trailSummary(
            ex.steps || []
          )}
        </p>
      ) : null}


      {ex.answer ? (
        <section
          id={`answer-${ex.id}`}
          className="answer"
          data-fresh={
            ex.fresh
              ? "true"
              : "false"
          }
        >
          <Answer
            text={ex.answer}
            papers={papers}
            onHot={onHot}
            onOpenPaper={onOpenPaper}
          />
        </section>
      ) : null}


      {ex.error ? (
        <div
          className="notice"
          role="alert"
        >
          <p>
            {ex.error}
          </p>

          <button
            type="button"
            className="btn"
            onClick={() =>
              onRetry(ex)
            }
            disabled={busy}
          >
            Try again
          </button>
        </div>
      ) : null}


      {isLast &&
      ex.status === "done" &&
      !busy ? (
        <div className="followups">
          {followUps.map(
            (text) => (
              <button
                type="button"
                key={text}
                onClick={() =>
                  onAsk(text)
                }
              >
                {text}
              </button>
            )
          )}
        </div>
      ) : null}

    </article>
  );
}