import {
  useCallback,
  useEffect,
  useReducer,
  useRef,
  useState
} from "react";

import {
  fetchHealth,
  streamChat
} from "./api.js";

import {
  initState,
  reducer,
  saveSession,
  uid
} from "./state.js";

import Shelf from "./components/Shelf.jsx";
import Exchange from "./components/Exchange.jsx";
import Composer from "./components/Composer.jsx";
import Empty from "./components/Empty.jsx";


const FOLLOW_UPS = [
  "Where are the biggest research gaps?",
  "Suggest research topics I could pursue.",
  "Which paper should I read first?"
];


function explain(error) {
  const message =
    error && error.message
      ? error.message
      : "";

  if (
    /failed to fetch|networkerror|load failed/i.test(
      message
    )
  ) {
    return "The app lost its connection to the agent. Check that python server.py is still running, then try again.";
  }

  return (
    message ||
    "Something went wrong while talking to the agent."
  );
}


export default function App() {
  const [
    state,
    dispatch
  ] = useReducer(
    reducer,
    undefined,
    initState
  );

  const [health, setHealth] =
    useState({
      status: "checking"
    });

  const [folded, setFolded] =
    useState(
      () =>
        typeof window !==
          "undefined" &&
        window.innerHeight < 680
    );

  const [hotId, setHotId] =
    useState(null);

  const [openId, setOpenId] =
    useState(null);

  const [draft, setDraft] =
    useState("");

  const deskRef = useRef(null);

  const stick = useRef(true);

  const busyRef =
    useRef(false);

  const threadRef =
    useRef(state.threadId);

  threadRef.current =
    state.threadId;

  const online =
    health.status === "online";


  // Keep the session in the browser.
  useEffect(() => {
    saveSession(state);
  }, [state]);


  // Check whether the Python backend is running.
  useEffect(() => {
    let cancelled = false;
    let timer;

    const check = async () => {
      try {
        const info =
          await fetchHealth();

        if (cancelled) {
          return;
        }

        setHealth({
          status: "online",
          info
        });

        dispatch({
          type: "boot",
          bootId: info.boot_id
        });

      } catch {
        if (cancelled) {
          return;
        }

        setHealth({
          status: "offline"
        });

        timer = setTimeout(
          check,
          4000
        );
      }
    };

    check();

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, []);


  // Send a question to the research agent.
  const send = useCallback(
    async (text) => {
      const question =
        String(text || "").trim();

      if (
        !question ||
        busyRef.current
      ) {
        return;
      }

      busyRef.current = true;

      stick.current = true;

      setDraft("");

      setOpenId(null);

      const exId = uid("x");

      dispatch({
        type: "ask",
        id: exId,
        question
      });

      try {
        await streamChat({
          message: question,
          threadId:
            threadRef.current,

          /*
           * Every SSE event from Python
           * comes through here.
           *
           * For example:
           *
           * {
           *   type: "step",
           *   text: "Found 5 papers on OpenAlex..."
           * }
           */
          onEvent: (event) =>
            dispatch({
              type: "event",
              exId,
              event
            })
        });

      } catch (error) {
        dispatch({
          type: "event",
          exId,
          event: {
            type: "error",
            message:
              explain(error)
          }
        });

      } finally {
        dispatch({
          type: "event",
          exId,
          event: {
            type: "done"
          }
        });

        busyRef.current = false;
      }
    },
    []
  );


  const retry = useCallback(
    (ex) => {
      dispatch({
        type: "drop",
        id: ex.id
      });

      send(ex.question);
    },
    [send]
  );


  const askAbout = useCallback(
    (paper) => {
      send(
        `Tell me about “${paper.title}”: what it does, what it found, and what it leaves open.`
      );
    },
    [send]
  );


  const newTopic = () => {
    if (
      window.confirm(
        "Start a new topic? This clears the conversation and the shelf."
      )
    ) {
      setOpenId(null);

      setHotId(null);

      dispatch({
        type: "reset"
      });
    }
  };


  const openFromText =
    useCallback(
      (id) => {
        if (folded) {
          setFolded(false);

          setTimeout(
            () => setOpenId(id),
            420
          );
        } else {
          setOpenId(id);
        }
      },
      [folded]
    );


  // Current exchange.
  const last =
    state.exchanges[
      state.exchanges.length - 1
    ];

  const lastId =
    last
      ? last.id
      : null;

  const lastStatus =
    last
      ? last.status
      : null;

  const lastFresh =
    last
      ? last.fresh
      : false;

  const stepCount =
    last
      ? last.steps.length
      : 0;


  // Follow the agent while it is working.
  useEffect(() => {
    const desk =
      deskRef.current;

    if (
      desk &&
      lastStatus === "working" &&
      stick.current
    ) {
      desk.scrollTo({
        top: desk.scrollHeight,
        behavior: "smooth"
      });
    }
  }, [
    lastId,
    lastStatus,
    stepCount
  ]);


  // Scroll to the answer when finished.
  useEffect(() => {
    if (
      !state.busy &&
      lastStatus === "done" &&
      lastFresh &&
      lastId
    ) {
      const el =
        document.getElementById(
          `answer-${lastId}`
        );

      if (el) {
        el.scrollIntoView({
          behavior: "smooth",
          block: "start"
        });
      }
    }
  }, [
    state.busy,
    lastId,
    lastStatus,
    lastFresh
  ]);


  const onDeskScroll = () => {
    const desk =
      deskRef.current;

    if (desk) {
      stick.current =
        desk.scrollHeight -
          desk.scrollTop -
          desk.clientHeight <
        140;
    }
  };


  const showEmpty =
    state.exchanges.length === 0;


  return (
    <div className="app">

      <Shelf
        papers={state.papers}
        busy={state.busy}
        folded={folded}

        onToggleFold={() => {
          setOpenId(null);

          setFolded(
            (f) => !f
          );
        }}

        online={online}

        onNewTopic={newTopic}

        canReset={
          !state.busy &&
          (
            state.exchanges.length >
              0 ||
            state.papers.length >
              0
          )
        }

        hotId={hotId}
        openId={openId}

        onOpen={setOpenId}

        onAsk={askAbout}

        canAsk={
          online &&
          !state.busy
        }
      />


      <main
        className="desk"
        ref={deskRef}
        onScroll={onDeskScroll}
      >

        <div className="desk__inner">

          {health.status ===
          "offline" ? (
            <div
              className="notice notice--wide"
              role="alert"
            >
              <p>
                The agent is not
                reachable. Open a
                terminal in the project
                folder and run{" "}
                <code>
                  python server.py
                </code>
                . This page connects
                on its own once the
                server is up.
              </p>
            </div>
          ) : null}

          {showEmpty ? (
            <Empty
              onPick={send}
              disabled={
                !online ||
                state.busy
              }
            />
          ) : (
            state.exchanges.map(
              (ex, i) => (
                <Exchange
                  key={ex.id}
                  ex={ex}
                  papers={
                    state.papers
                  }
                  isLast={
                    i ===
                    state.exchanges.length -
                      1
                  }
                  busy={
                    state.busy
                  }
                  onHot={
                    setHotId
                  }
                  onOpenPaper={
                    openFromText
                  }
                  onRetry={
                    retry
                  }
                  onAsk={
                    send
                  }
                  followUps={
                    FOLLOW_UPS
                  }
                />
              )
            )
          )}

        </div>
      </main>


      <Composer
        value={draft}
        onChange={setDraft}
        onSend={send}
        busy={state.busy}
        online={online}
      />

    </div>
  );
}
