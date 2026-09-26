const API_URL =
  import.meta.env.VITE_API_BASE_URL ||
  "https://beyond-paper-inky.vercel.app";


export async function fetchHealth(signal) {

  const res = await fetch(
    `${API_URL}/`,
    {
      method: "GET",
      signal
    }
  );


  if (!res.ok) {

    throw new Error(
      `Health check failed with status ${res.status}`
    );

  }


  return res.json();
}


// Extracts a usable error message from a non-OK JSON response,
// checking both `detail` (FastAPI's HTTPException shape) and
// `error` (legacy shape) so either produces a real message.
async function extractErrorMessage(res) {

  try {

    const body = await res.json();

    if (typeof body.detail === "string") {
      return body.detail;
    }

    if (typeof body.error === "string") {
      return body.error;
    }

  } catch {
    // Ignore invalid/non-JSON error response.
  }

  return "";
}


export async function streamChat({
  message,
  onEvent,
  signal
}) {

  const res = await fetch(
    `${API_URL}/research`,
    {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
      },

      body: JSON.stringify({
        topic: message
      }),

      signal
    }
  );


  if (!res.ok) {

    const detail = await extractErrorMessage(res);

    throw new Error(
      detail ||
      `The server answered with status ${res.status}.`
    );

  }

  if (!res.body) {

    // Some environments (very old browsers, certain proxies)
    // don't support streaming response bodies.
    throw new Error(
      "Streaming responses are not supported in this environment."
    );

  }


  // -------------------------------------------------------
  // PARSE THE SSE STREAM
  // -------------------------------------------------------
  //
  // The backend sends one "data: <json>\n\n" block per event.
  // We buffer raw bytes, split on the blank-line event
  // separator, and JSON.parse each event's payload as it
  // arrives - this is what lets onEvent fire progressively
  // instead of only once at the very end.

  const reader = res.body.getReader();
  const decoder = new TextDecoder();

  let buffer = "";
  let finalData = null;

  while (true) {

    const { value, done } = await reader.read();

    if (done) {
      break;
    }

    buffer += decoder.decode(
      value,
      { stream: true }
    );

    let separatorIndex;

    while (
      (separatorIndex = buffer.indexOf("\n\n")) !== -1
    ) {

      const rawEvent = buffer.slice(0, separatorIndex);
      buffer = buffer.slice(separatorIndex + 2);

      const dataLine = rawEvent
        .split("\n")
        .find((line) => line.startsWith("data: "));

      if (!dataLine) {
        continue;
      }

      let parsed;

      try {
        parsed = JSON.parse(dataLine.slice(6));
      } catch {
        continue;
      }

      switch (parsed.type) {

        case "progress":
          onEvent({
            type: "progress",
            node: parsed.node,
            message: parsed.message
          });
          break;

        case "answer":
          finalData = parsed;
          onEvent({
            type: "answer",
            text: JSON.stringify(
              parsed.assessments || [],
              null,
              2
            )
          });
          break;

        case "error":
          onEvent({
            type: "error",
            message: parsed.message
          });
          throw new Error(parsed.message);

        case "done":
          onEvent({
            type: "done",
            status: parsed.status
          });
          break;

        default:
          break;
      }
    }
  }

  return finalData || { assessments: [] };
}
