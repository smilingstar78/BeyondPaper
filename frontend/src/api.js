// Talks to the Python server (server.py).

export async function fetchHealth(signal) {
  const res = await fetch("/api/health", { signal });

  if (!res.ok) {
    throw new Error(
      `Health check failed with status ${res.status}`
    );
  }

  return res.json();
}

/**
 * Sends one message and receives Server-Sent Events
 * from the Python backend.
 *
 * Backend events look like:
 *
 * event: step
 * data: {"id":"step-openalex","text":"Found 5 papers on OpenAlex..."}
 *
 * event: paper
 * data: {"id":"p-1","title":"..."}
 *
 * event: answer
 * data: {"text":"..."}
 *
 * event: done
 * data: {"status":"completed"}
 */

export async function streamChat({
  message,
  threadId,
  onEvent,
  signal
}) {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      message,
      thread_id: threadId
    }),
    signal
  });

  if (!res.ok || !res.body) {
    let detail = "";

    try {
      const body = await res.json();

      detail =
        typeof body.detail === "string"
          ? body.detail
          : "";
    } catch {
      // Ignore invalid error response.
    }

    throw new Error(
      detail ||
        `The server answered with status ${res.status}.`
    );
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();

  let buffer = "";

  let eventType = null;
  let eventData = "";

  function emitEvent() {
    if (!eventData) {
      eventType = null;
      return;
    }

    try {
      const parsed = JSON.parse(eventData);

      onEvent({
        type: eventType || parsed.type || "message",
        ...parsed
      });
    } catch (error) {
      console.warn(
        "Could not parse SSE data:",
        eventData,
        error
      );
    }

    eventType = null;
    eventData = "";
  }

  for (;;) {
    const { value, done } = await reader.read();

    if (done) {
      break;
    }

    buffer += decoder.decode(value, {
      stream: true
    });

    let newline;

    while ((newline = buffer.indexOf("\n")) >= 0) {
      const rawLine = buffer.slice(
        0,
        newline
      );

      buffer = buffer.slice(
        newline + 1
      );

      const line = rawLine.trim();

      // Empty line means the current SSE event is complete.
      if (!line) {
        emitEvent();
        continue;
      }

      // Read event type.
      if (line.startsWith("event:")) {
        eventType = line
          .slice(6)
          .trim();

        continue;
      }

      // Read event data.
      if (line.startsWith("data:")) {
        const data = line
          .slice(5)
          .trim();

        eventData += data;

        continue;
      }
    }
  }

  // Flush any remaining decoder content.
  buffer += decoder.decode();

  if (buffer.trim()) {
    const lines = buffer.split("\n");

    for (const rawLine of lines) {
      const line = rawLine.trim();

      if (!line) {
        emitEvent();
        continue;
      }

      if (line.startsWith("event:")) {
        eventType = line
          .slice(6)
          .trim();
      } else if (line.startsWith("data:")) {
        const data = line
          .slice(5)
          .trim();

        eventData += data;
      }
    }
  }

  // Emit the final event if necessary.
  emitEvent();
}