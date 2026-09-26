const API_URL =
import.meta.env.VITE_API_BASE_URL ||
"https://beyond-paper-inky.vercel.app";

export async function fetchHealth(signal) {
const res = await fetch(
`${API_URL}/api/health`,
{
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

export async function streamChat({
message,
threadId,
onEvent,
signal
}) {
const res = await fetch(
`${API_URL}/api/chat`,
{
method: "POST",
  headers: {
    "Content-Type": "application/json"
  },

  body: JSON.stringify({
    message
  }),

  signal
}

);

if (!res.ok) {
let detail = "";

try {
  const body = await res.json();

  detail =
    typeof body.detail === "string"
      ? body.detail
      : typeof body.error === "string"
      ? body.error
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

while (true) {
const { value, done } = await reader.read();

if (done) {
  break;
}

buffer += decoder.decode(value, {
  stream: true
});

const events = buffer.split("\n\n");

buffer = events.pop() || "";

for (const eventBlock of events) {
  const lines = eventBlock.split("\n");

  let eventType = "message";
  let eventData = "";

  for (const line of lines) {
    if (line.startsWith("event:")) {
      eventType = line
        .slice(6)
        .trim();
    }

    if (line.startsWith("data:")) {
      eventData += line
        .slice(5)
        .trim();
    }
  }

  if (!eventData) {
    continue;
  }

  let parsedData = eventData;

  try {
    parsedData = JSON.parse(eventData);
  } catch {
    // Keep plain text if it isn't JSON.
  }

  onEvent({
    type: eventType,
    ...(typeof parsedData === "object" && parsedData !== null
      ? parsedData
      : { text: parsedData })
  });
}

}
}
