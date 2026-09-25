const API_URL =
  "https://beyond-paper-ac0zabblg-smilingstar78s-projects.vercel.app";


export async function fetchHealth(signal) {

  const res = await fetch(
    `${API_URL}/`,
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
    `${API_URL}/research`,
    {
      method: "POST",

      headers: {
        "Content-Type": "application/json"
      },

      body: JSON.stringify({
        topic: message
      }),

      signal
    }
  );


  if (!res.ok) {

    let detail = "";

    try {

      const body =
        await res.json();

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


  const data =
    await res.json();


  onEvent({
    type: "answer",
    text: JSON.stringify(
      data.assessments,
      null,
      2
    )
  });


  onEvent({
    type: "done",
    status: "completed"
  });


  return data;
}
