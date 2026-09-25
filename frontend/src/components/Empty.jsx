const STARTERS = [
  "What is the state of research on agentic AI, and where does it fall short?",
  "Find open problems in retrieval-augmented generation.",
  "Suggest thesis topics where federated learning meets healthcare.",
];

export default function Empty({ onPick, disabled }) {
  return (
    <section className="empty">
      <h2 className="empty__title">What should we read up on?</h2>
      <p className="empty__lede">
        Name a field or ask a question. The agent searches arXiv and OpenAlex, reads the strongest papers, and points
        out what nobody has answered yet.
      </p>

      <ul className="starters">
        {STARTERS.map((text) => (
          <li key={text}>
            <button type="button" onClick={() => onPick(text)} disabled={disabled}>
              {text}
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
