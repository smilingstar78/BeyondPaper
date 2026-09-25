import { useEffect, useRef } from "react";

export default function Composer({ value, onChange, onSend, busy, online }) {
  const ref = useRef(null);
  const disabled = busy || !online;

  // Grow with the text, up to about six lines.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 168) + "px";
  }, [value]);

  const submit = () => {
    if (!disabled && value.trim()) onSend(value);
  };

  const placeholder = !online
    ? "Waiting for the agent to come online"
    : busy
    ? "The agent is working on your last question"
    : "Ask about a field or a paper";

  return (
    <footer className="composer">
      <div className="composer__inner">
        <textarea
          ref={ref}
          rows={1}
          value={value}
          placeholder={placeholder}
          aria-label="Your question"
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              submit();
            }
          }}
        />
        <button type="button" className="btn btn--solid composer__send" onClick={submit} disabled={disabled || !value.trim()}>
          {busy ? "Researching" : "Ask"}
        </button>
      </div>
      <p className="composer__hint">Enter sends. Shift and Enter start a new line.</p>
    </footer>
  );
}
