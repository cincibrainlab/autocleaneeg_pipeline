import { useState, useRef, type KeyboardEvent } from "react";
import { X } from "lucide-react";

interface TagInputProps {
  value: string[];
  onChange: (tags: string[]) => void;
  placeholder?: string;
  disabled?: boolean;
}

export default function TagInput({
  value,
  onChange,
  placeholder = "Type and press Enter",
  disabled = false,
}: TagInputProps) {
  const [input, setInput] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const addTag = (raw: string) => {
    const tag = raw.trim();
    if (tag && !value.includes(tag)) {
      onChange([...value, tag]);
    }
    setInput("");
  };

  const removeTag = (idx: number) => {
    onChange(value.filter((_, i) => i !== idx));
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" || e.key === ",") {
      e.preventDefault();
      addTag(input);
    } else if (e.key === "Backspace" && !input && value.length > 0) {
      removeTag(value.length - 1);
    }
  };

  return (
    <div
      onClick={() => inputRef.current?.focus()}
      className={[
        "flex flex-wrap gap-1.5 rounded-md border border-border bg-surface-100 px-2 py-1.5 min-h-[38px] cursor-text",
        disabled ? "opacity-60 cursor-not-allowed" : "focus-within:ring-1 focus-within:ring-brand/50",
      ].join(" ")}
    >
      {value.map((tag, i) => (
        <span
          key={i}
          className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-surface-50 text-xs font-mono text-zinc-300 border border-border-subtle"
        >
          {tag}
          {!disabled && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                removeTag(i);
              }}
              className="text-zinc-500 hover:text-zinc-300"
            >
              <X className="w-3 h-3" />
            </button>
          )}
        </span>
      ))}
      <input
        ref={inputRef}
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={handleKeyDown}
        onBlur={() => { if (input.trim()) addTag(input); }}
        placeholder={value.length === 0 ? placeholder : ""}
        disabled={disabled}
        className="flex-1 min-w-[120px] bg-transparent text-sm text-zinc-200 placeholder-zinc-600 outline-none py-0.5"
      />
    </div>
  );
}
