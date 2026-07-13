import type { ReactNode } from "react";

interface CodeViewerProps {
  lines: string[];
  colorize?: (line: string) => ReactNode;
  maxHeight?: string;
}

/**
 * Reusable code viewer with line numbers, theme-aware surfaces, and optional syntax highlighting.
 * Used by Settings (YAML) and Tasks (Python source).
 */
export default function CodeViewer({
  lines,
  colorize,
  maxHeight = "600px",
}: CodeViewerProps) {
  return (
    <div className="overflow-x-auto overflow-y-auto bg-surface-500" style={{ maxHeight }}>
      <table className="w-full border-collapse">
        <tbody>
          {lines.map((line, i) => (
            <tr key={i} className="hover:bg-surface-50/30 transition-colors">
              <td className="px-3 py-0 text-right text-[11px] font-mono text-zinc-400 select-none w-10 align-top leading-relaxed">
                {i + 1}
              </td>
              <td className="px-3 py-0 text-xs font-mono whitespace-pre leading-relaxed text-zinc-300">
                {colorize ? colorize(line) : line}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
