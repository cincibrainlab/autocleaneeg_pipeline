import type { ReactNode } from "react";

interface CodeViewerProps {
  lines: string[];
  colorize?: (line: string) => ReactNode;
  maxHeight?: string;
}

/**
 * Reusable code viewer with line numbers, dark background, and optional syntax highlighting.
 * Used by Settings (YAML) and Tasks (Python source).
 */
export default function CodeViewer({
  lines,
  colorize,
  maxHeight = "600px",
}: CodeViewerProps) {
  return (
    <div className="overflow-x-auto overflow-y-auto bg-[#0A0A0A]" style={{ maxHeight }}>
      <table className="w-full border-collapse">
        <tbody>
          {lines.map((line, i) => (
            <tr key={i} className="hover:bg-white/[0.02] transition-colors">
              <td className="px-3 py-0 text-right text-[11px] font-mono text-zinc-700 select-none w-10 align-top leading-relaxed">
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
