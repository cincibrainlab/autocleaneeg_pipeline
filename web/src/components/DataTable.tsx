import { Inbox } from "lucide-react";
import type { ReactNode } from "react";

export interface Column<T> {
  key: string;
  header: string;
  render?: (row: T) => ReactNode;
  className?: string;
}

interface DataTableProps<T> {
  columns: Column<T>[];
  data: T[];
  onRowClick?: (row: T) => void;
  rowClassName?: (row: T) => string;
  emptyMessage?: ReactNode;
  loading?: boolean;
}

function SkeletonRows({ cols }: { cols: number }) {
  return (
    <>
      {Array.from({ length: 5 }).map((_, rowIdx) => (
        <tr key={rowIdx} className="border-b border-border-subtle">
          {Array.from({ length: cols }).map((_, colIdx) => (
            <td key={colIdx} className="px-4 py-3">
              <div className="h-4 w-3/4 rounded bg-surface-50 animate-pulse" />
            </td>
          ))}
        </tr>
      ))}
    </>
  );
}

export default function DataTable<T extends Record<string, unknown>>({
  columns,
  data,
  onRowClick,
  rowClassName,
  emptyMessage = "No data available",
  loading = false,
}: DataTableProps<T>) {
  return (
    <div>
      <table className="w-full">
        <thead>
          <tr className="bg-surface-100 border-b border-border">
            {columns.map((col) => (
              <th
                key={col.key}
                className={`px-4 py-2.5 text-left text-xs uppercase text-zinc-500 font-medium tracking-wider ${col.className || ""}`}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {loading ? (
            <SkeletonRows cols={columns.length} />
          ) : data.length === 0 ? (
            <tr>
              <td
                colSpan={columns.length}
                className="px-4 py-12 text-center"
              >
                <div className="flex flex-col items-center gap-2 text-zinc-500">
                  <Inbox className="w-8 h-8" />
                  <p className="text-sm">{emptyMessage}</p>
                </div>
              </td>
            </tr>
          ) : (
            data.map((row, idx) => (
              <tr
                key={idx}
                onClick={onRowClick ? () => onRowClick(row) : undefined}
                className={[
                  "border-b border-border-subtle transition-colors duration-150",
                  onRowClick
                    ? "cursor-pointer hover:bg-surface-50/30"
                    : "hover:bg-surface-50/30",
                  rowClassName ? rowClassName(row) : "",
                ].join(" ")}
              >
                {columns.map((col) => (
                  <td
                    key={col.key}
                    className={`px-4 py-3 text-sm text-zinc-300 ${col.className || ""}`}
                  >
                    {col.render
                      ? col.render(row)
                      : String(row[col.key] ?? "")}
                  </td>
                ))}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}
