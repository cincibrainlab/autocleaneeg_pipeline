import { CheckCircle2, GitBranch, Zap, BookOpen } from "lucide-react";
import { useTutorial } from "../../contexts/TutorialContext";

const NEXT_STEPS = [
  {
    icon: GitBranch,
    title: "Create another route",
    description: "Map additional folders and tasks to scale up processing.",
    href: "/routes",
  },
  {
    icon: Zap,
    title: "Try Live mode",
    description: "Promote your route to live once you've validated it.",
    href: "/routes",
  },
  {
    icon: BookOpen,
    title: "Read the docs",
    description: "Explore advanced configuration and pipeline options.",
    href: "https://docs.autocleaneeg.org/welcome",
    external: true,
  },
];

export default function TutorialComplete() {
  const { completeTutorial } = useTutorial();

  return (
    <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div
        className="w-full max-w-md mx-4 rounded-xl border border-border bg-surface-200 p-8 shadow-2xl"
        style={{ animation: "tutorial-enter 0.25s ease-out both" }}
      >
        {/* Checkmark */}
        <div className="flex justify-center mb-6">
          <div
            className="w-16 h-16 rounded-full bg-brand/15 border border-brand/30 flex items-center justify-center"
            style={{ animation: "tutorial-scale-in 0.35s cubic-bezier(0.34,1.56,0.64,1) both" }}
          >
            <CheckCircle2 className="w-8 h-8 text-brand" />
          </div>
        </div>

        {/* Heading */}
        <h2 className="text-xl font-bold text-zinc-100 text-center mb-1">
          You're all set!
        </h2>
        <p className="text-sm text-zinc-400 text-center mb-8">
          1 route created, 1 file processed.
        </p>

        {/* Next-step cards */}
        <div className="space-y-2 mb-8">
          {NEXT_STEPS.map(({ icon: Icon, title, description, href, external }) => (
            <a
              key={title}
              href={href}
              {...(external ? { target: "_blank", rel: "noopener noreferrer" } : {})}
              className="flex items-start gap-3 rounded-lg border border-border bg-surface-100 px-4 py-3 hover:bg-surface-50/40 transition-colors group"
            >
              <div className="mt-0.5 w-7 h-7 rounded-md bg-brand/10 flex items-center justify-center flex-shrink-0 group-hover:bg-brand/20 transition-colors">
                <Icon className="w-3.5 h-3.5 text-brand" />
              </div>
              <div>
                <p className="text-sm font-medium text-zinc-200">{title}</p>
                <p className="text-xs text-zinc-500 mt-0.5">{description}</p>
              </div>
            </a>
          ))}
        </div>

        <button
          onClick={completeTutorial}
          className="w-full rounded-lg px-4 py-2.5 text-sm font-semibold bg-brand text-brand-900 hover:bg-brand-500 transition-colors"
        >
          Finish Tutorial
        </button>
      </div>
    </div>
  );
}
