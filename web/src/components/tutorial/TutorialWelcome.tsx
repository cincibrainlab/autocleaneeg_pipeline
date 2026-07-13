import { Brain } from "lucide-react";
import { useTutorial } from "../../contexts/TutorialContext";

const FLOW_STEPS = [
  { label: "Dashboard", color: "bg-zinc-600" },
  { label: "Routes", color: "bg-brand/60" },
  { label: "Settings", color: "bg-brand/70" },
  { label: "Service", color: "bg-brand/80" },
  { label: "Queue", color: "bg-brand" },
];

export default function TutorialWelcome() {
  const { startTutorial, skipTutorial, isActive, nextStep } = useTutorial();

  const handleStart = async () => {
    if (isActive) {
      // Already active (re-opened welcome), just advance
      nextStep();
    } else {
      await startTutorial();
      // startTutorial sets step 0; we need to advance to step 1
      nextStep();
    }
  };

  return (
    <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div
        className="w-full max-w-md mx-4 rounded-xl border border-border bg-surface-200 p-8 shadow-2xl"
        style={{ animation: "tutorial-enter 0.25s ease-out both" }}
      >
        {/* Icon */}
        <div className="flex justify-center mb-6">
          <div className="w-16 h-16 rounded-full bg-brand/15 border border-brand/30 flex items-center justify-center">
            <Brain className="w-8 h-8 text-brand" />
          </div>
        </div>

        {/* Heading */}
        <h2 className="text-xl font-bold text-zinc-100 text-center mb-2">
          Welcome to AutoCleanEEG Serve
        </h2>
        <p className="text-sm text-zinc-400 text-center mb-8">
          Let's set up your first processing route in about 2 minutes.
        </p>

        {/* Flow diagram */}
        <div className="flex items-center justify-center gap-0 mb-8">
          {FLOW_STEPS.map((step, i) => (
            <div key={step.label} className="flex items-center">
              <div className="flex flex-col items-center gap-1.5">
                <div
                  className={`w-3 h-3 rounded-full ${step.color} ring-2 ring-offset-2 ring-offset-surface-200 ring-transparent`}
                />
                <span className="text-[10px] text-zinc-500 whitespace-nowrap">
                  {step.label}
                </span>
              </div>
              {i < FLOW_STEPS.length - 1 && (
                <div className="w-8 h-px bg-border mx-1 mb-4" />
              )}
            </div>
          ))}
        </div>

        {/* Actions */}
        <div className="flex flex-col gap-3">
          <button
            onClick={handleStart}
            className="w-full rounded-lg px-4 py-2.5 text-sm font-semibold bg-brand text-brand-900 hover:bg-brand-500 transition-colors"
          >
            Start Tutorial
          </button>
          <button
            onClick={skipTutorial}
            className="w-full text-center text-sm text-zinc-500 hover:text-zinc-300 transition-colors py-1"
          >
            Skip for now
          </button>
        </div>
      </div>
    </div>
  );
}
