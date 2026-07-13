import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useNavigate } from "react-router-dom";
import { X } from "lucide-react";
import { useTutorial } from "../../contexts/TutorialContext";
import { TUTORIAL_STEPS, PROGRESS_STEPS } from "./tutorialSteps";
import TutorialWelcome from "./TutorialWelcome";
import TutorialComplete from "./TutorialComplete";

// ── Rect helpers ──────────────────────────────────────────────────────

interface Rect {
  top: number;
  left: number;
  width: number;
  height: number;
}

const PADDING = 10;
const CARD_WIDTH = 320;
const CARD_HEIGHT_ESTIMATE = 180;

function getTargetRect(el: HTMLElement): Rect {
  const r = el.getBoundingClientRect();
  return {
    top: r.top - PADDING,
    left: r.left - PADDING,
    width: r.width + PADDING * 2,
    height: r.height + PADDING * 2,
  };
}

function rectsEqual(a: Rect | null, b: Rect | null): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  return a.top === b.top && a.left === b.left && a.width === b.width && a.height === b.height;
}

type CardSide = "below" | "above" | "right" | "left";

function computeCardPosition(
  rect: Rect,
  vw: number,
  vh: number
): { side: CardSide; top: number; left: number } {
  const spaceBelow = vh - rect.top - rect.height;
  const spaceAbove = rect.top;
  const spaceRight = vw - rect.left - rect.width;

  const CARD_H = CARD_HEIGHT_ESTIMATE;
  const CARD_W = CARD_WIDTH;
  const GAP = 14;

  if (spaceBelow >= CARD_H + GAP) {
    return {
      side: "below",
      top: rect.top + rect.height + GAP,
      left: Math.min(Math.max(rect.left, GAP), vw - CARD_W - GAP),
    };
  }
  if (spaceAbove >= CARD_H + GAP) {
    return {
      side: "above",
      top: Math.max(rect.top - CARD_H - GAP, GAP),
      left: Math.min(Math.max(rect.left, GAP), vw - CARD_W - GAP),
    };
  }
  if (spaceRight >= CARD_W + GAP) {
    return {
      side: "right",
      top: Math.min(Math.max(rect.top, GAP), vh - CARD_H - GAP),
      left: rect.left + rect.width + GAP,
    };
  }
  return {
    side: "left",
    top: Math.min(Math.max(rect.top, GAP), vh - CARD_H - GAP),
    left: Math.max(rect.left - CARD_W - GAP, GAP),
  };
}

// ── SVG Spotlight ─────────────────────────────────────────────────────

function SpotlightSvg({ rect }: { rect: Rect }) {
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const rx = 10;

  const cutoutPath = [
    `M ${rect.left + rx} ${rect.top}`,
    `H ${rect.left + rect.width - rx}`,
    `Q ${rect.left + rect.width} ${rect.top} ${rect.left + rect.width} ${rect.top + rx}`,
    `V ${rect.top + rect.height - rx}`,
    `Q ${rect.left + rect.width} ${rect.top + rect.height} ${rect.left + rect.width - rx} ${rect.top + rect.height}`,
    `H ${rect.left + rx}`,
    `Q ${rect.left} ${rect.top + rect.height} ${rect.left} ${rect.top + rect.height - rx}`,
    `V ${rect.top + rx}`,
    `Q ${rect.left} ${rect.top} ${rect.left + rx} ${rect.top}`,
    "Z",
  ].join(" ");

  return (
    <svg
      className="fixed inset-0 pointer-events-none"
      style={{ zIndex: 61, width: "100vw", height: "100vh" }}
      viewBox={`0 0 ${vw} ${vh}`}
      preserveAspectRatio="none"
    >
      <defs>
        <mask id="spotlight-mask">
          <rect width={vw} height={vh} fill="white" />
          <path d={cutoutPath} fill="black" />
        </mask>
      </defs>
      <rect
        width={vw}
        height={vh}
        fill="rgba(0,0,0,0.65)"
        mask="url(#spotlight-mask)"
      />
      <path
        d={cutoutPath}
        fill="none"
        stroke="rgba(62,207,142,0.35)"
        strokeWidth="2"
      />
    </svg>
  );
}

// ── Step Card ─────────────────────────────────────────────────────────

interface StepCardProps {
  step: (typeof TUTORIAL_STEPS)[number];
  cardPos: { top: number; left: number };
  onNext: () => void;
  onSkip: () => void;
  currentStep: number;
}

function StepCard({ step, cardPos, onNext, onSkip, currentStep }: StepCardProps) {
  const progressIndex = PROGRESS_STEPS.findIndex((s) => s.id === step.id);

  return (
    <div
      className="fixed z-[65] rounded-lg border border-brand/40 bg-surface-200 shadow-xl"
      style={{
        top: cardPos.top,
        left: cardPos.left,
        width: CARD_WIDTH,
        animation: "tutorial-enter 0.2s ease-out both",
        pointerEvents: "all",
      }}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 pt-3 pb-1">
        <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
          Step {currentStep} of {PROGRESS_STEPS.length}
        </span>
        <button
          onClick={onSkip}
          className="text-zinc-600 hover:text-zinc-300 transition-colors"
          title="Skip tutorial"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Content */}
      <div className="px-4 pb-3">
        <h3 className="text-sm font-semibold text-zinc-100 mb-1.5">{step.title}</h3>
        <p className="text-xs text-zinc-400 leading-relaxed">{step.description}</p>
      </div>

      {/* Footer */}
      <div className="px-4 pb-4 flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          {PROGRESS_STEPS.map((s, i) => (
            <div
              key={s.id}
              className={[
                "rounded-full transition-all duration-200",
                i === progressIndex
                  ? "w-4 h-1.5 bg-brand"
                  : i < progressIndex
                  ? "w-1.5 h-1.5 bg-brand/50"
                  : "w-1.5 h-1.5 bg-zinc-700",
              ].join(" ")}
            />
          ))}
        </div>

        {step.action === "next" && (
          <button
            onClick={onNext}
            className="rounded-md px-3 py-1.5 text-xs font-semibold bg-brand text-brand-900 hover:bg-brand-500 transition-colors"
          >
            Next
          </button>
        )}
        {step.action === "waiting" && (
          <span className="text-[11px] text-zinc-500 italic">
            Waiting for action...
          </span>
        )}
      </div>
    </div>
  );
}

// ── Main Overlay ──────────────────────────────────────────────────────

export default function TutorialOverlay() {
  const { isActive, currentStep, getTarget, nextStep, skipTutorial } =
    useTutorial();
  const navigate = useNavigate();

  const step = TUTORIAL_STEPS[currentStep];
  const [cardPos, setCardPos] = useState({ top: 60, left: 60 });
  const [targetRect, setTargetRect] = useState<Rect | null>(null);
  const rafRef = useRef<number | null>(null);
  const lastRectRef = useRef<Rect | null>(null);

  // Navigate to the correct page when step changes
  useEffect(() => {
    if (!isActive || !step?.route) return;
    const delay = setTimeout(() => navigate(step.route!), 100);
    return () => clearTimeout(delay);
  }, [isActive, currentStep, step, navigate]);

  // Position tracking via RAF — only sets state when rect actually changes
  const updatePosition = useCallback(() => {
    if (!step?.targetId) {
      if (lastRectRef.current !== null) {
        lastRectRef.current = null;
        setTargetRect(null);
      }
      return;
    }
    const el = getTarget(step.targetId);
    if (!el) {
      if (lastRectRef.current !== null) {
        lastRectRef.current = null;
        setTargetRect(null);
      }
      return;
    }
    const rect = getTargetRect(el);
    if (!rectsEqual(rect, lastRectRef.current)) {
      lastRectRef.current = rect;
      setTargetRect(rect);
      const vw = window.innerWidth;
      const vh = window.innerHeight;
      const pos = computeCardPosition(rect, vw, vh);
      setCardPos({ top: pos.top, left: pos.left });
    }
  }, [step, getTarget]);

  useEffect(() => {
    if (!isActive) return;
    let running = true;
    const loop = () => {
      if (!running) return;
      updatePosition();
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
    return () => {
      running = false;
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [isActive, updatePosition]);

  if (!isActive || !step) return null;

  if (step.isModal) {
    if (step.id === "welcome") {
      return createPortal(<TutorialWelcome />, document.body);
    }
    if (step.id === "complete") {
      return createPortal(<TutorialComplete />, document.body);
    }
    return null;
  }

  return createPortal(
    <>
      {targetRect ? (
        <SpotlightSvg rect={targetRect} />
      ) : (
        <div
          className="fixed inset-0 bg-black/50 pointer-events-none"
          style={{ zIndex: 61 }}
        />
      )}

      {targetRect && (
        <div
          className="fixed rounded-lg pointer-events-none tutorial-glow"
          style={{
            zIndex: 63,
            top: targetRect.top,
            left: targetRect.left,
            width: targetRect.width,
            height: targetRect.height,
          }}
        />
      )}

      <StepCard
        step={step}
        cardPos={cardPos}
        onNext={nextStep}
        onSkip={skipTutorial}
        currentStep={currentStep}
      />
    </>,
    document.body
  );
}
