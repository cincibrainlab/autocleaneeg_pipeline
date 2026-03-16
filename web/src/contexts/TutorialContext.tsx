import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { api } from "../lib/api";
import type { TutorialSuggestedRoute } from "../lib/api";
import { TUTORIAL_STEPS } from "../components/tutorial/tutorialSteps";

// ── Persisted state shape ────────────────────────────────────────────

interface TutorialPersistedState {
  currentStep: number;
  isActive: boolean;
  completed: boolean;
}

const STORAGE_KEY = "autoclean-tutorial-state";

function loadState(): TutorialPersistedState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw) as TutorialPersistedState;
  } catch {
    // ignore parse errors
  }
  return { currentStep: 0, isActive: false, completed: false };
}

function saveState(s: TutorialPersistedState) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(s));
  } catch {
    // ignore storage errors
  }
}

// ── Context shape ────────────────────────────────────────────────────

export interface TutorialContextValue {
  currentStep: number;
  isActive: boolean;
  completed: boolean;
  tutorialData: { sampleFile: string; suggestedRoute: TutorialSuggestedRoute } | null;
  /** Get a target element by step ID (reads from mutable ref) */
  getTarget: (stepId: string) => HTMLElement | undefined;
  startTutorial: () => Promise<void>;
  nextStep: () => void;
  skipTutorial: () => void;
  completeTutorial: () => void;
  registerTarget: (stepId: string, el: HTMLElement | null) => void;
}

export const TutorialContext = createContext<TutorialContextValue | null>(null);

export function useTutorial(): TutorialContextValue {
  const ctx = useContext(TutorialContext);
  if (!ctx) throw new Error("useTutorial must be used inside TutorialProvider");
  return ctx;
}

// ── Provider ─────────────────────────────────────────────────────────

export function TutorialProvider({ children }: { children: React.ReactNode }) {
  const persisted = useRef(loadState()).current;
  const [currentStep, setCurrentStep] = useState(persisted.currentStep);
  const [isActive, setIsActive] = useState(persisted.isActive);
  const [completed, setCompleted] = useState(persisted.completed);
  const [tutorialData, setTutorialData] = useState<TutorialContextValue["tutorialData"]>(null);

  // Mutable ref for targets — never triggers re-renders
  const targetsRef = useRef<Map<string, HTMLElement>>(new Map());

  // Persist whenever relevant state changes
  useEffect(() => {
    saveState({ currentStep, isActive, completed });
  }, [currentStep, isActive, completed]);

  const registerTarget = useCallback(
    (stepId: string, el: HTMLElement | null) => {
      if (el) {
        targetsRef.current.set(stepId, el);
      } else {
        targetsRef.current.delete(stepId);
      }
      // No state update — the overlay reads targets imperatively
    },
    []
  );

  const getTarget = useCallback((stepId: string) => {
    return targetsRef.current.get(stepId);
  }, []);

  const startTutorial = useCallback(async () => {
    try {
      const result = await api.tutorialSetup();
      if (result.sample_file && result.suggested_route) {
        setTutorialData({
          sampleFile: result.sample_file,
          suggestedRoute: result.suggested_route,
        });
      }
    } catch (err) {
      console.warn("Tutorial setup failed, continuing without sample data:", err);
    }
    setCurrentStep(0);
    setIsActive(true);
    setCompleted(false);
  }, []);

  const nextStep = useCallback(() => {
    setCurrentStep((prev) => {
      const next = prev + 1;
      if (next >= TUTORIAL_STEPS.length) {
        setIsActive(false);
        setCompleted(true);
        return prev;
      }
      return next;
    });
  }, []);

  const skipTutorial = useCallback(() => {
    setIsActive(false);
    setCurrentStep(0);
    api.tutorialCleanup().catch(() => {});
  }, []);

  const completeTutorial = useCallback(() => {
    setIsActive(false);
    setCompleted(true);
    setCurrentStep(0);
    api.tutorialCleanup().catch(() => {});
  }, []);

  const value = useMemo<TutorialContextValue>(
    () => ({
      currentStep,
      isActive,
      completed,
      tutorialData,
      getTarget,
      startTutorial,
      nextStep,
      skipTutorial,
      completeTutorial,
      registerTarget,
    }),
    [currentStep, isActive, completed, tutorialData, getTarget, startTutorial, nextStep, skipTutorial, completeTutorial, registerTarget]
  );

  return (
    <TutorialContext.Provider value={value}>
      {children}
    </TutorialContext.Provider>
  );
}
