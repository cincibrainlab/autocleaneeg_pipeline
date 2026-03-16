import { useCallback } from "react";
import { useTutorial } from "../contexts/TutorialContext";

/**
 * Returns a ref callback that registers the given element as the spotlight
 * target for the specified step ID.
 *
 * Usage:
 *   const ref = useTutorialTarget("new-route-button");
 *   <button ref={ref}>New Route</button>
 */
export function useTutorialTarget(stepId: string): (el: HTMLElement | null) => void {
  const { registerTarget } = useTutorial();

  const refCallback = useCallback(
    (el: HTMLElement | null) => {
      registerTarget(stepId, el);
    },
    // stepId is static per call site; registerTarget is stable from useCallback
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [stepId, registerTarget]
  );

  return refCallback;
}
