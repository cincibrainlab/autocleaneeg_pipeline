export type TutorialStepId =
  | "welcome"
  | "dashboard-overview"
  | "create-route-button"
  | "route-form"
  | "apply-config"
  | "start-service"
  | "watch-queue"
  | "complete";

export type StepAction = "next" | "waiting" | "none";

export interface TutorialStep {
  /** 0-based index */
  index: number;
  id: TutorialStepId;
  /** Target element ID registered via useTutorialTarget */
  targetId: string | null;
  /** Page path the tutorial auto-navigates to for this step */
  route: string | null;
  title: string;
  description: string;
  /** What the primary button does */
  action: StepAction;
  /** If true, render a full-screen modal rather than a spotlight card */
  isModal: boolean;
}

export const TUTORIAL_STEPS: TutorialStep[] = [
  {
    index: 0,
    id: "welcome",
    targetId: null,
    route: "/",
    title: "Welcome to AutoCleanEEG Serve",
    description:
      "Let's set up your first processing route in about 2 minutes. Routes are the core project unit, and Queue, Results, and Exclude now follow that route context.",
    action: "next",
    isModal: true,
  },
  {
    index: 1,
    id: "dashboard-overview",
    targetId: "dashboard-stats",
    route: "/",
    title: "This is your Dashboard",
    description:
      "These cards give you a live snapshot of routes, queue, settings, and service status. Use them to jump into the route workflow, then follow Queue into Results and Exclude.",
    action: "next",
    isModal: false,
  },
  {
    index: 2,
    id: "create-route-button",
    targetId: "new-route-button",
    route: "/routes",
    title: "Create a Processing Route",
    description:
      "A route tells AutoCleanEEG Serve which folder to watch, which file patterns to match, and which task to run. Click \"New Route\" to open the form.",
    action: "waiting",
    isModal: false,
  },
  {
    index: 3,
    id: "route-form",
    targetId: "route-modal",
    route: "/routes",
    title: "Fill in the Route Details",
    description:
      "The form is pre-filled with a sample configuration. Review the fields, then click \"Create Route\" to save.",
    action: "waiting",
    isModal: false,
  },
  {
    index: 4,
    id: "apply-config",
    targetId: "apply-button",
    route: "/settings",
    title: "Apply the Configuration",
    description:
      "Routes are compiled into a serve config. Click \"Apply\" to deploy the current settings so the service can use them.",
    action: "waiting",
    isModal: false,
  },
  {
    index: 5,
    id: "start-service",
    targetId: "service-control",
    route: "/service",
    title: "Start the Processing Service",
    description:
      "The service watches your configured folders and dispatches EEG files for processing. Click \"Start Service\" to begin.",
    action: "waiting",
    isModal: false,
  },
  {
    index: 6,
    id: "watch-queue",
    targetId: "queue-table",
    route: "/queue",
    title: "Watch Your File Process",
    description:
      "Files discovered by the service appear here. Queue stays global, but you can filter by route when many routes exist. The tutorial file will move from Pending to Processed once the pipeline finishes.",
    action: "waiting",
    isModal: false,
  },
  {
    index: 7,
    id: "complete",
    targetId: null,
    route: null,
    title: "You're all set!",
    description:
      "1 route created, 1 file processed. From here, keep using route-scoped Queue, Results, and Exclude for day-to-day review, and use Tasks, Montages, Events, and Settings as supporting utilities.",
    action: "none",
    isModal: true,
  },
];

export const TOTAL_STEPS = TUTORIAL_STEPS.length;
/** Steps 0 and 7 are modals; the visible progress range is 1-6 */
export const PROGRESS_STEPS = TUTORIAL_STEPS.filter((s) => !s.isModal);
