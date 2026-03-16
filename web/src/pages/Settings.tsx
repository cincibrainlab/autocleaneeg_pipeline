import { useState, useEffect, useRef } from "react";
import { CheckCircle2, AlertTriangle, AlertCircle, RefreshCw, Upload, ChevronDown, ChevronRight, FileText, X } from "lucide-react";
import { usePolling } from "../hooks/usePolling";
import { api } from "../lib/api";
import type { ValidationResponse } from "../lib/api";
import ErrorBanner from "../components/ErrorBanner";
import CodeViewer from "../components/CodeViewer";
import { useTutorial } from "../contexts/TutorialContext";
import { useTutorialTarget } from "../hooks/useTutorialTarget";

export default function Settings() {
  const {
    data: configData,
    error: configError,
    loading: configLoading,
    refresh: refreshConfig,
  } = usePolling(api.getConfigYaml, 30000);
  const { data: health } = usePolling(api.getHealth, 30000);

  const [validation, setValidation] = useState<ValidationResponse | null>(null);
  const [validating, setValidating] = useState(false);
  const [deploying, setDeploying] = useState(false);
  const [notice, setNotice] = useState<{ type: "success" | "error"; text: string } | null>(null);
  const noticeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [errorsOpen, setErrorsOpen] = useState(true);
  const [warningsOpen, setWarningsOpen] = useState(true);

  // Tutorial integration
  const { isActive, currentStep, nextStep } = useTutorial();
  const applyButtonRef = useTutorialTarget("apply-button");

  useEffect(() => {
    handleValidate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Clear notice timer on unmount
  useEffect(() => {
    return () => {
      if (noticeTimerRef.current) clearTimeout(noticeTimerRef.current);
    };
  }, []);

  const handleValidate = async () => {
    setValidating(true);
    try {
      const res = await api.validateConfig();
      setValidation(res);
    } catch (err) {
      setValidation({
        valid: false,
        errors: [err instanceof Error ? err.message : String(err)],
        warnings: [],
      });
    } finally {
      setValidating(false);
    }
  };

  const handleDeploy = async () => {
    setDeploying(true);
    setNotice(null);
    try {
      const res = await api.deployConfig();
      if (res.success) {
        setNotice({ type: "success", text: res.message || "Configuration applied successfully" });
        // Advance tutorial from apply-config (step 4) to start-service (step 5)
        if (isActive && currentStep === 4) {
          nextStep();
        }
      } else {
        setNotice({ type: "error", text: res.message || "Deploy failed" });
      }
      handleValidate();
      refreshConfig();
    } catch (err) {
      setNotice({ type: "error", text: err instanceof Error ? err.message : String(err) });
    } finally {
      setDeploying(false);
      if (noticeTimerRef.current) clearTimeout(noticeTimerRef.current);
      noticeTimerRef.current = setTimeout(() => setNotice(null), 6000);
    }
  };

  const status = (() => {
    if (!validation) return "checking";
    if (validation.errors.length > 0) return "errors";
    if (validation.warnings.length > 0) return "warnings";
    return "valid";
  })();

  const hasErrors = validation && validation.errors.length > 0;
  const hasWarnings = validation && validation.warnings.length > 0;

  const yamlLines = (configData?.content || "").split("\n");
  const mode = health?.mode ?? "test";
  const configFile = `serve-${mode}.yaml`;

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <h2 className="text-xl font-semibold text-zinc-100">Settings</h2>
        <div className="flex items-center gap-2">
          <button
            onClick={() => { handleValidate(); refreshConfig(); }}
            disabled={validating}
            className="rounded-md px-3 py-1.5 text-sm font-medium border border-border text-zinc-300 hover:bg-surface-50 transition-colors duration-150 flex items-center gap-2 disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${validating ? "animate-spin" : ""}`} />
            Validate
          </button>
          <button
            ref={applyButtonRef}
            onClick={handleDeploy}
            disabled={deploying || status === "errors"}
            title={status === "errors" ? "Fix errors before applying" : "Copy config to deploy/"}
            className="rounded-md px-3 py-1.5 text-sm font-medium bg-brand text-surface-500 hover:bg-brand-500 transition-colors duration-150 flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Upload className="w-3.5 h-3.5" />
            {deploying ? "Applying..." : "Apply"}
          </button>
        </div>
      </div>

      {configError && <ErrorBanner message={configError} />}

      {/* Notice */}
      {notice && (
        <div
          className={`rounded-lg px-4 py-2 text-sm font-medium flex items-center justify-between ${
            notice.type === "success"
              ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/30"
              : "bg-red-500/10 text-red-400 border border-red-500/30"
          }`}
        >
          {notice.text}
          <button onClick={() => setNotice(null)} className="opacity-60 hover:opacity-100">
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* Validation issues */}
      {hasErrors && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/5 overflow-hidden">
          <button
            onClick={() => setErrorsOpen(!errorsOpen)}
            className="w-full px-4 py-3 flex items-center gap-2 text-left hover:bg-red-500/5 transition-colors"
          >
            {errorsOpen ? <ChevronDown className="w-3.5 h-3.5 text-red-400" /> : <ChevronRight className="w-3.5 h-3.5 text-red-400" />}
            <AlertCircle className="w-4 h-4 text-red-400" />
            <span className="text-sm font-semibold text-red-400">
              {validation!.errors.length} {validation!.errors.length === 1 ? "Error" : "Errors"}
            </span>
            <span className="text-xs text-red-400/60 ml-1">— must fix before applying</span>
          </button>
          {errorsOpen && (
            <ul className="px-4 pb-3 space-y-1.5 ml-6">
              {validation!.errors.map((err, i) => (
                <li key={i} className="text-sm text-red-400/80 flex items-start gap-2">
                  <span className="text-red-500/40 select-none">-</span>
                  <span>{err}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {hasWarnings && (
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 overflow-hidden">
          <button
            onClick={() => setWarningsOpen(!warningsOpen)}
            className="w-full px-4 py-3 flex items-center gap-2 text-left hover:bg-amber-500/5 transition-colors"
          >
            {warningsOpen ? <ChevronDown className="w-3.5 h-3.5 text-amber-400" /> : <ChevronRight className="w-3.5 h-3.5 text-amber-400" />}
            <AlertTriangle className="w-4 h-4 text-amber-400" />
            <span className="text-sm font-semibold text-amber-400">
              {validation!.warnings.length} {validation!.warnings.length === 1 ? "Warning" : "Warnings"}
            </span>
          </button>
          {warningsOpen && (
            <ul className="px-4 pb-3 space-y-1.5 ml-6">
              {validation!.warnings.map((w, i) => (
                <li key={i} className="text-sm text-amber-400/80 flex items-start gap-2">
                  <span className="text-amber-500/40 select-none">-</span>
                  <span>{w}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {/* YAML viewer */}
      <div className="rounded-lg border border-border bg-surface-100 overflow-hidden">
        {/* Toolbar */}
        <div className="px-5 py-3 border-b border-border flex items-center justify-between">
          <div className="flex items-center gap-3">
            <FileText className="w-4 h-4 text-zinc-500" />
            <span className="text-sm font-semibold text-zinc-100">{configFile}</span>
            <span className="text-xs text-zinc-600">
              ({mode === "live" ? "Live" : "Testing"} lane)
            </span>
          </div>
          <div className="flex items-center gap-3">
            {/* Status indicator */}
            {status === "valid" && (
              <div className="flex items-center gap-1.5 text-xs text-brand">
                <CheckCircle2 className="w-3.5 h-3.5" />
                Valid
              </div>
            )}
            {status === "errors" && (
              <div className="flex items-center gap-1.5 text-xs text-red-400">
                <AlertCircle className="w-3.5 h-3.5" />
                Invalid
              </div>
            )}
            {status === "warnings" && (
              <div className="flex items-center gap-1.5 text-xs text-amber-400">
                <AlertTriangle className="w-3.5 h-3.5" />
                Warnings
              </div>
            )}
            {status === "checking" && (
              <div className="flex items-center gap-1.5 text-xs text-zinc-500">
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                Checking
              </div>
            )}
            <span className="text-xs text-zinc-600">{yamlLines.length} lines</span>
          </div>
        </div>

        {/* Code area */}
        {configLoading && !configData ? (
          <div className="p-6 text-sm text-zinc-600 bg-[#0A0A0A]">Loading configuration...</div>
        ) : yamlLines.length === 0 || !configData?.content ? (
          <div className="p-6 text-center bg-[#0A0A0A]">
            <FileText className="w-8 h-8 text-zinc-700 mx-auto mb-2" />
            <p className="text-sm text-zinc-500">No configuration found</p>
            <p className="text-xs text-zinc-600 mt-1">Create routes first, then sync to generate the config</p>
          </div>
        ) : (
          <CodeViewer lines={yamlLines} colorize={colorizeYaml} />
        )}
      </div>
    </div>
  );
}

// Simple YAML syntax highlighting
function colorizeYaml(line: string): React.ReactNode {
  // Comment lines
  if (/^\s*#/.test(line)) {
    return <span className="text-zinc-600">{line}</span>;
  }
  // Key: value lines
  const match = line.match(/^(\s*)([\w_-]+)(:)(.*)/);
  if (match) {
    const [, indent, key, colon, rest] = match as RegExpMatchArray;
    return (
      <>
        {indent ?? ""}
        <span className="text-cyan-400">{key ?? ""}</span>
        <span className="text-zinc-500">{colon ?? ""}</span>
        {colorizeValue(rest ?? "")}
      </>
    );
  }
  // List items
  const listMatch = line.match(/^(\s*)(- )(.*)/);
  if (listMatch) {
    const [, indent, dash, rest] = listMatch as RegExpMatchArray;
    return (
      <>
        {indent}
        <span className="text-zinc-500">{dash}</span>
        <span className="text-zinc-300">{rest}</span>
      </>
    );
  }
  return line;
}

function colorizeValue(val: string): React.ReactNode {
  const trimmed = val.trim();
  if (!trimmed) return val;
  // Boolean
  if (/^(true|false)$/i.test(trimmed)) {
    return <span className="text-amber-400"> {trimmed}</span>;
  }
  // Number
  if (/^-?\d+(\.\d+)?$/.test(trimmed)) {
    return <span className="text-purple-400"> {trimmed}</span>;
  }
  // String values
  return <span className="text-emerald-400">{val}</span>;
}
