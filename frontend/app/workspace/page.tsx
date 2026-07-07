import { WorkbenchShell } from "@/components/workspace/WorkbenchShell";

/**
 * /workspace — the Scientific Modeling IDE four-pane workbench.
 *
 * This is the primary surface of BioDynamics v4. The WorkbenchShell owns the
 * four-pane layout (Project / Scientific Workspace / Validation / AI Assistant)
 * and wires the global Zustand store + SSE chat stream + AI Assistant panel.
 *
 * The page itself is a thin server-component wrapper so that the client-side
 * WorkbenchShell can own all interactivity (per Next.js 16 App Router).
 */
export default function WorkspacePage() {
  return <WorkbenchShell />;
}
