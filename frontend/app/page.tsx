import { redirect } from "next/navigation";

/**
 * Home page — redirects to the Scientific Modeling IDE workbench.
 *
 * The legacy single-page chat UI that lived here has been refactored into the
 * four-pane WorkbenchShell (Task C.1): the chat experience now lives in the
 * collapsible AI Assistant pane of /workspace. A dedicated landing page will
 * be designed in Task C.10.
 */
export default function HomePage() {
  redirect("/workspace");
}
