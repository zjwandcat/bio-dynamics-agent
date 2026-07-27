import { redirect } from "next/navigation";

/**
 * /workspace — 旧四栏 IDE 已迁移至 /advanced。
 *
 * 极简 Auto-Chat 重构后，主入口改为 `/`（对话式）。此路由保留为
 * 301-style 重定向，避免旧书签 / 文档链接 404。
 */
export default function WorkspaceRedirect() {
  redirect("/advanced");
}
