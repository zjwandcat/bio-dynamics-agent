"use client";

import * as React from "react";
import { cn } from "@/lib/utils";

/**
 * Minimal self-contained Tabs primitive (shadcn-compatible API surface).
 *
 * Built without an external base-ui/radix dependency so the pathway detail
 * panel can render its Ontology / Reaction / Evidence tabs. Controlled or
 * uncontrolled; dark-theme styled to match the workbench zinc palette.
 */

interface TabsContextValue {
  value: string;
  setValue: (value: string) => void;
}

const TabsContext = React.createContext<TabsContextValue | null>(null);

function useTabsContext(component: string): TabsContextValue {
  const ctx = React.useContext(TabsContext);
  if (!ctx) {
    throw new Error(`<${component}> must be used within <Tabs>`);
  }
  return ctx;
}

export interface TabsProps extends React.ComponentProps<"div"> {
  /** Controlled active value. */
  value?: string;
  /** Initial active value when uncontrolled. */
  defaultValue?: string;
  /** Called when the active tab changes. */
  onValueChange?: (value: string) => void;
}

export function Tabs({
  value,
  defaultValue,
  onValueChange,
  className,
  children,
  ...props
}: TabsProps) {
  const [internalValue, setInternalValue] = React.useState(
    defaultValue ?? ""
  );
  const activeValue = value ?? internalValue;

  const setValue = React.useCallback(
    (next: string) => {
      if (value === undefined) setInternalValue(next);
      onValueChange?.(next);
    },
    [value, onValueChange]
  );

  const ctx = React.useMemo<TabsContextValue>(
    () => ({ value: activeValue, setValue }),
    [activeValue, setValue]
  );

  return (
    <TabsContext.Provider value={ctx}>
      <div className={cn("flex flex-col", className)} {...props}>
        {children}
      </div>
    </TabsContext.Provider>
  );
}

export function TabsList({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      role="tablist"
      className={cn(
        "inline-flex h-8 shrink-0 items-center gap-1 rounded-lg border border-zinc-800 bg-zinc-900/70 p-1",
        className
      )}
      {...props}
    />
  );
}

export interface TabsTriggerProps
  extends React.ComponentProps<"button"> {
  value: string;
}

export function TabsTrigger({
  value,
  className,
  ...props
}: TabsTriggerProps) {
  const ctx = useTabsContext("TabsTrigger");
  const active = ctx.value === value;
  return (
    <button
      type="button"
      role="tab"
      aria-selected={active}
      data-state={active ? "active" : "inactive"}
      onClick={() => ctx.setValue(value)}
      className={cn(
        "inline-flex h-6 items-center justify-center rounded-md px-2.5 text-xs font-medium whitespace-nowrap transition-colors outline-none focus-visible:ring-2 focus-visible:ring-blue-500/40",
        active
          ? "bg-zinc-100 text-zinc-900 shadow-sm"
          : "text-zinc-400 hover:bg-zinc-800/60 hover:text-zinc-200",
        className
      )}
      {...props}
    />
  );
}

export interface TabsContentProps extends React.ComponentProps<"div"> {
  value: string;
}

export function TabsContent({
  value,
  className,
  children,
  ...props
}: TabsContentProps) {
  const ctx = useTabsContext("TabsContent");
  if (ctx.value !== value) return null;
  return (
    <div
      role="tabpanel"
      className={cn("min-h-0 flex-1 overflow-auto", className)}
      {...props}
    >
      {children}
    </div>
  );
}
