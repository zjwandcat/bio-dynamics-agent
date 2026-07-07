"use client";

import React, {
  useState,
  useRef,
  useCallback,
  useImperativeHandle,
  forwardRef,
} from "react";
import {
  Upload,
  FileCode,
  CheckCircle2,
  AlertCircle,
  Loader2,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { API_BASE, V4_PREFIX } from "@/lib/api";

/**
 * SBML Upload — drag-and-drop + file browser sub-component for the InputArea
 * "SBML Upload" tab (Task C.6).
 *
 * Responsibilities:
 * - File selection via drag-and-drop or a hidden `<input type="file">` browser.
 * - Client-side validation: accepted extensions (.xml / .sbml) and a max size
 *   (default 10MB).
 * - Imperative `upload()` method (exposed via ref) that POSTs the file as
 *   `multipart/form-data` to `/api/v4/sbml/import` and reports real XHR upload
 *   progress.
 * - Displays the selected file name, upload progress bar, and parse status
 *   (done / error) inline so the parent InputArea submit button can stay
 *   focused on orchestration.
 *
 * The parent owns the submit button; it calls `sbmlRef.current?.upload()` when
 * the user submits in SBML mode. This keeps the upload lifecycle (progress +
 * status) visually self-contained inside this component while the parent stays
 * in control of when the network call fires.
 */

export type SbmlUploadStatus =
  | "idle"
  | "selected"
  | "uploading"
  | "done"
  | "error";

/** Shape of the `/api/v4/sbml/import` response (kept permissive). */
export interface SbmlUploadResult {
  run_id?: string;
  pathway_class?: string;
  species_count?: number;
  reaction_count?: number;
  message?: string;
  [key: string]: unknown;
}

export interface SbmlUploadHandle {
  /** Upload the currently-selected file. Resolves with the parsed result or null. */
  upload: () => Promise<SbmlUploadResult | null>;
  /** Reset back to idle (clears file + status). */
  reset: () => void;
}

export interface SbmlUploadProps {
  /** Called whenever the selected file changes (null when cleared). */
  onFileSelect?: (file: File | null) => void;
  /** Called when the upload completes successfully. */
  onUploaded?: (result: SbmlUploadResult) => void;
  /** Called when the upload fails. */
  onError?: (error: string) => void;
  /** Max file size in bytes (default 10MB). */
  maxSize?: number;
}

const ACCEPTED_EXT = [".xml", ".sbml"];

function hasAcceptedExt(name: string): boolean {
  const lower = name.toLowerCase();
  return ACCEPTED_EXT.some((ext) => lower.endsWith(ext));
}

export const SbmlUpload = forwardRef<SbmlUploadHandle, SbmlUploadProps>(
  function SbmlUpload(
    { onFileSelect, onUploaded, onError, maxSize = 10 * 1024 * 1024 },
    ref
  ) {
    const [file, setFile] = useState<File | null>(null);
    const [status, setStatus] = useState<SbmlUploadStatus>("idle");
    const [progress, setProgress] = useState(0);
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<SbmlUploadResult | null>(null);
    const [dragOver, setDragOver] = useState(false);
    const inputRef = useRef<HTMLInputElement>(null);
    const xhrRef = useRef<XMLHttpRequest | null>(null);

    const reset = useCallback(() => {
      xhrRef.current?.abort();
      xhrRef.current = null;
      setFile(null);
      setStatus("idle");
      setProgress(0);
      setError(null);
      setResult(null);
      onFileSelect?.(null);
      if (inputRef.current) inputRef.current.value = "";
    }, [onFileSelect]);

    const validate = useCallback(
      (f: File): string | null => {
        if (!hasAcceptedExt(f.name)) {
          return `Unsupported file type. Accepted: ${ACCEPTED_EXT.join(", ")}`;
        }
        if (f.size > maxSize) {
          return `File too large (max ${Math.round(maxSize / 1024 / 1024)}MB)`;
        }
        return null;
      },
      [maxSize]
    );

    const selectFile = useCallback(
      (f: File | null) => {
        if (!f) return;
        const validationError = validate(f);
        if (validationError) {
          setError(validationError);
          setStatus("error");
          return;
        }
        setError(null);
        setResult(null);
        setProgress(0);
        setFile(f);
        setStatus("selected");
        onFileSelect?.(f);
      },
      [onFileSelect, validate]
    );

    const upload = useCallback((): Promise<SbmlUploadResult | null> => {
      if (!file) {
        const msg = "No file selected";
        setError(msg);
        setStatus("error");
        onError?.(msg);
        return Promise.resolve(null);
      }

      setStatus("uploading");
      setProgress(0);
      setError(null);

      return new Promise<SbmlUploadResult | null>((resolve) => {
        const xhr = new XMLHttpRequest();
        xhrRef.current = xhr;
        const formData = new FormData();
        formData.append("file", file);

        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable) {
            setProgress(Math.round((e.loaded / e.total) * 100));
          }
        };

        xhr.onload = () => {
          xhrRef.current = null;
          let parsed: SbmlUploadResult | null = null;
          try {
            parsed = xhr.responseText
              ? (JSON.parse(xhr.responseText) as SbmlUploadResult)
              : null;
          } catch {
            parsed = null;
          }

          if (xhr.status >= 200 && xhr.status < 300) {
            setStatus("done");
            setResult(parsed);
            onUploaded?.(parsed ?? {});
            resolve(parsed);
          } else {
            const msg =
              parsed?.message ||
              `Upload failed: HTTP ${xhr.status} ${xhr.statusText}`;
            setError(msg);
            setStatus("error");
            onError?.(msg);
            resolve(null);
          }
        };

        xhr.onerror = () => {
          xhrRef.current = null;
          const msg = "Network error during upload";
          setError(msg);
          setStatus("error");
          onError?.(msg);
          resolve(null);
        };

        xhr.onabort = () => {
          xhrRef.current = null;
          setStatus("selected");
          setError(null);
          resolve(null);
        };

        xhr.open("POST", `${API_BASE}${V4_PREFIX}/sbml/import`);
        xhr.send(formData);
      });
    }, [file, onUploaded, onError]);

    useImperativeHandle(ref, () => ({ upload, reset }), [upload, reset]);

    const handleDrop = (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const f = e.dataTransfer.files?.[0];
      if (f) selectFile(f);
    };

    const openBrowser = () => inputRef.current?.click();

    return (
      <div className="space-y-2">
        <input
          ref={inputRef}
          type="file"
          accept=".xml,.sbml,application/xml,text/xml"
          className="hidden"
          onChange={(e) => selectFile(e.target.files?.[0] ?? null)}
        />

        {!file && (
          <div
            onDrop={handleDrop}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={(e) => {
              e.preventDefault();
              setDragOver(false);
            }}
            onClick={openBrowser}
            className={cn(
              "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed px-4 py-8 text-center transition-colors",
              dragOver
                ? "border-blue-500 bg-blue-500/5"
                : "border-zinc-700 bg-zinc-950/40 hover:border-zinc-600 hover:bg-zinc-900/40"
            )}
          >
            <Upload className="h-6 w-6 text-zinc-500" />
            <div className="text-xs text-zinc-300">
              Drag &amp; drop an SBML file, or{" "}
              <span className="text-blue-400 underline">browse</span>
            </div>
            <div className="text-[10px] text-zinc-500">
              Accepts .xml / .sbml · max 10MB
            </div>
          </div>
        )}

        {file && (
          <div className="space-y-2">
            <div className="flex items-center gap-2 rounded-md border border-zinc-700 bg-zinc-900/60 px-2.5 py-2">
              <FileCode className="h-4 w-4 shrink-0 text-emerald-400" />
              <div className="min-w-0 flex-1">
                <div className="truncate text-xs text-zinc-200">{file.name}</div>
                <div className="text-[10px] text-zinc-500">
                  {(file.size / 1024).toFixed(1)} KB
                </div>
              </div>
              {status === "done" && (
                <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-400" />
              )}
              {status === "error" && (
                <AlertCircle className="h-4 w-4 shrink-0 text-red-400" />
              )}
              {status === "uploading" && (
                <Loader2 className="h-4 w-4 shrink-0 animate-spin text-blue-400" />
              )}
              <button
                type="button"
                onClick={reset}
                className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
                aria-label="Remove file"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </div>

            {status === "uploading" && (
              <div className="space-y-1">
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-800">
                  <div
                    className="h-full bg-blue-500 transition-all"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <div className="text-[10px] text-zinc-500">
                  Uploading {progress}%
                </div>
              </div>
            )}

            {status === "done" && (
              <div className="rounded-md border border-emerald-700/40 bg-emerald-900/10 px-2.5 py-2 text-[11px] text-emerald-300">
                <div className="flex items-center gap-1.5 font-medium">
                  <CheckCircle2 className="h-3 w-3" /> Import complete
                </div>
                {(result?.species_count !== undefined ||
                  result?.reaction_count !== undefined) && (
                  <div className="mt-1 text-emerald-400/80">
                    {result?.species_count !== undefined &&
                      `${result.species_count} species · `}
                    {result?.reaction_count !== undefined &&
                      `${result.reaction_count} reactions`}
                  </div>
                )}
              </div>
            )}

            {status === "error" && error && (
              <div className="rounded-md border border-red-700/40 bg-red-900/10 px-2.5 py-2 text-[11px] text-red-300">
                <div className="flex items-center gap-1.5">
                  <AlertCircle className="h-3 w-3" /> {error}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    );
  }
);
