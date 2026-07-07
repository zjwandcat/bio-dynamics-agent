/**
 * Vitest setup — registers @testing-library/jest-dom custom matchers
 * (toBeInTheDocument, toHaveTextContent, etc.) and provides a minimal
 * jsdom polyfill for APIs the components rely on.
 */
import "@testing-library/jest-dom/vitest";

// jsdom does not implement ResizeObserver / IntersectionObserver — stub them
// so components using ScrollArea / recharts don't blow up during tests.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
class IntersectionObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}

// Guard against re-declaration in case the global already exists.
if (typeof globalThis.ResizeObserver === "undefined") {
  (globalThis as Record<string, unknown>).ResizeObserver = ResizeObserverStub;
}
if (typeof globalThis.IntersectionObserver === "undefined") {
  (globalThis as Record<string, unknown>).IntersectionObserver =
    IntersectionObserverStub;
}

// matchMedia is not implemented in jsdom.
if (typeof window !== "undefined" && !window.matchMedia) {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }),
  });
}

// scrollTo is not implemented in jsdom.
if (typeof window !== "undefined" && !window.scrollTo) {
  Object.defineProperty(window, "scrollTo", {
    writable: true,
    value: () => {},
  });
}
