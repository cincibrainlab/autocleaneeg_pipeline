import "@testing-library/jest-dom/vitest";
import { vi } from "vitest";

HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
  clearRect: () => {},
  fillRect: () => {},
  beginPath: () => {},
  moveTo: () => {},
  lineTo: () => {},
  stroke: () => {},
  fillStyle: "",
  strokeStyle: "",
  lineWidth: 1,
})) as unknown as typeof HTMLCanvasElement.prototype.getContext;
