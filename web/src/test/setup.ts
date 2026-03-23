import "@testing-library/jest-dom/vitest";
import { vi } from "vitest";

HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
  clearRect: () => {},
  fillRect: () => {},
  beginPath: () => {},
  moveTo: () => {},
  lineTo: () => {},
  stroke: () => {},
  strokeRect: () => {},
  fillText: () => {},
  fillStyle: "",
  strokeStyle: "",
  lineWidth: 1,
  font: "",
  textBaseline: "alphabetic",
})) as unknown as typeof HTMLCanvasElement.prototype.getContext;

HTMLCanvasElement.prototype.setPointerCapture = vi.fn();
HTMLCanvasElement.prototype.releasePointerCapture = vi.fn();
HTMLCanvasElement.prototype.hasPointerCapture = vi.fn(() => true);
