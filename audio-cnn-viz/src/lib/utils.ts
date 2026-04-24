import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import type { LayerData, VisualizationData } from "./types";
import { int } from "zod/v4";
import { ESC50_EMOJI_MAP } from "./constancts";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function splitLayers(
  visualization: Record<string, LayerData> | undefined,
) {
  if (!visualization) {
    return { main: [], internals: {} };
  }

  const main: [string, LayerData][] = [];
  const internals: Record<string, [string, LayerData][]> = {};

  for (const [name, data] of Object.entries(visualization)) {
    if (!name.includes(".")) {
      main.push([name, data]);
    } else {
      const [parent, child] = name.split(".");
      if (parent === undefined) continue;

      internals[parent] ??= [];
      internals[parent].push([name, data]);
    }
  }
}

export const getEmojiForClass = (className: string): string => {
  return ESC50_EMOJI_MAP[className] ?? "🔈";
};

export function getColor(value: number): [number, number, number] {
  let r, g, b;
  if (value > 0) {
    r = 255 * (1 - value * 0.8);
    g = 255 * (1 - value * 0.5);
    b = 255;
  } else {
    r = 255;
    g = 255 * (1 + value * 0.5);
    b = 255 * (1 + value * 0.8);
  }
  return [Math.round(r), Math.round(g), Math.round(b)];
}
