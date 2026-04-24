import React from "react";
import type { ColorScaleProps } from "~/lib/types";

const ColorScale = ({ width, height, min, max }: ColorScaleProps) => {
  const gradient = "linear-gradient(to right, red, white, blue)";
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-stone-500">{min}</span>

      <div
        className="rounded border border-stone-500"
        style={{
          width: `${width}px`,
          height: `${height}px`,
          background: gradient,
        }}
      ></div>
      <span className="text-xs text-stone-500">{max}</span>
    </div>
  );
};

export default ColorScale;
