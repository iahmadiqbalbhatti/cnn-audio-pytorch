export interface Prediction {
  class: string;
  confidence: number;
}

export interface LayerData {
  shape: number[];
  values: number[][]; // Flattened array of values
}

export type VisualizationData = Record<string, LayerData>;

export interface WaveformData {
  values: number[]; // Flattened array of waveform values
  sample_rate: number;
  duration: number;
}

export interface ApiResponse {
  predictions: Prediction[];
  visualizations: VisualizationData;
  input_spectrogram: LayerData;
  waveform: WaveformData;
}

export interface ColorScaleProps {
  width: number;
  height: number;
  min: number;
  max: number;
}

export interface FeatureMapProp {
  data: number[][];
  title: string;
  internal?: boolean;
  spectrogram?: boolean;
}