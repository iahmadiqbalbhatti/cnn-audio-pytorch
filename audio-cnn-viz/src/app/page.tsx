"use client";
import { Car } from "@phosphor-icons/react";
import { useState } from "react";
import ColorScale from "~/components/ColorScale";
import { Badge } from "~/components/ui/badge";
import { Button } from "~/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import FeatureMap from "~/components/ui/FeatureMap";
import { Progress } from "~/components/ui/progress";
import { env } from "~/env";
import type { ApiResponse } from "~/lib/types";
import { getEmojiForClass, splitLayers } from "~/lib/utils";

export default function HomePage() {
  const [vizData, setVizData] = useState<ApiResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setFileName(file.name);
    setIsLoading(true);
    setError(null);
    setVizData(null);

    const reader = new FileReader();
    reader.readAsArrayBuffer(file);
    reader.onload = async () => {
      try {
        const arrayBuffer = reader.result as ArrayBuffer;
        const base64String = btoa(
          new Uint8Array(arrayBuffer).reduce(
            (data, byte) => data + String.fromCharCode(byte),
            "",
          ),
        );
        if (!env.NEXT_PUBLIC_AUDIO_CNN_INFERENCE_ENDPOINT) {
          throw new Error("API endpoint is not configured");
        }
        const response = await fetch(
          env.NEXT_PUBLIC_AUDIO_CNN_INFERENCE_ENDPOINT,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ audio_data: base64String }),
          },
        );

        if (!response.ok) {
          throw new Error(`API Error: ${response.statusText}`);
        }
        const data = (await response.json()) as ApiResponse;

        setVizData(data);
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setIsLoading(false);
      }
    };

    reader.onerror = () => {
      setError("Failed to read the file. Please try again.");
      setIsLoading(false);
    };
  };


  const { main, internals } = (vizData
    ? splitLayers(vizData.visualizations)
    : null) ?? { main: [], internals: {} };

  return (
    <main className="min-h-screen bg-stone-50 p-8">
      <div className="mx-auto max-w-[60%]">
        <div className="mb-12 text-center">
          <h1 className="mb-4 text-4xl font-bold">Audio CNN Visualization</h1>

          <p className="text-xl text-gray-600">
            Explore the inner workings of Convolutional Neural Networks for
            audio processing.
          </p>

          <p className="text-md mb-8 text-stone-600">
            Upload a .WAV file to see the model&apos;s predictions and featured
            visualizations.
          </p>

          <div className="flex flex-col items-center">
            <div className="relative inline-block">
              <input
                type="file"
                accept=".wav"
                id="file-upload"
                disabled={isLoading}
                onChange={handleFileChange}
                className="absolute inset-0 h-full w-full cursor-pointer opacity-0"
              />
              <Button className="border-stone-300" variant="outline">
                {isLoading ? "Analyzing..." : "Upload .WAV File"}
              </Button>
            </div>

            {fileName && (
              <Badge
                variant={"secondary"}
                className="mt-4 rounded bg-stone-200 px-4 py-2 text-stone-700"
              >
                {fileName}
              </Badge>
            )}

            {error && (
              <Card className="my-8 border-red-200 bg-red-50">
                <CardContent>
                  <p className="text-red-600">Error: {error}</p>
                </CardContent>
              </Card>
            )}

            {vizData && (
              <div className="mt-4 space-y-8">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-stone-900">
                      Top Predictions
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {vizData.predictions.slice(0, 3).map((pred, i) => (
                        <div key={pred.class} className="space-y-2">
                          <div className="flex items-center justify-between">
                            <div className="text-md font-medium text-stone-700">
                              {getEmojiForClass(pred.class)}{" "}
                              <span>{pred.class.replaceAll("_", " ")}</span>
                            </div>
                            <Badge variant={i === 0 ? "default" : "secondary"}>
                              {(pred.confidence * 100).toFixed(1)}%
                            </Badge>
                          </div>
                          <Progress
                            value={pred.confidence * 100}
                            className="h-2"
                          />
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
                <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                  <Card>
                    <CardHeader className="text-stone-900">
                      <CardTitle className="text-stone-900">
                        Input Spectrogram
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <FeatureMap
                        data={vizData.input_spectrogram.values}
                        title={`${vizData.input_spectrogram.shape.join(" x ")}`}
                        spectrogram
                      />
                      <ColorScale width={200} height={16} min={-1} max={1} />
                    </CardContent>
                  </Card>
                </div>
                {/* <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                  <Card>
                    <CardHeader className="text-stone-900">
                      <CardTitle className="text-stone-900">
                        Input Spectrogram
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <FeatureMap
                        data={vizData.input_spectrogram.values}
                        title={`${vizData.input_spectrogram.shape.join(" x ")}`}
                        spectrogram
                      />

                      <div className="mt-5 flex justify-end">
                        <ColorScale width={200} height={16} min={-1} max={1} />
                      </div> 
                    </CardContent>
                  </Card>
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-stone-900">
                        Audio Waveform
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <Waveform
                        data={vizData.waveform.values}
                        title={`${vizData.waveform.duration.toFixed(2)}s * ${vizData.waveform.sample_rate}Hz`}
                      />
                    </CardContent>
                  </Card>
                </div> */}
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
