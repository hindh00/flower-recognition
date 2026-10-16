"use client";

import { useCallback, useState } from "react";
import { resizeImage } from "@/app/lib/resizeImage";
import { PredictionResult, type Detection, type Prediction } from "@/app/components/PredictionResult";

type Status = "idle" | "loading" | "done" | "error";

export function UploadForm() {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [lowConfidence, setLowConfidence] = useState(false);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [previewAspectRatio, setPreviewAspectRatio] = useState<number | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFile = useCallback(async (file: File) => {
    if (!file.type.startsWith("image/")) {
      setStatus("error");
      setError("Please choose an image file.");
      return;
    }

    setPreviewUrl(URL.createObjectURL(file));
    setPredictions([]);
    setLowConfidence(false);
    setDetections([]);
    setPreviewAspectRatio(null);
    setStatus("loading");
    setError(null);

    try {
      const resized = await resizeImage(file);
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "image/jpeg" },
        body: resized,
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? "Prediction failed");

      setPredictions(data.predictions);
      setLowConfidence(Boolean(data.low_confidence));
      setDetections(Array.isArray(data.detections) ? data.detections : []);
      setStatus("done");
    } catch (err) {
      setStatus("error");
      setError(err instanceof Error ? err.message : "Something went wrong");
    }
  }, []);

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  };

  return (
    <div className="flex w-full max-w-md flex-col items-center gap-6">
      <label
        htmlFor="file-input"
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={onDrop}
        className={`flex w-full cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-8 text-center transition-colors ${
          isDragging
            ? "border-emerald-500 bg-emerald-50 dark:bg-emerald-950/30"
            : "border-neutral-300 dark:border-neutral-700"
        }`}
      >
        {previewUrl ? (
          // Wrapper sized to match the rendered <img>'s own box (via
          // aspect-ratio) so the percentage-based detection overlays below
          // line up with the actual photo instead of a letterboxed gap.
          // Edge case accepted, not solved: very wide/panoramic photos can
          // be clipped by the parent's max-width before aspect-ratio sizing
          // fully resolves, throwing off the overlay slightly.
          <div className="relative inline-block max-h-64 max-w-full" style={{ aspectRatio: previewAspectRatio ?? undefined }}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={previewUrl}
              alt="Selected flower"
              className="h-full w-full rounded-lg object-contain"
              onLoad={(e) => setPreviewAspectRatio(e.currentTarget.naturalWidth / e.currentTarget.naturalHeight)}
            />
            {detections.map((d, i) => {
              const [x1, y1, x2, y2] = d.box;
              return (
                <div
                  key={i}
                  className="absolute rounded-sm border-2 border-emerald-400"
                  style={{
                    left: `${x1 * 100}%`,
                    top: `${y1 * 100}%`,
                    width: `${(x2 - x1) * 100}%`,
                    height: `${(y2 - y1) * 100}%`,
                  }}
                />
              );
            })}
          </div>
        ) : (
          <>
            <p className="text-sm font-medium">Drop a flower photo here, or click to choose one</p>
            <p className="text-xs text-neutral-500">JPG or PNG</p>
          </>
        )}
        <input id="file-input" type="file" accept="image/*" onChange={onInputChange} className="hidden" />
      </label>

      {status === "loading" && <p className="text-sm text-neutral-500">Identifying...</p>}
      {status === "error" && <p className="text-sm text-red-600">{error}</p>}
      {status === "done" && (
        <PredictionResult predictions={predictions} lowConfidence={lowConfidence} detectionCount={detections.length} />
      )}
    </div>
  );
}
