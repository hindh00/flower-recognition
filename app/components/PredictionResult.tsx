export type Prediction = {
  label: string;
  confidence: number;
};

export type Detection = {
  box: [number, number, number, number];
  confidence: number;
};

export function PredictionResult({
  predictions,
  lowConfidence,
  detectionCount = 0,
}: {
  predictions: Prediction[];
  lowConfidence?: boolean;
  detectionCount?: number;
}) {
  if (predictions.length === 0) return null;
  const top = predictions[0];

  return (
    <div className="w-full max-w-md space-y-4">
      <div className="text-center">
        <p className="text-sm text-neutral-500">{lowConfidence ? "Not sure — closest match" : "Predicted"}</p>
        <p className="text-2xl font-semibold capitalize">{top.label}</p>
        <p className="text-sm text-neutral-500">{(top.confidence * 100).toFixed(1)}% confidence</p>
        {detectionCount > 1 && (
          <p className="mt-2 text-xs text-neutral-500">
            Detected {detectionCount} flowers — averaging predictions across all of them.
          </p>
        )}
        {lowConfidence && (
          <p className="mt-2 text-xs text-amber-600 dark:text-amber-500">
            The model isn&apos;t confident about this one — it may be a species outside its training set, or the
            photo may have multiple flowers/a busy background. See other close matches below.
          </p>
        )}
      </div>

      <div className="space-y-2">
        <p className="text-xs uppercase tracking-wide text-neutral-500">Top 5</p>
        {predictions.map((p) => (
          <div key={p.label} className="flex items-center gap-3 text-sm">
            <span className="w-32 shrink-0 truncate capitalize">{p.label}</span>
            <div className="h-2 flex-1 rounded-full bg-neutral-200 dark:bg-neutral-800">
              <div
                className="h-2 rounded-full bg-emerald-600 dark:bg-emerald-500"
                style={{ width: `${Math.max(p.confidence * 100, 2)}%` }}
              />
            </div>
            <span className="w-12 shrink-0 text-right text-neutral-500">{(p.confidence * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}
