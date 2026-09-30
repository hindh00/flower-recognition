import { UploadForm } from "@/app/components/UploadForm";

export default function Home() {
  return (
    <main className="flex flex-1 flex-col items-center justify-center gap-8 p-8">
      <div className="text-center">
        <h1 className="text-3xl font-semibold">Flower Recognition</h1>
        <p className="mt-2 max-w-md text-sm text-neutral-500">
          Upload a photo and a fine-tuned EfficientNet model will classify it as one of 102 flower
          species from the Oxford 102 Flowers dataset.
        </p>
      </div>
      <UploadForm />
    </main>
  );
}
