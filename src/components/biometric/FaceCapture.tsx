import { useState, useRef } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Camera, Check } from "lucide-react";

interface FaceCaptureProps {
  onCapture: (data: { img1: string; img2: string; img3: string }) => void;
  captured: boolean;
}

export const FaceCapture = ({ onCapture, captured }: FaceCaptureProps) => {
  const [capturedImages, setCapturedImages] = useState<string[]>([]);
  const [isCapturing, setIsCapturing] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const startCamera = async () => {
    try {
      // Request camera with mobile-friendly constraints
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      });
      
      streamRef.current = stream;
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        // Ensure video plays immediately
        try {
          await videoRef.current.play();
          console.log("Camera stream started successfully");
        } catch (playError) {
          console.error("Error playing video:", playError);
        }
      }
      
      setIsCapturing(true);
    } catch (error) {
      console.error("Camera access denied:", error);
      alert(`Camera access is required for facial capture. Error: ${error}`);
    }
  };

  const captureImage = () => {
    if (videoRef.current && videoRef.current.videoWidth > 0) {
      const canvas = document.createElement("canvas");
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const ctx = canvas.getContext("2d");
      
      if (ctx) {
        ctx.drawImage(videoRef.current, 0, 0);
        const base64 = canvas.toDataURL("image/jpeg", 0.8);
        
        const newImages = [...capturedImages, base64];
        setCapturedImages(newImages);
        
        // If we have 3 images, complete capture
        if (newImages.length === 3) {
          onCapture({
            img1: newImages[0],
            img2: newImages[1],
            img3: newImages[2]
          });
          stopCamera();
        }
      }
    } else {
      alert("Camera not ready. Please wait and try again.");
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setIsCapturing(false);
  };

  const resetCapture = () => {
    setCapturedImages([]);
    startCamera();
  };

  return (
    <Card className="p-6 bg-card">
      <div className="flex items-start gap-4">
        <div className={`flex-shrink-0 w-12 h-12 rounded-full flex items-center justify-center ${
          captured ? "bg-success text-success-foreground" : "bg-primary text-primary-foreground"
        }`}>
          {captured ? <Check className="h-6 w-6" /> : <Camera className="h-6 w-6" />}
        </div>
        
        <div className="flex-1 min-w-0">
          <h3 className="font-semibold text-lg mb-2">
            Facial Recognition {capturedImages.length > 0 && `(${capturedImages.length}/3)`}
          </h3>
          
          {!isCapturing && capturedImages.length === 0 && (
            <div className="space-y-4">
              <p className="text-sm text-muted-foreground">
                Capture 3 clear photos of your face for biometric authentication
              </p>
              <Button onClick={startCamera} variant="outline" className="w-full sm:w-auto">
                <Camera className="mr-2 h-4 w-4" />
                Start Camera
              </Button>
            </div>
          )}

          {isCapturing && capturedImages.length < 3 && (
            <div className="space-y-4">
              <div className="relative rounded-lg overflow-hidden bg-black aspect-video max-w-md">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full h-full object-cover"
                />
                <div className="absolute top-2 right-2 bg-primary text-primary-foreground px-3 py-1 rounded-full text-sm font-semibold">
                  {capturedImages.length}/3
                </div>
              </div>
              <div className="flex gap-2">
                <Button onClick={captureImage} className="flex-1 sm:flex-initial">
                  Capture Image {capturedImages.length + 1}
                </Button>
                <Button onClick={stopCamera} variant="outline">
                  Cancel
                </Button>
              </div>
              {capturedImages.length > 0 && (
                <div className="flex gap-2 flex-wrap">
                  {capturedImages.map((img, idx) => (
                    <div key={idx} className="w-16 h-16 rounded border-2 border-success overflow-hidden">
                      <img src={img} alt={`Capture ${idx + 1}`} className="w-full h-full object-cover" />
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {capturedImages.length === 3 && (
            <div className="space-y-4">
              <p className="text-sm text-success">✓ All 3 images captured successfully</p>
              <div className="flex gap-2 flex-wrap">
                {capturedImages.map((img, idx) => (
                  <div key={idx} className="w-24 h-24 rounded border-2 border-success overflow-hidden">
                    <img src={img} alt={`Capture ${idx + 1}`} className="w-full h-full object-cover" />
                  </div>
                ))}
              </div>
              <Button 
                onClick={resetCapture} 
                variant="outline"
                className="w-full sm:w-auto"
              >
                Retake All
              </Button>
            </div>
          )}
        </div>
      </div>
    </Card>
  );
};
