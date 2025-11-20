import { useState, useRef } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Camera, Check } from "lucide-react";

interface FaceCaptureProps {
  onCapture: (data: string) => void;
  captured: boolean;
}

export const FaceCapture = ({ onCapture, captured }: FaceCaptureProps) => {
  const [preview, setPreview] = useState<string | null>(null);
  const [isCapturing, setIsCapturing] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: "user" } 
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setIsCapturing(true);
    } catch (error) {
      console.error("Camera access denied:", error);
      alert("Camera access is required for facial capture");
    }
  };

  const captureImage = () => {
    if (videoRef.current) {
      const canvas = document.createElement("canvas");
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.drawImage(videoRef.current, 0, 0);
        const base64 = canvas.toDataURL("image/jpeg", 0.8);
        setPreview(base64);
        onCapture(base64);
        stopCamera();
      }
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setIsCapturing(false);
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
          <h3 className="font-semibold text-lg mb-2">Facial Recognition</h3>
          
          {!isCapturing && !preview && (
            <div className="space-y-4">
              <p className="text-sm text-muted-foreground">
                Capture a clear photo of your face for biometric authentication
              </p>
              <Button onClick={startCamera} variant="outline" className="w-full sm:w-auto">
                <Camera className="mr-2 h-4 w-4" />
                Start Camera
              </Button>
            </div>
          )}

          {isCapturing && (
            <div className="space-y-4">
              <div className="relative rounded-lg overflow-hidden bg-black aspect-video max-w-md">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full h-full object-cover"
                />
              </div>
              <div className="flex gap-2">
                <Button onClick={captureImage} className="flex-1 sm:flex-initial">
                  Capture Face
                </Button>
                <Button onClick={stopCamera} variant="outline">
                  Cancel
                </Button>
              </div>
            </div>
          )}

          {preview && (
            <div className="space-y-4">
              <div className="relative rounded-lg overflow-hidden bg-black aspect-video max-w-md">
                <img src={preview} alt="Face preview" className="w-full h-full object-cover" />
              </div>
              <Button 
                onClick={() => {
                  setPreview(null);
                  startCamera();
                }} 
                variant="outline"
                className="w-full sm:w-auto"
              >
                Retake
              </Button>
            </div>
          )}
        </div>
      </div>
    </Card>
  );
};
