import { useState, useRef } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Hand, Check, Circle } from "lucide-react";

interface GestureCaptureProps {
  onCapture: (data: string) => void;
  captured: boolean;
}

export const GestureCapture = ({ onCapture, captured }: GestureCaptureProps) => {
  const [isRecording, setIsRecording] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: "user" } 
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setIsRecording(true);
      startRecording(stream);
    } catch (error) {
      console.error("Camera access denied:", error);
      alert("Camera access is required for gesture capture");
    }
  };

  const startRecording = (stream: MediaStream) => {
    const mediaRecorder = new MediaRecorder(stream);
    mediaRecorderRef.current = mediaRecorder;
    chunksRef.current = [];

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) {
        chunksRef.current.push(e.data);
      }
    };

    mediaRecorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: "video/webm" });
      
      // Convert to base64
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64 = reader.result as string;
        setPreview(base64);
        onCapture(base64);
      };
      reader.readAsDataURL(blob);
      
      stopCamera();
    };

    mediaRecorder.start();

    // Auto-stop after 2 seconds
    setTimeout(() => {
      if (mediaRecorderRef.current?.state === "recording") {
        mediaRecorderRef.current.stop();
        setIsRecording(false);
      }
    }, 2000);
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
  };

  return (
    <Card className="p-6 bg-card">
      <div className="flex items-start gap-4">
        <div className={`flex-shrink-0 w-12 h-12 rounded-full flex items-center justify-center ${
          captured ? "bg-success text-success-foreground" : "bg-primary text-primary-foreground"
        }`}>
          {captured ? <Check className="h-6 w-6" /> : <Hand className="h-6 w-6" />}
        </div>
        
        <div className="flex-1 min-w-0">
          <h3 className="font-semibold text-lg mb-2">Gesture Recognition</h3>
          
          {!isRecording && !preview && (
            <div className="space-y-4">
              <p className="text-sm text-muted-foreground">
                Record a 1-2 second hand gesture or pattern for authentication
              </p>
              <Button onClick={startCamera} variant="outline" className="w-full sm:w-auto">
                <Hand className="mr-2 h-4 w-4" />
                Start Recording
              </Button>
            </div>
          )}

          {isRecording && (
            <div className="space-y-4">
              <div className="relative rounded-lg overflow-hidden bg-black aspect-video max-w-md">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full h-full object-cover"
                />
                <div className="absolute top-4 right-4 flex items-center gap-2 bg-destructive/90 text-destructive-foreground px-3 py-1 rounded-full">
                  <Circle className="h-3 w-3 fill-current animate-pulse" />
                  <span className="text-sm font-medium">Recording</span>
                </div>
              </div>
              <p className="text-sm text-muted-foreground">
                Perform your gesture now... (auto-stops in 2s)
              </p>
            </div>
          )}

          {preview && (
            <div className="space-y-4">
              <div className="rounded-lg overflow-hidden bg-muted aspect-video max-w-md flex items-center justify-center">
                <p className="text-muted-foreground">Gesture recorded successfully</p>
              </div>
              <Button 
                onClick={() => {
                  setPreview(null);
                  startCamera();
                }} 
                variant="outline"
                className="w-full sm:w-auto"
              >
                Re-record
              </Button>
            </div>
          )}
        </div>
      </div>
    </Card>
  );
};
