import { useState, useRef, useEffect } from "react";
import { Mic, Square, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";
import { API_ENDPOINTS } from "@/config/api";

interface VoiceRecorderProps {
  onTranscript: (transcript: string, summary: string) => void;
  isRecording: boolean;
  setIsRecording: (value: boolean) => void;
}

export const VoiceRecorder = ({ onTranscript, isRecording, setIsRecording }: VoiceRecorderProps) => {
  const [transcript, setTranscript] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [liveTranscript, setLiveTranscript] = useState(""); // For live transcript display
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const websocketRef = useRef<WebSocket | null>(null);
  const accumulatedAudioRef = useRef<Float32Array>(new Float32Array());
  const { toast } = useToast();

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (websocketRef.current) {
        websocketRef.current.close();
      }
    };
  }, []);

  const startRecording = async () => {
    try {
      // Check if we're in a secure context (required for MediaDevices API)
      if (location.protocol !== 'https:' && location.hostname !== 'localhost' && location.hostname !== '127.0.0.1') {
        throw new Error("MediaDevices API requires a secure context (HTTPS). Please use HTTPS or localhost.");
      }
      
      // Check if mediaDevices API is available
      if (!navigator.mediaDevices) {
        throw new Error("MediaDevices API is not available in your browser. Please use a modern browser with HTTPS.");
      }
      
      // Get microphone access with better settings
      // Use more compatible constraints
      const constraints = {
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1
        }
      };
      
      // Try with sample rate constraint first, then fallback without it
      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({ 
          audio: {
            ...constraints.audio,
            sampleRate: 16000
          } 
        });
      } catch (error) {
        console.warn("Could not set sample rate constraint, trying without it:", error);
        stream = await navigator.mediaDevices.getUserMedia(constraints);
      }
      streamRef.current = stream;
      
      // Create MediaRecorder with webm codec
      let mimeType = 'audio/webm;codecs=opus';
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = 'audio/webm';
        if (!MediaRecorder.isTypeSupported(mimeType)) {
          mimeType = 'audio/mp4'; // Fallback for Safari
        }
      }
      
      const mediaRecorder = new MediaRecorder(stream, { mimeType });
      
      // Reset audio chunks
      audioChunksRef.current = [];
      
      // Connect to WebSocket for streaming transcription
      const wsUrl = `${API_ENDPOINTS.health.replace('/health', '').replace('http://', 'ws://').replace('https://', 'wss://')}/ws/transcribe`;
      console.log("WebSocket URL:", wsUrl); // Debug log
      websocketRef.current = new WebSocket(wsUrl);
      
      websocketRef.current.onopen = () => {
        console.log("WebSocket connection opened");
        // Send reset message to clear any previous state
        if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN) {
          websocketRef.current.send(JSON.stringify({ type: "reset" }));
        }
        // Set a flag to indicate WebSocket is ready
        (window as any).websocketReady = true;
      };
      
      websocketRef.current.onmessage = (event) => {
        console.log("Received WebSocket message:", event.data);
        try {
          const data = JSON.parse(event.data);
          if (data.partial_transcript) {
            setLiveTranscript(data.partial_transcript);
          }
        } catch (e) {
          console.error("Error parsing WebSocket message:", e);
        }
      };
      
      websocketRef.current.onerror = (error) => {
        console.error("WebSocket error:", error);
        console.log("WebSocket URL that failed:", wsUrl); // Debug log
        // Don't show toast for WebSocket errors as it's not critical
        console.log("Live transcription will not be available, but recording will still work");
      };
      
      // Collect audio data as it becomes available and send to WebSocket
      mediaRecorder.ondataavailable = async (event) => {
        if (event.data.size > 0) {
          console.log("Received audio data chunk:", event.data.size, "bytes");
          audioChunksRef.current.push(event.data);
          
          // Send audio data to WebSocket for live transcription
          if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN) {
            try {
              // Convert blob to ArrayBuffer then send as binary data
              const arrayBuffer = await event.data.arrayBuffer();
              console.log("Sending audio data to WebSocket:", arrayBuffer.byteLength, "bytes");
              websocketRef.current.send(arrayBuffer);
            } catch (e) {
              console.error("Error sending audio data to WebSocket:", e);
            }
          } else {
            console.log("WebSocket not ready, current state:", websocketRef.current?.readyState);
          }
        }
      };
      
      // Handle recording stop
      mediaRecorder.onstop = async () => {
        // Stop all tracks
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop());
          streamRef.current = null;
        }
        
        // Close WebSocket connection
        if (websocketRef.current) {
          websocketRef.current.close();
          websocketRef.current = null;
        }
        
        // Combine all audio chunks into a single blob
        if (audioChunksRef.current.length > 0) {
          const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });
          
          if (audioBlob.size > 0) {
            await processAudio(audioBlob);
          } else {
            toast({
              title: "Recording error",
              description: "No audio was recorded. Please try again.",
              variant: "destructive",
            });
            setIsRecording(false);
            setIsProcessing(false);
            setLiveTranscript("");
          }
        } else {
          toast({
            title: "Recording error",
            description: "No audio was recorded. Please try again.",
            variant: "destructive",
          });
          setIsRecording(false);
          setIsProcessing(false);
          setLiveTranscript("");
        }
      };
      
      // Handle recording errors
      mediaRecorder.onerror = (event) => {
        console.error("MediaRecorder error:", event);
        toast({
          title: "Recording error",
          description: "An error occurred during recording. Please try again.",
          variant: "destructive",
        });
        stopRecording();
      };
      
      // Start recording with 1 second chunks
      try {
        mediaRecorder.start(1000);
      } catch (startError) {
        console.error("Error starting MediaRecorder:", startError);
        // Try without timeslice
        try {
          mediaRecorder.start();
        } catch (fallbackError) {
          throw new Error("Failed to start recording: " + fallbackError.message);
        }
      }
      mediaRecorderRef.current = mediaRecorder;
      setIsRecording(true);
      setTranscript("Recording... Please speak clearly");
      setLiveTranscript("Listening..."); // Initial live transcript message
      
      toast({
        title: "Recording started",
        description: "Speak clearly into your microphone",
      });
    } catch (error) {
      console.error("Error starting recording:", error);
      let errorMessage = "Unable to access microphone. Please check permissions and try again.";
      
      // Provide more specific error messages
      if (error instanceof Error) {
        if (error.message.includes("MediaDevices API is not available")) {
          errorMessage = "Your browser does not support microphone access. Please use a modern browser (Chrome, Firefox, Edge) and ensure you're using HTTPS.";
        } else if (error.message.includes("Permission denied")) {
          errorMessage = "Microphone permission denied. Please allow microphone access in your browser settings.";
        } else if (error.message.includes("Permission dismissed")) {
          errorMessage = "Microphone permission was dismissed. Please click the microphone button again and allow microphone access.";
        }
      }
      
      toast({
        title: "Microphone error",
        description: errorMessage,
        variant: "destructive",
      });
      setIsRecording(false);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setTranscript("Processing...");
      setLiveTranscript("Processing audio...");
    }
    
    // Also stop the media stream if it exists
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    // Close WebSocket connection
    if (websocketRef.current) {
      websocketRef.current.close();
      websocketRef.current = null;
    }
  };

  const processAudio = async (audioBlob: Blob) => {
    setIsProcessing(true);
    setLiveTranscript("Transcribing audio...");
    
    try {
      // Validate audio blob
      if (audioBlob.size === 0) {
        throw new Error("Recorded audio is empty");
      }
      
      // Send audio to backend for transcription + summarization
      const formData = new FormData();
      formData.append('file', audioBlob, 'recording.webm');
      
      const response = await fetch(API_ENDPOINTS.voiceToNotes, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        let errorMessage = `API error: ${response.status}`;
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorMessage;
        } catch (e) {
          errorMessage = response.statusText || errorMessage;
        }
        throw new Error(errorMessage);
      }
      
      const data = await response.json();
      
      // Handle the response
      const finalTranscript = data.transcript || "No speech detected in the audio.";
      const finalSummary = data.summary || "No summary available - no speech detected.";
      
      // If we get empty results, provide better feedback
      if ((!finalTranscript || finalTranscript === "No speech detected in the audio.") && 
          (!finalSummary || finalSummary === "No summary available - no speech detected.")) {
        toast({
          title: "No speech detected",
          description: "The system couldn't detect any recognizable speech in your recording. Please try speaking more clearly and at a normal volume.",
          variant: "destructive",
        });
      }
      
      onTranscript(finalTranscript, finalSummary);
      setTranscript("");
      setLiveTranscript(""); // Clear live transcript
      
      if (finalTranscript !== "No speech detected in the audio.") {
        toast({
          title: "Note saved",
          description: "Your voice note has been transcribed and summarized.",
        });
      }
    } catch (error) {
      console.error("Processing error:", error);
      toast({
        title: "Processing failed",
        description: error instanceof Error ? error.message : "Failed to process audio. Make sure your backend is running.",
        variant: "destructive",
      });
      setTranscript("");
      setLiveTranscript("");
    } finally {
      setIsProcessing(false);
      // Clean up
      audioChunksRef.current = [];
    }
  };

  return (
    <Card className="p-6 mb-8">
      <div className="flex flex-col items-center gap-4">
        <div className="text-center">
          <h2 className="text-xl font-semibold text-foreground mb-2">Record Voice Note</h2>
          <p className="text-sm text-muted-foreground">
            {isRecording ? "Recording... Click stop when done" : "Click the microphone to start recording"}
          </p>
        </div>

        <Button
          size="lg"
          onClick={isRecording ? stopRecording : startRecording}
          disabled={isProcessing}
          className="w-20 h-20 rounded-full"
          variant={isRecording ? "destructive" : "default"}
        >
          {isProcessing ? (
            <Loader2 className="w-8 h-8 animate-spin" />
          ) : isRecording ? (
            <Square className="w-8 h-8" />
          ) : (
            <Mic className="w-8 h-8" />
          )}
        </Button>

        {/* Live transcript display during recording */}
        {isRecording && liveTranscript && (
          <div className="w-full mt-4 p-4 bg-muted rounded-lg">
            <p className="text-sm text-muted-foreground mb-1">Live Transcript:</p>
            <p className="text-foreground">{liveTranscript}</p>
          </div>
        )}

        {/* Status messages */}
        {transcript && transcript !== "Recording..." && transcript !== "Processing..." && (
          <div className="w-full mt-4 p-4 bg-muted rounded-lg">
            <p className="text-sm text-muted-foreground mb-1">Status:</p>
            <p className="text-foreground">{transcript}</p>
          </div>
        )}
      </div>
    </Card>
  );
};