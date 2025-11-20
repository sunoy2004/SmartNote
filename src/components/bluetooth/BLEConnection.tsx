import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Bluetooth, Loader2, CheckCircle2, XCircle } from "lucide-react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { toast } from "sonner";

interface BLEConnectionProps {
  biometricData: {
    face: string | null;
    voice: string | null;
    gesture: string | null;
  };
  mode: "enroll" | "authenticate";
  onClose: () => void;
}

export const BLEConnection = ({ biometricData, mode, onClose }: BLEConnectionProps) => {
  const [isScanning, setIsScanning] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [result, setResult] = useState<"success" | "failed" | null>(null);
  const [device, setDevice] = useState<any>(null);

  const scanAndConnect = async () => {
    try {
      setIsScanning(true);
      
      // Check if Web Bluetooth API is available
      if (!(navigator as any).bluetooth) {
        toast.error("Web Bluetooth API is not available in your browser");
        return;
      }

      // Request BLE device
      const bleDevice = await (navigator as any).bluetooth.requestDevice({
        filters: [{ namePrefix: "Arduino" }],
        optionalServices: ["generic_access"]
      });

      setDevice(bleDevice);
      
      // Connect to the device
      const server = await bleDevice.gatt?.connect();
      
      if (server?.connected) {
        setIsConnected(true);
        toast.success("Connected to Arduino BLE device");
      }
    } catch (error) {
      console.error("BLE connection error:", error);
      toast.error("Failed to connect to BLE device");
    } finally {
      setIsScanning(false);
    }
  };

  const sendData = async () => {
    if (!device || !isConnected) {
      toast.error("No device connected");
      return;
    }

    try {
      setIsSending(true);

      // Prepare payload
      const payload = JSON.stringify({
        face: biometricData.face,
        voice: biometricData.voice,
        gesture: biometricData.gesture,
        mode: mode
      });

      // In a real implementation, you would send this via BLE characteristic
      // For now, we'll simulate the response
      console.log("Sending data:", { mode, dataLength: payload.length });

      // Simulate processing time
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Simulate Arduino response (replace with actual BLE read)
      const simulatedResponse = Math.random() > 0.3 
        ? (mode === "enroll" ? "ENROLL_SUCCESS" : "AUTH_SUCCESS")
        : (mode === "enroll" ? "ENROLL_FAILED" : "AUTH_FAILED");

      handleResponse(simulatedResponse);
    } catch (error) {
      console.error("Error sending data:", error);
      toast.error("Failed to send data to device");
      setResult("failed");
    } finally {
      setIsSending(false);
    }
  };

  const handleResponse = (response: string) => {
    if (response.includes("SUCCESS")) {
      setResult("success");
      toast.success(
        mode === "enroll" 
          ? "User enrolled successfully!" 
          : "Authentication successful!"
      );
    } else {
      setResult("failed");
      toast.error(
        mode === "enroll" 
          ? "Enrollment failed. Please try again." 
          : "Authentication failed. Please try again."
      );
    }
  };

  return (
    <Dialog open onOpenChange={onClose}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>BLE Device Connection</DialogTitle>
        </DialogHeader>

        <div className="space-y-4">
          {!isConnected && !result && (
            <Card className="p-6 border-dashed">
              <div className="text-center space-y-4">
                <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center mx-auto">
                  <Bluetooth className="h-8 w-8 text-primary" />
                </div>
                <div>
                  <h3 className="font-semibold mb-2">Connect to Arduino BLE</h3>
                  <p className="text-sm text-muted-foreground">
                    Click the button below to scan and connect to your Arduino Nano BLE Rev2
                  </p>
                </div>
                <Button 
                  onClick={scanAndConnect} 
                  disabled={isScanning}
                  className="w-full"
                >
                  {isScanning ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Scanning...
                    </>
                  ) : (
                    <>
                      <Bluetooth className="mr-2 h-4 w-4" />
                      Scan for Devices
                    </>
                  )}
                </Button>
              </div>
            </Card>
          )}

          {isConnected && !result && (
            <Card className="p-6">
              <div className="text-center space-y-4">
                <div className="w-16 h-16 rounded-full bg-success/10 flex items-center justify-center mx-auto">
                  <CheckCircle2 className="h-8 w-8 text-success" />
                </div>
                <div>
                  <h3 className="font-semibold mb-2">Device Connected</h3>
                  <p className="text-sm text-muted-foreground mb-1">
                    {device?.name || "Arduino BLE Device"}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    Ready to transmit biometric data
                  </p>
                </div>
                <Button 
                  onClick={sendData} 
                  disabled={isSending}
                  className="w-full"
                >
                  {isSending ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Sending Data...
                    </>
                  ) : (
                    `Send ${mode === "enroll" ? "Enrollment" : "Authentication"} Data`
                  )}
                </Button>
              </div>
            </Card>
          )}

          {result && (
            <Card className={`p-6 ${result === "success" ? "border-success" : "border-destructive"}`}>
              <div className="text-center space-y-4">
                <div className={`w-16 h-16 rounded-full flex items-center justify-center mx-auto ${
                  result === "success" ? "bg-success/10" : "bg-destructive/10"
                }`}>
                  {result === "success" ? (
                    <CheckCircle2 className="h-8 w-8 text-success" />
                  ) : (
                    <XCircle className="h-8 w-8 text-destructive" />
                  )}
                </div>
                <div>
                  <h3 className="font-semibold mb-2">
                    {result === "success" 
                      ? (mode === "enroll" ? "Enrollment Successful!" : "Authentication Successful!")
                      : (mode === "enroll" ? "Enrollment Failed" : "Authentication Failed")
                    }
                  </h3>
                  <p className="text-sm text-muted-foreground">
                    {result === "success"
                      ? mode === "enroll" 
                        ? "User biometric data has been stored successfully"
                        : "User identity verified successfully"
                      : "Please try again or contact support"
                    }
                  </p>
                </div>
                <Button onClick={onClose} className="w-full">
                  Close
                </Button>
              </div>
            </Card>
          )}
        </div>

        <div className="mt-4 p-4 bg-muted rounded-lg">
          <p className="text-xs text-muted-foreground">
            <strong>Note:</strong> Web Bluetooth requires HTTPS. Make sure your app is served over HTTPS 
            or use localhost for testing. The Arduino Nano BLE Rev2 must be properly configured 
            to receive and process the biometric data payload.
          </p>
        </div>
      </DialogContent>
    </Dialog>
  );
};
