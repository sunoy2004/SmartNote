import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Bluetooth, Loader2, CheckCircle2, XCircle } from "lucide-react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { toast } from "sonner";

interface BLEConnectionProps {
  biometricData: {
    face: { img1: string; img2: string; img3: string } | null;
    voice: string | null;
    gesture: string | null;
  };
  mode: "enroll" | "authenticate";
  onClose: () => void;
}

// Arduino Nano BLE Rev2 Service and Characteristic UUIDs
// Replace these with your actual UUIDs from Arduino sketch
const SERVICE_UUID = "19b10000-e8f2-537e-4f6c-d104768a1214";
const CHARACTERISTIC_UUID = "19b10001-e8f2-537e-4f6c-d104768a1214";

export const BLEConnection = ({ biometricData, mode, onClose }: BLEConnectionProps) => {
  const [isScanning, setIsScanning] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [result, setResult] = useState<"success" | "failed" | null>(null);
  const [device, setDevice] = useState<any>(null);
  const [characteristic, setCharacteristic] = useState<any>(null);
  const [errorMessage, setErrorMessage] = useState<string>("");

  const scanAndConnect = async () => {
    try {
      setIsScanning(true);
      setErrorMessage("");
      
      // Check if Web Bluetooth API is available
      if (!(navigator as any).bluetooth) {
        const msg = "Web Bluetooth is not supported. Use Chrome on Android/Desktop or Edge. HTTPS required.";
        setErrorMessage(msg);
        toast.error(msg);
        return;
      }

      // Check if running on HTTPS or localhost
      if (window.location.protocol !== "https:" && !window.location.hostname.includes("localhost")) {
        const msg = "Web Bluetooth requires HTTPS. Deploy to a secure server or use localhost.";
        setErrorMessage(msg);
        toast.error(msg);
        return;
      }

      // Request BLE device with correct service UUID
      const bleDevice = await (navigator as any).bluetooth.requestDevice({
        filters: [{ namePrefix: "Arduino" }],
        optionalServices: [SERVICE_UUID]
      });

      setDevice(bleDevice);
      console.log("Device selected:", bleDevice.name);
      
      // Connect to GATT server
      const server = await bleDevice.gatt?.connect();
      console.log("GATT server connected");
      
      // Get the service
      const service = await server.getPrimaryService(SERVICE_UUID);
      console.log("Service found");
      
      // Get the characteristic
      const char = await service.getCharacteristic(CHARACTERISTIC_UUID);
      console.log("Characteristic found");
      
      setCharacteristic(char);
      setIsConnected(true);
      toast.success(`Connected to ${bleDevice.name}`);
      
    } catch (error: any) {
      console.error("BLE connection error:", error);
      const msg = error.message || "Failed to connect to BLE device";
      setErrorMessage(msg);
      toast.error(msg);
    } finally {
      setIsScanning(false);
    }
  };

  const sendData = async () => {
    if (!characteristic || !isConnected) {
      toast.error("No device connected");
      return;
    }

    try {
      setIsSending(true);

      // Prepare payload
      const payload = JSON.stringify({
        mode: mode,
        face: biometricData.face,
        voice: biometricData.voice,
        gesture: biometricData.gesture
      });

      console.log("Preparing to send data:", { mode, dataLength: payload.length });

      // Convert payload to chunks (BLE has MTU limits, typically 20-512 bytes)
      const CHUNK_SIZE = 512; // Adjust based on your Arduino's MTU
      const encoder = new TextEncoder();
      const payloadBytes = encoder.encode(payload);
      
      // Send data in chunks
      for (let i = 0; i < payloadBytes.length; i += CHUNK_SIZE) {
        const chunk = payloadBytes.slice(i, i + CHUNK_SIZE);
        await characteristic.writeValue(chunk);
        console.log(`Sent chunk ${Math.floor(i / CHUNK_SIZE) + 1}/${Math.ceil(payloadBytes.length / CHUNK_SIZE)}`);
        await new Promise(resolve => setTimeout(resolve, 50)); // Small delay between chunks
      }

      // Send END signal
      const endSignal = encoder.encode("END");
      await characteristic.writeValue(endSignal);
      console.log("END signal sent");

      // Wait for Arduino response
      toast.info("Waiting for Arduino response...");
      
      // Start notifications to receive response
      await characteristic.startNotifications();
      
      // Listen for response
      characteristic.addEventListener('characteristicvaluechanged', handleCharacteristicValueChanged);
      
    } catch (error: any) {
      console.error("Error sending data:", error);
      toast.error(`Failed to send data: ${error.message}`);
      setResult("failed");
      setIsSending(false);
    }
  };

  const handleCharacteristicValueChanged = (event: any) => {
    const value = event.target.value;
    const decoder = new TextDecoder();
    const response = decoder.decode(value);
    
    console.log("Received from Arduino:", response);
    handleResponse(response);
    setIsSending(false);
    
    // Stop listening
    event.target.removeEventListener('characteristicvaluechanged', handleCharacteristicValueChanged);
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

        {errorMessage && (
          <div className="mt-4 p-4 bg-destructive/10 border border-destructive rounded-lg">
            <p className="text-sm text-destructive font-semibold">Error:</p>
            <p className="text-xs text-destructive mt-1">{errorMessage}</p>
          </div>
        )}

        <div className="mt-4 p-4 bg-muted rounded-lg">
          <p className="text-xs text-muted-foreground">
            <strong>Requirements:</strong> Web Bluetooth requires HTTPS (or localhost). 
            Arduino service UUID: <code className="bg-background px-1 py-0.5 rounded text-xs">{SERVICE_UUID}</code>
            <br />
            <strong>Browser Support:</strong> Chrome (Android/Desktop), Edge, Opera. Not supported in Safari/Firefox.
          </p>
        </div>
      </DialogContent>
    </Dialog>
  );
};
